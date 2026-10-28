"""The morning probe run (Task 12a, spec 5.1): the sealed probes, rerun unchanged, on the checkpoint
that is serving after the night.

``run_morning`` answers each of the person's sealed probes with every arm (``arms.reply``), judges
each arm against arm 1 in both orders (``score.score``; ``taste_ab`` by log-probability, no judge),
and writes ``scores/night-NN.json`` in the layout ``reports.results`` reads: per arm and dimension,
the tie-inclusive score against arm 1 (arm 1 is 0.5 against itself by definition), plus the two
headline pairs (3 against 3x, 4 against 2C). The file holds numbers, the night, the label and the
checkpoint id, never a reply or a probe, and nothing is written to history, evidence or memory (RF5).

- Glimmer is woken first (``Morning.wake``); if it can't be, the run stops with ``glimmer_not_ready``
  before any call, and the score file records ``wake_seconds``.
- A night that already has a score file is refused before any call: the first score stands.
- The checkpoint given must be the one serving; arms 3 and 4 are pinned to it for the whole run.
- Arm 3x needs the registered donor's promoted adapter. Without one, 3x is left out and recorded
  under ``unavailable`` (specificity is then inconclusive); it is never filled in.
- Every paid call is held first, the way ``night/holds.py`` does it: the arms on ``modal_usd`` and
  ``baseten_sleep_usd``, the judge on ``meta_usd``. The judge's hold is taken before any arm runs, so
  a pool without room stops the run before the first call.
- With ``concurrency`` above 1 every arm's question is prepared first, on the calling thread
  (``arms.prepare``: adapters, context C, the stored evidence), then the questions are asked side by
  side, then every (pair, probe) is judged side by side, at most ``concurrency`` at once
  (``parallel.Calls``). Replies and outcomes are gathered by arm, pair and probe, never by when they
  came back, so the score file is the serial run's given the same replies. The holds' bounds still
  cover the worst case: the judge's counts calls, and the GPU bills wall time, which calls in
  parallel only shorten.
- A daily morning may run a ``ProbeSubset`` (``DAILY``, the daily runner's development nights):
  fewer arms, and a fixed number of each kind of probe, picked by id alone (``subset_probes``), so
  every daily morning asks the same probes whatever came back before. Its file says ``subset`` and
  leaves out the comparisons whose arms didn't answer; the holds scale to what it asks. A final run
  never takes a subset (refused before any call), and a run without one is the full run, unchanged.
- A judge answer that can't be read is asked once more with more room (``MuseJudge.again``), at most
  ``JUDGE_RETRY_SHARE`` of the run's judge calls, which the judge's hold covers. A probe still without a
  verdict is ``no verdict``: left out of its comparison's score, given an outcome row ``no_verdict``,
  counted per comparison under ``no_verdict`` (written only when there is one) and in
  ``MorningResult.no_verdict``. One unreadable answer no longer stops the morning (Teo's night 1 did,
  on 2026-10-28), but a judge that can't be read does, loudly: past ``NO_VERDICT_MAX_SHARE`` of a
  comparison's judged probes the run raises ``TooManyNoVerdicts`` and writes no score, so a rerun may
  score the night. A comparison with no verdict at all is never written as a 0.50 tie: it is missing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import date
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

from anjo.sleep.budget import Bound, Ledger
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import held, usd
from anjo.sleep.parallel import Calls
from anjo.sleep.probes.arms import ARMS, Arm, ArmDeps, ArmReply, DonorError, Question, ask, prepare
from anjo.sleep.probes.clients import AdapterRegistry
from anjo.sleep.probes.score import (
    JUDGE_RETRY_MAX_TOKENS,
    REPORTED,
    BootstrapConfig,
    JudgeError,
    NoVerdict,
    PairScore,
    PairwiseJudge,
    ProbeOutcome,
    RoomyJudge,
    Verdict,
    probe_outcome,
    score_outcomes,
)
from anjo.sleep.reports import results
from anjo.sleep.serving.backend import phase
from anjo.sleep.serving.client import GlimmerNotReady, WakeResult
from anjo.sleep.types import CheckpointRef, Probe

ARM_ORDER: Final[tuple[Arm, ...]] = ("1", "2C", "2F", "3", "3x", "4")
BASELINE: Final[Arm] = "1"
PAIRS: Final[dict[str, tuple[Arm, Arm]]] = {"3_vs_3x": ("3", "3x"), "4_vs_2C": ("4", "2C")}
ARM_CALL_S: Final = 60.0  # the most one arm reply holds the serving GPU
JUDGE_USD_PER_CALL: Final = 0.02  # a Muse Spark judge call, until the Oct 30 timing measures it
JUDGE_RETRY_SHARE: Final = 0.1  # retries of unreadable answers a run may send, as a share of its judge calls (1 in 257 seen)
NO_VERDICT: Final = "no_verdict"
# Past this share of a comparison's judged probes without a verdict, the morning stops. One answer in
# 257 was unreadable on Teo's night 1, and a probe goes without a verdict only when an answer and its
# retry both fail, so chance leaves a comparison far below a quarter; a judge whose format changed
# fails most of them. Under it, a comparison's score still rests on three quarters of its probes.
NO_VERDICT_MAX_SHARE: Final = 0.25
ARM_POOLS: Final = ("modal_usd", "baseten_sleep_usd")
JUDGE_POOLS: Final = ("meta_usd",)
SCORES: Final = "scores"
DONOR_UNAVAILABLE: Final = "donor_unavailable"
logger = logging.getLogger(__name__)


class TooManyNoVerdicts(RuntimeError):
    """The judge gave no readable verdict on more than ``NO_VERDICT_MAX_SHARE`` of a comparison's
    probes: its answers can't be read, so the morning stops and scores nothing."""


class MorningError(RuntimeError):
    """The morning run was refused: a rerun, a stale checkpoint, a probe of another person, or a
    final run given a subset."""


@dataclass(frozen=True)
class ProbeSubset:
    """A share of the morning run, committed before any result: the arms that answer (arm 1 among
    them) and how many probes of each kind are asked. Its numbers change only under a new name."""

    name: str
    arms: tuple[Arm, ...]
    per_kind: Mapping[str, int]

    def __post_init__(self) -> None:
        if BASELINE not in self.arms or len(self.arms) < 2 or not set(self.arms) <= set(ARM_ORDER):
            raise ValueError("a subset keeps arm 1 and at least one more of the six arms")
        if any(n < 0 for n in self.per_kind.values()):
            raise ValueError("a subset asks no negative count of a kind")


# Committed 2026-10-28, before any daily result: arm 3 against arm 1 is the question a daily night
# answers (did tonight's weights help); 14 of the 80 probes, every kind with a judge or a log-prob,
# pushback left to the full runs. 22 judge calls and 28 replies, against the full run's 910 and 480.
DAILY: Final = ProbeSubset(name="daily-v1", arms=("1", "3"), per_kind=MappingProxyType(
    {"new_work": 5, "vocab": 3, "tone_hold": 3, "taste_ab": 3, "pushback": 0}))


def _rank(subset: ProbeSubset, probe: Probe) -> str:
    return hashlib.sha256(f"{subset.name}:{probe.id}".encode()).hexdigest()


def subset_probes(probes: Sequence[Probe], subset: ProbeSubset) -> list[Probe]:
    """Each kind's first ``per_kind`` probes by the hash of the subset's name and the probe's id, in
    the subset's kind order: the same probes whatever order they come in. A kind with fewer probes
    gives all it has."""
    chosen: list[Probe] = []
    for kind, count in subset.per_kind.items():
        chosen.extend(sorted((p for p in probes if p.kind == kind), key=partial(_rank, subset))[:count])
    return chosen


@dataclass(frozen=True)
class Morning:
    """What a morning run reaches outside through. ``arms.adapters`` is the registry of serving
    adapters; ``sealed`` gives a person's sealed probes; ``root`` defaults to the results tree."""

    arms: ArmDeps
    sealed: Callable[[str], Sequence[Probe]]
    judge: PairwiseJudge
    prices: Prices
    ledger: Ledger | None = None
    root: Path | None = None
    timer: Callable[[], float] = time.monotonic
    wake: Callable[[], WakeResult] | None = None  # wakes Glimmer before any arm calls it
    sleep: Callable[[], None] | None = None  # puts it back to sleep when the run ends, however it ends
    concurrency: int = 1  # model calls at once (``parallel.Calls``); 1 is the serial run
    retries: Callable[[], int] = field(default=lambda: 0)  # the arms' running count of requests sent again (the night log's)


@dataclass(frozen=True)
class MorningResult:
    path: Path
    replies: int
    judge_calls: int
    no_verdict: int = 0  # (comparison, probe) pairs the judge gave no readable verdict on


class _Pinned:
    """The adapter registry with the person's serving checkpoint fixed for the whole run."""

    def __init__(self, inner: AdapterRegistry, person_id: str, checkpoint: CheckpointRef | None) -> None:
        self._inner, self._person, self._checkpoint = inner, person_id, checkpoint

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        return self._checkpoint if person_id == self._person else self._inner.last_promoted(person_id)


class _Counting:
    """The judge, its calls counted for the hold: every call, from any thread. An unreadable answer is
    asked once more with more room while ``retries`` last; one still unreadable raises ``NoVerdict``.
    Its log lines name the probe and how the answer ended, never its words."""

    def __init__(self, inner: PairwiseJudge, retries: int = 0) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._retries = retries
        self.calls = 0

    def _count(self, *, retry: bool = False) -> bool:
        with self._lock:
            if retry and self._retries <= 0:
                return False
            self._retries -= retry
            self.calls += 1
            return True

    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        self._count()
        try:
            return self._inner.verdict(probe, first, second)
        except NoVerdict:
            raise
        except JudgeError as error:
            if not self._count(retry=True):
                logger.warning("judge: probe %s unreadable (%s), no retry left: no verdict", probe.id, error)
                raise NoVerdict(str(error)) from None
            logger.warning("judge: probe %s unreadable (%s), asked once more with room for %d tokens",
                           probe.id, error, JUDGE_RETRY_MAX_TOKENS)
        again = self._inner.again if isinstance(self._inner, RoomyJudge) else self._inner.verdict
        try:
            return again(probe, first, second)
        except (JudgeError, ModelApiError) as error:
            reason = str(error) if isinstance(error, JudgeError) else type(error).__name__
            logger.warning("judge: probe %s unreadable again (%s): no verdict", probe.id, reason)
            raise NoVerdict(reason) from None


def _next_file(folder: Path, night: date) -> Path:
    """The score file for ``night``, refusing a night that is already scored."""
    taken = 0
    for path in folder.glob("night-*.json") if folder.is_dir() else ():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("night") == night.isoformat():
            raise MorningError(f"{night.isoformat()} is already scored ({path.name}); the first score stands")
        taken = max(taken, int(path.stem.rsplit("-", 1)[1]))
    return folder / f"night-{taken + 1:02d}.json"


def _stat(wins: int, ties: int, losses: int, score: float, decisive: float | None,
          ci: tuple[float, float] | None) -> dict[str, object]:
    return {"n": wins + ties + losses, "wins": wins, "ties": ties, "losses": losses, "score": score,
            "decisive_share": decisive, "ci": None if ci is None else list(ci)}


def _stats(done: PairScore) -> dict[str, dict[str, object]]:
    """Per dimension and overall: the counts, the tie-inclusive score, the decisive share and the CI."""
    found = {name: _stat(t.wins, t.ties, t.losses, t.score, t.decisive_share, t.ci)
             for name, t in sorted(done.per_dimension.items()) if name != REPORTED}
    found[results.OVERALL] = _stat(done.wins, done.ties, done.losses, done.score, done.decisive_share, done.ci)
    return found


def _all_ties(against: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Arm 1 against itself: every probe a tie, by definition."""
    return {name: _stat(0, int(str(stat["n"])), 0, 0.5, None, (0.5, 0.5)) for name, stat in against.items()}


def _rows(pair: str, probes: Sequence[Probe], found: Sequence[ProbeOutcome | None]) -> list[dict[str, object]]:
    """Ids and codes only: enough to recount any result, with no text. A probe with no verdict says so."""
    return [{"probe": p.id, "pair": pair, "kind": p.kind, "outcome": NO_VERDICT, "flip": False} if f is None else
            {"probe": f.probe_id, "pair": pair, "kind": f.kind, "outcome": f.outcome, "flip": f.flipped}
            for p, f in zip(probes, found, strict=True)]


def _answer(
    user_id: str, probes: Sequence[Probe], deps: ArmDeps, calls: Calls, arms: Sequence[Arm] = ARM_ORDER,
) -> tuple[dict[Arm, list[ArmReply]], dict[str, str]]:
    """Every arm's questions prepared here, in arm and probe order, then asked side by side."""
    questions: dict[Arm, list[Question]] = {}
    unavailable: dict[str, str] = {}
    for arm in arms:
        try:
            questions[arm] = [prepare(arm, user_id, p, deps=deps) for p in probes]
        except DonorError:
            if arm != "3x":
                raise
            unavailable[arm] = DONOR_UNAVAILABLE  # a code: score files hold ids, codes and numbers only
    answers = iter(calls.map(partial(ask, deps=deps), [q for asked in questions.values() for q in asked]))
    return {arm: [next(answers) for _ in asked] for arm, asked in questions.items()}, unavailable


def _comparisons(replies: Mapping[Arm, Sequence[ArmReply]]) -> list[tuple[str, Arm, Arm]]:
    """Each arm against arm 1, then the headline pairs whose arms both answered, in file order."""
    found = [(f"{arm}_vs_1", arm, BASELINE) for arm in ARM_ORDER if arm != BASELINE and arm in replies]
    return found + [(name, a, b) for name, (a, b) in PAIRS.items() if a in replies and b in replies]


def _judged(
    probes: Sequence[Probe], replies: Mapping[Arm, Sequence[ArmReply]], judge: PairwiseJudge, calls: Calls,
) -> list[tuple[str, Arm, Arm, list[ProbeOutcome | None]]]:
    """Every comparison's per-probe outcomes, judged side by side, gathered by comparison and probe;
    None where the judge gave no readable verdict."""
    comparisons = _comparisons(replies)
    jobs = [(first, second, index) for _, first, second in comparisons for index in range(len(probes))]

    def outcome(job: tuple[Arm, Arm, int]) -> ProbeOutcome | None:
        first, second, index = job
        try:
            return probe_outcome(probes[index], replies[first][index], replies[second][index], judge)
        except NoVerdict:
            return None

    found = calls.map(outcome, jobs)
    size = len(probes)
    return [(name, a, b, found[n * size:(n + 1) * size]) for n, (name, a, b) in enumerate(comparisons)]


def _worst_judge_calls(probes: Sequence[Probe], arms: Sequence[Arm] = ARM_ORDER) -> int:
    judged = sum(1 for p in probes if p.kind != "taste_ab")
    pairs = sum(1 for a, b in PAIRS.values() if a in arms and b in arms)
    return (len(arms) - 1 + pairs) * judged * 2


def run_morning(
    user_id: str, night: date, checkpoint: CheckpointRef | None, *, morning: Morning,
    label: str = "development", promoted: bool = True, subset: ProbeSubset | None = None,
) -> MorningResult:
    """``_run_morning`` as one phase: Glimmer goes back to sleep when the run ends, however it ends."""
    with phase(morning.sleep):
        return _run_morning(user_id, night, checkpoint, morning=morning, label=label, promoted=promoted, subset=subset)


def _run_morning(
    user_id: str, night: date, checkpoint: CheckpointRef | None, *, morning: Morning,
    label: str = "development", promoted: bool = True, subset: ProbeSubset | None = None,
) -> MorningResult:
    """Score every arm on ``user_id``'s sealed probes, on the checkpoint serving after ``night``.

    ``checkpoint`` is the night's own (``NightResult.checkpoint``) and must be the one serving now;
    anything else is refused. A night that promoted nothing passes ``promoted=False``: the checkpoint
    that was serving is scored, and the file says ``not_promoted``."""
    if label not in results.LABELS:
        raise ValueError(f"label is one of {', '.join(results.LABELS)}, got {label!r}")
    if subset is not None and label == "final":
        raise MorningError("a final run asks the full set of probes and arms, never a subset")
    folder = results.results_root(morning.root) / user_id / SCORES
    target = _next_file(folder, night)
    serving = morning.arms.adapters.last_promoted(user_id)
    if promoted:
        if checkpoint is None:
            raise MorningError("a promoted night gives the checkpoint it promoted")
        if serving != checkpoint:
            raise MorningError("the night's checkpoint is not the one serving now; nothing was scored")
    else:
        checkpoint = serving
    probes = list(morning.sealed(user_id))
    if subset is not None:
        probes = subset_probes(probes, subset)
    arms = ARM_ORDER if subset is None else tuple(a for a in ARM_ORDER if a in subset.arms)
    if not probes:
        raise MorningError(f"{user_id} has no sealed probes")
    if any(p.person_id != user_id for p in probes):
        raise MorningError("a sealed probe belongs to another person")
    woke = None if morning.wake is None else morning.wake()
    if woke is not None and not woke.ok:
        raise GlimmerNotReady()  # nothing was called or written
    deps = replace(morning.arms, adapters=_Pinned(morning.arms.adapters, user_id, checkpoint), until=night,
                   context_cache={})
    worst_judge = _worst_judge_calls(probes, arms)
    retries = math.ceil(worst_judge * JUDGE_RETRY_SHARE)
    judge = _Counting(morning.judge, retries)
    key = f"morning:{night.isoformat()}:{user_id}"
    arm_s = len(probes) * len(arms) * ARM_CALL_S
    started = morning.timer()
    with ExitStack() as holds:
        holds.enter_context(held(morning.ledger, f"{key}:judge", JUDGE_POOLS,
                                 usd((worst_judge + retries) * JUDGE_USD_PER_CALL),
                                 lambda: judge.calls * JUDGE_USD_PER_CALL))
        holds.enter_context(held(morning.ledger, f"{key}:arms", ARM_POOLS,
                                 Bound(morning.prices.serve_gpu_per_hour, arm_s),
                                 lambda: morning.prices.serve_gpu_per_hour * max(0.0, morning.timer() - started) / 3600.0))
        calls = Calls(morning.concurrency)
        replies, unavailable = _answer(user_id, probes, deps, calls, arms)
        config = BootstrapConfig.from_thresholds()
        rows: list[dict[str, object]] = []
        stats: dict[str, dict[str, dict[str, object]]] = {}
        pairs: dict[str, dict[str, dict[str, object]]] = {}
        no_verdict: dict[str, int] = {}
        judged = sum(1 for p in probes if p.kind != "taste_ab")
        for name, first, second, found in _judged(probes, replies, judge, calls):
            kept = [(p, f) for p, f in zip(probes, found, strict=True) if f is not None]
            if missing := len(found) - len(kept):
                no_verdict[name] = missing
                if missing > NO_VERDICT_MAX_SHARE * judged:
                    raise TooManyNoVerdicts(
                        f"the judge gave no readable verdict on {missing} of {judged} probes in {name}, more than "
                        f"{NO_VERDICT_MAX_SHARE:.0%}: its answers can't be read (has their format changed?), so "
                        "nothing was scored")
            if not kept:
                continue  # no probe has a verdict: the comparison is missing, never a 0.50 tie
            scored = _stats(score_outcomes([p for p, _ in kept], [f.outcome for _, f in kept], bootstrap=config))
            if second == BASELINE:  # the arms are sorted when written; the pairs keep _comparisons' order
                stats[first] = scored
            else:
                pairs[name] = scored
            rows.extend(_rows(name, probes, found))
    if stats:  # arm 1 against itself takes the shape of an arm that was scored
        stats[BASELINE] = _all_ties(next(iter(stats.values())))
    record = {
        "night": night.isoformat(), "label": label, "probes": len(probes),
        "checkpoint": None if checkpoint is None else checkpoint.id,
        "checkpoint_basis": "promoted" if promoted else "not_promoted",
        "arms": dict(sorted(stats.items())), "pairs": pairs, "unavailable": unavailable, "outcomes": rows,
    }
    record["seconds"] = round(morning.timer() - started, 3)  # the arms and the judge, wall time
    if subset is not None:
        record["subset"] = subset.name
    if no_verdict:
        record[NO_VERDICT] = no_verdict
    if woke is not None:
        record["wake_seconds"] = woke.seconds
    folder.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.tmp")
    pending.write_text(json.dumps(record, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(pending, target)
    return MorningResult(target, sum(len(r) for r in replies.values()), judge.calls, sum(no_verdict.values()))


__all__ = ["ARMS", "DAILY", "Morning", "MorningError", "MorningResult", "ProbeSubset", "TooManyNoVerdicts", "run_morning",
           "subset_probes"]
