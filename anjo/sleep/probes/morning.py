"""The morning probe run (Task 12a, spec 5.1): the sealed probes, rerun unchanged, on the checkpoint
that is serving after the night.

``run_morning`` answers each of the person's sealed probes with every arm (``arms.reply``), judges
each arm against arm 1 in both orders (``score.score``; ``taste_ab`` by log-probability, no judge),
and writes ``scores/night-NN.json`` in the layout ``reports.results`` reads: per arm and dimension,
the tie-inclusive score against arm 1 (arm 1 is 0.5 against itself by definition), plus the two
headline pairs (3 against 3x, 4 against 2C). The file holds numbers, the night, the label and the
checkpoint id, never a reply or a probe, and nothing is written to history, evidence or memory (RF5).

- A night that already has a score file is refused before any call: the first score stands.
- The checkpoint given must be the one serving; arms 3 and 4 are pinned to it for the whole run.
- Arm 3x needs the registered donor's promoted adapter. Without one, 3x is left out and recorded
  under ``unavailable`` (specificity is then inconclusive); it is never filled in.
- Every paid call is held first, the way ``night/holds.py`` does it: the arms on ``modal_usd`` and
  ``baseten_sleep_usd``, the judge on ``meta_usd``. The judge's hold is taken before any arm runs, so
  a pool without room stops the run before the first call.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Bound, Ledger
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import held, usd
from anjo.sleep.probes.arms import ARMS, Arm, ArmDeps, ArmReply, DonorError, reply
from anjo.sleep.probes.clients import AdapterRegistry
from anjo.sleep.probes.score import (
    REPORTED,
    BootstrapConfig,
    PairScore,
    PairwiseJudge,
    ProbeOutcome,
    Verdict,
    pair_outcomes,
    score_outcomes,
)
from anjo.sleep.reports import results
from anjo.sleep.types import CheckpointRef, Probe

ARM_ORDER: Final[tuple[Arm, ...]] = ("1", "2C", "2F", "3", "3x", "4")
BASELINE: Final[Arm] = "1"
PAIRS: Final[dict[str, tuple[Arm, Arm]]] = {"3_vs_3x": ("3", "3x"), "4_vs_2C": ("4", "2C")}
ARM_CALL_S: Final = 60.0  # the most one arm reply holds the serving GPU
JUDGE_USD_PER_CALL: Final = 0.02  # a Muse Spark judge call, until the Oct 30 timing measures it
ARM_POOLS: Final = ("modal_usd", "baseten_sleep_usd")
JUDGE_POOLS: Final = ("meta_usd",)
SCORES: Final = "scores"
DONOR_UNAVAILABLE: Final = "donor_unavailable"


class MorningError(RuntimeError):
    """The morning run was refused: a rerun, a stale checkpoint, or a probe of another person."""


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


@dataclass(frozen=True)
class MorningResult:
    path: Path
    replies: int
    judge_calls: int


class _Pinned:
    """The adapter registry with the person's serving checkpoint fixed for the whole run."""

    def __init__(self, inner: AdapterRegistry, person_id: str, checkpoint: CheckpointRef | None) -> None:
        self._inner, self._person, self._checkpoint = inner, person_id, checkpoint

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        return self._checkpoint if person_id == self._person else self._inner.last_promoted(person_id)


class _Counting:
    def __init__(self, inner: PairwiseJudge) -> None:
        self._inner = inner
        self.calls = 0

    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        self.calls += 1
        return self._inner.verdict(probe, first, second)


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


def _rows(pair: str, probes: Sequence[Probe], found: Sequence[ProbeOutcome]) -> list[dict[str, object]]:
    """Ids and codes only: enough to recount any result, with no text."""
    return [{"probe": f.probe_id, "pair": pair, "kind": f.kind, "outcome": f.outcome, "flip": f.flipped}
            for f in found]


def _answer(
    user_id: str, probes: Sequence[Probe], deps: ArmDeps,
) -> tuple[dict[Arm, list[ArmReply]], dict[str, str]]:
    replies: dict[Arm, list[ArmReply]] = {}
    unavailable: dict[str, str] = {}
    for arm in ARM_ORDER:
        try:
            replies[arm] = [reply(arm, user_id, p, deps=deps) for p in probes]
        except DonorError:
            if arm != "3x":
                raise
            unavailable[arm] = DONOR_UNAVAILABLE  # a code: score files hold ids, codes and numbers only
    return replies, unavailable


def _worst_judge_calls(probes: Sequence[Probe]) -> int:
    judged = sum(1 for p in probes if p.kind != "taste_ab")
    return (len(ARM_ORDER) - 1 + len(PAIRS)) * judged * 2


def run_morning(
    user_id: str, night: date, checkpoint: CheckpointRef | None, *, morning: Morning,
    label: str = "development", promoted: bool = True,
) -> MorningResult:
    """Score every arm on ``user_id``'s sealed probes, on the checkpoint serving after ``night``.

    ``checkpoint`` is the night's own (``NightResult.checkpoint``) and must be the one serving now;
    anything else is refused. A night that promoted nothing passes ``promoted=False``: the checkpoint
    that was serving is scored, and the file says ``not_promoted``."""
    if label not in results.LABELS:
        raise ValueError(f"label is one of {', '.join(results.LABELS)}, got {label!r}")
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
    if not probes:
        raise MorningError(f"{user_id} has no sealed probes")
    if any(p.person_id != user_id for p in probes):
        raise MorningError("a sealed probe belongs to another person")
    deps = replace(morning.arms, adapters=_Pinned(morning.arms.adapters, user_id, checkpoint), until=night,
                   context_cache={})
    judge = _Counting(morning.judge)
    key = f"morning:{night.isoformat()}:{user_id}"
    arm_s = len(probes) * len(ARM_ORDER) * ARM_CALL_S
    started = morning.timer()
    with ExitStack() as holds:
        holds.enter_context(held(morning.ledger, f"{key}:judge", JUDGE_POOLS,
                                 usd(_worst_judge_calls(probes) * JUDGE_USD_PER_CALL),
                                 lambda: judge.calls * JUDGE_USD_PER_CALL))
        holds.enter_context(held(morning.ledger, f"{key}:arms", ARM_POOLS,
                                 Bound(morning.prices.serve_gpu_per_hour, arm_s),
                                 lambda: morning.prices.serve_gpu_per_hour * max(0.0, morning.timer() - started) / 3600.0))
        replies, unavailable = _answer(user_id, probes, deps)
        config = BootstrapConfig.from_thresholds()
        rows: list[dict[str, object]] = []
        stats: dict[str, dict[str, dict[str, object]]] = {}
        pairs: dict[str, dict[str, dict[str, object]]] = {}

        def compare(name: str, first: Arm, second: Arm, into: dict[str, dict[str, dict[str, object]]], key: str) -> None:
            found = pair_outcomes(probes, replies[first], replies[second], judge)
            into[key] = _stats(score_outcomes(probes, [f.outcome for f in found], bootstrap=config))
            rows.extend(_rows(name, probes, found))

        for arm in ARM_ORDER:
            if arm != BASELINE and arm in replies:
                compare(f"{arm}_vs_1", arm, BASELINE, stats, arm)
        for name, (first, second) in PAIRS.items():
            if first in replies and second in replies:
                compare(name, first, second, pairs, name)
    stats[BASELINE] = _all_ties(next(iter(stats.values())))
    record = {
        "night": night.isoformat(), "label": label, "probes": len(probes),
        "checkpoint": None if checkpoint is None else checkpoint.id,
        "checkpoint_basis": "promoted" if promoted else "not_promoted",
        "arms": dict(sorted(stats.items())), "pairs": pairs, "unavailable": unavailable, "outcomes": rows,
    }
    folder.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.tmp")
    pending.write_text(json.dumps(record, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(pending, target)
    return MorningResult(target, sum(len(r) for r in replies.values()), judge.calls)


__all__ = ["ARMS", "Morning", "MorningError", "MorningResult", "run_morning"]
