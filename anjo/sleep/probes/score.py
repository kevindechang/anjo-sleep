"""Pairwise scoring (Task 2, spec 5.1 and 5.1b).

``score`` compares two arms' replies to the same probes:

- ``taste_ab`` by log-probability forced choice (``logprob``): no judge, no sampling.
- Every other kind by a pairwise judge, run in both orders. A verdict that flips with the order
  counts as a tie, and ties count half.
- ``neutral=True`` rewrites both replies into a plain voice first (``new_work`` and ``vocab``
  only, as a diagnostic).

``PairScore`` carries the tie-inclusive score, the decisive share, the paired block-bootstrap
interval and the per-life, per-dimension and per-kind results. The interval resamples within
each (life, kind): a block is one held-out source item with every probe derived from it, or a
single probe for ``vocab``, ``tone_hold`` and ``pushback``. A shared ``taste_ab`` item is one
block across the lives that share it. The resamples, confidence and seed come from the registered
``prereg/thresholds.json``, which this module reads and never writes.
"""
from __future__ import annotations

import json
import math
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast, runtime_checkable

from anjo.sleep.context.template import render
from anjo.sleep import baseten_api, fireworks_api
from anjo.sleep.model_api import ChatResult
from anjo.sleep.probes.arms import ArmReply
from anjo.sleep.probes.logprob import expected_margin
from anjo.sleep.probes.neutral import rewrite as neutral_rewrite
from anjo.sleep.types import Probe

Verdict = Literal["A", "B", "tie"]
JUDGE_PROMPT: Final = Path(__file__).parent / "prompts" / "judge_v1.md"
THRESHOLDS: Final = Path(__file__).parents[1] / "prereg" / "thresholds.json"
JUDGE_MODEL: Final = "muse-spark"
JUDGE_TEMPERATURE: Final = 0.0
JUDGE_TIMEOUT_S: Final = 180.0
# A judge answer that can't be read is asked once more with this cap (sent as max_completion_tokens).
# The first call sends none, so the provider's default applies. Teo's night 1 (2026-10-28, 257 calls
# of muse-spark-1.3) wrote 797 to 8,944 tokens (p50 2,558, p99 7,233); its one unreadable answer came
# among calls of 1.8k to 4.5k, nowhere near a cap. 16,384 is 1.8 times the longest seen.
JUDGE_RETRY_MAX_TOKENS: Final = 16_384
NEUTRAL_KINDS: Final = frozenset({"new_work", "vocab"})
REPORTED: Final = "reported"
_POINTS: Final = {"win": 1.0, "tie": 0.5, "loss": 0.0}
BlockKey = tuple[str, str, str]


class JudgeError(RuntimeError):
    """A judge's answer was not one of A, B or tie."""


class NoVerdict(JudgeError):
    """A probe the judge gave no readable verdict on, after its one retry: counted and shown, never
    scored, never a crash."""


# ── Judges ────────────────────────────────────────────────────────────────────


class PairwiseJudge(Protocol):
    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        """Which of the two replies, shown in this order, is more useful to this person."""


@runtime_checkable
class RoomyJudge(PairwiseJudge, Protocol):
    def again(self, probe: Probe, first: str, second: str) -> Verdict:
        """``verdict`` asked once more, with more room to answer (``JUDGE_RETRY_MAX_TOKENS``)."""


class JudgeChat(Protocol):
    """``meta_api.chat`` fits this: a chat whose reply can be capped (``max_tokens``)."""

    def __call__(
        self, model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, timeout_s: float, *,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> ChatResult: ...


def judge_prompt(probe: Probe, brief: str, first: str, second: str) -> str:
    return cast(str, render(JUDGE_PROMPT.read_text(encoding="utf-8"), {
        "BRIEF": brief, "KIND": probe.kind, "PROMPT": probe.prompt,
        "REPLY_A": first, "REPLY_B": second,
    }))


def parse_verdict(text: str) -> Verdict:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise JudgeError("the judge's answer is not JSON") from exc
    verdict = data.get("verdict") if isinstance(data, dict) else None
    if verdict == "A" or verdict == "B" or verdict == "tie":
        return verdict
    raise JudgeError("the judge's verdict is not A, B or tie")


class JudgeTokens:
    """The judge's calls that reported their usage, and their tokens in and out, from any thread. A
    reply that reports none (0 in and 0 out) is left out: the morning settles it at its hold's per-call
    bound and marks it estimated."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls = self._in = self._out = 0

    def record(self, result: ChatResult) -> None:
        if result.input_tokens <= 0 and result.output_tokens <= 0:
            return
        with self._lock:
            self._calls, self._in, self._out = (self._calls + 1, self._in + result.input_tokens,
                                                self._out + result.output_tokens)

    def read(self) -> tuple[int, int, int]:
        """Calls with usage, tokens in, tokens out."""
        with self._lock:
            return self._calls, self._in, self._out


@dataclass(frozen=True)
class MuseJudge:
    """Muse Spark with ``judge_v1.md``. ``brief_for`` gives the judge brief (``judge_brief``); ``tokens``,
    when given, records every reply's usage, an unreadable one's too (it is billed all the same)."""

    chat: JudgeChat
    brief_for: Callable[[str], str]
    tokens: JudgeTokens | None = None
    model: str = JUDGE_MODEL

    def __post_init__(self) -> None:
        if self.model in (baseten_api.SPARK_OPEN, fireworks_api.SPARK_OPEN, "muse-spark-open"):
            raise ValueError("Spark Open cannot be used as a judge")
        if self.model not in (JUDGE_MODEL, fireworks_api.DEEPSEEK_V4P1_FLASH):
            raise ValueError("unsupported judge model")

    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        return self._ask(probe, first, second, None)

    def again(self, probe: Probe, first: str, second: str) -> Verdict:
        return self._ask(probe, first, second, JUDGE_RETRY_MAX_TOKENS)

    def _ask(self, probe: Probe, first: str, second: str, max_tokens: int | None) -> Verdict:
        """One judge call; an unreadable answer's error names how it ended and its length, never its words."""
        prompt = judge_prompt(probe, self.brief_for(probe.person_id), first, second)
        messages = [{"role": "user", "content": prompt}]
        result = (self.chat(self.model, messages, None, JUDGE_TIMEOUT_S, temperature=JUDGE_TEMPERATURE)
                  if max_tokens is None else
                  self.chat(self.model, messages, None, JUDGE_TIMEOUT_S, temperature=JUDGE_TEMPERATURE,
                            max_tokens=max_tokens))
        if self.tokens is not None:
            self.tokens.record(result)
        try:
            return parse_verdict(result.text)
        except JudgeError as error:
            raise JudgeError(f"{error} (finish {str(result.finish_reason)[:16]}, {result.output_tokens} tokens out)") from None


Outcome = Literal["win", "tie", "loss"]


def judge_orders(judge: PairwiseJudge, probe: Probe, reply_a: str, reply_b: str) -> tuple[Outcome, bool]:
    """Arm A's outcome, and whether the two orders disagreed (which makes it a tie)."""
    seen_first = {"A": "win", "B": "loss", "tie": "tie"}[judge.verdict(probe, reply_a, reply_b)]
    seen_second = {"A": "loss", "B": "win", "tie": "tie"}[judge.verdict(probe, reply_b, reply_a)]
    if seen_first == seen_second:
        return seen_first, False  # type: ignore[return-value]
    return "tie", True


def judge_both_orders(judge: PairwiseJudge, probe: Probe, reply_a: str, reply_b: str) -> Outcome:
    """Arm A's outcome: judged with A first and again with B first; a flip is a tie."""
    return judge_orders(judge, probe, reply_a, reply_b)[0]


def taste_outcome(probe: Probe, reply_a: ArmReply, reply_b: ArmReply) -> Outcome:
    """Arm A's outcome on a forced choice. An order flip by either arm is a tie (spec 5.1);
    otherwise the arm that picked the expected option wins, and equal picks tie."""
    if reply_a.choice is None or reply_b.choice is None:
        raise ValueError(f"{probe.id} is taste_ab: both replies need a forced-choice result (choice)")
    right_a = reply_a.choice.correct(probe.expected)
    right_b = reply_b.choice.correct(probe.expected)
    if right_a is None or right_b is None or right_a == right_b:
        return "tie"
    return "win" if right_a else "loss"


# ── Registered rules ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BootstrapConfig:
    resamples: int
    seed: int
    ci: float

    @classmethod
    def from_thresholds(cls, path: Path = THRESHOLDS) -> BootstrapConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))["bootstrap"]
        return cls(int(raw["resamples"]), int(raw["seed"]), float(raw["ci"]))


@lru_cache(maxsize=1)
def dimension_of() -> Mapping[str, str]:
    """probe kind -> dimension (knowledge, interaction), pushback as ``reported``."""
    raw = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    mapping = {kind: name for name, dim in raw["dimensions"].items() for kind in dim["kinds"]}
    mapping.update({kind: REPORTED for kind in raw["reported_only_kinds"]})
    return MappingProxyType(mapping)


# ── Results ───────────────────────────────────────────────────────────────────


class _Counts:
    wins: int
    ties: int
    losses: int

    @property
    def n(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def score(self) -> float:
        """Tie-inclusive: a win counts 1, a tie half. 0.5 when there is nothing to score."""
        return (self.wins + 0.5 * self.ties) / self.n if self.n else 0.5

    @property
    def decisive_share(self) -> float | None:
        """W / (W + L), or ``None`` when no pair was decisive."""
        decided = self.wins + self.losses
        return self.wins / decided if decided else None


@dataclass(frozen=True)
class Tally(_Counts):
    wins: int
    ties: int
    losses: int
    ci: tuple[float, float] | None = None


@dataclass(frozen=True)
class PairScore(_Counts):
    """Arm A against arm B over the probes given. ``ci`` is the block-bootstrap interval of
    ``score``; each dimension carries its own. It covers these lives' probes, conditional on
    these lives, and is not a population estimate.

    The headline (``wins``, ``ties``, ``losses``, ``score``, ``decisive_share``, ``ci``) leaves
    out the reported-only kinds (``pushback``); they appear in ``per_kind`` and in the
    ``reported`` dimension only. The registered win rule applies per dimension, so "ahead in 2
    of 3 lives" reads ``per_life_dimension[(life, dimension)]``; ``per_life`` pools every
    dimension and is descriptive only."""

    wins: int
    ties: int
    losses: int
    ci: tuple[float, float]
    per_life: Mapping[str, Tally] = field(hash=False)
    per_dimension: Mapping[str, Tally] = field(hash=False)
    per_kind: Mapping[str, Tally] = field(hash=False)
    per_life_dimension: Mapping[tuple[str, str], Tally] = field(hash=False)

    def __post_init__(self) -> None:
        for name in ("per_life", "per_dimension", "per_kind", "per_life_dimension"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _tally(outcomes: Sequence[Outcome]) -> Tally:
    return Tally(outcomes.count("win"), outcomes.count("tie"), outcomes.count("loss"))


# ── The block bootstrap ───────────────────────────────────────────────────────


def _block_key(probe: Probe) -> BlockKey:
    source = probe.source_ids[0] if probe.source_ids else probe.id
    if probe.kind == "taste_ab":
        return (probe.kind, "", source)  # shared across the lives that share the item
    if probe.kind == "new_work":
        return (probe.kind, probe.person_id, source)
    return (probe.kind, probe.person_id, probe.id)


def bootstrap_blocks(probes: Sequence[Probe]) -> dict[BlockKey, list[int]]:
    """Block key -> the indices of the probes in it."""
    blocks: dict[BlockKey, list[int]] = {}
    for index, probe in enumerate(probes):
        blocks.setdefault(_block_key(probe), []).append(index)
    return dict(sorted(blocks.items()))


def _strata(probes: Sequence[Probe], blocks: Mapping[BlockKey, list[int]]) -> list[list[BlockKey]]:
    """Blocks grouped by (life, kind). Lives that share a ``taste_ab`` item form one group."""
    parent: dict[str, str] = {}

    def find(life: str) -> str:
        parent.setdefault(life, life)
        while parent[life] != life:
            parent[life] = parent[parent[life]]
            life = parent[life]
        return life

    for key, indices in blocks.items():
        if key[0] == "taste_ab":
            lives = sorted({probes[i].person_id for i in indices})
            for other in lives[1:]:
                parent[find(other)] = find(lives[0])
    groups: dict[tuple[str, str], list[BlockKey]] = {}
    for key, indices in blocks.items():
        life = find(probes[indices[0]].person_id) if key[0] == "taste_ab" else key[1]
        groups.setdefault((key[0], life), []).append(key)
    return [sorted(groups[name]) for name in sorted(groups)]


def _rank(q: float, n: int) -> int:
    """Nearest-rank index ``ceil(q * n) - 1``, clamped to the sample. The product is rounded
    first so float error (0.975 * 10000) can't move it a whole rank."""
    return min(n - 1, max(0, math.ceil(round(q * n, 9)) - 1))


def _interval(values: list[float], ci: float) -> tuple[float, float]:
    """The percentile interval by one convention for both ends: nearest-rank, ``values[ceil(q*n) - 1]``
    for q = tail and q = 1 - tail. ``thresholds.json`` names no other method."""
    values.sort()
    tail = (1.0 - ci) / 2.0
    return values[_rank(tail, len(values))], values[_rank(1.0 - tail, len(values))]


def _bootstrap(
    probes: Sequence[Probe], points: Sequence[float], config: BootstrapConfig,
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    blocks = bootstrap_blocks(probes)
    dims = dimension_of()
    stats = {
        key: (dims[probes[idx[0]].kind], sum(points[i] for i in idx), len(idx))
        for key, idx in blocks.items()
    }
    strata = _strata(probes, blocks)
    names = sorted({dim for dim, _, _ in stats.values()})
    for key in stats:  # sums of floats: add in a fixed order, whatever the probe order was
        dim, _, count = stats[key]
        stats[key] = (dim, math.fsum(points[i] for i in blocks[key]), count)
    rng = random.Random(config.seed)
    overall: list[float] = []
    by_dimension: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(config.resamples):
        total = [0.0, 0]  # the headline: reported-only kinds are not in it
        parts = {name: [0.0, 0] for name in names}
        for group in strata:
            for key in rng.choices(group, k=len(group)):
                dim, got, count = stats[key]
                if dim != REPORTED:
                    total[0] += got
                    total[1] += count
                parts[dim][0] += got
                parts[dim][1] += count
        if total[1]:
            overall.append(total[0] / total[1])
        for name in names:
            by_dimension[name].append(parts[name][0] / parts[name][1])
    return (
        _interval(overall, config.ci) if overall else (0.5, 0.5),
        {name: _interval(values, config.ci) for name, values in by_dimension.items()},
    )


# ── score ─────────────────────────────────────────────────────────────────────


def _grouped(
    probes: Sequence[Probe], outcomes: Sequence[Outcome], by: Callable[[Probe], str],
) -> dict[str, list[Outcome]]:
    groups: dict[str, list[Outcome]] = {}
    for probe, outcome in zip(probes, outcomes, strict=True):
        groups.setdefault(by(probe), []).append(outcome)
    return groups


def _grouped2(
    probes: Sequence[Probe], outcomes: Sequence[Outcome], dims: Mapping[str, str],
) -> dict[tuple[str, str], list[Outcome]]:
    groups: dict[tuple[str, str], list[Outcome]] = {}
    for probe, outcome in zip(probes, outcomes, strict=True):
        groups.setdefault((probe.person_id, dims[probe.kind]), []).append(outcome)
    return groups


@dataclass(frozen=True)
class ProbeOutcome:
    """One probe's outcome for arm A against arm B, with whether an order flip made it a tie.
    ``margin_a`` and ``margin_b`` are the two arms' log-probability margins toward the expected
    option (``logprob.expected_margin``): a ``taste_ab`` diagnostic, ``None`` for every other kind."""

    probe_id: str
    kind: str
    outcome: Outcome
    flipped: bool
    margin_a: float | None = None
    margin_b: float | None = None


def pair_outcomes(
    probes: Sequence[Probe],
    replies_a: Sequence[ArmReply],
    replies_b: Sequence[ArmReply],
    judge: PairwiseJudge,
    *,
    neutral: bool = False,
    rewrite: Callable[[str], str] = neutral_rewrite,
) -> list[ProbeOutcome]:
    """Every probe's outcome for arm A against arm B. The three sequences line up by index."""
    if not len(probes) == len(replies_a) == len(replies_b):
        raise ValueError("probes, replies_a and replies_b must align one to one")
    if neutral and any(p.kind not in NEUTRAL_KINDS for p in probes):
        raise ValueError("neutral re-judging runs on new_work and vocab only")
    return [probe_outcome(probe, reply_a, reply_b, judge, neutral=neutral, rewrite=rewrite)
            for probe, reply_a, reply_b in zip(probes, replies_a, replies_b, strict=True)]


def probe_outcome(
    probe: Probe, reply_a: ArmReply, reply_b: ArmReply, judge: PairwiseJudge, *, neutral: bool = False,
    rewrite: Callable[[str], str] = neutral_rewrite,
) -> ProbeOutcome:
    """One probe's outcome for arm A against arm B: forced choice for ``taste_ab``, else the judge
    in both orders."""
    if probe.kind == "taste_ab":
        choice_a, choice_b = reply_a.choice, reply_b.choice
        flip = choice_a is not None and choice_b is not None and (choice_a.flipped or choice_b.flipped)
        outcome = taste_outcome(probe, reply_a, reply_b)
        return ProbeOutcome(
            probe.id, probe.kind, outcome, flip,
            None if choice_a is None else expected_margin(choice_a, probe),
            None if choice_b is None else expected_margin(choice_b, probe),
        )
    text_a, text_b = reply_a.text, reply_b.text
    if neutral:
        text_a, text_b = rewrite(text_a), rewrite(text_b)
    outcome, flip = judge_orders(judge, probe, text_a, text_b)
    return ProbeOutcome(probe.id, probe.kind, outcome, flip)


def score(
    probes: Sequence[Probe],
    replies_a: Sequence[ArmReply],
    replies_b: Sequence[ArmReply],
    judge: PairwiseJudge,
    *,
    neutral: bool = False,
    rewrite: Callable[[str], str] = neutral_rewrite,
    bootstrap: BootstrapConfig | None = None,
) -> PairScore:
    """Arm A's ``PairScore`` against arm B. The three sequences line up by index.

    ``rewrite`` is the style-neutral rewrite used when ``neutral`` is true (Muse Spark by
    default); ``bootstrap`` defaults to the registered thresholds.
    """
    found = pair_outcomes(probes, replies_a, replies_b, judge, neutral=neutral, rewrite=rewrite)
    return score_outcomes(probes, [f.outcome for f in found], bootstrap=bootstrap)


def score_outcomes(
    probes: Sequence[Probe], outcomes: Sequence[Outcome], *, bootstrap: BootstrapConfig | None = None,
) -> PairScore:
    """The ``PairScore`` of per-probe outcomes (the second half of ``score``)."""
    config = bootstrap if bootstrap is not None else BootstrapConfig.from_thresholds()
    dims = dimension_of()
    total = _tally([o for p, o in zip(probes, outcomes, strict=True) if dims[p.kind] != REPORTED])
    if not probes:
        return PairScore(0, 0, 0, (0.5, 0.5), {}, {}, {}, {})
    points = [_POINTS[outcome] for outcome in outcomes]
    interval, by_dimension = _bootstrap(probes, points, config)
    dimension_tallies = {
        name: Tally(t.wins, t.ties, t.losses, by_dimension[name])
        for name, t in ((n, _tally(o)) for n, o in _grouped(probes, outcomes, lambda p: dims[p.kind]).items())
    }
    return PairScore(
        total.wins, total.ties, total.losses, interval,
        per_life={k: _tally(v) for k, v in _grouped(probes, outcomes, lambda p: p.person_id).items()},
        per_dimension=dimension_tallies,
        per_kind={k: _tally(v) for k, v in _grouped(probes, outcomes, lambda p: p.kind).items()},
        per_life_dimension={
            (life, dim): _tally(v)
            for (life, dim), v in _grouped2(probes, outcomes, dims).items()
        },
    )
