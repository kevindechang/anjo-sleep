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
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Protocol

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.template import render
from anjo.sleep.probes.arms import ArmReply
from anjo.sleep.probes.neutral import rewrite as neutral_rewrite
from anjo.sleep.types import Probe

Verdict = Literal["A", "B", "tie"]
JUDGE_PROMPT: Final = Path(__file__).parent / "prompts" / "judge_v1.md"
THRESHOLDS: Final = Path(__file__).parents[1] / "prereg" / "thresholds.json"
JUDGE_MODEL: Final = "muse-spark"
JUDGE_TEMPERATURE: Final = 0.0
JUDGE_TIMEOUT_S: Final = 180.0
NEUTRAL_KINDS: Final = frozenset({"new_work", "vocab"})
REPORTED: Final = "reported"
_POINTS: Final = {"win": 1.0, "tie": 0.5, "loss": 0.0}
BlockKey = tuple[str, str, str]


class JudgeError(RuntimeError):
    """A judge's answer was not one of A, B or tie."""


# ── Judges ────────────────────────────────────────────────────────────────────


class PairwiseJudge(Protocol):
    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        """Which of the two replies, shown in this order, is more useful to this person."""


def judge_prompt(probe: Probe, brief: str, first: str, second: str) -> str:
    return render(JUDGE_PROMPT.read_text(encoding="utf-8"), {
        "BRIEF": brief, "KIND": probe.kind, "PROMPT": probe.prompt,
        "REPLY_A": first, "REPLY_B": second,
    })


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


@dataclass(frozen=True)
class MuseJudge:
    """Muse Spark with ``judge_v1.md``. ``brief_for`` gives the judge brief (``judge_brief``)."""

    chat: ChatFn
    brief_for: Callable[[str], str]

    def verdict(self, probe: Probe, first: str, second: str) -> Verdict:
        prompt = judge_prompt(probe, self.brief_for(probe.person_id), first, second)
        result = self.chat(
            JUDGE_MODEL, [{"role": "user", "content": prompt}], None, JUDGE_TIMEOUT_S,
            temperature=JUDGE_TEMPERATURE,
        )
        return parse_verdict(result.text)


Outcome = Literal["win", "tie", "loss"]


def judge_both_orders(judge: PairwiseJudge, probe: Probe, reply_a: str, reply_b: str) -> Outcome:
    """Arm A's outcome: judged with A first and again with B first; a flip is a tie."""
    seen_first = {"A": "win", "B": "loss", "tie": "tie"}[judge.verdict(probe, reply_a, reply_b)]
    seen_second = {"A": "loss", "B": "win", "tie": "tie"}[judge.verdict(probe, reply_b, reply_a)]
    return seen_first if seen_first == seen_second else "tie"  # type: ignore[return-value]


def taste_outcome(probe: Probe, reply_a: ArmReply, reply_b: ArmReply) -> Outcome:
    """Arm A's outcome on a forced choice: the better standing wins (a flip stands between)."""
    if reply_a.choice is None or reply_b.choice is None:
        raise ValueError(f"{probe.id} is taste_ab: both replies need a forced-choice result (choice)")
    standing_a = reply_a.choice.standing(probe.expected)
    standing_b = reply_b.choice.standing(probe.expected)
    if standing_a == standing_b:
        return "tie"
    return "win" if standing_a > standing_b else "loss"


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
    these lives, and is not a population estimate."""

    wins: int
    ties: int
    losses: int
    ci: tuple[float, float]
    per_life: Mapping[str, Tally] = field(hash=False)
    per_dimension: Mapping[str, Tally] = field(hash=False)
    per_kind: Mapping[str, Tally] = field(hash=False)

    def __post_init__(self) -> None:
        for name in ("per_life", "per_dimension", "per_kind"):
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
    return blocks


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
    return list(groups.values())


def _interval(values: list[float], ci: float) -> tuple[float, float]:
    values.sort()
    tail = (1.0 - ci) / 2.0
    low = values[int(tail * len(values))]
    high = values[min(len(values) - 1, int((1.0 - tail) * len(values)))]
    return low, high


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
    rng = random.Random(config.seed)
    overall: list[float] = []
    by_dimension: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(config.resamples):
        total = [0.0, 0]
        parts = {name: [0.0, 0] for name in names}
        for group in strata:
            for key in rng.choices(group, k=len(group)):
                dim, got, count = stats[key]
                total[0] += got
                total[1] += count
                parts[dim][0] += got
                parts[dim][1] += count
        overall.append(total[0] / total[1])
        for name in names:
            by_dimension[name].append(parts[name][0] / parts[name][1])
    return (
        _interval(overall, config.ci),
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
    if not len(probes) == len(replies_a) == len(replies_b):
        raise ValueError("probes, replies_a and replies_b must align one to one")
    if neutral and any(p.kind not in NEUTRAL_KINDS for p in probes):
        raise ValueError("neutral re-judging runs on new_work and vocab only")
    config = bootstrap if bootstrap is not None else BootstrapConfig.from_thresholds()
    outcomes: list[Outcome] = []
    for probe, reply_a, reply_b in zip(probes, replies_a, replies_b, strict=True):
        if probe.kind == "taste_ab":
            outcomes.append(taste_outcome(probe, reply_a, reply_b))
            continue
        text_a, text_b = reply_a.text, reply_b.text
        if neutral:
            text_a, text_b = rewrite(text_a), rewrite(text_b)
        outcomes.append(judge_both_orders(judge, probe, text_a, text_b))
    total = _tally(outcomes)
    if not probes:
        return PairScore(0, 0, 0, (0.5, 0.5), {}, {}, {})
    points = [_POINTS[outcome] for outcome in outcomes]
    interval, by_dimension = _bootstrap(probes, points, config)
    dims = dimension_of()
    dimension_tallies = {
        name: Tally(t.wins, t.ties, t.losses, by_dimension[name])
        for name, t in ((n, _tally(o)) for n, o in _grouped(probes, outcomes, lambda p: dims[p.kind]).items())
    }
    return PairScore(
        total.wins, total.ties, total.losses, interval,
        per_life={k: _tally(v) for k, v in _grouped(probes, outcomes, lambda p: p.person_id).items()},
        per_dimension=dimension_tallies,
        per_kind={k: _tally(v) for k, v in _grouped(probes, outcomes, lambda p: p.kind).items()},
    )
