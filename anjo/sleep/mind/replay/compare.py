"""The 17b test's arms (Task 17b): M, M+replay and M+shuffled on the held-out APD lives.

All three are Task 17's MIND arm (``MindArm()``) on the same world query, screen, resolver, brief
and budget; only the thinker differs: base Glimmer for M, thinker-1's adapter for M+replay and
thinker-shuffled's for M+shuffled (``served.arm_thinker``), each prompt dated on its night
(the arms set the thinkers' shared clock as each night starts). They run on held-out lives only, and
each run is scored by Task 17's ``score_sheet`` under the audited judge and rescored under Gemini,
as M and B are; ``reports/replay.py`` applies Task 17's primary rule to the pairs.

**A failed call voids the comparison.** Task 17's APD arms have no rule for a thinker call that
fails (the port returns empty text charged its cap, and the run goes on). Here every thinker
records how its calls ended, ``ArmRuns.failed`` counts the failed ones per arm, and an arm with
any is never scored: ``score_replay_arms`` refuses, and the report says the comparison is invalid
(``reports.replay.invalid``), with no sentence that any arm beats another.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from anjo.sleep.mind.apd.arms import MindCore, mind_core, run_arm
from anjo.sleep.mind.apd.score import ApdJudge, ScoreSheet, score_life, score_sheet
from anjo.sleep.mind.apd.types import ArmModels, ArmName, ArmRun, Life, LifeScore, ProblemFile
from anjo.sleep.mind.inquiry.types import Budget
from anjo.sleep.mind.replay.sample import RecordingThinker
from anjo.sleep.mind.replay.served import NightClock, failed_calls
from anjo.sleep.mind.replay.spend import Holds, held_gpu

REPLAY_ARMS: Final[tuple[ArmName, ...]] = ("M", "M+replay", "M+shuffled")


@dataclass(frozen=True)
class ArmThinkers:
    """The one thing the three arms don't share, and the clock their prompts are dated by."""

    m: RecordingThinker
    replay: RecordingThinker
    shuffled: RecordingThinker
    clock: NightClock = field(default_factory=NightClock, compare=False)

    def of(self, arm: ArmName) -> RecordingThinker:
        found: dict[ArmName, RecordingThinker] = {"M": self.m, "M+replay": self.replay, "M+shuffled": self.shuffled}
        return found[arm]


@dataclass(frozen=True)
class ArmRuns:
    """Every run, and each arm's failed thinker calls."""

    runs: tuple[ArmRun, ...]
    failed: Mapping[ArmName, int] = field(hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "failed", MappingProxyType(dict(self.failed)))

    @property
    def valid(self) -> bool:
        return not any(self.failed.values())


@dataclass(frozen=True)
class ArmScores:
    """Every run's audited score sheet and its Gemini rescore."""

    audited: tuple[ScoreSheet, ...]
    gemini: tuple[LifeScore, ...]


def run_replay_arms(
    lives: Sequence[Life], models: ArmModels, thinkers: ArmThinkers, *, budget: Budget = Budget(),
    core: MindCore | None = None, brief: str | None = None, holds: Holds | None = None,
) -> ArmRuns:
    """Each held-out life under each of the three arms; ``models.thinker`` is replaced per arm.
    With ``holds``, each run holds its Glimmer time first (``spend``)."""
    if any(life.split != "held_out" for life in lives):
        raise ValueError("the 17b test runs on held-out lives only")
    mind = core or mind_core()
    runs: list[ArmRun] = []
    for life in lives:
        for arm in REPLAY_ARMS:
            with held_gpu(holds, f"arm:{life.id}:{arm}", len(life.days)):
                arm_models = dataclasses.replace(models, thinker=thinkers.of(arm), clock=thinkers.clock)
                runs.append(run_arm(arm, life, arm_models, budget=budget, core=mind, brief=brief))
    return ArmRuns(tuple(runs), {arm: failed_calls(thinkers.of(arm).calls) for arm in REPLAY_ARMS})


def score_replay_arms(
    lives: Sequence[Life], problems: Mapping[str, ProblemFile], runs: ArmRuns, *, audited: ApdJudge,
    gemini: ApdJudge,
) -> ArmScores:
    """Every run scored, or none: an arm with a failed thinker call is never scored."""
    if not runs.valid:
        raise ValueError(f"an arm with a failed thinker call is never scored ({dict(runs.failed)})")
    held = {life.id: life for life in lives}
    return ArmScores(
        audited=tuple(score_sheet(held[r.life_id], problems[r.life_id], r, audited, view="audited") for r in runs.runs),
        gemini=tuple(score_life(held[r.life_id], problems[r.life_id], r, gemini, view="gemini") for r in runs.runs),
    )
