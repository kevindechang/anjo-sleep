"""The 17b test's arms (Task 17b): M, M+replay and M+shuffled on the held-out APD lives.

All three are Task 17's MIND arm (``MindArm()``) on the same world query, screen, resolver, brief
and budget; only the thinker differs: base Glimmer for M, thinker-1's adapter for M+replay and
thinker-shuffled's for M+shuffled (``served.arm_thinker``). They run on held-out lives only, and
each run is scored by Task 17's ``score_sheet`` under the audited judge and rescored under Gemini,
as M and B are; ``reports/replay.py`` applies Task 17's primary rule to the pairs.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.mind.apd.arms import MindCore, run_arm
from anjo.sleep.mind.apd.score import ApdJudge, ScoreSheet, score_life, score_sheet
from anjo.sleep.mind.apd.types import ArmModels, ArmName, ArmRun, Life, LifeScore, ProblemFile
from anjo.sleep.mind.inquiry.ports import Thinker
from anjo.sleep.mind.inquiry.types import Budget

REPLAY_ARMS: Final[tuple[ArmName, ...]] = ("M", "M+replay", "M+shuffled")


@dataclass(frozen=True)
class ArmThinkers:
    """The one thing the three arms don't share."""

    m: Thinker
    replay: Thinker
    shuffled: Thinker

    def of(self, arm: ArmName) -> Thinker:
        found: dict[ArmName, Thinker] = {"M": self.m, "M+replay": self.replay, "M+shuffled": self.shuffled}
        return found[arm]


@dataclass(frozen=True)
class ArmScores:
    """Every run's audited score sheet and its Gemini rescore."""

    audited: tuple[ScoreSheet, ...]
    gemini: tuple[LifeScore, ...]


def run_replay_arms(
    lives: Sequence[Life], models: ArmModels, thinkers: ArmThinkers, *, budget: Budget = Budget(),
    core: MindCore | None = None, brief: str | None = None,
) -> tuple[ArmRun, ...]:
    """Each held-out life under each of the three arms; ``models.thinker`` is replaced per arm."""
    if any(life.split != "held_out" for life in lives):
        raise ValueError("the 17b test runs on held-out lives only")
    return tuple(
        run_arm(arm, life, dataclasses.replace(models, thinker=thinkers.of(arm)), budget=budget, core=core, brief=brief)
        for life in lives for arm in REPLAY_ARMS
    )


def score_replay_arms(
    lives: Sequence[Life], problems: Mapping[str, ProblemFile], runs: Sequence[ArmRun], *, audited: ApdJudge,
    gemini: ApdJudge,
) -> ArmScores:
    held = {life.id: life for life in lives}
    return ArmScores(
        audited=tuple(score_sheet(held[r.life_id], problems[r.life_id], r, audited, view="audited") for r in runs),
        gemini=tuple(score_life(held[r.life_id], problems[r.life_id], r, gemini, view="gemini") for r in runs),
    )
