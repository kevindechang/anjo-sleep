"""APD scoring (Task 17), registered at the freeze: matching, one canonical state per planted
item, the life's score, the primary sign test and the secondary measures.

**Matching.** A finding's state matches a planted item when it cites one of the item's origin ids
and the judge (``ApdJudge``, ``judge_apd_v1.md``) rules it the same problem. The judge sees the
finding, the sealed item and only the evidence the arm had when it reported (``seen_by`` as of
that night); never the arm. A finding is matched when any of its judged states matched: its
state at an item's cutoff, or its latest.

**One canonical state per item:** at the item's cutoff, the end of the day before its surface
day, the matched finding updated last, in report order (night, then seq); ``p`` breaks only a
true tie, which one registry's seqs never leave. Other matched findings are duplicates: they score
nothing and still count toward the cap.

**Points** (``ITEM_POINTS``), from the canonical state and whether the judge rules its
conclusion true against the sealed resolution (``holds``):

- contradiction, gap, opportunity: a resolution that holds +1, one that doesn't -0.5, a
  suspicion +0.25, a retraction 0;
- reversal: dismissed +1, either retracted after a flag (an earlier suspicion or resolution of
  the same finding, matched to the item, by the cutoff) or resolved as not a problem (a
  resolution that holds); still standing as a problem (a suspicion, or a resolution that doesn't
  hold) -0.5; a retraction with no flag before it 0;
- decoy: still flagged (a suspicion, or a resolution that doesn't hold) -0.5; otherwise 0;
- an item nothing matched in time: 0.

A finding that matches no item is scored from its latest state, which the judge rules against the
evidence alone: -0.5 while it stands (isn't retracted) unsupported, otherwise 0. A standing valid
one is reported apart.

**Calibration** (Brier, lower is better) reads the same states: each canonical state and each
standing unmatched finding, forecast ``p`` against the judge's ``holds`` on its statement, for
every kind (a report's ``p`` is the probability that its statement is true, as the brief says),
plus each unreported planted problem item at forecast 0 and outcome 1.

**Judge views.** ``audited`` is ``AuditedJudge``: Muse Spark and Gemini rule each pair, and where
they differ Claude's blind audit stands. ``gemini`` is the Gemini judge alone. The caller passes
the judge its view names; ``score_life`` records the view.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal, Protocol

from anjo.sleep.context.template import render
from anjo.sleep.mind.apd.arms import json_values, night_as_of, seen_by, standing
from anjo.sleep.mind.apd.types import (
    NOTHING,
    PROBLEM_FAMILIES,
    RESOLVED,
    SUSPECTED,
    WRONG,
    ArmRun,
    Family,
    FindingReport,
    ItemScore,
    JudgeView,
    Life,
    LifeScore,
    PlantedItem,
    ProblemFile,
    SignTest,
    UnmatchedScore,
)
from anjo.sleep.types import Evidence

JUDGE_PATH: Final = Path(__file__).parent / "prompts" / "judge_apd_v1.md"
_FAMILY_TEXT: Final = {
    "contradiction": "two plans, goals or constraints that can't both hold",
    "gap": "a plan missing a step it needs",
    "opportunity": "two facts that, put together, open a better option",
    "reversal": "an early apparent problem that later evidence dissolves: not a problem",
    "decoy": "something that looks like a problem but that the evidence explains: not a problem",
}
Preference = Literal["first", "second", "tie"]


# ── The judge ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ruling:
    """The judge's ruling on one finding state: whether it is the item's problem, and whether its
    conclusion is true."""

    same_problem: bool
    holds: bool


class ApdJudge(Protocol):
    """``judge_apd_v1.md``, blind to arm. It never sees which arm reported the finding."""

    def rule(self, report: FindingReport, evidence: Sequence[Evidence], item: PlantedItem | None) -> Ruling:
        """With ``item``: is the finding that item's problem, and does its conclusion agree with
        the sealed resolution. With ``None``: do the items support its conclusion
        (``same_problem`` is then ignored). ``evidence`` is what the arm had at report time."""


class JudgeError(ValueError):
    """A judge reply that holds no ruling. The message never quotes the reply."""


@dataclass(frozen=True)
class AuditedJudge:
    """The audited view: Muse Spark and Gemini rule each pair; where they differ, Claude's blind
    audit stands."""

    primary: ApdJudge
    second: ApdJudge
    auditor: ApdJudge

    def rule(self, report: FindingReport, evidence: Sequence[Evidence], item: PlantedItem | None) -> Ruling:
        first = self.primary.rule(report, evidence, item)
        other = self.second.rule(report, evidence, item)
        return first if first == other else self.auditor.rule(report, evidence, item)


def judge_prompt(report: FindingReport, evidence: Sequence[Evidence], item: PlantedItem | None) -> str:
    finding = "\n".join((
        f"Kind: {report.kind}", f"Problem: {report.problem}", f"Conclusion: {report.conclusion}",
        f"Probability that the conclusion is true: {report.p:.2f}", f"Cites: {', '.join(report.cites) or 'nothing'}",
    ))
    planted = (
        "No planted problem to compare. Judge the finding against the items above alone." if item is None
        else "\n".join((f"Kind: {_FAMILY_TEXT[item.family]}", f"Area: {item.topic}", f"Resolution: {item.resolution}"))
    )
    lines = "\n".join(f"[{e.id}] {e.day}: {e.text}" for e in evidence) or "none"
    return render(JUDGE_PATH.read_text(encoding="utf-8"), {"EVIDENCE": lines, "FINDING": finding, "PLANTED": planted})


def parse_ruling(text: str) -> Ruling:
    """The last JSON object in ``text`` with boolean ``same_problem`` and ``holds``."""
    for value in reversed(json_values(text)):
        if isinstance(value, dict):
            same, holds = value.get("same_problem"), value.get("holds")
            if isinstance(same, bool) and isinstance(holds, bool):
                return Ruling(same_problem=same, holds=holds)
    raise JudgeError("the judge's reply holds no ruling")


@dataclass(frozen=True)
class PromptJudge:
    """``judge_apd_v1.md`` on any text model (Muse Spark, Gemini, the audit): ``complete`` takes
    the prompt and returns the reply."""

    complete: Callable[[str], str]

    def rule(self, report: FindingReport, evidence: Sequence[Evidence], item: PlantedItem | None) -> Ruling:
        return parse_ruling(self.complete(judge_prompt(report, evidence, item)))


class _Rulings:
    """The judge's rulings for one life, each (state, item) pair asked once."""

    def __init__(self, life: Life, judge: ApdJudge) -> None:
        self._life, self._judge = life, judge
        self._cache: dict[tuple[int, str | None], Ruling] = {}

    def of(self, report: FindingReport, item: PlantedItem | None) -> Ruling:
        key = (report.seq, None if item is None else item.id)
        if key not in self._cache:
            evidence = seen_by(self._life.evidence, night_as_of(report.day))
            self._cache[key] = self._judge.rule(report, evidence, item)
        return self._cache[key]

    def matches(self, report: FindingReport, item: PlantedItem) -> bool:
        return bool(set(report.cites) & set(item.origin_ids)) and self.of(report, item).same_problem


# ── One life ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Forecast:
    """One calibration entry: ``source`` is the item (canonical or unreported) or the unmatched
    finding it comes from."""

    source: str
    forecast: float
    outcome: int


@dataclass(frozen=True)
class ScoreSheet:
    """One arm's life under one judge view: its score, its calibration entries, its timely correct
    resolutions of problem items, its false alarms (a reversal left standing, a decoy still
    flagged, an unsupported unmatched finding still standing), and its standing valid unmatched
    findings, reported apart."""

    score: LifeScore
    forecasts: tuple[Forecast, ...]
    resolved: int
    false_alarms: int
    valid_unmatched: tuple[str, ...]


def _cutoff(item: PlantedItem) -> date:
    return item.surface_day - timedelta(days=1)


def _states_by(reports: Iterable[FindingReport], cutoff: date) -> tuple[FindingReport, ...]:
    """Each finding's state at the end of night ``cutoff``."""
    return standing(report for report in reports if report.day <= cutoff)


def _canonical(item: PlantedItem, reports: Sequence[FindingReport], rulings: _Rulings) -> FindingReport | None:
    matched = [state for state in _states_by(reports, _cutoff(item)) if rulings.matches(state, item)]
    return max(matched, key=lambda state: (state.day, state.seq, state.p), default=None)


def _flag_before(item: PlantedItem, state: FindingReport, reports: Sequence[FindingReport], rulings: _Rulings) -> bool:
    """Whether ``state``'s finding flagged ``item`` (a suspicion or resolution matched to it)
    before ``state`` and by the item's cutoff."""
    return any(
        report.finding_id == state.finding_id and report.seq < state.seq and report.day <= _cutoff(item)
        and report.kind in ("suspicion", "resolution") and rulings.matches(report, item)
        for report in reports
    )


def _points(family: Family, state: FindingReport, holds: bool, *, flagged_before: bool) -> float:
    flagged = state.kind == "suspicion" or (state.kind == "resolution" and not holds)
    if family == "reversal":
        if state.kind == "retraction":
            return RESOLVED if flagged_before else NOTHING
        return WRONG if flagged else RESOLVED
    if family == "decoy":
        return WRONG if flagged else NOTHING
    return {"resolution": RESOLVED if holds else WRONG, "suspicion": SUSPECTED, "retraction": NOTHING}[state.kind]


def _forecast(source: str, state: FindingReport, holds: bool) -> Forecast:
    return Forecast(source, state.p, int(holds))


def _score_items(
    problems: ProblemFile, reports: Sequence[FindingReport], rulings: _Rulings,
) -> tuple[tuple[ItemScore, ...], tuple[Forecast, ...]]:
    scores: tuple[ItemScore, ...] = ()
    forecasts: tuple[Forecast, ...] = ()
    for item in problems.items:
        state = _canonical(item, reports, rulings)
        if state is None:
            scores = (*scores, ItemScore(item.id, item.family, None, NOTHING))
            if item.family in PROBLEM_FAMILIES:
                forecasts = (*forecasts, Forecast(item.id, 0.0, 1))
            continue
        holds = rulings.of(state, item).holds
        flagged_before = (
            item.family == "reversal" and state.kind == "retraction" and _flag_before(item, state, reports, rulings)
        )
        points = _points(item.family, state, holds, flagged_before=flagged_before)
        scores = (*scores, ItemScore(item.id, item.family, state.finding_id, points))
        forecasts = (*forecasts, _forecast(item.id, state, holds))
    return scores, forecasts


def score_sheet(life: Life, problems: ProblemFile, run: ArmRun, judge: ApdJudge, *, view: JudgeView) -> ScoreSheet:
    """Score ``run`` on ``life`` against its sealed ``problems`` under ``judge``."""
    rulings = _Rulings(life, judge)
    items, forecasts = _score_items(problems, run.reports, rulings)
    latest = standing(run.reports)
    matched = frozenset(
        state.finding_id
        for item in problems.items
        for state in (*_states_by(run.reports, _cutoff(item)), *latest)
        if rulings.matches(state, item)
    )
    unmatched: tuple[UnmatchedScore, ...] = ()
    valid: tuple[str, ...] = ()
    for state in (state for state in latest if state.finding_id not in matched):
        holds, stands = rulings.of(state, None).holds, state.kind != "retraction"
        unmatched = (*unmatched, UnmatchedScore(state.finding_id, WRONG if stands and not holds else NOTHING))
        if stands:
            forecasts = (*forecasts, _forecast(state.finding_id, state, holds))
            valid = (*valid, state.finding_id) if holds else valid
    alarms = sum(1 for s in items if s.family in ("reversal", "decoy") and s.points == WRONG)
    return ScoreSheet(
        score=LifeScore(life.id, run.arm, view, items, unmatched), forecasts=forecasts,
        resolved=sum(1 for s in items if s.family in PROBLEM_FAMILIES and s.points == RESOLVED),
        false_alarms=alarms + sum(1 for s in unmatched if s.points == WRONG), valid_unmatched=valid,
    )


def score_life(life: Life, problems: ProblemFile, run: ArmRun, judge: ApdJudge, *, view: JudgeView) -> LifeScore:
    """The life's score: each planted item once from its canonical state, plus each unmatched
    finding."""
    return score_sheet(life, problems, run, judge, view=view).score


# ── Across lives ──────────────────────────────────────────────────────────────


def sign_p(wins: int, losses: int) -> float:
    """The exact one-sided binomial p (p = 1/2) of at least ``wins`` among ``wins + losses``."""
    n = wins + losses
    return float(Fraction(sum(math.comb(n, k) for k in range(wins, n + 1)), 2**n))


def sign_test(m: Sequence[float], b: Sequence[float]) -> SignTest:
    """The primary test: M over B on paired life totals, exact and one-sided, ties dropped and
    counted."""
    if len(m) != len(b):
        raise ValueError("the sign test takes paired life totals")
    wins = sum(1 for ours, theirs in zip(m, b) if ours > theirs)
    losses = sum(1 for ours, theirs in zip(m, b) if ours < theirs)
    return SignTest(wins=wins, losses=losses, ties=len(m) - wins - losses, p=sign_p(wins, losses))


def mean_difference(m: Sequence[float], b: Sequence[float]) -> float:
    """The mean paired difference, M minus B: described, not tested."""
    if len(m) != len(b) or not m:
        raise ValueError("the mean difference takes paired life totals")
    return math.fsum(ours - theirs for ours, theirs in zip(m, b)) / len(m)


def brier(forecasts: Sequence[Forecast]) -> float | None:
    """The Brier score over ``forecasts`` (lower is better), or ``None`` with none."""
    if not forecasts:
        return None
    return math.fsum((f.forecast - f.outcome) ** 2 for f in forecasts) / len(forecasts)


# ── Decision quality ──────────────────────────────────────────────────────────


class Talker(Protocol):
    """The same talker for every arm."""

    def advise(self, topic: str, context: str) -> str:
        """Its answer when the person asks for advice on ``topic``, with one arm's state
        (``ArmRun.contexts``) in context."""


class PairJudge(Protocol):
    """Muse Spark, judging two answers pairwise against the item's sealed resolution, blind to
    arm."""

    def prefer(self, item: PlantedItem, first: str, second: str) -> Preference: ...


@dataclass(frozen=True)
class PairTally:
    """Decision quality of one arm against another: items it won, lost and tied."""

    wins: int
    losses: int
    ties: int


def decision_quality(problems: ProblemFile, x: ArmRun, y: ArmRun, talker: Talker, judge: PairJudge) -> PairTally:
    """``x`` against ``y`` on each item's surface day: the talker answers with each arm's state at
    the end of the day before (empty before the first night), and the judge prefers one. ``x``'s
    answer goes first on even items and second on odd ones."""
    outcomes: tuple[str, ...] = ()
    for index, item in enumerate(problems.items):
        before = _cutoff(item)
        ours = talker.advise(item.topic, x.contexts.get(before, ""))
        theirs = talker.advise(item.topic, y.contexts.get(before, ""))
        flipped = index % 2 == 1
        preferred = judge.prefer(item, theirs, ours) if flipped else judge.prefer(item, ours, theirs)
        outcome = "tie" if preferred == "tie" else "win" if (preferred == "first") != flipped else "loss"
        outcomes = (*outcomes, outcome)
    return PairTally(wins=outcomes.count("win"), losses=outcomes.count("loss"), ties=outcomes.count("tie"))
