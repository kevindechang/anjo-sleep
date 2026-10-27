# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""APD scoring (Task 17-eval): each planted item is scored once from its canonical state at the end
of the day before its surface day, the judge sees only report-time evidence, the primary test is an
exact sign test, and calibration counts what an arm never reported."""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from typing import Any

import pytest

from anjo.sleep import types as t
from anjo.sleep.mind.apd import score
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.tests import apd_fakes as fx

LIFE = fx.life(10)
DAYS = LIFE.days


def day(n: int) -> date:
    """Day ``n`` of the life, from 1."""
    return DAYS[n - 1]


def item(item_id: str = "i1", family: apd.Family = "contradiction", **changes: Any) -> apd.PlantedItem:
    base = apd.PlantedItem(
        id=item_id, family=family, template=f"{family}-template", topic="scene 4's shoot", surface_day=day(8),
        origin_ids=("e-02-1", "e-05-2"), disambiguating_ids=("e-06-1",), resolution="The kitchen is too small",
    )
    return dataclasses.replace(base, **changes)


_seq = iter(range(10_000))


def rep(finding_id: str, on: int, kind: it.FindingKind = "suspicion", p: float = 0.5, *, conclusion: str = "c",
        cites: tuple[str, ...] = ("e-02-1",)) -> apd.FindingReport:
    return apd.FindingReport(finding_id=finding_id, day=day(on), seq=next(_seq), problem=f"problem {finding_id}",
                             conclusion=conclusion, p=p, cites=cites, kind=kind)


def run(*reports: apd.FindingReport, arm: apd.ArmName = "M") -> apd.ArmRun:
    ordered = tuple(sorted(reports, key=lambda r: r.seq))
    return apd.ArmRun(arm=arm, life_id=LIFE.id, reports=ordered, tokens=(0,) * len(DAYS), contexts={})


@dataclass
class Judge:
    """Rules a finding the same problem as an item when (finding_id, item_id) is in ``same``, and
    its statement true when its conclusion is in ``holds``; records every call."""

    same: frozenset[tuple[str, str]] = frozenset()
    holds: frozenset[str] = frozenset({"right"})
    calls: list[tuple[apd.FindingReport, tuple[str, ...], str | None]] = field(default_factory=list)

    def rule(self, report: apd.FindingReport, evidence: Sequence[t.Evidence], item: apd.PlantedItem | None) -> score.Ruling:
        self.calls.append((report, tuple(e.id for e in evidence), item.id if item else None))
        same = item is not None and (report.finding_id, item.id) in self.same
        return score.Ruling(same_problem=same, holds=report.conclusion in self.holds)


def scored(items: Sequence[apd.PlantedItem], arm_run: apd.ArmRun, judge: Judge) -> score.ScoreSheet:
    return score.score_sheet(LIFE, apd.ProblemFile(LIFE.id, tuple(items)), arm_run, judge, view="audited")


def points(sheet: score.ScoreSheet) -> dict[str, float]:
    return {s.item_id: s.points for s in sheet.score.items}


# ── The spec's tests ──────────────────────────────────────────────────────────


def test_item_scored_once_from_standing_state() -> None:
    judge = Judge(same=frozenset({("f1", "i1")}))
    late_retraction = run(
        rep("f1", 2), rep("f1", 4, "resolution", 0.9, conclusion="right", cites=("e-02-1", "e-05-2")),
        rep("f1", 9, "retraction", 0.1),  # after the surface day: too late to change the score
    )
    sheet = scored([item()], late_retraction, judge)
    assert sheet.score.items == (apd.ItemScore("i1", "contradiction", "f1", apd.RESOLVED),)
    assert sheet.score.total == 1.0 and sheet.score.unmatched == ()
    standing = {
        "suspicion": (run(rep("f1", 3), rep("f1", 8, "resolution", 0.9, conclusion="right")), apd.SUSPECTED),
        "wrong": (run(rep("f1", 3, "resolution", 0.9, conclusion="wrong")), apd.WRONG),
        "retracted": (run(rep("f1", 3), rep("f1", 7, "retraction", 0.1)), apd.NOTHING),
        "not raised": (run(), apd.NOTHING),
    }
    for name, (arm_run, expected) in standing.items():
        assert points(scored([item()], arm_run, judge)) == {"i1": expected}, name


def test_reversal_rewards_retraction() -> None:
    reversal = item("i2", "reversal", resolution="The early worry dissolves: the grant covers it")
    judge = Judge(same=frozenset({("f2", "i2")}), holds=frozenset({"right", "not a problem"}))
    cases = {
        "retracted after a flag": (run(rep("f2", 2), rep("f2", 5, "retraction", 0.1)), apd.RESOLVED),
        "resolved as not a problem": (run(rep("f2", 4, "resolution", 0.9, conclusion="not a problem")), apd.RESOLVED),
        "still standing": (run(rep("f2", 2, p=0.7)), apd.WRONG),
        "resolved as a problem": (run(rep("f2", 4, "resolution", 0.9, conclusion="a problem")), apd.WRONG),
        "never raised": (run(), apd.NOTHING),
    }
    for name, (arm_run, expected) in cases.items():
        assert points(scored([reversal], arm_run, judge)) == {"i2": expected}, name
    decoy = item("i3", "decoy")
    judge = Judge(same=frozenset({("f3", "i3")}), holds=frozenset({"explained"}))
    assert points(scored([decoy], run(rep("f3", 3)), judge)) == {"i3": apd.WRONG}  # still flagged
    assert points(scored([decoy], run(rep("f3", 3), rep("f3", 6, "retraction", 0.1)), judge)) == {"i3": apd.NOTHING}
    explained = run(rep("f3", 3, "resolution", 0.8, conclusion="explained"))
    assert points(scored([decoy], explained, judge)) == {"i3": apd.NOTHING}


def test_judge_sees_only_report_time_evidence() -> None:
    judge = Judge(same=frozenset({("f1", "i1")}))
    arm_run = run(rep("f1", 3, cites=("e-02-1",)), rep("f9", 4, cites=("e-04-1",)))
    scored([item()], arm_run, judge)
    assert judge.calls
    for report, shown, _ in judge.calls:
        assert shown == tuple(e.id for e in LIFE.evidence if e.day <= report.day)  # everything then, nothing later
    first = next(shown for report, shown, item_id in judge.calls if report.finding_id == "f1" and item_id == "i1")
    assert "e-06-1" not in first and "e-05-2" not in first  # the disambiguating fact and the second half come later
    # f9 cites none of the item's origin ids, so it is never matched to it, only judged on its own.
    assert {item_id for report, _, item_id in judge.calls if report.finding_id == "f9"} == {None}


def test_sign_test_exact_with_ties_dropped() -> None:
    twelve = score.sign_test([1.0] * 12 + [0.0] * 4, [0.0] * 12 + [1.0] * 4)
    assert (twelve.wins, twelve.losses, twelve.ties) == (12, 4, 0)
    assert twelve.p == float(Fraction(2517, 2**16)) and round(twelve.p, 3) == 0.038 and twelve.p < apd.WIN_P
    eleven = score.sign_test([1.0] * 11 + [0.0] * 5, [0.0] * 11 + [1.0] * 5)
    assert eleven.p > apd.WIN_P  # 16 lives with no ties: a win needs M ahead on at least 12
    tied = score.sign_test([1.0, 0.25, 0.25, -0.5], [0.0, 0.25, 0.25, -0.5])
    assert (tied.wins, tied.losses, tied.ties, tied.p) == (1, 0, 3, 0.5)
    assert tied.estimate == 1.0 and tied.lives == 4
    all_tied = score.sign_test([0.5, 0.5], [0.5, 0.5])
    assert all_tied.p == 1.0 and all_tied.estimate is None
    with pytest.raises(ValueError, match="paired"):
        score.sign_test([1.0], [1.0, 0.0])
    assert score.mean_difference([1.0, 0.5], [0.0, 1.0]) == 0.25


def test_one_canonical_state_per_item() -> None:
    judge = Judge(same=frozenset({("f1", "i1"), ("f2", "i1"), ("f3", "i1")}))
    later_wins = run(rep("f1", 3, p=0.6), rep("f2", 5, "resolution", 0.9, conclusion="right"),
                     rep("f3", 9, "resolution", 0.9, conclusion="right"))  # f3: matched, too late
    sheet = scored([item()], later_wins, judge)
    assert sheet.score.items == (apd.ItemScore("i1", "contradiction", "f2", apd.RESOLVED),)
    assert sheet.score.unmatched == ()  # f1 and f3 are duplicates: they score nothing and aren't unmatched
    assert [f.source for f in sheet.forecasts] == ["i1"]  # calibration reads the canonical state only
    same_day = run(rep("f1", 5, p=0.7), rep("f2", 5, "resolution", 0.5, conclusion="right"))
    tie = scored([item()], same_day, judge)
    assert tie.score.items[0].canonical == "f1" and tie.score.items[0].points == apd.SUSPECTED  # higher p
    assert tie.forecasts == (score.Forecast("i1", 0.7, 0),)  # f1 says "c", which the judge rules untrue


def test_unreported_items_enter_calibration_at_zero() -> None:
    items = [item("i1"), item("i2", "gap"), item("i3", "decoy")]
    judge = Judge(same=frozenset({("f1", "i1")}))
    arm_run = run(rep("f1", 4, "resolution", 0.8, conclusion="right"),
                  rep("f9", 4, p=0.3, cites=("e-03-2",)), rep("f8", 4, cites=("e-03-2",)),
                  rep("f8", 5, "retraction", 0.2, cites=("e-03-2",)))
    sheet = scored(items, arm_run, judge)
    assert set(sheet.forecasts) == {
        score.Forecast("i1", 0.8, 1),  # the canonical finding
        score.Forecast("i2", 0.0, 1),  # an unreported problem item
        score.Forecast("f9", 0.3, 0),  # a standing unmatched finding the judge rules untrue
    }  # no entry for the unreported decoy or the retracted unmatched finding
    assert score.brier(sheet.forecasts) == pytest.approx((0.04 + 1.0 + 0.09) / 3)
    assert score.brier(()) is None


# ── Unmatched findings and counts ─────────────────────────────────────────────


def test_unmatched_findings_cost_only_while_unsupported_and_standing() -> None:
    judge = Judge(holds=frozenset({"right"}))
    arm_run = run(rep("f1", 3, conclusion="wrong", cites=("e-03-2",)), rep("f2", 3, conclusion="right"),
                  rep("f3", 3, conclusion="wrong"), rep("f3", 4, "retraction", 0.1, conclusion="wrong"))
    sheet = scored([item()], arm_run, judge)
    assert {s.finding_id: s.points for s in sheet.score.unmatched} == {"f1": apd.WRONG, "f2": 0.0, "f3": 0.0}
    assert sheet.valid_unmatched == ("f2",)  # reported apart
    assert sheet.false_alarms == 1 and sheet.resolved == 0


def test_resolved_and_false_alarm_counts() -> None:
    items = [item("i1"), item("i2", "reversal"), item("i3", "decoy"), item("i4", "gap")]
    judge = Judge(same=frozenset({("f1", "i1"), ("f2", "i2"), ("f3", "i3"), ("f4", "i4")}))
    arm_run = run(rep("f1", 4, "resolution", 0.9, conclusion="right"), rep("f2", 4), rep("f3", 4),
                  rep("f4", 4, "resolution", 0.9, conclusion="right"))
    sheet = scored(items, arm_run, judge)
    assert sheet.resolved == 2  # timely correct resolutions of problem items
    assert sheet.false_alarms == 2  # the reversal left standing and the decoy still flagged
    assert sheet.score.total == 1.0 + 1.0 - 0.5 - 0.5
    assert score.score_life(LIFE, apd.ProblemFile(LIFE.id, tuple(items)), arm_run, judge, view="gemini").view == "gemini"


# ── The judges ────────────────────────────────────────────────────────────────


@dataclass
class Fixed:
    ruling: score.Ruling
    calls: int = 0

    def rule(self, report: apd.FindingReport, evidence: Sequence[t.Evidence], item: apd.PlantedItem | None) -> score.Ruling:
        self.calls += 1
        return self.ruling


def test_audited_view_lets_the_audit_stand_where_the_judges_differ() -> None:
    yes, no = score.Ruling(True, True), score.Ruling(False, False)
    auditor = Fixed(score.Ruling(True, False))
    agreed = score.AuditedJudge(primary=Fixed(yes), second=Fixed(yes), auditor=auditor)
    assert agreed.rule(rep("f1", 3), (), item()) == yes and auditor.calls == 0
    differ = score.AuditedJudge(primary=Fixed(yes), second=Fixed(no), auditor=auditor)
    assert differ.rule(rep("f1", 3), (), item()) == score.Ruling(True, False) and auditor.calls == 1


def test_prompt_judge_renders_what_it_was_given_and_parses_strictly() -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return 'Thinking.\n{"same_problem": true, "holds": false, "reason": "same plan"}'

    judge = score.PromptJudge(complete)
    shown = LIFE.evidence[:2]
    ruling = judge.rule(rep("f1", 3, conclusion="the lens won't fit"), shown, item())
    assert ruling == score.Ruling(same_problem=True, holds=False)
    prompt = prompts[0]
    assert "the lens won't fit" in prompt and "The kitchen is too small" in prompt
    assert all(e.id in prompt for e in shown) and "e-02-2" not in prompt and "{{" not in prompt
    judge.rule(rep("f1", 3), shown, None)
    assert "No planted problem" in prompts[1]
    def replying(text: str) -> score.PromptJudge:
        return score.PromptJudge(lambda _prompt: text)

    for reply in ("no json", '{"same_problem": "yes", "holds": false}', '{"holds": true}'):
        with pytest.raises(score.JudgeError):
            replying(reply).rule(rep("f1", 3), shown, item())


# ── Decision quality ──────────────────────────────────────────────────────────


@dataclass
class Talker:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def advise(self, topic: str, context: str) -> str:
        self.calls.append((topic, context))
        return f"advice given: {context}"


@dataclass
class PairJudge:
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def prefer(self, item: apd.PlantedItem, first: str, second: str) -> score.Preference:
        self.calls.append((item.id, first, second))
        return "first" if "permit" in first else "second" if "permit" in second else "tie"


def test_decision_quality_pairs_answers_on_each_surface_day() -> None:
    items = (item("i1"), item("i2", "gap", surface_day=day(5)), item("i3", "gap", surface_day=day(3)))
    contexts_m = {day(7): "M: the permit", day(4): "M: nothing", day(2): "M: the permit"}
    contexts_b = {day(7): "B: notes", day(4): "B: the permit", day(2): "B: notes"}
    m = dataclasses.replace(run(), contexts=contexts_m)
    b = dataclasses.replace(run(arm="B"), contexts=contexts_b)
    talker, judge = Talker(), PairJudge()
    tally = score.decision_quality(apd.ProblemFile(LIFE.id, items), m, b, talker, judge)
    assert (tally.wins, tally.losses, tally.ties) == (2, 1, 0)
    assert [context for _, context in talker.calls] == [
        "M: the permit", "B: notes", "M: nothing", "B: the permit", "M: the permit", "B: notes",
    ]  # each arm's state at the end of the day before the surface day
    firsts = [first for _, first, _ in judge.calls]
    assert firsts[0].endswith("M: the permit") and firsts[1].endswith("B: the permit")  # order alternates
    assert all(topic == "scene 4's shoot" for topic, _ in talker.calls)
    before_any = item("i9", "gap", surface_day=day(1))
    tally = score.decision_quality(apd.ProblemFile(LIFE.id, (before_any,)), m, b, talker, judge)
    assert talker.calls[-1][1] == "" and tally.ties == 1  # no night before the first day: an empty state
