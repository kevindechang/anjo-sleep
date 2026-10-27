# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The APD report (Task 17-eval): the claim table picks the spec's sentence for a win, an
inconclusive result or a loss, and the per-life table shows both judge views."""
from __future__ import annotations

import dataclasses

import pytest

from anjo.sleep.mind.apd import score
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.reports import apd as report


def _sign(wins: int, losses: int, ties: int = 0) -> apd.SignTest:
    from anjo.sleep.mind.apd.score import sign_p

    return apd.SignTest(wins=wins, losses=losses, ties=ties, p=sign_p(wins, losses))


def _pilot(**changes: object) -> report.Pilot:
    base = report.Pilot(sign=_sign(12, 4), mean_audited=0.4, mean_gemini=0.3, m_resolved=9, m_false_alarms=2,
                        b_resolved=5, b_false_alarms=4)
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def test_claim_selector_uses_the_spec_sentences() -> None:
    win = _pilot()
    assert report.outcome(win) == "win"
    assert report.claim(win) == (
        "In a 16-life pilot with problems nobody mentioned, MIND resolved 9 before the person raised them "
        "(2 false alarms); an equally resourced agent with a persistent notebook resolved 5 (4)."
    )
    for name, pilot in {
        "not significant": _pilot(sign=_sign(11, 5)),
        "negative under Gemini": _pilot(mean_gemini=-0.1),
        "no mean lead": _pilot(mean_audited=0.0),
    }.items():
        assert report.outcome(pilot) == "inconclusive", name
        assert report.claim(pilot) == (
            "MIND and an equally resourced notebook agent resolved a similar number (9 and 5); "
            "MIND's record shows why each belief moved."
        ), name


def test_a_loss_is_stated_plainly() -> None:
    loss = _pilot(sign=_sign(3, 13), mean_audited=-0.4, mean_gemini=-0.2, m_resolved=2, b_resolved=8)
    assert report.outcome(loss) == "loss"
    assert report.claim(loss) == (
        "In a 16-life pilot with problems nobody mentioned, an equally resourced agent with a persistent notebook "
        "outscored MIND: it resolved 8 before the person raised them (4 false alarms), and MIND resolved 2 (2)."
    )
    assert report.outcome(dataclasses.replace(loss, mean_gemini=0.1)) == "inconclusive"
    for pilot in (loss, _pilot(), _pilot(sign=_sign(8, 8))):
        assert "conscious" not in report.claim(pilot).lower()


def test_pilot_size_takes_the_right_article() -> None:
    assert report.claim(_pilot(sign=_sign(10, 1, 1))).startswith("In a 12-life pilot")
    assert report.claim(_pilot(sign=_sign(8, 0, 3))).startswith("In an 11-life pilot")
    assert report.claim(_pilot(sign=_sign(8, 0))).startswith("In an 8-life pilot")


def test_estimand_reports_ties_beside_it() -> None:
    assert report.estimand(_sign(9, 3, 4)) == (
        "P(M > B | M ≠ B) = 0.75 on the 12 lives where their scores differ (4 tied); "
        "exact one-sided sign test p = 0.073. A pilot of 16 lives."
    )
    assert report.estimand(_sign(0, 0, 16)).startswith("P(M > B | M ≠ B) = n/a on the 0 lives")


def test_per_life_table_shows_both_views() -> None:
    def life_score(life: str, arm: apd.ArmName, view: apd.JudgeView, total: float) -> apd.LifeScore:
        items = (apd.ItemScore("i1", "contradiction", "f1" if total else None, total),)
        return apd.LifeScore(life, arm, view, items, ())

    scores = [
        life_score("life-2", "M", "audited", 1.0), life_score("life-2", "B", "audited", 0.25),
        life_score("life-1", "M", "audited", 0.0), life_score("life-1", "B", "audited", 1.0),
        life_score("life-1", "M", "gemini", 0.25), life_score("life-1", "B", "gemini", 1.0),
        life_score("life-1", "A", "audited", 1.0),  # other arms are left out
    ]
    assert report.per_life_table(scores).splitlines() == [
        "| Life | M (audited) | B (audited) | M − B (audited) | M (Gemini) | B (Gemini) | M − B (Gemini) |",
        "|---|---|---|---|---|---|---|",
        "| life-1 | 0.00 | 1.00 | -1.00 | 0.25 | 1.00 | -0.75 |",
        "| life-2 | 1.00 | 0.25 | +0.75 | - | - | - |",
    ]


def test_pilot_pairs_m_and_b_by_life_under_both_judges() -> None:
    def life_score(life: str, arm: apd.ArmName, view: apd.JudgeView, points: float) -> apd.LifeScore:
        return apd.LifeScore(life, arm, view, (apd.ItemScore("i1", "gap", "f1" if points else None, points),), ())

    def sheet(life: str, arm: apd.ArmName, points: float, resolved: int, alarms: int) -> score.ScoreSheet:
        return score.ScoreSheet(life_score(life, arm, "audited", points), (), resolved, alarms, ())

    m = [sheet("life-1", "M", 1.0, 1, 0), sheet("life-2", "M", 0.25, 0, 1)]
    b = [sheet("life-2", "B", 0.0, 0, 2), sheet("life-1", "B", 1.0, 1, 0)]
    gemini_m = [life_score("life-1", "M", "gemini", 0.25), life_score("life-2", "M", "gemini", 0.25)]
    gemini_b = [life_score("life-1", "B", "gemini", 0.0), life_score("life-2", "B", "gemini", 0.0)]
    pilot = report.pilot_of(m, b, gemini_m, gemini_b)
    assert (pilot.sign.wins, pilot.sign.losses, pilot.sign.ties) == (1, 0, 1)
    assert (pilot.mean_audited, pilot.mean_gemini) == (0.125, 0.25)
    assert (pilot.m_resolved, pilot.m_false_alarms, pilot.b_resolved, pilot.b_false_alarms) == (1, 1, 1, 2)
    with pytest.raises(ValueError, match="same lives"):
        report.pilot_of(m, b[:1], gemini_m, gemini_b)
    with pytest.raises(ValueError, match="Gemini|gemini"):
        report.pilot_of(m, b, gemini_b, gemini_m)
