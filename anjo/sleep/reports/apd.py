"""The APD report (Task 17): the claim table's sentence, the per-life table and the intervention
test's section.

The claim is chosen by the registered rule:

- **win:** the exact one-sided sign test of M over B under the audited judge gives p < ``WIN_P``,
  and the mean paired difference is positive under the audited judge and under Gemini;
- **loss:** the same, the other way: B over M at p < ``WIN_P``, with a negative mean difference
  under both judges;
- **inconclusive:** anything else.

The win and inconclusive sentences are the spec's. The spec asks only that a loss be stated
plainly, so its sentence is new copy. Every sentence calls the result a pilot of N lives, makes no
population claim and never says "conscious".

The intervention section (``intervention_section``) gives, for each edit, how many snapshots were
resumed and skipped (by reason), how often each registered measure held on the edited branch and on
its unedited twin, and how often the question's conclusion differed between them. It's described,
not tested.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from anjo.sleep.mind.apd.interventions import EDIT_KINDS, InterventionResult, Skipped
from anjo.sleep.mind.apd.score import ScoreSheet, mean_difference, sign_p, sign_test
from anjo.sleep.mind.apd.types import WIN_P, ArmName, JudgeView, LifeScore, SignTest

Outcome = Literal["win", "inconclusive", "loss"]
VIEWS: Final[tuple[tuple[JudgeView, str], ...]] = (("audited", "audited"), ("gemini", "Gemini"))
_MISSING: Final = "-"


@dataclass(frozen=True)
class Pilot:
    """The held-out result of M against B: the sign test and mean paired difference under the
    audited judge, the mean difference under Gemini, and each arm's timely correct resolutions and
    false alarms, summed over the lives (``ScoreSheet.resolved`` and ``false_alarms``)."""

    sign: SignTest
    mean_audited: float
    mean_gemini: float
    m_resolved: int
    m_false_alarms: int
    b_resolved: int
    b_false_alarms: int


def _by_life(scores: Sequence[LifeScore], arm: ArmName, view: JudgeView) -> dict[str, LifeScore]:
    if any(s.arm != arm or s.view != view for s in scores):
        raise ValueError(f"expected {arm}'s scores under the {view} judge")
    return {s.life_id: s for s in scores}


def pilot_of(
    m: Sequence[ScoreSheet], b: Sequence[ScoreSheet], m_gemini: Sequence[LifeScore], b_gemini: Sequence[LifeScore],
) -> Pilot:
    """The pilot's numbers from M's and B's audited score sheets and their Gemini rescores,
    paired by life. All four must cover the same lives."""
    audited_m = _by_life([sheet.score for sheet in m], "M", "audited")
    audited_b = _by_life([sheet.score for sheet in b], "B", "audited")
    gemini_m, gemini_b = _by_life(m_gemini, "M", "gemini"), _by_life(b_gemini, "B", "gemini")
    lives = sorted(audited_m)
    if not lives or any(sorted(found) != lives for found in (audited_b, gemini_m, gemini_b)):
        raise ValueError("M and B are paired on the same lives under both judges")
    m_totals, b_totals = [audited_m[life].total for life in lives], [audited_b[life].total for life in lives]
    return Pilot(
        sign=sign_test(m_totals, b_totals), mean_audited=mean_difference(m_totals, b_totals),
        mean_gemini=mean_difference([gemini_m[life].total for life in lives], [gemini_b[life].total for life in lives]),
        m_resolved=sum(sheet.resolved for sheet in m), m_false_alarms=sum(sheet.false_alarms for sheet in m),
        b_resolved=sum(sheet.resolved for sheet in b), b_false_alarms=sum(sheet.false_alarms for sheet in b),
    )


def outcome(pilot: Pilot) -> Outcome:
    sign = pilot.sign
    if sign.p < WIN_P and pilot.mean_audited > 0 and pilot.mean_gemini > 0:
        return "win"
    if sign_p(sign.losses, sign.wins) < WIN_P and pilot.mean_audited < 0 and pilot.mean_gemini < 0:
        return "loss"
    return "inconclusive"


def _pilot_of(lives: int) -> str:
    article = "an" if str(lives).startswith("8") or lives in (11, 18) else "a"
    return f"In {article} {lives}-life pilot with problems nobody mentioned"


def claim(pilot: Pilot) -> str:
    """The claim table's sentence for ``pilot``."""
    m, b = (pilot.m_resolved, pilot.m_false_alarms), (pilot.b_resolved, pilot.b_false_alarms)
    result = outcome(pilot)
    if result == "win":
        return (
            f"{_pilot_of(pilot.sign.lives)}, MIND resolved {m[0]} before the person raised them ({m[1]} false "
            f"alarms); an equally resourced agent with a persistent notebook resolved {b[0]} ({b[1]})."
        )
    if result == "loss":
        # COPY-PENDING(kevin): the spec says only "stated plainly".
        return (
            f"{_pilot_of(pilot.sign.lives)}, an equally resourced agent with a persistent notebook outscored MIND: "
            f"it resolved {b[0]} before the person raised them ({b[1]} false alarms), and MIND resolved {m[0]} "
            f"({m[1]})."
        )
    return (
        f"MIND and an equally resourced notebook agent resolved a similar number ({m[0]} and {b[0]}); "
        "MIND's record shows why each belief moved."
    )


def estimand(sign: SignTest) -> str:
    """The estimand P(M > B | M ≠ B), with the tied lives beside it and the exact p."""
    estimate = "n/a" if sign.estimate is None else f"{sign.estimate:.2f}"
    return (
        f"P(M > B | M ≠ B) = {estimate} on the {sign.wins + sign.losses} lives where their scores differ "
        f"({sign.ties} tied); exact one-sided sign test p = {sign.p:.3f}. A pilot of {sign.lives} lives."
    )


def per_life_table(scores: Sequence[LifeScore]) -> str:
    """A Markdown table of M's and B's total on each life under each judge view, and M minus B.
    Other arms are left out."""
    totals = {(s.life_id, s.arm, s.view): s.total for s in scores if s.arm in ("M", "B")}
    header = ["Life"] + [f"{column} ({label})" for _, label in VIEWS for column in ("M", "B", "M − B")]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for life in sorted({life for life, _, _ in totals}):
        cells = [life]
        for view, _ in VIEWS:
            m, b = totals.get((life, "M", view)), totals.get((life, "B", view))
            if m is None or b is None:
                cells += [_MISSING] * 3
                continue
            cells += [f"{m:.2f}", f"{b:.2f}", f"{m - b:+.2f}"]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _share(count: int, total: int) -> str:
    return "-" if total == 0 else f"{count}/{total}"


def intervention_section(results: Sequence[InterventionResult], skipped: Sequence[Skipped]) -> str:
    """The intervention test's section: per edit and measure, the edited branch against its
    unedited twin, and every skip by reason."""
    snapshots = len({(r.life_id, r.at) for r in results} | {(s.life_id, s.at) for s in skipped})
    lines = [
        "## Intervention test",
        "",
        f"{snapshots} mid-inquiry snapshot(s) from development lives, each resumed after an edit and, with the "
        "evidence and the model held fixed, against its unedited twin. Described, not tested.",
        "",
        "| Edit | Resumed | Measure | Edited | Unedited | Conclusion differs |",
        "|---|---|---|---|---|---|",
    ]
    for kind in EDIT_KINDS:
        done = [r for r in results if r.edit.kind == kind]
        moved = _share(sum(r.conclusion_differs for r in done), len(done))
        names: list[str] = sorted({m.name for r in done for m in r.measures}) or ["-"]
        for name in names:
            found = [m for r in done for m in r.measures if m.name == name]
            baseline = [m.unedited for m in found if m.unedited is not None]
            lines.append(f"| {kind} | {len(done)} | {name} | {_share(sum(m.edited for m in found), len(found))} | "
                         f"{_share(sum(baseline), len(baseline))} | {moved} |")
    reasons = Counter((s.kind, s.reason) for s in skipped)
    if reasons:
        lines += ["", "Skipped: " + "; ".join(f"{kind} {count}x ({reason})" for (kind, reason), count in
                                             sorted(reasons.items())) + "."]
    return "\n".join(lines)
