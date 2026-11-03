"""The 17b report section (Task 17b): M+replay against M and against M+shuffled.

Each pair is judged on the held-out lives by Task 17's primary rule: the exact one-sided sign
test of M+replay over the other arm on paired life totals under the audited judge gives
p < ``WIN_P``, and the mean paired difference is positive under the audited judge and under
Gemini (``beats``). Replay is claimed only if it beats both, M (did replay help?) and M+shuffled
(did choosing the best branch help, beyond training on any branch?).

The claim's words are the spec's; the sentence for no claim is new copy. Both wait on Kevin. No
sentence calls it overnight personal learning, and none says "next-investigation choice": that
step isn't isolated here. Every sentence calls the result a pilot of N lives. When any arm's
thinker call failed, no arm is scored, and the section says only that the comparison is invalid
(``invalid``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.mind.apd.score import mean_difference, sign_test
from anjo.sleep.mind.apd.types import WIN_P, ArmName, JudgeView, LifeScore, SignTest
from anjo.sleep.reports.apd import VIEWS

REPLAY: Final[ArmName] = "M+replay"
OTHERS: Final[tuple[ArmName, ...]] = ("M", "M+shuffled")
_MISSING: Final = "-"


@dataclass(frozen=True)
class Versus:
    """M+replay against ``other`` on the held-out lives: the sign test and mean paired difference
    under the audited judge, and the mean paired difference under Gemini."""

    other: ArmName
    sign: SignTest
    mean_audited: float
    mean_gemini: float


def _totals(scores: Sequence[LifeScore], arm: ArmName, view: JudgeView) -> dict[str, float]:
    return {s.life_id: s.total for s in scores if s.arm == arm and s.view == view}


def versus(audited: Sequence[LifeScore], gemini: Sequence[LifeScore], other: ArmName) -> Versus:
    """The pair's numbers, paired by life. Both arms must cover the same lives under both judges."""
    ours_a, theirs_a = _totals(audited, REPLAY, "audited"), _totals(audited, other, "audited")
    ours_g, theirs_g = _totals(gemini, REPLAY, "gemini"), _totals(gemini, other, "gemini")
    lives = sorted(ours_a)
    if not lives or any(sorted(found) != lives for found in (theirs_a, ours_g, theirs_g)):
        raise ValueError(f"{REPLAY} and {other} are paired on the same lives under both judges")
    ours, theirs = [ours_a[life] for life in lives], [theirs_a[life] for life in lives]
    return Versus(
        other=other, sign=sign_test(ours, theirs), mean_audited=mean_difference(ours, theirs),
        mean_gemini=mean_difference([ours_g[life] for life in lives], [theirs_g[life] for life in lives]),
    )


def beats(pair: Versus) -> bool:
    """Task 17's primary rule, M+replay in M's place and ``pair.other`` in B's."""
    return pair.sign.p < WIN_P and pair.mean_audited > 0 and pair.mean_gemini > 0


def claimed(pairs: Sequence[Versus]) -> bool:
    """Replay is claimed only if it beats both M and M+shuffled."""
    return sorted(pair.other for pair in pairs) == sorted(OTHERS) and all(beats(pair) for pair in pairs)


def claim(pairs: Sequence[Versus]) -> str:
    lives = pairs[0].sign.lives
    if claimed(pairs):
        # Copy: the spec's claim, in a pilot's words.
        return f"In a {lives}-life pilot, replaying simulated lives improved her inquiry performance."
    # Copy: the spec gives no sentence for no claim.
    return (
        f"In a {lives}-life pilot, replaying simulated lives didn't beat both the unreplayed thinker and a "
        "thinker trained on random branches, so we make no claim for it."
    )


def _pair_line(pair: Versus) -> str:
    estimate = "n/a" if pair.sign.estimate is None else f"{pair.sign.estimate:.2f}"
    verdict = "beats it" if beats(pair) else "doesn't beat it"
    return (
        f"- {REPLAY} against {pair.other}: P({REPLAY} > {pair.other} | they differ) = {estimate} on "
        f"{pair.sign.wins + pair.sign.losses} lives ({pair.sign.ties} tied); exact one-sided sign test "
        f"p = {pair.sign.p:.3f}; mean paired difference {pair.mean_audited:+.2f} (audited), "
        f"{pair.mean_gemini:+.2f} (Gemini). By Task 17's rule it {verdict}."
    )


def table(scores: Sequence[LifeScore]) -> str:
    """A Markdown table of the three arms' totals on each life under each judge view."""
    arms: tuple[ArmName, ...] = ("M", REPLAY, "M+shuffled")
    totals = {(s.life_id, s.arm, s.view): s.total for s in scores}
    header = ["Life"] + [f"{arm} ({label})" for _, label in VIEWS for arm in arms]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for life in sorted({life for life, _, _ in totals}):
        cells = [life] + [
            _MISSING if (life, arm, view) not in totals else f"{totals[(life, arm, view)]:.2f}"
            for view, _ in VIEWS for arm in arms
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def section(audited: Sequence[LifeScore], gemini: Sequence[LifeScore]) -> str:
    """The report's replay section: the claim, both pairs and the per-life table."""
    pairs = [versus(audited, gemini, other) for other in OTHERS]
    return "\n\n".join((
        "## Replay (Task 17b)", claim(pairs), "\n".join(_pair_line(pair) for pair in pairs),
        table([*audited, *gemini]),
    )) + "\n"


def invalid(failed: Mapping[ArmName, int]) -> str:
    """The section when a thinker call failed: the counts, and no sentence that any arm wins or loses."""
    counts = ", ".join(f"{arm} {count}" for arm, count in failed.items())
    # Copy: the spec gives no sentence for a failed run.
    return "\n\n".join((
        "## Replay (Task 17b)",
        f"The comparison is invalid: thinker calls failed during the arms' runs ({counts}). An arm with a "
        "failed call is never scored, so this pilot makes no comparison between the arms.",
    )) + "\n"
