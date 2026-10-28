# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The 17b test (Task 17b): M, M+replay and M+shuffled on held-out lives, differing only in the
thinker; replay claimed only if it beats both by Task 17's primary rule; and the ``--fake`` CLI,
end to end."""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.inquiry.types import ModelReply
from anjo.sleep.mind.replay import cli, compare, fakes
from anjo.sleep.reports import replay as report


class Spy:
    """A thinker that records what it was sent and answers as the fake does."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        self.sent.append(rendered)
        return fakes.FakeGlimmer().propose(rendered, max_tokens)


def test_the_17b_arms_run_on_held_out_lives_only_and_differ_only_in_the_thinker() -> None:
    lives = [fakes.fake_life(n, split="held_out") for n in range(2)]
    thinkers = compare.ArmThinkers(m=Spy(), replay=Spy(), shuffled=Spy())
    runs = compare.run_replay_arms(lives, fakes.fake_models(), thinkers, brief="BRIEF")
    assert [(run.life_id, run.arm) for run in runs] == [(life.id, arm) for life in lives for arm in compare.REPLAY_ARMS]
    for arm in compare.REPLAY_ARMS:
        sent = thinkers.of(arm).sent  # type: ignore[attr-defined]
        assert sent and all(text.startswith("BRIEF\n\n") for text in sent)
    m, replay = runs[0], runs[1]
    assert [t.prompt for t in m.turns] == [t.prompt for t in replay.turns]  # same thinker replies, same run
    with pytest.raises(ValueError, match="held-out"):
        compare.run_replay_arms([fakes.fake_life(0)], fakes.fake_models(), thinkers)


def _scores(view: apd.JudgeView, totals: dict[apd.ArmName, Sequence[float]]) -> list[apd.LifeScore]:
    out = []
    for arm, values in totals.items():
        for n, value in enumerate(values):
            items = (apd.ItemScore(f"i{n}", "contradiction", None if value == 0 else "f", value),)
            out.append(apd.LifeScore(f"life-{n}", arm, view, items, ()))
    return out


WIN = [1.0] * 6
LOW = [0.0] * 6


@pytest.mark.parametrize(("m", "shuffled", "won"), [
    (LOW, LOW, True), (WIN, LOW, False), (LOW, WIN, False), (LOW, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], False),
])
def test_replay_is_claimed_only_if_it_beats_both(m: list[float], shuffled: list[float], won: bool) -> None:
    totals: dict[apd.ArmName, Sequence[float]] = {"M": m, "M+replay": WIN, "M+shuffled": shuffled}
    audited, gemini = _scores("audited", totals), _scores("gemini", totals)
    pairs = [report.versus(audited, gemini, other) for other in report.OTHERS]
    assert report.claimed(pairs) is won
    text = report.section(audited, gemini)
    assert ("replaying simulated lives improved her inquiry performance" in text) is won
    assert "pilot" in text and "6-life" in text
    for never in ("overnight", "next-investigation", "personal learning"):
        assert never not in text
    assert "| life-0 |" in text and "M+shuffled (Gemini)" in text


def test_a_gemini_disagreement_withholds_the_claim() -> None:
    totals: dict[apd.ArmName, Sequence[float]] = {"M": LOW, "M+replay": WIN, "M+shuffled": LOW}
    gemini_totals: dict[apd.ArmName, Sequence[float]] = {"M": WIN, "M+replay": LOW, "M+shuffled": LOW}
    pairs = [report.versus(_scores("audited", totals), _scores("gemini", gemini_totals), o) for o in report.OTHERS]
    assert not report.claimed(pairs)
    with pytest.raises(ValueError, match="paired"):
        report.versus(_scores("audited", totals), _scores("gemini", {"M": LOW}), "M")


def test_the_fake_cli_runs_end_to_end() -> None:
    printed: list[str] = []
    assert cli.main(["--fake", "--lives", "2", "--held-out", "3"], out=printed.append) == 0
    text = "\n".join(printed)
    for needed in ("Start states:", "labelled", "undetermined", "branches", "thinker-1:", "thinker-shuffled:",
                   "Replay training plan", "A100 minutes", "compute units", "## Replay (Task 17b)", "3-life pilot"):
        assert needed in text


def test_a_live_run_stops_without_a_go() -> None:
    printed: list[str] = []
    assert cli.main([], out=printed.append) == 2
    assert printed == [cli.LIVE_STOP]
