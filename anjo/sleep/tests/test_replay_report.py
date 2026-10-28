# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The 17b test (Task 17b): M, M+replay and M+shuffled on held-out lives, differing only in the
thinker; replay claimed only if it beats both by Task 17's primary rule; and the ``--fake`` CLI,
end to end."""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import date

import pytest

from anjo.sleep.mind.apd import fakes as apd_fakes
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.inquiry.types import ModelReply, RunStart
from anjo.sleep.mind.replay import cli, compare, fakes, served
from anjo.sleep.mind.replay.served import NightClock
from anjo.sleep.mind.replay.types import CallEnd
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.reports import replay as report
from anjo.sleep.serving.client import ServingError
from anjo.sleep.types import CheckpointRef


class Spy:
    """A thinker that records what it was sent, and the night it was sent on (``clock``), and
    answers as the fake does; ``fail`` makes those calls fail, as a served call that errs does."""

    def __init__(self, clock: NightClock | None = None, fail: frozenset[int] = frozenset()) -> None:
        self.sent: list[str] = []
        self.days: list[date | None] = []
        self.calls: list[CallEnd] = []
        self.clock, self.fail = clock, fail

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        self.sent.append(rendered)
        self.days.append(None if self.clock is None else self.clock.day)
        if len(self.calls) in self.fail:
            self.calls.append(CallEnd(None, failed=True))
            return ModelReply(text="", tokens=max_tokens, model=GLIMMER)
        self.calls.append(CallEnd("stop"))
        return fakes.FakeGlimmer().propose(rendered, max_tokens)


def test_the_17b_arms_run_on_held_out_lives_only_and_differ_only_in_the_thinker() -> None:
    lives = [fakes.fake_life(n, split="held_out") for n in range(2)]
    clock = NightClock()
    thinkers = compare.ArmThinkers(m=Spy(clock), replay=Spy(clock), shuffled=Spy(clock), clock=clock)
    arms = compare.run_replay_arms(lives, fakes.fake_models(), thinkers, brief="BRIEF")
    runs = arms.runs
    assert arms.valid and dict(arms.failed) == dict.fromkeys(compare.REPLAY_ARMS, 0)
    assert [(run.life_id, run.arm) for run in runs] == [(life.id, arm) for life in lives for arm in compare.REPLAY_ARMS]
    for arm in compare.REPLAY_ARMS:
        spy = thinkers.of(arm)
        assert isinstance(spy, Spy) and spy.sent and all(text.startswith("BRIEF\n\n") for text in spy.sent)
        assert set(spy.days) <= set(lives[0].days) | set(lives[1].days) and None not in spy.days
    m, replay = runs[0], runs[1]
    assert [t.prompt for t in m.turns] == [t.prompt for t in replay.turns]  # same thinker replies, same run
    with pytest.raises(ValueError, match="held-out"):
        compare.run_replay_arms([fakes.fake_life(0)], fakes.fake_models(), thinkers)


def test_each_arm_prompt_is_dated_on_the_night_it_belongs_to() -> None:
    life = fakes.fake_life(0, split="held_out")
    clock = NightClock()
    spy = Spy(clock)
    run = compare.run_replay_arms([life], fakes.fake_models(), compare.ArmThinkers(spy, Spy(clock), Spy(clock), clock),
                                  brief="BRIEF").runs[0]
    nights = [e.step.entry.night for e in run.events if isinstance(e.step.entry, RunStart)]
    assert None not in spy.days and set(spy.days) == set(nights) == set(life.days)  # every night's, and only its own


def test_a_failed_call_voids_the_comparison() -> None:
    lives = [fakes.fake_life(n, split="held_out") for n in range(2)]
    thinkers = compare.ArmThinkers(m=Spy(), replay=Spy(fail=frozenset({4})), shuffled=Spy())
    arms = compare.run_replay_arms(lives, fakes.fake_models(), thinkers, brief="BRIEF")
    assert not arms.valid and dict(arms.failed) == {"M": 0, "M+replay": 1, "M+shuffled": 0}
    problems = {life.id: fakes.fake_problems(life) for life in lives}
    with pytest.raises(ValueError, match="never scored"):
        compare.score_replay_arms(lives, problems, arms, audited=fakes.FakeJudge(), gemini=fakes.FakeJudge())
    text = report.invalid(arms.failed)
    assert "invalid" in text and "M+replay 1" in text
    assert "beat" not in text and "improved" not in text


def test_an_adapter_is_refused_before_any_call_unless_it_matches_the_servers_rendering() -> None:
    poster = _NoCalls()
    ref = CheckpointRef("p-0-n0", "apd-replay", date(2026, 9, 17), "volume://p-0-n0", None, rendering="other/low")
    with apd_fakes.serving(apd_fakes.poster_transport(poster)) as colab:
        for adapter in (ref, dataclasses.replace(ref, rendering=None)):
            with pytest.raises(ServingError, match="rendering"):
                served.arm_thinker(colab, adapter, NightClock())
    assert not poster.paths


class _NoCalls:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def post(self, path: str, body: object) -> dict[str, object]:
        self.paths.append(path)
        raise AssertionError("no call is made")

    def close(self) -> None:
        return None


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
