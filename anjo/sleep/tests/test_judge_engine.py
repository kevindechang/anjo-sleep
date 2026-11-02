# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""One judge's night end to end on the fakes (``night/judge.py``): plain SFT at the profile's fixed
steps on the warm-first trainer, the keep-warm started when extract ends and stopped when training
does, its time held and settled on ``modal_usd``, and the night under the shared lock. No Modal."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Bound, Ledger
from anjo.sleep.night import judge
from anjo.sleep.night.deps import NightDeps, NightOptions, Trainer
from anjo.sleep.night.state import day_window
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.tests.night_fakes import NIGHT, World, thread
from anjo.sleep.tests.test_judge_night import DEMO, PROFILE, UID
from anjo.sleep.train import warm_job
from anjo.sleep.train.lora import LoraCfg
from anjo.sleep.train.runners import choice
from anjo.sleep.train.runners.warm import WarmFirstRunner
from anjo.sleep.train.trainer import LoraTrainer


class Pings:
    """The ``Warmer``'s host side: counts starts and stops, and reports the span its pings ran."""

    def __init__(self, span_s: float | None = None) -> None:
        self.started = self.stopped = 0
        self.span_s = span_s

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


def test_the_trainer_warms_from_the_nights_start_unless_the_host_says_later() -> None:
    assert judge.warm_at({}) == "start" and judge.warm_at({judge.WARM_AT_ENV: "practice"}) == "practice"
    with pytest.raises(judge.JudgeNightError):
        judge.warm_at({judge.WARM_AT_ENV: "brief"})


def test_by_default_the_first_event_starts_the_keep_warm(tmp_path: Path) -> None:
    pings = Pings()
    window = judge.WarmWindow(pings, Ledger(tmp_path / "b.jsonl"), "modal_train:x:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    observe = judge.watching(None, window, "start")
    observe(UID, "close_day", "running", {})
    observe(UID, "extract", "running", {})
    assert pings.started == 1


def test_the_serving_gpu_starts_waking_on_its_own_thread_and_a_failure_is_kept() -> None:
    from anjo.sleep.serving.client import ServingError, WakeResult

    early = judge.EarlyWake(lambda: WakeResult(279.0, True))
    early.start()
    assert early.thread is not None
    early.thread.join(5.0)
    assert early.result == WakeResult(279.0, True) and early.error is None

    def down() -> WakeResult:
        raise ServingError("cold")

    failed = judge.EarlyWake(down)
    failed.start()
    assert failed.thread is not None
    failed.thread.join(5.0)
    assert failed.error == "ServingError"
    judge.EarlyWake(None).start()  # no wake wired: nothing to do


def test_a_judge_is_a_real_person_on_a_deep_night_with_the_profile() -> None:
    options = judge.judge_options(NIGHT, PROFILE)
    assert options.personas == {} and options.allow_real and options.force_deep and options.mind_inquiry
    assert options.demo == PROFILE and options.practice_scenes == PROFILE.practice_scenes
    assert judge.judge_steps(PROFILE).steps == PROFILE.train_steps == 8


def test_the_keep_warm_is_held_and_settled_at_its_span_plus_the_scale_down(tmp_path: Path) -> None:
    ledger, pings = Ledger(tmp_path / "budget.jsonl"), Pings()
    window = judge.WarmWindow(pings, ledger, "modal_train:2026-11-03:judge-1:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    window.begin()
    window.begin()  # once
    [held] = ledger.open_holds()
    assert held.pools == ("modal_usd",) and pings.started == 1
    pings.span_s = 1000.0
    window.end()
    window.end()  # once
    assert pings.stopped == 1 and ledger.open_holds() == ()
    assert window.spent == pytest.approx((1000.0 + 90.0) / 3600.0 * 2.50)
    assert ledger.settled([held.job]) == pytest.approx(window.spent)


def test_a_short_keep_warm_counts_the_container_start_a_ping_may_have_begun(tmp_path: Path) -> None:
    ledger, pings = Ledger(tmp_path / "budget.jsonl"), Pings()
    window = judge.WarmWindow(pings, ledger, "modal_train:x:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    window.begin()
    pings.span_s = 30.0  # the night broke half a minute in, while the container may still be starting
    window.end()
    assert judge.WARM_START_S == 15 * 60
    assert window.spent == pytest.approx((judge.WARM_START_S + 90.0) / 3600.0 * 2.50)


def test_a_keep_warm_that_never_pinged_releases_its_hold(tmp_path: Path) -> None:
    ledger, pings = Ledger(tmp_path / "budget.jsonl"), Pings(span_s=None)
    window = judge.WarmWindow(pings, ledger, "modal_train:x:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    window.begin()
    window.end()
    assert ledger.open_holds() == () and ledger.used("modal_usd") == 0.0 and window.spent == 0.0


def test_a_keep_warm_past_its_hold_settles_all_it_cost_on_a_second_hold(tmp_path: Path) -> None:
    ledger, pings = Ledger(tmp_path / "budget.jsonl"), Pings()
    window = judge.WarmWindow(pings, ledger, "modal_train:x:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    window.begin()
    [held] = ledger.open_holds()
    pings.span_s = judge.WARM_HOLD_S  # the last ping's latency ran it past the deadline: 90 s over
    window.end()
    cost = (judge.WARM_HOLD_S + 90.0) / 3600.0 * 2.50
    assert window.spent == pytest.approx(cost)  # never cut to the hold
    assert ledger.open_holds() == () and ledger.overages() == ()
    assert ledger.settled([held.job]) == pytest.approx(held.bound.amount)
    assert ledger.settled([f"{held.job}:over"]) == pytest.approx(cost - held.bound.amount)
    assert ledger.used("modal_usd") == pytest.approx(cost)


def test_a_keep_warm_past_its_hold_with_no_room_left_settles_it_whole(tmp_path: Path) -> None:
    ledger, pings = Ledger(tmp_path / "budget.jsonl"), Pings()
    window = judge.WarmWindow(pings, ledger, "modal_train:x:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    window.begin()
    [held] = ledger.open_holds()
    room = CAPS["modal_usd"] - ledger.used("modal_usd")
    filler = ledger.reserve("modal_train:filler", ("modal_usd",), Bound(1.0, room * 3600.0))
    assert filler is not None  # the pool is now full
    pings.span_s = judge.WARM_HOLD_S
    window.end()
    cost = (judge.WARM_HOLD_S + 90.0) / 3600.0 * 2.50
    assert ledger.settled([held.job]) == pytest.approx(cost)  # recorded in full, not cut to the hold
    assert ledger.overages() == (held.job,)  # and the pool waits for the coordinator to clear it


def test_the_keep_warm_stops_when_training_ends_even_when_it_started_with_the_night(tmp_path: Path) -> None:
    pings = Pings(span_s=60.0)
    window = judge.WarmWindow(pings, Ledger(tmp_path / "b.jsonl"), "modal_train:x:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    observe = judge.watching(None, window, "start")
    for step, state in (("extract", "running"), ("train", "running"), ("train", "ok"), ("gate", "running"),
                        ("refine", "ok")):
        observe(UID, step, state, {})
    assert pings.started == 1 and pings.stopped == 1 and window.ended  # not warm through the gates


def test_a_cold_only_host_warms_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    window = judge.WarmWindow(None, ledger, "modal_train:x:warm", scaledown_s=90.0)
    window.begin()
    window.end()
    assert ledger.open_holds() == () and window.spent == 0.0


def test_the_observer_starts_and_stops_the_keep_warm_and_passes_every_event_on(tmp_path: Path) -> None:
    pings, seen = Pings(), []
    window = judge.WarmWindow(pings, Ledger(tmp_path / "b.jsonl"), "modal_train:x:warm", scaledown_s=90.0)  # type: ignore[arg-type]
    observe = judge.watching(lambda *event: seen.append(event[1:3]), window, "extract")
    observe(UID, "extract", "running", {})
    assert pings.started == 0
    observe(UID, "extract", "ok", {})
    assert pings.started == 1
    observe(UID, "train", "running", {})
    assert pings.stopped == 0
    observe(UID, "train", "ok", {})
    assert pings.stopped == 1 and window.ended
    assert seen == [("extract", "running"), ("extract", "ok"), ("train", "running"), ("train", "ok")]


class Warm:
    """The warm calls' host side: counts keep-warm calls (``ModalCalls`` too, never reached here)."""

    def __init__(self) -> None:
        self.pings = 0
        self.cancelled: list[str] = []

    def warm(self) -> str:
        self.pings += 1
        return f"fc-warm-{self.pings}"

    def cancel_input(self, call_id: str) -> bool:
        self.cancelled.append(call_id)
        return True


def test_a_judges_night_runs_on_the_warm_first_trainer_under_the_shared_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(choice, "colab_installed", lambda: False)
    world = World(tmp_path / "world")
    world.plan(UID, NIGHT)
    world.targets(UID).upsert(thread(UID))
    built: dict[str, Any] = {}

    woke: list[str] = []

    def wake() -> WakeResult:
        woke.append("wake")
        return WakeResult(1.0, True)

    def wiring(trainer: Trainer, options: NightOptions) -> NightDeps:
        built.update(trainer=trainer, options=options)
        return world.deps(settings=replace(DEMO, demo=options.demo), wake=wake)  # the fakes train; the real trainer is checked

    seen: list[tuple[str, str]] = []
    warm = Warm()
    start, end = day_window(NIGHT)
    done = judge.execute_judge(
        wiring=wiring, job=judge.JudgeNight(user=UID, night=NIGHT, window=(start, end), as_of=end), profile=PROFILE,
        data_dir=tmp_path / "data", ledger=Ledger(tmp_path / "data" / "budget.jsonl"),
        observer=lambda uid, step, state, receipt: seen.append((step, state)),
        environ={warm_job.SCALEDOWN_ENV: "60"}, warm_calls=lambda: warm,  # type: ignore[arg-type, return-value]
        publisher=lambda: (lambda folder, name: None),
    )
    assert done.result.status == "promoted"
    trainer = built["trainer"]
    assert isinstance(trainer, LoraTrainer) and isinstance(trainer._runner, WarmFirstRunner)
    assert trainer._cfg == LoraCfg() and trainer._steps == judge.judge_steps(PROFILE)
    assert built["options"].demo == PROFILE and built["options"].allow_real
    assert ("extract", "ok") in seen and ("train", "ok") in seen
    assert done.warm_usd > 0.0  # held and settled from the night's start until training's end
    assert woke  # the serving GPU was woken (early, and again where the night needs Glimmer)


def run_named(night_id: str, execute: Any) -> int:
    """The app side's entry, as ``main`` finds it by name."""
    CALLED.append((night_id, execute))
    return 7


CALLED: list[tuple[str, Any]] = []


def test_the_night_process_hands_the_named_app_entry_the_night() -> None:
    assert judge.main([f"{__name__}:run_named", "jn-0123456789abcdef"]) == 7
    assert CALLED == [("jn-0123456789abcdef", judge.execute_record)]
    with pytest.raises(SystemExit):
        judge.main(["no-colon", "jn-0123456789abcdef"])
    with pytest.raises(SystemExit):
        judge.main([])



def test_the_keep_warms_hold_is_settled_before_the_shared_lock_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    monkeypatch.setattr(choice, "colab_installed", lambda: False)
    order: list[str] = []

    @contextmanager
    def shared_night(ledger: Ledger, sweep: Any) -> Any:
        try:
            yield None
        finally:
            order.append("lock released")

    monkeypatch.setattr(judge, "shared_night", shared_night)
    monkeypatch.setattr(judge.WarmWindow, "end", lambda self: order.append("warm settled"))

    def broken(trainer: Trainer, options: NightOptions) -> NightDeps:
        raise RuntimeError("the wiring broke")

    start, end = day_window(NIGHT)
    with pytest.raises(RuntimeError):
        judge.execute_judge(
            wiring=broken, job=judge.JudgeNight(user=UID, night=NIGHT, window=(start, end), as_of=end),
            profile=PROFILE, data_dir=tmp_path / "data", ledger=Ledger(tmp_path / "data" / "budget.jsonl"),
            environ={}, warm_calls=lambda: Warm(),  # type: ignore[arg-type, return-value]
            publisher=lambda: (lambda folder, name: None),
        )
    assert order == ["warm settled", "lock released"]  # no other night's sweep can settle it first
