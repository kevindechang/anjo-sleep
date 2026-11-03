# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""The judges' talker: waking Glimmer on arrival (debounced, capped a day), keeping her warm only
while a judge is present, stopping the GPU once none is, the spend it costs with its alerts and
hard cap, and the route that never makes a judge wait on a cold start. Every backend is a fake: no
Modal, SageMaker or Meta call."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.app import judge_cohort, judge_talker
from anjo.sleep.app.judge_gpu import HOLD_KIND, GpuHolds
from anjo.sleep.app.judge_talker import (
    JudgeTalker,
    SpendTracker,
    TalkerSettings,
    WindowFile,
)
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.serving.wake import WarmWindow, read_window

T0 = 1_800_000_000.0


@dataclass
class FakeServing:
    name: str = "modal"
    wake_ok: bool = True
    ping_ok: bool = True
    wakes: int = 0
    pings: int = 0
    slept: int = 0

    def wake(self) -> WakeResult:
        self.wakes += 1
        return WakeResult(seconds=200.0, ok=self.wake_ok)

    def ping(self) -> bool:
        self.pings += 1
        return self.ping_ok

    def sleep_now(self) -> None:
        self.slept += 1

    def client(self) -> Any:
        raise AssertionError("the talker never sends a request itself")

    def upload_adapter(self, folder: Path, name: str) -> None:
        raise AssertionError("no adapter moves here")


@dataclass
class Rig:
    talker: JudgeTalker
    serving: FakeServing
    clock: list[float]
    spawned: list[Callable[[], None]]
    stops: list[str]
    ledger: Ledger
    stop_results: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    window: Path | None = None

    def settled(self) -> float:
        """Dollars the judges' GPU holds settled on the ledger ($1 a second here)."""
        settled: float = sum(usd for job, usd in self.ledger.settlements().items() if job.startswith(HOLD_KIND))
        return settled

    def open_holds(self) -> list[str]:
        return [held.job for held in self.ledger.open_holds() if held.job.startswith(HOLD_KIND)]

    def at(self, seconds: float) -> None:
        self.clock[0] = T0 + seconds

    def run_wakes(self) -> None:
        while self.spawned:
            self.spawned.pop(0)()


def rig(tmp_path: Path, *, backend: str = "modal", modal_usd: float = 1e9, **settings: Any) -> Rig:
    serving = FakeServing(name=backend)
    clock = [T0]
    spawned: list[Callable[[], None]] = []
    stops: list[str] = []
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": modal_usd})
    made = TalkerSettings(**{"rate_per_hour": 3600.0, "budget_usd": 1000.0, "soft_usd_per_day": 0.0,
                             "hard_usd_per_day": 0.0, **settings})
    results: list[tuple[str, dict[str, Any]]] = []

    def stop() -> tuple[str, dict[str, Any]]:
        stops.append("stop")
        return results.pop(0) if results else ("ok", {"containers": 1})

    window = tmp_path / "keep_warm.json"
    talker = JudgeTalker(
        serving, made, spend=SpendTracker(tmp_path / "spend.json", tmp_path / "alerts.jsonl", made),
        stop=stop, window=WindowFile(window), holds=GpuHolds(lambda: ledger, made.rate_per_hour, owner="1-test"),
        clock=lambda: clock[0], spawn=spawned.append, keeper=lambda _run: None, sleep=lambda _s: None,
    )
    return Rig(talker, serving, clock, spawned, stops, ledger, results, window)


@pytest.fixture(autouse=True)
def spark_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_META_API_KEY", "test-meta-key")


def test_a_judge_arriving_wakes_her_once_and_spark_answers_meanwhile(tmp_path: Path) -> None:
    r = rig(tmp_path)
    assert r.talker.status().view() == {"state": "cold", "eta_s": 240, "model": "spark-fallback", "reason": None,
                                        "fallback_via": "muse-spark", "chat_adapter": False}

    r.talker.arrive()
    r.at(10)
    r.talker.arrive()  # inside the debounce, with a wake running: no second wake
    assert len(r.spawned) == 1
    waking = r.talker.status()
    assert (waking.state, waking.model, waking.eta_s) == ("waking", "spark-fallback", 230)
    assert not r.talker.glimmer_first()

    r.run_wakes()
    assert r.serving.wakes == 1
    assert r.talker.status().view() == {"state": "warm", "eta_s": None, "model": "glimmer", "reason": None,
                                        "fallback_via": None, "chat_adapter": False}
    assert r.talker.glimmer_first()
    assert r.serving.pings == 0  # Modal is never pinged to learn whether she is up


def test_wakes_are_debounced_and_capped_a_day(tmp_path: Path) -> None:
    r = rig(tmp_path, wakes_per_day=1)
    r.serving.wake_ok = False
    r.talker.arrive()
    r.run_wakes()
    r.at(30)
    r.talker.arrive()  # within the debounce: no try
    assert r.spawned == []
    r.at(90)
    r.talker.arrive()  # past it, but the day's one wake is spent
    assert r.spawned == []
    assert r.talker.status().reason == "wakes_spent"


def test_wake_off_never_wakes(tmp_path: Path) -> None:
    r = rig(tmp_path, wake_on=False)
    r.talker.arrive()
    assert r.spawned == []
    assert r.talker.status().reason == "wake_off"


def test_no_heartbeat_stops_the_gpu_and_the_spend(tmp_path: Path) -> None:
    r = rig(tmp_path, idle_s=180.0)
    r.talker.arrive()
    r.run_wakes()
    r.at(100)
    assert r.talker.tick() is True and r.stops == []  # present: nothing stops
    r.at(150)
    r.talker.arrive()  # a heartbeat: the lease runs to 330
    r.at(300)
    assert r.talker.tick() is True and r.stops == []
    r.at(331)
    assert r.talker.tick() is False
    assert r.stops == ["stop"]
    assert r.talker.status().state == "cold"
    assert r.settled() == pytest.approx(331.0) and r.open_holds() == []  # the stretch, settled on its holds
    r.at(1000)
    assert r.talker.tick() is False and r.stops == ["stop"]  # nothing paid since: nothing more to do
    assert SpendTracker(tmp_path / "spend.json", tmp_path / "alerts.jsonl", r.talker.settings).view(T0 + 1000) \
        == {"day_usd": pytest.approx(331.0), "total_usd": pytest.approx(331.0)}


def test_a_stop_skipped_for_a_turn_is_tried_again(tmp_path: Path) -> None:
    r = rig(tmp_path, idle_s=60.0)
    r.stop_results.extend([("skipped", {"why": "recent_turn"}), ("ok", {"containers": 1})])
    r.talker.arrive()
    r.run_wakes()
    r.at(61)
    assert r.talker.tick() is True
    r.at(91)
    assert r.talker.tick() is False
    assert r.stops == ["stop", "stop"]


def test_a_failed_stop_charges_her_idle_window(tmp_path: Path) -> None:
    r = rig(tmp_path, idle_s=60.0)
    r.stop_results.append(("failed", {"error": "CalledProcessError"}))
    r.talker.arrive()
    r.run_wakes()
    r.at(61)
    assert r.talker.tick() is False
    assert r.settled() == pytest.approx(61.0 + 600.0) and r.open_holds() == []


def test_the_keeper_pings_while_a_judge_is_present(tmp_path: Path) -> None:
    r = rig(tmp_path, idle_s=600.0)
    r.talker.arrive()
    r.run_wakes()
    r.at(10)
    r.talker.tick()
    assert r.serving.pings == 1
    r.at(100)
    r.talker.tick()
    assert r.serving.pings == 1  # not again before KEEP_WARM_PING_S
    r.at(260)
    r.serving.ping_ok = False
    r.talker.tick()
    assert r.serving.pings == 2
    assert not r.talker.up()  # she didn't answer: down until a wake says otherwise


def test_the_keep_warm_window_follows_the_lease_and_closes(tmp_path: Path) -> None:
    r = rig(tmp_path, idle_s=180.0)
    assert r.window is not None
    r.talker.arrive()
    r.run_wakes()
    held = read_window(r.window)
    assert held is not None and held.until == pytest.approx(T0 + 180.0)
    r.at(100)
    r.talker.arrive()
    held = read_window(r.window)
    assert held is not None and held.until == pytest.approx(T0 + 280.0)
    r.at(281)
    r.talker.tick()
    held = read_window(r.window)
    assert held is not None and not held.open_at(T0 + 281.5)


def test_someone_elses_longer_window_is_never_shortened(tmp_path: Path) -> None:
    path = tmp_path / "keep_warm.json"
    from anjo.sleep.serving import wake

    wake._write_window(path, WarmWindow(T0 + 5000.0, T0))
    window = WindowFile(path)
    window.extend(T0 + 100.0, T0)
    window.close(T0 + 50.0)
    held = read_window(path)
    assert held is not None and held.until == T0 + 5000.0


def test_spend_alerts_fire_once_at_70_and_90_percent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    settings = TalkerSettings(rate_per_hour=3600.0, budget_usd=10.0, soft_usd_per_day=0.0, hard_usd_per_day=0.0)
    spend = SpendTracker(tmp_path / "spend.json", tmp_path / "alerts.jsonl", settings)
    spend.add(6.0, T0)
    assert not (tmp_path / "alerts.jsonl").exists()
    spend.add(1.5, T0)
    spend.add(2.0, T0)
    spend.add(0.2, T0)
    lines = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert [(a["scope"], a["fraction"]) for a in lines] == [("budget", 0.7), ("budget", 0.9)]
    assert "JUDGE GPU SPEND ALERT" in caplog.text
    again = SpendTracker(tmp_path / "spend.json", tmp_path / "alerts.jsonl", settings)  # after a restart
    again.add(0.1, T0)
    assert len((tmp_path / "alerts.jsonl").read_text().splitlines()) == 2


def test_the_day_soft_cap_alerts_and_resets_the_next_day(tmp_path: Path) -> None:
    settings = TalkerSettings(rate_per_hour=3600.0, budget_usd=0.0, soft_usd_per_day=10.0, hard_usd_per_day=0.0)
    spend = SpendTracker(tmp_path / "spend.json", tmp_path / "alerts.jsonl", settings)
    spend.add(7.0, T0)
    spend.add(7.0, T0 + 86_400)
    scopes = [json.loads(line)["scope"] for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert scopes == ["day_soft", "day_soft"]
    assert spend.view(T0 + 86_400)["day_usd"] == pytest.approx(7.0)


def test_only_the_hard_cap_changes_what_a_judge_gets(tmp_path: Path) -> None:
    r = rig(tmp_path, hard_usd_per_day=100.0, idle_s=600.0)
    r.talker.arrive()
    r.run_wakes()
    r.at(50)
    r.talker.tick()
    assert r.talker.glimmer_first()  # $50 of $100: still Glimmer
    r.at(101)
    assert r.talker.tick() is False  # past the hard cap: the GPU stops though the lease holds
    assert r.stops == ["stop"]
    status = r.talker.status()
    assert (status.state, status.model, status.reason) == ("cold", "spark-fallback", "budget")
    assert not r.talker.glimmer_first()
    r.talker.arrive()
    assert r.spawned == []  # and no wake


def test_sagemaker_status_asks_with_one_cached_ping_while_a_judge_is_present(tmp_path: Path) -> None:
    r = rig(tmp_path, backend="sagemaker", wake_on=False)
    assert r.talker.status().state == "cold"
    assert r.serving.pings == 0  # no judge present: not asked
    r.talker.arrive()
    assert r.talker.status().state == "warm"
    assert r.talker.status().state == "warm"
    assert r.serving.pings == 1


def test_without_a_spark_key_the_label_stays_and_the_model_under_it_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert rig(tmp_path).talker.status().fallback_via == "muse-spark"
    monkeypatch.delenv("ANJO_META_API_KEY")
    status = rig(tmp_path).talker.status()
    assert (status.model, status.fallback_via) == ("spark-fallback", "frontier")


def test_settings_from_env() -> None:
    assert TalkerSettings.from_env("sagemaker", {judge_talker.RATE_ENV: "2"}).wake_eta_s == 900.0
    made = TalkerSettings.from_env("modal", {
        judge_talker.IDLE_MIN_ENV: "5", judge_talker.WAKE_ENV: "0", judge_talker.RATE_ENV: "1.95",
        judge_talker.HARD_ENV: "nope", judge_talker.HARD_TOTAL_ENV: "40",
    })
    assert (made.idle_s, made.wake_on, made.rate_per_hour, made.hard_usd_per_day, made.hard_usd_total) == \
        (300.0, False, 1.95, 12.0, 40.0)
    assert (made.budget_usd, made.soft_usd_per_day, made.wake_eta_s, made.wakes_per_day) == (19.0, 6.0, 240.0, 30)


# ── The route ─────────────────────────────────────────────────────────────────


@pytest.fixture
def judge_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Rig]:
    from anjo.core import capabilities
    from anjo.sleep.app import runtime
    from anjo.sleep.serving import config

    monkeypatch.setattr(capabilities, "CAPABILITY_ENV_VARS", dict(capabilities.CAPABILITY_ENV_VARS))
    monkeypatch.setattr(capabilities, "CAPABILITY_COHORTS", dict(capabilities.CAPABILITY_COHORTS))
    monkeypatch.setattr(capabilities, "_UNSWEPT_CAPABILITIES", set(capabilities._UNSWEPT_CAPABILITIES))
    monkeypatch.setenv("ANJO_SLEEP_ENABLED", "1")
    monkeypatch.setenv(config.REVISION_ENV, "0123456789abcdef0123456789abcdef01234567")
    runtime.set_adapter_source(None)
    made = rig(tmp_path)
    judge_talker.set_talker(made.talker)
    judge_cohort.add_judge("judge-1")
    yield made
    judge_talker.set_talker(None)


def test_the_route_leaves_everyone_but_judges_alone(judge_env: Rig) -> None:
    assert judge_talker.judge_route("someone-else") is None


def test_a_judge_waits_on_no_cold_start(judge_env: Rig) -> None:
    from anjo.sleep.app.runtime import HOST as GLIMMER_HOST
    from anjo.sleep.app.spark_talker import HOST as SPARK_HOST

    cold = judge_talker.judge_route("judge-1")
    assert cold is not None and [q.pin.host for q in cold.served] == [SPARK_HOST]
    judge_env.talker.arrive()
    judge_env.run_wakes()
    warm = judge_talker.judge_route("judge-1")
    assert warm is not None and [q.pin.host for q in warm.served] == [GLIMMER_HOST, SPARK_HOST]
    assert warm.served[0].pin.user_id == "judge-1" and warm.served[0].pin.adapter is None


def _promoted(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

    from anjo.sleep.app import runtime
    from anjo.sleep.types import CheckpointRef

    trained = CheckpointRef("p-1a2b3c4d-n1", "judge-1", date(2026, 11, 3), "x", None)
    monkeypatch.setattr(runtime, "_adapter_source", [lambda user_id: trained if user_id == "judge-1" else None])


def test_a_judges_chat_answers_with_notes_not_the_adapter_by_default(judge_env: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """The final's registered result (2026-11-03): adapter plus notes lost to notes alone (32%), so a
    judge's chat is base Glimmer with the overnight notes; the adapter shows in the wake's before/after."""
    monkeypatch.delenv(judge_talker.CHAT_ADAPTER_ENV, raising=False)
    _promoted(monkeypatch)
    judge_env.talker.arrive()
    judge_env.run_wakes()
    warm = judge_talker.judge_route("judge-1")
    assert warm is not None and warm.served[0].pin.adapter is None
    assert judge_talker.talker().status().view()["chat_adapter"] is False


def test_the_chat_adapter_switch_puts_the_adapter_back(judge_env: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(judge_talker.CHAT_ADAPTER_ENV, "1")
    _promoted(monkeypatch)
    judge_env.talker.arrive()
    judge_env.run_wakes()
    warm = judge_talker.judge_route("judge-1")
    assert warm is not None and warm.served[0].pin.adapter == "p-1a2b3c4d-n1"
    assert judge_talker.talker().status().view()["chat_adapter"] is True


def test_the_sleep_route_fallthrough_keeps_a_judges_chat_on_notes(judge_env: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """No Spark key while cold: judge_route steps aside and sleep_route decides. It must not put the
    adapter back on a judge's chat (Codex review, 2026-11-03); everyone else keeps theirs."""
    from anjo.sleep.app import runtime

    monkeypatch.delenv("ANJO_META_API_KEY")
    monkeypatch.delenv(judge_talker.CHAT_ADAPTER_ENV, raising=False)
    _promoted(monkeypatch)
    assert judge_talker.judge_route("judge-1") is None
    fallthrough = runtime.sleep_route("judge-1")
    assert fallthrough is not None and fallthrough.served[0].pin.adapter is None
    monkeypatch.setenv(judge_talker.CHAT_ADAPTER_ENV, "1")
    switched = runtime.sleep_route("judge-1")
    assert switched is not None and switched.served[0].pin.adapter == "p-1a2b3c4d-n1"


def test_the_kill_switch_still_holds_for_judges(judge_env: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app.flags import sleep_enabled_for

    assert sleep_enabled_for("judge-1")
    monkeypatch.setenv("ANJO_SLEEP_ENABLED", "0")
    assert not sleep_enabled_for("judge-1")
    assert judge_talker.judge_route("judge-1") is None


def test_no_spark_key_while_cold_falls_through_to_the_sleep_route(judge_env: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_META_API_KEY")
    assert judge_talker.judge_route("judge-1") is None


def test_any_backend_error_ends_the_wake_and_the_ping(tmp_path: Path) -> None:
    """A cloud SDK's own error (not a ServingError) still ends the wake: the waking state never sticks."""
    r = rig(tmp_path, backend="sagemaker")

    def broken() -> Any:
        raise RuntimeError("the SDK has no credentials")

    r.serving.wake = broken  # type: ignore[method-assign]
    r.serving.ping = broken  # type: ignore[method-assign]
    r.talker.arrive()
    r.run_wakes()
    assert not r.talker.waking()
    assert r.talker.status().state == "cold"  # its ping failed too: not up
    r.at(61)
    r.talker.arrive()
    assert len(r.spawned) == 1  # past the debounce: tried again


# ── GPU time held on the ledger before it is spent ───────────────────────────


def test_a_wake_is_held_on_the_ledger_before_it_starts(tmp_path: Path) -> None:
    r = rig(tmp_path)
    r.talker.arrive()
    assert r.serving.wakes == 0 and len(r.spawned) == 1  # the wake hasn't run yet...
    (held,) = r.ledger.open_holds()  # ...but its worst case is already held
    assert held.job.startswith(HOLD_KIND) and held.pools == ("modal_usd",)
    assert held.bound.amount == pytest.approx(600.0 + 600.0)  # the wake's timeout, then her idle window


def test_no_ledger_room_wakes_nothing_and_spark_answers_labelled(tmp_path: Path) -> None:
    r = rig(tmp_path, modal_usd=1000.0)  # less than one wake's worst case
    r.talker.arrive()
    assert r.spawned == [] and r.open_holds() == []
    status = r.talker.status()
    assert (status.state, status.model, status.reason) == ("cold", "spark-fallback", "budget")
    assert not r.talker.glimmer_first()


def test_room_other_jobs_hold_is_not_the_judges_and_freed_room_wakes_her(tmp_path: Path) -> None:
    from anjo.sleep.budget import Bound

    r = rig(tmp_path, modal_usd=2000.0)
    night = r.ledger.reserve("night_gpu:someone", ("modal_usd",), Bound(3600.0, 1000.0))  # $1000 of the $2000
    assert night is not None
    r.talker.arrive()
    assert r.spawned == [] and r.talker.status().reason == "budget"
    night.settle(0.0)
    r.at(61)  # past the debounce
    r.talker.arrive()
    assert len(r.spawned) == 1 and r.talker.status().reason is None


def test_a_present_judge_holds_ahead_in_blocks_and_the_stretch_settles_what_it_cost(tmp_path: Path) -> None:
    r = rig(tmp_path, idle_s=180.0)
    r.talker.arrive()
    r.run_wakes()
    for t in range(60, 1801, 60):  # heartbeats for half an hour
        r.at(t)
        r.talker.arrive()
        r.talker.tick()
        covered = r.talker._holds.covered_until()
        assert covered is not None and covered >= T0 + t + 180 + 600  # always held past the lease and its tail
    assert len(r.open_holds()) >= 2
    r.at(1981)
    assert r.talker.tick() is False and r.stops == ["stop"]
    assert r.settled() == pytest.approx(1981.0) and r.open_holds() == []


def test_an_anonymous_prewake_takes_one_small_hold_and_never_extends_past_it(tmp_path: Path) -> None:
    r = rig(tmp_path)
    r.talker.arrive(lease_s=300.0, anonymous=True)
    r.run_wakes()
    (held,) = r.ledger.open_holds()
    for t in (200, 550, 700, 900):  # prewakes again and again, no code ever entered
        r.at(t)
        r.talker.arrive(lease_s=300.0, anonymous=True)
    assert [h.job for h in r.ledger.open_holds()] == [held.job]  # still the one hold
    assert r.talker._lease_until <= T0 + held.bound.amount - 600.0  # the lease stays inside it
    r.at(601)
    assert r.talker.tick() is False and r.stops == ["stop"]
    assert r.settled() == pytest.approx(601.0) and r.settled() <= held.bound.amount


def test_a_hold_for_a_wake_that_never_ran_is_released(tmp_path: Path) -> None:
    r = rig(tmp_path, wakes_per_day=0)  # the day's wakes are spent after the hold is taken
    r.talker.arrive()
    assert r.spawned == [] and len(r.open_holds()) == 1
    r.at(200)
    assert r.talker.tick() is False
    assert r.open_holds() == [] and r.settled() == 0.0


def test_time_past_the_holds_is_held_and_settled_or_recorded_as_an_overage(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": 2400.0})
    holds = GpuHolds(lambda: ledger, 3600.0, owner="1-test")
    assert holds.cover(T0 + 1000.0, T0)
    holds.settle(1500.0)  # 500 s past its hold: a hold of its own
    assert sum(ledger.settlements().values()) == pytest.approx(1500.0) and ledger.overages() == ()
    assert holds.cover(T0 + 400.0, T0)  # the least block, 900 s: the pool is now full
    holds.settle(1000.0)  # 100 s past it, and no room for a hold of its own: an overage, recorded in full
    assert sum(ledger.settlements().values()) == pytest.approx(2500.0) and len(ledger.overages()) == 1


def test_a_stopped_processs_holds_are_settled_at_their_bound_and_live_ones_left(tmp_path: Path) -> None:
    import os

    from anjo.sleep.app.judge_gpu import settle_orphans
    from anjo.sleep.budget import Bound

    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": 1e9})
    for job in (f"{HOLD_KIND}4242-dead:0", f"{HOLD_KIND}{os.getpid()}-old:0", f"{HOLD_KIND}1-test:0",
                f"{HOLD_KIND}5151-live:0", "night_gpu:other"):
        assert ledger.reserve(job, ("modal_usd",), Bound(3600.0, 100.0)) is not None
    assert settle_orphans(ledger, "1-test", alive=lambda pid: pid == 5151) == 2
    assert sorted(ledger.settlements()) == [f"{HOLD_KIND}4242-dead:0", f"{HOLD_KIND}{os.getpid()}-old:0"]
    assert all(usd == pytest.approx(100.0) for usd in ledger.settlements().values())
    assert sorted(h.job for h in ledger.open_holds()) == [f"{HOLD_KIND}1-test:0", f"{HOLD_KIND}5151-live:0",
                                                          "night_gpu:other"]
