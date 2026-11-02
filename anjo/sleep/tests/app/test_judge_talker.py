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
from anjo.sleep.app.judge_talker import (
    JudgeTalker,
    SpendTracker,
    TalkerSettings,
    WindowFile,
)
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
    paid: list[float]
    stop_results: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    window: Path | None = None

    def at(self, seconds: float) -> None:
        self.clock[0] = T0 + seconds

    def run_wakes(self) -> None:
        while self.spawned:
            self.spawned.pop(0)()


def rig(tmp_path: Path, *, backend: str = "modal", **settings: Any) -> Rig:
    serving = FakeServing(name=backend)
    clock = [T0]
    spawned: list[Callable[[], None]] = []
    stops: list[str] = []
    paid: list[float] = []
    made = TalkerSettings(**{"rate_per_hour": 3600.0, "budget_usd": 1000.0, "soft_usd_per_day": 0.0,
                             "hard_usd_per_day": 0.0, **settings})
    results: list[tuple[str, dict[str, Any]]] = []

    def stop() -> tuple[str, dict[str, Any]]:
        stops.append("stop")
        return results.pop(0) if results else ("ok", {"containers": 1})

    window = tmp_path / "keep_warm.json"

    def pay(seconds: float) -> bool:
        paid.append(seconds)
        return True

    talker = JudgeTalker(
        serving, made, spend=SpendTracker(tmp_path / "spend.json", tmp_path / "alerts.jsonl", made),
        stop=stop, window=WindowFile(window), pay=pay,
        clock=lambda: clock[0], spawn=spawned.append, keeper=lambda _run: None, sleep=lambda _s: None,
    )
    return Rig(talker, serving, clock, spawned, stops, paid, results, window)


@pytest.fixture(autouse=True)
def spark_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_META_API_KEY", "test-meta-key")


def test_a_judge_arriving_wakes_her_once_and_spark_answers_meanwhile(tmp_path: Path) -> None:
    r = rig(tmp_path)
    assert r.talker.status().view() == {"state": "cold", "eta_s": 240, "model": "spark-fallback", "reason": None}

    r.talker.arrive()
    r.at(10)
    r.talker.arrive()  # inside the debounce, with a wake running: no second wake
    assert len(r.spawned) == 1
    waking = r.talker.status()
    assert (waking.state, waking.model, waking.eta_s) == ("waking", "spark-fallback", 230)
    assert not r.talker.glimmer_first()

    r.run_wakes()
    assert r.serving.wakes == 1
    assert r.talker.status().view() == {"state": "warm", "eta_s": None, "model": "glimmer", "reason": None}
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
    assert r.paid == [pytest.approx(331.0)]  # the stretch, debited once to the ledger
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
    assert r.paid == [pytest.approx(61.0 + 600.0)]


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


def test_without_a_spark_key_the_label_is_the_frontier_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_META_API_KEY")
    assert rig(tmp_path).talker.status().model == "frontier-fallback"


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
