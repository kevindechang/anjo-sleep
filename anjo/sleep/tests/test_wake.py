# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Waking Glimmer before use, and the keep-warm window: a fake server and an injected clock, so no
network and no real sleeping."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.serving import client, wake
from anjo.sleep.serving.client import WakeResult

URL = "https://glimmer.example.test"
TOKEN = "-".join(("fake", "bearer", "value"))  # built at runtime: no key-shaped literal for the secret scan


@pytest.fixture(autouse=True)
def on_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests serve through Modal, the fallback backend; SageMaker's are in test_sagemaker*.py."""
    monkeypatch.setenv("ANJO_SLEEP_BACKEND", "modal")


class Clock:
    """Time that only moves when something sleeps or the server takes its time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def server(clock: Clock, ready_at: float, seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        clock.now += 1.0  # each poll takes a second
        if clock.now < ready_at:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": []})

    return httpx.MockTransport(handle)


def test_wake_polls_until_the_server_answers() -> None:
    clock = Clock()
    seen: list[httpx.Request] = []
    done = client.wake(URL, TOKEN, 600.0, transport=server(clock, 30.0, seen), clock=clock, sleep=clock.sleep)
    assert done.ok and 30.0 <= done.seconds < 60.0
    assert {r.url.path for r in seen} == {"/v1/models"} and len(seen) > 2
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in seen)
    assert clock.sleeps == sorted(clock.sleeps) and max(clock.sleeps) <= client.WAKE_BACKOFF_MAX_S  # backoff, capped


def test_wake_respects_total_timeout() -> None:
    clock = Clock()
    done = client.wake(URL, TOKEN, 100.0, transport=server(clock, 10**9), clock=clock, sleep=clock.sleep)
    assert not done.ok and 100.0 <= done.seconds <= 100.0 + client.WAKE_POLL_TIMEOUT_S
    assert client.WAKE_TOTAL_TIMEOUT_S == 600.0


def test_wake_never_logs_token(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    clock = Clock()

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused for {request.headers['authorization']}")

    with caplog.at_level(logging.DEBUG):
        done = client.wake(URL, TOKEN, 30.0, transport=httpx.MockTransport(broken), clock=clock, sleep=clock.sleep)
    assert not done.ok
    out = capsys.readouterr()
    assert TOKEN not in caplog.text and TOKEN not in out.out + out.err and TOKEN not in repr(done)


def test_wake_needs_https_and_a_token() -> None:
    with pytest.raises(client.ServingError):
        client.wake("http://x.test", TOKEN, 10.0)
    with pytest.raises(client.ServingError):
        client.wake(URL, "", 10.0)


def test_keep_warm_pings_under_scaledown_and_stops_at_cap() -> None:
    clock = Clock()
    seen: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(clock.now)
        return httpx.Response(200, json={})

    done = wake.keep_warm(URL, TOKEN, 10_000, transport=httpx.MockTransport(handle), clock=clock, sleep=clock.sleep)
    assert wake.KEEP_WARM_PING_S < 600 and wake.KEEP_WARM_PING_S == 240.0 and wake.MAX_KEEP_WARM_MIN == 180
    gaps = [b - a for a, b in zip(seen, seen[1:], strict=False)]
    assert gaps and all(g <= wake.KEEP_WARM_PING_S < 600 for g in gaps)  # never idle for Modal's 600 s
    assert seen[-1] <= wake.MAX_KEEP_WARM_MIN * 60  # it stops at the cap
    assert done.seconds >= wake.MAX_KEEP_WARM_MIN * 60 - wake.KEEP_WARM_PING_S and done.pings == len(seen)
    assert clock.now <= wake.MAX_KEEP_WARM_MIN * 60 + 1


def test_keep_warm_reports_pings_that_failed() -> None:
    clock = Clock()
    done = wake.keep_warm(URL, TOKEN, 10, transport=httpx.MockTransport(lambda r: httpx.Response(503)),
                          clock=clock, sleep=clock.sleep)
    assert done.failed == done.pings > 0


def test_cli_wake_prints_seconds_not_the_token(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(wake.URL_ENV, URL)
    monkeypatch.setenv(wake.TOKEN_ENV, TOKEN)
    monkeypatch.setattr(client, "wake", lambda url, token, timeout_s: WakeResult(41.5, True))
    assert wake.main(["--wake"]) == 0
    shown = capsys.readouterr().out
    assert "41.5" in shown and TOKEN not in shown and URL not in shown
    monkeypatch.setattr(client, "wake", lambda url, token, timeout_s: WakeResult(600.0, False))
    assert wake.main(["--wake"]) == 1


def test_cli_without_the_environment_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(wake.URL_ENV, raising=False)
    monkeypatch.delenv(wake.TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit):
        wake.main(["--wake"])


def test_night_and_morning_record_wake_seconds(tmp_path: Path) -> None:
    from dataclasses import replace

    from anjo.sleep.tests.night_fakes import NIGHT, World, summary
    from anjo.sleep.tests.test_morning import NIGHT as MORNING_NIGHT
    from anjo.sleep.tests.test_morning import Rig

    from anjo.sleep.night.run import run_night_batch_with

    calls: list[str] = []

    def fake_wake() -> WakeResult:
        calls.append("wake")
        return WakeResult(12.5, True)

    world = World(tmp_path / "night")
    world.plan("p1", NIGHT)
    result = run_night_batch_with(world.deps(wake=fake_wake), {"p1": NIGHT})["p1"]
    assert result.status == "promoted" and calls == ["wake", "wake"]  # before practice, and again before the gates
    assert summary(result.log_path)["wake_seconds"] == 12.5

    rig = Rig(tmp_path / "morning")
    rig.morning = replace(rig.morning, wake=fake_wake)
    done = rig.run(night=MORNING_NIGHT)
    assert json.loads(done.path.read_text(encoding="utf-8"))["wake_seconds"] == 12.5 and len(calls) == 3


def test_a_night_that_cannot_wake_glimmer_fails_with_a_code(tmp_path: Path) -> None:
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.night_fakes import NIGHT, World, step

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = run_night_batch_with(world.deps(wake=lambda: WakeResult(600.0, False)), {"p1": NIGHT})["p1"]
    assert result.status == "failed" and result.failed_at == "practice"
    assert "glimmer_not_ready" in step(result.log_path, "practice")["error"]


def test_a_morning_that_cannot_wake_glimmer_scores_nothing(tmp_path: Path) -> None:
    from dataclasses import replace

    from anjo.sleep.tests.test_morning import Rig

    rig = Rig(tmp_path)
    rig.morning = replace(rig.morning, wake=lambda: WakeResult(600.0, False))
    with pytest.raises(client.GlimmerNotReady, match="glimmer_not_ready"):
        rig.run()
    assert rig.model.calls == [] and not rig.scores.exists()


def test_keep_warm_exits_1_on_partial_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(wake.URL_ENV, URL)
    monkeypatch.setenv(wake.TOKEN_ENV, TOKEN)
    monkeypatch.setattr(client, "wake", lambda url, token, timeout_s: WakeResult(1.0, True))
    monkeypatch.setattr(wake, "keep_warm", lambda *a, **k: wake.KeepWarmResult(60.0, 3, 1))
    assert wake.main(["--keep-warm-minutes", "1"]) == 1
    monkeypatch.setattr(wake, "keep_warm", lambda *a, **k: wake.KeepWarmResult(60.0, 3, 0))
    assert wake.main(["--keep-warm-minutes", "1"]) == 0


def test_gates_wake_again_and_record_both_wakes(tmp_path: Path) -> None:
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.night_fakes import NIGHT, World, step, summary

    seconds = iter([12.5, 3.25])
    calls: list[str] = []

    def fake_wake() -> WakeResult:
        calls.append("wake")
        return WakeResult(next(seconds), True)

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = run_night_batch_with(world.deps(wake=fake_wake), {"p1": NIGHT})["p1"]
    assert result.status == "promoted" and calls == ["wake", "wake"]
    assert summary(result.log_path)["wake_seconds"] == 12.5 and summary(result.log_path)["gates_wake_seconds"] == 3.25
    assert step(result.log_path, "gates")["wake_seconds"] == 3.25


def test_a_cold_server_before_gates_fails_that_person_at_gates(tmp_path: Path) -> None:
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.night_fakes import NIGHT, World, step

    answers = iter([WakeResult(1.0, True), WakeResult(600.0, False)])
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = run_night_batch_with(world.deps(wake=lambda: next(answers)), {"p1": NIGHT})["p1"]
    assert result.status == "failed" and result.failed_at == "gates"
    assert "glimmer_not_ready" in step(result.log_path, "gates")["error"]


def test_a_wake_that_raises_is_a_recorded_failure_for_each_person(tmp_path: Path) -> None:
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.night_fakes import NIGHT, World, step

    def broken() -> WakeResult:
        raise ValueError("ANJO_SLEEP_MODAL_URL is not set")

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    world.plan("p2", NIGHT)
    done = run_night_batch_with(world.deps(wake=broken), {"p1": NIGHT, "p2": NIGHT})
    assert {r.status for r in done.values()} == {"failed"} and {r.failed_at for r in done.values()} == {"practice"}
    assert all(step(r.log_path, "practice")["error"] == "ValueError" for r in done.values())


def test_keep_warm_reports_each_ping_to_its_listener() -> None:
    clock, heard = Clock(), list[bool]()
    answers = iter([True, False, True])
    done = wake.keep_warm_with(lambda: next(answers), 2 * wake.KEEP_WARM_PING_S / 60, clock=clock,
                               sleep=clock.sleep, on_ping=heard.append)
    assert heard == [True, False, True] and done.pings == 3 and done.failed == 1


def test_a_keep_warm_window_is_written_while_it_pings_and_removed_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The live page reads the window (``read_window``) to answer inside it on either backend
    without asking Glimmer anything: on Modal a request would start a container."""
    import time

    monkeypatch.setenv(wake.URL_ENV, URL)
    monkeypatch.setenv(wake.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(wake.DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(client, "wake", lambda url, token, timeout_s: WakeResult(1.0, True))
    path = wake.window_path(tmp_path)
    seen: list[wake.WarmWindow | None] = []

    def pinging(url: str, token: str, minutes: float, *, on_ping: Any = None, **_: Any) -> wake.KeepWarmResult:
        assert wake.read_window(path) is None  # no window before her first answered ping
        on_ping(True)
        seen.append(wake.read_window(path))
        return wake.KeepWarmResult(60.0, 1, 0)

    monkeypatch.setattr(wake, "keep_warm", pinging)
    assert wake.main(["--keep-warm-minutes", "30"]) == 0
    (window,) = seen
    assert window is not None and window.open_at(time.time())
    assert window.until - window.beat == pytest.approx(30 * 60, abs=5)
    assert wake.read_window(path) is None  # the window ends with the process
    assert TOKEN not in path.parent.joinpath(wake.WINDOW_FILE + ".part").as_posix()


def test_a_window_whose_pings_stopped_is_closed() -> None:
    window = wake.WarmWindow(until=10_000.0, beat=1_000.0)
    assert window.open_at(1_000.0) and window.open_at(1_000.0 + wake.BEAT_GRACE_S - 1)
    assert not window.open_at(1_000.0 + wake.BEAT_GRACE_S)  # the keep-warm process died
    assert not window.open_at(999.0) and not wake.WarmWindow(until=1_100.0, beat=1_000.0).open_at(1_100.0)
    assert wake.BEAT_GRACE_S < 600  # closed before Modal's scale-down could take her


def test_an_unreadable_window_file_is_no_window(tmp_path: Path) -> None:
    path = wake.window_path(tmp_path)
    assert wake.read_window(path) is None
    path.parent.mkdir(parents=True)
    for junk in ("{torn", "[]", '{"until": "soon", "beat": 1}', '{"until": 5}'):
        path.write_text(junk)
        assert wake.read_window(path) is None, junk
