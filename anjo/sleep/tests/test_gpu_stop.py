# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's GPU stop (``serving.stop``): when the night's outermost phase ends, the serving
GPU's containers stop instead of idling through Modal's window, unless a live Glimmer request
(``serving.activity``) could be cut. Fakes only: no Modal call, no subprocess."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.serving import activity
from anjo.sleep.serving.backend import phase
from anjo.sleep.serving.config import SCALEDOWN_S
from anjo.sleep.serving.stop import QUIET_S, GpuStop, ModalCli
from anjo.sleep.serving.wake import WarmWindow, _write_window
from anjo.sleep.tests.night_fakes import NIGHT, World, step, summary
from anjo.sleep.night.run import run_night_batch_with

APP = "anjo-sleep-serve"
UID = "p1"


class Containers:
    """Modal's containers as the stop sees them: ``found`` running for the app until stopped."""

    def __init__(self, *found: str, fail: Exception | None = None, before_stop: object = None) -> None:
        self.found, self.stopped, self.listed = list(found), list[str](), 0
        self.fail, self.before_stop = fail, before_stop

    def running(self, app: str) -> list[str]:
        assert app == APP
        self.listed += 1
        return [c for c in self.found if c not in self.stopped]

    def stop(self, container: str) -> None:
        if self.fail is not None:
            raise self.fail
        self.stopped.append(container)


def _host(tmp_path: Path) -> Path:
    """The host's turns folder, as the chat route's host marks it when it registers."""
    folder = activity.turns_dir(tmp_path)
    activity.track(folder)
    return folder


def _stop(folder: Path | None, containers: Containers, window: Path | None = None,
          ) -> tuple[GpuStop, list[tuple[str, dict[str, Any]]]]:
    reports: list[tuple[str, dict[str, Any]]] = []
    stop = GpuStop(containers, APP, turns=folder, window=window)
    stop.watch(lambda status, fields: reports.append((status, dict(fields))))
    return stop, reports


# ── the host's turns ─────────────────────────────────────────────────────────


def test_a_turn_is_in_flight_from_its_start_to_its_end_even_when_it_raises(tmp_path: Path) -> None:
    folder = _host(tmp_path)
    assert activity.seen(folder, time.time()) == activity.Seen(tracked=True, in_flight=0, since_last=None)
    with activity.turn(folder):
        inside = activity.seen(folder, time.time())
    assert inside.in_flight == 1 and inside.since_last is not None and inside.since_last < 5
    with pytest.raises(RuntimeError), activity.turn(folder):
        raise RuntimeError("the turn broke off")
    after = activity.seen(folder, time.time())
    assert after.in_flight == 0 and after.since_last is not None and after.since_last < 5  # its end is its last


def test_a_marker_a_dead_process_left_is_stale_and_a_host_never_marked_is_untracked(tmp_path: Path) -> None:
    folder = activity.turns_dir(tmp_path)
    assert activity.seen(folder, time.time()).tracked is False  # the host never registered its chat route
    activity.track(folder)
    old = folder / activity.IN_FLIGHT / "4242-dead"
    old.touch()
    long_ago = time.time() - activity.STALE_S - 1
    os.utime(old, (long_ago, long_ago))  # no request runs this long: its process died mid-turn
    assert activity.seen(folder, time.time()).in_flight == 0
    with activity.turn(None):  # no data dir: nothing marked, and the turn still runs
        pass


# ── the stop ─────────────────────────────────────────────────────────────────


def test_a_quiet_host_stops_its_containers_and_reports_the_idle_window_saved(tmp_path: Path) -> None:
    folder = _host(tmp_path)
    with activity.turn(folder, clock=lambda: time.time() - QUIET_S - 60):  # a chat that ended 11 minutes ago
        pass
    containers = Containers("c1")
    stop, reports = _stop(folder, containers)
    stop()
    assert containers.stopped == ["c1"]
    [(status, fields)] = reports
    assert status == "ok" and fields["containers"] == 1 and fields["seconds_saved"] == SCALEDOWN_S
    assert fields["graceful"] is True and fields["turn_during_stop"] is False and fields["since_last_turn_s"] > QUIET_S


@pytest.mark.parametrize("why", ["in_flight", "recent_turn", "keep_warm", "untracked", "no_data_dir", "unreadable"])
def test_a_stop_that_could_cut_a_live_request_is_skipped_and_the_idle_window_left_to_do_it(
    tmp_path: Path, why: str,
) -> None:
    folder: Path | None = _host(tmp_path)
    window = tmp_path / "warm_window.json"
    if why == "untracked":
        (activity.turns_dir(tmp_path) / activity.TRACKED).unlink()
    elif why == "no_data_dir":
        folder = None
    elif why == "unreadable":
        (activity.turns_dir(tmp_path) / activity.LAST).write_text("not json")
    elif why == "recent_turn":
        with activity.turn(folder, clock=lambda: time.time() - 300):  # five minutes ago: she may write again
            pass
    elif why == "keep_warm":
        _write_window(window, WarmWindow(time.time() + 3600, time.time()))
    containers = Containers("c1")
    stop, reports = _stop(folder, containers, window)
    if why == "in_flight":
        with activity.turn(folder):  # Kevin's turn streams while the night ends
            stop()
    else:
        stop()
    assert containers.stopped == [] and reports == [("skipped", reports[0][1])]
    assert reports[0][1]["why"] == why and reports[0][1]["seconds_saved"] == 0


def test_the_host_is_checked_again_right_before_each_stop(tmp_path: Path) -> None:
    """A turn that starts while the containers are listed (the CLI takes seconds) stops the stop."""
    folder = _host(tmp_path)
    started = activity.turn(folder)

    class Slow(Containers):
        def running(self, app: str) -> list[str]:
            started.__enter__()  # a chat turn starts mid-listing
            return super().running(app)

    containers = Slow("c1")
    stop, reports = _stop(folder, containers)
    stop()
    started.__exit__(None, None, None)
    assert containers.stopped == [] and reports[0][0] == "skipped" and reports[0][1]["why"] == "in_flight"


@pytest.mark.parametrize("where", ["list", "stop"])
def test_a_stop_that_fails_is_logged_and_never_raises(tmp_path: Path, where: str) -> None:
    folder = _host(tmp_path)
    containers = Containers("c1", fail=RuntimeError("modal said no") if where == "stop" else None)
    if where == "list":
        containers.running = lambda app: (_ for _ in ()).throw(FileNotFoundError("modal"))  # type: ignore[method-assign]
    stop, reports = _stop(folder, containers)
    stop()  # never raises: the night is done whatever the stop does
    status, fields = reports[0]
    assert status == "failed" and fields["error"] == ("RuntimeError" if where == "stop" else "FileNotFoundError")
    assert fields["containers"] == 0 and fields["seconds_saved"] == 0


def test_nothing_running_is_nothing_to_stop(tmp_path: Path) -> None:
    stop, reports = _stop(_host(tmp_path), Containers())
    stop()
    assert reports == [("skipped", {**reports[0][1], "why": "none_running", "seconds_saved": 0})]


def test_modal_cli_lists_the_apps_containers_and_stops_one_gracefully() -> None:
    calls: list[tuple[list[str], float]] = []
    listed = [{"container_id": "ta-1", "app_id": "ap-1", "app_name": APP, "start_time": "now"},
              {"container_id": "ta-2", "app_id": "ap-2", "app_name": "anjo-sleep-train", "start_time": "now"}]

    def run(args: Sequence[str], timeout_s: float) -> str:
        calls.append((list(args), timeout_s))
        return json.dumps(listed) if args[1:3] == ["container", "list"] else ""

    cli = ModalCli(run=run)
    assert cli.running(APP) == ["ta-1"]  # another app's container is never touched
    cli.stop("ta-1")
    assert [args for args, _ in calls] == [["modal", "container", "list", "--json"],
                                           ["modal", "container", "stop", "ta-1", "--yes", "--graceful"]]
    with pytest.raises(ValueError):
        ModalCli(run=lambda args, timeout_s: "{not a list").running(APP)


# ── the night ────────────────────────────────────────────────────────────────


def test_the_night_stops_the_gpu_when_its_outermost_phase_ends_and_logs_it(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=12)
    containers = Containers("c1")
    stop = GpuStop(containers, APP, turns=_host(tmp_path / "data"), window=None)
    with phase():  # the daily's: the night, then the morning, in one phase
        result = run_night_batch_with(world.deps(gpu_stop=stop), {UID: NIGHT})[UID]
        assert containers.listed == 0  # the morning still needs her: nothing stops before the phase ends
    assert containers.stopped == ["c1"]
    logged = step(result.log_path, "gpu_stop")
    assert (logged["status"], logged["containers"], logged["seconds_saved"]) == ("ok", 1, SCALEDOWN_S)
    assert logged["uses"] == "new" and summary(result.log_path)["status"] == result.status


def test_a_failed_stop_never_fails_the_night(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=12)
    stop = GpuStop(Containers("c1", fail=RuntimeError("modal said no")), APP, turns=_host(tmp_path / "data"),
                   window=None)
    result = run_night_batch_with(world.deps(gpu_stop=stop), {UID: NIGHT})[UID]
    assert result.status == "promoted" and result.failed_at is None
    logged = step(result.log_path, "gpu_stop")
    assert (logged["status"], logged["error"]) == ("failed", "RuntimeError")
