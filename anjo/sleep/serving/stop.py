"""The night's GPU stop on Modal: when the night's outermost phase ends (``backend.phase``; the
daily's ends after its morning), its serving containers stop instead of idling through Modal's
scale-down window (``config.SCALEDOWN_S``), which bills about $0.33 a night on the L40S.

It stops from this host, through Modal's own CLI (``modal container list --json``, then ``modal
container stop <id> --yes --graceful``): no change to the serve app, no redeploy, no undeploy. A
graceful stop lets a request the container already runs finish.

**It never cuts a live chat.** Kevin's and Teo's chats, and the live page, use the same container.
Right before each stop (the listing takes seconds, so again after it) the stop reads the host's
turns (``serving.activity``) and skips, leaving Modal's window to do it, when:

- a Glimmer request is in flight (``in_flight``);
- one started or ended within ``QUIET_S``, Modal's own window: she may write again (``recent_turn``);
- a keep-warm window is open (``keep_warm``, ``wake.WarmWindow``);
- the host doesn't mark its turns, or there's no data dir, or they can't be read (``untracked``,
  ``no_data_dir``, ``unreadable``): turns it can't see are turns it could cut.

A turn that starts in the second the stop takes finds the container stopping: its request fails
before its first token, and the chat route's fallback answers it (``app.runtime``). The record says
whether one did (``turn_during_stop``).

Each night of the batch logs the outcome as its ``gpu_stop`` step (``watch``): the containers
stopped and the seconds of idle window saved (``SCALEDOWN_S`` each: the night's last request ended
moments before), or why it skipped. A stop that fails is logged (``failed``, its error's type) and
never raises: the night is done whatever the stop does.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from anjo.sleep.serving import activity
from anjo.sleep.serving.config import SCALEDOWN_S
from anjo.sleep.serving.wake import read_window

QUIET_S: Final = float(SCALEDOWN_S)  # a turn inside Modal's own window keeps the container, as Modal would
CLI_TIMEOUT_S: Final = 60.0

StopStatus = Literal["ok", "skipped", "failed"]
Report = Callable[[StopStatus, Mapping[str, Any]], None]
Runner = Callable[[Sequence[str], float], str]  # the command's arguments and timeout; its stdout

logger = logging.getLogger(__name__)


class Containers(Protocol):
    def running(self, app: str) -> list[str]: ...

    def stop(self, container: str) -> None: ...


def _run(args: Sequence[str], timeout_s: float) -> str:
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout_s, check=True).stdout


@dataclass(frozen=True)
class ModalCli:
    """The serve app's containers through Modal's CLI on this host."""

    run: Runner = _run
    binary: str = "modal"
    timeout_s: float = CLI_TIMEOUT_S

    def running(self, app: str) -> list[str]:
        rows = json.loads(self.run([self.binary, "container", "list", "--json"], self.timeout_s))
        if not isinstance(rows, list):
            raise ValueError("modal container list gave no list")
        return [str(row["container_id"]) for row in rows if isinstance(row, dict) and row.get("app_name") == app]

    def stop(self, container: str) -> None:
        self.run([self.binary, "container", "stop", container, "--yes", "--graceful"], self.timeout_s)


@dataclass(eq=False)  # one stop is one object: ``phase`` runs each pending sleep once, by identity
class GpuStop:
    """The night's phase sleep on Modal (``NightDeps.gpu_stop``): stop ``app``'s containers unless a
    live request could be cut, and report the outcome to every night watching (``watch``)."""

    containers: Containers
    app: str
    turns: Path | None  # the host's turns folder (``activity.turns_dir``); None: no data dir
    window: Path | None  # the keep-warm window file (``wake.window_path``)
    clock: Callable[[], float] = time.time
    quiet_s: float = QUIET_S
    _reports: list[Report] = field(default_factory=list)

    def watch(self, report: Report) -> None:
        self._reports.append(report)

    def __call__(self) -> None:
        try:
            status, fields = self._stop()
        except Exception as error:  # a stop that fails never fails the night: Modal's window still ends it
            status, fields = "failed", {"error": type(error).__name__, "containers": 0, "seconds_saved": 0}
        reports, self._reports = self._reports, []
        for report in reports:
            try:
                report(status, fields)
            except OSError as error:
                logger.warning("the GPU stop's outcome went unlogged (%s)", type(error).__name__)

    def _busy(self) -> tuple[str | None, float | None]:
        """Why a stop now could cut a live request (None: nothing could), and seconds since her last turn."""
        if self.turns is None:
            return "no_data_dir", None
        now = self.clock()
        try:
            seen = activity.seen(self.turns, now)
        except (OSError, ValueError):
            return "unreadable", None
        window = None if self.window is None else read_window(self.window)
        why = ("untracked" if not seen.tracked else "in_flight" if seen.in_flight
               else "recent_turn" if seen.since_last is not None and seen.since_last < self.quiet_s
               else "keep_warm" if window is not None and window.open_at(now) else None)
        return why, seen.since_last

    def _stop(self) -> tuple[StopStatus, dict[str, Any]]:
        why, since = self._busy()
        found = [] if why is not None else self.containers.running(self.app)
        stopped: list[str] = []
        error: str | None = None
        for container in found:
            why, since = self._busy()  # again, right before the stop
            if why is not None:
                break
            try:
                self.containers.stop(container)
            except Exception as failed:  # the rest stay up for Modal's window; reported below, never raised
                error = type(failed).__name__
                break
            stopped.append(container)
        if error is not None:
            return "failed", {"error": error, "containers": len(stopped), "seconds_saved": len(stopped) * SCALEDOWN_S}
        known = {"since_last_turn_s": None if since is None else round(since, 1)}
        if not stopped:
            return "skipped", {"why": why or "none_running", "seconds_saved": 0, **known}
        after, _ = self._busy()  # quiet right before the stop: a turn now began while it ran
        return "ok", {"containers": len(stopped), "seconds_saved": len(stopped) * SCALEDOWN_S, "graceful": True,
                      "turn_during_stop": after in ("in_flight", "recent_turn"), **({"why": why} if why else {}),
                      **known}
