"""The night's GPU stop on Modal: when the night's outermost phase ends (``backend.phase``; the
daily's ends after its morning), its serving containers stop instead of idling through Modal's
scale-down window (``config.SCALEDOWN_S``), which bills about $0.33 a night on the L40S.

It stops from this host, through Modal's own CLI run by the night's own Python (``python -m modal
container list --json``, then ``container stop <id> --yes --graceful``): no change to the serve app,
no redeploy, no undeploy. A graceful stop lets a request the container already runs finish.

**It never cuts a live chat.** People's chats and the live page use the same container.
Before it lists the containers, and again right before each stop, the stop reads the host's turns
(``serving.activity``) and skips, leaving Modal's window to do it, when:

- another night or simulated day holds the night lock (``night_running``): its Glimmer isn't ours.
  The probe (``sweep.held_elsewhere``) takes the lock for an instant, so a night or simulated day
  that starts in that instant is refused as busy (``NightBusy``), as when two start at once;
- a Glimmer request is in flight (``in_flight``);
- one started or ended within ``QUIET_S``, Modal's own window: she may write again (``recent_turn``);
- a keep-warm window is open (``keep_warm``, ``wake.WarmWindow``);
- the host doesn't mark its turns, or there's no data dir, or its turns or keep-warm window can't be
  read (``untracked``, ``no_data_dir``, ``unreadable``): turns it can't see are turns it could cut.

The last look and the stop are one step for a turn: the stop writes its ``stopping`` mark before
that look and removes it after the stop (``activity.stopping``), and a turn writes its own marker
before it looks for that mark. Whichever comes second sees the other: the stop skips, or the turn is
refused before its request is sent (``activity.GlimmerStopping``) and the chat route's fallback
answers it (``app.runtime``). When both look at once, both back off: no turn is ever cut.

Each night of the batch logs the outcome as its ``gpu_stop`` step (``watch``): the containers
stopped and the seconds of idle window saved (``SCALEDOWN_S`` each: the night's last request ended
moments before), or why it skipped. A stop that fails goes to the host's log and to the night's
(``failed``, its error's type) and never raises: the night is done whatever the stop does.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from anjo.sleep.serving import activity
from anjo.sleep.serving.config import SCALEDOWN_S
from anjo.sleep.serving.wake import read_window

QUIET_S: Final = float(SCALEDOWN_S)  # a turn inside Modal's own window keeps the container, as Modal would
CLI_TIMEOUT_S: Final = 60.0  # each CLI call, bounded well under ``activity.STOPPING_STALE_S``

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
    """The serve app's containers through Modal's CLI on this host, run by this Python: a service's
    PATH may not reach the venv's ``modal`` script."""

    run: Runner = _run
    command: tuple[str, ...] = (sys.executable, "-m", "modal")
    timeout_s: float = CLI_TIMEOUT_S

    def running(self, app: str) -> list[str]:
        rows = json.loads(self.run([*self.command, "container", "list", "--json"], self.timeout_s))
        if not isinstance(rows, list):
            raise ValueError("modal container list gave no list")
        return [str(row["container_id"]) for row in rows if isinstance(row, dict) and row.get("app_name") == app]

    def stop(self, container: str) -> None:
        self.run([*self.command, "container", "stop", container, "--yes", "--graceful"], self.timeout_s)


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
    night_running: Callable[[], bool] = lambda: False  # another run holds the night lock (``sweep.held_elsewhere``)
    _reports: list[Report] = field(default_factory=list)

    def watch(self, report: Report) -> None:
        self._reports.append(report)

    def __call__(self) -> None:
        try:
            status, fields = self._stop()
        except Exception as error:  # a stop that fails never fails the night: Modal's window still ends it
            status, fields = "failed", {"error": type(error).__name__, "containers": 0, "seconds_saved": 0}
        if status == "failed":  # on the host's log whoever watches: the containers idle out instead
            logger.warning("the night's GPU stop failed (%s); Modal's idle window ends the containers",
                           fields["error"])
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
        if self.night_running():
            return "night_running", None
        now = self.clock()
        try:
            seen = activity.seen(self.turns, now)
        except (OSError, ValueError):
            return "unreadable", None
        window = None if self.window is None else read_window(self.window)
        lost = window is None and self.window is not None and self.window.exists()  # a window it can't read
        why = ("untracked" if not seen.tracked else "in_flight" if seen.in_flight
               else "recent_turn" if seen.since_last is not None and seen.since_last < self.quiet_s
               else "unreadable" if lost
               else "keep_warm" if window is not None and window.open_at(now) else None)
        return why, seen.since_last

    def _stop(self) -> tuple[StopStatus, dict[str, Any]]:
        why, since = self._busy()
        found = [] if why is not None or self.turns is None else self.containers.running(self.app)
        stopped: list[str] = []
        error: str | None = None
        if found and self.turns is not None:
            with activity.stopping(self.turns) as refresh:  # removed once the last stop returns
                stopped, error, why, since = self._each(found, refresh)
        if error is not None:
            return "failed", {"error": error, "containers": len(stopped), "seconds_saved": len(stopped) * SCALEDOWN_S}
        known = {"since_last_turn_s": None if since is None else round(since, 1)}
        if not stopped:
            return "skipped", {"why": why or "none_running", "seconds_saved": 0, **known}
        return "ok", {"containers": len(stopped), "seconds_saved": len(stopped) * SCALEDOWN_S, "graceful": True,
                      **({"why": why} if why else {}), **known}

    def _each(self, found: list[str], refresh: Callable[[], None],
              ) -> tuple[list[str], str | None, str | None, float | None]:
        """Stop each container, looking at the host right before each: the containers stopped, the
        stop's error, and why it stopped early (None: it didn't), with seconds since her last turn."""
        stopped: list[str] = []
        error: str | None = None
        why: str | None = None
        since: float | None = None
        for container in found:
            refresh()  # first: a turn that starts after the look below sees the mark and is refused
            why, since = self._busy()
            if why is not None:
                break
            try:
                self.containers.stop(container)
            except Exception as failed:  # the rest stay up for Modal's window; reported, never raised
                error = type(failed).__name__
                break
            stopped.append(container)
        return stopped, error, why, since
