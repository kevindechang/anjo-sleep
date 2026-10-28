"""The Glimmer requests people are waiting on right now, kept where another process can read them, so
the night's GPU stop (``serving.stop``) never cuts one.

The host's chat route (``app.runtime``) and the live page (``app.demo_live``) mark each Glimmer
request they send, from its start to its end (``turn``), in ``<data dir>/sleep/serving/glimmer_turns``:

- ``in_flight/``: one file per request while it runs, gone when it ends, however it ends;
- ``last.json``: the wall time of the latest start or end, written before the request's file goes,
  so no instant shows neither;
- ``tracked``: written when the host registers its chat route (``track``). A host that never wrote
  it (an app from before this, another data dir) has turns the stop can't see, so it never stops.
  A turn whose marker can't be written (a full disk) runs unmarked and removes it: no stop follows
  until the host registers again, and one already stopping still refuses it (``stopping``).
- ``stopping``: the stop's own mark (``stopping``), written before its last look at ``in_flight/``.
  A turn writes its marker before it looks for this one, so whichever comes second sees the other:
  the stop skips, or the turn never reaches Glimmer (``GlimmerStopping``) and the chat route's
  fallback answers it before any word is shown.

A marker older than ``STALE_S`` is a process that died mid-request: none runs that long (the chat
route's first token is due within 20 s). A ``stopping`` mark older than ``STOPPING_STALE_S`` is a stop
that died: the stop refreshes it before each container, each CLI call bounded well under it.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.serving.client import ServingError

TURNS_DIR: Final = "glimmer_turns"
IN_FLIGHT: Final = "in_flight"
LAST: Final = "last.json"
TRACKED: Final = "tracked"
STOPPING: Final = "stopping"
STALE_S: Final = 3600.0
STOPPING_STALE_S: Final = 300.0

logger = logging.getLogger(__name__)


class GlimmerStopping(ServingError):
    """The night is stopping Glimmer's container: this request never reached it."""


def turns_dir(data_dir: Path) -> Path:
    """The host's turns folder under its data dir (beside the keep-warm window, ``wake.window_path``)."""
    return data_dir / "sleep" / "serving" / TURNS_DIR


def track(folder: Path) -> None:
    """The host marks its turns from now on: the stop may read them."""
    (folder / IN_FLIGHT).mkdir(parents=True, exist_ok=True)
    (folder / TRACKED).touch()


def _last(folder: Path, at: float) -> None:
    staged = folder / f".{LAST}.{os.getpid()}.{uuid.uuid4().hex}"
    staged.write_text(json.dumps({"at": at}), encoding="utf-8")
    os.replace(staged, folder / LAST)


def _stopping(folder: Path, now: float) -> bool:
    try:
        return now - (folder / STOPPING).stat().st_mtime < STOPPING_STALE_S
    except FileNotFoundError:
        return False


def _stopping_unmarked(folder: Path, now: float) -> bool:
    """``_stopping`` for a turn that runs unmarked: untracked, the host holds off every later stop, but
    one already past its last look sees no marker. A mark it can't read counts as none (logged): the
    folder the stop must write its mark in is failing, and chats aren't refused while it does."""
    try:
        return _stopping(folder, now)
    except OSError as error:
        logger.warning("an unmarked Glimmer turn can't read the GPU stop's mark (%s)", type(error).__name__)
        return False


def _untrack(folder: Path, error: OSError) -> None:
    logger.warning("a Glimmer turn runs unmarked (%s): the host stops counting as tracked, so no GPU stop "
                   "follows until it registers again", type(error).__name__)
    try:
        (folder / TRACKED).unlink(missing_ok=True)
    except OSError as failed:
        logger.error("the host's turns stay marked tracked (%s) though one ran unmarked", type(failed).__name__)


def _unmark_refused(marker: Path) -> None:
    try:
        marker.unlink(missing_ok=True)
    except OSError as error:  # left behind, it holds off stops until it goes stale: never a cut
        logger.warning("a refused Glimmer turn's marker stays (%s)", type(error).__name__)


@contextmanager
def turn(folder: Path | None, *, clock: Callable[[], float] = time.time) -> Iterator[None]:
    """One Glimmer request in flight from its start to its end (None: no data dir, nothing marked).
    Raises ``GlimmerStopping``, before the request is sent, while the night stops the container."""
    marker = None
    refused = False
    if folder is not None:
        try:
            (folder / IN_FLIGHT).mkdir(parents=True, exist_ok=True)
            marker = folder / IN_FLIGHT / f"{os.getpid()}-{uuid.uuid4().hex}"
            marker.touch(exist_ok=False)
            refused = _stopping(folder, clock())  # after the marker: a stop that looks later sees it
            if not refused:
                _last(folder, clock())
        except OSError as error:
            marker = None
            _untrack(folder, error)
            refused = _stopping_unmarked(folder, clock())
    if refused:
        if marker is not None:
            _unmark_refused(marker)
        raise GlimmerStopping("the night is stopping Glimmer's container")
    try:
        yield
    finally:
        if folder is not None and marker is not None:
            try:
                _last(folder, clock())
                marker.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("a Glimmer turn's end went unmarked (%s)", type(error).__name__)


@contextmanager
def stopping(folder: Path) -> Iterator[Callable[[], None]]:
    """The stop's mark, written by the refresh it yields (called before each container's last look,
    so a long stop's mark never goes stale) and removed after the stop however it ends."""
    mark = folder / STOPPING

    def refresh() -> None:
        mark.touch()

    try:
        yield refresh
    finally:
        mark.unlink(missing_ok=True)


@dataclass(frozen=True)
class Seen:
    tracked: bool  # the host marks its turns (``track``)
    in_flight: int  # requests running now
    since_last: float | None  # seconds since the latest start or end; None: none marked yet


def seen(folder: Path, now: float) -> Seen:
    """The host's turns as they stand at ``now`` (wall seconds). A folder that can't be read raises
    (``OSError``, ``ValueError``): the caller treats it as busy."""
    if not (folder / TRACKED).is_file():
        return Seen(tracked=False, in_flight=0, since_last=None)
    flying = folder / IN_FLIGHT
    live = [m for m in flying.iterdir() if now - m.stat().st_mtime < STALE_S] if flying.is_dir() else []
    last = folder / LAST
    if not last.is_file():
        return Seen(tracked=True, in_flight=len(live), since_last=None)
    data = json.loads(last.read_text(encoding="utf-8"))
    at = data.get("at") if isinstance(data, dict) else None
    if not isinstance(at, int | float) or isinstance(at, bool):
        raise ValueError(f"{LAST} holds no time")
    return Seen(tracked=True, in_flight=len(live), since_last=max(0.0, now - float(at)))
