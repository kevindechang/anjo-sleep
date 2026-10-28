"""The Glimmer requests people are waiting on right now, kept where another process can read them, so
the night's GPU stop (``serving.stop``) never cuts one.

The host's chat route (``app.runtime``) and the live page (``app.demo_live``) mark each Glimmer
request they send, from its start to its end (``turn``), in ``<data dir>/sleep/serving/glimmer_turns``:

- ``in_flight/``: one file per request while it runs, gone when it ends, however it ends;
- ``last.json``: the wall time of the latest start or end, written before the request's file goes,
  so no instant shows neither;
- ``tracked``: written when the host registers its chat route (``track``). A host that never wrote
  it (an app from before this, another data dir) has turns the stop can't see, so it never stops.

A file older than ``STALE_S`` is a process that died mid-request: none runs that long (the chat
route's first token is due within 20 s). Marking never stops a request: a write that fails is
logged and the request runs unmarked.
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

TURNS_DIR: Final = "glimmer_turns"
IN_FLIGHT: Final = "in_flight"
LAST: Final = "last.json"
TRACKED: Final = "tracked"
STALE_S: Final = 3600.0

logger = logging.getLogger(__name__)


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


@contextmanager
def turn(folder: Path | None, *, clock: Callable[[], float] = time.time) -> Iterator[None]:
    """One Glimmer request in flight from its start to its end (None: no data dir, nothing marked)."""
    marker = None
    if folder is not None:
        try:
            (folder / IN_FLIGHT).mkdir(parents=True, exist_ok=True)
            marker = folder / IN_FLIGHT / f"{os.getpid()}-{uuid.uuid4().hex}"
            marker.touch(exist_ok=False)
            _last(folder, clock())
        except OSError as error:
            logger.warning("a Glimmer turn runs unmarked (%s): the night's GPU stop can't see it", type(error).__name__)
    try:
        yield
    finally:
        if folder is not None and marker is not None:
            try:
                _last(folder, clock())
                marker.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("a Glimmer turn's end went unmarked (%s)", type(error).__name__)


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
