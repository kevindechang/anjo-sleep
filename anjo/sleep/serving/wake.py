"""Wake Glimmer, and keep it warm for a live window (``python -m anjo.sleep.serving.wake``).

    python -m anjo.sleep.serving.wake --wake                      # wake it, print the seconds
    python -m anjo.sleep.serving.wake --keep-warm-minutes 90      # wake it, then keep it up 90 min

The server scales to zero after ``config.SCALEDOWN_S`` (600 s) idle, and a cold start of the 30B
model takes minutes, so a simulated day, a night, a morning run and every live demo wake it first
(``client.wake``). A keep-warm window sends ``GET /v1/models`` (with the bearer token) every
``KEEP_WARM_PING_S`` (240 s) until the minutes pass. That request goes through Modal's proxy to
vLLM, and a request in flight or just finished is what holds a container past its scale-down
window, so it resets the idle timer. It costs no tokens and no GPU work, and it checks the token
as well. A 1-token completion would reset it too, but it would spend GPU time for nothing.

**Nothing can stay up unbounded.** The window is capped at ``MAX_KEEP_WARM_MIN`` (180 minutes), and
if this process dies, the pings stop and Modal scales down by itself after 600 idle seconds. Keeping
the L40S up costs about $2 an hour, so every window needs Kevin's OK.

The URL and token come from ``ANJO_SLEEP_MODAL_URL`` and ``ANJO_SLEEP_MODAL_TOKEN``, and are never
printed.

**The window file.** While a window runs, each answered ping writes its end and the time of that
ping to ``$ANJO_DATA_DIR/sleep/serving/warm_window.json`` (``WarmWindow``); the file goes when the
process ends. A window refuses to start without ``ANJO_DATA_DIR``, before anything wakes: run from a
shell without the host's environment file, it would write where the page never looks. The live demo page reads it to answer inside the window on either backend without a
request of its own, which on Modal would start a container, and charges nothing to its own warm-time
cap inside it. A window whose pings stopped for ``BEAT_GRACE_S`` (the process died) reads as closed.

**On SageMaker** (``ANJO_SLEEP_BACKEND``, ``serving.backend``) the same command wakes the inference
component (copy count 1, then the poll) and pings through the endpoint. A keep-warm window runs as
one phase, so its end, an error or Ctrl-C sets the copy count back to 0, and the idle alarm does it
after 15 minutes without a call if this process dies. ``--wake`` alone leaves Glimmer up for the
demo that follows, unless the wake failed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeGuard

import httpx

from anjo.sleep.serving import backend, client
from anjo.sleep.serving.config import ServeConfigError

URL_ENV: Final = backend.MODAL_URL_ENV
TOKEN_ENV: Final = backend.MODAL_TOKEN_ENV
KEEP_WARM_PING_S: Final = 240.0  # well under Modal's 600 s scale-down window
MAX_KEEP_WARM_MIN: Final = 180
BEAT_GRACE_S: Final = KEEP_WARM_PING_S + 60.0  # a window this long without an answered ping is closed
DATA_DIR_ENV: Final = "ANJO_DATA_DIR"
WINDOW_FILE: Final = "warm_window.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmWindow:
    """A keep-warm window in progress: when it ends, and its last answered ping (epoch seconds)."""

    until: float
    beat: float

    def open_at(self, now: float) -> bool:
        return now < self.until and 0.0 <= now - self.beat < BEAT_GRACE_S


def window_path(data_dir: Path) -> Path:
    return data_dir / "sleep" / "serving" / WINDOW_FILE


def read_window(path: Path) -> WarmWindow | None:
    """The window ``path`` holds, or None: no file, or one that can't be read, is no window."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    until, beat = data.get("until"), data.get("beat")
    if not (_finite(until) and _finite(beat)):
        return None
    return WarmWindow(float(until), float(beat))


def _finite(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _write_window(path: Path, window: WarmWindow) -> None:
    part = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps({"until": window.until, "beat": window.beat}), encoding="utf-8")
        os.replace(part, path)
    except OSError as error:  # the demo then sees no window: it answers less, never more
        logger.warning("the keep-warm window file wasn't written: %s", type(error).__name__)


@dataclass(frozen=True)
class KeepWarmResult:
    seconds: float
    pings: int
    failed: int


def keep_warm(
    base_url: str, token: str, minutes: float, *, transport: httpx.BaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
    on_ping: Callable[[bool], None] | None = None,
) -> KeepWarmResult:
    """Ping Modal every ``KEEP_WARM_PING_S`` until ``minutes`` (at most ``MAX_KEEP_WARM_MIN``) pass."""
    return keep_warm_with(lambda: client.ping(base_url, token, transport=transport), minutes,
                          clock=clock, sleep=sleep, on_ping=on_ping)


def keep_warm_with(
    ping: Callable[[], bool], minutes: float, *, clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None, on_ping: Callable[[bool], None] | None = None,
) -> KeepWarmResult:
    """Call ``ping`` every ``KEEP_WARM_PING_S`` until ``minutes`` (at most ``MAX_KEEP_WARM_MIN``) pass,
    telling ``on_ping`` whether each one was answered."""
    now, pause = clock or time.monotonic, sleep or time.sleep
    window_s = min(minutes, MAX_KEEP_WARM_MIN) * 60.0
    started = now()
    pings = failed = 0
    while True:
        pings += 1
        answered = ping()
        if not answered:
            failed += 1
        if on_ping is not None:
            on_ping(answered)
        left = window_s - (now() - started)
        if left <= 0:
            return KeepWarmResult(now() - started, pings, failed)
        pause(min(KEEP_WARM_PING_S, left))


def _keep_warm(serving: backend.Serving, minutes: float, on_ping: Callable[[bool], None]) -> KeepWarmResult:
    if isinstance(serving, backend.ModalServing):
        return keep_warm(serving.url, serving.token, minutes, on_ping=on_ping)  # Modal's window, as before
    return keep_warm_with(serving.ping, minutes, on_ping=on_ping)


def window_file() -> Path | None:
    """Where the live page reads the window: under ``$ANJO_DATA_DIR`` resolved as the app resolves it
    (``anjo.core.paths``), or None when it isn't set."""
    root = os.environ.get(DATA_DIR_ENV, "").strip()
    return window_path(Path(root).expanduser().resolve()) if root else None


def _window_beat(path: Path, minutes: float) -> Callable[[bool], None]:
    """The listener that writes the window file after each answered ping. Wall time: the page
    reading it runs in another process."""
    until = time.time() + minutes * 60.0

    def beat(answered: bool) -> None:
        if answered:
            _write_window(path, WarmWindow(until, time.time()))

    return beat


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wake Glimmer, and optionally keep it warm.")
    parser.add_argument("--wake", action="store_true", help="wake it and print the seconds it took")
    parser.add_argument("--keep-warm-minutes", type=float, help=f"wake it, then keep it up (at most {MAX_KEEP_WARM_MIN})")
    args = parser.parse_args(argv)
    if not args.wake and args.keep_warm_minutes is None:
        parser.error("pass --wake or --keep-warm-minutes N")
    if args.keep_warm_minutes is not None and args.keep_warm_minutes <= 0:
        parser.error("--keep-warm-minutes is above 0")
    try:
        serving = backend.from_env()
    except ServeConfigError as error:
        raise SystemExit(str(error)) from None
    if args.keep_warm_minutes is None:
        woke = None
        try:
            woke = serving.wake()
        finally:
            if woke is None or not woke.ok:
                serving.sleep_now()  # a wake that failed or raised leaves nothing up
        print(json.dumps({"wake_seconds": round(woke.seconds, 1), "ok": woke.ok}))
        return 0 if woke.ok else 1
    path = window_file()
    if path is None:  # before anything wakes
        parser.error(f"--keep-warm-minutes needs {DATA_DIR_ENV}, the app's data dir, where the live page reads the window")
    with backend.phase(serving.sleep_now):  # the window ends asleep, however it ends
        woke = serving.wake()
        print(json.dumps({"wake_seconds": round(woke.seconds, 1), "ok": woke.ok}))
        if not woke.ok:
            return 1
        minutes = min(args.keep_warm_minutes, MAX_KEEP_WARM_MIN)
        try:
            done = _keep_warm(serving, minutes, _window_beat(path, minutes))
        finally:
            path.unlink(missing_ok=True)
        print(json.dumps({"keep_warm_minutes": minutes, "pings": done.pings, "failed_pings": done.failed,
                          "seconds": round(done.seconds, 1)}))
        return 0 if done.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
