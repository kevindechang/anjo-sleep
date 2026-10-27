"""Wake Glimmer on Modal, and keep it warm for a live window (``python -m anjo.sleep.serving.wake``).

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
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import httpx

from anjo.sleep.serving import client

URL_ENV: Final = "ANJO_SLEEP_MODAL_URL"
TOKEN_ENV: Final = "ANJO_SLEEP_MODAL_TOKEN"
KEEP_WARM_PING_S: Final = 240.0  # well under Modal's 600 s scale-down window
MAX_KEEP_WARM_MIN: Final = 180


@dataclass(frozen=True)
class KeepWarmResult:
    seconds: float
    pings: int
    failed: int


def keep_warm(
    base_url: str, token: str, minutes: float, *, transport: httpx.BaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
) -> KeepWarmResult:
    """Ping every ``KEEP_WARM_PING_S`` until ``minutes`` (at most ``MAX_KEEP_WARM_MIN``) pass."""
    window_s = min(minutes, MAX_KEEP_WARM_MIN) * 60.0
    started = clock()
    pings = failed = 0
    while True:
        pings += 1
        if not client.ping(base_url, token, transport=transport):
            failed += 1
        left = window_s - (clock() - started)
        if left <= 0:
            return KeepWarmResult(clock() - started, pings, failed)
        sleep(min(KEEP_WARM_PING_S, left))


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wake Glimmer on Modal, and optionally keep it warm.")
    parser.add_argument("--wake", action="store_true", help="wake it and print the seconds it took")
    parser.add_argument("--keep-warm-minutes", type=float, help=f"wake it, then keep it up (at most {MAX_KEEP_WARM_MIN})")
    args = parser.parse_args(argv)
    if not args.wake and args.keep_warm_minutes is None:
        parser.error("pass --wake or --keep-warm-minutes N")
    if args.keep_warm_minutes is not None and args.keep_warm_minutes <= 0:
        parser.error("--keep-warm-minutes is above 0")
    url, token = _env(URL_ENV), _env(TOKEN_ENV)
    woke = client.wake(url, token, client.WAKE_TOTAL_TIMEOUT_S)
    print(json.dumps({"wake_seconds": round(woke.seconds, 1), "ok": woke.ok}))
    if not woke.ok:
        return 1
    if args.keep_warm_minutes is None:
        return 0
    minutes = min(args.keep_warm_minutes, MAX_KEEP_WARM_MIN)
    done = keep_warm(url, token, minutes)
    print(json.dumps({"keep_warm_minutes": minutes, "pings": done.pings, "failed_pings": done.failed,
                      "seconds": round(done.seconds, 1)}))
    return 0 if done.failed < done.pings else 1


if __name__ == "__main__":
    raise SystemExit(main())
