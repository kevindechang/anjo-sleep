"""The batch's GPU session on the serving server (Task 5): ``modal_usd`` holds the batch's worst
case before the first due person practises, and is settled with what the session really held.

The worst case is the per-person hold (practice, a Colab training while the server may stay warm,
and three gates) times the due people, at the serving GPU's hourly price. A batch the pool can't
hold stops with ``BudgetStop`` before any GPU work, which fails every due person's night at
practice (RF1) and leaves the rest of the batch's steps to run.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import date
from typing import Final

from anjo.sleep.budget import Bound, Ledger
from anjo.sleep.stops import BudgetStop

POOL: Final = "modal_usd"
DEFAULT_HOLD_PER_PERSON_S: Final = 3 * 3600.0  # practice, the Colab training's hard timeout, gates


class ReservedGpu:
    """A ``GpuSession`` that reserves ``modal_usd`` for the batch."""

    def __init__(
        self, ledger: Ledger, night: date, *, usd_per_hour: float,
        hold_per_person_s: float = DEFAULT_HOLD_PER_PERSON_S, timer: Callable[[], float] = time.monotonic,
        stamp: Callable[[], int] = time.time_ns,
    ) -> None:
        self._ledger = ledger
        self._night = night
        self._rate = usd_per_hour
        self._hold = hold_per_person_s
        self._timer = timer
        self._stamp = stamp

    @contextmanager
    def _held(self, due: tuple[str, ...]) -> Iterator[None]:
        if not due:
            yield
            return
        digest = hashlib.sha256("\x1f".join(sorted(due)).encode()).hexdigest()[:12]
        job = f"night_gpu:{self._night.isoformat()}:{digest}:{self._stamp()}"
        reservation = self._ledger.reserve(job, (POOL,), Bound(self._rate, self._hold * len(due)))
        if reservation is None:
            raise BudgetStop(f"{POOL} can't hold this batch's worst case")
        started = self._timer()
        try:
            yield
        finally:
            reservation.settle(self._rate * max(0.0, self._timer() - started) / 3600.0)

    def open(self, due: Sequence[str]) -> AbstractContextManager[None]:
        return self._held(tuple(due))
