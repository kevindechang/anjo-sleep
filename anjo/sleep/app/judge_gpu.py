"""The judges' serving GPU on the budget ledger: held before it is spent, settled when it ends.

``judge_talker`` never wakes Glimmer or keeps her warm for a judge without ledger room for it first:

- **Before** a wake or a longer lease, ``GpuHolds.cover`` holds ``modal_usd`` (``WARM_POOLS``, the live
  page's pool) for the GPU time up to the moment it could last: the lease's end plus the backend's idle
  window (the tail if the stop fails), and while waking, the wake's whole timeout. Room comes in blocks
  of at least ``MIN_BLOCK_S``, one ledger hold each (``judge_gpu:<owner>:<n>``). The ledger refuses past
  the pool's cap or while other nights hold its room, and then the talker wakes nothing and Spark
  answers, labelled ``reason: budget``.
- **When the stretch ends**, ``settle`` charges the real time: each hold in order up to its bound, the
  rest on a hold of its own (or, refused, on the last hold as an overage: the money is spent), and a
  hold nothing reached is released.
- **A restart mid-stretch** leaves its holds open in the ledger at their full bound, so nothing is lost
  from the count. ``settle_orphans`` (run when the next talker starts) settles each one whose process
  is gone at that bound: work may have run, so the worst case is the honest charge, as the night's
  sweep settles its own kinds. It is not left to the night's sweep, which runs while this server is
  live and would settle a hold still in use.
"""
from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from typing import Final

from anjo.sleep.budget import Bound, BudgetError, Ledger, Pool, Reservation

logger = logging.getLogger(__name__)

HOLD_KIND: Final = "judge_gpu:"
POOLS: Final[tuple[Pool, ...]] = ("modal_usd",)  # demo_live.WARM_POOLS: the serving GPU's pool
MIN_BLOCK_S: Final = 900.0  # the least a new hold covers, so a present judge takes one every 15 minutes
_HOUR_S: Final = 3600.0
LEDGER_FAILURES: Final = (BudgetError, OSError)


def _owner() -> str:
    return f"{os.getpid()}-{secrets.token_hex(4)}"


class GpuHolds:
    """This process's holds for the stretch Glimmer is up (or waking) for the judges."""

    def __init__(self, ledger: Callable[[], Ledger], rate_per_hour: float, *, owner: str | None = None) -> None:
        self._ledger = ledger
        self._rate = rate_per_hour
        self.owner = owner or _owner()
        self._held: list[Reservation] = []
        self._held_s = 0.0
        self._start: float | None = None
        self._count = 0

    @property
    def held(self) -> bool:
        return bool(self._held)

    def covered_until(self) -> float | None:
        """The moment the stretch's holds run out at, or None with none held."""
        return None if self._start is None or not self._held else self._start + self._held_s

    def covers(self, until: float) -> bool:
        """Whether the holds already reach ``until``."""
        covered = self.covered_until()
        return covered is not None and until <= covered

    def cover(self, until: float, now: float) -> bool:
        """Hold ledger room for the GPU from the stretch's start to ``until``; True when it is held
        (already, or now), False when the ledger refuses or can't be read."""
        if self._start is None or not self._held:
            self._start = now
        gap = until - (self._start + self._held_s)
        if gap <= 0:
            return True
        seconds = max(gap, MIN_BLOCK_S)
        found = self._reserve(seconds)
        if found is None:
            return False
        self._held.append(found)
        self._held_s += seconds
        return True

    def _reserve(self, seconds: float) -> Reservation | None:
        key = f"{HOLD_KIND}{self.owner}:{self._count}"
        try:
            found = self._ledger().reserve(key, POOLS, Bound(self._rate, seconds))
        except LEDGER_FAILURES as error:
            logger.error("the judges' GPU hold couldn't be taken: %s", type(error).__name__)
            return None
        self._count += 1
        if found is None:
            logger.warning("the budget ledger refused the judges' GPU time ($%.2f)", self._rate * seconds / _HOUR_S)
        return found

    def settle(self, seconds: float) -> None:
        """The stretch ended after ``seconds`` of GPU time: settle every hold with its share."""
        cost = max(0.0, seconds) * self._rate / _HOUR_S
        held, self._held, self._held_s, self._start = self._held, [], 0.0, None
        shares = []
        for hold in held:
            shares.append(min(cost, hold.bound.amount))
            cost -= shares[-1]
        if cost > 1e-9:  # time past its holds: a hold of its own, settled at once
            extra = self._reserve(cost * _HOUR_S / self._rate)
            if extra is not None:
                self._close(extra, extra.bound.amount)
            elif held:  # refused: the money is spent, so the last hold records it as an overage
                logger.error("the judges' GPU ran $%.4f past its holds; recorded as an overage", cost)
                shares[-1] += cost
            else:
                logger.error("the judges' GPU ran $%.4f with no hold the ledger would take", cost)
        for hold, share in zip(held, shares, strict=True):
            self._close(hold, share)

    def _close(self, hold: Reservation, share: float) -> None:
        try:
            if share > 0:
                hold.settle(share)
            else:
                hold.release()
        except LEDGER_FAILURES as error:
            logger.error("a judges' GPU hold wasn't settled: %s", type(error).__name__)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def settle_orphans(ledger: Ledger, owner: str, alive: Callable[[int], bool] = _alive) -> int:
    """Settle at full bound every open judge GPU hold left by a process that is gone (``owner`` is
    this process's own, never touched). Returns how many."""
    settled = 0
    for held in ledger.open_holds():
        if not held.job.startswith(HOLD_KIND):
            continue
        holder = held.job[len(HOLD_KIND):].split(":", 1)[0]
        pid = holder.split("-", 1)[0]
        if holder == owner or not pid.isdigit():
            continue
        if int(pid) != os.getpid() and alive(int(pid)):
            continue  # another live process's stretch
        held.settle(held.bound.amount)
        settled += 1
    if settled:
        logger.warning("settled %d judge GPU holds a stopped process left open, at their full bound", settled)
    return settled
