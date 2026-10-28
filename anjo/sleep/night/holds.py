"""Budget holds for the night's paid steps (Task 5).

Every paid step reserves its worst case before its first call (``hold``) and settles what it
metered, or releases the hold, when it ends (``held``). A pool without room stops the step with
``BudgetStop`` before any call. A job's key names the night, the step and the person, with an
attempt number: a rerun takes the next attempt after one that ended, so nothing is held or charged
twice. A hold a killed run left open is settled at its full bound by the night command's sweep
(``sweep.py``) before the rerun starts.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from anjo.sleep.budget import Bound, BudgetError, Ledger, Pool, Reservation
from anjo.sleep.stops import BudgetStop

if TYPE_CHECKING:  # the training runners hold through here, and need none of the night's ports
    from anjo.sleep.night.deps import Prices

MAX_ATTEMPTS: Final = 50
HOUR_S: Final = 3600.0
SPARK_MAX_IN_TOKENS: Final = 32_768  # the most one Spark Open call reads
SPARK_MAX_OUT_TOKENS: Final = 8_192  # and writes


def spark_usd(prices: Prices, tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * prices.spark_in_per_m + tokens_out * prices.spark_out_per_m) / 1e6


def spark_worst_usd(prices: Prices, calls: int) -> float:
    """The most ``calls`` Spark Open calls can cost."""
    return spark_usd(prices, calls * SPARK_MAX_IN_TOKENS, calls * SPARK_MAX_OUT_TOKENS)


def usd(amount: float) -> Bound:
    """A worst case in dollars, as the ledger's rate over an hour."""
    return Bound(amount, HOUR_S)


def attempt_key(base: str, attempt: int) -> str:
    return f"{base}:a{attempt}"


def hold(ledger: Ledger, base: str, pools: tuple[Pool, ...], bound: Bound) -> Reservation:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            found = ledger.reserve(attempt_key(base, attempt), pools, bound)
        except BudgetError:  # this attempt ended, or holds another bound: try the next one
            continue
        if found is None:
            raise BudgetStop(f"{'/'.join(pools)} can't hold the worst case of {base}")
        return found
    raise BudgetError(f"{base} has used all {MAX_ATTEMPTS} attempts")


@contextmanager
def held(
    ledger: Ledger | None, base: str, pools: tuple[Pool, ...], bound: Bound, actual: Callable[[], float],
) -> Iterator[None]:
    """Hold ``bound`` while the step runs, then settle ``actual()`` (or release a step that spent
    nothing). Without a ledger (tests, the public example) nothing is held."""
    if ledger is None:
        yield
        return
    reservation = hold(ledger, base, pools, bound)
    try:
        yield
    finally:
        spent = actual()
        if spent > 0:
            reservation.settle(spent)
        else:
            reservation.release()
