"""A run's spend ceiling (``--max-usd``) for APD's paid commands (Task 17).

The pools' caps bound what all of MIND may spend; the ceiling bounds one command: ``generate``,
``run``, ``timing`` or ``judge``. Before each new unit of work (a generation attempt, a life-arm
run, timing's 17b sample or intervention probe, a new judge call) the command sums what its own
holds have settled (``RunLedger.run_usd``: the keys this command reserved, never the pools' totals,
which hold earlier runs' and Sleep's spend too), and once that reaches the ceiling it starts nothing
more (``CeilingReached``), so it takes no new hold. A unit already running finishes under its own
hold, so a command passes its ceiling by at most the units in flight: one, or with ``--parallel N``
up to N life-arms. Colab's compute units aren't dollars and are capped apart (``colab_cu``).

After each life generated and each life-arm run, the command prints what it has settled per life
or per arm-day, so a run on one life can be extrapolated before the rest. With life-arms running in
parallel, the nights the running ones already settled count too, so the figure leans high until
the batch ends.
"""
from __future__ import annotations

import argparse
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from anjo.sleep.budget import CAPS, POOLS, Bound, Ledger, Pool, Reservation

DEFAULT_MAX_USD: Final = 20.0
COMMANDS: Final = ("generate", "generate-pairs", "run", "timing", "judge")
UNITS: Final[Mapping[str, str]] = {  # what the ceiling is checked before, as the plan names it
    "generate": "generation attempt", "generate-pairs": "pair generation attempt", "run": "life-arm run",
    "timing": "life-arm run or timing step", "judge": "judge call",
}
USD_POOLS: Final = frozenset(pool for pool in POOLS if pool.endswith("_usd"))


def dollars(value: float) -> str:
    """A ceiling as printed: cents, or four places below a dollar."""
    return f"${value:,.2f}" if value >= 1 else f"${value:.4f}"


class CeilingReached(RuntimeError):
    """What this command has settled reached its ceiling: nothing new starts."""


def max_usd(text: str) -> float:
    """``--max-usd``: a finite dollar amount above 0."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--max-usd is a dollar amount, not {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"--max-usd is a dollar amount above 0, not {text!r}")
    return value


class RunLedger(Ledger):
    """The shared ledger as one command sees it: the same file and caps, keeping the key and pools
    of every hold this command takes, so its own spend is summed apart from everyone else's."""

    def __init__(self, path: Path, *, caps: Mapping[Pool, float] = CAPS) -> None:
        super().__init__(path, caps=caps)
        self._mine: dict[str, tuple[Pool, ...]] = {}
        self._lock = threading.Lock()

    def reserve(self, job: str, pools: tuple[Pool, ...], bound: Bound) -> Reservation | None:
        found = super().reserve(job, pools, bound)
        if found is not None:
            with self._lock:
                self._mine[found.job] = found.pools
        return found

    def run_usd(self) -> float:
        """What this command's dollar holds have settled; open and released ones add nothing."""
        with self._lock:
            jobs = [job for job, pools in self._mine.items() if set(pools) <= USD_POOLS]
        return self.settled(jobs)


@dataclass
class Ceiling:
    """One command's ceiling over ``ledger``, and how far its current step has got: ``done`` of
    ``total`` units (``None`` when unknown ahead), and the arm-days of its finished life-arms."""

    max_usd: float
    ledger: RunLedger
    unit: str = "unit(s)"
    total: int | None = None
    done: int = 0
    arm_days: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def begin(self, unit: str, total: int | None) -> None:
        """A step of ``total`` units, each a ``unit``, as the stop message counts them."""
        with self._lock:
            self.unit, self.total, self.done, self.arm_days = unit, total, 0, 0

    def check(self) -> None:
        """Before a new unit: ``CeilingReached`` once this command has settled its ceiling."""
        spent = self.ledger.run_usd()
        if spent >= self.max_usd:
            raise CeilingReached(f"this run settled ${spent:.4f}, at or past its {dollars(self.max_usd)} ceiling")

    def counted(self) -> None:
        with self._lock:
            self.done += 1

    def life_generated(self, what: str = "life") -> str:
        """One more life (or pair) generated: what this run has settled, and per one."""
        with self._lock:
            self.done += 1
            lives = self.done
        spent = self.ledger.run_usd()
        many = "lives" if what == "life" else f"{what}s"
        return (f"this run's spend: ${spent:.4f} settled for {lives} {what if lives == 1 else many} generated, "
                f"${spent / lives:.4f} per {what} generated (ceiling {dollars(self.max_usd)})")

    def arm_ran(self, days: int) -> str:
        """One more life-arm run of ``days`` nights: what this run has settled, and per arm-day."""
        with self._lock:
            self.done += 1
            self.arm_days += days
            runs, arm_days = self.done, self.arm_days
        spent = self.ledger.run_usd()
        per = spent / arm_days if arm_days else 0.0
        return (f"this run's spend: ${spent:.4f} settled for {runs} life-arm run(s), {arm_days} arm-days, "
                f"${per:.5f} per arm-day (ceiling {dollars(self.max_usd)})")

    def stopped(self) -> str:
        """The stop message, once the units in flight have finished."""
        with self._lock:
            done, total, unit = self.done, self.total, self.unit
        got = f"{done} {unit}" if total is None else f"{done} of {total} {unit}"
        return (f"Ceiling: this run settled ${self.ledger.run_usd():.4f} of its {dollars(self.max_usd)} ceiling "
                f"(--max-usd) after {got}; nothing new started, and what was running finished under its own "
                "hold. What finished is kept, and a rerun with a higher --max-usd goes on from there.")
