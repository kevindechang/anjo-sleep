"""The sweep at the start of every night command (Task 5).

A run killed hard (SIGKILL, power loss) runs no ``finally``: its holds stay open at their full
bound forever, and its Colab VM may still run. So ``swept_night``, before any new hold:

1. takes the night lock (``night.lock`` beside the ledger). The OS frees an ``flock`` when its
   process dies, however it dies, so while the lock is held no other night run is live on this
   host. A second command stops with ``NightBusy`` and touches nothing;
2. stops every Colab session the runner made (the night command passes that step in, so this
   module never imports training). A stop that fails raises, and every hold stays open;
3. settles every open hold of the night's own kinds at its full bound. Work may have run, so the
   worst case is the honest charge: a release could undercount.

The lock is held until the run ends. The night log records the counts (``sweep``), never a key.
"""
from __future__ import annotations

import fcntl
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from anjo.sleep.budget import Ledger

NIGHT_KINDS: Final = ("night:", "night_gpu:", "colab_train:", "sim:")
LOCK_NAME: Final = "night.lock"


class NightBusy(RuntimeError):
    """Another night run is live on this host."""


@dataclass(frozen=True)
class Sweep:
    holds_settled: int
    sessions_stopped: int
    sessions_not_ours: int


def _settle_stale(ledger: Ledger) -> int:
    stale = [held for held in ledger.open_holds() if held.job.startswith(NIGHT_KINDS)]
    for held in stale:
        held.settle(held.bound.amount)
    return len(stale)


@contextmanager
def swept_night(ledger: Ledger, stop_sessions: Callable[[], tuple[int, int]]) -> Iterator[Sweep]:
    """Hold the night lock for the run, after stopping leftover VMs and settling stale holds."""
    path = ledger.path.with_name(LOCK_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise NightBusy("another night run is live on this host; nothing was swept") from None
        try:
            stopped, not_ours = stop_sessions()  # VMs first: a hold is settled only once its VM is down
            yield Sweep(_settle_stale(ledger), stopped, not_ours)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
