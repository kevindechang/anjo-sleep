"""MIND's lock and sweep (Tasks 17 and 17b), beside the night's (``night.sweep``).

A MIND command killed hard runs no ``finally``: its holds stay open at their full bound, and its
Colab session may still run. So every APD command, and replay's training when it trains, first
takes MIND's lock (``mind.lock``, beside the ledger; in the data dir when nothing is held). The OS
frees an ``flock`` when its process dies, however it dies, so while the lock is held no other MIND
command is live on this host; a second stops with ``MindBusy`` and touches nothing. Only the
holder sweeps (``swept_mind``), before any new hold:

1. it stops every MIND Colab session (``anjo-mind-…``). A stop that fails raises, and every hold
   stays open;
2. it settles each night a killed run's journal holds open, at what it priced plus one call's
   worst case (the command passes this step in);
3. it settles every other open hold of MIND's kinds (``MIND_KINDS``) at its full bound: work may
   have run, so the worst case is the honest charge.

MIND's kinds and the night's (``night.sweep.NIGHT_KINDS``) share no prefix, and the two locks are
separate files, so a Sleep night and a MIND command run side by side and neither sweep settles the
other's holds or stops the other's sessions. The lock is re-entrant within a command: a step that
takes it inside a command that holds it neither waits nor sweeps.
"""
from __future__ import annotations

import fcntl
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Ledger
from anjo.sleep.train.runners.colab import HOLD_KINDS, MIND_SESSION_PREFIX

MIND_KINDS: Final = ("apd:", "replay:", *(f"{kind}:" for kind in HOLD_KINDS[MIND_SESSION_PREFIX]),
                     f"colab_serve:{MIND_SESSION_PREFIX}")
LOCK_NAME: Final = "mind.lock"
_HELD: ContextVar[frozenset[Path]] = ContextVar("mind_lock_held", default=frozenset())


class MindBusy(RuntimeError):
    """Another MIND command is live on this host."""


@dataclass(frozen=True)
class MindSweep:
    sessions_stopped: int
    sessions_not_ours: int  # another tool's, only counted
    journals_usd: float  # what killed runs' journals settled
    holds_settled: int  # at their full bound


def lock_path(ledger: Ledger | None, root: Path) -> Path:
    """MIND's lock: beside the ledger, or in ``root`` when nothing is held."""
    return (root / LOCK_NAME if ledger is None else ledger.path.with_name(LOCK_NAME)).resolve()


@contextmanager
def mind_lock(path: Path) -> Iterator[bool]:
    """MIND's lock for this command; yields whether this call took it (``False``: the command
    around it holds it already)."""
    if path in _HELD.get():
        yield False
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise MindBusy("another MIND command is live on this host; nothing was swept") from None
        token = _HELD.set(_HELD.get() | {path})
        try:
            yield True
        finally:
            _HELD.reset(token)
            fcntl.flock(lock, fcntl.LOCK_UN)


def settle_stale(ledger: Ledger) -> int:
    stale = [held for held in ledger.open_holds() if held.job.startswith(MIND_KINDS)]
    for held in stale:
        held.settle(held.bound.amount)
    return len(stale)


@contextmanager
def swept_mind(
    path: Path, ledger: Ledger | None, stop_sessions: Callable[[], tuple[int, int]],
    recover_journals: Callable[[], float],
) -> Iterator[MindSweep | None]:
    """MIND's lock for the command, after stopping leftover sessions and settling stale holds;
    ``None`` when the command around it holds the lock (and swept)."""
    with mind_lock(path) as took:
        if not took:
            yield None
            return
        stopped, not_ours = stop_sessions()  # VMs first: a hold is settled only once its VM is down
        journals = recover_journals()
        yield MindSweep(stopped, not_ours, journals, 0 if ledger is None else settle_stale(ledger))
