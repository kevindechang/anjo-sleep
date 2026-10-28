"""Spark Open's spend, journaled call by call (Task 17): what settles a hold whose process died.

A run's ``SparkCalls`` (``live``) writes one line per event to its journal, each appended under
the file's lock and fsynced before the call that made it returns (``append_records``):

- ``held``: a night's hold was taken, under its ledger key;
- ``call``: one Spark Open call came back, with its real tokens and their price; a failed try
  (``failed``) is priced at one call's worst case, since what it cost is unknown;
- ``closed``: the night was settled at what its calls priced;
- ``recovered``: ``recover`` settled a night its process never closed.

A process killed mid-night leaves its hold open in the ledger and no ``closed`` line. ``recover``
settles each such hold at what its journal priced plus one call's worst case, for the call that
may have been in flight, which no journal can see; never above the hold. So the ledger stays true
however a run ends, and the journal says what each night really spent (``spent``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Ledger
from anjo.sleep.evidence.jsonl import PLAIN, append_records, read_records
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_worst_usd

OPS: Final = frozenset({"held", "call", "closed", "recovered"})


@dataclass(frozen=True)
class Call:
    tokens_in: int
    tokens_out: int
    usd: float
    failed: bool = False


@dataclass(frozen=True)
class Night:
    """One hold's journal: its calls, and how it ended (``closed``, or settled by ``recover`` at
    ``recovered`` dollars), if it has."""

    job: str
    calls: tuple[Call, ...]
    closed: bool
    recovered: float | None

    @property
    def usd(self) -> float:
        """What the night's calls priced."""
        return sum(call.usd for call in self.calls)

    @property
    def spent(self) -> float:
        """What the ledger settled the night at, or will: the recovery's amount, else its calls."""
        return self.usd if self.recovered is None else self.recovered


class SpendJournal:
    """The journal at ``path``: append-only, one process or thread writing a night at a time."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def held(self, job: str) -> None:
        self._write({"op": "held", "job": job})

    def call(self, job: str, tokens_in: int, tokens_out: int, usd: float, *, failed: bool = False) -> None:
        record: dict[str, object] = {"op": "call", "job": job, "in": tokens_in, "out": tokens_out, "usd": usd}
        self._write({**record, "failed": True} if failed else record)

    def closed(self, job: str) -> None:
        self._write({"op": "closed", "job": job})

    def recovered(self, job: str, usd: float) -> None:
        self._write({"op": "recovered", "job": job, "usd": usd})

    def jobs(self) -> tuple[Night, ...]:
        """Each night the journal holds, in the order its holds were taken."""
        calls: dict[str, list[Call]] = {}
        closed: set[str] = set()
        recovered: dict[str, float] = {}
        for record in read_records(self.path, PLAIN):
            op, job = record.get("op"), str(record.get("job"))
            if op not in OPS:
                raise ValueError(f"{self.path.name} holds an unknown event {op!r}")
            if op == "held":
                calls.setdefault(job, [])
            elif op == "call":
                calls.setdefault(job, []).append(Call(int(record["in"]), int(record["out"]), float(record["usd"]),
                                                      record.get("failed") is True))
            elif op == "closed":
                closed.add(job)
            else:
                recovered[job] = float(record["usd"])
        return tuple(Night(job, tuple(found), job in closed, recovered.get(job)) for job, found in calls.items())

    def spent(self) -> float:
        return sum(night.spent for night in self.jobs())

    def retire(self) -> Path | None:
        """Move the journal aside (``<name>.discarded-<n>.jsonl``) for a fresh attempt."""
        if not self.path.exists():
            return None
        n = 1
        while (moved := self.path.with_name(f"{self.path.stem}.discarded-{n}.jsonl")).exists():
            n += 1
        os.replace(self.path, moved)
        return moved

    def _write(self, record: dict[str, object]) -> None:
        append_records(self.path, PLAIN, [record])


def recover(ledger: Ledger, journal: SpendJournal, prices: Prices) -> float:
    """Settle each night ``journal`` held and never closed whose hold is still open: its calls'
    price plus one call's worst case, at most the hold. Returns the total it settled."""
    holds = {held.job: held for held in ledger.open_holds()}
    settled = 0.0
    for night in journal.jobs():
        held = holds.get(night.job)
        if night.closed or night.recovered is not None or held is None:
            continue
        amount = min(night.usd + spark_worst_usd(prices, 1), held.bound.amount)
        held.settle(amount)
        journal.recovered(night.job, amount)
        settled += amount
    return settled
