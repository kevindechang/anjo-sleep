"""The dream curriculum: what the night's dreams pick, she rehearses (the one system's step 1 to 2).

After the day's events are observed and the core has dreamt, each open thread that cites a real
event scores how unsure or at risk she is about it: its risk, plus the predicted error of her own
forecast (``self_error``), plus the strongest ignition naming it (as either slot of a collision).
The top ``threads`` threads share the rehearsal ``budget`` (counted in events) by their surprise:
the mean surprise weight of the events each cites (the Sleep link's weights, 1.0 for an event
without one). Every picked thread gets one rehearsal first, in rank order; the rest goes, one at
a time, to the thread furthest below its share that still has an event to give, so a share a
thread can't spend passes on. Each thread gives its cited events strongest write first.

Practice is then seeded on the picked events (``evidence_ids``) instead of the day's newest ones,
under the same scenes and the same token cap. Pure: no torch, no clock, no model call.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from anjo.sleep.mind.core.types import Ignition, Predictions

MAX_THREADS: Final = 3  # threads a night rehearses
DEFAULT_WEIGHT: Final = 1.0


@dataclass(frozen=True)
class Thread:
    slot: int
    score: float  # risk + self_error + the strongest ignition naming it
    cites: tuple[str, ...]  # its real events, strongest write first
    share: float = 0.0  # its share of the rehearsal budget


@dataclass(frozen=True)
class Picks:
    threads: tuple[Thread, ...]
    evidence_ids: tuple[str, ...]  # what practice is seeded on: each thread's allotment, in rank order


def rank_threads(predictions: Predictions, ignitions: Sequence[Ignition] = ()) -> tuple[Thread, ...]:
    """Every open thread that cites a real event, the most unsure or at risk first (ties: lower slot)."""
    lit: dict[int, float] = {}
    for ignition in ignitions:
        for slot in (ignition.slot, ignition.other):
            if slot is not None:
                lit[slot] = max(lit.get(slot, 0.0), ignition.score)
    threads = [
        Thread(s.slot, s.risk + s.self_error + lit.get(s.slot, 0.0), s.cites)
        for s in predictions.slots if s.cites
    ]
    return tuple(sorted(threads, key=lambda t: (-t.score, t.slot)))


def _allot(sizes: Sequence[int], shares: Sequence[float], budget: int) -> list[int]:
    given = [1 if n < budget else 0 for n in range(len(sizes))]
    left = budget - sum(given)
    while left > 0:
        room = [n for n in range(len(sizes)) if given[n] < sizes[n]]
        if not room:
            break
        n = max(room, key=lambda i: (shares[i] * budget - given[i], -i))
        given[n] += 1
        left -= 1
    return given


def pick(
    predictions: Predictions, ignitions: Sequence[Ignition], weights: Mapping[str, float], *, budget: int,
    threads: int = MAX_THREADS,
) -> Picks:
    """The night's rehearsal: the top ``threads`` threads and ``budget`` of their cited events."""
    if budget < 1 or threads < 1:
        raise ValueError("a rehearsal budget and a thread count are positive")
    top = rank_threads(predictions, ignitions)[:threads]
    if not top:
        return Picks((), ())
    surprise = [math.fsum(weights.get(c, DEFAULT_WEIGHT) for c in t.cites) / len(t.cites) for t in top]
    total = math.fsum(surprise)
    shares = [s / total if total > 0 else 1.0 / len(top) for s in surprise]
    given = _allot([len(t.cites) for t in top], shares, budget)
    chosen = tuple(Thread(t.slot, t.score, t.cites, share) for t, share in zip(top, shares, strict=True))
    ids = [c for t, n in zip(top, given, strict=True) for c in t.cites[:n]]
    return Picks(chosen, tuple(dict.fromkeys(ids)))


def record(picks: Picks) -> dict[str, Any]:
    """The picks for the night log: slots, numbers and event ids, never words."""
    return {
        "threads": [{"slot": t.slot, "score": round(t.score, 6), "share": round(t.share, 6), "cites": list(t.cites)}
                    for t in picks.threads],
        "evidence_ids": list(picks.evidence_ids),
    }
