"""The blind pair (Task 13a): which side is which stays on the server until the vote.

``open`` keeps a pair under an opaque random id: which arm is on each side, with its prompt length
and cost per turn. The page gets the id and the two texts in that order, nothing else, so nothing
sent before the vote carries the mapping. ``vote`` takes left, right, tie or neither, writes the vote
through ``record`` (the choice, the side mapping and the time: never a word of the question or of
either reply), and only then reveals the sides. A pair takes one vote, and a pair with a withheld
side takes none.

Pairs live in memory for ``PAIR_TTL_S``, the oldest leaving first past ``MAX_PAIRS``. A restart
forgets the open ones, and a vote on one is refused as expired.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Literal, get_args

from anjo.sleep.demo.limits import Refused

Choice = Literal["left", "right", "tie", "neither"]
CHOICES: Final[tuple[Choice, ...]] = get_args(Choice)
PAIR_TTL_S: Final = 30 * 60
MAX_PAIRS: Final = 5_000


@dataclass(frozen=True)
class SideFacts:
    """What the reveal shows of one side. ``cost_usd`` is None when it wasn't measured."""

    arm: str
    prompt_tokens: int
    cost_usd: float | None


@dataclass(frozen=True)
class Vote:
    choice: Choice
    left: str
    right: str
    at: str

    def record(self) -> dict[str, str]:
        """The stored line: the choice, the side mapping and the time, nothing else."""
        return {"choice": self.choice, "left": self.left, "right": self.right, "at": self.at}


@dataclass(frozen=True)
class Reveal:
    vote: Vote
    left: SideFacts
    right: SideFacts


@dataclass(frozen=True)
class _Pair:
    left: SideFacts
    right: SideFacts
    votable: bool
    expires: float
    voted: bool = False


class PairRegistry:
    def __init__(self, *, ttl_s: float = PAIR_TTL_S, max_pairs: int = MAX_PAIRS) -> None:
        self._ttl_s = ttl_s
        self._max = max_pairs
        self._pairs: OrderedDict[str, _Pair] = OrderedDict()
        self._lock = threading.Lock()

    def open(self, pair_id: str, *, left: SideFacts, right: SideFacts, votable: bool, now: datetime) -> None:
        with self._lock:
            self._sweep(now)
            if pair_id in self._pairs:
                raise ValueError("a pair id is used once")
            self._pairs[pair_id] = _Pair(left, right, votable, now.timestamp() + self._ttl_s)
            while len(self._pairs) > self._max:
                self._pairs.popitem(last=False)

    def vote(self, pair_id: str, choice: str, *, now: datetime, record: Callable[[Vote], None]) -> Reveal:
        """Record the vote, then reveal. ``record`` raising leaves the pair open to vote again."""
        picked = next((each for each in CHOICES if each == choice), None)
        if picked is None:
            raise Refused(422, "bad_choice")
        with self._lock:
            self._sweep(now)
            pair = self._pairs.get(pair_id)
            if pair is None:
                raise Refused(404, "pair_gone")
            if pair.voted:
                raise Refused(409, "already_voted")
            if not pair.votable:
                raise Refused(409, "not_votable")
            vote = Vote(picked, pair.left.arm, pair.right.arm, now.isoformat(timespec="seconds"))
            record(vote)
            self._pairs[pair_id] = replace(pair, voted=True)
            return Reveal(vote, pair.left, pair.right)

    def _sweep(self, now: datetime) -> None:
        moment = now.timestamp()
        for pair_id in [key for key, pair in self._pairs.items() if pair.expires <= moment]:
            del self._pairs[pair_id]
