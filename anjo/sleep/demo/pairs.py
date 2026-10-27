"""The blind pair (Task 13a): which side is which stays on the server until the vote.

``open`` keeps a pair under an opaque random id: which arm is on each side, with its prompt length
and cost per turn. The page gets the id and the two texts in that order, nothing else, so nothing
sent before the vote carries the mapping. ``vote`` takes left, right, tie or neither, writes the vote
through ``record`` (the choice, the side mapping and the time: never a word of the question or of
either reply), and only then reveals the sides. A pair takes one vote, and a pair with a withheld
side takes none.

**Per visit.** A pair belongs to the visit that opened it; to any other it doesn't exist. A visit
takes at most ``VOTES_PER_VISIT`` votes and holds at most ``OPEN_PER_VISIT`` pairs open, so one
visit's flood can't crowd the registry. ``check_room`` refuses an ask before it spends its calls on a
pair ``open`` would refuse. One vote per starter scene is the address's, a day
(``Limiter.claim_scene_vote``), since a reload starts a new visit.

Pairs live in memory for ``PAIR_TTL_S``. Past ``MAX_PAIRS`` a new pair is refused (``busy``) rather
than the oldest dropped, so a flood never expires someone's pair before they vote. A restart forgets
the open ones, and a vote on one is refused as expired.
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
OPEN_PER_VISIT: Final = 8  # unvoted pairs a visit may hold at once
VOTES_PER_VISIT: Final = 15  # the five scenes and ten questions of their own
MAX_PAIRS: Final = 20_000  # above every visit's open pairs together (Limits.visits_per_day × 8 < 20,000)
MAX_TRACKED: Final = 10_000  # visits whose votes are remembered; past it the least recent go first


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
    visit: str
    scene: str | None  # a starter scene's id; None for a question of the visitor's own
    left: SideFacts
    right: SideFacts
    votable: bool
    expires: float
    voted: bool = False

    @property
    def waiting(self) -> bool:
        """Open for this visit's vote: what ``OPEN_PER_VISIT`` counts."""
        return self.votable and not self.voted


class PairRegistry:
    def __init__(self, *, ttl_s: float = PAIR_TTL_S, max_pairs: int = MAX_PAIRS) -> None:
        self._ttl_s = ttl_s
        self._max = max_pairs
        self._pairs: OrderedDict[str, _Pair] = OrderedDict()
        self._waiting: dict[str, int] = {}
        self._votes: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def open(
        self, pair_id: str, *, visit: str, scene: str | None = None, left: SideFacts, right: SideFacts,
        votable: bool, now: datetime,
    ) -> None:
        with self._lock:
            self._sweep(now)
            if pair_id in self._pairs:
                raise ValueError("a pair id is used once")
            self._room(visit, votable)
            pair = _Pair(visit, scene, left, right, votable, now.timestamp() + self._ttl_s)
            self._pairs[pair_id] = pair
            self._count(pair, 1)

    def check_room(self, visit: str, now: datetime) -> None:
        """Refuse now what opening one more of ``visit``'s pairs would: too many open, or no room."""
        with self._lock:
            self._sweep(now)
            self._room(visit, votable=True)

    def scene_of(self, pair_id: str, *, visit: str) -> str | None:
        """The starter scene of ``visit``'s pair, or None: a question of its own, or no such pair."""
        with self._lock:
            pair = self._pairs.get(pair_id)
            return None if pair is None or pair.visit != visit else pair.scene

    def vote(
        self, pair_id: str, choice: str, *, visit: str, now: datetime, record: Callable[[Vote], None],
    ) -> Reveal:
        """Record the vote, then reveal. ``record`` raising leaves the pair open to vote again."""
        picked = next((each for each in CHOICES if each == choice), None)
        if picked is None:
            raise Refused(422, "bad_choice")
        with self._lock:
            self._sweep(now)
            pair = self._pairs.get(pair_id)
            if pair is None or pair.visit != visit:
                raise Refused(404, "pair_gone")
            self._check(pair)
            vote = Vote(picked, pair.left.arm, pair.right.arm, now.isoformat(timespec="seconds"))
            record(vote)
            self._count(pair, -1)
            self._pairs[pair_id] = replace(pair, voted=True)
            self._remember(self._votes, visit, self._votes.get(visit, 0) + 1)
            return Reveal(vote, pair.left, pair.right)

    def _room(self, visit: str, votable: bool) -> None:
        if votable and self._waiting.get(visit, 0) >= OPEN_PER_VISIT:
            raise Refused(429, "too_many_open")
        if len(self._pairs) >= self._max:
            raise Refused(503, "busy")

    def _check(self, pair: _Pair) -> None:
        if pair.voted:
            raise Refused(409, "already_voted")
        if not pair.votable:
            raise Refused(409, "not_votable")
        if self._votes.get(pair.visit, 0) >= VOTES_PER_VISIT:
            raise Refused(429, "votes_spent")

    def _count(self, pair: _Pair, change: int) -> None:
        if not pair.waiting:
            return
        left = self._waiting.get(pair.visit, 0) + change
        if left > 0:
            self._waiting[pair.visit] = left
        else:
            self._waiting.pop(pair.visit, None)

    def _sweep(self, now: datetime) -> None:
        """Every pair lives ``ttl_s`` and a vote keeps its place, so the oldest is always first."""
        moment = now.timestamp()
        while self._pairs:
            pair = next(iter(self._pairs.values()))
            if pair.expires > moment:
                return
            self._pairs.popitem(last=False)
            self._count(pair, -1)

    @staticmethod
    def _remember(table: OrderedDict[str, int], key: str, value: int) -> None:
        table[key] = value
        table.move_to_end(key)
        while len(table) > MAX_TRACKED:
            table.popitem(last=False)
