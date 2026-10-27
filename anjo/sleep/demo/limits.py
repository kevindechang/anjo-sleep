"""The live page's abuse limits (Task 13a), after the removed public web demo's design.

The page is public and unauthenticated, and a visitor's own question costs a model call per side
(and, from the full step, per slider stop), so every limit is held on the server:

- per client, counted by a keyed hash of the address (``visits.client_key``), never the address:
  visits a day, requests a minute and model calls a minute;
- per visit: questions of their own;
- for everyone: model calls in flight, model calls a day and tokens a day.

**Worst-case admission.** An ask holds its worst case in the same locked step as the checks: every
call at Glimmer's whole window, which prompt and reply together never exceed. When the calls
return, ``settle`` puts what they used in place of the hold; a call that failed keeps its worst
case, since it may have run. ``release`` gives back a hold whose calls never ran.

**Fail closed.** A change to the day's counters is saved to the ``CounterStore`` before memory takes
it. If the counters can't be read, or a change can't be saved, the request is refused (503) and
memory keeps the last saved day, so a restart never hands out a fresh one. A refund that can't be
saved is dropped: the day stays charged at the worst case. The minute windows, the calls in flight
and each visit's questions live in this process only (the app runs one worker), so a restart resets
them; the daily counts carry over.

Nothing here reads the clock: every call is given ``now``, an aware time.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

logger = logging.getLogger(__name__)

WINDOW_S: Final = 60.0
MAX_TRACKED: Final = 10_000  # clients or visits kept in memory; past it the least recent go first
GLIMMER_WINDOW_TOKENS: Final = 131_072  # arms.GLIMMER_WINDOW_TOKENS: no call can use more


@dataclass(frozen=True)
class Limits:
    visits_per_client_per_day: int = 6
    visits_per_day: int = 600
    requests_per_client_per_minute: int = 30
    calls_per_client_per_minute: int = 6  # three questions of their own: two calls each
    asks_per_visit: int = 10
    max_inflight_calls: int = 8
    calls_per_day: int = 400
    tokens_per_day: int = 20_000_000
    tokens_per_call: int = GLIMMER_WINDOW_TOKENS
    wakes_per_day: int = 3  # only when waking on demand is switched on


class Refused(Exception):
    """A request the demo turns away. ``reason`` is a key of ``words.COPY``, the visitor's text."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class CounterError(RuntimeError):
    """The day's counters can't be read or saved."""


class CounterStore(Protocol):
    def load(self) -> Mapping[str, Any] | None:
        """The saved counters, or None before the first save. Raises ``CounterError``."""

    def save(self, counters: Mapping[str, Any]) -> None:
        """Replace the saved counters. Raises ``CounterError``."""


class JsonCounterStore:
    """One JSON file, written beside itself and renamed over the old one, so it is never torn."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> Mapping[str, Any] | None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as error:
            raise CounterError(f"the demo counters can't be read ({type(error).__name__})") from None
        try:
            data = json.loads(raw)
        except ValueError:
            raise CounterError("the demo counters are not JSON") from None
        if not isinstance(data, dict):
            raise CounterError("the demo counters are not an object")
        return data

    def save(self, counters: Mapping[str, Any]) -> None:
        part = self._path.with_name(f"{self._path.name}.part")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "w", encoding="utf-8") as handle:
                json.dump(dict(counters), handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(part, self._path)
        except OSError as error:
            raise CounterError(f"the demo counters can't be saved ({type(error).__name__})") from None


@dataclass(frozen=True)
class Day:
    """One UTC day's counts. ``visits_by_client`` is keyed by ``client_key``, never an address."""

    day: str
    visits: int = 0
    visits_by_client: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    calls: int = 0
    tokens: int = 0
    wakes: int = 0

    def dump(self) -> dict[str, Any]:
        return {"day": self.day, "visits": self.visits, "visits_by_client": dict(self.visits_by_client),
                "calls": self.calls, "tokens": self.tokens, "wakes": self.wakes}


def _count(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CounterError(f"the demo counters' {key} is not a count")
    return value


def _parse(data: Mapping[str, Any] | None, today: str) -> Day:
    if data is None or data.get("day") != today:
        return Day(today)
    by_client = data.get("visits_by_client", {})
    if not isinstance(by_client, dict):
        raise CounterError("the demo counters' visits_by_client is not an object")
    return Day(
        today, visits=_count(data, "visits"),
        visits_by_client=MappingProxyType({str(k): _count(by_client, k) for k in by_client}),
        calls=_count(data, "calls"), tokens=_count(data, "tokens"), wakes=_count(data, "wakes"),
    )


@dataclass(frozen=True)
class Hold:
    """An ask's claim on the limits until it is settled or released."""

    key: int
    day: str
    visit: str
    calls: int
    tokens: int


class Limiter:
    def __init__(self, store: CounterStore, limits: Limits = Limits()) -> None:
        self._store = store
        self.limits = limits
        self._lock = threading.Lock()
        self._day: Day | None = None
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._asks: OrderedDict[str, int] = OrderedDict()
        self._inflight: dict[int, Hold] = {}
        self._keys = count(1)

    def request(self, client: str, now: datetime) -> None:
        """One request from ``client``: refused past the per-minute request cap."""
        with self._lock:
            hits = self._window(f"request:{client}", 1, self.limits.requests_per_client_per_minute, now)
            hits.append(now.timestamp())

    def visit(self, client: str, now: datetime) -> None:
        """A new visit from ``client``: refused past the per-client or the global daily cap."""
        with self._lock:
            day = self._today(now)
            if day.visits >= self.limits.visits_per_day:
                raise Refused(429, "resting_today")
            had = day.visits_by_client.get(client, 0)
            if had >= self.limits.visits_per_client_per_day:
                raise Refused(429, "visits_spent")
            by_client = MappingProxyType({**day.visits_by_client, client: had + 1})
            self._commit(replace(day, visits=day.visits + 1, visits_by_client=by_client))

    def admit(self, client: str, visit: str, calls: int, now: datetime) -> Hold:
        """Hold ``calls`` model calls at their worst case, or refuse. Every check and the hold are
        one locked step, so concurrent asks can't pass a cap together."""
        limits = self.limits
        worst = calls * limits.tokens_per_call
        with self._lock:
            day = self._today(now)
            if day.calls + calls > limits.calls_per_day or day.tokens + worst > limits.tokens_per_day:
                raise Refused(429, "resting_today")
            if self._asks.get(visit, 0) >= limits.asks_per_visit:
                raise Refused(429, "visit_spent")
            hits = self._window(f"calls:{client}", calls, limits.calls_per_client_per_minute, now)
            if sum(hold.calls for hold in self._inflight.values()) + calls > limits.max_inflight_calls:
                raise Refused(503, "busy")
            self._commit(replace(day, calls=day.calls + calls, tokens=day.tokens + worst))
            hits.extend([now.timestamp()] * calls)
            self._remember(self._asks, visit, self._asks.get(visit, 0) + 1)
            hold = Hold(next(self._keys), day.day, visit, calls, worst)
            self._inflight[hold.key] = hold
            return hold

    def settle(self, hold: Hold, used_tokens: int, now: datetime) -> None:
        """The calls ran and used ``used_tokens`` in all: that replaces the worst case."""
        with self._lock:
            if self._inflight.pop(hold.key, None) is not None:
                self._refund(hold, 0, hold.tokens - used_tokens, now)

    def release(self, hold: Hold, now: datetime) -> None:
        """No call ran: the hold, and the visit's question, go back."""
        with self._lock:
            if self._inflight.pop(hold.key, None) is not None:
                self._refund(hold, hold.calls, hold.tokens, now)
                if self._asks.get(hold.visit, 0) > 0:
                    self._asks[hold.visit] -= 1

    def claim_wake(self, now: datetime) -> bool:
        """Count one wake on demand, or False when the day's wakes are spent."""
        with self._lock:
            day = self._today(now)
            if day.wakes >= self.limits.wakes_per_day:
                return False
            self._commit(replace(day, wakes=day.wakes + 1))
            return True

    # ── under the lock ────────────────────────────────────────────────────────

    def _today(self, now: datetime) -> Day:
        today = now.astimezone(timezone.utc).date().isoformat()
        if self._day is not None and self._day.day == today:
            return self._day
        try:
            day = _parse(self._store.load(), today)
        except CounterError as error:
            logger.error("demo counters unreadable, the demo is closed until they load: %s", error)
            raise Refused(503, "resting") from None
        self._day = day
        return day

    def _commit(self, day: Day) -> None:
        try:
            self._store.save(day.dump())
        except CounterError as error:
            logger.error("demo counters not saved, the change is refused: %s", error)
            raise Refused(503, "resting") from None
        self._day = day

    def _refund(self, hold: Hold, calls: int, tokens: int, now: datetime) -> None:
        try:
            day = self._today(now)
            if day.day == hold.day:  # a hold from an earlier day went with that day
                self._commit(replace(day, calls=max(0, day.calls - calls), tokens=max(0, day.tokens - tokens)))
        except Refused:
            logger.warning("a demo refund wasn't saved; the day stays charged at the worst case")

    def _window(self, key: str, adding: int, cap: int, now: datetime) -> deque[float]:
        hits = self._hits.get(key)
        if hits is None:
            hits = deque()
        cutoff = now.timestamp() - WINDOW_S
        while hits and hits[0] <= cutoff:
            hits.popleft()
        self._remember(self._hits, key, hits)
        if len(hits) + adding > cap:
            raise Refused(429, "slow_down")
        return hits

    @staticmethod
    def _remember(table: OrderedDict[str, Any], key: str, value: Any) -> None:
        table[key] = value
        table.move_to_end(key)
        while len(table) > MAX_TRACKED:
            table.popitem(last=False)
