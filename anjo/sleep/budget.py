"""The spend ledger every paid Anjo Sleep job goes through (spec §12.1, Task 4).

A job reserves its estimate before it starts and settles what it really cost when it ends.
``reserve`` debits every pool it names at once, all or none, and refuses past any cap, so a
judge's night debits its provider's pool and ``judge_usd`` together. Pools named in one call
share a unit: dollars, or Colab compute units.

The caps are §12.1's. Baseten's $100 credit is two pools, and Sleep has priority: when
``baseten_sleep_usd`` runs short it may take MIND's unspent balance, so the two together never
pass $100 and MIND never passes its own $40.

The ledger is an append-only JSON Lines file in the data dir, one event per line, and every
reserve and settle holds an exclusive file lock, so the server and a night process can share
it. Nothing here reads the wall clock except to stamp an event.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, get_args

from anjo.sleep.serving.config import SCALEDOWN_S, SERVE_TIMEOUT_S

Pool = Literal[
    "modal_usd", "colab_cu", "meta_usd", "baseten_sleep_usd", "baseten_mind_usd", "vertex_usd",
    "judge_usd",
]
POOLS: Final[tuple[Pool, ...]] = get_args(Pool)

CAPS: Final[Mapping[Pool, float]] = MappingProxyType({
    "modal_usd": 500.0,  # of which about $120 is warm hours for the live page in judging week
    "colab_cu": 300.0,  # training (Task 5) and MIND's synthetic runs (Tasks 17, 17b)
    "meta_usd": 140.0,  # $10 under the $150 credit, for billing lag
    "baseten_sleep_usd": 60.0,  # practice, the simulator, extraction and the brief
    "baseten_mind_usd": 40.0,  # Tasks 17 and 17b
    "vertex_usd": 50.0,  # the Gemini diagnostic judge
    "judge_usd": 100.0,  # judges' sleep-now nights, inside the totals above
})
UNITS: Final[Mapping[Pool, str]] = MappingProxyType(
    {pool: "cu" if pool == "colab_cu" else "usd" for pool in POOLS}
)
BASETEN_TOTAL: Final = CAPS["baseten_sleep_usd"] + CAPS["baseten_mind_usd"]

# The time caps every paid job runs under (§12.1). A GPU runner passes these as hard timeouts.
TRAINING_TIMEOUT_S: Final = 2 * 3600
EVAL_TIMEOUT_S: Final = 3600
SERVING_IDLE_S: Final = SCALEDOWN_S  # serving scales down after 10 minutes idle
SERVING_WINDOW_S: Final = SERVE_TIMEOUT_S  # one serving container lasts at most 3 hours

LEDGER_NAME: Final = "budget.jsonl"
DATA_DIR_ENV: Final = "ANJO_DATA_DIR"
_EPSILON: Final = 1e-9


class BudgetError(ValueError):
    """A reserve or settle the ledger can't take: an unknown pool, mixed units, a bad amount,
    a reservation settled twice, or an unreadable ledger."""


def _amount(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise BudgetError(f"{label} must be a finite number")
    if value < 0:
        raise BudgetError(f"{label} can't be negative")
    return float(value)


def _pools(pools: tuple[Pool, ...]) -> tuple[Pool, ...]:
    if not pools:
        raise BudgetError("a reservation names at least one pool")
    unknown = [pool for pool in pools if pool not in CAPS]
    if unknown:
        raise BudgetError(f"unknown pool {unknown[0]!r}")
    if len(set(pools)) != len(pools):
        raise BudgetError("a reservation names each pool once")
    if len({UNITS[pool] for pool in pools}) != 1:
        raise BudgetError("pools reserved together share a unit")
    return pools


@dataclass(frozen=True)
class Reservation:
    """One job's hold on its pools. ``settle`` records what it really cost, once."""

    id: str
    job: str
    pools: tuple[Pool, ...]
    est: float
    ledger: Ledger = field(repr=False, compare=False)

    def settle(self, actual: float) -> None:
        self.ledger.settle(self, actual)


class Ledger:
    """The spend ledger at ``path``: an event log folded on every read."""

    def __init__(self, path: Path, *, caps: Mapping[Pool, float] = CAPS) -> None:
        self._path = path
        self._caps = dict(caps)

    @property
    def path(self) -> Path:
        return self._path

    def reserve(self, job: str, pools: tuple[Pool, ...], est: float) -> Reservation | None:
        """Hold ``est`` on every pool in ``pools``, or nothing when any would pass its cap."""
        if not isinstance(job, str) or not job.strip():
            raise BudgetError("a reservation names its job")
        checked = _pools(pools)
        amount = _amount(est, "est")
        if amount == 0:
            raise BudgetError("a reservation holds a positive estimate")
        with self._locked():
            used = self._used(self._events())
            if any(amount > self._room(pool, used) + _EPSILON for pool in checked):
                return None
            reservation = Reservation(uuid.uuid4().hex, job, checked, amount, self)
            self._append({
                "op": "reserve", "id": reservation.id, "job": job, "pools": list(checked),
                "amount": amount, "at": _now(),
            })
        return reservation

    def settle(self, reservation: Reservation, actual: float) -> None:
        """Replace the reservation's estimate with what the job really cost. A cost above the
        estimate is still recorded: the money is spent, so the next reserve sees it."""
        amount = _amount(actual, "actual")
        with self._locked():
            state = _fold(self._events())
            if reservation.id not in state:
                raise BudgetError("this ledger holds no such reservation")
            if state[reservation.id].settled:
                raise BudgetError("a reservation is settled once")
            self._append({"op": "settle", "id": reservation.id, "amount": amount, "at": _now()})

    def used(self, pool: Pool) -> float:
        """Settled costs plus open estimates on ``pool``."""
        _pools((pool,))
        return self._used(self._events())[pool]

    def remaining(self, pool: Pool) -> float:
        """What ``pool`` may still take, Sleep's claim on MIND's Baseten balance included."""
        _pools((pool,))
        return max(0.0, self._room(pool, self._used(self._events())))

    def _room(self, pool: Pool, used: Mapping[Pool, float]) -> float:
        baseten = used["baseten_sleep_usd"] + used["baseten_mind_usd"]
        total = self._caps["baseten_sleep_usd"] + self._caps["baseten_mind_usd"]
        if pool == "baseten_sleep_usd":
            return total - baseten
        if pool == "baseten_mind_usd":
            return min(self._caps[pool] - used[pool], total - baseten)
        return self._caps[pool] - used[pool]

    def _used(self, events: list[dict[str, object]]) -> dict[Pool, float]:
        used: dict[Pool, float] = dict.fromkeys(POOLS, 0.0)
        for entry in _fold(events).values():
            for pool in entry.pools:
                used[pool] += entry.amount
        return used

    def _events(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        events: list[dict[str, object]] = []
        for number, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                raise BudgetError(f"{self._path.name} line {number} can't be read") from None
            if not isinstance(value, dict):
                raise BudgetError(f"{self._path.name} line {number} is not an event")
            events.append(value)
        return events

    def _append(self, event: Mapping[str, object]) -> None:
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path.with_name(self._path.name + ".lock"), "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass(frozen=True)
class _Entry:
    pools: tuple[Pool, ...]
    amount: float
    settled: bool


def _fold(events: list[dict[str, object]]) -> dict[str, _Entry]:
    """Each reservation's pools and its current amount: the settled cost, else the estimate."""
    entries: dict[str, _Entry] = {}
    for event in events:
        op, ident, amount = event.get("op"), event.get("id"), event.get("amount")
        if not isinstance(ident, str) or not isinstance(amount, int | float):
            raise BudgetError("a ledger event names its reservation and amount")
        if op == "reserve":
            raw = event.get("pools")
            if not isinstance(raw, list) or not all(pool in CAPS for pool in raw):
                raise BudgetError("a reserve event names known pools")
            pools = tuple(pool for pool in POOLS if pool in raw)
            entries[ident] = _Entry(pools, float(amount), False)
        elif op == "settle" and ident in entries and not entries[ident].settled:
            entries[ident] = _Entry(entries[ident].pools, float(amount), True)
        else:
            raise BudgetError(f"a ledger event {op!r} doesn't follow its reservation")
    return entries


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_ledger(environ: Mapping[str, str] | None = None) -> Ledger:
    """The ledger in the data dir: ``$ANJO_DATA_DIR/sleep/budget.jsonl``."""
    env = os.environ if environ is None else environ
    root = env.get(DATA_DIR_ENV, "").strip()
    if not root:
        raise BudgetError(f"{DATA_DIR_ENV} names the data dir the ledger lives in")
    return Ledger(Path(root) / "sleep" / LEDGER_NAME)


def reserve(job: str, pools: tuple[Pool, ...], est: float) -> Reservation | None:
    """``Ledger.reserve`` on the data dir's ledger."""
    return default_ledger().reserve(job, pools, est)
