"""The spend ledger every paid Anjo Sleep job goes through (spec §12.1, Task 4).

A job holds its worst case before it starts and settles what it really cost when it ends. The
hold is a ``Bound``: the job's hard timeout at the most its GPU or API can cost per hour, never a
caller's hopeful estimate. ``reserve`` debits every pool it names at once, all or none, and
refuses past any cap, so a judge's night debits its provider's pool and ``judge_usd`` together.
Pools named in one call share a unit: dollars, or Colab compute units.

**Job keys.** A job's name is its key. Reserving a key that already holds is that same hold, so a
job that crashed and is run again lands in its first hold rather than taking a second. Its
settlement is the whole cost under the key, every attempt included. A hold counts at its full
bound until it is settled or released, however long ago its hard timeout passed: a hold is never
freed by time. ``release`` frees a hold whose job spent nothing.

**Overages.** A settlement above its hold is recorded in full, since the money is spent, and the
pools it named then refuse every reserve until the coordinator clears it (``clear``). The cost
still counts after the clearing.

The caps are §12.1's. Baseten's $100 credit is two pools, and Sleep has priority: when
``baseten_sleep_usd`` runs short it may take MIND's unspent balance, so the two together never
pass $100 and MIND never passes its own $40.

The ledger is an append-only JSON Lines file in the data dir (``anjo/sleep/evidence/jsonl.py``,
in plain text: it holds no person's words), one event per line. Every reserve, settle, release
and clear holds an exclusive file lock, so the server and a night process can share it. Every
event is checked when the ledger is read, and one that fails refuses the whole ledger; a torn last
line from a killed write is left out and cut off before the next append. Nothing here reads the
wall clock except to stamp an event.
"""
from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, get_args

from anjo.sleep.evidence.jsonl import PLAIN, LogError, append_records, read_records
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

# The time caps every paid job runs under (§12.1). A GPU runner passes these as hard timeouts,
# and a job's Bound is its rate over the same timeout.
TRAINING_TIMEOUT_S: Final = 2 * 3600
EVAL_TIMEOUT_S: Final = 3600
SERVING_IDLE_S: Final = SCALEDOWN_S  # serving scales down after 10 minutes idle
SERVING_WINDOW_S: Final = SERVE_TIMEOUT_S  # one serving container lasts at most 3 hours

LEDGER_NAME: Final = "budget.jsonl"
DATA_DIR_ENV: Final = "ANJO_DATA_DIR"
_EPSILON: Final = 1e-9
_HOUR_S: Final = 3600.0
_FIELDS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "reserve": frozenset({"op", "job", "pools", "amount", "rate_per_hour", "timeout_s", "at"}),
    "settle": frozenset({"op", "job", "amount", "at"}),
    "release": frozenset({"op", "job", "at"}),
    "clear": frozenset({"op", "job", "at"}),
})


class BudgetError(ValueError):
    """A reserve, settle, release or clear the ledger can't take (an unknown pool, mixed units, a
    bad amount, a key held with another bound, a hold already ended), or an unreadable ledger."""


def _amount(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise BudgetError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise BudgetError(f"{label} must be {'positive' if positive else 'zero or more'}")
    return float(value)


def _pools(pools: object) -> tuple[Pool, ...]:
    if not isinstance(pools, tuple | list) or not pools:
        raise BudgetError("a reservation names at least one pool")
    unknown = [pool for pool in pools if pool not in CAPS]
    if unknown:
        raise BudgetError(f"unknown pool {unknown[0]!r}")
    if len(set(pools)) != len(pools):
        raise BudgetError("a reservation names each pool once")
    if len({UNITS[pool] for pool in pools}) != 1:
        raise BudgetError("pools reserved together share a unit")
    return tuple(pool for pool in POOLS if pool in pools)


def _job(job: object) -> str:
    if not isinstance(job, str) or not job.strip():
        raise BudgetError("a reservation names its job")
    return job


@dataclass(frozen=True)
class Bound:
    """A job's worst case: its hard timeout (the one its runner enforces) at the most its GPU or
    API can cost per hour, in the pool's unit."""

    rate_per_hour: float
    timeout_s: float

    def __post_init__(self) -> None:
        _amount(self.rate_per_hour, "rate_per_hour", positive=True)
        _amount(self.timeout_s, "timeout_s", positive=True)

    @property
    def amount(self) -> float:
        return self.rate_per_hour * self.timeout_s / _HOUR_S


@dataclass(frozen=True)
class Reservation:
    """One job's hold on its pools, under its key ``job``. ``settle`` records what it really
    cost, once; ``release`` frees it when it spent nothing."""

    job: str
    pools: tuple[Pool, ...]
    bound: Bound
    ledger: Ledger = field(repr=False, compare=False)

    def settle(self, actual: float) -> None:
        self.ledger.settle(self, actual)

    def release(self) -> None:
        self.ledger.release(self.job)


@dataclass(frozen=True)
class _Hold:
    pools: tuple[Pool, ...]
    bound: Bound
    spent: float | None = None
    released: bool = False
    cleared: bool = False

    @property
    def open(self) -> bool:
        return self.spent is None and not self.released

    @property
    def counted(self) -> float:
        if self.released:
            return 0.0
        return self.bound.amount if self.spent is None else self.spent

    @property
    def overage(self) -> bool:
        """Settled above its hold and not yet cleared."""
        return self.spent is not None and self.spent > self.bound.amount + _EPSILON and not self.cleared


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Ledger:
    """The spend ledger at ``path``: an event log folded on every read."""

    def __init__(
        self, path: Path, *, caps: Mapping[Pool, float] = CAPS,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._path = path
        self._caps = dict(caps)
        self._clock = clock

    @property
    def path(self) -> Path:
        return self._path

    def reserve(self, job: str, pools: tuple[Pool, ...], bound: Bound) -> Reservation | None:
        """Hold ``bound`` on every pool in ``pools`` under the key ``job``, or nothing when any
        pool would pass its cap or waits on a cleared overage. A key that already holds returns
        that same hold."""
        key, checked = _job(job), _pools(pools)
        if not isinstance(bound, Bound):
            raise BudgetError("a reservation holds a Bound: the rate over the hard timeout")
        with self._locked():
            holds = self._holds()
            if key in holds:
                return self._same(key, holds[key], checked, bound)
            used, blocked = _used(holds), _blocked(holds)
            if any(pool in blocked or bound.amount > self._room(pool, used) + _EPSILON for pool in checked):
                return None
            self._append({
                "op": "reserve", "job": key, "pools": list(checked), "amount": bound.amount,
                "rate_per_hour": bound.rate_per_hour, "timeout_s": bound.timeout_s,
            })
        return Reservation(key, checked, bound, self)

    def _same(self, key: str, hold: _Hold, pools: tuple[Pool, ...], bound: Bound) -> Reservation:
        if not hold.open:
            raise BudgetError(f"job {key!r} has ended; a new run takes a new key")
        if hold.pools != pools or hold.bound != bound:
            raise BudgetError(f"job {key!r} is held with other pools or another bound")
        return Reservation(key, pools, bound, self)

    def settle(self, reservation: Reservation, actual: float) -> None:
        """Replace the hold with what the job really cost, every attempt under its key included.
        A cost above the hold is recorded in full, and its pools then wait for ``clear``."""
        amount = _amount(actual, "actual")
        with self._locked():
            self._open(reservation.job)
            self._append({"op": "settle", "job": reservation.job, "amount": amount})

    def release(self, job: str) -> None:
        """Free an open hold whose job spent nothing (it never started)."""
        with self._locked():
            self._open(job)
            self._append({"op": "release", "job": job})

    def clear(self, job: str) -> None:
        """The coordinator's clearing of a settled overage: its pools take reserves again."""
        with self._locked():
            hold = self._holds().get(job)
            if hold is None or not hold.overage:
                raise BudgetError(f"job {job!r} has no overage to clear")
            self._append({"op": "clear", "job": job})

    def open_holds(self) -> tuple[Reservation, ...]:
        """Every hold not yet settled or released, in ledger order."""
        return tuple(Reservation(job, hold.pools, hold.bound, self) for job, hold in self._holds().items() if hold.open)

    def overages(self) -> tuple[str, ...]:
        """The jobs whose overage still blocks their pools, in ledger order."""
        return tuple(job for job, hold in self._holds().items() if hold.overage)

    def used(self, pool: Pool) -> float:
        """Settled costs plus open holds on ``pool``."""
        _pools((pool,))
        return _used(self._holds())[pool]

    def remaining(self, pool: Pool) -> float:
        """What ``pool`` may still take, Sleep's claim on MIND's Baseten balance included; none
        while an overage on it waits to be cleared."""
        _pools((pool,))
        holds = self._holds()
        if pool in _blocked(holds):
            return 0.0
        return max(0.0, self._room(pool, _used(holds)))

    def _open(self, job: str) -> _Hold:
        hold = self._holds().get(job)
        if hold is None:
            raise BudgetError(f"this ledger has no hold for job {job!r}")
        if hold.released:
            raise BudgetError(f"job {job!r} was released")
        if hold.spent is not None:
            raise BudgetError(f"job {job!r} is already settled; a hold is settled once")
        return hold

    def _room(self, pool: Pool, used: Mapping[Pool, float]) -> float:
        baseten = used["baseten_sleep_usd"] + used["baseten_mind_usd"]
        total = self._caps["baseten_sleep_usd"] + self._caps["baseten_mind_usd"]
        if pool == "baseten_sleep_usd":
            return total - baseten
        if pool == "baseten_mind_usd":
            return min(self._caps[pool] - used[pool], total - baseten)
        return self._caps[pool] - used[pool]

    def _holds(self) -> dict[str, _Hold]:
        try:
            events = read_records(self._path, PLAIN)
        except LogError as error:
            raise BudgetError(str(error)) from None
        return _fold(events, self._path.name)

    def _append(self, event: Mapping[str, object]) -> None:
        append_records(self._path, PLAIN, [{**event, "at": self._clock().isoformat()}])

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path.with_name(self._path.name + ".lock"), "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _used(holds: Mapping[str, _Hold]) -> dict[Pool, float]:
    used: dict[Pool, float] = dict.fromkeys(POOLS, 0.0)
    for hold in holds.values():
        for pool in hold.pools:
            used[pool] += hold.counted
    return used


def _blocked(holds: Mapping[str, _Hold]) -> frozenset[Pool]:
    return frozenset(pool for hold in holds.values() if hold.overage for pool in hold.pools)


def _fold(events: list[dict[str, Any]], name: str) -> dict[str, _Hold]:
    """Each job's hold, after checking every event against the ones before it."""
    holds: dict[str, _Hold] = {}
    for number, event in enumerate(events, 1):
        try:
            job, hold = _apply(event, holds)
        except BudgetError as error:
            raise BudgetError(f"{name} line {number}: {error}") from None
        holds[job] = hold
    return holds


def _apply(event: Mapping[str, Any], holds: Mapping[str, _Hold]) -> tuple[str, _Hold]:
    op = event.get("op")
    if not isinstance(op, str) or op not in _FIELDS or set(event) != _FIELDS[op]:
        raise BudgetError(f"an event {op!r} carries exactly its own fields")
    job = _job(event["job"])
    _stamp(event["at"])
    if op == "reserve":
        if job in holds:
            raise BudgetError(f"job {job!r} is reserved twice")
        bound = Bound(event["rate_per_hour"], event["timeout_s"])
        amount = _amount(event["amount"], "amount", positive=True)
        if not math.isclose(amount, bound.amount, rel_tol=1e-9):
            raise BudgetError("a hold is its rate over its hard timeout")
        return job, _Hold(_pools(event["pools"]), bound)
    hold = holds.get(job)
    if op == "clear":
        if hold is None or not hold.overage:
            raise BudgetError(f"job {job!r} has no overage to clear")
        return job, replace(hold, cleared=True)
    if hold is None or not hold.open:
        raise BudgetError(f"a {op} needs job {job!r} to hold")
    if op == "settle":
        return job, replace(hold, spent=_amount(event["amount"], "amount"))
    return job, replace(hold, released=True)


def _stamp(value: object) -> None:
    try:
        stamped = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        stamped = None
    if stamped is None or stamped.tzinfo is None:
        raise BudgetError("an event is stamped with an aware ISO time")


def default_ledger(environ: Mapping[str, str] | None = None) -> Ledger:
    """The ledger in the data dir: ``$ANJO_DATA_DIR/sleep/budget.jsonl``."""
    env = os.environ if environ is None else environ
    root = env.get(DATA_DIR_ENV, "").strip()
    if not root:
        raise BudgetError(f"{DATA_DIR_ENV} names the data dir the ledger lives in")
    return Ledger(Path(root) / "sleep" / LEDGER_NAME)


def reserve(job: str, pools: tuple[Pool, ...], bound: Bound) -> Reservation | None:
    """``Ledger.reserve`` on the data dir's ledger."""
    return default_ledger().reserve(job, pools, bound)


def release(job: str) -> None:
    """``Ledger.release`` on the data dir's ledger."""
    default_ledger().release(job)
