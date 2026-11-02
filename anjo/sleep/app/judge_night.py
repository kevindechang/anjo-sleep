"""Judges' "Sleep now" nights on the sleep host (2026-11-02): who may start one, the night records the
night view polls, the child process each night runs in, and the judges' budget.

**Who.** The judge cohort is the judge API's (sign-in, accounts, caps). It plugs in here with
``set_judge_ports(JudgePorts(is_judge, caps))``: ``is_judge(user_id)`` and
``caps.try_start_night(user_id)``, which checks the per-judge and global daily caps and counts the
night when it answers ``ok`` (else ``per_judge``, ``global`` or ``running``). Until it does, nobody is a
judge and both routes answer 404.

**Records.** One JSON file per night under ``<data>/sleep/judge_nights/`` (``JobStore``): the judge,
the status (``queued``, ``running``, ``done``, ``failed``), the stages (``judge_stages``), the note and
whether tonight's adapter is live, written by the night's child process as its steps go, under one
file lock, each write atomic.

**Dispatch.** ``pump`` runs on every start and every poll: a running night whose process is gone is
marked failed (its judges' hold settled at its full bound), and queued nights start while fewer than
``ANJO_SLEEP_JUDGE_CONCURRENCY`` (2) run. Each night is ``python -m anjo.sleep.night.judge``
(``judge_child.run_night``, by name) in its own session, with MIND's loop on (``ANJO_SLEEP_MIND_LOOP=1``) and Spark Open on the provider
``ANJO_SLEEP_JUDGE_SPARK`` names (passed on as the night's ``ANJO_SLEEP_SPARK``; unset: the host's own).

**Which night.** A judge's night reads the chat since their last night (the last week, the first
time), and is dated the day after their last judge night or promoted night when that is later than
today, so each night's evidence is new to it (``night/run.py``'s ``_since``).

**Budget.** Each night holds ``ANJO_SLEEP_JUDGE_NIGHT_USD`` ($10) on ``judge_usd`` under
``judge-night:<id>`` when it is admitted, so a queued night counts against the budget before its process
starts; the night's process takes that same hold and settles what it cost (the night's own cost, and
keeping the trainer warm). A new night is refused (429, a plain message) when the judges' spend plus
that hold would pass ``ANJO_SLEEP_JUDGE_BUDGET_USD`` ($100, ``judge_usd``'s cap, never above it), and a
night the caps refuse lets its hold go. At 70% and 90% of it one alert record each goes to
``<data>/sleep/judge_alerts.jsonl``.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from anjo.sleep.app.judge_stages import close, empty
from anjo.sleep.app.storage import SLEEP_DIR, checkpoint_registry, data_root
from anjo.sleep.budget import CAPS, LEDGER_NAME, Ledger, Reservation
from anjo.sleep.night.holds import usd

StartVerdict = Literal["ok", "per_judge", "global", "running"]
JOBS_DIR: Final = "judge_nights"
ALERTS_FILE: Final = "judge_alerts.jsonl"
POOL: Final = "judge_usd"
HOLD_PREFIX: Final = "judge-night:"  # no night kind's prefix: the night sweep never settles it
CONCURRENCY_ENV: Final = "ANJO_SLEEP_JUDGE_CONCURRENCY"
BUDGET_ENV: Final = "ANJO_SLEEP_JUDGE_BUDGET_USD"
NIGHT_USD_ENV: Final = "ANJO_SLEEP_JUDGE_NIGHT_USD"
SPARK_ENV: Final = "ANJO_SLEEP_JUDGE_SPARK"
SLEEP_SPARK_ENV: Final = "ANJO_SLEEP_SPARK"  # the night's own Spark Open provider setting
DEFAULT_CONCURRENCY: Final = 2
DEFAULT_NIGHT_USD: Final = 10.0
ALERT_AT: Final = (0.7, 0.9)
LOOKBACK: Final = timedelta(days=7)
ACTIVE: Final = ("queued", "running")
DIED: Final = "The night stopped unexpectedly."
_ID: Final = re.compile(r"jn-[0-9a-f]{16}")
ENTRY: Final = "anjo.sleep.app.judge_child:run_night"  # the night process's app side, named for the engine's entry
logger = logging.getLogger(__name__)


class JudgeSettingError(ValueError):
    """A judges' setting holds a value the host can't run."""


class JudgeCaps(Protocol):
    def try_start_night(self, user_id: str) -> StartVerdict:
        """Check the caps and, on ``ok``, count the night."""
        ...


@dataclass(frozen=True)
class JudgePorts:
    is_judge: Callable[[str], bool]
    caps: JudgeCaps


class _NoCaps:
    def try_start_night(self, user_id: str) -> StartVerdict:
        return "global"


NO_JUDGES: Final = JudgePorts(is_judge=lambda user_id: False, caps=_NoCaps())
_PORTS: list[JudgePorts] = [NO_JUDGES]


def set_judge_ports(ports: JudgePorts | None) -> None:
    """The judge cohort's two answers (the judge API sets them at startup); None: nobody is a judge."""
    _PORTS[:] = [ports or NO_JUDGES]


def judge_ports() -> JudgePorts:
    return _PORTS[0]


def _number(environ: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise JudgeSettingError(f"{name} is a number") from None
    if not low <= value <= high:
        raise JudgeSettingError(f"{name} is {low:g} to {high:g}")
    return value


def concurrency(environ: Mapping[str, str]) -> int:
    return int(_number(environ, CONCURRENCY_ENV, DEFAULT_CONCURRENCY, 1, 4))


def budget_usd(environ: Mapping[str, str]) -> float:
    return _number(environ, BUDGET_ENV, CAPS[POOL], 1.0, CAPS[POOL])


def night_usd(environ: Mapping[str, str]) -> float:
    return _number(environ, NIGHT_USD_ENV, DEFAULT_NIGHT_USD, 1.0, 50.0)


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The night process's environment: the host's, MIND's loop on, and Spark Open's provider set to the
    judges' setting when there is one (the night checks the value as it checks the host's own)."""
    spark = environ.get(SPARK_ENV, "").strip().casefold()
    return {**environ, "ANJO_SLEEP_MIND_LOOP": "1", **({SLEEP_SPARK_ENV: spark} if spark else {})}


def _iso(at: datetime) -> str:
    return at.astimezone(timezone.utc).isoformat()


def check_id(night_id: str) -> str:
    if not _ID.fullmatch(night_id):
        raise KeyError(night_id)
    return night_id


class JobStore:
    """The night records: one JSON file per night, writes under one lock, each atomic."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, night_id: str) -> Path:
        return self.root / f"{check_id(night_id)}.json"

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def get(self, night_id: str) -> dict[str, Any] | None:
        try:
            path = self._path(night_id)
        except KeyError:
            return None
        if not path.is_file():
            return None
        found: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return found

    def _write(self, job: Mapping[str, Any]) -> None:
        path = self._path(str(job["night_id"]))
        pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        pending.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
        os.replace(pending, path)

    def put(self, job: Mapping[str, Any]) -> None:
        with self.locked():
            self._write(job)

    def update(self, night_id: str, change: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        """Apply ``change`` to the night's record under the lock (it returns a new record)."""
        with self.locked():
            return self._update_held(night_id, change)

    def _update_held(self, night_id: str, change: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        job = self.get(night_id)
        if job is None:
            raise KeyError(night_id)
        new = change(job)
        self._write(new)
        return new

    def all(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        found = [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("jn-*.json")]
        return sorted(found, key=lambda job: str(job["created_at"]))

    def for_user(self, user_id: str) -> list[dict[str, Any]]:
        return [job for job in self.all() if job["user_id"] == user_id]


def job_store() -> JobStore:
    return JobStore(data_root() / SLEEP_DIR / JOBS_DIR)


def ledger() -> Ledger:
    return Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)


def plan(user_id: str, earlier: list[dict[str, Any]], promoted: date | None, now: datetime) -> dict[str, Any]:
    """The night's date, window and clock (the module's docstring: "Which night")."""
    last = earlier[-1] if earlier else None
    start = datetime.fromisoformat(last["as_of"]) if last else now - LOOKBACK
    nights = [now.date()]
    if last:
        nights.append(date.fromisoformat(last["night"]) + timedelta(days=1))
    if promoted is not None:
        nights.append(promoted + timedelta(days=1))
    return {"night": max(nights).isoformat(), "window": [_iso(start), _iso(now)], "as_of": _iso(now)}


def new_id() -> str:
    return f"jn-{secrets.token_hex(8)}"


def hold_key(night_id: str) -> str:
    return f"{HOLD_PREFIX}{night_id}"


def night_hold(book: Ledger, night_id: str, environ: Mapping[str, str]) -> Reservation | None:
    """The night's hold on ``judge_usd``: the one taken when it was admitted, else a new one (or
    ``None`` when the pool has no room for it)."""
    key = hold_key(night_id)
    held = next((hold for hold in book.open_holds() if hold.job == key), None)
    return held or book.reserve(key, (POOL,), usd(night_usd(environ)))


def new_job(user_id: str, store: JobStore, now: datetime, night_id: str | None = None) -> dict[str, Any]:
    """A queued night for ``user_id`` (the caller holds the store's lock)."""
    serving = checkpoint_registry().last_promoted(user_id)
    job = {
        "night_id": night_id or new_id(), "user_id": user_id, "status": "queued", "phase": 1,
        "stages": empty(), "note": None, "adapter_live": False, "created_at": _iso(now), "pid": None,
        "error": None, "cost_usd": None,
        **plan(user_id, store.for_user(user_id), serving.night if serving else None, now),
    }
    store._write(job)
    return job


# --- dispatch -------------------------------------------------------------------------------------

_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


def _spawn(night_id: str) -> int:
    log = job_store().root / f"{night_id}.log"
    with open(log, "ab") as out:
        child = subprocess.Popen(  # noqa: S603 - our own module, a checked id, no shell
            [sys.executable, "-m", "anjo.sleep.night.judge", ENTRY, check_id(night_id)],
            env=child_env(os.environ), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _CHILDREN[child.pid] = child
    return child.pid


def _alive(pid: int) -> bool:
    child = _CHILDREN.get(pid)
    if child is not None:
        return child.poll() is None  # reaps it once it ends
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _settle_died(book: Ledger, night_id: str) -> None:
    """A night whose process died settles its judges' hold at the full bound: what it spent is unknown."""
    for held in book.open_holds():
        if held.job == hold_key(night_id):
            held.settle(held.bound.amount)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def pump(store: JobStore, *, book: Callable[[], Ledger] | None = None, environ: Mapping[str, str] | None = None,
         spawn: Callable[[str], int] | None = None, alive: Callable[[int], bool] | None = None,
         now: Callable[[], datetime] = _utc_now) -> None:
    """Mark the nights whose process is gone failed, and start queued nights while there is room."""
    env = os.environ if environ is None else environ
    book, spawn, alive = book or ledger, spawn or _spawn, alive or _alive
    with store.locked():
        running = 0
        for job in store.all():
            if job["status"] == "running" and job.get("pid") and not alive(int(job["pid"])):
                at = _iso(now())
                store._update_held(job["night_id"], lambda j: {
                    **j, "status": "failed", "error": DIED, "ended_at": at, "stages": close(j["stages"], at, failed=True)})
                _settle_died(book(), job["night_id"])
            elif job["status"] == "running":
                running += 1
        for job in store.all():
            if job["status"] != "queued" or running >= concurrency(env):
                continue
            pid = spawn(job["night_id"])
            store._update_held(job["night_id"], lambda j: {**j, "status": "running", "pid": pid})
            running += 1


# --- starting a night -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Refused:
    """Why a night didn't start: the HTTP status, a plain message, and the running night's id on a 409
    (so the page can go back to watching it)."""

    status: int
    detail: str
    night_id: str | None = None


PER_JUDGE: Final = "You've used tonight's nights. Chat still works; try again tomorrow."
GLOBAL: Final = "Anjo has slept as much as she can today. Chat still works; try again tomorrow."
BUDGET: Final = "The judges' sleep budget is used up. Chat still works."
RUNNING: Final = "A night is already running."


def check_alerts(book: Ledger, path: Path, budget: float, now: datetime) -> list[float]:
    """Write an alert record for each of ``ALERT_AT`` the judges' spend has reached, once each."""
    used = book.used(POOL)
    seen = set()
    if path.is_file():
        seen = {json.loads(line)["threshold"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    new = [at for at in ALERT_AT if used >= at * budget and at not in seen]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as out:
            for at in new:
                out.write(json.dumps({"at": _iso(now), "threshold": at, "used_usd": round(used, 2),
                                      "budget_usd": budget}) + "\n")
                logger.warning("the judges' sleep spend reached %d%% of $%.0f", round(at * 100), budget)
    return new


def start_night(user_id: str, *, store: JobStore | None = None, book: Callable[[], Ledger] | None = None,
                environ: Mapping[str, str] | None = None, ports: JudgePorts | None = None,
                now: Callable[[], datetime] = _utc_now, spawn: Callable[[str], int] | None = None,
                alive: Callable[[int], bool] | None = None) -> str | Refused:
    """Queue ``user_id``'s night and start it when there is room: its id, or why not (409 or 429)."""
    env, jobs, judges = os.environ if environ is None else environ, store or job_store(), ports or judge_ports()
    book = book or ledger
    pump(jobs, book=book, environ=env, spawn=spawn, alive=alive, now=now)
    spend = book()
    check_alerts(spend, jobs.root.parent / ALERTS_FILE, budget_usd(env), now())
    with jobs.locked():
        active = [job["night_id"] for job in jobs.for_user(user_id) if job["status"] in ACTIVE]
        if active:
            return Refused(409, RUNNING, active[-1])
        night_id = new_id()
        held = night_hold(spend, night_id, env) if spend.used(POOL) + night_usd(env) <= budget_usd(env) else None
        if held is None:  # held at admission: queued nights count before their process starts
            return Refused(429, BUDGET)
        verdict = judges.caps.try_start_night(user_id)
        if verdict != "ok":
            held.release()
            if verdict == "running":
                return Refused(409, RUNNING)
            return Refused(429, PER_JUDGE if verdict == "per_judge" else GLOBAL)
        new_job(user_id, jobs, now(), night_id)
    pump(jobs, book=book, environ=env, spawn=spawn, alive=alive, now=now)
    return night_id


def night_view(job: Mapping[str, Any]) -> dict[str, Any]:
    """The night route's answer: the contract's shape, nothing of the record's own bookkeeping."""
    from anjo.sleep.app.judge_stages import as_list

    return {
        "status": job["status"], "phase": job["phase"], "stages": as_list(job["stages"]),
        **({"note": job["note"]} if job.get("note") else {}), "adapter_live": bool(job.get("adapter_live")),
    }

