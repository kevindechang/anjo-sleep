"""The judge cohort (Meta hackathon judging, Nov 2026): fresh accounts that ``POST
/api/sleep/judge/start`` makes for whoever holds the judge code, on the sleep host only.

A judge account is tagged here, in ``<data dir>/sleep/judges/accounts.json`` (ids and times only), so
the ops task can find and delete every one after the event (``judge_ids``) and the caps below can
count them. Nothing else marks it: every other account stays on its own path.

**What being a judge turns on, and nothing more:**

- the ``sleep`` flag (``flags.sleep_enabled_for``), under its kill switch ``ANJO_SLEEP_ENABLED``;
- the caller's own MIND state (``GET /api/sleep/me/mind``), with an empty held-out filter: a judge
  has no persona file and no sealed tests, and a caller only ever reads their own account;
- nights on a real account (``night_allowed``): the night command's ``--allow-real`` equivalent,
  scoped to this cohort and nothing else.

**What the night worker imports** (``POST /api/sleep/me/sleep-now`` and the night job):

- ``is_judge(user_id)`` and ``night_allowed(user_id)``;
- ``claim_night(user_id)`` before a night starts (raises ``CapReached``: 429 at the per-judge cap,
  ``ANJO_SLEEP_JUDGE_NIGHTS_PER_DAY``, default 3, or the cohort's, ``ANJO_SLEEP_JUDGE_NIGHTS_TOTAL_PER_DAY``,
  default 60; 503 when the counters can't be read), and ``release_night(user_id)`` when the claimed
  night never started;
- ``save_before_after(user_id, question=..., before=..., after=...)``: the wake's before/after pair,
  stored encrypted in the judge's own sleep folder (erased with it), read by ``GET /api/sleep/me/wake``;
- ``live_mind.paused(user_id)`` (in ``live_mind``): hold the per-turn MIND pass off while a night
  writes the same workspace log.

The counters live in ``<data dir>/sleep/judges/counters.json`` (one UTC day; a new day starts at 0),
written before a claim counts, so a restart never hands out a fresh day.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.limits import CounterError, JsonCounterStore

logger = logging.getLogger(__name__)

CODE_ENV: Final = "ANJO_SLEEP_JUDGE_CODE"
NIGHTS_PER_DAY_ENV: Final = "ANJO_SLEEP_JUDGE_NIGHTS_PER_DAY"
NIGHTS_TOTAL_ENV: Final = "ANJO_SLEEP_JUDGE_NIGHTS_TOTAL_PER_DAY"
MAX_ACCOUNTS_ENV: Final = "ANJO_SLEEP_JUDGE_MAX_ACCOUNTS"
LIVE_PER_DAY_ENV: Final = "ANJO_SLEEP_JUDGE_LIVE_PER_DAY"
LIVE_TOKENS_ENV: Final = "ANJO_SLEEP_JUDGE_LIVE_TOKENS_PER_DAY"
START_PER_IP_ENV: Final = "ANJO_SLEEP_JUDGE_START_PER_IP"
PREWAKE_PER_IP_ENV: Final = "ANJO_SLEEP_JUDGE_PREWAKE_PER_IP"
PREWAKE_PER_DAY_ENV: Final = "ANJO_SLEEP_JUDGE_PREWAKE_PER_DAY"
DEFAULTS: Final[Mapping[str, int]] = {
    NIGHTS_PER_DAY_ENV: 3, NIGHTS_TOTAL_ENV: 60, MAX_ACCOUNTS_ENV: 300,
    LIVE_PER_DAY_ENV: 8,  # MIND's per-turn passes a judge gets a day (each one Spark Open call or two)
    LIVE_TOKENS_ENV: 12_000,  # their thinker tokens a day, apart from the night's 20k (live_mind.live_budget)
    START_PER_IP_ENV: 20,  # judge/start attempts, right code or wrong, from one address a day
    PREWAKE_PER_IP_ENV: 6, PREWAKE_PER_DAY_ENV: 60,  # judge/prewake calls that may start a wake
}
JUDGES_DIR: Final = "judges"
ACCOUNTS_FILE: Final = "accounts.json"
COUNTERS_FILE: Final = "counters.json"
BEFORE_AFTER_FILE: Final = "judge_before_after.json"  # in the judge's own sleep folder
TAG: Final = "judge"


class CapReached(Exception):
    """A claim the cohort's caps refuse. ``status`` is the HTTP status, ``reason`` a short code."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def setting(name: str, environ: Mapping[str, str] | None = None) -> int:
    """A whole-number setting from the environment; a missing or malformed one is its default."""
    raw = (os.environ if environ is None else environ).get(name, "").strip()
    if raw.isdigit():
        return int(raw)
    if raw:
        logger.warning("%s is not a whole number; using %d", name, DEFAULTS[name])
    return DEFAULTS[name]


def judges_dir() -> Path:
    """``<data dir>/sleep/judges``, read when called (tests move the data dir)."""
    from anjo.core import paths

    return Path(paths.DATA_ROOT) / "sleep" / JUDGES_DIR


def _today(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()


# ── The registry ──────────────────────────────────────────────────────────────

_lock = threading.Lock()
_cache: dict[str, Any] = {"key": None, "ids": {}}


def _accounts_path() -> Path:
    return judges_dir() / ACCOUNTS_FILE


def _read_accounts(path: Path) -> dict[str, str]:
    """Each judge id and when it joined; empty when there is no file. A file that can't be read
    names nobody (fail closed: no account gains the cohort's access by an unreadable file)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        logger.error("the judge registry can't be read (%s); it names nobody", type(error).__name__)
        return {}
    rows = data.get("accounts") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    return {str(row["user_id"]): str(row.get("joined_at", "")) for row in rows
            if isinstance(row, dict) and isinstance(row.get("user_id"), str) and row.get("tag") == TAG}


def _accounts() -> dict[str, str]:
    """The registry, kept until the file changes (by modification time and size): ``is_judge`` runs
    on every flag read."""
    path = _accounts_path()
    try:
        stat = path.stat()
        key: tuple[str, int, int] | None = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    with _lock:
        if key is not None and _cache["key"] == key:
            return dict(_cache["ids"])
        ids = _read_accounts(path) if key is not None else {}
        _cache["key"], _cache["ids"] = key, ids
        return dict(ids)


def is_judge(user_id: str) -> bool:
    return bool(user_id) and user_id in _accounts()


def judge_ids() -> tuple[str, ...]:
    """Every judge account, for the cleanup after the event."""
    return tuple(sorted(_accounts()))


def joined_at(user_id: str) -> datetime | None:
    raw = _accounts().get(user_id)
    try:
        return datetime.fromisoformat(raw) if raw else None
    except ValueError:
        return None


def add_judge(user_id: str, now: datetime | None = None) -> None:
    """Tag ``user_id`` as a judge: written beside the file and renamed over it, so it is never torn.
    Raises ``CapReached`` (429) past ``ANJO_SLEEP_JUDGE_MAX_ACCOUNTS``."""
    path = _accounts_path()
    with _lock:
        current = _read_accounts(path)
        if user_id in current:
            return
        if len(current) >= setting(MAX_ACCOUNTS_ENV):
            raise CapReached(429, "judges_full")
        joined = (now or datetime.now(timezone.utc)).isoformat()
        rows = [{"user_id": uid, "joined_at": at, "tag": TAG} for uid, at in {**current, user_id: joined}.items()]
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.part")
        part.write_text(json.dumps({"accounts": rows}, sort_keys=True), encoding="utf-8")
        os.replace(part, path)
        _cache["key"] = None


def night_allowed(user_id: str) -> bool:
    """The ``--allow-real`` equivalent for the judge cohort only: True for a judge on the sleep host."""
    from anjo.sleep.app.flags import sleep_host

    return sleep_host() and is_judge(user_id)


# ── The day's counters ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Check:
    """One count a claim takes: ``counter`` (with ``member``, a per-judge or per-client count), at
    most ``cap`` a day; ``reason`` is what a refusal says."""

    counter: str
    member: str | None
    cap: int
    reason: str

    @property
    def key(self) -> str:
        return self.counter if self.member is None else f"{self.counter}:{self.member}"


class DailyCounters:
    """The cohort's counts for one UTC day, saved before they count (``demo.limits``' store)."""

    def __init__(self, path: Path) -> None:
        self._store = JsonCounterStore(path)
        self._lock = threading.Lock()

    def _load(self, today: str) -> dict[str, int]:
        data = self._store.load()
        if data is None or data.get("day") != today:
            return {}
        counts = data.get("counts")
        if not isinstance(counts, dict):
            raise CounterError("the judge counters are not an object")
        return {str(k): int(v) for k, v in counts.items() if isinstance(v, int) and not isinstance(v, bool)}

    def claim(self, now: datetime | None, *checks: Check) -> None:
        """Every check's count goes up by one, or none does: ``CapReached`` names the first full one."""
        today = _today(now)
        with self._lock:
            try:
                counts = self._load(today)
                for check in checks:
                    if counts.get(check.key, 0) >= check.cap:
                        raise CapReached(429, check.reason)
                self._store.save({"day": today, "counts": {**counts, **{c.key: counts.get(c.key, 0) + 1 for c in checks}}})
            except CounterError as error:
                logger.error("the judge counters can't be used: %s", error)
                raise CapReached(503, "counters_unreadable") from None

    def release(self, now: datetime | None, *keys: str) -> None:
        """Give back one of each count (a claim whose work never started). Best effort."""
        today = _today(now)
        with self._lock:
            try:
                counts = self._load(today)
                self._store.save({"day": today, "counts": {**counts, **{k: max(0, counts.get(k, 0) - 1) for k in keys}}})
            except CounterError as error:
                logger.warning("a judge count wasn't given back: %s", error)

    def count(self, now: datetime | None, key: str) -> int:
        with self._lock:
            try:
                return self._load(_today(now)).get(key, 0)
            except CounterError:
                return 0


_counters: list[DailyCounters] = []


def counters() -> DailyCounters:
    path = judges_dir() / COUNTERS_FILE
    with _lock:
        if not _counters or _counters[0]._store._path != path:
            _counters[:] = [DailyCounters(path)]
        return _counters[0]


def _night_checks(user_id: str) -> tuple[Check, Check]:
    return (Check("nights", None, setting(NIGHTS_TOTAL_ENV), "nights_spent_today"),
            Check("nights", user_id, setting(NIGHTS_PER_DAY_ENV), "your_nights_spent_today"))


def claim_night(user_id: str, now: datetime | None = None) -> None:
    """One of today's nights for ``user_id``, or ``CapReached`` (429 at a cap, 503 unreadable).
    Refused (403) for anyone outside the cohort."""
    if not night_allowed(user_id):
        raise CapReached(403, "not_a_judge")
    counters().claim(now, *_night_checks(user_id))


def release_night(user_id: str, now: datetime | None = None) -> None:
    """Give back a claimed night that never started."""
    counters().release(now, *(check.key for check in _night_checks(user_id)))


def claim_live(user_id: str, now: datetime | None = None) -> None:
    """One of today's per-turn MIND passes for ``user_id`` (``live_mind``), or ``CapReached``."""
    counters().claim(now, Check("live", user_id, setting(LIVE_PER_DAY_ENV), "live_spent_today"))


# ── The wake's before/after pair (written by the night, read by me/wake) ──────


def _before_after_path(user_id: str) -> Path:
    from anjo.sleep.app.storage import sleep_dir

    path: Path = sleep_dir(user_id) / BEFORE_AFTER_FILE
    return path


def save_before_after(user_id: str, *, question: str, before: str, after: str, night: str | None = None,
                      before_model: str | None = None, after_model: str | None = None,
                      promoted: bool | None = None, question_source: str | None = None) -> None:
    """The same question answered before the night (``before``, by ``before_model``) and by tonight's
    adapter (``after``, ``after_model``), whether the gate ``promoted`` it and where the question came
    from (``question_source``: ``personal`` or ``gate_everyday``), encrypted at rest like
    every sleep file, under the account's writer guard."""
    from anjo.sleep.app.storage import CODEC, writer_guard

    record = {"question": question, "before": before, "after": after, "night": night,
              "before_model": before_model, "after_model": after_model, "promoted": promoted,
              "question_source": question_source,
              "at": datetime.now(timezone.utc).isoformat()}
    path = _before_after_path(user_id)
    with writer_guard(user_id)():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.part")
        part.write_text(CODEC.encode(json.dumps(record, sort_keys=True)), encoding="utf-8")
        os.replace(part, path)


def load_before_after(user_id: str) -> dict[str, Any] | None:
    """The pair the night saved last, or None (none yet, or it can't be read)."""
    from anjo.sleep.app.storage import CODEC

    path = _before_after_path(user_id)
    try:
        data = json.loads(CODEC.decode(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as error:
        logger.error("a judge's before/after pair can't be read: %s", type(error).__name__)
        return None
    if not isinstance(data, dict) or not all(isinstance(data.get(k), str) for k in ("question", "before", "after")):
        return None
    labels = {k: data[k] for k in ("before_model", "after_model", "question_source") if isinstance(data.get(k), str)}
    promoted = {"promoted": data["promoted"]} if isinstance(data.get("promoted"), bool) else {}
    return {"question": data["question"], "before": data["before"], "after": data["after"], "night": data.get("night"),
            **labels, **promoted}
