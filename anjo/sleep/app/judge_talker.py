"""Glimmer for the judges: whether she is up, waking her when a judge arrives, keeping her warm
only while one is present, what that costs, and the route that picks Glimmer or Muse Spark per turn.

**Judges never wait on a cold start.** ``judge_route`` (the first route override on the sleep host)
gives a judge's turn Glimmer first only while she is known to be up, with Muse Spark behind her
(``spark_talker``); while she is cold or waking, or past the hard spend cap, Spark answers alone,
labelled by ``GET /api/sleep/me/talker``. A judge with a promoted adapter gets it, through the same
pin ``runtime.sleep_route`` builds. Without ``ANJO_META_API_KEY`` there is no Spark pin, and the
route falls through to ``sleep_route`` and the chat route's own fallback, as for everyone flagged.

**Up, found without waking her** (``JudgeTalker.up``): a keep-warm window someone holds
(``serving.wake``'s file), or our own mark: a wake or ping that answered keeps her up for the
backend's idle window less a minute (Modal's ``SCALEDOWN_S``, SageMaker's ``IDLE_MINUTES``). On Modal
she is never asked: any request starts a container. On SageMaker, ``status`` may ask with one ping
(kept ``PING_TTL_S``) while a judge is present: a stopped copy refuses at once.

**Warm while present.** Every judge call (``judge/start``, ``judge/prewake``, ``me/*``, a chat turn)
and the page's ``me/heartbeat`` (every 60 s while the tab shows) extends the cohort's lease to
``ANJO_SLEEP_JUDGE_IDLE_MINUTES`` (default 3) from now; ``judge/prewake`` holds it
``ANJO_SLEEP_JUDGE_PREWAKE_MINUTES`` (default 5) for the code entry. While the lease holds, a keeper
thread pings her every ``KEEP_WARM_PING_S`` and keeps the keep-warm window file open to the lease's
end, so the night's own GPU stop leaves her up for the judge. When the lease lapses (no heartbeat
from any judge), the keeper stops paying at once instead of idling through the backend's window:
the night's ``GpuStop`` with the lease as its quiet time (Modal: the serving containers stop; on
SageMaker the copy count goes to 0), which never cuts a turn in flight, another run's night or
someone else's keep-warm window.

**Waking** (``arrive``): when no wake is running, not more often than ``WAKE_DEBOUNCE_S``, at most
``ANJO_SLEEP_JUDGE_WAKES_PER_DAY`` (default 30) a day, and never with ``ANJO_SLEEP_JUDGE_WAKE=0``,
through ``serving.backend`` (Modal or SageMaker, whichever ``ANJO_SLEEP_BACKEND`` names).

**Spend alerts, not degradation.** Every second the cohort keeps her up or waking is charged at
``ANJO_SLEEP_JUDGE_GPU_USD_PER_HOUR`` (default: ``ANJO_SLEEP_PRICES``' serving rate, else 1.95), kept in
``<data dir>/sleep/judges/spend.json`` and debited to the host's budget ledger (``modal_usd``) when each
paid stretch ends. Crossing 70% and 90% of the judges' budget (``ANJO_SLEEP_JUDGE_BUDGET_USD``, default
19) or of the day's soft cap (``ANJO_SLEEP_JUDGE_SOFT_USD_PER_DAY``, default 6) writes one record to
``<data dir>/sleep/judges/alerts.jsonl`` and a warning line, for the host operator to top the budget up.
Only the hard cap (``ANJO_SLEEP_JUDGE_HARD_USD_PER_DAY``, default 12, about six hours warm; and
``ANJO_SLEEP_JUDGE_HARD_USD_TOTAL`` when set) changes anything a judge sees: no more wakes or warm time,
and Spark answers, labelled ``reason: budget``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from anjo.core.anjo_model.assignment import OVERRIDE_CAUSE, ROUTE_ANJO, ModelRoute, QualifiedPin
from anjo.sleep.app.judge_cohort import CapReached, Check, counters, is_judge, judges_dir
from anjo.sleep.app.spark_talker import LABEL as SPARK_LABEL
from anjo.sleep.app.spark_talker import spark_configured, spark_pin
from anjo.sleep.serving.backend import Serving
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.serving.config import SCALEDOWN_S, ServeConfigError
from anjo.sleep.serving.wake import KEEP_WARM_PING_S, WarmWindow, read_window

logger = logging.getLogger(__name__)

WAKE_ENV: Final = "ANJO_SLEEP_JUDGE_WAKE"
IDLE_MIN_ENV: Final = "ANJO_SLEEP_JUDGE_IDLE_MINUTES"
PREWAKE_MIN_ENV: Final = "ANJO_SLEEP_JUDGE_PREWAKE_MINUTES"
WAKE_ETA_ENV: Final = "ANJO_SLEEP_JUDGE_WAKE_ETA_S"
WAKES_ENV: Final = "ANJO_SLEEP_JUDGE_WAKES_PER_DAY"
RATE_ENV: Final = "ANJO_SLEEP_JUDGE_GPU_USD_PER_HOUR"
BUDGET_ENV: Final = "ANJO_SLEEP_JUDGE_BUDGET_USD"
SOFT_ENV: Final = "ANJO_SLEEP_JUDGE_SOFT_USD_PER_DAY"
HARD_ENV: Final = "ANJO_SLEEP_JUDGE_HARD_USD_PER_DAY"
HARD_TOTAL_ENV: Final = "ANJO_SLEEP_JUDGE_HARD_USD_TOTAL"
DEFAULT_RATE: Final = 1.95  # an L40S an hour on Modal, list price 2026-11-02
DEFAULT_ETA_S: Final[Mapping[str, float]] = {"modal": 240.0, "sagemaker": 900.0}  # until measured
ALERT_FRACTIONS: Final = (0.7, 0.9)
WAKE_DEBOUNCE_S: Final = 60.0
WARM_MARGIN_S: Final = 60.0
PING_TTL_S: Final = 20.0
TICK_S: Final = 30.0
SPEND_FILE: Final = "spend.json"
ALERTS_FILE: Final = "alerts.jsonl"
VIA_SPARK: Final = "muse-spark"  # what answers under SPARK_LABEL with ANJO_META_API_KEY set
VIA_FRONTIER: Final = "frontier"  # without it: the chat route's own frontier fallback
RETRY_STOP: Final = frozenset({"in_flight", "recent_turn"})  # a stop skipped for a turn: try again next tick
UNSEEN_STOP: Final = frozenset({"untracked", "unreadable", "no_data_dir"})  # skipped blind: she idles out

State = Literal["warm", "waking", "cold"]
StopResult = tuple[str, Mapping[str, Any]]


def _number(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default
    return value if math.isfinite(value) and value >= 0 else default


def _serve_rate() -> float | None:
    from anjo.sleep.app.demo_live import serve_rate

    rate: float | None = serve_rate()
    return rate


@dataclass(frozen=True)
class TalkerSettings:
    wake_on: bool = True
    idle_s: float = 180.0
    prewake_s: float = 300.0
    wake_eta_s: float = 240.0
    wakes_per_day: int = 30
    rate_per_hour: float = DEFAULT_RATE
    budget_usd: float = 19.0
    soft_usd_per_day: float = 6.0
    hard_usd_per_day: float = 12.0
    hard_usd_total: float | None = None

    @classmethod
    def from_env(cls, backend: str | None, environ: Mapping[str, str] | None = None) -> TalkerSettings:
        env = os.environ if environ is None else environ
        base = cls()
        rate = _number(env, RATE_ENV, None)
        return cls(
            wake_on=env.get(WAKE_ENV, "").strip() != "0",
            idle_s=60.0 * (_number(env, IDLE_MIN_ENV, base.idle_s / 60.0) or base.idle_s / 60.0),
            prewake_s=60.0 * (_number(env, PREWAKE_MIN_ENV, base.prewake_s / 60.0) or base.prewake_s / 60.0),
            wake_eta_s=_number(env, WAKE_ETA_ENV, DEFAULT_ETA_S.get(backend or "", base.wake_eta_s)) or base.wake_eta_s,
            wakes_per_day=int(_number(env, WAKES_ENV, float(base.wakes_per_day)) or 0),
            rate_per_hour=rate if rate is not None else (_serve_rate() or DEFAULT_RATE),
            budget_usd=_number(env, BUDGET_ENV, base.budget_usd) or 0.0,
            soft_usd_per_day=_number(env, SOFT_ENV, base.soft_usd_per_day) or 0.0,
            hard_usd_per_day=_number(env, HARD_ENV, base.hard_usd_per_day) or 0.0,
            hard_usd_total=_number(env, HARD_TOTAL_ENV, None),
        )


# ── Spend ─────────────────────────────────────────────────────────────────────


def _day(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).date().isoformat()


class SpendTracker:
    """The judges' GPU dollars: today's and in all, saved after each charge, with the alerts each
    crossing writes once. A file that can't be read starts from zero and says so in the log: the
    judges' experience comes first, and the alerts still fire from there."""

    def __init__(self, path: Path, alerts: Path, settings: TalkerSettings) -> None:
        self._path = path
        self._alerts = alerts
        self._settings = settings
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None

    def _load(self, today: str) -> dict[str, Any]:
        if self._data is None:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = raw if isinstance(raw, dict) else {}
            except FileNotFoundError:
                self._data = {}
            except (OSError, ValueError) as error:
                logger.error("the judges' spend can't be read (%s); counting from zero", type(error).__name__)
                self._data = {}
        data = self._data
        if data.get("day") != today:
            data.update(day=today, day_usd=0.0)
        data.setdefault("total_usd", 0.0)
        data.setdefault("alerted", [])
        return data

    def _save(self, data: Mapping[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            part = self._path.with_name(f"{self._path.name}.part")
            part.write_text(json.dumps(dict(data), sort_keys=True), encoding="utf-8")
            os.replace(part, self._path)
        except OSError as error:
            logger.error("the judges' spend wasn't saved: %s", type(error).__name__)

    def add(self, seconds: float, at: float) -> None:
        if seconds <= 0:
            return
        usd = seconds * self._settings.rate_per_hour / 3600.0
        with self._lock:
            data = self._load(_day(at))
            before = (float(data["day_usd"]), float(data["total_usd"]))
            data["day_usd"], data["total_usd"] = before[0] + usd, before[1] + usd
            fired = self._crossed(data, before, at)
            self._save(data)
        for record in fired:
            self._alert(record)

    def _crossed(self, data: dict[str, Any], before: tuple[float, float], at: float) -> list[dict[str, Any]]:
        s, day = self._settings, data["day"]
        lines: list[tuple[str, str, float, float, float]] = []  # key, scope, fraction, cap, spent
        for fraction in ALERT_FRACTIONS:
            lines.append((f"budget:{fraction}", "budget", fraction, s.budget_usd, data["total_usd"]))
            lines.append((f"{day}:soft:{fraction}", "day_soft", fraction, s.soft_usd_per_day, data["day_usd"]))
        lines.append((f"{day}:hard", "day_hard", 1.0, s.hard_usd_per_day, data["day_usd"]))
        if s.hard_usd_total is not None:
            lines.append(("hard_total", "total_hard", 1.0, s.hard_usd_total, data["total_usd"]))
        fired = []
        for key, scope, fraction, cap, spent in lines:
            if cap > 0 and spent >= fraction * cap and key not in data["alerted"]:
                data["alerted"].append(key)
                fired.append({"at": datetime.fromtimestamp(at, timezone.utc).isoformat(), "kind": "judge_gpu_spend",
                              "scope": scope, "fraction": fraction, "cap_usd": round(cap, 2),
                              "spent_usd": round(spent, 4), "day_usd": round(data["day_usd"], 4),
                              "total_usd": round(data["total_usd"], 4), "rate_per_hour": s.rate_per_hour})
        return fired

    def _alert(self, record: Mapping[str, Any]) -> None:
        logger.warning("JUDGE GPU SPEND ALERT: %s at %.0f%% of $%.2f (spent $%.2f today, $%.2f in all)",
                       record["scope"], 100 * float(record["fraction"]), record["cap_usd"],
                       record["day_usd"], record["total_usd"])
        try:
            self._alerts.parent.mkdir(parents=True, exist_ok=True)
            with open(self._alerts, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        except OSError as error:
            logger.error("a judge spend alert wasn't written: %s", type(error).__name__)

    def hard_reached(self, at: float) -> bool:
        s = self._settings
        with self._lock:
            data = self._load(_day(at))
            day_hit = s.hard_usd_per_day > 0 and data["day_usd"] >= s.hard_usd_per_day
            total_hit = s.hard_usd_total is not None and data["total_usd"] >= s.hard_usd_total
            return bool(day_hit or total_hit)

    def view(self, at: float) -> dict[str, Any]:
        with self._lock:
            data = self._load(_day(at))
            return {"day_usd": round(data["day_usd"], 4), "total_usd": round(data["total_usd"], 4)}


# ── The keep-warm window file ─────────────────────────────────────────────────


class WindowFile:
    """``serving.wake``'s keep-warm window, which the night's GPU stop and the live demo read. The
    judges only ever extend it while Glimmer is up and close what they opened; a longer window
    someone else holds is never shortened."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._mine: float | None = None

    def until(self, now: float) -> float | None:
        window = None if self._path is None else read_window(self._path)
        return window.until if window is not None and window.open_at(now) else None

    def extend(self, until: float, beat: float) -> None:
        if self._path is None:
            return
        held = read_window(self._path)
        if held is not None and held.until >= until and held.until != self._mine:
            return  # someone else's, longer
        self._write(WarmWindow(until, beat))
        self._mine = until

    def close(self, now: float) -> None:
        """End the window the judges opened, if it is still theirs."""
        if self._path is None or self._mine is None:
            return
        held = read_window(self._path)
        if held is not None and held.until == self._mine and held.until > now:
            self._write(WarmWindow(now, held.beat))
        self._mine = None

    def _write(self, window: WarmWindow) -> None:
        from anjo.sleep.serving import wake

        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        wake._write_window(self._path, window)  # the one writer of the file's format


# ── The talker ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TalkerStatus:
    """What answers a judge now. ``model`` is ``glimmer`` or always ``SPARK_LABEL`` while she isn't
    up; ``fallback_via`` names the model under that label (``fallback_via()``), ``None`` when warm."""

    state: State
    model: str
    eta_s: int | None = None
    reason: str | None = None
    fallback_via: str | None = None

    def view(self) -> dict[str, Any]:
        return {"state": self.state, "eta_s": self.eta_s, "model": self.model, "reason": self.reason,
                "fallback_via": self.fallback_via}


def fallback_via() -> str:
    """The model behind the ``spark-fallback`` label: Muse Spark when its key is set, else the
    chat route's own frontier fallback."""
    return VIA_SPARK if spark_configured() else VIA_FRONTIER


def _in_thread(name: str) -> Callable[[Callable[[], None]], None]:
    def spawn(run: Callable[[], None]) -> None:
        threading.Thread(target=run, name=name, daemon=True).start()

    return spawn


class JudgeTalker:
    """The cohort's view of Glimmer, its wakes, its lease and its keeper. ``clock`` is epoch
    seconds; ``spawn`` runs the wake and the keeper off the request thread (tests run them inline
    and drive ``tick`` themselves)."""

    def __init__(
        self, serving: Serving | None, settings: TalkerSettings, *, spend: SpendTracker,
        stop: Callable[[], StopResult], window: WindowFile, pay: Callable[[float], bool] = lambda _s: True,
        clock: Callable[[], float] = time.time, spawn: Callable[[Callable[[], None]], None] | None = None,
        keeper: Callable[[Callable[[], None]], None] | None = None, sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._serving = serving
        self.settings = settings
        self._spend = spend
        self._stop = stop
        self._window = window
        self._pay = pay
        self._clock = clock
        self._spawn = spawn or _in_thread("sleep-judge-wake")
        self._keeper = keeper or _in_thread("sleep-judge-keeper")
        self._sleep = sleep
        self._lock = threading.RLock()
        self._warm_until = 0.0
        self._lease_until = 0.0
        self._paying_since: float | None = None
        self._paid_s = 0.0  # this stretch's seconds, debited to the ledger when it ends
        self._waking_since: float | None = None
        self._last_wake_try = -math.inf
        self._last_ping = -math.inf
        self._last_beat: float | None = None
        self._asked: tuple[float, bool] | None = None
        self._keeping = False

    @property
    def backend(self) -> str | None:
        return None if self._serving is None else self._serving.name

    def _idle_window(self) -> float:
        if self.backend == "sagemaker":
            from anjo.sleep.serving.sagemaker.settings import IDLE_MINUTES

            return float(IDLE_MINUTES) * 60.0
        return float(SCALEDOWN_S)

    def _mark_up(self, now: float) -> None:
        self._warm_until = now + self._idle_window() - WARM_MARGIN_S
        self._last_beat = now
        if self._paying_since is None:
            self._paying_since = now
        if now < self._lease_until:
            self._window.extend(self._lease_until, now)

    # ── what judges see ───────────────────────────────────────────────────────

    def capped(self, now: float | None = None) -> bool:
        return self._spend.hard_reached(self._clock() if now is None else now)

    def up(self, *, ask: bool = False) -> bool:
        """Whether Glimmer is up, found without waking her (see the module's note)."""
        if self._serving is None:
            return False
        now = self._clock()
        if self._window.until(now) is not None or now < self._warm_until:
            return True
        if not ask or self._serving.name == "modal" or now >= self._lease_until:
            return False
        if self._asked is not None and now - self._asked[0] < PING_TTL_S:
            return self._asked[1]
        answered = self._ping()
        with self._lock:
            self._asked = (now, answered)
            if answered:
                self._mark_up(now)
        return answered

    def glimmer_first(self) -> bool:
        """Whether a judge's turn goes to Glimmer first: she is up, and the hard cap isn't reached."""
        return not self.capped() and self.up()

    def waking(self) -> bool:
        return self._waking_since is not None

    def status(self) -> TalkerStatus:
        now = self._clock()
        via = fallback_via()

        def fallback(state: State, *, eta_s: int | None = None, reason: str | None = None) -> TalkerStatus:
            return TalkerStatus(state, SPARK_LABEL, eta_s=eta_s, reason=reason, fallback_via=via)

        if self._serving is None:
            return fallback("cold", reason="no_backend")
        if self.capped(now):
            return fallback("cold", reason="budget")
        if self.up(ask=True):
            return TalkerStatus("warm", "glimmer")
        since = self._waking_since
        if since is not None:
            return fallback("waking", eta_s=max(0, round(self.settings.wake_eta_s - (now - since))))
        if not self.settings.wake_on:
            return fallback("cold", reason="wake_off")
        if counters().count(None, "wakes") >= self.settings.wakes_per_day:
            return fallback("cold", reason="wakes_spent")
        return fallback("cold", eta_s=round(self.settings.wake_eta_s))

    # ── presence ──────────────────────────────────────────────────────────────

    def arrive(self, *, lease_s: float | None = None) -> None:
        """A judge is here: hold the lease, keep her warm, and wake her if she sleeps (debounced,
        capped per day). Never blocks the request: the wake and the keeper run off its thread."""
        now = self._clock()
        wake = False
        with self._lock:
            self._lease_until = max(self._lease_until, now + (self.settings.idle_s if lease_s is None else lease_s))
            if self._serving is None or self.capped(now):
                return
            self._start_keeper()
            if self.up():
                if self._paying_since is None:
                    self._paying_since = now
                if self._last_beat is not None:
                    self._window.extend(self._lease_until, self._last_beat)
                return
            if self._waking_since is not None or not self.settings.wake_on or now - self._last_wake_try < WAKE_DEBOUNCE_S:
                return
            self._last_wake_try = now
            try:
                counters().claim(None, Check("wakes", None, self.settings.wakes_per_day, "wakes_spent"))
            except CapReached as refused:
                logger.warning("a judge wake was refused: %s", refused.reason)
                return
            self._waking_since = now
            if self._paying_since is None:
                self._paying_since = now
            wake = True
        if wake:
            self._spawn(self._wake)

    def _wake(self) -> None:
        from anjo.sleep.app.storage import glimmer_turns
        from anjo.sleep.serving import activity

        from anjo.core.best_effort import best_effort

        assert self._serving is not None
        result: WakeResult | None = None
        try:
            # Any backend's error (a cloud SDK's own included) ends the wake, never the waking state:
            # the next arrival past the debounce tries again, and Spark answers meanwhile.
            with best_effort("judge wake", type_only=True), activity.turn(glimmer_turns()):  # no GPU stop cuts it
                result = self._serving.wake()
        finally:
            with self._lock:
                self._waking_since = None
                if result is not None and result.ok:
                    self._mark_up(self._clock())
                elif result is not None:
                    logger.warning("a judge wake timed out after %.0f s", result.seconds)

    # ── the keeper ────────────────────────────────────────────────────────────

    def _start_keeper(self) -> None:
        if not self._keeping:
            self._keeping = True
            self._keeper(self._keep)

    def _keep(self) -> None:
        try:
            while self.tick():
                self._sleep(TICK_S)
        except BaseException:
            with self._lock:
                self._keeping = False
            raise

    def _ping(self) -> bool:
        from anjo.core.best_effort import best_effort

        assert self._serving is not None
        answered = False
        with best_effort("judge keep-warm ping", type_only=True):  # any backend error: not up
            answered = bool(self._serving.ping())
        return answered

    def _charge(self, now: float) -> None:
        if self._paying_since is not None:
            seconds = max(0.0, now - self._paying_since)
            self._spend.add(seconds, now)
            self._paid_s += seconds
            self._paying_since = now

    def _end_stretch(self, now: float, tail_s: float = 0.0) -> None:
        """Stop paying: charge the tail (an idle window that still runs), debit the stretch."""
        self._charge(now)
        if tail_s:
            self._spend.add(tail_s, now)
            self._paid_s += tail_s
        paid, self._paid_s, self._paying_since = self._paid_s, 0.0, None
        if paid and not self._pay(paid):
            logger.warning("the budget ledger didn't take the judges' %.0f warm seconds", paid)

    def tick(self) -> bool:
        """The keeper's step: charge the time paid since the last one, ping while a judge is present,
        stop paying once none is. False (the keeper ends, under the lock, so an ``arrive`` after it
        starts a new one) when there is nothing left to do."""
        now = self._clock()
        with self._lock:
            self._charge(now)
            present = now < self._lease_until and not self.capped(now)
            if present:
                if not (self.up() and now - self._last_ping >= KEEP_WARM_PING_S):
                    return True
                self._last_ping = now
                action = "ping"
            elif self._waking_since is not None:
                return True  # stop once the wake has ended
            elif self._paying_since is None:
                self._keeping = False
                return False
            else:
                action = "stop"
        if action == "ping":
            answered = self._ping()
            with self._lock:
                if answered:
                    self._mark_up(now)
                else:
                    self._warm_until = 0.0
                    self._window.close(now)
                    self._end_stretch(now)
            return True
        status, fields = self._stop()
        why = str(fields.get("why", ""))
        with self._lock:
            if status == "skipped" and why in RETRY_STOP:
                return True
            self._warm_until = 0.0
            self._window.close(now)
            idle_on = status == "failed" or why in UNSEEN_STOP  # her idle window runs out on its own
            self._end_stretch(now, self._idle_window() if idle_on else 0.0)
            logger.info("no judge present: Glimmer's stop %s%s", status, f" ({why})" if why else "")
            if self._clock() < self._lease_until and not self.capped():
                return True  # a judge came back during the stop: the next arrive wakes her
            self._keeping = False
            return False


# ── The host's talker ─────────────────────────────────────────────────────────


class _SageMakerCopies:
    """``GpuStop``'s containers on SageMaker: the one copy, stopped by setting the count to 0."""

    def __init__(self, serving: Serving) -> None:
        self._serving = serving

    def running(self, app: str) -> list[str]:
        return ["copies"]

    def stop(self, container: str) -> bool:
        self._serving.sleep_now()
        return False


def backend_stop(serving: Serving, quiet_s: float) -> Callable[[], StopResult]:
    """The night's own GPU stop, quiet for ``quiet_s`` after a turn: skips while a turn is in flight
    or recent, another run holds the night lock, or someone's keep-warm window is open."""
    from anjo.sleep.app.storage import SLEEP_DIR, data_root, glimmer_turns
    from anjo.sleep.budget import LEDGER_NAME, Ledger
    from anjo.sleep.night.sweep import held_elsewhere
    from anjo.sleep.serving import config
    from anjo.sleep.serving.stop import Containers, GpuStop, ModalCli
    from anjo.sleep.serving.wake import window_path

    def run() -> StopResult:
        try:
            containers: Containers = ModalCli() if serving.name == "modal" else _SageMakerCopies(serving)
            app = config.serve_app_name(config.serve_precision()) if serving.name == "modal" else "sagemaker"
        except ServeConfigError as error:
            return "failed", {"error": type(error).__name__}
        ledger = Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)
        stop = GpuStop(containers, app, turns=glimmer_turns(), window=window_path(data_root()), quiet_s=quiet_s,
                       night_running=lambda: held_elsewhere(ledger))
        reported: list[StopResult] = []
        stop.watch(lambda status, fields: reported.append((status, dict(fields))))
        stop()
        return reported[0] if reported else ("failed", {"error": "unreported"})

    return run


def _serving() -> Serving | None:
    from anjo.sleep.serving import backend

    try:
        return backend.from_env()
    except ServeConfigError as error:
        logger.warning("the judges' talker has no serving backend: %s", error)
        return None


def build_talker() -> JudgeTalker:
    from anjo.sleep.app.demo_live import ledger_pay
    from anjo.sleep.app.storage import data_root
    from anjo.sleep.serving.wake import window_path

    serving = _serving()
    settings = TalkerSettings.from_env(None if serving is None else serving.name)
    folder = judges_dir()
    stop = backend_stop(serving, settings.idle_s) if serving is not None else (lambda: ("skipped", {"why": "no_backend"}))
    return JudgeTalker(serving, settings, spend=SpendTracker(folder / SPEND_FILE, folder / ALERTS_FILE, settings),
                       stop=stop, window=WindowFile(window_path(data_root())), pay=ledger_pay(settings.rate_per_hour))


_talker_lock = threading.Lock()
_talker: list[JudgeTalker] = []


def talker() -> JudgeTalker:
    """One talker per process: its lease, wakes and keeper live in it."""
    with _talker_lock:
        if not _talker:
            _talker.append(build_talker())
        return _talker[0]


def set_talker(made: JudgeTalker | None) -> None:
    """Replace the process's talker (tests)."""
    with _talker_lock:
        _talker[:] = [] if made is None else [made]


# ── The route ─────────────────────────────────────────────────────────────────


def judge_route(user_id: str) -> ModelRoute | None:
    """The first route override on the sleep host: a judge's turn goes to Glimmer first while she is
    up (their own adapter once one is promoted), with Muse Spark behind; to Spark alone while she is
    cold, waking or past the hard cap. Anyone else, or no Spark key while she is cold: None, so the
    next override (``sleep_route``) decides as before."""
    if not is_judge(user_id):
        return None
    from anjo.sleep.app.flags import sleep_enabled_for
    from anjo.sleep.app.runtime import current_adapter, glimmer_pin

    if not sleep_enabled_for(user_id):
        return None
    pins: list[QualifiedPin] = []
    if talker().glimmer_first():
        adapter = current_adapter(user_id)
        pin = glimmer_pin(user_id, None if adapter is None else adapter.id,
                          adapter_rendering=None if adapter is None else adapter.rendering)
        pins.append(QualifiedPin(pin, None, None))
    if spark_configured():
        pins.append(QualifiedPin(spark_pin(user_id), None, None))
    if not pins:
        return None
    route: ModelRoute = ModelRoute(ROUTE_ANJO, None, OVERRIDE_CAUSE, None, tuple(pins))
    return route
