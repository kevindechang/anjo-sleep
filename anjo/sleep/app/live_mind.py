"""MIND after each judge's reply (the judge cohort only): one small pass over what they said since
the last pass, so the panel beside the chat moves while they talk, not only after a night.

**One pass** (``run_pass``), at most ``ANJO_SLEEP_JUDGE_LIVE_PER_DAY`` a judge a day (default 8,
``judge_cohort.claim_live``):

1. the judge's own turns since the last pass (or since they joined) go through the night's own
   extraction (``HistorySource.record_day``, Spark Open), into their evidence store and targets;
2. MIND's notice pass reads the fresh items (``MindNight.notice``, the night's own, under the same
   holds on the budget ledger): it may open questions, with alternatives and their credences. The
   thinker is base Glimmer while she is up (``judge_talker``), Spark Open otherwise, and the record
   says which (``glimmer`` or ``spark-open``). A live pass names today as its night, so its tokens
   count against tonight's MIND budget (``night.mind.tonight``: 12 cycles, 20k tokens);
3. her chooser runs on the state as it now stands (``StateChooser``, the night's policy) and the
   record keeps every want it weighed and what it would pick. It acts on nothing: studying,
   inquiring and the note stay the night's.

The record (``<sleep folder>/live_mind.json``, encrypted like every sleep file, erased with them)
holds ids, counts and codes, never a conversation word; ``judge_views`` reads it for
``GET /api/sleep/me/mind``.

**Never two at once, never under a night.** One pass per judge runs at a time; a reply during it asks
for one more after it. A pass skips while any night holds the host's night lock
(``sweep.held_elsewhere``) or while the night worker holds this judge ``paused``; the pass asked for
then runs when the pause ends.

**What the night worker imports:** ``paused(user_id)`` around a night that writes the judge's
workspace, and ``extracted_until(user_id)``: the end of the window the live passes extracted, so the
night's own extraction can start there instead of reading those turns again.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.app.judge_cohort import CapReached, claim_live, is_judge, joined_at

logger = logging.getLogger(__name__)

RECORD_FILE: Final = "live_mind.json"
FIRST_WINDOW: Final = timedelta(days=1)  # a judge with no join time: their last day
GLIMMER_THINKER: Final = "glimmer"
SPARK_THINKER: Final = "spark-open"
LIVE_ALLOWANCE: Final = (2, 1, 1.0)  # what a night offers her chooser, for the live pick: actions, calls, dollars
LIVE_SEED: Final = 0


# ── The record ────────────────────────────────────────────────────────────────


def _record_path(user_id: str) -> Path:
    from anjo.sleep.app.storage import sleep_dir

    path: Path = sleep_dir(user_id) / RECORD_FILE
    return path


def load_record(user_id: str) -> dict[str, Any] | None:
    """The last pass's record, or None (none yet, or it can't be read)."""
    from anjo.sleep.app.storage import CODEC

    try:
        data = json.loads(CODEC.decode(_record_path(user_id).read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as error:
        logger.error("a judge's live MIND record can't be read: %s", type(error).__name__)
        return None
    return data if isinstance(data, dict) else None


def save_record(user_id: str, record: Mapping[str, Any]) -> None:
    from anjo.sleep.app.storage import CODEC, writer_guard

    path = _record_path(user_id)
    with writer_guard(user_id)():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.part")
        part.write_text(CODEC.encode(json.dumps(dict(record), sort_keys=True)), encoding="utf-8")
        os.replace(part, path)


def extracted_until(user_id: str) -> datetime | None:
    """Where the live passes' extraction stopped: the night's own may start here."""
    record = load_record(user_id)
    raw = None if record is None else record.get("cursor")
    try:
        return datetime.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


# ── One pass ──────────────────────────────────────────────────────────────────


def _job(user_id: str, now: datetime) -> str:
    who = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    return f"sleep-judge-live-{who}-{now:%Y%m%dT%H%M%S}"


def _models(glimmer: bool) -> Callable[[str, Any], Any]:
    """The pass's model ports: base Glimmer's thinker while she is up, Spark Open's otherwise; the
    resolver is always Spark Open, as at night."""
    from anjo.sleep.night.mind import MindModels
    from anjo.sleep.spark_inquiry import SparkResolver, SparkThinker, SparkWorldQuery, spark_open

    def models(_user_id: str, spark: Any) -> MindModels:
        resolver = SparkResolver(chat=spark(spark_open))
        if glimmer:
            from anjo.sleep.app.mind_night import GlimmerThinker, GlimmerWorldQuery, glimmer_chat
            from anjo.sleep.provenance import GLIMMER
            from anjo.sleep.serving import backend

            chat = glimmer_chat(backend.from_env().client().with_load_wait(0).once(), seed=LIVE_SEED, wake=None)
            return MindModels(thinker=GlimmerThinker(chat), world=GlimmerWorldQuery(chat), resolver=resolver,
                              model=GLIMMER)
        chat = spark(spark_open)
        return MindModels(thinker=SparkThinker(chat), world=SparkWorldQuery(chat), resolver=resolver,
                          model=SPARK_THINKER)

    return models


@dataclass(frozen=True)
class PassPorts:
    """What a pass calls out to; tests replace them."""

    extract: Callable[[str, datetime, datetime], list[Any]]
    mind: Callable[[bool], Any]  # thinker on Glimmer? -> a MindNight
    glimmer_up: Callable[[], bool]
    night_running: Callable[[], bool]


def _extract(user_id: str, start: datetime, end: datetime) -> list[Any]:
    from anjo.sleep.app.history_source import HistorySource

    added: list[Any] = HistorySource().record_day(user_id, end.date(), (start, end), as_of=end)[0]
    return added


def _mind(glimmer: bool) -> Any:
    from anjo.sleep.app.night import prices
    from anjo.sleep.app.storage import SLEEP_DIR, data_root, workspace_log
    from anjo.sleep.budget import LEDGER_NAME, Ledger
    from anjo.sleep.night.mind import MindNight

    return MindNight(models=_models(glimmer), log=workspace_log, prices=prices(),
                     ledger=Ledger(data_root() / SLEEP_DIR / LEDGER_NAME))


def _glimmer_up() -> bool:
    from anjo.sleep.app.judge_talker import talker

    return talker().glimmer_first()


def _night_running() -> bool:
    from anjo.sleep.app.storage import SLEEP_DIR, data_root
    from anjo.sleep.budget import LEDGER_NAME, Ledger
    from anjo.sleep.night.sweep import held_elsewhere

    return bool(held_elsewhere(Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)))


def default_ports() -> PassPorts:
    return PassPorts(extract=_extract, mind=_mind, glimmer_up=_glimmer_up, night_running=_night_running)


def _call(user_id: str, now: datetime, mind: Any) -> Any:
    from anjo.sleep.app.night import prices
    from anjo.sleep.app.storage import evidence_store, target_store
    from anjo.sleep.night.deps import MindCall
    from anjo.sleep.night.meter import SpendMeter, WonderMeter

    rates = prices()
    return MindCall(user_id=user_id, night=now.date(), as_of=now, at=now, store=evidence_store(user_id),
                    targets=target_store(user_id),
                    spend=SpendMeter(spark_in_per_m=rates.spark_in_per_m, spark_out_per_m=rates.spark_out_per_m),
                    wonder=WonderMeter(0, 0.0), actions_left=0, job=_job(user_id, now), cancelled=lambda: False)


def _choose(user_id: str, now: datetime, mind: Any, call: Any) -> Any:
    """Her chooser on the state as it stands: what she would pick, acting on nothing."""
    from anjo.sleep.app.intentions import LedgerIntentions
    from anjo.sleep.app.policy import build_policy
    from anjo.sleep.mind import points
    from anjo.sleep.mind.choose import StateChooser
    from anjo.sleep.mind.inquiry.findings import PERSON_KINDS
    from anjo.sleep.mind.wants import own_wants_on
    from anjo.sleep.types import ChoiceContext, NightAllowance

    own = [item.created_at for item in call.store.stored(now.date()) if item.kind in PERSON_KINDS]
    raw = points.RawSnapshot(tuple(call.targets.open_targets(now)), tuple(mind.results(call)), now,
                             max(own) if own else None, intentions=LedgerIntentions().intentions(user_id, now),
                             questions=tuple(mind.questions(call)))
    chooser = StateChooser()
    state = points.build(raw, inquiry_offered=True, own_wants=own_wants_on(os.environ))
    actions, calls, dollars = LIVE_ALLOWANCE
    return chooser.choose("away", chooser.derive(state), ChoiceContext(build_policy(), NightAllowance(actions, calls, dollars)))


def _choice_record(choice: Any) -> dict[str, Any]:
    return {
        "action": choice.action, "target_id": choice.target_id, "question_id": choice.question_id,
        "kind": None if choice.want is None else choice.want.kind,
        "considered": [{"want_id": c.want_id, "kind": c.kind, "target_id": c.target_id, "action": c.action,
                        "strength": c.strength, "status": c.status, "reason": c.reason} for c in choice.considered],
    }


@contextmanager
def _glimmer_turn(glimmer: bool) -> Iterator[None]:
    if not glimmer:
        yield
        return
    from anjo.sleep.app.storage import glimmer_turns
    from anjo.sleep.serving import activity

    with activity.turn(glimmer_turns()):
        yield


def run_pass(user_id: str, *, ports: PassPorts | None = None, now: datetime | None = None) -> dict[str, Any]:
    """One live pass for ``user_id`` (see the module's note); returns the record it saved, or why
    it saved none."""
    p = ports or default_ports()
    at = now or datetime.now(timezone.utc)
    if not is_judge(user_id):
        return {"state": "not_a_judge"}
    if p.night_running():
        return {"state": "night_running"}
    try:
        claim_live(user_id, at)
    except CapReached as refused:
        return {"state": "capped", "reason": refused.reason}
    previous = load_record(user_id) or {}
    start = extracted_until(user_id) or joined_at(user_id) or at - FIRST_WINDOW
    glimmer = p.glimmer_up()
    record: dict[str, Any] = {"at": at.isoformat(), "thinker": GLIMMER_THINKER if glimmer else SPARK_THINKER,
                              "cursor": previous.get("cursor"), "passes": int(previous.get("passes", 0)) + 1}
    try:
        fresh = p.extract(user_id, min(start, at), at) if start < at else []
        record["cursor"] = at.isoformat()
        mind = p.mind(glimmer)
        call = _call(user_id, at, mind)
        with _glimmer_turn(glimmer):  # her request in flight: no GPU stop cuts it
            record["notice"] = dict(mind.notice(call, fresh))
        record["choice"] = _choice_record(_choose(user_id, at, mind, call))
        record["state"] = "ok"
    except LIVE_FAILURES as error:
        logger.warning("a judge's live MIND pass failed: %s", type(error).__name__)
        record.update(state="failed", error=type(error).__name__)
    save_record(user_id, record)
    return record


def _failures() -> tuple[type[BaseException], ...]:
    from anjo.sleep.model_api import ModelApiError
    from anjo.sleep.serving.activity import GlimmerStopping
    from anjo.sleep.serving.config import ServeConfigError

    return (ModelApiError, GlimmerStopping, ServeConfigError, ValueError, LookupError, OSError)


LIVE_FAILURES: Final = _failures()


# ── The runner ────────────────────────────────────────────────────────────────


def _in_thread(run: Callable[[], None]) -> None:
    threading.Thread(target=run, name="sleep-judge-live-mind", daemon=True).start()


class LiveRunner:
    """One pass per judge at a time; a pass asked for during one, or during a pause, runs after."""

    def __init__(self, run: Callable[[str], Mapping[str, Any]] = run_pass,
                 spawn: Callable[[Callable[[], None]], None] = _in_thread) -> None:
        self._run = run
        self._spawn = spawn
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._again: set[str] = set()
        self._paused: dict[str, int] = {}

    def schedule(self, user_id: str) -> bool:
        """Ask for a pass; False when it waits for the running one or a pause to end."""
        with self._lock:
            if user_id in self._running or self._paused.get(user_id, 0):
                self._again.add(user_id)
                return False
            self._running.add(user_id)
        self._spawn(lambda: self._loop(user_id))
        return True

    def _loop(self, user_id: str) -> None:
        while True:
            try:
                self._run(user_id)
            finally:
                with self._lock:
                    again = user_id in self._again and not self._paused.get(user_id, 0)
                    if again:
                        self._again.discard(user_id)
                    else:
                        self._running.discard(user_id)
            if not again:
                return

    @contextmanager
    def paused(self, user_id: str) -> Iterator[None]:
        with self._lock:
            self._paused[user_id] = self._paused.get(user_id, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                left = self._paused.get(user_id, 1) - 1
                if left:
                    self._paused[user_id] = left
                else:
                    self._paused.pop(user_id, None)
                rerun = not left and user_id in self._again and user_id not in self._running
                if rerun:
                    self._again.discard(user_id)
            if rerun:
                self.schedule(user_id)

    def busy(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._running


_runner: list[LiveRunner] = []
_runner_lock = threading.Lock()


def runner() -> LiveRunner:
    with _runner_lock:
        if not _runner:
            _runner.append(LiveRunner())
        return _runner[0]


def set_runner(made: LiveRunner | None) -> None:
    """Replace the process's runner (tests)."""
    with _runner_lock:
        _runner[:] = [] if made is None else [made]


@contextmanager
def paused(user_id: str) -> Iterator[None]:
    """Hold this judge's live passes off (a night writes the same workspace log)."""
    with runner().paused(user_id):
        yield
