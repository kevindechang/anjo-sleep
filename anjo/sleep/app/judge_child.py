"""One judge's night, in its own process (2026-11-02): ``python -m anjo.sleep.night.judge
anjo.sleep.app.judge_child:run_night <id>``. The engine's entry hands ``run_night`` its night
(``night/judge.py``'s ``execute_record``), so nothing in the app imports the trainer (the import
contract "Sleep serving and app do not import the Sleep trainer"), as the night command's wiring.

It takes the night's hold on ``judge_usd`` (the one its admission took), runs ``night/judge.py``'s
night on the app's wiring with the judge's note held back until the night ends (``JudgeNotifier``),
writes each step into the night's record as the night view's stages, makes tonight's adapter servable
when it was promoted (``serve``), then settles the hold and checks the judges' spend alerts.

MIND on a judge's night runs only under the judge's live-pass lock (``MindLock``, the lock
``live_mind.live_pass_guard`` takes). At MIND's first use the night (which holds the night lock by
then, so no new live pass starts) waits for a pass under way (``wait_for_live_pass``) and takes the
lock; only once it has it is MIND on, and each MIND call then holds the lock for its own length, so the
page sees a live pass ``running`` only while MIND works. A pass still holding the lock after
``PASS_WAIT_S`` and ``PASS_WAIT_S`` more leaves MIND resting for the rest of the night: its ports
answer unavailable, so the engine runs the night as a night without MIND (no notice pass, no
questions, practice without MIND's loop), and the night view shows MIND's stages ``skipped`` with a
plain line (``judge_stages.MIND_RESTED``). The rest of the night goes on; MIND never runs unlocked.

The night's extraction starts where the live passes' stopped (``AfterLive``): the turns a pass already
extracted are in the judge's evidence store, and extracting them again under the night's own day and
window would store each a second time. As extraction starts it waits for a pass under way (whose end
moves ``live_mind.extracted_until``) and reads the cursor then: under the night lock, no new pass
extracts. Only a judge's night does this; every other night's wiring is the app's own.

The note: phase 1 writes it and the refine step may write it again after the gates, so the night
view shows each version as it comes, and the judge's outreach gets one note, the last, when the
night ends. Nothing is pushed: a judge is watching the night view.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol

from anjo.sleep.app import judge_night as jn
from anjo.sleep.app.judge_stages import Stages, close, fold, phase, rest_mind
from anjo.sleep.budget import Ledger, Reservation
from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.night.demo import from_env
from anjo.sleep.night.deps import NightDeps, NightOptions, Trainer
from anjo.sleep.ports import EvidenceSource, Notifier
from anjo.sleep.types import CheckpointRef, Evidence, NightResult

LIVE: Final = "Live: her next reply comes from tonight's weights"
LOADS_LATER: Final = "Loads with her next reply"
NOTHING_TO_SERVE: Final = "Nothing new to serve"
PASS_WAIT_S: Final = 120.0  # how long a night waits for a live MIND pass already under way, then for its lock
logger = logging.getLogger(__name__)


class Outcome(Protocol):
    """``night/judge.py``'s ``JudgeOutcome``, as this side reads it."""

    @property
    def result(self) -> NightResult: ...

    @property
    def warm_usd(self) -> float: ...


Execute = Callable[..., Outcome]  # ``night/judge.py``'s ``execute_record``


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JudgeNotifier:
    """``Notifier`` that keeps the night's latest note (shown on the night view through ``on_note``)
    and gives the judge's outreach only that one, at ``flush``."""

    def __init__(self, inner: Notifier, on_note: Callable[[str], None]) -> None:
        self._inner, self._on_note = inner, on_note
        self.latest: tuple[str, str, str] | None = None

    def queue(self, user_id: str, message: str, source: str) -> None:
        self.latest = (user_id, message, source)
        self._on_note(message)

    def push(self, user_id: str, message: str) -> bool:
        return False

    def flush(self) -> bool:
        if self.latest is None:
            return False
        self._inner.queue(*self.latest)
        return True


class Recorder:
    """The night's observer: each step event folded into the record's stages (``judge_stages.fold``)."""

    def __init__(self, store: jn.JobStore, night_id: str, label: Callable[[str], str | None]) -> None:
        self._store, self._id, self._label = store, night_id, label
        self._lock = threading.Lock()  # the background study reports from its own thread
        self.seen = False
        self.mind_rested = False

    def observe(self, user_id: str, step: str, state: str, receipt: Mapping[str, object]) -> None:
        self.seen = True
        at = _now()
        with self._lock:
            self._store.update(self._id, lambda job: self._folded(job, step, state, dict(receipt), at))

    def _folded(self, job: dict[str, Any], step: str, state: str, receipt: dict[str, Any], at: str) -> dict[str, Any]:
        stages: Stages = fold(job["stages"], step, state, receipt, at, self._label)
        if self.mind_rested:  # a step's own line never replaces why MIND didn't run
            stages = rest_mind(stages, at)
        return {**job, "stages": stages, "phase": phase(stages)}

    def rest_mind(self) -> None:
        """MIND rests tonight (``MindLock``): its stages say so from now on."""
        at = _now()
        with self._lock:
            self.mind_rested = True
            self._store.update(self._id, lambda job: {**job, "stages": rest_mind(job["stages"], at)})

    def note(self, text: str) -> None:
        self._store.update(self._id, lambda job: {**job, "note": text})

    def serve(self, state: str, summary: str, live: bool) -> None:
        at = _now()
        self._store.update(self._id, lambda job: {**job, "adapter_live": live, "stages": {
            **job["stages"], "serve": {**job["stages"]["serve"], "state": state, "summary": summary,
                                       "started_at": job["stages"]["serve"]["started_at"] or at, "ended_at": at}}})

    def finish(self, status: str, error: str | None, cost: float | None) -> None:
        at = _now()
        self._store.update(self._id, lambda job: {
            **job, "status": status, "error": error, "cost_usd": cost, "ended_at": at,
            "stages": close(job["stages"], at, failed=status == "failed")})


@dataclass(frozen=True)
class LivePorts:
    """``live_mind``'s calls the night keeps live passes off its MIND and its extraction with:
    ``wait(user_id, timeout_s)`` (True once no pass is running), ``guard(user_id, timeout_s=...)``, the
    judge's cross-process live-pass lock (``busy`` when it can't be had in time), and ``cursor``."""

    wait: Callable[[str, float], bool]
    guard: Callable[..., AbstractContextManager[object]]
    busy: type[Exception]
    cursor: Callable[[str], datetime | None]  # where the live passes' extraction stopped


def live_ports() -> LivePorts:
    from anjo.sleep.app import live_mind

    return LivePorts(wait=live_mind.wait_for_live_pass, guard=live_mind.live_pass_guard,
                     busy=live_mind.LiveBusy, cursor=live_mind.extracted_until)


def _utc(at: datetime) -> datetime:
    return at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)


class AfterLive:
    """``EvidenceSource`` for a judge's night: the inner source over only the part of each window after
    the judge's live passes' extraction (the module's docstring); a window they covered reads nothing."""

    def __init__(self, inner: EvidenceSource, ports: LivePorts | None = None, timeout_s: float = PASS_WAIT_S) -> None:
        self._inner, self._ports, self._timeout_s = inner, ports, timeout_s

    def extract_day(self, user_id: str, day: date, window: tuple[datetime, datetime],
                    as_of: datetime | None = None) -> list[Evidence]:
        ports = self._ports or live_ports()
        if not ports.wait(user_id, self._timeout_s):
            logger.warning("a live MIND pass still ran after %.0f s; the night reads the cursor as it stands",
                           self._timeout_s)
        start, end = window
        cursor = ports.cursor(user_id)
        after = None if cursor is None else _utc(cursor)
        if after is not None and after >= end:
            return []  # the live passes read all of it
        if after is not None and after > start:
            window = (after, end)
        return self._inner.extract_day(user_id, day, window, as_of)


class MindResting(RuntimeError):
    """A MIND call on a night whose MIND rests: the engine asked a port that answered unavailable."""


class MindLock:
    """The judge's live-pass lock as MIND's precondition (the module's docstring). ``ready`` decides
    once, at MIND's first use, and only after the lock was really had; ``held`` holds it around one MIND
    call (re-entrant within the night, one call at a time)."""

    def __init__(self, user_id: str, ports: LivePorts | None = None, timeout_s: float = PASS_WAIT_S,
                 on_rest: Callable[[], None] = lambda: None) -> None:
        self._user, self._ports, self._timeout_s, self._on_rest = user_id, ports, timeout_s, on_rest
        self._decide = threading.Lock()
        self._calls = threading.RLock()
        self._depth = 0
        self._stack = ExitStack()
        self.state = "untried"  # then "open" (the lock was had) or "resting" (it wasn't)

    def _live(self) -> LivePorts:
        return self._ports or live_ports()

    def ready(self) -> bool:
        with self._decide:
            if self.state == "untried":
                self.state = "open" if self._had() else "resting"
                if self.state == "resting":
                    self._on_rest()
        return self.state == "open"

    def _had(self) -> bool:
        ports = self._live()
        if not ports.wait(self._user, self._timeout_s):
            logger.warning("a live MIND pass still ran after %.0f s; the night waits for its lock", self._timeout_s)
        try:
            with ports.guard(self._user, timeout_s=self._timeout_s):
                return True
        except ports.busy:
            logger.warning("a live MIND pass held the judge's lock past %.0f s more: MIND rests tonight",
                           self._timeout_s)
            return False

    @contextmanager
    def held(self) -> Iterator[None]:
        if not self.ready():
            raise MindResting("MIND rests tonight: a live pass held the judge's lock")
        with self._calls:
            if self._depth == 0:
                self._stack.enter_context(self._live().guard(self._user, timeout_s=self._timeout_s))
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    self._stack.close()


class LockedMind:
    """A MIND port (the night's ``Notice`` or ``Inquiry``) on a judge's night: unavailable while MIND
    rests, and each of its calls under the judge's live-pass lock (``MindLock.held``)."""

    def __init__(self, inner: Any, lock: MindLock) -> None:
        self._inner, self._lock = inner, lock

    @property
    def available(self) -> bool:
        return bool(self._inner.available) and self._lock.ready()

    def __getattr__(self, name: str) -> Any:
        found = getattr(self._inner, name)
        if not callable(found):
            return found

        def locked(*args: Any, **kwargs: Any) -> Any:
            with self._lock.held():
                return found(*args, **kwargs)

        return locked


def target_label(user_id: str) -> Callable[[str], str | None]:
    from anjo.sleep.app.storage import target_store

    def label(target_id: str) -> str | None:
        found = target_store(user_id).get(target_id)
        return found.label if found is not None else None

    return label


def judge_wiring(notifier: Notifier, live: LivePorts | None = None,
                 mind: MindLock | None = None) -> Callable[[Trainer, NightOptions], NightDeps]:
    """The app's night wiring, with the judge's note held back, extraction after the live passes' and
    MIND under the judge's live-pass lock."""
    from anjo.sleep.app.night import build

    def wire(trainer: Trainer, options: NightOptions) -> NightDeps:
        deps = build(trainer, options)
        locked = {} if mind is None else {"notice": LockedMind(deps.notice, mind), "inquiry": LockedMind(deps.inquiry, mind)}
        return replace(deps, notifier=notifier, extractor=lambda chat: AfterLive(deps.extractor(chat), live), **locked)

    return wire


def serve(checkpoint: CheckpointRef | None, recorder: Recorder,
          ensure: Callable[[CheckpointRef], None] | None = None) -> bool:
    """Make a promoted adapter servable now; the night stands whatever this does."""
    if checkpoint is None:
        recorder.serve("skipped", NOTHING_TO_SERVE, False)
        return False
    if ensure is None:
        from anjo.sleep.serving import backend

        ensure = backend.from_env().client().ensure_served
    from anjo.sleep.serving.client import ServingError

    try:
        ensure(checkpoint)
    except (ServingError, OSError, TimeoutError) as error:
        logger.warning("tonight's adapter isn't live yet (%s)", type(error).__name__)
        recorder.serve("failed", LOADS_LATER, False)
        return False
    recorder.serve("done", LIVE, True)
    return True


def _job(record: Mapping[str, Any]) -> dict[str, Any]:
    """The night's ``JudgeNight`` fields, from its record (``execute_record`` builds the night from them)."""
    start, end = (datetime.fromisoformat(at) for at in record["window"])
    return {"user": str(record["user_id"]), "night": date.fromisoformat(record["night"]), "window": (start, end),
            "as_of": datetime.fromisoformat(record["as_of"])}


def _settle(held: Reservation, outcome: Outcome | None, recorder: Recorder) -> float | None:
    """What the night cost on ``judge_usd``: its own cost and the warm trainer's, settled once. A night
    that broke after it began (no result) settles its whole bound; one that broke before, nothing."""
    if outcome is not None:
        spent = outcome.result.cost_usd + outcome.warm_usd
        held.settle(spent)
        return spent
    if recorder.seen:
        held.settle(held.bound.amount)
        return None
    held.release()
    return 0.0


def run(night_id: str, *, store: jn.JobStore, book: Ledger, environ: Mapping[str, str], execute: Execute,
        notifier: Notifier | None = None, ensure: Callable[[CheckpointRef], None] | None = None,
        live: LivePorts | None = None) -> int:
    record = store.get(night_id)
    if record is None:
        return 2
    recorder = Recorder(store, night_id, target_label(str(record["user_id"])))
    held = jn.night_hold(book, night_id, environ)
    if held is None:
        recorder.finish("failed", jn.BUDGET, None)
        return 1
    if notifier is None:
        from anjo.sleep.app.delivery import OutreachNotifier

        notifier = OutreachNotifier()
    held_note = JudgeNotifier(notifier, recorder.note)
    mind = MindLock(str(record["user_id"]), live, on_rest=recorder.rest_mind)
    outcome: Outcome | None = None
    try:
        outcome = execute(wiring=judge_wiring(held_note, live, mind), job=_job(record), profile=from_env(environ),
                          data_dir=Path(environ[DATA_DIR_ENV]), ledger=book, observer=recorder.observe,
                          environ=environ)
        result: NightResult = outcome.result
        held_note.flush()
        serve(result.checkpoint if result.status == "promoted" else None, recorder, ensure)
        status, why = ("failed", f"stopped at {result.failed_at}") if result.status == "failed" else ("done", None)
    except Exception as error:  # the night's record says it failed; the process exits non-zero
        logger.exception("a judge's night failed")
        status, why = "failed", type(error).__name__
    finally:
        cost = _settle(held, outcome, recorder)
        jn.check_alerts(book, store.root.parent / jn.ALERTS_FILE, jn.budget_usd(environ), datetime.now(timezone.utc))
    recorder.finish(status, why, cost)
    return 0 if status == "done" else 1


def run_night(night_id: str, execute: Execute) -> int:
    """The night process's entry (``night/judge.py``'s ``main`` calls it with ``execute_record``)."""
    logging.basicConfig(level=logging.INFO)
    return run(jn.check_id(night_id), store=jn.job_store(), book=jn.ledger(), environ=dict(os.environ),
               execute=execute)
