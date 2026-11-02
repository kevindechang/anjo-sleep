"""One judge's night, in its own process (2026-11-02): ``python -m anjo.sleep.night.judge
anjo.sleep.app.judge_child:run_night <id>``. The engine's entry hands ``run_night`` its night
(``night/judge.py``'s ``execute_record``), so nothing in the app imports the trainer (the import
contract "Sleep serving and app do not import the Sleep trainer"), as the night command's wiring.

It takes the night's hold on ``judge_usd`` (the one its admission took), runs ``night/judge.py``'s
night on the app's wiring with the judge's note held back until the night ends (``JudgeNotifier``),
writes each step into the night's record as the night view's stages, makes tonight's adapter servable
when it was promoted (``serve``), then settles the hold and checks the judges' spend alerts.

The note: phase 1 writes it and the refine step may write it again after the gates, so the night
view shows each version as it comes, and the judge's outreach gets one note, the last, when the
night ends. Nothing is pushed: a judge is watching the night view.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol

from anjo.sleep.app import judge_night as jn
from anjo.sleep.app.judge_stages import Stages, close, fold, phase
from anjo.sleep.budget import Ledger, Reservation
from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.night.demo import from_env
from anjo.sleep.night.deps import NightDeps, NightOptions, Trainer
from anjo.sleep.ports import Notifier
from anjo.sleep.types import CheckpointRef, NightResult

LIVE: Final = "Live: her next reply comes from tonight's weights"
LOADS_LATER: Final = "Loads with her next reply"
NOTHING_TO_SERVE: Final = "Nothing new to serve"
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

    def observe(self, user_id: str, step: str, state: str, receipt: Mapping[str, object]) -> None:
        self.seen = True
        at = _now()
        with self._lock:
            self._store.update(self._id, lambda job: self._folded(job, step, state, dict(receipt), at))

    def _folded(self, job: dict[str, Any], step: str, state: str, receipt: dict[str, Any], at: str) -> dict[str, Any]:
        stages: Stages = fold(job["stages"], step, state, receipt, at, self._label)
        return {**job, "stages": stages, "phase": phase(stages)}

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


def target_label(user_id: str) -> Callable[[str], str | None]:
    from anjo.sleep.app.storage import target_store

    def label(target_id: str) -> str | None:
        found = target_store(user_id).get(target_id)
        return found.label if found is not None else None

    return label


def judge_wiring(notifier: Notifier) -> Callable[[Trainer, NightOptions], NightDeps]:
    from anjo.sleep.app.night import build

    return lambda trainer, options: replace(build(trainer, options), notifier=notifier)


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
        notifier: Notifier | None = None, ensure: Callable[[CheckpointRef], None] | None = None) -> int:
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
    outcome: Outcome | None = None
    try:
        outcome = execute(wiring=judge_wiring(held_note), job=_job(record), profile=from_env(environ),
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
