"""``run_night`` and ``run_night_batch`` (Task 5): one person's night, or many people's in one batch.

Steps, in order, each writing its log record before the next starts (``nightlog.py``):

1. ``close_day``: the night's window (the calendar day, or an accelerated run's simulated day).
2. ``extract``: the day's evidence and targets (Task 3). ``sleep`` (2b, with a sleep policy): a nap or
   a deep sleep (``night/sleep_pressure.py``), decided on the CPU steps' counts. A nap (``napped``)
   wakes no Glimmer: it skips the probes and steps 5 to 9, and defers MIND's notice and ``inquire`` and
   the note to her next deep sleep (``night/sleep_night.py``). ``notice`` (2c, Task 17): with
   ``mind_inquiry`` on, one thinker pass over the person's new items, every one since her last night
   that woke, may open questions (``night/mind.py``); else a no-op. ``dream`` (2d, for a person MIND's
   core serves; a nap too, on the CPU): she dreams, opens questions and picks what practice rehearses
   (``core_steps.py``).
3. ``probes``: Kevin's night probes are generated and sealed, a synthetic person's were sealed once,
   before any training input exists.
4. ``choose``: her chooser (Task 7a's ``StateChooser``; ``RuleChooser`` is the fallback) picks one action
   with the night's allowance. A study runs homework (Task 7's ``do_homework``), stores its
   artifact as hosted evidence and its ``research`` result, and marks the target's revision acted.
   Wonder's dollars are what the person's rolling $3-a-month cap has left; with none left no study
   runs, and the record says so (``wonder_month``). No study is offered while the homework port is
   a no-op. ``hold_note`` is written at step 10. A chosen study records her expectation once the
   estimator is registered (Task 16), and a synthetic life records the offer's decision point.
   With ``mind_inquiry`` on, her questions join the snapshot and ``inquire`` runs MIND (Task 17).
   With ``ANJO_SLEEP_AGENT_BASELINE=1`` the night runs the comparison's conventional agent right
   after it (``agent_baseline``): one stateless pass a night that leaves ``agent_note`` items only
   the morning arm 2A reads.
6. ``brief``: both briefs fold the evidence no night has folded yet (before practice, so the
   training brief is the pre-practice one). A brief that breaks a rule twice (``context.brief``)
   keeps both briefs she had: the step is logged ``failed`` with the night it kept (``kept_brief``),
   tonight's items wait for the next fold, and the night goes on with the kept briefs.
5. ``practice``, 6. ``dataset``, 7. ``train``, 8. ``gates``: only on a training night, one whose
   cleared, non-practice evidence since the last trained night reaches ``min_new_examples``.
   Under the MIND loop (``ANJO_SLEEP_MIND_LOOP=1`` with MIND on) practice first takes her focus from
   her workspace at the night's start, she commits a prediction about her own learning on each focus
   target, and that night's practice record resolves it by code (``mind/loop_focus.py``,
   ``mind/self_predict.py``); off, practice and its record are today's.
   The last trained night is the night of the checkpoint that serves (the one the next night
   trains from), so a failed or rejected night's evidence stays new. Every due person in a batch
   runs inside one GPU session (a shared one is in each summary as ``gpu_batch``, so the month's spend
   splits it). Below it, the night is ``nothing_new`` and those items carry over.
9. ``promote``: every gate that ran passed (a switched-off gate, ``skipped``, is left out), or the
   night is rejected and the previous checkpoint serves.
10. ``note``: a ``hold_note`` choice, or a second offer (``allowed_actions={"hold_note"}``) after a
    result with a verified source, or an inquiry's anchored one (with ``mind_inquiry`` on, each finding
    with an unconsumed anchor shares here); at most one note, which the writer may decline. A synthetic life
    records this offer's decision point too, before the note settles its target, offered or not: a
    night with no second offer records one that allows nothing. A nap writes none: nothing shares, and
    the note is ``deferred``, its target unsettled.
11. ``finish``: the ``NightResult``, every receipt, and the night told as one loop (``story.py``).

A failure, a budget stop or a cancel stops that person's night at that step: the previous
checkpoint keeps serving, no note is held, and the rest of the batch goes on (RF1). Every date is
the person's own night's, never the wall clock's: her snapshot, its eligibility and her expectations
read the night's clock (``as_of``, the virtual night end). Records are stamped at the window's
wall-clock end or ``as_of``, whichever is earlier (``PersonNight.stamp``), so readers on either clock see
them; the wall-clock end alone drives budgets and timeouts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any, Final

from anjo.sleep.night.deps import (
    STUDY_ACTIONS,
    MindCall,
    NightDeps,
    StudyOutcome,
)
from anjo.sleep.serving.backend import phase
from anjo.sleep.serving.client import GlimmerNotReady
from anjo.sleep.night import channel_night
from anjo.sleep.night.core_steps import CoreSteps
from anjo.sleep.harness import harness_on
from anjo.sleep.harness.loop import HARNESS_MAX_CALLS, HarnessCall, run_harness
from anjo.sleep.night.holds import SPARK_MAX_OUT_TOKENS, held, spark_worst_usd, usd
from anjo.sleep.night.demo import PROBES_SKIPPED
from anjo.sleep.night.sweep import Sweep
from anjo.sleep.practice.full import TEACHER_ALARM_SHARE
from anjo.sleep.night.gpu import batch_digest
from anjo.sleep.night.meter import SpendMeter, WonderMeter, WonderSpent
from anjo.sleep.night.past import paid_within, spent_within
from anjo.sleep.night.sleep_night import (
    CAPPED_WHY,
    DEFER_WHY,
    catch_up,
    deferred_note,
    mind_on,
    offered_tonight,
    owed_tonight,
    put_off,
    sleep_step,
    unnoticed,
)
from anjo.sleep.night.state import GPU_STEPS as GPU_STEPS
from anjo.sleep.night.state import (
    PersonNight,
    checkpoint_id,
    kept_brief,
    kept_summary,
    note_error,
    safe_error,
    study_outcome,
)
from anjo.sleep.night.story import KEPT
from anjo.sleep.agent_baseline.types import AgentCall
from anjo.sleep.context.brief import MAX_TOKENS as BRIEF_MAX_TOKENS
from anjo.sleep.context.brief import BriefError
from anjo.sleep.night.state import Window as Window
from anjo.sleep.night.state import day_window as day_window
from anjo.sleep.mind import expect, loop_focus, points, receipt, self_predict
from anjo.sleep.mind.wants import own_wants_on
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.sim.day import DayRun
from anjo.sleep.night.nightlog import NightLog, Uses, gate_record, night_folder, summaries
from anjo.sleep.practice.run import PracticeRun
from anjo.sleep.note.hold import HELD_BACK_LINE, HeldNote
from anjo.sleep.note.hold import SOURCE as NOTE_SOURCE
from anjo.sleep.note.hold import Fact, write_note
from anjo.sleep.stops import BudgetStop, Cancelled
from anjo.sleep.train.dataset import build_night_dataset
from anjo.sleep.train.lora import fresh_adapter_name
from anjo.sleep.train.terms import attach_night_terms
from anjo.sleep.types import (
    MAX_NIGHT_ACTIONS,
    Choice,
    ChoiceContext,
    CostPart,
    Evidence,
    NightResult,
    NightStatus,
    Result,
    Target,
    WantSnapshot,
)

PERSON_KINDS: Final = frozenset({"lesson", "correction", "preference", "tone_request"})
USES: Final[Mapping[str, tuple[Uses, str | None]]] = {
    "sweep": ("new", None),
    "simulate": ("existing", "chat route"),
    "close_day": ("new", None),
    "extract": ("existing", "history store"),
    "notice": ("new", None),
    "dream": ("new", None),
    "sleep": ("new", None),
    "probes": ("new", None),
    "inquire": ("new", None),
    "agent": ("new", None),
    "choose": ("new", None),
    "brief": ("new", None),
    "practice": ("new", None),
    "harness": ("new", None),
    "dataset": ("new", None),
    "train": ("new", None),
    "gates": ("existing", "crisis check"),
    "promote": ("new", None),
    "note": ("existing", "outreach"),
    "finish": ("new", None),
    # A judge's night only (``night/demo.py``): the background study's end, MIND's early commitment and
    # the note's rewrite after the gates.
    "homework": ("new", None),
    "self_predict": ("new", None),
    "refine": ("existing", "outreach"),
}
# Worst cases for the paid steps' holds (holds.py): Spark Open calls per step, and the rest.
EXTRACT_MAX_CALLS: Final = 2  # the day's one call, and one retry
BRIEF_MAX_CALLS: Final = (
    4  # the served and the training brief, each retried once (each held at its cap)
)
PROBES_WORST_USD: Final = 0.50  # Muse Spark writing a real person's night probes (about 8 calls)
NOTE_MAX_S: Final = (
    300.0  # one Glimmer call on the serving GPU, inside the client's request timeout
)

# A study that fails this way stops the study, not the night: a Wonder error or timeout, a meta_usd
# hold with no room, a request that can't fit its bound, a page in an unknown encoding.
STUDY_FAILURES: Final = (ModelApiError, BudgetStop, ValueError, LookupError)
# What ends the MIND loop's share of a night, not the night: a log that can't be read or doesn't
# fold, a summary or focus record of the wrong shape, a prompt file that can't be read. The loop
# degrades to today's practice and the night trains; a cancel or a budget stop still reaches it.
LOOP_FAILURES: Final = (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError)
# What ends the harness step, not the night: the proposal, the vet or a probe call failed, or its
# log couldn't be read or its code broke. The night keeps the list it had and goes on to train; a
# cancel still stops it (``Cancelled`` is caught before this, never read as a harness failure).
HARNESS_FAILURES: Final = (
    ModelApiError, BudgetStop, ValueError, KeyError, TypeError, OSError, RuntimeError,
    AttributeError, ImportError, LookupError, AssertionError,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Focus:
    """The MIND loop's share of a night's practice: the targets her questions are working on, the
    practice step's record so far (None: the loop is off), the call the loop's writes go through, the
    predictions she committed before practice, and the finished nights whose summaries hold her
    baseline. Off, every field is empty and practice is today's."""

    targets: tuple[loop_focus.FocusTarget, ...] = ()
    record: Mapping[str, Any] | None = None
    call: MindCall | None = None
    commitment: self_predict.Commitment = self_predict.Commitment()
    history: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class _Study:
    """A judge's study running in the background (``night/demo.py``): what it studies, the Wonder calls
    spent before it started, and the call itself, which only touches the night's Wonder meter."""

    target: Target
    choice: Choice
    calls_before: int
    future: Future[StudyOutcome] = field(compare=False)


@dataclass(frozen=True)
class _Written:
    """A judge's phase-1 note and what it was written from, for its rewrite after the gates."""

    choice: Choice
    note: HeldNote
    topic: str
    facts: tuple[Fact, ...]


def _kept_note(kept: _Written, error: str) -> dict[str, Any]:
    """The refine step's receipt when the phase-1 note stands because the rewrite or its queueing failed."""
    return {"outcome": "kept", "note_id": kept.note.id, "note_error": error}


class NightRunner(CoreSteps):
    """Runs a batch of nights over one set of ports."""

    def __init__(self, deps: NightDeps) -> None:
        self.deps = deps
        self._timer = deps.timer or time.monotonic
        # A judge's night only (``settings.demo``): each person's background study, the MIND loop's
        # commitment made in phase 1, and the phase-1 note. Empty on every other night.
        self._studies: dict[str, _Study] = {}
        self._early: dict[str, _Focus] = {}
        self._written: dict[str, _Written] = {}
        self.observer_errors: list[str] = []  # each observer failure's step and error type (the view only)

    # ── the batch ────────────────────────────────────────────────────────────

    def run_batch(
        self,
        nights: Mapping[str, date],
        windows: Mapping[str, Window] | None = None,
        days: Mapping[str, DayRun] | None = None,
        sweep: Sweep | None = None,
        as_of: Mapping[str, datetime] | None = None,
    ) -> dict[str, NightResult]:
        """The batch as one phase: Glimmer goes back to sleep when it ends, however it ends."""
        with phase(self.deps.gpu_stop or self.deps.sleep):
            return self._run_batch(nights, windows, days, sweep, as_of or {})

    def _run_batch(
        self,
        nights: Mapping[str, date],
        windows: Mapping[str, Window] | None,
        days: Mapping[str, DayRun] | None,
        sweep: Sweep | None,
        as_of: Mapping[str, datetime],
    ) -> dict[str, NightResult]:
        days = days or {}
        channel_night.check(self.deps, nights)  # the thought channel's ports, before any step
        windows = {**(windows or {}), **{uid: day.window for uid, day in days.items()}}
        people = []
        for uid, night in nights.items():
            window = windows.get(uid) or day_window(night)
            people.append(self._start(uid, night, window, as_of.get(uid) or window[1]))
        for p in people:
            if sweep is not None:  # what the command's start swept for this batch, as counts
                p.log.step(
                    "sweep",
                    "skipped" if sweep.sessions_skipped else "ok",
                    uses="new",
                    system=None,
                    holds_settled=sweep.holds_settled,
                    sessions_stopped=sweep.sessions_stopped,
                    sessions_not_ours=sweep.sessions_not_ours,
                    **({"sessions_skipped": sweep.sessions_skipped} if sweep.sessions_skipped else {}),
                )
            if (
                p.uid in days
            ):  # a short day (a turn Glimmer couldn't answer) is recorded, never an error
                day = days[p.uid]
                if (
                    day.failure is not None
                ):  # the day never started: that is a failure, not a short day
                    p.failed_at, p.status = "simulate", "failed"
                    p.log.step(
                        "simulate",
                        "failed",
                        uses="existing",
                        system="chat route",
                        error=day.failure,
                        wake_seconds=day.wake_seconds,
                    )
                    continue
                extra: dict[str, Any] = (
                    {} if day.wake_seconds is None else {"wake_seconds": day.wake_seconds}
                )
                p.log.step(
                    "simulate",
                    "ok",
                    uses="existing",
                    system="chat route",
                    answered=day.answered,
                    failed_turns=day.failed,
                    short_day=day.short,
                    **extra,
                )
            for step, fn in (
                ("close_day", self._close_day),
                ("extract", self._extract),
                *((("sleep", self._sleep),) if self.deps.sleep_policy else ()),
                ("notice", self._notice),
                *self._dreams(p),
                ("probes", self._probes),
                ("choose", self._choose),
                *((("inquire", self._quiet_inquire),) if self.deps.settings.quiet_inquiry and self._mind() else ()),
                *((("agent", self._agent),) if self.deps.baseline is not None else ()),
                ("brief", self._brief),
                *self._phase_one(),
            ):
                if (why := put_off(self.deps, p, step) or self._demo_skips(step)) is not None:
                    self._skip(p, step, why)
                    continue
                self._do(p, step, fn)
        due = [p for p in people if p.due and p.failed_at is None]
        if due:
            self._gpu_job(due)
        for p in due:
            self._do(p, "promote", self._promote)
        results = {}
        for p in people:
            if p.failed_at is None and not p.due:
                why = (
                    put_off(self.deps, p, GPU_STEPS[0])
                    or "below min_new_examples: no practice, training, gates or promotion"
                )
                p.status = p.status or "nothing_new"
                for step in GPU_STEPS:
                    self._skip(p, step, why)
            if self.deps.settings.demo is None:
                self._do(p, "note", self._note)
            else:  # a judge's note was written in phase 1: the gates' verdict may rewrite it
                self._do(p, "refine", self._refine)
            results[p.uid] = self._finish(p)
        return results

    # ── a judge's night (``night/demo.py``) ─────────────────────────────────

    def _phase_one(self) -> tuple[tuple[str, Callable[[PersonNight], Mapping[str, Any]]], ...]:
        """The steps only a judge's night adds to phase 1, after the brief: the background study's end,
        MIND's commitment about her own practice, and the note. None on any other night."""
        if self.deps.settings.demo is None:
            return ()
        return (("homework", self._homework), ("self_predict", self._self_predict), ("note", self._early_note))

    def _demo_skips(self, step: str) -> str | None:
        """Why a judge's night skips ``step`` (only the probes, unless its profile writes them)."""
        demo = self.deps.settings.demo
        return PROBES_SKIPPED if demo is not None and step == "probes" and not demo.probes else None

    def _homework(self, p: PersonNight) -> Mapping[str, Any]:
        """Wait for the study ``choose`` started in the background, and keep what it found."""
        pending = self._studies.pop(p.uid, None)
        if pending is None:
            return {"noop": True, "why": "no study was chosen tonight"}
        outcome = pending.future.result()  # the call catches its own failures (``_study_call``)
        done = dict(self._study_end(p, pending.target, outcome, pending.calls_before))
        done["receipt"] = receipt.record(pending.choice, study_outcome(outcome.stopped))
        return done

    def _self_predict(self, p: PersonNight) -> Mapping[str, Any]:
        """MIND's loop commits its prediction about her own practice in phase 1, before practice
        (which then resolves it); a night that won't practise commits nothing."""
        if p.napped or not p.due:
            return {"noop": True, "why": "no practice tonight, so nothing to predict"}
        loop = self._early[p.uid] = self._focus(p)
        if loop.record is None:
            return {"noop": True, "why": "MIND's loop is off"}
        return {"loop": loop.record, "committed": [
            {"target_id": c.target_id, "label": c.label, "prediction_id": c.prediction_id, "claim": c.claim}
            for c in loop.commitment.predictions]}

    def _early_note(self, p: PersonNight) -> Mapping[str, Any]:
        """The note at the end of phase 1, Glimmer woken first (her writer is Glimmer). A wake that
        fails leaves the note's own write to fail and be recorded; it never fails the night."""
        woke: dict[str, Any] = {}
        if self.deps.wake is not None:
            try:
                p.wake_seconds = self.deps.wake().seconds
            except Exception as error:  # the note records its own failure; the night goes on to practice
                woke = {"wake_error": note_error("wake", error)}
        return {**self._note(p), **woke}

    def _refine(self, p: PersonNight) -> Mapping[str, Any]:
        """After the gates: a kept night's note is written again by the promoted adapter; a rejected
        night's phase-1 note gets the held-back line; any other night's note stands. A rewrite that fails
        or is declined keeps the phase-1 note, and never fails the night."""
        kept = self._written.get(p.uid)
        if kept is None:
            return {"noop": True, "why": "no note was written in phase 1"}
        if p.status == "rejected":
            text = f"{kept.note.text}\n\n{HELD_BACK_LINE}"
            return self._requeue(p, kept, text, "held_back_line")
        if p.status != "promoted":
            return {"noop": True, "why": "nothing was promoted tonight: the phase-1 note stands"}
        interim = NightResult(status="promoted", failed_at=None, checkpoint=p.checkpoint, gates=p.gates,
                              choices=tuple(p.choices), cost_usd=0.0, cost_parts={}, log_path=p.log.path)
        rate, started = self.deps.prices.serve_gpu_per_hour, self._timer()
        try:
            with held(self.deps.ledger, self._job(p, "refine"), ("modal_usd",), usd(rate * NOTE_MAX_S / 3600.0),
                      lambda: rate * max(0.0, self._timer() - started) / 3600.0):
                note, outcome = write_note(p.uid, kept.choice, interim, writer=self.deps.note_writer, topic=kept.topic,
                                           facts=kept.facts, night=p.night, check=self.deps.note_check)
        except Exception as error:  # the phase-1 note stands
            return _kept_note(kept, note_error("rewrite", error))
        if note is None:
            return {"outcome": "kept", "note_id": kept.note.id, "why": f"the rewrite was {outcome}"}
        return self._requeue(p, replace(kept, note=note), note.text, "refined")

    def _requeue(self, p: PersonNight, kept: _Written, text: str, outcome: str) -> Mapping[str, Any]:
        try:
            self.deps.notifier.queue(p.uid, text, NOTE_SOURCE)
        except Exception as error:  # the phase-1 note was queued already and stands
            return _kept_note(kept, note_error("queue", error))
        self._written[p.uid] = replace(kept, note=replace(kept.note, text=text))
        return {"outcome": outcome, "note_id": kept.note.id, "sources": len(kept.note.sources)}

    def _gpu_job(self, due: list[PersonNight]) -> None:
        """Practice, dataset, training and gates for the due people, in one GPU session."""
        if not self._wake(due):
            return
        uids = [p.uid for p in due]
        if len(uids) > 1:  # a shared session: each one's share of its spend (``past.her_share``)
            for p in due:
                p.gpu_batch = (batch_digest(uids), len(uids))
        try:
            with self.deps.gpu.open(uids):
                for p in due:
                    for step, fn in (
                        ("practice", self._practice),
                        *((("harness", self._harness),) if harness_on(os.environ) else ()),
                        ("dataset", self._dataset),
                        ("train", self._train),
                        ("gates", self._gates),
                    ):
                        self._do(p, step, fn)
        except Exception as error:  # the session itself failed: every person it held stops
            self._fail_ungated(due, error)

    def _wake(self, due: list[PersonNight]) -> bool:
        """Wake Glimmer before the first Modal step. The seconds go in each night's summary; a
        server that can't be woken fails these nights at practice with a code, not a message."""
        wake = self.deps.wake
        if wake is None:
            return True
        try:
            woke = wake()
        except (
            Exception
        ) as error:  # a bad URL or token: a recorded failure for each person, not an aborted batch
            for p in due:
                self._fail(p, "practice", error)
            return False
        for p in due:
            p.wake_seconds = woke.seconds
        if woke.ok:
            return True
        for p in due:
            self._fail(p, "practice", GlimmerNotReady())
        return False

    def _wake_again(self, p: PersonNight) -> float | None:
        """Wake Glimmer again right before the gates; a server that stays cold fails this person."""
        if self.deps.wake is None:
            return None
        woke = self.deps.wake()
        p.gates_wake_seconds = woke.seconds
        if not woke.ok:
            raise GlimmerNotReady()
        return woke.seconds

    def _fail_ungated(self, due: list[PersonNight], error: Exception) -> None:
        for p in due:
            if p.failed_at is None and not p.gated:
                self._fail(p, "practice", error)

    # ── step plumbing ────────────────────────────────────────────────────────

    def _start(self, uid: str, night: date, window: Window, as_of: datetime) -> PersonNight:
        deps = self.deps
        deps.cancel(
            uid
        )  # before any state loads: the first look at the flag is the baseline a change is read against
        log = NightLog(night_folder(deps.logs(uid), night))
        if (
            deps.gpu_stop is not None
        ):  # when the batch's outermost phase ends, the stop's outcome or why it skipped
            deps.gpu_stop.watch(
                lambda status, fields: log.step(
                    "gpu_stop", status, uses="new", system=None, **fields
                )
            )
        history = summaries(deps.logs(uid))
        actions_left = min(
            MAX_NIGHT_ACTIONS,
            max(
                0,
                deps.settings.rolling_actions
                - paid_within(history, window[1], timedelta(hours=24)),
            ),
        )
        wonder_spent = spent_within(history, window[1], timedelta(days=30))
        return PersonNight(
            uid=uid,
            night=night,
            window=window,
            log=log,
            spend=SpendMeter(
                spark_in_per_m=deps.prices.spark_in_per_m,
                spark_out_per_m=deps.prices.spark_out_per_m,
            ),
            wonder=WonderMeter(
                deps.settings.homework_call_cap,
                max(0.0, deps.settings.wonder_usd_month - wonder_spent),
            ),
            actions_left=actions_left,
            store=deps.evidence(uid),
            targets=deps.targets(uid),
            as_of=as_of,
            wonder_month_spent=wonder_spent,
        )

    def _do(
        self, p: PersonNight, step: str, fn: Callable[[PersonNight], Mapping[str, Any]]
    ) -> bool:
        if p.failed_at is not None:
            return False
        if self.deps.cancel(p.uid):
            return self._fail(p, step, Cancelled("the night was cancelled"))
        self._observe(p, step, "running", {})
        try:
            receipt = dict(fn(p))
        except Exception as error:  # a failure stops this person's night at this step (RF1)
            return False if self._cancelled_mid_step(p, step) else self._fail(p, step, error)
        if self._cancelled_mid_step(p, step):
            return False
        noop = bool(receipt.pop("noop", False))
        uses, system = USES[
            step
        ]  # a step that kept what it had failed, and the night goes on (``kept_brief``)
        p.log.step(
            step,
            "failed" if KEPT in receipt else "ok",
            uses=uses,
            system=system,
            noop=noop,
            **receipt,
        )
        p.receipts[step] = receipt
        self._observe(p, step, "failed" if KEPT in receipt else "ok", {**receipt, "noop": noop})
        return True

    def _observe(self, p: PersonNight, step: str, state: str, receipt: Mapping[str, Any]) -> None:
        """Tell the observer (a judge's night view) a step's state. It never changes the night: an
        observer that fails is logged and the night goes on."""
        observer = self.deps.observer
        if observer is None:
            return
        try:
            observer(p.uid, step, state, receipt)
        except Exception as error:  # the view is optional; the night is not
            self.observer_errors.append(note_error(step, error))
            logger.warning("the night's observer failed at %s (%s)", step, type(error).__name__)

    def _cancelled_mid_step(self, p: PersonNight, step: str) -> bool:
        """The person was deleted while ``step`` ran (RF2): remove whatever it wrote, record nothing
        (the night log's folder is gone with the delete), and stop this person's night."""
        if not self.deps.cancel(p.uid):
            return False
        self.deps.purge(p.uid)
        p.failed_at, p.status = step, "failed"
        return True

    def _fail(self, p: PersonNight, step: str, error: BaseException) -> bool:
        p.failed_at, p.status = step, "failed"
        uses, system = USES[step]
        p.log.step(step, "failed", uses=uses, system=system, error=safe_error(error))
        self._observe(p, step, "failed", {"error": safe_error(error)})
        return False

    def _skip(self, p: PersonNight, step: str, why: str) -> None:
        uses, system = USES[step]
        p.log.step(step, "skipped", uses=uses, system=system, why=why)
        self._observe(p, step, "skipped", {"why": why})

    # ── steps 1 to 4 and the brief ──────────────────────────────────────────

    def _job(self, p: PersonNight, step: str) -> str:
        return f"night:{step}:{p.uid}:{p.night.isoformat()}"

    def _metered(
        self, p: PersonNight, part: CostPart, step: str, calls: int, out: int = SPARK_MAX_OUT_TOKENS
    ) -> Any:
        """Hold ``step``'s Spark Open worst case (``calls``, each writing up to ``out``) on
        ``baseten_sleep_usd``; settle what ``part`` metered."""
        before = p.spend.parts().get(part, 0.0)
        return held(
            self.deps.ledger,
            self._job(p, step),
            ("baseten_sleep_usd",),
            usd(spark_worst_usd(self.deps.prices, calls, out)),
            lambda: p.spend.parts().get(part, 0.0) - before,
        )

    def _close_day(self, p: PersonNight) -> Mapping[str, Any]:
        start, end = p.window
        if end <= start:
            raise ValueError("a night's window ends after it starts")
        return {"window": [start.isoformat(), end.isoformat()], "night": p.night.isoformat()}

    def _extract(self, p: PersonNight) -> Mapping[str, Any]:
        p.spend.part = "extract"
        source = self.deps.extractor(p.spend.wrap_extract(self.deps.extract_chat))
        with self._metered(
            p,
            "extract",
            "extract",
            EXTRACT_MAX_CALLS * (1 + len(owed := owed_tonight(self.deps, p))),
        ):
            catch_up(
                source, p, owed
            )  # the days capped naps left, oldest first (sleep_night), then tonight's
            items = source.extract_day(p.uid, p.night, p.window, p.as_of)
        since = self._since(p)
        fresh = [
            item
            for item in p.store.trainable(p.night)
            if item.kind != "practice" and (since is None or item.day > since)
        ]
        p.due = len(fresh) >= self.deps.settings.min_new_examples
        kinds: dict[str, int] = {}
        for item in items:
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
        return {
            "items": len(items),
            "kinds": kinds,
            "new_since_last_training": len(fresh),
            "min_new_examples": self.deps.settings.min_new_examples,
            "training_night": p.due,
        }

    def _since(self, p: PersonNight) -> date | None:
        """The last trained night: the night of the checkpoint that serves, which the next night
        trains from. A failed or rejected night's training never served, so its evidence stays new."""
        serving = self.deps.registry.last_promoted(p.uid)
        return None if serving is None else serving.night

    def _mind(self) -> bool:
        """MIND's inquiry runs only with the setting on (Task 17); off, the night is Task 7a's."""
        return mind_on(self.deps)

    def _loop(self) -> bool:
        """The MIND loop (practice draws her focus, and she predicts her own practice): on only with
        ``ANJO_SLEEP_MIND_LOOP=1`` and MIND's inquiry on. Off, every night is byte-identical."""
        return loop_focus.loop_on(os.environ) and self._mind()

    def _notice(self, p: PersonNight) -> Mapping[str, Any]:
        notice = self.deps.notice
        if not (self.deps.settings.mind_inquiry and notice.available):
            return {"noop": True, "port": "notice (Task 17): no-op"}
        done, since = notice.notice(self._call(p, "notice"), unnoticed(p)), p.notice_from
        return (
            {**done, "notice_from": since.isoformat()}
            if since is not None and since < p.night
            else done
        )  # after naps

    def _sleep(self, p: PersonNight) -> Mapping[str, Any]:
        """Nap or deep sleep (``sleep_night.py``): a nap wakes no Glimmer and trains nothing."""
        return sleep_step(self.deps, p)

    def _probes(self, p: PersonNight) -> Mapping[str, Any]:
        if self.deps.is_synthetic(p.uid):  # sealed once: nothing is called tonight
            sealed, how = self.deps.probes.register(p.uid, p.night, p.store)
        else:  # Muse Spark writes them, unmetered: the hold is settled whole
            with held(
                self.deps.ledger,
                self._job(p, "probes"),
                ("meta_usd",),
                usd(PROBES_WORST_USD),
                lambda: PROBES_WORST_USD,
            ):
                sealed, how = self.deps.probes.register(p.uid, p.night, p.store)
        p.sealed = list(sealed)
        return {"sealed": len(p.sealed), "how": how}

    def _raw(self, p: PersonNight) -> points.RawSnapshot:
        """Her state's raw input to ``snapshot.build`` (Task 7a), on the person's night clock, before
        any eligibility filter. With ``mind_inquiry`` on (Task 17) her questions come from her
        workspace, after the close marks tonight's closes owe; else none."""
        own = [item.created_at for item in p.store.stored(p.night) if item.kind in PERSON_KINDS]
        if self._mind():
            self.deps.inquiry.settle(self._call(p, "choose"))
        targets = p.targets.open_targets(p.as_of)
        questions = self.deps.inquiry.questions(self._call(p, "choose")) if self._mind() else ()
        return points.RawSnapshot(
            tuple(targets),
            self._results(p),
            p.as_of,
            max(own) if own else None,
            intentions=self.deps.intentions.intentions(p.uid, p.as_of),
            questions=questions,
        )

    def _point(
        self,
        p: PersonNight,
        phase: points.Phase,
        raw: points.RawSnapshot,
        ctx: ChoiceContext,
        choice: Choice | None,
    ) -> None:
        """Task 16: every night of a synthetic life records both offers' decision points."""
        if self.deps.is_synthetic(p.uid):
            p.log.point(points.point_record(points.DecisionPoint(p.night, phase, raw, ctx, choice)))

    def _expect(self, p: PersonNight, raw: points.RawSnapshot, choice: Choice) -> str | None:
        """Task 16: a chosen study records how likely she thinks the person is to need it."""
        if self.deps.estimator is None or choice.want is None or not self.deps.is_synthetic(p.uid):
            return None
        if choice.want.kind not in expect.STUDY_KINDS:
            return None  # the registered estimator was fit on the person's pulls, not her own
        if choice.action == "inquire" and all(
            q.view.id != choice.question_id for q in raw.questions
        ):
            return None  # a question the snapshot doesn't hold has no first target to expect on
        target_id = points.study_target(choice, raw)
        target = next((t for t in raw.targets if t.id == target_id), None)
        if target is None:
            return None
        made = expect.expectation(target, choice.want.kind, p.as_of, self.deps.estimator)
        if all(
            kept.id != made.id for kept in p.targets.expectations()
        ):  # a rerun of the night records it once
            p.targets.record_expectation(made)
        return made.id

    def _choose(self, p: PersonNight) -> Mapping[str, Any]:
        deps = self.deps
        raw = self._raw(p)
        offered = offered_tonight(deps, p)
        state = points.build(raw, inquiry_offered="inquire" in offered, own_wants=own_wants_on(os.environ))
        ctx = ChoiceContext(
            policy=deps.policy,
            allowance=p.wonder.allowance(p.actions_left),
            allowed_actions=offered,
        )
        choice = deps.chooser.choose("away", deps.chooser.derive(state), ctx)
        p.choices.append(choice)
        self._point(p, "pre_study", raw, ctx, choice)
        action = choice.action
        cap = deps.settings.wonder_usd_month
        month = {
            "cap_usd": cap,
            "spent_usd": round(p.wonder_month_spent, 6),
            "left_usd": round(max(0.0, cap - p.wonder_month_spent), 6),
        }
        homework = (
            (CAPPED_WHY if p.capped else "available")
            if deps.homework.available
            else "no-op: no study offered"
        )
        if deps.homework.available and month["left_usd"] <= 0:
            homework = f"capped: the monthly Wonder cap of ${cap:.2f} is spent, so no study runs"
        done: dict[str, Any] = {
            "action": action,
            "target_id": choice.target_id,
            "offered": sorted(offered),
            "allowance": {
                "actions_left": p.actions_left,
                "calls_left": p.wonder.calls_left,
                "usd_left": p.wonder.usd_left,
            },
            "wonder_month": month,
            "ports": {
                "chooser": type(deps.chooser).__name__,
                "homework": homework,
                "inquiry": (DEFER_WHY if p.napped else "available")
                if self._mind()
                else "no-op (Task 17)",
                "expectations": "registered"
                if deps.estimator
                else "estimator not registered (Task 16)",
            },
            "not_offered": sorted(
                {c.want_id for c in choice.considered if c.reason == "action_not_offered"}
            ),
            "expectation": self._expect(p, raw, choice),
        }
        outcome: receipt.Outcome | None = None
        if action in STUDY_ACTIONS and deps.settings.demo is not None:  # a judge's: the study runs alongside
            done.update(self._study_start(p, state, choice))
        elif action in STUDY_ACTIONS:
            done.update(self._study(p, state, choice))
            outcome = study_outcome(p.study.stopped if p.study is not None else None)
        elif action == "inquire":
            done.update(self._inquire(p, choice))
            outcome = study_outcome(p.inquiry.stopped if p.inquiry is not None else None)
        elif action == "hold_note":
            p.note_choice = choice  # written at step 10, whose receipt records the outcome
        done["receipt"] = receipt.record(choice, outcome)
        return done

    def _study(self, p: PersonNight, state: WantSnapshot, choice: Choice) -> Mapping[str, Any]:
        target = next(t for t in state.targets if t.id == choice.target_id)
        p.actions_left -= 1  # a study that started is a spent action, even if it then fails
        p.paid_actions += 1
        calls_before = p.wonder.calls
        return self._study_end(p, target, self._study_call(p, target, choice), calls_before)

    def _study_start(self, p: PersonNight, state: WantSnapshot, choice: Choice) -> Mapping[str, Any]:
        """A judge's study, started in the background (only the Wonder calls run there; what it found
        is kept on this thread at ``homework``). The action is spent now, as ``_study`` spends it."""
        target = next(t for t in state.targets if t.id == choice.target_id)
        p.actions_left -= 1
        p.paid_actions += 1
        calls_before = p.wonder.calls
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="night-study")
        future = pool.submit(self._study_call, p, target, choice)
        pool.shutdown(wait=False)  # the thread finishes its one call; ``homework`` (or ``_finish``) waits
        self._studies[p.uid] = _Study(target, choice, calls_before, future)
        self._observe(p, "homework", "running", {"background": True})
        return {"study": {"background": True}}

    def _study_call(self, p: PersonNight, target: Target, choice: Choice) -> StudyOutcome:
        """The study's Wonder calls, its failures caught: they stop the study, never the night."""
        try:
            return self.deps.homework.study(
                p.uid, target, p.night, p.wonder, action=choice.action or "", at=p.stamp
            )
        except WonderSpent:
            return StudyOutcome(result=None, stopped="budget")
        except STUDY_FAILURES as error:  # the study failed, not the night: its calls stay debited
            return StudyOutcome(
                result=None, stopped="failed", receipt={"error": type(error).__name__}
            )

    def _study_end(
        self, p: PersonNight, target: Target, outcome: StudyOutcome, calls_before: int
    ) -> Mapping[str, Any]:
        if (
            outcome.stopped is not None and p.wonder.calls == calls_before
        ):  # it never started: no action spent
            p.actions_left += 1
            p.paid_actions -= 1
        if (
            outcome.stopped is None
        ):  # a study the budget stopped, or that failed, didn't run: the want stays
            p.targets.mark_acted(target.id, target.revision)
        if outcome.evidence:
            p.store.add(list(outcome.evidence))
        if outcome.result is not None:
            p.targets.add_result(outcome.result)
        p.study = outcome
        return {
            "study": {
                **outcome.receipt,
                "result": outcome.result.id if outcome.result else None,
                "stopped": outcome.stopped,
                "homework_items": len(outcome.evidence),
                "wonder_calls": p.wonder.calls,
                "wonder_usd": p.wonder.usd,
            }
        }

    def _quiet_inquire(self, p: PersonNight) -> Mapping[str, Any]:
        """An alone night still visits the controller when the chooser found no live question."""
        if p.napped:
            return {"deferred": True, "why": DEFER_WHY}  # the monthly cap still wins
        if p.inquiry is not None:
            return {"already_ran": True}
        return self._inquire(p, None)

    def _inquire(self, p: PersonNight, choice: Choice | None) -> Mapping[str, Any]:
        if choice is not None and choice.question_id is None:
            raise ValueError("an inquire choice names its question")
        outcome = self.deps.inquiry.inquire(self._call(p, "inquire"), None if choice is None else choice.question_id)
        if outcome.result is not None:
            p.targets.add_result(outcome.result)
        if outcome.research_calls > 0:  # only research is paid; thinker cycles are her own Glimmer
            p.actions_left -= 1
            p.paid_actions += 1
        p.inquiry = outcome
        return {
            "inquiry": {
                "cycles": outcome.cycles,
                "research_calls": outcome.research_calls,
                "result": outcome.result.id if outcome.result else None,
                **outcome.receipt,
            }
        }

    def _agent(self, p: PersonNight) -> Mapping[str, Any]:
        """The comparison's conventional agent (arm 2A): one stateless pass after MIND's step.

        It reads tonight's evidence and the served brief, and leaves notes only arm 2A reads. It
        never naps: a scheduled agent has no sleep pressure, and its own hold on
        ``baseten_sleep_usd`` (the ledger remaps it when Sleep's Spark Open is switched) is what
        stops it when the comparison's credit is spent."""
        baseline = self.deps.baseline
        if baseline is None:
            raise RuntimeError("the night has no baseline agent to run")
        latest = self.deps.briefs(p.uid).latest()
        outcome = baseline.night(AgentCall(
            user_id=p.uid, night=p.night, as_of=p.as_of, at=p.stamp, store=p.store, wonder=p.wonder,
            spend=p.spend, job=self._job(p, "agent"), brief=None if latest is None else latest.served,
            cancelled=lambda: self.deps.cancel(p.uid),
        ))
        p.agent = outcome
        return {"notes": len(outcome.notes), "tokens_used": outcome.tokens_used, "calls": outcome.calls,
                "usd": round(outcome.usd, 6), "stopped": outcome.stopped}

    def _brief(self, p: PersonNight) -> Mapping[str, Any]:
        briefs = self.deps.briefs(p.uid)
        new = briefs.unfolded(p.store.stored(p.night))
        if not new:
            return {"folded": 0, "why": "no evidence since the last fold"}
        p.spend.part = "brief"
        try:
            with self._metered(p, "brief", "brief", BRIEF_MAX_CALLS, BRIEF_MAX_TOKENS):
                folded = briefs.fold(
                    p.night,
                    new,
                    at=p.stamp,
                    hosted_in_training=self.deps.hosted_in_training,
                    chat=p.spend.wrap_chat(self.deps.spark_chat),
                    mind_refresh=self._mind() and any(
                        item.kind == "homework" and item.id.startswith("mind-") for item in new
                    ),
                )
        except (
            BriefError
        ) as error:  # written badly twice: nothing is stored, and she keeps the briefs she had
            latest = briefs.latest()
            return kept_brief(None if latest is None else latest.night, len(new), error)
        return {"folded": len(folded.folded)}

    # ── the GPU steps ────────────────────────────────────────────────────────

    def _serving(self, p: PersonNight, part: str, started: tuple[float, int]) -> dict[str, Any]:
        seconds, retries = max(0.0, self._timer() - started[0]), self.deps.retries() - started[1]
        p.spend.add(part, seconds / 3600.0 * self.deps.prices.serve_gpu_per_hour)  # type: ignore[arg-type]
        return {
            "seconds": seconds,
            **({"retries": retries} if retries else {}),
        }  # requests sent again: a count

    def _practice(self, p: PersonNight) -> Mapping[str, Any]:
        kept = self.deps.transcripts(p.uid).for_night(p.night)
        if kept:  # a rerun of the night reuses the practice it already paid for
            p.store.add([transcript.evidence() for transcript in kept])
            if self._loop():  # a rerun skips resolution: close what the earlier attempt left pending
                self._censor(p, self._call(p, "practice"), self_predict.record(p.night, kept),
                             "practice was reused, so the loop could not resolve it")
            return {
                "scenes": len(kept),
                "reused": True,
                "reused_from": p.night.isoformat(),
                "homework": None,
            }
        started = self._timer(), self.deps.retries()
        p.spend.part = "practice"
        runner, channel = channel_night.practice(
            self.deps, p.uid, p.spend.wrap_chat(self.deps.spark_chat), p.night
        )
        # the MIND loop's commitments, before any scene exists (a judge's were made in phase 1)
        loop = self._early.pop(p.uid) if p.uid in self._early else self._focus(p)
        scene_kwargs = {"focus": loop.targets} if loop.targets else {}  # off, practice is called as today
        with self._metered(p, "practice", "practice", runner.max_spark_calls + (1 if loop.targets else 0)):
            run = runner.run(p.uid, p.night, seeded := self._seeded(p), None, at=p.stamp, **scene_kwargs)
        p.spend.add(
            "practice", run.critic_usd
        )  # a third-party critic's, held per call on its own pool
        self.deps.transcripts(p.uid).add(list(run.transcripts))
        p.store.add([transcript.evidence() for transcript in run.transcripts])
        if loop.record is not None and loop.call is not None:  # resolved from this night's record, by code
            loop = replace(loop, record=self._resolved(p, run, loop))
        share = run.teacher_targets / len(run.transcripts) if run.transcripts else 0.0
        return {
            "scenes": len(run.transcripts),
            "dropped_cued": run.dropped_cued,
            "dropped_cut": run.dropped_cut,
            "f_items": len(run.f_ids),
            "pairs": run.pairs,
            "teacher_targets": run.teacher_targets,
            "critic_unreadable": run.critic_unreadable,
            "targets": {
                "teacher": run.teacher_targets,
                "glimmer": len(run.transcripts) - run.teacher_targets,
                "teacher_share": round(share, 4),
            },
            "teacher_alarm": share
            > TEACHER_ALARM_SHARE,  # the critic alone promoted the teacher this often
            "homework": None,
            **run.spend_record(),
            **channel,
            **self._seed_record(p, seeded),
            **self._serving(p, "practice", started),
            **({} if loop.record is None else {"loop": loop.record}),
        }

    def _harness(self, p: PersonNight) -> Mapping[str, Any]:
        """Harness learning, after practice: one vetted strategy edit, probed contrastively.

        Off, the step never runs. On, it never fails the night: a failed proposal, vet or probe is
        logged by its error's type, the list stays as it was, and dataset and training go on. Its
        Spark Open calls go through the night's metered client and its Glimmer probes through the
        serving GPU the night already holds, both charged to the practice part."""
        strategies, glimmer = self.deps.strategies, self.deps.glimmer
        if strategies is None or glimmer is None:
            return {"noop": True, "why": "the harness switch is on, but its ports aren't wired"}
        before = p.spend.parts().get("practice", 0.0)
        started = self._timer(), self.deps.retries()
        p.spend.part = "practice"
        store = None
        calls = 0
        metered = p.spend.wrap_chat(self.deps.spark_chat)

        def counted(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return metered(*args, **kwargs)

        try:
            store = strategies(p.uid)
            call = HarnessCall(
                user_id=p.uid, night=p.night, seed=self.deps.settings.seed, at=p.stamp, store=store,
                transcripts=self.deps.transcripts(p.uid).for_night(p.night), glimmer=glimmer,
                chat=counted, adapter=self.deps.registry.last_promoted(p.uid),
                cancelled=lambda: self.deps.cancel(p.uid),
            )
            with self._metered(p, "practice", "harness", HARNESS_MAX_CALLS):
                done = run_harness(call)
        except Cancelled:  # a cancelled night stops; only a harness failure is swallowed
            raise
        except HARNESS_FAILURES as error:  # the harness failed, not the night: the list stays
            logger.warning("harness: the loop didn't finish (%s); the list stays", type(error).__name__)
            size = 0
            if store is not None:
                try:
                    size = len(store.lines(until=p.night))
                except HARNESS_FAILURES:
                    pass  # the log can't be read either: the receipt still says no edit
            served = self._serving(p, "practice", started)
            return {
                "error": safe_error(error), "why": "the harness failed; the list stays",
                "promoted": False, "list_size": size, "calls": calls,
                "usd": round(p.spend.parts().get("practice", 0.0) - before, 6), **served,
            }
        served = self._serving(p, "practice", started)
        return {
            **done, "usd": round(p.spend.parts().get("practice", 0.0) - before, 6), **served,
        }

    def _focus(self, p: PersonNight) -> _Focus:
        """MIND's share of practice (``ANJO_SLEEP_MIND_LOOP=1`` with MIND on): the targets her own
        questions are working on, and the prediction she commits about herself before any scene is
        written. Off, practice is today's and the record carries nothing. A loop that can't read her
        focus or commit does the same and says why; a commitment it already made is censored."""
        if not self._loop():
            return _Focus()
        call = self._call(p, "practice")
        history: tuple[Mapping[str, object], ...] = ()
        try:
            history = tuple(summaries(self.deps.logs(p.uid)))
            found = self.deps.inquiry.focus(call)
            committed = (self.deps.inquiry.self_predict(call, found.targets, history) if found.targets
                         else self_predict.Commitment())
        except LOOP_FAILURES as error:  # the loop failed, not the night: practice is today's
            logger.warning("MIND's loop didn't start practice (%s); practice is today's", type(error).__name__)
            why = f"the loop failed before practice ({type(error).__name__})"
            record = self_predict.record(p.night, ())
            return _Focus((), self_predict.receipt(
                (), record, self_predict.Commitment(), self._censor(p, call, record, why), why=why),
                call, self_predict.Commitment(), ())
        return _Focus(found.targets, self_predict.receipt(
            found.targets, self_predict.record(p.night, ()), committed, (), why=found.why), call, committed, history)

    def _resolved(self, p: PersonNight, run: PracticeRun, loop: _Focus) -> Mapping[str, Any]:
        """Resolve her prediction on each focus target from the night's practice record: the counts,
        the baseline its claim named and the verdict, which moves her credence through the kernel. A
        resolution that fails censors what it left pending and still records the night's practice."""
        assert loop.call is not None
        record = self_predict.record(p.night, run.transcripts)
        try:
            outcomes = (self.deps.inquiry.self_resolve(loop.call, loop.commitment.predictions, record, loop.history)
                        if loop.commitment.predictions else ())
        except LOOP_FAILURES as error:  # the loop failed, not the night: close what it left pending
            logger.warning("MIND's loop didn't resolve practice (%s)", type(error).__name__)
            outcomes = self._censor(p, loop.call, record,
                                    f"the loop couldn't resolve it ({type(error).__name__})",
                                    loop.commitment.predictions)
        why = None if loop.record is None else loop.record.get("why")
        return self_predict.receipt(loop.targets, record, loop.commitment, outcomes,
                                    why=None if why is None else str(why))

    def _censor(
        self, p: PersonNight, call: MindCall, record: self_predict.Record, why: str,
        predictions: Sequence[self_predict.SelfPrediction] = (),
    ) -> tuple[self_predict.Outcome, ...]:
        """Settle the loop's unresolved predictions as censored, so MIND's inquiry resolver never
        settles one from unrelated evidence. A censor that fails leaves them pending, which still
        moves no credence, and never fails the night."""
        try:
            return self.deps.inquiry.self_censor(call, predictions, record, why)
        except LOOP_FAILURES as error:
            logger.warning("MIND's loop couldn't censor its predictions (%s)", type(error).__name__)
            return ()

    def _dataset(self, p: PersonNight) -> Mapping[str, Any]:
        settings = self.deps.settings
        data, probe_dropped = build_night_dataset(
            p.store,
            self._since(p),
            p.night,
            p.sealed,
            self.deps.decontam,
            settings.token_cap,
            transcripts=self.deps.transcripts(p.uid),
            seed=settings.seed,
            dropout=settings.dropout,
            hosted_in_training=self.deps.hosted_in_training,
        )
        if not data.examples:
            raise ValueError("no training example survived the dataset's checks")
        p.data = data = self._weighted(p, data)
        done = {
            "examples": len(data.examples),
            "tokens": data.tokens,
            "token_cap": settings.token_cap,
            "new": sum(e.new for e in data.examples),
            "replay": sum(not e.new for e in data.examples),
            "context_dropped": sum(e.context_dropped for e in data.examples),
            **(
                {"mind_weighted": sum(e.weight != 1.0 for e in data.examples)}
                if p.dreamt.state_id
                else {}
            ),
            "dropped": dict(data.dropped),
            "probe_dropped": probe_dropped,
            "context_paraphrase_kept": data.context_paraphrase_kept,
            "sealed": len(p.sealed),
        }
        if self.deps.terms.seed:  # sft_dpo_seed's rows (train/terms.py); off, the night is today's
            p.data, done["terms"] = attach_night_terms(
                self.deps, p.uid, p.night, p.store, p.sealed, data, p.spend
            )
        return done

    def _train(self, p: PersonNight) -> Mapping[str, Any]:
        registry = self.deps.registry
        assert p.data is not None
        prev = registry.last_promoted(p.uid)
        number = len(registry.trained_nights(p.uid) - {p.night}) + 1
        name = fresh_adapter_name(p.uid, number, registry.trained_names(p.uid))
        registry.record_trained(p.uid, name, p.night)  # before training: a name is never reused
        ckpt, receipt = channel_night.train(self.deps, p.uid, p.night, prev, p.data, name)
        p.checkpoint = ckpt
        minutes = receipt.get("gpu_minutes")
        if isinstance(minutes, int | float) and math.isfinite(minutes):
            p.spend.add("training", float(minutes) / 60.0 * self.deps.prices.train_gpu_per_hour)
        return {
            "adapter": ckpt.id,
            "parent": ckpt.parent_id,
            "from_last_promoted": prev.id if prev else None,
            "receipt": dict(receipt),
            **self._surprise(p, receipt),
        }

    def _gates(self, p: PersonNight) -> Mapping[str, Any]:
        assert p.checkpoint is not None
        woke = self._wake_again(p)  # training takes long enough for Modal to idle out
        started = self._timer(), self.deps.retries()
        for name, check in self.deps.gates.for_person(p.uid).items():
            p.gates[name] = check(p.checkpoint)
        p.gated = True
        demo = self.deps.settings.demo
        return {
            **({} if demo is None else {"fast_gate": demo.gate_receipt()}),
            "gates": {name: gate_record(result) for name, result in p.gates.items()},
            **self._serving(p, "gates", started),
            **({} if woke is None else {"wake_seconds": woke}),
        }

    def _promote(self, p: PersonNight) -> Mapping[str, Any]:
        assert p.checkpoint is not None
        off = sorted(
            name for name, result in p.gates.items() if result.skipped
        )  # switched off: left out
        failed, noted = (
            sorted(n for n, r in p.gates.items() if not (r.passed or r.skipped)),
            {"skipped": off} if off else {},
        )
        if failed or len(off) == len(p.gates):
            p.status = "rejected"
            return {
                "promoted": None,
                "rejected_by": failed,
                "serving": checkpoint_id(self.deps.registry.last_promoted(p.uid)),
                **noted,
            }
        unit = self._promoted_unit(
            p, p.checkpoint.id
        )  # her unit first: unkept, nothing is promoted
        self.deps.registry.promote(p.uid, p.checkpoint)
        p.status = "promoted"
        return {"promoted": p.checkpoint.id, "serving": p.checkpoint.id, **noted, **unit}

    # ── step 10, the note, and step 11 ──────────────────────────────────────

    def _second_offer(self, p: PersonNight) -> Choice | None:
        verified = (
            p.study is not None
            and p.study.result is not None
            and p.study.result.verified_sources > 0
        )
        if (
            self._mind() and not p.napped
        ):  # a finding with an unconsumed anchor shares now (Task 17)
            p.shared = tuple(r.id for r in self.deps.inquiry.share(self._call(p, "note")))
        anchored = (p.inquiry is not None and p.inquiry.result is not None) or bool(p.shared)
        offer = p.note_choice is None and (verified or anchored)
        if not (offer or self.deps.is_synthetic(p.uid)):
            return None
        raw = self._raw(p)
        allowed = (
            frozenset({"hold_note"}) if offer else frozenset()
        )  # what this point really offered
        ctx = ChoiceContext(
            policy=self.deps.policy,
            allowance=p.wonder.allowance(max(0, p.actions_left)),
            allowed_actions=allowed,
        )
        if not offer:
            self._point(p, "post_result", raw, ctx, None)
            return None
        choice = self.deps.chooser.choose("away", self.deps.chooser.derive(points.build(
            raw, inquiry_offered=ctx.allows("inquire"), own_wants=own_wants_on(os.environ))), ctx)
        p.choices.append(choice)
        self._point(p, "post_result", raw, ctx, choice)
        if choice.action != "hold_note":
            p.receipts["second_offer"] = receipt.record(choice, None)
            return None
        return choice

    def _note(self, p: PersonNight) -> Mapping[str, Any]:
        done = self._held_note(p)
        return {**done, "shared": list(p.shared)} if p.shared else done

    def _held_note(self, p: PersonNight) -> Mapping[str, Any]:
        offered = self._second_offer(p)  # a step-4 note still records the post-result point
        choice = p.note_choice or offered
        if choice is None or choice.target_id is None:
            return {
                "outcome": "no_reason",
                "offered_again": "second_offer" in p.receipts,
                "second_offer_receipt": p.receipts.get("second_offer"),
            }
        if (
            p.napped
        ):  # the writer is Glimmer: the note waits, its target unsettled, for her next deep sleep
            return deferred_note(choice, choice.target_id)
        if p.actions_left < 1:
            return {"outcome": "no_allowance"}
        target_id = choice.target_id
        interim = NightResult(
            status=p.status or "nothing_new",
            failed_at=None,
            checkpoint=p.checkpoint,
            gates=p.gates,
            choices=tuple(p.choices),
            cost_usd=0.0,
            cost_parts={},
            log_path=p.log.path,
        )
        rate, started = self.deps.prices.serve_gpu_per_hour, self._timer()
        try:  # the note is optional: whatever fails here, the night keeps its status and checkpoint
            with held(
                self.deps.ledger,
                self._job(p, "note"),
                ("modal_usd",),
                usd(rate * NOTE_MAX_S / 3600.0),
                lambda: rate * max(0.0, self._timer() - started) / 3600.0,
            ):
                topic, facts = self._topic(p, target_id), self._facts(p, target_id)
                note, outcome = write_note(
                    p.uid,
                    choice,
                    interim,
                    writer=self.deps.note_writer,
                    topic=topic,
                    facts=facts,
                    night=p.night,
                    check=self.deps.note_check,
                )
        except Exception as error:
            return self._note_failed(choice, target_id, "write", error)
        if note is not None:
            try:
                self.deps.notifier.queue(p.uid, note.text, NOTE_SOURCE)
            except Exception as error:
                return self._note_failed(choice, target_id, "queue", error)
            if self.deps.settings.demo is not None:  # a judge's: the gates' verdict may rewrite it (``_refine``)
                self._written[p.uid] = _Written(choice, note, topic, tuple(facts))
        pushed, push_error = False, None
        if note is not None:
            p.actions_left -= 1
            p.paid_actions += 1
            try:
                pushed = self.deps.notifier.push(p.uid, note.text)
            except Exception as error:  # the note is queued; only its announcement failed
                push_error = note_error("push", error)
        if outcome in ("held", "declined"):
            p.targets.settle(target_id)
        done: dict[str, Any] = {
            "outcome": "note_declined" if outcome == "declined" else outcome,
            "target_id": target_id,
            "pushed": pushed,
            "second_offer": choice is not p.note_choice,
            "receipt": receipt.record(
                choice, "declined" if outcome == "declined" else "ran" if note else None
            ),
        }
        if push_error is not None:
            done["note_error"] = push_error
        if note is not None:
            done.update(
                {"note_id": note.id, "sources": len(note.sources), "held_back": note.held_back}
            )
        return done

    def _note_failed(
        self, choice: Choice, target_id: str, stage: str, error: Exception
    ) -> Mapping[str, Any]:
        """A note that couldn't be written or queued: recorded by stage and error type, never words."""
        outcome: receipt.Outcome = "budget" if isinstance(error, BudgetStop) else "failed"
        return {
            "outcome": "note_failed",
            "target_id": target_id,
            "note_error": note_error(stage, error),
            "receipt": receipt.record(choice, outcome),
        }

    def _topic(self, p: PersonNight, target_id: str) -> str:
        targets: Sequence[Target] = p.targets.open_targets(p.as_of)
        return next((t.label for t in targets if t.id == target_id), "something you are working on")

    def _results(self, p: PersonNight) -> tuple[Result, ...]:
        """The target log's results; with MIND on, a shared finding's only while it stands (Task 17)."""
        return (
            self.deps.inquiry.results(self._call(p, "choose"))
            if self._mind()
            else tuple(p.targets.results())
        )

    def _facts(self, p: PersonNight, target_id: str) -> list[Fact]:
        wanted = {
            r.homework_id for r in self._results(p) if r.target_id == target_id and not r.consumed
        }
        items: list[Evidence] = [
            e for e in p.store.stored(p.night) if e.kind == "homework" and e.id in wanted
        ]
        facts = [Fact(text=item.text, source=item.source) for item in items]
        return (
            facts + self.deps.inquiry.facts(self._call(p, "note"), target_id)
            if self._mind()
            else facts
        )

    def _finish(self, p: PersonNight) -> NightResult:
        pending = self._studies.pop(p.uid, None)  # a judge's night that stopped before ``homework``
        if pending is not None and (error := pending.future.exception()) is not None:  # waits for it to end
            logger.warning("tonight's background study failed (%s)", type(error).__name__)  # not kept either way
        parts = p.spend.parts()
        if p.agent is None:  # off: no cost line, so a night without the arm is byte-identical
            parts.pop("agent", None)
        status: NightStatus = p.status or "failed"
        result = NightResult(
            status=status,
            failed_at=p.failed_at,
            checkpoint=p.checkpoint if status == "promoted" else None,
            gates=p.gates,
            choices=tuple(p.choices),
            cost_usd=math.fsum(parts.values()),
            cost_parts=parts,
            log_path=p.log.path,
        )
        uses, system = USES["finish"]
        p.log.step("finish", "ok", uses=uses, system=system, night_status=status)
        p.log.finish(
            result,
            {
                "user_id_hash": None,
                "night": p.night.isoformat(),
                "night_end": p.end.isoformat(),
                "trained": p.checkpoint is not None,
                "trained_adapter": checkpoint_id(p.checkpoint),
                **({} if p.wake_seconds is None else {"wake_seconds": p.wake_seconds}),
                **(
                    {}
                    if p.gates_wake_seconds is None
                    else {"gates_wake_seconds": p.gates_wake_seconds}
                ),
                "paid_actions": p.paid_actions,
                "wonder_calls": p.wonder.calls,
                "wonder_usd": p.wonder.usd,
                "steps": [record["step"] for record in p.log.steps()],
                **self._kept(p),
                **({} if p.agent is None else {
                    "agent": {"notes": list(p.agent.notes), "tokens_used": p.agent.tokens_used,
                              "stopped": p.agent.stopped}}),
                **({"sleep": p.receipts["sleep"]} if "sleep" in p.receipts else {}),
                **({"mind_loop": p.receipts["practice"]["loop"]}
                   if "loop" in (p.receipts.get("practice") or {}) else {}),
                **kept_summary(p.receipts),
                **(
                    {}
                    if p.gpu_batch is None
                    else {"gpu_batch": {"digest": p.gpu_batch[0], "people": p.gpu_batch[1]}}
                ),
            },
            night=p.night,
        )
        return result


def run_night_batch_with(
    deps: NightDeps,
    nights: Mapping[str, date],
    *,
    windows: Mapping[str, Window] | None = None,
    accelerated: bool = False,
    days: Mapping[str, DayRun] | None = None,
    sweep: Sweep | None = None,
    as_of: Mapping[str, datetime] | None = None,
) -> dict[str, NightResult]:
    """``run_night_batch`` over the given ports. ``accelerated`` runs pass each simulated day's
    window; the steps are the same. ``days`` are simulated days run just before the night: each
    one's window is the night's, and its log starts with the day's counts. ``sweep`` is what the
    night command's start swept (``sweep.py``), logged first. ``as_of`` is each person's night
    clock, the virtual night end that her snapshot, its dates and her expectations read; it is the
    window's end in real time (the two clocks are equal) and, on an accelerated run, the end of
    the virtual date."""
    if accelerated and not (windows or days):
        raise ValueError("an accelerated run passes each person's simulated window")
    if accelerated:
        as_of = {**{uid: day_window(night)[1] for uid, night in nights.items()}, **(as_of or {})}
    return NightRunner(deps).run_batch(nights, windows, days, sweep, as_of)


_DEFAULT: list[Callable[[], NightDeps]] = []


def set_default_deps(factory: Callable[[], NightDeps] | None) -> None:
    """The ports ``run_night`` and ``run_night_batch`` use (the command line sets them)."""
    _DEFAULT[:] = [] if factory is None else [factory]


def _deps() -> NightDeps:
    if not _DEFAULT:
        raise RuntimeError(
            "no night ports are set; run the night through python -m anjo.sleep.night"
        )
    return _DEFAULT[0]()


def run_night_batch(
    nights: dict[str, date],
    *,
    windows: dict[str, Window] | None = None,
    accelerated: bool = False,
) -> dict[str, NightResult]:
    return run_night_batch_with(_deps(), nights, windows=windows, accelerated=accelerated)


def run_night(
    user_id: str,
    night: date,
    *,
    window: Window | None = None,
    accelerated: bool = False,
) -> NightResult:
    """A batch of one."""
    windows = {user_id: window} if window is not None else None
    return run_night_batch({user_id: night}, windows=windows, accelerated=accelerated)[user_id]
