"""``run_night`` and ``run_night_batch`` (Task 5): one person's night, or many people's in one batch.

Steps, in order, each writing its log record before the next starts (``nightlog.py``):

1. ``close_day``: the night's window (the calendar day, or an accelerated run's simulated day).
2. ``extract``: the day's evidence and targets (Task 3). ``notice`` (2b, Task 17): with ``mind_inquiry``
   on, one thinker pass over the person's new items may open questions (``night/mind.py``); else a no-op.
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
6. ``brief``: both briefs fold the evidence no night has folded yet (before practice, so the
   training brief is the pre-practice one).
5. ``practice``, 6. ``dataset``, 7. ``train``, 8. ``gates``: only on a training night, one whose
   cleared, non-practice evidence since the last trained night reaches ``min_new_examples``.
   The last trained night is the night of the checkpoint that serves (the one the next night
   trains from), so a failed or rejected night's evidence stays new. Every due person in a batch
   runs inside one GPU session. Below it, the night is ``nothing_new`` and those items carry over.
9. ``promote``: every gate passed, or the night is rejected and the previous checkpoint serves.
10. ``note``: a ``hold_note`` choice, or a second offer (``allowed_actions={"hold_note"}``) after a
    result with a verified source, or an inquiry's anchored one (with ``mind_inquiry`` on, each finding
    with an unconsumed anchor shares here); at most one note, which the writer may decline. A synthetic life
    records this offer's decision point too, before the note settles its target, offered or not: a
    night with no second offer records one that allows nothing.
11. ``finish``: the ``NightResult`` and every receipt.

A failure, a budget stop or a cancel stops that person's night at that step: the previous
checkpoint keeps serving, no note is held, and the rest of the batch goes on (RF1). Every date is
the person's own night's, never the wall clock's: her snapshot, its eligibility and her expectations
read the night's clock (``as_of``, the virtual night end). Records are stamped at the window's
wall-clock end or ``as_of``, whichever is earlier (``_Night.stamp``), so readers on either clock see
them; the wall-clock end alone drives budgets and timeouts.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Final

from anjo.sleep.night.deps import (
    ALL_ACTIONS,
    STUDY_ACTIONS,
    InquiryOutcome,
    MindCall,
    NightDeps,
    StudyOutcome,
)
from anjo.sleep.serving.backend import phase
from anjo.sleep.serving.client import GlimmerNotReady
from anjo.sleep.night.holds import held, spark_worst_usd, usd
from anjo.sleep.night.sweep import Sweep
from anjo.sleep.practice.full import TEACHER_ALARM_SHARE
from anjo.sleep.night.meter import SpendMeter, WonderMeter, WonderSpent
from anjo.sleep.mind import expect, points, receipt
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.sim.day import DayRun
from anjo.sleep.night.nightlog import NightLog, Uses, gate_record, night_folder, summaries
from anjo.sleep.note.hold import SOURCE as NOTE_SOURCE
from anjo.sleep.note.hold import Fact, write_note
from anjo.sleep.ports import EvidenceStore, TargetStore
from anjo.sleep.stops import BudgetStop, Cancelled
from anjo.sleep.train.dataset import Dataset, build_night_dataset
from anjo.sleep.train.lora import fresh_adapter_name
from anjo.sleep.train.terms import attach_night_terms
from anjo.sleep.types import (
    MAX_NIGHT_ACTIONS,
    CheckpointRef,
    Choice,
    ChoiceContext,
    Evidence,
    GateResult,
    NightResult,
    NightStatus,
    Probe,
    Result,
    Target,
    WantSnapshot,
)

Window = tuple[datetime, datetime]
PERSON_KINDS: Final = frozenset({"lesson", "correction", "preference", "tone_request"})
USES: Final[Mapping[str, tuple[Uses, str | None]]] = {
    "sweep": ("new", None), "simulate": ("existing", "chat route"), "close_day": ("new", None), "extract": ("existing", "history store"), "notice": ("new", None),
    "probes": ("new", None), "choose": ("new", None), "brief": ("new", None),
    "practice": ("new", None), "dataset": ("new", None), "train": ("new", None),
    "gates": ("existing", "crisis check"), "promote": ("new", None), "note": ("existing", "outreach"),
    "finish": ("new", None),
}
# Worst cases for the paid steps' holds (holds.py): Spark Open calls per step, and the rest.
EXTRACT_MAX_CALLS: Final = 2  # the day's one call, and one retry
BRIEF_MAX_CALLS: Final = 2  # the served and the training brief
PROBES_WORST_USD: Final = 0.50  # Muse Spark writing a real person's night probes (about 8 calls)
NOTE_MAX_S: Final = 300.0  # one Glimmer call on the serving GPU, inside the client's request timeout

GPU_STEPS: Final = ("practice", "dataset", "train", "gates", "promote")
# A study that fails this way stops the study, not the night: a Wonder error or timeout, a meta_usd
# hold with no room, a request that can't fit its bound, a page in an unknown encoding.
STUDY_FAILURES: Final = (ModelApiError, BudgetStop, ValueError, LookupError)


def day_window(night: date) -> Window:
    start = datetime.combine(night, dtime.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


@dataclass
class _Night:
    """One person's night as it runs: the working state the steps fill in."""

    uid: str
    night: date
    window: Window
    log: NightLog
    spend: SpendMeter
    wonder: WonderMeter
    actions_left: int
    store: EvidenceStore
    targets: TargetStore
    as_of: datetime  # the night's clock: the virtual night end, which the window's end is in real time
    failed_at: str | None = None
    status: NightStatus | None = None
    due: bool = False
    gated: bool = False
    sealed: list[Probe] = field(default_factory=list)
    choices: list[Choice] = field(default_factory=list)
    note_choice: Choice | None = None
    study: StudyOutcome | None = None
    inquiry: InquiryOutcome | None = None
    shared: tuple[str, ...] = ()  # results MIND's findings shared at step 10 (Task 17)
    data: Dataset | None = None
    checkpoint: CheckpointRef | None = None
    gates: dict[str, GateResult] = field(default_factory=dict)
    paid_actions: int = 0
    wake_seconds: float | None = None
    gates_wake_seconds: float | None = None
    wonder_month_spent: float = 0.0
    receipts: dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> datetime:
        """The window's wall-clock end: budget holds and timeouts read it, never a date."""
        return self.window[1]

    @property
    def stamp(self) -> datetime:
        """What the night's records are created at: never after the wall clock, never after its own."""
        return min(self.end, self.as_of)


def _safe_error(error: BaseException) -> str:
    """The error's type, and its message only when Sleep raised it (never another library's text,
    which could quote a request)."""
    name = type(error).__name__
    return f"{name}: {error}" if type(error).__module__.startswith("anjo.sleep") else name


class NightRunner:
    """Runs a batch of nights over one set of ports."""

    def __init__(self, deps: NightDeps) -> None:
        self.deps = deps
        self._timer = deps.timer or time.monotonic

    # ── the batch ────────────────────────────────────────────────────────────

    def run_batch(
        self, nights: Mapping[str, date], windows: Mapping[str, Window] | None = None,
        days: Mapping[str, DayRun] | None = None, sweep: Sweep | None = None,
        as_of: Mapping[str, datetime] | None = None,
    ) -> dict[str, NightResult]:
        """The batch as one phase: Glimmer goes back to sleep when it ends, however it ends."""
        with phase(self.deps.sleep):
            return self._run_batch(nights, windows, days, sweep, as_of or {})

    def _run_batch(
        self, nights: Mapping[str, date], windows: Mapping[str, Window] | None,
        days: Mapping[str, DayRun] | None, sweep: Sweep | None, as_of: Mapping[str, datetime],
    ) -> dict[str, NightResult]:
        days = days or {}
        windows = {**(windows or {}), **{uid: day.window for uid, day in days.items()}}
        people = []
        for uid, night in nights.items():
            window = windows.get(uid) or day_window(night)
            people.append(self._start(uid, night, window, as_of.get(uid) or window[1]))
        for p in people:
            if sweep is not None:  # what the command's start swept for this batch, as counts
                p.log.step("sweep", "ok", uses="new", system=None, holds_settled=sweep.holds_settled,
                           sessions_stopped=sweep.sessions_stopped, sessions_not_ours=sweep.sessions_not_ours)
            if p.uid in days:  # a short day (a turn Glimmer couldn't answer) is recorded, never an error
                day = days[p.uid]
                if day.failure is not None:  # the day never started: that is a failure, not a short day
                    p.failed_at, p.status = "simulate", "failed"
                    p.log.step("simulate", "failed", uses="existing", system="chat route", error=day.failure,
                               wake_seconds=day.wake_seconds)
                    continue
                extra: dict[str, Any] = {} if day.wake_seconds is None else {"wake_seconds": day.wake_seconds}
                p.log.step("simulate", "ok", uses="existing", system="chat route", answered=day.answered,
                           failed_turns=day.failed, short_day=day.short, **extra)
            for step, fn in (("close_day", self._close_day), ("extract", self._extract), ("notice", self._notice),
                             ("probes", self._probes), ("choose", self._choose), ("brief", self._brief)):
                self._do(p, step, fn)
        due = [p for p in people if p.due and p.failed_at is None]
        if due:
            self._gpu_job(due)
        for p in due:
            self._do(p, "promote", self._promote)
        results = {}
        for p in people:
            if p.failed_at is None and not p.due:
                p.status = "nothing_new"
                for step in GPU_STEPS:
                    self._skip(p, step, "below min_new_examples: no practice, training, gates or promotion")
            self._do(p, "note", self._note)
            results[p.uid] = self._finish(p)
        return results

    def _gpu_job(self, due: list[_Night]) -> None:
        """Practice, dataset, training and gates for the due people, in one GPU session."""
        if not self._wake(due):
            return
        try:
            with self.deps.gpu.open([p.uid for p in due]):
                for p in due:
                    for step, fn in (("practice", self._practice), ("dataset", self._dataset),
                                     ("train", self._train), ("gates", self._gates)):
                        self._do(p, step, fn)
        except Exception as error:  # the session itself failed: every person it held stops
            self._fail_ungated(due, error)

    def _wake(self, due: list[_Night]) -> bool:
        """Wake Glimmer before the first Modal step. The seconds go in each night's summary; a
        server that can't be woken fails these nights at practice with a code, not a message."""
        wake = self.deps.wake
        if wake is None:
            return True
        try:
            woke = wake()
        except Exception as error:  # a bad URL or token: a recorded failure for each person, not an aborted batch
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

    def _wake_again(self, p: _Night) -> float | None:
        """Wake Glimmer again right before the gates; a server that stays cold fails this person."""
        if self.deps.wake is None:
            return None
        woke = self.deps.wake()
        p.gates_wake_seconds = woke.seconds
        if not woke.ok:
            raise GlimmerNotReady()
        return woke.seconds

    def _fail_ungated(self, due: list[_Night], error: Exception) -> None:
        for p in due:
            if p.failed_at is None and not p.gated:
                self._fail(p, "practice", error)

    # ── step plumbing ────────────────────────────────────────────────────────

    def _start(self, uid: str, night: date, window: Window, as_of: datetime) -> _Night:
        deps = self.deps
        deps.cancel(uid)  # before any state loads: the first look at the flag is the baseline a change is read against
        log = NightLog(night_folder(deps.logs(uid), night))
        history = summaries(deps.logs(uid))
        actions_left = min(MAX_NIGHT_ACTIONS, max(0, deps.settings.rolling_actions - _paid_within(history, window[1], timedelta(hours=24))))
        wonder_spent = _spent_within(history, window[1], timedelta(days=30))
        return _Night(
            uid=uid, night=night, window=window, log=log,
            spend=SpendMeter(spark_in_per_m=deps.prices.spark_in_per_m, spark_out_per_m=deps.prices.spark_out_per_m),
            wonder=WonderMeter(deps.settings.homework_call_cap, max(0.0, deps.settings.wonder_usd_month - wonder_spent)),
            actions_left=actions_left, store=deps.evidence(uid), targets=deps.targets(uid), as_of=as_of,
            wonder_month_spent=wonder_spent,
        )

    def _do(self, p: _Night, step: str, fn: Callable[[_Night], Mapping[str, Any]]) -> bool:
        if p.failed_at is not None:
            return False
        if self.deps.cancel(p.uid):
            return self._fail(p, step, Cancelled("the night was cancelled"))
        try:
            receipt = dict(fn(p))
        except Exception as error:  # a failure stops this person's night at this step (RF1)
            return False if self._cancelled_mid_step(p, step) else self._fail(p, step, error)
        if self._cancelled_mid_step(p, step):
            return False
        noop = bool(receipt.pop("noop", False))
        uses, system = USES[step]
        p.log.step(step, "ok", uses=uses, system=system, noop=noop, **receipt)
        p.receipts[step] = receipt
        return True

    def _cancelled_mid_step(self, p: _Night, step: str) -> bool:
        """The person was deleted while ``step`` ran (RF2): remove whatever it wrote, record nothing
        (the night log's folder is gone with the delete), and stop this person's night."""
        if not self.deps.cancel(p.uid):
            return False
        self.deps.purge(p.uid)
        p.failed_at, p.status = step, "failed"
        return True

    def _fail(self, p: _Night, step: str, error: BaseException) -> bool:
        p.failed_at, p.status = step, "failed"
        uses, system = USES[step]
        p.log.step(step, "failed", uses=uses, system=system, error=_safe_error(error))
        return False

    def _skip(self, p: _Night, step: str, why: str) -> None:
        uses, system = USES[step]
        p.log.step(step, "skipped", uses=uses, system=system, why=why)

    # ── steps 1 to 4 and the brief ──────────────────────────────────────────

    def _job(self, p: _Night, step: str) -> str:
        return f"night:{step}:{p.uid}:{p.night.isoformat()}"

    def _metered(self, p: _Night, part: str, step: str, calls: int) -> Any:
        """Hold ``step``'s Spark Open worst case on ``baseten_sleep_usd``; settle what ``part`` metered."""
        before = p.spend.parts().get(part, 0.0)  # type: ignore[call-overload]
        return held(self.deps.ledger, self._job(p, step), ("baseten_sleep_usd",), usd(spark_worst_usd(self.deps.prices, calls)),
                    lambda: p.spend.parts().get(part, 0.0) - before)  # type: ignore[call-overload]

    def _close_day(self, p: _Night) -> Mapping[str, Any]:
        start, end = p.window
        if end <= start:
            raise ValueError("a night's window ends after it starts")
        return {"window": [start.isoformat(), end.isoformat()], "night": p.night.isoformat()}

    def _extract(self, p: _Night) -> Mapping[str, Any]:
        p.spend.part = "extract"
        source = self.deps.extractor(p.spend.wrap_extract(self.deps.extract_chat))
        with self._metered(p, "extract", "extract", EXTRACT_MAX_CALLS):
            items = source.extract_day(p.uid, p.night, p.window, p.as_of)
        since = self._since(p)
        fresh = [
            item for item in p.store.trainable(p.night)
            if item.kind != "practice" and (since is None or item.day > since)
        ]
        p.due = len(fresh) >= self.deps.settings.min_new_examples
        kinds: dict[str, int] = {}
        for item in items:
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
        return {"items": len(items), "kinds": kinds, "new_since_last_training": len(fresh),
                "min_new_examples": self.deps.settings.min_new_examples, "training_night": p.due}

    def _since(self, p: _Night) -> date | None:
        """The last trained night: the night of the checkpoint that serves, which the next night
        trains from. A failed or rejected night's training never served, so its evidence stays new."""
        serving = self.deps.registry.last_promoted(p.uid)
        return None if serving is None else serving.night

    def _mind(self) -> bool:
        """MIND's inquiry runs only with the setting on (Task 17); off, the night is Task 7a's."""
        return self.deps.settings.mind_inquiry and self.deps.inquiry.available

    def _call(self, p: _Night, step: str) -> MindCall:
        return MindCall(user_id=p.uid, night=p.night, as_of=p.as_of, at=p.stamp, store=p.store, targets=p.targets,
                        spend=p.spend, wonder=p.wonder, actions_left=p.actions_left, job=self._job(p, step),
                        cancelled=lambda: self.deps.cancel(p.uid))

    def _notice(self, p: _Night) -> Mapping[str, Any]:
        notice = self.deps.notice
        if not (self.deps.settings.mind_inquiry and notice.available):
            return {"noop": True, "port": "notice (Task 17): no-op"}
        new = [item for item in p.store.for_day(p.night) if not item.held_out]
        return notice.notice(self._call(p, "notice"), new)

    def _probes(self, p: _Night) -> Mapping[str, Any]:
        if self.deps.is_synthetic(p.uid):  # sealed once: nothing is called tonight
            sealed, how = self.deps.probes.register(p.uid, p.night, p.store)
        else:  # Muse Spark writes them, unmetered: the hold is settled whole
            with held(self.deps.ledger, self._job(p, "probes"), ("meta_usd",), usd(PROBES_WORST_USD),
                      lambda: PROBES_WORST_USD):
                sealed, how = self.deps.probes.register(p.uid, p.night, p.store)
        p.sealed = list(sealed)
        return {"sealed": len(p.sealed), "how": how}

    def _raw(self, p: _Night) -> points.RawSnapshot:
        """Her state's raw input to ``snapshot.build`` (Task 7a), on the person's night clock, before
        any eligibility filter. With ``mind_inquiry`` on (Task 17) her questions come from her
        workspace, after the close marks tonight's closes owe; else none."""
        own = [item.created_at for item in p.store.stored(p.night) if item.kind in PERSON_KINDS]
        if self._mind():
            self.deps.inquiry.settle(self._call(p, "choose"))
        targets = p.targets.open_targets(p.as_of)
        questions = self.deps.inquiry.questions(self._call(p, "choose")) if self._mind() else ()
        return points.RawSnapshot(
            tuple(targets), self._results(p), p.as_of, max(own) if own else None,
            intentions=self.deps.intentions.intentions(p.uid, p.as_of), questions=questions,
        )

    def _point(self, p: _Night, phase: points.Phase, raw: points.RawSnapshot, ctx: ChoiceContext,
               choice: Choice | None) -> None:
        """Task 16: every night of a synthetic life records both offers' decision points."""
        if self.deps.is_synthetic(p.uid):
            p.log.point(points.point_record(points.DecisionPoint(p.night, phase, raw, ctx, choice)))

    def _expect(self, p: _Night, raw: points.RawSnapshot, choice: Choice) -> str | None:
        """Task 16: a chosen study records how likely she thinks the person is to need it."""
        if self.deps.estimator is None or choice.want is None:
            return None
        if choice.action == "inquire" and all(q.view.id != choice.question_id for q in raw.questions):
            return None  # a question the snapshot doesn't hold has no first target to expect on
        target_id = points.study_target(choice, raw)
        target = next((t for t in raw.targets if t.id == target_id), None)
        if target is None:
            return None
        made = expect.expectation(target, choice.want.kind, p.as_of, self.deps.estimator)
        if all(kept.id != made.id for kept in p.targets.expectations()):  # a rerun of the night records it once
            p.targets.record_expectation(made)
        return made.id

    def _offered(self) -> frozenset[str]:
        offered = set(ALL_ACTIONS)
        if not self.deps.homework.available:
            offered -= STUDY_ACTIONS
        if not self._mind():
            offered.discard("inquire")
        return frozenset(offered)

    def _choose(self, p: _Night) -> Mapping[str, Any]:
        deps = self.deps
        raw = self._raw(p)
        state = points.build(raw)
        offered = self._offered()
        ctx = ChoiceContext(policy=deps.policy, allowance=p.wonder.allowance(p.actions_left), allowed_actions=offered)
        choice = deps.chooser.choose("away", deps.chooser.derive(state), ctx)
        p.choices.append(choice)
        self._point(p, "pre_study", raw, ctx, choice)
        action = choice.action
        cap = deps.settings.wonder_usd_month
        month = {"cap_usd": cap, "spent_usd": round(p.wonder_month_spent, 6),
                 "left_usd": round(max(0.0, cap - p.wonder_month_spent), 6)}
        homework = "available" if deps.homework.available else "no-op: no study offered"
        if deps.homework.available and month["left_usd"] <= 0:
            homework = f"capped: the monthly Wonder cap of ${cap:.2f} is spent, so no study runs"
        done: dict[str, Any] = {
            "action": action, "target_id": choice.target_id, "offered": sorted(offered),
            "allowance": {"actions_left": p.actions_left, "calls_left": p.wonder.calls_left, "usd_left": p.wonder.usd_left},
            "wonder_month": month,
            "ports": {
                "chooser": type(deps.chooser).__name__,
                "homework": homework,
                "inquiry": "available" if self._mind() else "no-op (Task 17)",
                "expectations": "registered" if deps.estimator else "estimator not registered (Task 16)",
            },
            "not_offered": sorted({c.want_id for c in choice.considered if c.reason == "action_not_offered"}),
            "expectation": self._expect(p, raw, choice),
        }
        outcome: receipt.Outcome | None = None
        if action in STUDY_ACTIONS:
            done.update(self._study(p, state, choice))
            outcome = _outcome(p.study.stopped if p.study is not None else None)
        elif action == "inquire":
            done.update(self._inquire(p, choice))
            outcome = _outcome(p.inquiry.stopped if p.inquiry is not None else None)
        elif action == "hold_note":
            p.note_choice = choice  # written at step 10, whose receipt records the outcome
        done["receipt"] = receipt.record(choice, outcome)
        return done

    def _study(self, p: _Night, state: WantSnapshot, choice: Choice) -> Mapping[str, Any]:
        deps = self.deps
        target = next(t for t in state.targets if t.id == choice.target_id)
        p.actions_left -= 1  # a study that started is a spent action, even if it then fails
        p.paid_actions += 1
        calls_before = p.wonder.calls
        try:
            outcome = deps.homework.study(p.uid, target, p.night, p.wonder, action=choice.action or "", at=p.stamp)
        except WonderSpent:
            outcome = StudyOutcome(result=None, stopped="budget")
        except STUDY_FAILURES as error:  # the study failed, not the night: its calls stay debited
            outcome = StudyOutcome(result=None, stopped="failed", receipt={"error": type(error).__name__})
        if outcome.stopped is not None and p.wonder.calls == calls_before:  # it never started: no action spent
            p.actions_left += 1
            p.paid_actions -= 1
        if outcome.stopped is None:  # a study the budget stopped, or that failed, didn't run: the want stays
            p.targets.mark_acted(target.id, target.revision)
        if outcome.evidence:
            p.store.add(list(outcome.evidence))
        if outcome.result is not None:
            p.targets.add_result(outcome.result)
        p.study = outcome
        return {"study": {**outcome.receipt, "result": outcome.result.id if outcome.result else None,
                          "stopped": outcome.stopped, "homework_items": len(outcome.evidence),
                          "wonder_calls": p.wonder.calls, "wonder_usd": p.wonder.usd}}

    def _inquire(self, p: _Night, choice: Choice) -> Mapping[str, Any]:
        if choice.question_id is None:
            raise ValueError("an inquire choice names its question")
        outcome = self.deps.inquiry.inquire(self._call(p, "inquire"), choice.question_id)
        if outcome.result is not None:
            p.targets.add_result(outcome.result)
        if outcome.research_calls > 0:  # only research is paid; thinker cycles are her own Glimmer
            p.actions_left -= 1
            p.paid_actions += 1
        p.inquiry = outcome
        return {"inquiry": {"cycles": outcome.cycles, "research_calls": outcome.research_calls,
                            "result": outcome.result.id if outcome.result else None, **outcome.receipt}}

    def _brief(self, p: _Night) -> Mapping[str, Any]:
        briefs = self.deps.briefs(p.uid)
        new = briefs.unfolded(p.store.stored(p.night))
        if not new:
            return {"folded": 0, "why": "no evidence since the last fold"}
        p.spend.part = "brief"
        with self._metered(p, "brief", "brief", BRIEF_MAX_CALLS):
            briefs.fold(p.night, new, at=p.stamp, hosted_in_training=self.deps.hosted_in_training,
                        chat=p.spend.wrap_chat(self.deps.spark_chat))
        return {"folded": len(new)}

    # ── the GPU steps ────────────────────────────────────────────────────────

    def _serving(self, p: _Night, part: str, started: tuple[float, int]) -> dict[str, Any]:
        seconds, retries = max(0.0, self._timer() - started[0]), self.deps.retries() - started[1]
        p.spend.add(part, seconds / 3600.0 * self.deps.prices.serve_gpu_per_hour)  # type: ignore[arg-type]
        return {"seconds": seconds, **({"retries": retries} if retries else {})}  # requests sent again: a count

    def _practice(self, p: _Night) -> Mapping[str, Any]:
        kept = self.deps.transcripts(p.uid).for_night(p.night)
        if kept:  # a rerun of the night reuses the practice it already paid for
            p.store.add([transcript.evidence() for transcript in kept])
            return {"scenes": len(kept), "reused": True, "homework": None}
        started = self._timer(), self.deps.retries()
        p.spend.part = "practice"
        runner = self.deps.practice(p.spend.wrap_chat(self.deps.spark_chat), p.night)
        with self._metered(p, "practice", "practice", runner.max_spark_calls):
            run = runner.run(p.uid, p.night, p.store.trainable(p.night), None, at=p.stamp)
        self.deps.transcripts(p.uid).add(list(run.transcripts))
        p.store.add([transcript.evidence() for transcript in run.transcripts])
        share = run.teacher_targets / len(run.transcripts) if run.transcripts else 0.0
        return {"scenes": len(run.transcripts), "dropped_cued": run.dropped_cued, "dropped_cut": run.dropped_cut,
                "f_items": len(run.f_ids), "pairs": run.pairs, "teacher_targets": run.teacher_targets,
                "critic_unreadable": run.critic_unreadable,
                "targets": {"teacher": run.teacher_targets, "glimmer": len(run.transcripts) - run.teacher_targets,
                            "teacher_share": round(share, 4)},
                "teacher_alarm": share > TEACHER_ALARM_SHARE,  # the critic alone promoted the teacher this often
                "homework": None, **({"spark": run.spark, "capped_scenes": run.capped_scenes} if run.spark else {}), **self._serving(p, "practice", started)}

    def _dataset(self, p: _Night) -> Mapping[str, Any]:
        settings = self.deps.settings
        data, probe_dropped = build_night_dataset(
            p.store, self._since(p), p.night, p.sealed, self.deps.decontam,
            settings.token_cap, transcripts=self.deps.transcripts(p.uid), seed=settings.seed,
            dropout=settings.dropout, hosted_in_training=self.deps.hosted_in_training,
        )
        if not data.examples:
            raise ValueError("no training example survived the dataset's checks")
        p.data = data
        done = {"examples": len(data.examples), "tokens": data.tokens, "token_cap": settings.token_cap,
                "new": sum(e.new for e in data.examples), "replay": sum(not e.new for e in data.examples),
                "context_dropped": sum(e.context_dropped for e in data.examples),
                "dropped": dict(data.dropped), "probe_dropped": probe_dropped,
                "context_paraphrase_kept": data.context_paraphrase_kept, "sealed": len(p.sealed)}
        if self.deps.terms.seed:  # sft_dpo_seed's rows (train/terms.py); off, the night is today's
            p.data, done["terms"] = attach_night_terms(self.deps, p.uid, p.night, p.store, p.sealed, data, p.spend)
        return done

    def _train(self, p: _Night) -> Mapping[str, Any]:
        registry = self.deps.registry
        assert p.data is not None
        prev = registry.last_promoted(p.uid)
        number = len(registry.trained_nights(p.uid) - {p.night}) + 1
        name = fresh_adapter_name(p.uid, number, registry.trained_names(p.uid))
        registry.record_trained(p.uid, name, p.night)  # before training: a name is never reused
        ckpt, receipt = self.deps.trainer.train(p.uid, p.night, prev, p.data, name)
        p.checkpoint = ckpt
        minutes = receipt.get("gpu_minutes")
        if isinstance(minutes, int | float) and math.isfinite(minutes):
            p.spend.add("training", float(minutes) / 60.0 * self.deps.prices.train_gpu_per_hour)
        return {"adapter": ckpt.id, "parent": ckpt.parent_id, "from_last_promoted": prev.id if prev else None,
                "receipt": dict(receipt)}

    def _gates(self, p: _Night) -> Mapping[str, Any]:
        assert p.checkpoint is not None
        woke = self._wake_again(p)  # training takes long enough for Modal to idle out
        started = self._timer(), self.deps.retries()
        for name, check in self.deps.gates.for_person(p.uid).items():
            p.gates[name] = check(p.checkpoint)
        p.gated = True
        return {"gates": {name: gate_record(result) for name, result in p.gates.items()},
                **self._serving(p, "gates", started),
                **({} if woke is None else {"wake_seconds": woke})}

    def _promote(self, p: _Night) -> Mapping[str, Any]:
        assert p.checkpoint is not None
        failed = sorted(name for name, result in p.gates.items() if not result.passed)
        if failed or not p.gates:
            p.status = "rejected"
            return {"promoted": None, "rejected_by": failed, "serving": _id(self.deps.registry.last_promoted(p.uid))}
        self.deps.registry.promote(p.uid, p.checkpoint)
        p.status = "promoted"
        return {"promoted": p.checkpoint.id, "serving": p.checkpoint.id}

    # ── step 10, the note, and step 11 ──────────────────────────────────────

    def _second_offer(self, p: _Night) -> Choice | None:
        verified = p.study is not None and p.study.result is not None and p.study.result.verified_sources > 0
        if self._mind():  # a finding with an unconsumed anchor shares now (Task 17)
            p.shared = tuple(r.id for r in self.deps.inquiry.share(self._call(p, "note")))
        anchored = (p.inquiry is not None and p.inquiry.result is not None) or bool(p.shared)
        offer = p.note_choice is None and (verified or anchored)
        if not (offer or self.deps.is_synthetic(p.uid)):
            return None
        raw = self._raw(p)
        allowed = frozenset({"hold_note"}) if offer else frozenset()  # what this point really offered
        ctx = ChoiceContext(policy=self.deps.policy, allowance=p.wonder.allowance(max(0, p.actions_left)),
                            allowed_actions=allowed)
        if not offer:
            self._point(p, "post_result", raw, ctx, None)
            return None
        choice = self.deps.chooser.choose("away", self.deps.chooser.derive(points.build(raw)), ctx)
        p.choices.append(choice)
        self._point(p, "post_result", raw, ctx, choice)
        if choice.action != "hold_note":
            p.receipts["second_offer"] = receipt.record(choice, None)
            return None
        return choice

    def _note(self, p: _Night) -> Mapping[str, Any]:
        done = self._held_note(p)
        return {**done, "shared": list(p.shared)} if p.shared else done

    def _held_note(self, p: _Night) -> Mapping[str, Any]:
        offered = self._second_offer(p)  # a step-4 note still records the post-result point
        choice = p.note_choice or offered
        if choice is None or choice.target_id is None:
            return {"outcome": "no_reason", "offered_again": "second_offer" in p.receipts,
                    "second_offer_receipt": p.receipts.get("second_offer")}
        if p.actions_left < 1:
            return {"outcome": "no_allowance"}
        target_id = choice.target_id
        interim = NightResult(status=p.status or "nothing_new", failed_at=None, checkpoint=p.checkpoint,
                              gates=p.gates, choices=tuple(p.choices), cost_usd=0.0, cost_parts={},
                              log_path=p.log.path)
        rate, started = self.deps.prices.serve_gpu_per_hour, self._timer()
        try:  # the note is optional: whatever fails here, the night keeps its status and checkpoint
            with held(self.deps.ledger, self._job(p, "note"), ("modal_usd",), usd(rate * NOTE_MAX_S / 3600.0),
                      lambda: rate * max(0.0, self._timer() - started) / 3600.0):
                note, outcome = write_note(
                    p.uid, choice, interim, writer=self.deps.note_writer, topic=self._topic(p, target_id),
                    facts=self._facts(p, target_id), night=p.night, check=self.deps.note_check,
                )
        except Exception as error:
            return self._note_failed(choice, target_id, "write", error)
        if note is not None:
            try:
                self.deps.notifier.queue(p.uid, note.text, NOTE_SOURCE)
            except Exception as error:
                return self._note_failed(choice, target_id, "queue", error)
        pushed, push_error = False, None
        if note is not None:
            p.actions_left -= 1
            p.paid_actions += 1
            try:
                pushed = self.deps.notifier.push(p.uid, note.text)
            except Exception as error:  # the note is queued; only its announcement failed
                push_error = _note_error("push", error)
        if outcome in ("held", "declined"):
            p.targets.settle(target_id)
        done: dict[str, Any] = {
            "outcome": "note_declined" if outcome == "declined" else outcome, "target_id": target_id,
            "pushed": pushed, "second_offer": choice is not p.note_choice,
            "receipt": receipt.record(choice, "declined" if outcome == "declined" else "ran" if note else None),
        }
        if push_error is not None:
            done["note_error"] = push_error
        if note is not None:
            done.update({"note_id": note.id, "sources": len(note.sources), "held_back": note.held_back})
        return done

    def _note_failed(self, choice: Choice, target_id: str, stage: str, error: Exception) -> Mapping[str, Any]:
        """A note that couldn't be written or queued: recorded by stage and error type, never words."""
        outcome: receipt.Outcome = "budget" if isinstance(error, BudgetStop) else "failed"
        return {"outcome": "note_failed", "target_id": target_id, "note_error": _note_error(stage, error),
                "receipt": receipt.record(choice, outcome)}

    def _topic(self, p: _Night, target_id: str) -> str:
        targets: Sequence[Target] = p.targets.open_targets(p.as_of)
        return next((t.label for t in targets if t.id == target_id), "something you are working on")

    def _results(self, p: _Night) -> tuple[Result, ...]:
        """The target log's results; with MIND on, a shared finding's only while it stands (Task 17)."""
        return self.deps.inquiry.results(self._call(p, "choose")) if self._mind() else tuple(p.targets.results())

    def _facts(self, p: _Night, target_id: str) -> list[Fact]:
        wanted = {r.homework_id for r in self._results(p) if r.target_id == target_id and not r.consumed}
        items: list[Evidence] = [e for e in p.store.stored(p.night) if e.kind == "homework" and e.id in wanted]
        facts = [Fact(text=item.text, source=item.source) for item in items]
        return facts + self.deps.inquiry.facts(self._call(p, "note"), target_id) if self._mind() else facts

    def _finish(self, p: _Night) -> NightResult:
        parts = p.spend.parts()
        status: NightStatus = p.status or "failed"
        result = NightResult(
            status=status, failed_at=p.failed_at, checkpoint=p.checkpoint if status == "promoted" else None,
            gates=p.gates, choices=tuple(p.choices), cost_usd=math.fsum(parts.values()), cost_parts=parts,
            log_path=p.log.path,
        )
        uses, system = USES["finish"]
        p.log.step("finish", "ok", uses=uses, system=system, night_status=status)
        p.log.finish(result, {
            "user_id_hash": None, "night": p.night.isoformat(), "night_end": p.end.isoformat(),
            "trained": p.checkpoint is not None, "trained_adapter": _id(p.checkpoint),
            **({} if p.wake_seconds is None else {"wake_seconds": p.wake_seconds}),
            **({} if p.gates_wake_seconds is None else {"gates_wake_seconds": p.gates_wake_seconds}),
            "paid_actions": p.paid_actions, "wonder_calls": p.wonder.calls, "wonder_usd": p.wonder.usd,
            "steps": [record["step"] for record in p.log.steps()],
        })
        return result


def _id(ckpt: CheckpointRef | None) -> str | None:
    return None if ckpt is None else ckpt.id


def _outcome(stopped: str | None) -> receipt.Outcome:
    return "budget" if stopped == "budget" else "failed" if stopped == "failed" else "ran"


def _note_error(stage: str, error: BaseException) -> str:
    return f"{stage}: {type(error).__name__}"


def _paid_within(history: Sequence[Mapping[str, Any]], end: datetime, span: timedelta) -> int:
    return sum(int(s.get("paid_actions", 0)) for s in history if _inside(s, end, span))


def _spent_within(history: Sequence[Mapping[str, Any]], end: datetime, span: timedelta) -> float:
    return math.fsum(float(s.get("wonder_usd", 0.0)) for s in history if _inside(s, end, span))


def _inside(summary: Mapping[str, Any], end: datetime, span: timedelta) -> bool:
    at = summary.get("night_end")
    if not isinstance(at, str):
        return False
    moment = datetime.fromisoformat(at)
    return end - span < moment <= end


def run_night_batch_with(
    deps: NightDeps, nights: Mapping[str, date], *, windows: Mapping[str, Window] | None = None,
    accelerated: bool = False, days: Mapping[str, DayRun] | None = None, sweep: Sweep | None = None,
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
        raise RuntimeError("no night ports are set; run the night through python -m anjo.sleep.night")
    return _DEFAULT[0]()


def run_night_batch(
    nights: dict[str, date], *, windows: dict[str, Window] | None = None, accelerated: bool = False,
) -> dict[str, NightResult]:
    return run_night_batch_with(_deps(), nights, windows=windows, accelerated=accelerated)


def run_night(
    user_id: str, night: date, *, window: Window | None = None, accelerated: bool = False,
) -> NightResult:
    """A batch of one."""
    windows = {user_id: window} if window is not None else None
    return run_night_batch({user_id: night}, windows=windows, accelerated=accelerated)[user_id]
