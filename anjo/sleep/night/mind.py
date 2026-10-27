"""MIND's inquiry in the night (Task 17): ``MindNight`` is the night's ``Notice`` and ``Inquiry`` port.

The night calls it only while ``NightSettings.mind_inquiry`` is on (``run.py``); off, the night is
Task 7a's as specced. Each person's workspace is her log (``JsonlWorkspaceLog`` on the app's sleep
folder), folded on every read, so it is exported, erased and replayed with her other sleep stores.

- **Step 2b, notice:** one thinker pass (``loop.notice``) over the person's own new items that her
  workspace hasn't received yet, so a rerun of the night passes nothing twice. The pass is written
  to her log in one record when it ends, so its items reach her only with the questions they opened.
  It opens questions and acts on nothing.
- **The snapshot's questions:** every question she holds, with its priority's parts. One
  ``continues`` while it is live, its next step is ready and tonight's budget lasts.
- **Step 4, inquire:** ``loop.run_inquiry`` on the chosen question under the night's one budget
  (12 cycles and 20k tokens, shared with the notice pass), or ``loop.resume_inquiry`` when a kill
  cut tonight's run off inside an ``INVESTIGATE``. ``research`` (Wonder) is offered only while
  Wonder's calls and dollars and the night's paid actions last. A question tonight's runs closed
  owes its close marks (``snapshot.close_marks``), read from her log: they are applied after the
  run however it ended, and again before each snapshot, so a kill or a failure loses none and a
  rerun repeats none. Only a close marks; a cycle alone marks nothing. A resolution citing a
  verified research source is stored as the night's homework artifact; nothing else a finding says
  is ever evidence.
- **Step 10, share:** each finding with an unconsumed anchor (``findings.share_result``) enters the
  target log as a ``Result`` that the second offer reads; the note's facts carry its statement. A
  shared result stands only while its finding does, unchanged, on the anchor it was shared on
  (``results``): the snapshot and the note leave out the rest.

**Holds.** Each step holds its Spark Open worst case (the screen and the resolver, at most
``MIND_SPARK_CALLS``) on ``baseten_mind_usd`` and its Glimmer worst case (the thinker and
``world_query``, at most ``MIND_GLIMMER_CALLS`` of at most ``GLIMMER_CALL_S`` each, and one wake)
on ``modal_usd``, and settles what it metered. Past either count a call returns empty and costs
nothing: the screen then refuses and the resolver says ``undetermined``. A hold with no room ends
the step cleanly (``budget``); a failure inside the run ends it as ``failed``. Neither fails the night.
A log that doesn't fold does, at the step that reads it (RF1), except at step 10, after promotion,
where it shares nothing. A cancel (RF2) stops the run before its next log write and reaches the
night, which purges.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final, Protocol

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import Ledger
from anjo.sleep.mind import snapshot
from anjo.sleep.mind.inquiry import controller, findings, kernel, loop
from anjo.sleep.mind.inquiry.ports import (
    InquiryPorts,
    PredictionResolver,
    Research,
    ScreenPort,
    Thinker,
    WorkspaceLog,
    WorldQuery,
)
from anjo.sleep.mind.inquiry.types import (
    A_READY,
    CYCLE_CAP,
    LIVE_STATUSES,
    OP_CLASSES,
    Budget,
    InquiryRun,
    InvestigateOp,
    ModelReply,
    NoticeOp,
    Question,
    RunStart,
    Workspace,
    WorkspaceEvent,
)
from anjo.sleep.mind.points import QuestionState
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import InquiryOutcome, MindCall, Prices
from anjo.sleep.night.holds import held, spark_usd, spark_worst_usd, usd
from anjo.sleep.note.hold import Fact
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.serving.client import REQUEST_TIMEOUT_S, WAKE_TOTAL_TIMEOUT_S
from anjo.sleep.spark_inquiry import ChatFn
from anjo.sleep.stops import BudgetStop, Cancelled
from anjo.sleep.types import Evidence, QuestionView, Result

logger = logging.getLogger(__name__)

MIND_SPARK_CALLS: Final = 48  # a step's screen and resolver calls: a few per cycle, 12 cycles
MIND_GLIMMER_CALLS: Final = 2 * CYCLE_CAP + 2  # a thinker and a world_query a cycle, the notice pass, a resumed tool
GLIMMER_CALL_S: Final = REQUEST_TIMEOUT_S  # one Glimmer call, inside the client's request timeout
SEARCH_POOL: Final = 100  # items asked of the store before practice scenes are dropped
# What ends a MIND run as failed, not the night: a model call that failed outside the loop's own
# handling, an item the store refuses, a lookup that found nothing.
MIND_FAILURES: Final = (ModelApiError, ValueError, LookupError)

SparkWrap = Callable[[ChatFn], ChatFn]


@dataclass(frozen=True)
class MindModels:
    """One person's model ports for a MIND step. ``model`` names the thinker, for an artifact's
    ancestry. The factory wraps every Spark Open ``ChatFn`` it builds in the step's ``SparkWrap``."""

    thinker: Thinker
    world: WorldQuery
    screen: ScreenPort
    resolver: PredictionResolver
    model: str


ModelsFor = Callable[[str, SparkWrap], MindModels]
ResearchFor = Callable[[MindCall], Research]


class NightLog(WorkspaceLog, Protocol):
    """Her workspace log, which can also take several events in one write."""

    def extend(self, events: Sequence[WorkspaceEvent]) -> None: ...


def _first_run(events: Sequence[WorkspaceEvent], night: date) -> int | None:
    return next((n for n, e in enumerate(events) if isinstance(e.step.entry, RunStart) and e.step.entry.night == night),
                None)


def tonight(events: Sequence[WorkspaceEvent], night: date) -> Budget:
    """The night's one budget, shared by its notice pass and its inquiries: from the first
    ``RunStart`` naming ``night``, a cycle per distinct cycle number and every token logged since."""
    first = _first_run(events, night)
    if first is None:
        return Budget()
    since = [event.step for event in events[first:]]
    return Budget().spend(cycles=len({s.cycle for s in since if s.cycle is not None}),
                          tokens=sum(s.tokens for s in since))


def closed_tonight(events: Sequence[WorkspaceEvent], night: date) -> tuple[Question, ...]:
    """The questions tonight's runs closed: closed in her log, and not before its first ``RunStart``
    naming ``night``."""
    first = _first_run(events, night)
    if first is None:
        return ()
    before = kernel.fold(events[:first])
    return tuple(
        q for q in kernel.fold(events).questions
        if q.status in snapshot.CLOSED_QUESTION
        and ((was := before.question(q.id)) is None or was.status not in snapshot.CLOSED_QUESTION)
    )


def cut_off(events: Sequence[WorkspaceEvent], night: date) -> bool:
    """Whether tonight's inquiry was killed inside an applied ``INVESTIGATE``, its tool unrun."""
    last = events[-1].step if events else None
    if last is None or not (last.applied and isinstance(last.entry, InvestigateOp)) or last.cycle is None:
        return False
    start = loop.resume(events).start
    return start.call == "inquiry" and start.night == night


@dataclass
class _Meters:
    """One step's model calls: Spark Open's counted and priced, Glimmer's counted and timed."""

    prices: Prices
    timer: Callable[[], float]
    spark_calls: int = 0
    spark_usd: float = 0.0
    glimmer_calls: int = 0
    glimmer_s: float = 0.0

    def spark(self, inner: ChatFn) -> ChatFn:
        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            if self.spark_calls >= MIND_SPARK_CALLS:  # past the hold's count: no call, nothing spent
                return ChatResult(text="", tool_calls=(), model=SPARK_OPEN, input_tokens=0, output_tokens=0,
                                  finish_reason=None)
            self.spark_calls += 1
            result = inner(messages, max_tokens)
            self.spark_usd += spark_usd(self.prices, result.input_tokens, result.output_tokens)
            return result
        return chat

    def glimmer(self, ask: Callable[[], ModelReply], model: str) -> ModelReply:
        if self.glimmer_calls >= MIND_GLIMMER_CALLS:
            return ModelReply(text="", tokens=0, model=model)
        self.glimmer_calls += 1
        started = self.timer()
        try:
            return ask()
        finally:
            self.glimmer_s += max(0.0, self.timer() - started)

    @property
    def glimmer_usd(self) -> float:
        return self.prices.serve_gpu_per_hour * self.glimmer_s / 3600.0

    def counts(self) -> dict[str, object]:
        return {"spark_calls": self.spark_calls, "glimmer_calls": self.glimmer_calls}


@dataclass(frozen=True)
class _TimedThinker:
    inner: Thinker
    meters: _Meters
    model: str

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        return self.meters.glimmer(lambda: self.inner.propose(rendered, max_tokens), self.model)


@dataclass(frozen=True)
class _TimedWorld:
    inner: WorldQuery
    meters: _Meters
    model: str

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        return self.meters.glimmer(lambda: self.inner.ask(text, max_tokens), self.model)


@dataclass(frozen=True)
class _Watched:
    """Her log, read against the night's cancel flag before every write (RF2)."""

    inner: WorkspaceLog
    cancelled: Callable[[], bool]

    def append(self, event: WorkspaceEvent) -> None:
        if self.cancelled():
            raise Cancelled("the night was cancelled")
        self.inner.append(event)

    def events(self) -> tuple[WorkspaceEvent, ...]:
        return self.inner.events()


@dataclass
class _Batched:
    """Her log for the notice pass: its events are held, then written in one record (``commit``),
    so the day's items reach her (``kernel.known_items``) only with the questions they opened. A
    kill or a failure before then writes nothing, and the next pass offers the items again. The
    cancel flag is read before each event and before the record (RF2)."""

    inner: NightLog
    cancelled: Callable[[], bool]
    held: list[WorkspaceEvent] = field(default_factory=list)

    def append(self, event: WorkspaceEvent) -> None:
        if self.cancelled():
            raise Cancelled("the night was cancelled")
        self.held.append(event)

    def events(self) -> tuple[WorkspaceEvent, ...]:
        return (*self.inner.events(), *self.held)

    def commit(self) -> None:
        if self.cancelled():
            raise Cancelled("the night was cancelled")
        self.inner.extend(tuple(self.held))


@dataclass(frozen=True)
class StoreSearch:
    """``evidence_search`` over the person's store: her own items and homework, never a practice
    scene (a model's rehearsal, not her life). ``get`` reads ``stored``, masked to ``as_of``."""

    store: EvidenceStore

    def search(self, query: str, as_of: datetime, k: int) -> list[Evidence]:
        found = self.store.search(query, as_of, SEARCH_POOL)
        return [item for item in found if item.kind != "practice"][: max(k, 0)]

    def get(self, ids: Sequence[str], as_of: datetime) -> tuple[Evidence, ...]:
        held_items = {i.id: i for i in self.store.stored(as_of.date()) if i.created_at <= as_of and i.kind != "practice"}
        return tuple(held_items[item_id] for item_id in ids if item_id in held_items)


@dataclass(frozen=True)
class _Step:
    """One step's ports, its meters, its search and its thinker's model."""

    ports: InquiryPorts
    meters: _Meters
    search: StoreSearch
    model: str


class MindNight:
    """The night's ``Notice`` and ``Inquiry`` ports over each person's workspace log."""

    available: bool = True

    def __init__(
        self, *, models: ModelsFor, log: Callable[[str], NightLog], prices: Prices,
        research: ResearchFor | None = None, ledger: Ledger | None = None,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._models = models
        self._log = log
        self._prices = prices
        self._research = research
        self._ledger = ledger
        self._timer = timer

    # ── step 2b ──────────────────────────────────────────────────────────────

    def notice(self, call: MindCall, new: Sequence[Evidence]) -> Mapping[str, object]:
        log = self._log(call.user_id)
        events = log.events()
        ws = kernel.fold(events)
        known = kernel.known_items(ws)
        fresh = [i for i in new if i.kind in findings.PERSON_KINDS and not i.held_out and i.id not in known]
        if not fresh:
            return {"new": 0, "questions_opened": 0}
        batched = _Batched(log, call.cancelled)
        step = self._step(call, batched, research=None)
        targets = call.targets.open_targets(call.as_of)
        try:
            run = self._held(call, step, lambda: loop.notice(
                ws, fresh, step.ports, tonight(events, call.night), night=call.night, targets=targets, as_of=call.as_of))
            batched.commit()
        except BudgetStop:
            return {"new": len(fresh), "questions_opened": 0, "stopped": "budget", **step.meters.counts()}
        except MIND_FAILURES as error:
            return {"new": len(fresh), "questions_opened": 0, "stopped": "failed", "error": type(error).__name__}
        opened = sum(1 for e in run.events if isinstance(e.step.entry, NoticeOp) and e.step.applied)
        return {"new": len(fresh), "questions_opened": opened, **_counts(run, step.meters)}

    # ── the snapshot: her questions, her closes, her standing results ────────

    def questions(self, call: MindCall) -> tuple[QuestionState, ...]:
        events = self._log(call.user_id).events()
        ws = kernel.fold(events)
        targets = call.targets.open_targets(call.as_of)
        lasts = tonight(events, call.night).stop_reason is None
        states = []
        for q in ws.questions:
            parts = controller.priority(ws, q.id, targets=targets, as_of=call.as_of)
            continues = q.status in LIVE_STATUSES and parts.a == A_READY and lasts
            states.append(QuestionState(QuestionView(q.id, q.target_ids, q.status, parts.value, continues), parts))
        return tuple(states)

    def settle(self, call: MindCall) -> list[str]:
        """Apply the close marks each question tonight's runs closed owes, read from her log: each of
        its targets at its current revision (tonight's evidence is in), skipping what is marked."""
        closed = closed_tonight(self._log(call.user_id).events(), call.night)
        targets = call.targets.open_targets(call.as_of)
        owed: dict[str, int] = {}
        for q in closed:
            owed.update(snapshot.close_marks(QuestionView(q.id, q.target_ids, q.status, 0.0, False), list(targets)))
        acted = {t.id: t.acted_revision for t in targets}
        marked = sorted(tid for tid, revision in owed.items() if acted.get(tid, revision) < revision)
        for target_id in marked:
            call.targets.mark_acted(target_id, owed[target_id])
        return marked

    def results(self, call: MindCall) -> tuple[Result, ...]:
        """The target log's results, less each shared finding's that no longer stands: its finding
        retracted or changed, or the anchor it was shared on gone (``findings.share_result``, asked
        again as of tonight)."""
        ws = kernel.fold(self._log(call.user_id).events())
        return self._standing(call, ws)

    def _standing(self, call: MindCall, ws: Workspace) -> tuple[Result, ...]:
        results = call.targets.results()
        targets = call.targets.open_targets(call.as_of)
        search = StoreSearch(call.store)
        by_result = {f"{f.id}.{f.kind}": f for f in ws.findings}
        kept = []
        for result in results:
            fid, _, kind = result.id.rpartition(".")
            if fid and kind in findings.SHARED_KINDS and not result.consumed:
                finding = by_result.get(result.id)
                if finding is None:
                    continue  # retracted, or a finding of another kind now
                again = findings.share_result(
                    ws, finding.question_id, cited=search.get(finding.cites, call.as_of), targets=targets,
                    results=[r for r in results if r.id != result.id], as_of=call.as_of)
                if again is None or (again.target_id, again.basis) != (result.target_id, result.basis):
                    continue  # its anchor lapsed
            kept.append(result)
        return tuple(kept)

    # ── step 4 ───────────────────────────────────────────────────────────────

    def inquire(self, call: MindCall, question_id: str) -> InquiryOutcome:
        log = self._log(call.user_id)
        events = log.events()
        before = kernel.fold(events)
        targets = call.targets.open_targets(call.as_of)
        offered = self._research if call.wonder.can_call() and call.actions_left > 0 else None
        step = self._step(call, _Watched(log, call.cancelled), research=offered(call) if offered is not None else None)
        resumed, wonder_calls = cut_off(events, call.night), call.wonder.calls
        run: InquiryRun | None = None
        stopped, said = None, {}
        try:
            run = self._held(call, step, lambda: loop.resume_inquiry(events, step.ports, targets=targets) if resumed
                             else loop.run_inquiry(before, step.ports, tonight(events, call.night), night=call.night,
                                                   targets=targets, as_of=call.as_of, focus=question_id))
        except BudgetStop:
            stopped, said = "budget", step.meters.counts()
        except MIND_FAILURES as error:
            stopped, said = "failed", {"error": type(error).__name__}
        consumed = self.settle(call)  # however the run ended: a question it closed owes its marks
        # Paid research is what reached Wonder, counted as it was called: a research call with no
        # Wonder call left spends no action, and one the run failed after still spends one.
        research_calls = call.wonder.calls - wonder_calls
        if run is None:
            return InquiryOutcome(result=None, research_calls=research_calls, stopped=stopped, receipt={
                "stopped": stopped, **said, "research_calls": research_calls, "consumed": consumed})
        cycles = len({e.step.cycle for e in run.events if e.step.cycle is not None})
        artifacts = self._artifacts(call, run.ws, step)
        receipt = {"question_id": question_id, "resumed": resumed, "research_offered": step.ports.research is not None,
                   "research_asked": run.research_calls, **_counts(run, step.meters), "consumed": consumed,
                   "artifacts": artifacts}
        return InquiryOutcome(result=None, cycles=cycles, research_calls=research_calls, receipt=receipt)

    def _artifacts(self, call: MindCall, ws: Workspace, step: _Step) -> list[str]:
        """A resolution citing a verified research source: the night's homework artifact, stored once."""
        stored = {item.id for item in call.store.stored(call.night)}
        made = []
        for finding in ws.findings:
            item = findings.artifact(ws, finding.question_id, cited=step.search.get(finding.cites, call.as_of),
                                     person_id=call.user_id, night=call.night, at=call.at, model=step.model)
            if item is not None and item.id not in stored:
                made.append(item)
        if made:
            call.store.add(made)
        return [item.id for item in made]

    # ── step 10 ──────────────────────────────────────────────────────────────

    def share(self, call: MindCall) -> tuple[Result, ...]:
        """Commit a result for each finding that has an unconsumed anchor tonight. It runs after the
        night's promotion, so a log that doesn't fold shares nothing rather than fail the night."""
        try:
            ws = kernel.fold(self._log(call.user_id).events())
        except ValueError as error:
            logger.warning("MIND's workspace didn't fold at step 10 (%s); nothing shared", type(error).__name__)
            return ()
        targets = call.targets.open_targets(call.as_of)
        search = StoreSearch(call.store)
        shared = []
        for finding in ws.findings:
            result = findings.share_result(
                ws, finding.question_id, cited=search.get(finding.cites, call.as_of), targets=targets,
                results=call.targets.results(), as_of=call.as_of,
            )
            if result is not None:
                call.targets.add_result(result)
                shared.append(result)
        return tuple(shared)

    def facts(self, call: MindCall, target_id: str) -> list[Fact]:
        """A standing shared finding's statement, for the note, with its source: the research
        source's URL, or the days the person said what it cites."""
        ws = kernel.fold(self._log(call.user_id).events())
        by_result = {f"{f.id}.{f.kind}": f for f in ws.findings}
        search = StoreSearch(call.store)
        out = []
        for result in self._standing(call, ws):
            finding = by_result.get(result.id)
            if finding is None or result.target_id != target_id or result.consumed:
                continue
            source = search.get([result.homework_id], call.as_of) if result.homework_id else ()
            if source:
                out.append(Fact(text=finding.statement, source=source[0].source))
                continue
            days = sorted({i.day.isoformat() for i in search.get(finding.cites, call.as_of) if i.kind in findings.PERSON_KINDS})
            # COPY-PENDING(kevin): how a note cites the person's own words.
            out.append(Fact(text=finding.statement, source=f"what you told me on {', '.join(days) or 'earlier days'}"))
        return out

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _step(self, call: MindCall, log: WorkspaceLog, *, research: Research | None) -> _Step:
        """A step's ports over ``log``, which reads the cancel flag before it writes."""
        meters = _Meters(self._prices, self._timer)
        models = self._models(call.user_id, meters.spark)
        search = StoreSearch(call.store)
        ports = InquiryPorts(
            thinker=_TimedThinker(models.thinker, meters, models.model), screen=models.screen, search=search,
            world=_TimedWorld(models.world, meters, models.model), resolver=models.resolver,
            log=log, research=research,
        )
        return _Step(ports, meters, search, models.model)

    def _held(self, call: MindCall, step: _Step, run: Callable[[], InquiryRun]) -> InquiryRun:
        """``run`` under the step's two holds; what it metered is settled, and charged to ``mind``,
        however it ends."""
        meters, rate = step.meters, self._prices.serve_gpu_per_hour
        try:
            with _holds(self._ledger, call.job, self._prices, rate, meters):
                return run()
        finally:
            call.spend.add("mind", meters.spark_usd + meters.glimmer_usd)


@contextmanager
def _holds(ledger: Ledger | None, job: str, prices: Prices, rate: float, meters: _Meters) -> Iterator[None]:
    spark_bound = usd(spark_worst_usd(prices, MIND_SPARK_CALLS))
    gpu_bound = usd(rate * (MIND_GLIMMER_CALLS * GLIMMER_CALL_S + WAKE_TOTAL_TIMEOUT_S) / 3600.0)
    with held(ledger, f"{job}:spark", ("baseten_mind_usd",), spark_bound, lambda: meters.spark_usd), \
            held(ledger, f"{job}:glimmer", ("modal_usd",), gpu_bound, lambda: meters.glimmer_usd):
        yield


def _counts(run: InquiryRun, meters: _Meters) -> dict[str, object]:
    """A run's receipt, content-free: operations by name, rejections by code, every decision with its
    numbers, the night's budget after it and the calls it made."""
    ops: dict[str, int] = {}
    rejected: dict[str, int] = {}
    for event in run.events:
        step = event.step
        if step.code is not None:
            rejected[step.code] = rejected.get(step.code, 0) + 1
        elif isinstance(step.entry, OP_CLASSES):
            ops[step.entry.name] = ops.get(step.entry.name, 0) + 1
    decisions = [
        {"kind": d.kind, "question_id": d.question_id, "reason": d.reason,
         "priorities": [{"question_id": p.question_id, "stake": p.stake, "u": round(p.u, 6), "a": p.a,
                         "value": round(p.value, 6)} for p in d.priorities]}
        for d in run.decisions
    ]
    budget = run.budget
    return {
        "ops": ops, "rejected": rejected, "decisions": decisions,
        "budget": {"cycles_used": budget.cycles_used, "cycle_cap": budget.cycle_cap,
                   "tokens_used": budget.tokens_used, "token_cap": budget.token_cap},
        "resolver_tokens": run.resolver_tokens, **meters.counts(),
    }
