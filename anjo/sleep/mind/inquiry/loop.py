"""MIND's inquiry loop: the notice pass and the inquiry cycles (Task 17).

The only module here that calls ports. Each call logs its ``RunStart`` first, then appends every
step's event to ``ports.log`` the moment it happens, so a kill loses nothing logged and
``kernel.fold`` of the log rebuilds the workspace. Every step is stamped ``as_of``, and every tool
is masked to it: searches and fetches see only items created by then, and the resolver only items
inside a prediction's window. No clock is read.

**A cycle,** on the question ``controller.decide`` picked:

1. each pending prediction on it with items the resolver hasn't seen is resolved, blind: the
   resolver gets the claim and the items in its window, never the hypotheses;
2. the loop fetches the question's items by id (``EvidenceSearch.get``), renders, and asks the
   thinker for one operation, with ``max_tokens`` the budget's ``tokens_left``;
3. the reply is parsed leniently; the kernel checks the operation, then the screen reads what she
   would investigate (a question's subject, a hypothesis, a research query), and the step is
   logged with its code and every token it cost, so a rejection costs the cycle;
4. an applied ``INVESTIGATE`` runs its tool and logs what came back.

Cycles are numbered from the budget's ``cycles_used``, so a resumed run continues the numbering.
After a ``park`` decision (K5) the loop lets the controller pick afresh, which passes over the
stalled question for the rest of the night. Predictions past their horizon are settled when an
inquiry run starts: resolved once more on any unseen items in their window, then censored if
still undetermined.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Final

from anjo.sleep.mind.inquiry import controller, kernel, parse, render
from anjo.sleep.mind.inquiry.ports import InquiryPorts
from anjo.sleep.mind.inquiry.types import (
    FULL_RULES,
    LIVE_STATUSES,
    TOOLS,
    Answered,
    Budget,
    ControlMode,
    Decision,
    EvidenceRef,
    HypothesizeOp,
    InquiryRun,
    InvestigateOp,
    KernelRules,
    ModelReply,
    NoticeOp,
    Observation,
    Op,
    Prediction,
    RejectCode,
    Rejected,
    Resolved,
    Retrieved,
    RunStart,
    ScreenPurpose,
    Step,
    Turn,
    Workspace,
    WorkspaceEvent,
)
from anjo.sleep.types import Evidence, Target

K_SEARCH: Final = 5  # items one evidence_search returns
MAX_RENDERED: Final = 10  # items a cycle's render shows
EARLIER_SHOWN: Final = 20  # earlier items the notice pass shows beside the day's new ones


@dataclass(frozen=True)
class _Context:
    """What stays fixed through one call."""

    ports: InquiryPorts
    as_of: datetime
    target_ids: frozenset[str]
    tools: frozenset[str]


@dataclass(frozen=True)
class _Run:
    """One call's state so far; every change makes a new one."""

    ws: Workspace
    budget: Budget
    rules: KernelRules
    events: tuple[WorkspaceEvent, ...] = ()
    decisions: tuple[Decision, ...] = ()
    turns: tuple[Turn, ...] = ()
    cycles: frozenset[int] = field(default_factory=frozenset)
    resolver_tokens: int = 0
    research_calls: int = 0


def _context(ports: InquiryPorts, as_of: datetime, targets: Sequence[Target]) -> _Context:
    tools = TOOLS if ports.research is not None else TOOLS - {"research"}
    return _Context(ports, as_of, frozenset(t.id for t in targets), frozenset(tools))


def _step(
    run: _Run, ctx: _Context, entry: Op | Observation | RunStart | None, *, code: RejectCode | None = None,
    cycle: int | None = None, focus: str | None = None, tokens: int = 0,
) -> Step:
    return Step(seq=run.ws.seq, at=ctx.as_of, entry=entry, code=code, cycle=cycle, focus=focus, tokens=tokens)


def _log(run: _Run, ctx: _Context, step: Step) -> _Run:
    """Apply ``step``, append its event to the log at once, and charge its cycle and tokens."""
    result = kernel.apply(run.ws, step, rules=run.rules)
    if isinstance(result, Rejected):
        raise RuntimeError(f"the kernel refused step {step.seq}, which the loop had admitted ({result.code})")
    ctx.ports.log.append(result.event)
    new_cycle = step.cycle is not None and step.cycle not in run.cycles
    return replace(
        run, ws=result.ws, events=(*run.events, result.event),
        budget=run.budget.spend(cycles=1 if new_cycle else 0, tokens=step.tokens),
        cycles=run.cycles | {step.cycle} if new_cycle and step.cycle is not None else run.cycles,
    )


def _turn(run: _Run, seq: int, prompt: str, reply: ModelReply) -> _Run:
    return replace(run, turns=(*run.turns, Turn(seq=seq, prompt=prompt, reply=reply)))


def _result(run: _Run) -> InquiryRun:
    return InquiryRun(
        ws=run.ws, events=run.events, decisions=run.decisions, turns=run.turns, budget=run.budget,
        resolver_tokens=run.resolver_tokens, research_calls=run.research_calls,
    )


def _refs(items: Sequence[Evidence]) -> tuple[EvidenceRef, ...]:
    return tuple(EvidenceRef(id=item.id, day=item.day, created_at=item.created_at) for item in items)


def _masked(items: Sequence[Evidence], as_of: datetime) -> tuple[Evidence, ...]:
    return tuple(item for item in items if not item.held_out and item.created_at <= as_of)


# ── Admission: the kernel, then the screen ────────────────────────────────────


def _screened_texts(op: Op) -> list[tuple[str, ScreenPurpose]]:
    if isinstance(op, NoticeOp):
        return [(op.subject, "question")]
    if isinstance(op, HypothesizeOp):
        return [(alt.statement, "hypothesis") for alt in op.hypotheses]
    if isinstance(op, InvestigateOp) and op.tool == "research":
        return [(op.query, "research_query")]
    return []


def _admit(run: _Run, ctx: _Context, op: Op, pending: int) -> tuple[RejectCode | None, int]:
    """The code ``op`` is logged with, and the screen's tokens. ``pending`` is the tokens this
    step has already cost but not yet charged."""
    code = kernel.check(run.ws, op, rules=run.rules, target_ids=ctx.target_ids, tools=ctx.tools)
    if code is not None:
        return code, 0
    spent = 0
    for text, purpose in _screened_texts(op):
        screened = ctx.ports.screen.screen(text, purpose, max(run.budget.tokens_left - pending - spent, 0))
        spent += screened.tokens
        if not screened.allowed:
            return "domain_screen", spent
    return None, spent


# ── Tools ─────────────────────────────────────────────────────────────────────


def _resolve(run: _Run, ctx: _Context, prediction: Prediction, cycle: int | None, focus: str | None) -> _Run:
    """Ask the resolver about ``prediction`` on every item in its window, blind."""
    refs = {ref.id: ref for ref in kernel.resolvable(run.ws, prediction, ctx.as_of)}
    evidence = _masked(ctx.ports.search.get(list(refs), ctx.as_of), ctx.as_of)
    seen = tuple(refs[item.id] for item in evidence if item.id in refs)
    if not seen:
        return run
    reply = ctx.ports.resolver.resolve(prediction.claim, [item for item in evidence if item.id in refs])
    run = replace(run, resolver_tokens=run.resolver_tokens + reply.tokens)
    return _log(run, ctx, _step(run, ctx, Resolved(prediction.id, reply.verdict, seen), cycle=cycle, focus=focus))


def _investigate(run: _Run, ctx: _Context, op: InvestigateOp, cycle: int, focus: str) -> _Run:
    ports = ctx.ports
    if op.tool == "world_query":
        reply = ports.world.ask(op.query, run.budget.tokens_left)
        entry: Observation = Answered(op.question_id, op.query, reply.text)
        return _log(run, ctx, _step(run, ctx, entry, cycle=cycle, focus=focus, tokens=reply.tokens))
    if op.tool == "research" and ports.research is not None:
        found = tuple(ports.research.research(op.question_id, op.query))
        run = replace(run, research_calls=run.research_calls + 1)
        entry = Retrieved(op.question_id, "research", op.query, _refs(found))
        return _log(run, ctx, _step(run, ctx, entry, cycle=cycle, focus=focus))
    found = _masked(ports.search.search(op.query, ctx.as_of, K_SEARCH), ctx.as_of)
    entry = Retrieved(op.question_id, "evidence_search", op.query, _refs(found))
    return _log(run, ctx, _step(run, ctx, entry, cycle=cycle, focus=focus))


def _rendered_ids(ws: Workspace, focus: str) -> list[str]:
    """The items a cycle shows: the question's origin and its hits. Past ``MAX_RENDERED``, the
    unassessed hits come first, then the origin, then the newest assessed hits."""
    question = ws.question(focus)
    origin = list(question.origin_evidence) if question is not None else []
    returned = [item.id for item in kernel.hits(ws, focus)]
    ordered = list(dict.fromkeys(origin + returned))
    if len(ordered) <= MAX_RENDERED:
        return ordered
    used = kernel.used_items(ws, focus)
    newest = [item for item in reversed(returned) if item not in origin]
    ranked = [i for i in newest if i not in used] + origin + [i for i in newest if i in used]
    keep = frozenset(list(dict.fromkeys(ranked))[:MAX_RENDERED])
    return [item for item in ordered if item in keep]


# ── A cycle ───────────────────────────────────────────────────────────────────


def _cycle(run: _Run, ctx: _Context, focus: str, cycle: int) -> _Run:
    for prediction in [p for p in run.ws.predictions if p.question_id == focus and p.pending]:
        if kernel.unseen(run.ws, prediction, ctx.as_of):
            run = _resolve(run, ctx, prediction, cycle, focus)
    evidence = _masked(ctx.ports.search.get(_rendered_ids(run.ws, focus), ctx.as_of), ctx.as_of)
    prompt = render.render(run.ws, focus, evidence)
    reply = ctx.ports.thinker.propose(prompt, run.budget.tokens_left)
    parsed = parse.parse_op(reply.text)
    if isinstance(parsed, Rejected):
        step = _step(run, ctx, None, code=parsed.code, cycle=cycle, focus=focus, tokens=reply.tokens)
        return _turn(_log(run, ctx, step), step.seq, prompt, reply)
    code, screened = _admit(run, ctx, parsed, reply.tokens)
    step = _step(run, ctx, parsed, code=code, cycle=cycle, focus=focus, tokens=reply.tokens + screened)
    run = _turn(_log(run, ctx, step), step.seq, prompt, reply)
    if code is None and isinstance(parsed, InvestigateOp):
        run = _investigate(run, ctx, parsed, cycle, focus)
    return run


def _settle_expired(run: _Run, ctx: _Context) -> _Run:
    """Predictions past their horizon: one last resolution on unseen items in their window (live
    questions only), then censored if still undetermined."""
    for prediction_id in [p.id for p in run.ws.predictions if p.pending and p.horizon <= ctx.as_of]:
        prediction = next(p for p in run.ws.predictions if p.id == prediction_id)
        question = run.ws.question(prediction.question_id)
        live = question is not None and question.status in LIVE_STATUSES
        if live and kernel.unseen(run.ws, prediction, ctx.as_of):
            run = _resolve(run, ctx, prediction, None, None)
        prediction = next(p for p in run.ws.predictions if p.id == prediction_id)
        if prediction.pending:
            run = _log(run, ctx, _step(run, ctx, Resolved(prediction_id, "censored", ())))
    return run


# ── The two loop functions ────────────────────────────────────────────────────


def _earlier(ws: Workspace, fresh: Sequence[Evidence], ctx: _Context) -> tuple[Evidence, ...]:
    """Items she already holds, for the notice pass: those on a target a new item names first,
    then the newest, up to ``EARLIER_SHOWN``, shown oldest first."""
    new_ids = frozenset(item.id for item in fresh)
    held = [item for item in kernel.known_items(ws) if item not in new_ids]
    items = _masked(ctx.ports.search.get(held, ctx.as_of), ctx.as_of)
    named = frozenset(item.target_id for item in fresh if item.target_id is not None)
    ranked = [i for i in reversed(items) if i.target_id in named] + [i for i in reversed(items) if i.target_id not in named]
    chosen = frozenset(item.id for item in ranked[:EARLIER_SHOWN])
    return tuple(item for item in items if item.id in chosen)


def notice(
    ws: Workspace,
    new: Sequence[Evidence],
    ports: InquiryPorts,
    budget: Budget,
    *,
    night: date,
    targets: Sequence[Target],
    as_of: datetime,
    wipe: bool = False,
) -> InquiryRun:
    """Night step 2b: one thinker pass over the day's new evidence, which may only open questions."""
    ctx = _context(ports, as_of, targets)
    run = _Run(ws, budget, FULL_RULES)
    start = RunStart(night, as_of, "notice", FULL_RULES, "priority", None, budget, wipe)
    run = _log(run, ctx, _step(run, ctx, start))
    fresh = tuple({item.id: item for item in _masked(new, as_of)}.values())
    arrived = Retrieved(None, "new_evidence", "", _refs(fresh))
    if not fresh or run.budget.tokens_left == 0:
        return _result(_log(run, ctx, _step(run, ctx, arrived)))
    prompt = render.render_notice(run.ws, fresh, _earlier(run.ws, fresh, ctx), targets)
    reply = ports.thinker.propose(prompt, run.budget.tokens_left)
    step = _step(run, ctx, arrived, tokens=reply.tokens)
    run = _turn(_log(run, ctx, step), step.seq, prompt, reply)
    parsed = parse.parse_notices(reply.text)
    if isinstance(parsed, Rejected):
        return _result(_log(run, ctx, _step(run, ctx, None, code=parsed.code)))
    for op in parsed:
        code, screened = _admit(run, ctx, op, 0)
        run = _log(run, ctx, _step(run, ctx, op, code=code, tokens=screened))
    return _result(run)


def run_inquiry(
    ws: Workspace,
    ports: InquiryPorts,
    budget: Budget,
    *,
    night: date,
    targets: Sequence[Target],
    as_of: datetime,
    focus: str | None = None,
    rules: KernelRules = FULL_RULES,
    control: ControlMode = "priority",
    wipe: bool = False,
) -> InquiryRun:
    """Night step 4 (and each APD night): cycles under ``controller.decide`` until it stops."""
    ctx = _context(ports, as_of, targets)
    run = _Run(ws, budget, rules)
    start = RunStart(night, as_of, "inquiry", rules, control, focus, budget, wipe)
    run = _settle_expired(_log(run, ctx, _step(run, ctx, start)), ctx)
    cycle, current = budget.cycles_used, focus
    while True:
        decision = controller.decide(run.ws, run.budget, focus=current, targets=targets, as_of=as_of, control=control)
        run = replace(run, decisions=(*run.decisions, decision))
        if decision.question_id is None:
            return _result(run)
        if decision.kind == "park":
            current = None
            continue
        current = decision.question_id
        run = _cycle(run, ctx, current, cycle)
        cycle += 1


# ── Kill-resume ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Resume:
    """Where a killed call picks up: the folded workspace, its latest ``RunStart``, the budget (that
    start's plus one cycle per distinct ``Step.cycle`` and every token logged since) and the focus
    (the last ``Step.focus`` since, else the start's)."""

    ws: Workspace
    start: RunStart
    budget: Budget
    focus: str | None


def resume(events: Sequence[WorkspaceEvent]) -> Resume:
    """``ValueError`` when the log doesn't fold or holds no run."""
    ws = kernel.fold(events)
    starts = [n for n, event in enumerate(events) if isinstance(event.step.entry, RunStart)]
    if not starts:
        raise ValueError("a log with no RunStart can't be resumed")
    start = events[starts[-1]].step.entry
    assert isinstance(start, RunStart)
    since = [event.step for event in events[starts[-1] + 1:]]
    cycles = len({step.cycle for step in since if step.cycle is not None})
    budget = start.budget.spend(cycles=cycles, tokens=sum(step.tokens for step in since))
    focus = next((step.focus for step in reversed(since) if step.focus is not None), start.focus)
    return Resume(ws=ws, start=start, budget=budget, focus=focus)


def resume_inquiry(events: Sequence[WorkspaceEvent], ports: InquiryPorts, *, targets: Sequence[Target]) -> InquiryRun:
    """Rebuild a killed inquiry from its log and run it on, with the start's night, ``as_of``,
    rules and control. ``ports.log`` is the log ``events`` came from."""
    point = resume(events)
    if point.start.call != "inquiry":
        raise ValueError("only an inquiry run resumes this way; rerun the notice pass")
    return run_inquiry(
        point.ws, ports, point.budget, night=point.start.night, targets=targets, as_of=point.start.as_of,
        focus=point.focus, rules=point.start.rules, control=point.start.control,
    )
