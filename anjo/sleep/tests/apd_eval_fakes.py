"""Offline fakes for the APD harness (Task 17-eval).

``FakeCore`` stands in for 17-core's kernel and loop with the contract's shapes and few of its
rules: every step is applied to a valid ``Workspace`` (an ASSESS by Bayes' rule), logged with its
digests before and after, and ``fold`` rebuilds the workspace from the log and checks them. Its loop calls the thinker once
per cycle on a short view of the focal question, runs ``evidence_search`` and ``world_query`` for
an INVESTIGATE, and stops when no question is live or the budget is spent. ``mind_reply`` is a
thinker script that walks a question through HYPOTHESIZE, PREDICT (a test of the lead),
INVESTIGATE and CONCLUDE, so a run's ops follow its state. Lives are small and synthetic.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from anjo.sleep import types as t
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.apd.arms import MindCore
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.ports import InquiryPorts, WorkspaceLog

DAY0 = date(2026, 10, 1)
SEARCH_K = 3
Reply = Callable[[str], str]


# ── A small synthetic life ────────────────────────────────────────────────────


def evidence(day: date, n: int, text: str, *, held_out: bool = False) -> t.Evidence:
    return t.Evidence(
        id=f"e-{day:%d}-{n}", person_id="life-1", day=day, kind="lesson", text=text, source=f"session:{day}",
        held_out=held_out, created_at=datetime(day.year, day.month, day.day, 20, 0, tzinfo=timezone.utc),
        model="spark-open-fake", ancestry=frozenset({"spark-open-fake"}),
    )


def target(day: date, target_id: str = "t-scene4") -> t.Target:
    return t.Target(
        id=target_id, person_id="life-1", kind="thread", label="Scene 4's shoot", check_after=None, status="open",
        revision=1, acted_revision=0, confidence=None, last_mentioned=day,
    )


def life(days: int = 3, *, split: apd.Split = "development", per_day: int = 2) -> apd.Life:
    """``days`` days; each day's first item mentions a plan (the fake notice pass opens a
    question on it), the rest are background."""
    span = tuple(DAY0 + timedelta(days=n) for n in range(days))
    items = tuple(
        evidence(day, n, f"day {day:%d} plan for scene 4" if n == 1 else f"day {day:%d} kitchen note {n}")
        for day in span for n in range(1, per_day + 1)
    )
    targets = {day: (target(DAY0),) for day in span}
    sessions = tuple(apd.Session(day, f"sessions of {day}", "spark-open-fake") for day in span)
    return apd.Life(
        id="life-1", field="indie film", seed=7, split=split, days=span, sessions=sessions, evidence=items,
        targets=targets,
    )


# ── Model ports ───────────────────────────────────────────────────────────────


@dataclass
class ScriptThinker:
    """A thinker whose reply is ``reply(prompt)``; it records every call and spends
    ``min(tokens, max_tokens)``."""

    reply: Reply
    tokens: int = 100
    model: str = "glimmer-fake"
    calls: list[tuple[str, int]] = field(default_factory=list)

    def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
        self.calls.append((rendered, max_tokens))
        return it.ModelReply(text=self.reply(rendered), tokens=min(self.tokens, max_tokens), model=self.model)


@dataclass
class World:
    tokens: int = 50
    calls: list[tuple[str, int]] = field(default_factory=list)

    def ask(self, text: str, max_tokens: int) -> it.ModelReply:
        self.calls.append((text, max_tokens))
        return it.ModelReply(text="general knowledge", tokens=min(self.tokens, max_tokens), model="glimmer-fake")


@dataclass
class Screen:
    def screen(self, text: str, purpose: it.ScreenPurpose, max_tokens: int) -> it.Screened:
        return it.Screened(allowed=True, tokens=0)


@dataclass
class Resolver:
    def resolve(self, claim: str, evidence: Sequence[t.Evidence]) -> it.ResolverReply:
        return it.ResolverReply(verdict="undetermined", tokens=5)


def models(thinker: ScriptThinker, world: World | None = None) -> apd.ArmModels:
    return apd.ArmModels(thinker=thinker, world=world or World(), screen=Screen(), resolver=Resolver())


# ── The fake kernel ───────────────────────────────────────────────────────────


def digest(ws: it.Workspace) -> str:
    return hashlib.sha256(repr(ws).encode()).hexdigest()


def apply(ws: it.Workspace, step: it.Step, *, rules: it.KernelRules = it.FULL_RULES) -> it.Applied | it.Rejected:
    changed: it.Workspace | it.Rejected = ws
    if step.code is None and step.applied:
        changed = _op(ws, step.entry, step)
    elif isinstance(step.entry, it.Edit):
        changed = _edit(ws, step.entry, step.seq)
    if isinstance(changed, it.Rejected):
        return changed
    if isinstance(step.entry, it.RunStart) and step.entry.wipe:
        after = it.Workspace(history=(step,))
    else:
        after = dataclasses.replace(changed, history=(*changed.history, step))
    return it.Applied(after, it.WorkspaceEvent(step, digest(ws), digest(after)))


def fold(events: Iterable[it.WorkspaceEvent]) -> it.Workspace:
    ws = it.Workspace()
    for event in events:
        if event.before != digest(ws):
            raise ValueError("an event's before isn't the digest so far")
        done = apply(ws, event.step)
        if isinstance(done, it.Rejected) or done.event.after != event.after:
            raise ValueError("an event doesn't reproduce its after digest")
        ws = done.ws
    return ws


def _set_status(ws: it.Workspace, question_id: str, status: it.QuestionStatus) -> it.Workspace:
    return dataclasses.replace(ws, questions=tuple(
        dataclasses.replace(q, status=status) if q.id == question_id else q for q in ws.questions
    ))


def _op(ws: it.Workspace, op: object, step: it.Step) -> it.Workspace | it.Rejected:
    if isinstance(op, it.NoticeOp):
        if not op.target_ids or not op.origin_evidence:
            return it.Rejected("no_targets" if not op.target_ids else "k3_citations")
        question = it.Question(
            id=f"q{step.seq}", target_ids=op.target_ids, kind=op.kind, origin_evidence=op.origin_evidence,
            status="open", opened_at=step.at, subject=op.subject,
        )
        return dataclasses.replace(ws, questions=(*ws.questions, question))
    held = ws.question(str(getattr(op, "question_id", "")))
    if held is None:
        return it.Rejected("unknown_ref")
    if isinstance(op, it.HypothesizeOp):
        return _hypothesize(ws, held, op)
    if isinstance(op, it.PredictOp):
        return _predict(ws, held, op, step)
    if isinstance(op, it.AssessOp):
        return _assess(ws, held, op)
    if isinstance(op, it.ConcludeOp):
        return _conclude(ws, held, op, step)
    if isinstance(op, it.RetractOp):
        return _retract(ws, held, op, step)
    if isinstance(op, it.ParkOp):
        return _set_status(ws, held.id, "parked")
    return ws


def _hypothesize(ws: it.Workspace, question: it.Question, op: it.HypothesizeOp) -> it.Workspace | it.Rejected:
    if ws.hypotheses_of(question.id):
        return it.Rejected("hypothesize_once")
    if not it.MIN_ALTERNATIVES <= len(op.hypotheses) <= it.MAX_ALTERNATIVES:
        return it.Rejected("hypothesis_count")
    total = sum(alt.credence for alt in op.hypotheses) + op.catch_all
    made = tuple(
        it.Hypothesis(f"{question.id}.h{n}", question.id, alt.statement, alt.credence / total)
        for n, alt in enumerate(op.hypotheses, start=1)
    )
    catch_all = it.Hypothesis(it.catch_all_id(question.id), question.id, it.CATCH_ALL, op.catch_all / total)
    return _set_status(dataclasses.replace(ws, hypotheses=(*ws.hypotheses, *made, catch_all)), question.id, "active")


def _predict(ws: it.Workspace, question: it.Question, op: it.PredictOp, step: it.Step) -> it.Workspace | it.Rejected:
    if set(op.p_true_given) != {h.id for h in ws.hypotheses_of(question.id)}:
        return it.Rejected("k1_hypotheses")
    prediction = it.Prediction(
        id=f"p{step.seq}", question_id=question.id, claim=op.claim, p_true_given=op.p_true_given,
        committed_at=step.at, horizon=step.at + timedelta(days=op.horizon_days), outcome=None, resolved_by=(),
    )
    return dataclasses.replace(ws, predictions=(*ws.predictions, prediction))


def _assess(ws: it.Workspace, question: it.Question, op: it.AssessOp) -> it.Workspace | it.Rejected:
    held = ws.hypotheses_of(question.id)
    likelihoods = op.likelihoods or {}
    if set(likelihoods) != {h.id for h in held}:
        return it.Rejected("k1_hypotheses")
    weighted = {h.id: h.credence * likelihoods[h.id] for h in held}
    total = sum(weighted.values())
    others = tuple(h for h in ws.hypotheses if h.question_id != question.id)
    update = it.Update(question.id, op.evidence_id, "assessed_evidence", (op.evidence_id,), likelihoods)
    return dataclasses.replace(
        ws, hypotheses=(*others, *(dataclasses.replace(h, credence=weighted[h.id] / total) for h in held)),
        updates=(*ws.updates, update),
    )


def _put_finding(ws: it.Workspace, finding: it.Finding) -> it.Workspace:
    others = tuple(f for f in ws.findings if f.question_id != finding.question_id)
    return dataclasses.replace(ws, findings=(*others, finding))


def _conclude(ws: it.Workspace, question: it.Question, op: it.ConcludeOp, step: it.Step) -> it.Workspace | it.Rejected:
    chosen = next((h for h in ws.hypotheses_of(question.id) if h.id == op.hypothesis_id), None)
    if chosen is None:
        return it.Rejected("unknown_ref")
    held = ws.finding_of(question.id)
    finding = it.Finding(
        id=held.id if held else f"f{step.seq}", question_id=question.id, statement=op.statement, p=chosen.credence,
        cites=op.cites, kind=op.kind, hypothesis_id=chosen.id,
    )
    status: it.QuestionStatus = "concluded" if op.kind == "resolution" else question.status
    return _set_status(_put_finding(ws, finding), question.id, status)


def _retract(ws: it.Workspace, question: it.Question, op: it.RetractOp, step: it.Step) -> it.Workspace:
    held = ws.finding_of(question.id)
    credences = {h.id: h.credence for h in ws.hypotheses_of(question.id)}
    finding = it.Finding(
        id=held.id if held else f"f{step.seq}", question_id=question.id, statement=op.statement,
        p=credences.get(held.hypothesis_id or "", 0.0) if held else 0.0, cites=op.cites, kind="retraction",
        hypothesis_id=held.hypothesis_id if held else None,
    )
    return _set_status(_put_finding(ws, finding), question.id, "retracted")


def _edit(ws: it.Workspace, edit: it.Edit, seq: int) -> it.Workspace:
    held = ws.hypotheses_of(edit.question_id)
    if not held:
        raise ValueError("the question has no hypotheses to edit")
    others = tuple(h for h in ws.hypotheses if h.question_id != edit.question_id)
    if edit.kind == "swap_lead":
        lead = max(held, key=lambda h: h.credence)
        other = next((h for h in held if h.id == edit.hypothesis_id), None)
        if other is None or other.id == lead.id:
            raise ValueError("swap_lead names a hypothesis other than the leader")
        swap = {lead.id: other.credence, other.id: lead.credence}
        return dataclasses.replace(ws, hypotheses=(
            *others, *(dataclasses.replace(h, credence=swap.get(h.id, h.credence)) for h in held),
        ))
    if edit.kind == "delete_prediction":
        if not any(p.id == edit.prediction_id and p.pending for p in ws.predictions):
            raise ValueError("delete_prediction names a pending prediction")
        return dataclasses.replace(ws, predictions=tuple(p for p in ws.predictions if p.id != edit.prediction_id))
    if len(held) - 1 >= it.MAX_ALTERNATIVES:
        raise ValueError("a set holds four alternatives")
    injected = it.Hypothesis(f"{edit.question_id}.x{seq}", edit.question_id, str(edit.statement), 0.5)
    catch_all = it.catch_all_id(edit.question_id)
    predictions = tuple(
        dataclasses.replace(p, p_true_given={**p.p_true_given, injected.id: p.p_true_given[catch_all]})
        if p.question_id == edit.question_id else p
        for p in ws.predictions
    )
    halved = tuple(dataclasses.replace(h, credence=h.credence / 2) for h in held)
    return dataclasses.replace(ws, hypotheses=(*others, *halved, injected), predictions=predictions)


# ── The fake loop ─────────────────────────────────────────────────────────────


@dataclass
class _Run:
    ws: it.Workspace
    log: WorkspaceLog
    events: list[it.WorkspaceEvent] = field(default_factory=list)

    def record(self, step: it.Step) -> int:
        done = apply(self.ws, step)
        if isinstance(done, it.Rejected):
            done = apply(self.ws, dataclasses.replace(step, code=done.code))
        assert isinstance(done, it.Applied)
        self.log.append(done.event)
        self.events.append(done.event)
        self.ws = done.ws
        return step.seq

    def at(self, as_of: datetime, entry: object, **fields: object) -> int:
        return self.record(it.Step(seq=self.ws.seq, at=as_of, entry=entry, **fields))  # type: ignore[arg-type]


def _live(ws: it.Workspace, question_id: str | None) -> bool:
    question = ws.question(question_id) if question_id else None
    return question is not None and question.status in it.LIVE_STATUSES


def view(ws: it.Workspace, question_id: str) -> str:
    """The fake render: the facts ``mind_reply`` steers by."""
    question = ws.question(question_id)
    assert question is not None
    held = ws.hypotheses_of(question_id)
    lead = max(held, key=lambda h: h.credence).id if held else "none"
    searched = any(
        isinstance(s.entry, it.Retrieved) and s.entry.question_id == question_id for s in ws.history
    )
    pending = [p.id for p in ws.predictions if p.question_id == question_id and p.pending]
    updates = [u for u in ws.updates if u.question_id == question_id]
    return "\n".join((
        f"FOCUS {question_id}", f"origin: {','.join(question.origin_evidence)}", f"hypotheses: {len(held)}",
        f"ids: {','.join(h.id for h in held)}", f"lead: {lead}", f"predictions: {len(pending)}",
        f"searched: {'yes' if searched else 'no'}", f"updates: {len(updates)}",
    ))


@dataclass
class FakeCore:
    """17-core's loop functions, recording what each call was given."""

    calls: list[dict[str, object]] = field(default_factory=list)
    searches: list[tuple[str, datetime, tuple[str, ...]]] = field(default_factory=list)

    def core(self) -> MindCore:
        return MindCore(notice=self.notice, run_inquiry=self.run_inquiry, apply=apply, fold=fold, digest=digest)

    def notice(
        self, ws: it.Workspace, new: Sequence[t.Evidence], ports: InquiryPorts, budget: it.Budget, *,
        night: date, targets: Sequence[t.Target], as_of: datetime, wipe: bool = False,
    ) -> it.InquiryRun:
        self.calls.append({"call": "notice", "night": night, "targets": tuple(targets), "as_of": as_of,
                           "budget": budget, "wipe": wipe, "new": tuple(e.id for e in new), "ports": ports})
        run = _Run(ws, ports.log)
        run.at(as_of, it.RunStart(night, as_of, "notice", it.FULL_RULES, "priority", None, budget, wipe))
        prompt = "NOTICE PASS\n" + "\n".join(f"[{e.id}] {e.text}" for e in new) + "\nTARGETS " + ",".join(
            x.id for x in targets
        )
        reply = ports.thinker.propose(prompt, budget.tokens_left)
        budget = budget.spend(tokens=reply.tokens)
        refs = tuple(it.EvidenceRef(e.id, e.day, e.created_at) for e in new)
        seq = run.at(as_of, it.Retrieved(None, "new_evidence", "", refs), tokens=reply.tokens)
        for op in _notices(reply.text):
            run.at(as_of, op)
        return it.InquiryRun(ws=run.ws, events=tuple(run.events), decisions=(), turns=(it.Turn(seq, prompt, reply),),
                             budget=budget)

    def run_inquiry(
        self, ws: it.Workspace, ports: InquiryPorts, budget: it.Budget, *, night: date,
        targets: Sequence[t.Target], as_of: datetime, focus: str | None = None,
        rules: it.KernelRules = it.FULL_RULES, control: it.ControlMode = "priority", wipe: bool = False,
    ) -> it.InquiryRun:
        self.calls.append({"call": "inquiry", "night": night, "targets": tuple(targets), "as_of": as_of,
                           "budget": budget, "wipe": wipe, "rules": rules, "control": control, "focus": focus,
                           "ports": ports})
        run = _Run(ws, ports.log)
        run.at(as_of, it.RunStart(night, as_of, "inquiry", rules, control, focus, budget, wipe))
        turns: list[it.Turn] = []
        cycle = 0
        while budget.stop_reason is None:
            question = focus if _live(run.ws, focus) else next(
                (q.id for q in reversed(run.ws.questions) if q.status in it.LIVE_STATUSES), None,
            )
            if question is None:
                break
            prompt = f"CYCLE {cycle}\n" + view(run.ws, question)
            reply = ports.thinker.propose(prompt, budget.tokens_left)
            budget = budget.spend(cycles=1, tokens=reply.tokens)
            seq, budget = self._cycle(run, ports, budget, reply, as_of, cycle, question)
            turns.append(it.Turn(seq, prompt, reply))
            cycle += 1
        return it.InquiryRun(ws=run.ws, events=tuple(run.events), decisions=(), turns=tuple(turns), budget=budget)

    def _cycle(
        self, run: _Run, ports: InquiryPorts, budget: it.Budget, reply: it.ModelReply, as_of: datetime,
        cycle: int, question: str,
    ) -> tuple[int, it.Budget]:
        try:
            op: it.Op | None = it.op_from_json(json.loads(reply.text))
        except (ValueError, it.OpFormatError):
            op = None
        seq = run.at(as_of, op, code=None if op else "malformed", cycle=cycle, focus=question, tokens=reply.tokens)
        if isinstance(op, it.InvestigateOp) and op.tool == "evidence_search":
            found = tuple(ports.search.search(op.query, as_of, SEARCH_K))
            self.searches.append((op.query, as_of, tuple(e.id for e in found)))
            refs = tuple(it.EvidenceRef(e.id, e.day, e.created_at) for e in found)
            run.at(as_of, it.Retrieved(op.question_id, "evidence_search", op.query, refs), cycle=cycle, focus=question)
        elif isinstance(op, it.InvestigateOp) and op.tool == "world_query":
            answer = ports.world.ask(op.query, budget.tokens_left)
            budget = budget.spend(tokens=answer.tokens)
            run.at(as_of, it.Answered(op.question_id, op.query, answer.text), cycle=cycle, focus=question,
                   tokens=answer.tokens)
        return seq, budget


def _notices(text: str) -> tuple[it.NoticeOp, ...]:
    try:
        found = json.loads(text)
    except ValueError:
        return ()
    ops = tuple(it.op_from_json(item) for item in found) if isinstance(found, list) else ()
    return tuple(op for op in ops if isinstance(op, it.NoticeOp))


# ── Thinker scripts ───────────────────────────────────────────────────────────

_ITEM = re.compile(r"^\[(\S+)\] .*\bplan\b", re.MULTILINE)
_FIELD = re.compile(r"^(\w+): (.*)$", re.MULTILINE)


def mind_reply(prompt: str) -> str:
    """The notice pass opens a gap question on each new item naming a plan; a cycle walks the
    focal question through HYPOTHESIZE, PREDICT, a search and a CONCLUDE on its leader."""
    if "NOTICE PASS" in prompt:
        targets = prompt.rsplit("TARGETS ", 1)[1].split(",")
        return json.dumps([
            {"op": "NOTICE", "kind": "gap", "subject": f"The plan in {item} misses a step", "target_ids": targets[:1],
             "origin_evidence": [item]}
            for item in _ITEM.findall(prompt)
        ])
    focal = prompt.split("FOCUS ", 1)[1]
    question = focal.split("\n", 1)[0]
    facts = dict(_FIELD.findall(focal))
    if facts["hypotheses"] == "0":
        return json.dumps({"op": "HYPOTHESIZE", "question_id": question, "catch_all": 0.1, "hypotheses": [
            {"statement": "The plan lacks a location permit", "credence": 0.6},
            {"statement": "The plan lacks a crew call", "credence": 0.3},
        ]})
    if facts["predictions"] == "0" and facts["searched"] == "no":
        tests = {h: 0.9 if h == facts["lead"] else 0.5 if h == it.catch_all_id(question) else 0.2
                 for h in facts["ids"].split(",")}
        return json.dumps({"op": "PREDICT", "question_id": question, "claim": "She mentions a permit", "horizon_days": 2,
                           "p_true_given": tests})
    if facts["searched"] == "no":
        return json.dumps({"op": "INVESTIGATE", "question_id": question, "tool": "evidence_search", "query": "plan scene"})
    lead = facts["lead"].rsplit(".", 1)[1]  # h1, h2, ...: id-free, so a resumed run's text matches
    return json.dumps({"op": "CONCLUDE", "question_id": question, "hypothesis_id": facts["lead"], "kind": "resolution",
                       "statement": f"The plan misses what {lead} says", "cites": facts["origin"].split(",")})
