"""The intervention test (Task 17): does her state cause what she does next, or only store it?

``draw_snapshots`` takes ``INTERVENTION_SNAPSHOTS`` mid-inquiry snapshots from development lives'
M logs, in an order a seed fixes. A snapshot is a prefix of the log that ends on a cycle step
inside an inquiry run that goes on, whose focus question is live and has hypotheses.

Each snapshot is resumed after each of the three edits, where it applies (``intervention_test``):

1. ``swap_lead``: the leader swaps credences with the strongest other alternative (skipped when the
   lead is tied, since the swap would name no clear leader);
2. ``delete_prediction``: the question's latest pending prediction is deleted (skipped with none);
3. ``inject_hypothesis``: a plausible wrong alternative joins at 0.5 (skipped when the question
   already holds four, since a set holds at most four).

Every skip is recorded. ``resume_after_edit`` runs each edit against its own unedited twin: both
on the snapshot's evidence, clock, targets, budget left and focus, each on its own fresh in-memory
log seeded with the snapshot, so each branch is a fold of its own log. With the evidence, the
budget and the model held fixed, a difference between the branches is the edit's doing (the
thinker should run at temperature 0, or sampling noise adds to it).

**Measured,** each on the edited branch and, where it means something, on its unedited twin as the
baseline, from the operations the branch applied on the edited question (``measures``):

- ``swap_lead``: ``acts_on_new_lead``, a PREDICT or ASSESS aimed at the new lead (``aims_at``);
- ``delete_prediction``: ``deleted_not_resolved``, no true or false verdict on the deleted
  prediction; ``replaced_or_moved_on``, a new PREDICT on the question, or the focus leaves it
  for another question, or it ends parked or abandoned;
- ``inject_hypothesis``: ``injected_tested``, a PREDICT or ASSESS aimed at it, and
  ``injected_dropped``, retracted, or left below its 0.5 and not leading after an update on
  evidence (no baseline: only the edited branch holds it).

Each is reported apart, and the question's conclusion is compared by the hypothesis it reports and
its kind (``Branch.conclusion``), never by its text.
"""
from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal, Protocol

from anjo.sleep.mind.apd.arms import MemoryLog, MindCore, load_brief, mind_core, mind_ports
from anjo.sleep.mind.apd.resume import resume_point
from anjo.sleep.mind.apd.types import INTERVENTION_SNAPSHOTS, ArmModels, ArmRun, Life
from anjo.sleep.mind.inquiry.ports import InquiryPorts
from anjo.sleep.mind.inquiry.types import (
    LIVE_STATUSES,
    MAX_ALTERNATIVES,
    OP_CLASSES,
    AssessOp,
    Budget,
    Edit,
    EditKind,
    FindingKind,
    Hypothesis,
    InquiryRun,
    Op,
    PredictOp,
    Question,
    Rejected,
    Resolved,
    RunStart,
    Step,
    Workspace,
    WorkspaceEvent,
    op_to_json,
)
from anjo.sleep.types import Target

SkipReason = Literal["tied_lead", "no_pending_prediction", "four_alternatives"]
MeasureName = Literal[
    "acts_on_new_lead", "deleted_not_resolved", "replaced_or_moved_on", "injected_tested", "injected_dropped",
]
EDIT_KINDS: Final[tuple[EditKind, ...]] = ("swap_lead", "delete_prediction", "inject_hypothesis")
INJECTED_AT: Final = 0.5  # an injected hypothesis's credence (Edit)
MOVED_ON: Final = frozenset({"parked", "abandoned"})


class WrongHypothesis(Protocol):
    """Writes the injected alternative: plausible for the question, and wrong by the life's sealed
    problem file."""

    def write(self, question: Question, hypotheses: Sequence[Hypothesis]) -> str: ...


@dataclass(frozen=True)
class Snapshot:
    """A prefix of a development life's M log, mid-inquiry on ``question_id``."""

    life_id: str
    events: tuple[WorkspaceEvent, ...]
    question_id: str


@dataclass(frozen=True)
class Branch:
    """What one branch did after the snapshot: every operation it applied (canonical JSON, for the
    record), the edited question's conclusion (the hypothesis its finding reports, and its kind)
    and its leading hypothesis at the end."""

    ops: tuple[str, ...]
    conclusion: tuple[str | None, FindingKind] | None
    leader: str | None


@dataclass(frozen=True)
class Measure:
    """One registered measure: whether the edited branch shows it, and whether its unedited twin
    does (``None`` where there is nothing to compare)."""

    name: MeasureName
    edited: bool
    unedited: bool | None


@dataclass(frozen=True)
class Resumed:
    """A snapshot resumed unedited and after an edit, and the workspace right after the edit."""

    unedited: InquiryRun
    edited: InquiryRun
    after_edit: Workspace


@dataclass(frozen=True)
class InterventionResult:
    life_id: str
    at: int  # the snapshot's length in events
    edit: Edit
    unedited: Branch
    edited: Branch
    measures: tuple[Measure, ...]

    @property
    def conclusion_differs(self) -> bool:
        return self.unedited.conclusion != self.edited.conclusion


@dataclass(frozen=True)
class Skipped:
    life_id: str
    at: int
    kind: EditKind
    reason: SkipReason


# ── The snapshots ─────────────────────────────────────────────────────────────


def _mid_inquiry(events: Sequence[WorkspaceEvent]) -> Iterator[tuple[int, str]]:
    """(length, focus) of every prefix that ends on a cycle step of an inquiry run that goes on."""
    call = None
    for n, event in enumerate(events[:-1], start=1):
        step = event.step
        call = step.entry.call if isinstance(step.entry, RunStart) else call
        if call == "inquiry" and step.cycle is not None and step.focus and not isinstance(events[n].step.entry, RunStart):
            yield n, step.focus


def draw_snapshots(
    runs: Sequence[tuple[Life, ArmRun]], core: MindCore, *, seed: int, count: int = INTERVENTION_SNAPSHOTS,
) -> tuple[Snapshot, ...]:
    """Up to ``count`` mid-inquiry snapshots from development lives' M runs, in the order ``seed``
    fixes."""
    for life, run in runs:
        if life.split != "development" or run.arm != "M" or run.life_id != life.id:
            raise ValueError("snapshots come from development lives' M runs")
    candidates = [(run, n, focus) for _, run in runs for n, focus in _mid_inquiry(run.events)]
    found: tuple[Snapshot, ...] = ()
    for run, n, focus in random.Random(seed).sample(candidates, len(candidates)):
        ws = core.fold(run.events[:n])
        question = ws.question(focus)
        if question is not None and question.status in LIVE_STATUSES and ws.hypotheses_of(focus):
            found = (*found, Snapshot(run.life_id, run.events[:n], focus))
        if len(found) == count:
            break
    return found


# ── The edits ─────────────────────────────────────────────────────────────────


def edit_for(kind: EditKind, ws: Workspace, question_id: str, wrong: WrongHypothesis) -> Edit | SkipReason:
    """The ``kind`` edit of ``question_id`` in ``ws``, or why it doesn't apply."""
    held = ws.hypotheses_of(question_id)
    if kind == "swap_lead":
        ranked = sorted(held, key=lambda h: (-h.credence, h.id))
        if ranked[0].credence == ranked[1].credence:
            return "tied_lead"
        runner_up = next(h for h in ranked[1:] if not h.is_catch_all)
        return Edit("swap_lead", question_id, hypothesis_id=runner_up.id)
    if kind == "delete_prediction":
        pending = [p for p in ws.predictions if p.question_id == question_id and p.pending]
        if not pending:
            return "no_pending_prediction"
        latest = max(pending, key=lambda p: (p.committed_at, p.id))
        return Edit("delete_prediction", question_id, prediction_id=latest.id)
    if len(held) - 1 >= MAX_ALTERNATIVES:
        return "four_alternatives"
    question = ws.question(question_id)
    assert question is not None
    return Edit("inject_hypothesis", question_id, statement=wrong.write(question, held))


def resume_after_edit(
    events: Sequence[WorkspaceEvent], edit: Edit, ports: InquiryPorts, budget: Budget, *, night: date,
    targets: Sequence[Target], as_of: datetime, core: MindCore | None = None,
) -> Resumed:
    """Resume the snapshot ``events`` twice, unedited and after ``edit`` (logged as a step stamped
    ``as_of``), each on its own fresh in-memory log seeded with ``events``, both on ``ports``'
    evidence at ``as_of``, focused on the edited question, under the snapshot run's rules and
    control."""
    core = core or mind_core()
    start = resume_point(events).start
    ws = core.fold(events)

    def resume(log: MemoryLog, at: Workspace) -> InquiryRun:
        return core.run_inquiry(
            at, dataclasses.replace(ports, log=log), budget, night=night, targets=targets, as_of=as_of,
            focus=edit.question_id, rules=start.rules, control=start.control,
        )

    unedited = resume(MemoryLog(events), ws)
    log = MemoryLog(events)
    applied = core.apply(ws, Step(seq=ws.seq, at=as_of, entry=edit), rules=start.rules)
    if isinstance(applied, Rejected):
        raise ValueError("an edit is a mark, recorded and never rejected")
    log.append(applied.event)
    return Resumed(unedited, resume(log, applied.ws), applied.ws)


# ── The measures ──────────────────────────────────────────────────────────────


def _ops_on(run: InquiryRun, question_id: str) -> tuple[Op, ...]:
    """The operations ``run`` applied on ``question_id``, in order."""
    return tuple(
        event.step.entry for event in run.events
        if event.step.code is None and isinstance(event.step.entry, OP_CLASSES)
        and getattr(event.step.entry, "question_id", None) == question_id
    )


def _alone(values: Mapping[str, float], key: str, better: Callable[[float, float], bool]) -> bool:
    others = [value for name, value in values.items() if name != key]
    return key in values and bool(others) and all(better(values[key], value) for value in others)


def aims_at(op: Op, hypothesis_id: str) -> bool:
    """A test or an assessment of ``hypothesis_id``: a PREDICT whose claim it alone makes most
    likely, or an ASSESS whose item bears on it alone most, for or against."""
    if isinstance(op, PredictOp):
        return _alone(op.p_true_given, hypothesis_id, float.__gt__)
    if isinstance(op, AssessOp):
        values = op.likelihoods if op.likelihoods is not None else op.credences or {}
        return _alone(values, hypothesis_id, float.__gt__) or _alone(values, hypothesis_id, float.__lt__)
    return False


def _acts_on(run: InquiryRun, question_id: str, hypothesis_id: str) -> bool:
    return any(aims_at(op, hypothesis_id) for op in _ops_on(run, question_id))


def _resolved(run: InquiryRun, prediction_id: str) -> bool:
    return any(
        isinstance(e.step.entry, Resolved) and e.step.entry.prediction_id == prediction_id
        and e.step.entry.outcome in ("true", "false")
        for e in run.events
    )


def _replaced_or_moved_on(run: InquiryRun, question_id: str) -> bool:
    question = run.ws.question(question_id)
    return (
        any(isinstance(op, PredictOp) for op in _ops_on(run, question_id))
        or any(e.step.focus not in (None, question_id) for e in run.events)
        or (question is not None and question.status in MOVED_ON)
    )


def _dropped(run: InquiryRun, after_edit: Workspace, question_id: str, hypothesis_id: str) -> bool:
    finding = run.ws.finding_of(question_id)
    if finding is not None and finding.kind == "retraction" and finding.hypothesis_id == hypothesis_id:
        return True
    held = {h.id: h.credence for h in run.ws.hypotheses_of(question_id)}
    updated = len([u for u in run.ws.updates if u.question_id == question_id]) > len(
        [u for u in after_edit.updates if u.question_id == question_id]
    )
    return (
        updated and held.get(hypothesis_id, 1.0) < INJECTED_AT
        and max(held, key=lambda h: (held[h], h)) != hypothesis_id
    )


def measures(edit: Edit, before: Workspace, resumed: Resumed) -> tuple[Measure, ...]:
    """The registered measures of ``edit`` on a snapshot whose workspace was ``before``."""
    question, edited, unedited = edit.question_id, resumed.edited, resumed.unedited
    if edit.kind == "swap_lead":
        lead = str(edit.hypothesis_id)
        return (Measure("acts_on_new_lead", _acts_on(edited, question, lead), _acts_on(unedited, question, lead)),)
    if edit.kind == "delete_prediction":
        deleted = str(edit.prediction_id)
        return (
            Measure("deleted_not_resolved", not _resolved(edited, deleted), not _resolved(unedited, deleted)),
            Measure("replaced_or_moved_on", _replaced_or_moved_on(edited, question),
                    _replaced_or_moved_on(unedited, question)),
        )
    known = {h.id for h in before.hypotheses_of(question)}
    injected = next(h.id for h in resumed.after_edit.hypotheses_of(question) if h.id not in known)
    return (
        Measure("injected_tested", _acts_on(edited, question, injected), None),
        Measure("injected_dropped", _dropped(edited, resumed.after_edit, question, injected), None),
    )


# ── The test ──────────────────────────────────────────────────────────────────


def _branch(run: InquiryRun, question_id: str) -> Branch:
    held = run.ws.hypotheses_of(question_id)
    finding = run.ws.finding_of(question_id)
    return Branch(
        ops=tuple(json.dumps(op_to_json(op), sort_keys=True) for op in _ops_on(run, question_id)),
        conclusion=None if finding is None else (finding.hypothesis_id, finding.kind),
        leader=max(held, key=lambda h: (h.credence, h.id)).id if held else None,
    )


def intervention_test(
    snapshots: Sequence[Snapshot], lives: Mapping[str, Life], models: ArmModels, wrong: WrongHypothesis, *,
    core: MindCore | None = None, brief: str | None = None,
) -> tuple[tuple[InterventionResult, ...], tuple[Skipped, ...]]:
    """Each snapshot resumed after each edit that applies, against its unedited twin; every
    snapshot an edit skips is recorded."""
    core = core or mind_core()
    text = load_brief() if brief is None else brief
    results: tuple[InterventionResult, ...] = ()
    skipped: tuple[Skipped, ...] = ()
    for snap in snapshots:
        life, point, ws = lives[snap.life_id], resume_point(snap.events), core.fold(snap.events)
        at = len(snap.events)
        for kind in EDIT_KINDS:
            edit = edit_for(kind, ws, snap.question_id, wrong)
            if not isinstance(edit, Edit):
                skipped = (*skipped, Skipped(snap.life_id, at, kind, edit))
                continue
            resumed = resume_after_edit(
                snap.events, edit, mind_ports(models, life, MemoryLog(), text), point.budget,
                night=point.start.night, targets=life.targets[point.start.night], as_of=point.start.as_of,
                core=core,
            )
            results = (*results, InterventionResult(
                snap.life_id, at, edit, _branch(resumed.unedited, snap.question_id),
                _branch(resumed.edited, snap.question_id), measures(edit, ws, resumed),
            ))
    return results, skipped
