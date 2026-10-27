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
thinker should run at temperature 0, or sampling noise adds to it). Measured: whether the next
operation differs, and whether the question's conclusion does.
"""
from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Iterator, Mapping, Sequence
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
    Budget,
    Edit,
    EditKind,
    Hypothesis,
    InquiryRun,
    Question,
    Rejected,
    RunStart,
    Step,
    Workspace,
    WorkspaceEvent,
    op_to_json,
)
from anjo.sleep.types import Target

SkipReason = Literal["tied_lead", "no_pending_prediction", "four_alternatives"]
EDIT_KINDS: Final[tuple[EditKind, ...]] = ("swap_lead", "delete_prediction", "inject_hypothesis")


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
    """What one branch did after the snapshot: every operation it proposed (canonical JSON), and
    the focus question's finding (kind, hypothesis) and leading hypothesis at its end."""

    ops: tuple[str, ...]
    finding: tuple[str, str | None] | None
    leader: str | None


@dataclass(frozen=True)
class InterventionResult:
    life_id: str
    at: int  # the snapshot's length in events
    edit: Edit
    unedited: Branch
    edited: Branch

    @property
    def next_op_differs(self) -> bool:
        return self.unedited.ops[:1] != self.edited.ops[:1]

    @property
    def conclusion_differs(self) -> bool:
        return self.unedited.finding != self.edited.finding


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
) -> tuple[InquiryRun, InquiryRun]:
    """Resume the snapshot ``events`` twice, unedited and after ``edit`` (logged as a step stamped
    ``as_of``), each on its own fresh in-memory log seeded with ``events``, both on ``ports``'
    evidence at ``as_of``, focused on the edited question, under the snapshot run's rules and
    control. Returns (unedited, edited)."""
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
    return unedited, resume(log, applied.ws)


# ── The test ──────────────────────────────────────────────────────────────────


def _branch(run: InquiryRun, question_id: str) -> Branch:
    held = run.ws.hypotheses_of(question_id)
    finding = run.ws.finding_of(question_id)
    return Branch(
        ops=tuple(
            json.dumps(op_to_json(event.step.entry), sort_keys=True)
            for event in run.events if isinstance(event.step.entry, OP_CLASSES)
        ),
        finding=None if finding is None else (finding.kind, finding.hypothesis_id),
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
            unedited, edited = resume_after_edit(
                snap.events, edit, mind_ports(models, life, MemoryLog(), text), point.budget,
                night=point.start.night, targets=life.targets[point.start.night], as_of=point.start.as_of,
                core=core,
            )
            results = (*results, InterventionResult(
                snap.life_id, at, edit, _branch(unedited, snap.question_id), _branch(edited, snap.question_id),
            ))
    return results, skipped
