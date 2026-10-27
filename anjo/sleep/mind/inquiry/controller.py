"""MIND's controller: which question gets the next cycle, and when the night's inquiry stops
(Task 17). Pure: it reads the workspace, the targets and ``as_of`` it is given.

``priority(q) = stake x u x a`` is a heuristic, not a value of information:

- **stake** is ``STAKE_DUE`` when one of the question's targets is an open commitment due within
  ``STAKE_WINDOW`` of ``as_of``, else ``STAKE_NONE``;
- **u** is 1.0 before hypotheses exist, else the set's normalized entropy, floored at ``U_FLOOR``;
- **a** is ``A_READY`` when a next step is ready (no hypotheses yet, no prediction yet, a pending
  prediction with evidence the resolver hasn't seen, an unassessed search hit, a resolution or a
  retraction ready), else ``A_WAITING``.

A question ready to resolve or retract scores ``stake x 1.0``, so a settled belief isn't starved
by its low entropy.

``decide`` stays on its ``focus`` while that question is live, not stalled and above
``PRIORITY_FLOOR``; otherwise it switches to the live question with the top priority (a tie goes
to the older question), or stops at a cap or when nothing clears the floor. **K5:** a question
whose latest ``STALL_LIMIT`` cycles tonight made no progress gets a ``park`` decision (reason
``stalled``), and ``decide`` passes over it for the rest of the night. Progress is an item
assessed, a prediction committed or resolved true or false, or a hypothesis added, whatever it
does to entropy, so counter-evidence counts. The park changes no status: the workspace can't wake
a parked question on new evidence about its targets (``EvidenceRef`` names no target), so the
guard sets it aside for the night instead, and the next night counts afresh. Only the thinker's
``PARK`` parks a question. M-control (``control="newest"``) works the newest live question for
the whole budget, with neither the floor nor K5.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import (
    A_READY,
    A_WAITING,
    LIVE_STATUSES,
    PRIORITY_FLOOR,
    STAKE_DUE,
    STAKE_NONE,
    STAKE_WINDOW,
    STALL_LIMIT,
    U_FLOOR,
    AssessOp,
    Budget,
    ControlMode,
    Decision,
    DecisionKind,
    DecisionReason,
    Hypothesis,
    HypothesizeOp,
    PredictOp,
    Priority,
    Question,
    Resolved,
    RunStart,
    Step,
    Workspace,
)
from anjo.sleep.types import Target

_PROGRESS_OPS = (AssessOp, PredictOp, HypothesizeOp)


def due(target: Target, as_of: datetime) -> bool:
    """An open commitment whose date falls within ``STAKE_WINDOW`` of ``as_of``."""
    if target.kind != "commitment" or target.status != "open" or target.check_after is None:
        return False
    return timedelta(0) <= target.check_after - as_of <= STAKE_WINDOW


def uncertainty(hypotheses: Sequence[Hypothesis]) -> float:
    """u: 1.0 before hypotheses exist, else normalized entropy floored at ``U_FLOOR``."""
    if not hypotheses:
        return 1.0
    entropy = -math.fsum(h.credence * math.log(h.credence) for h in hypotheses if h.credence > 0.0)
    return min(1.0, max(U_FLOOR, entropy / math.log(len(hypotheses))))


def _step_ready(ws: Workspace, question: Question, as_of: datetime) -> bool:
    hypotheses = ws.hypotheses_of(question.id)
    predictions = [p for p in ws.predictions if p.question_id == question.id]
    if not hypotheses or not predictions:
        return True
    if any(p.pending and kernel.unseen(ws, p, as_of) for p in predictions):
        return True
    used = kernel.used_items(ws, question.id)
    return any(item.id not in used for item in kernel.hits(ws, question.id))


def priority(ws: Workspace, question_id: str, *, targets: Sequence[Target], as_of: datetime) -> Priority:
    """stake x u x a for one question; ``ValueError`` if the workspace doesn't hold it."""
    question = ws.question(question_id)
    if question is None:
        raise ValueError(f"question {question_id} isn't held")
    held = frozenset(question.target_ids)
    stake = STAKE_DUE if any(t.id in held and due(t, as_of) for t in targets) else STAKE_NONE
    ready = kernel.resolution_ready(ws, question_id) or kernel.retraction_ready(ws, question_id)
    a = A_READY if ready or _step_ready(ws, question, as_of) else A_WAITING
    return Priority(question_id, stake, uncertainty(ws.hypotheses_of(question_id)), a, ready)


# ── K5: the progress guard ────────────────────────────────────────────────────


def _night(ws: Workspace) -> tuple[Step, ...]:
    """The steps of the latest run's night: from the first ``RunStart`` naming it."""
    starts = [(n, step.entry) for n, step in enumerate(ws.history) if isinstance(step.entry, RunStart)]
    if not starts:
        return ()
    night = starts[-1][1].night
    first = next(n for n, entry in starts if entry.night == night)
    return ws.history[first:]


def _progress(step: Step) -> bool:
    if isinstance(step.entry, Resolved):
        return step.entry.outcome in ("true", "false")
    return step.applied and isinstance(step.entry, _PROGRESS_OPS)


def _cycles(ws: Workspace) -> list[tuple[str, bool]]:
    """Each cycle of the night, in order: its focus and whether it made progress. A cycle is a run
    of consecutive steps with the same ``Step.cycle``."""
    cycles: list[tuple[int, str, bool]] = []
    for step in _night(ws):
        if step.cycle is None or step.focus is None:
            continue
        if cycles and cycles[-1][0] == step.cycle and cycles[-1][1] == step.focus:
            number, focus, moved = cycles[-1]
            cycles[-1] = (number, focus, moved or _progress(step))
        else:
            cycles.append((step.cycle, step.focus, _progress(step)))
    return [(focus, moved) for _, focus, moved in cycles]


def stalled(ws: Workspace, question_id: str) -> bool:
    """K5: the question's latest ``STALL_LIMIT`` cycles tonight made no progress."""
    count = 0
    for focus, moved in reversed(_cycles(ws)):
        if focus != question_id:
            continue
        if moved:
            break
        count += 1
    return count >= STALL_LIMIT


# ── The decision ──────────────────────────────────────────────────────────────


def decide(
    ws: Workspace,
    budget: Budget,
    *,
    focus: str | None,
    targets: Sequence[Target],
    as_of: datetime,
    control: ControlMode = "priority",
) -> Decision:
    """The next cycle's question, or stop, with every live question's priority and the budget."""
    live = [q for q in ws.questions if q.status in LIVE_STATUSES]
    priorities = tuple(priority(ws, q.id, targets=targets, as_of=as_of) for q in live)

    def made(kind: DecisionKind, question_id: str | None, reason: DecisionReason | None = None) -> Decision:
        return Decision(kind, question_id, reason, priorities, budget)

    cap = budget.stop_reason
    if cap is not None:
        return made("stop", None, cap)
    if not live:
        return made("stop", None, "below_floor")
    if control == "newest":
        newest = live[-1].id
        return made("continue" if newest == focus else "investigate", newest)
    by_id = {p.question_id: p for p in priorities}
    if focus in by_id:
        if stalled(ws, focus):
            return made("park", focus, "stalled")
        if by_id[focus].value >= PRIORITY_FLOOR:
            return made("continue", focus)
    open_ones = [p for p in priorities if p.question_id != focus and not stalled(ws, p.question_id)]
    best = max(open_ones, key=lambda p: p.value, default=None)
    if best is None or best.value < PRIORITY_FLOOR:
        return made("stop", None, "below_floor")
    return made("investigate", best.question_id)
