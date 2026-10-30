"""Her own inquiry state, as her own wants read it (Task 18). Pure: every time is on her clock.

``view`` is the one place a question's alternatives, commitments to the world and last working are
read, so ``wants`` weighs her own pulls from the same facts the controller and the render see:

- **credences** are the question's alternatives, the catch-all out. The kernel holds a set as one
  normalized vector without marking which alternatives contradict which, so a standoff between the
  top two is what a contradiction looks like from her state.
- **stake** and **uncertainty** come from ``controller.priority``'s parts, never recomputed here:
  the night's inquiry and her wants must weigh a question the same way.
- **prediction_due** is the earliest horizon among predictions she committed and hasn't settled.
  It is her own date, not the person's: it is the moment she said she would know.
- **resolved_prediction** and **finding** say her own thinking produced something; which kinds
  count as a finding is ``findings.SHARED_KINDS``, the kinds the night may share.
- **last_worked_at** is the latest step that took the question up, else the moment it opened.
"""
from __future__ import annotations

from typing import Final

from anjo.sleep.mind.inquiry.findings import SHARED_KINDS
from anjo.sleep.mind.inquiry.types import Priority, Question, Workspace
from anjo.sleep.types import OwnView

RESOLVED_OUTCOMES: Final[frozenset[str]] = frozenset({"true", "false"})  # she called it, and it settled


def view(ws: Workspace, question: Question, parts: Priority) -> OwnView:
    """The own view of one of her questions, with the priority's parts the night computed."""
    predictions = tuple(p for p in ws.predictions if p.question_id == question.id)
    pending = tuple(p.horizon for p in predictions if p.pending)
    worked = [step.at for step in ws.history if step.focus == question.id]
    return OwnView(
        question_id=question.id,
        target_ids=question.target_ids,
        status=question.status,
        credences=tuple(h.credence for h in ws.hypotheses_of(question.id) if not h.is_catch_all),
        stake=parts.stake,
        uncertainty=parts.u,
        prediction_due=min(pending, default=None),
        resolved_prediction=any(p.outcome in RESOLVED_OUTCOMES for p in predictions),
        finding=any(f.question_id == question.id and f.kind in SHARED_KINDS for f in ws.findings),
        last_worked_at=max(worked, default=question.opened_at),
    )
