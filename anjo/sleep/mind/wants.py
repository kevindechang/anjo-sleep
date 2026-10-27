"""Wants, computed from state and never stored (Task 7a).

Built now: ``follow_up``, ``understand`` and ``share``, at the ``away`` opportunity. A want's
strength is a function of its target's stored fields, bounded to [0, 1]. Time alone never creates
or strengthens one: the clock only matters through a date the person set (``check_after``), and
nothing here reads it (``as_of`` is passed in).

``explore``, ``repair``, ``know`` and ``settle``, the ``open`` and ``turn`` opportunities and
learned weights are autonomy Phase 2.
"""
from __future__ import annotations

from datetime import datetime
from typing import Final

from anjo.sleep.mind.snapshot import LIVE_QUESTION, continuing_targets
from anjo.sleep.types import QuestionView, Result, Target, Want, WantSnapshot

COMMITMENT_CLASS: Final = "user_commitment"
THREAD_CLASS: Final = "active_thread"

PREPARE_STRENGTH: Final = 0.6  # a plan she hasn't prepared for
PREPARE_DUE_STRENGTH: Final = 0.9  # ... whose date the person set has come
NOTE_DUE_STRENGTH: Final = 0.8  # a due plan, prepared
SHARE_BASE: Final = 0.6  # a result she can say; each source (or own item cited) adds a tenth
SHARE_STEP: Final = 0.1
SHARE_STEPS: Final = 3
MAX_INQUIRY_PRIORITY: Final = 2.0  # priority is at most 2, so half of it stays on one 0-1 scale


def derive_wants(snapshot: WantSnapshot) -> tuple[Want, ...]:
    """Every want the snapshot's state pulls, in want-id order."""
    wants: list[Want] = []
    for target in snapshot.targets:
        if target.status != "open":
            continue
        pull = _follow_up(target, snapshot) if target.kind == "commitment" else _understand(target, snapshot)
        if pull is not None:
            wants.append(pull)
    kinds = {t.id: t.kind for t in snapshot.targets}
    for target_id in sorted({r.target_id for r in snapshot.results}):
        note = _share(target_id, kinds.get(target_id, "thread"), snapshot)
        if note is not None:
            wants.append(note)
    return tuple(sorted(wants, key=lambda want: want.id))


def _due(target: Target, as_of: datetime) -> bool:
    return target.check_after is not None and target.check_after <= as_of


def _preparation_open(target: Target, snapshot: WantSnapshot) -> bool:
    """Open until a result with a verified source lands on the commitment, consumed or not. It
    also needs a revision a study hasn't consumed, so a study that found no source waits for the
    person to say something new."""
    if target.unacted <= 0:
        return False
    return not any(r.target_id == target.id and r.verified_sources >= 1 for r in snapshot.results)


def _follow_up(target: Target, snapshot: WantSnapshot) -> Want | None:
    due = _due(target, snapshot.as_of)
    if _preparation_open(target, snapshot):
        return Want(
            id=f"follow_up:{target.id}", kind="follow_up", target_id=target.id,
            action="prepare_commitment", priority_class=COMMITMENT_CLASS,
            strength=PREPARE_DUE_STRENGTH if due else PREPARE_STRENGTH,
        )
    if due and not target.due_noted:
        return Want(
            id=f"follow_up:{target.id}", kind="follow_up", target_id=target.id,
            action="hold_note", priority_class=COMMITMENT_CLASS, strength=NOTE_DUE_STRENGTH,
        )
    return None


def _inquiring(target: Target, questions: tuple[QuestionView, ...]) -> QuestionView | None:
    """The open or active question on ``target`` with the largest priority (smallest id on a tie).
    With no new evidence only a question that continues may be chosen: one that isn't ready stays
    asleep."""
    covering = [
        q for q in questions
        if q.status in LIVE_QUESTION and target.id in q.target_ids and (target.unacted > 0 or q.continues)
    ]
    return min(covering, key=lambda q: (-q.priority, q.id), default=None)


def _understand(target: Target, snapshot: WantSnapshot) -> Want | None:
    question = _inquiring(target, snapshot.questions)
    continuing = target.id in continuing_targets(snapshot.questions)
    if target.unacted <= 0 and not continuing:
        return None
    if question is not None:
        priority = min(max(question.priority, 0.0), MAX_INQUIRY_PRIORITY)
        return Want(
            id=f"understand:{target.id}", kind="understand", target_id=target.id,
            action="inquire", priority_class=THREAD_CLASS,
            strength=priority / MAX_INQUIRY_PRIORITY, question_id=question.id,
        )
    if target.unacted <= 0:
        return None
    # n / (n + 1) grows with the person's unacted revisions and stays inside [0, 1); a thread she
    # already understands pulls less.
    unacted = target.unacted
    confidence = 0.0 if target.confidence is None else target.confidence
    return Want(
        id=f"understand:{target.id}", kind="understand", target_id=target.id,
        action="revisit_thread", priority_class=THREAD_CLASS,
        strength=(unacted / (unacted + 1)) * (1.0 - confidence),
    )


def _share(target_id: str, kind: str, snapshot: WantSnapshot) -> Want | None:
    """A committed result that changed an open target since the person last spoke. The snapshot
    admits results only on open targets and passes every open commitment, so a result whose target
    isn't in ``snapshot.targets`` is on a thread that has nothing new to study."""
    pending = [
        r for r in snapshot.results
        if r.target_id == target_id and not r.consumed and _after_last_message(r, snapshot)
    ]
    if not pending:
        return None
    return Want(
        id=f"share:{target_id}", kind="share", target_id=target_id, action="hold_note",
        priority_class=COMMITMENT_CLASS if kind == "commitment" else THREAD_CLASS,
        strength=max(_share_strength(r) for r in pending),
    )


def _after_last_message(result: Result, snapshot: WantSnapshot) -> bool:
    return snapshot.last_message_at is None or result.created_at > snapshot.last_message_at


def _share_strength(result: Result) -> float:
    count = result.verified_sources if result.basis == "research" else result.user_cites
    return SHARE_BASE + SHARE_STEP * min(count, SHARE_STEPS)
