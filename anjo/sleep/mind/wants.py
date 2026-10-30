"""Wants, computed from state and never stored (Task 7a, Task 18).

A want's strength is a function of its target's stored fields or of her own inquiry's state,
bounded to [0, 1]. Two families of pull:

- **the person's**, at the ``away`` opportunity: ``follow_up``, ``understand`` and ``share`` from
  their revisions, dates and results. The clock reaches these only through a date they set
  (``check_after``);
- **her own**, whenever MIND's inquiry is on: ``settle``, ``know`` and ``share`` from her own
  workspace (``own.py``). A week with no messages still pulls, because a prediction she committed
  comes due, two alternatives she holds stay neck and neck, a question she hasn't worked stays
  uncertain, or her own finding bears on the person's live plan. Nothing here blocks an action
  derived from her state: the person's own settings (``paused_by_you``, ``quiet_hours``) and the
  night's allowance reach ``choose`` for it as they do for every other want, as an action not
  offered or no budget.

``ANJO_SLEEP_MIND_OWN_WANTS=0`` (``own_wants_on``) turns every own source off, for the ablation:
her wants then come only from the person again. ``explore`` and ``repair``, the ``open`` and
``turn`` opportunities and learned weights are still later autonomy phases.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Final, Literal

from anjo.sleep.mind.inquiry.types import LIVE_STATUSES, STAKE_DUE
from anjo.sleep.mind.snapshot import CLOSED_QUESTION, EVIDENCE_WINDOW, LIVE_QUESTION, continuing_targets
from anjo.sleep.types import OwnView, QuestionView, Result, Target, Want, WantSnapshot

OwnKind = Literal["know", "settle"]

COMMITMENT_CLASS: Final = "user_commitment"
THREAD_CLASS: Final = "active_thread"

PREPARE_STRENGTH: Final = 0.6  # a plan she hasn't prepared for
PREPARE_DUE_STRENGTH: Final = 0.9  # ... whose date the person set has come
NOTE_DUE_STRENGTH: Final = 0.8  # a due plan, prepared
SHARE_BASE: Final = 0.6  # a result she can say; each source (or own item cited) adds a tenth
SHARE_STEP: Final = 0.1
SHARE_STEPS: Final = 3
MAX_INQUIRY_PRIORITY: Final = 2.0  # priority is at most 2, so half of it stays on one 0-1 scale

# ── Her own pulls (Task 18): named bounds, each derived from her state, never tuned ────────────
OWN_WANTS_ENV: Final = "ANJO_SLEEP_MIND_OWN_WANTS"  # "0": only the person's state pulls again
SETTLE_DUE_SHARE: Final = 0.5  # what a prediction pulls the moment its own date comes
SETTLE_OVERDUE_FULL: Final = timedelta(days=14)  # a fortnight overdue pulls it all the way
CONTRADICTION_GAP: Final = 0.15  # her top two alternatives this close: neither leads
KNOW_STALE_FULL: Final = timedelta(days=7)  # a week unworked pulls a know all the way


def own_wants_on(env: Mapping[str, str]) -> bool:
    """Her own pulls' switch: on unless ``ANJO_SLEEP_MIND_OWN_WANTS`` says "0"."""
    return env.get(OWN_WANTS_ENV, "").strip() != "0"


def derive_wants(snapshot: WantSnapshot) -> tuple[Want, ...]:
    """Every want the snapshot's state pulls, in want-id order."""
    wants: list[Want] = []
    for target in snapshot.targets:
        if target.status != "open":
            continue
        pull = _follow_up(target, snapshot) if target.kind == "commitment" else _understand(target, snapshot)
        if pull is not None:
            # Only inquiry-enabled snapshots carry questions. Plans outside the existing
            # seven-day follow-up window compete on strength while she has live inquiry work.
            if (snapshot.inquiry_offered and target.kind == "commitment"
                    and any(q.status in LIVE_QUESTION for q in snapshot.questions)
                    and (target.check_after is None or target.check_after > snapshot.as_of + EVIDENCE_WINDOW)):
                pull = replace(pull, priority_class=THREAD_CLASS)
            wants.append(pull)
        if (snapshot.inquiry_offered and target.kind == "commitment"
                and _inquiring(target, snapshot.questions) is not None):
            inquiry = _understand(target, snapshot)
            if inquiry is not None:
                wants.append(inquiry)
    kinds = {t.id: t.kind for t in snapshot.targets}
    for target_id in sorted({r.target_id for r in snapshot.results}):
        note = _share(target_id, kinds.get(target_id, "thread"), snapshot)
        if note is not None:
            wants.append(note)
    if snapshot.own_wants:
        # A result waiting to be said is already the note on its target; her own note never repeats it.
        said = {want.target_id for want in wants if want.kind == "share"}
        for view in snapshot.own:
            wants.extend(_own_pull(view, snapshot, said))
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
    """Made since the person last wrote. A night stamps the day's evidence and its own results at the
    same moment (its window's end, or its clock), so a result stamped then is news."""
    return snapshot.last_message_at is None or result.created_at >= snapshot.last_message_at


def _share_strength(result: Result) -> float:
    count = result.verified_sources if result.basis == "research" else result.user_cites
    return SHARE_BASE + SHARE_STEP * min(count, SHARE_STEPS)


# ── Her own pulls (Task 18) ───────────────────────────────────────────────────


def _own_pull(view: OwnView, snapshot: WantSnapshot, said: set[str]) -> tuple[Want, ...]:
    """What one of her questions pulls at the ``away`` opportunity: at most one inquiry (a settle
    before a know) while the night may inquire, and at most one note. ``said`` is the targets a
    result's share already covers."""
    if view.status in CLOSED_QUESTION:
        note = _own_note(view, snapshot, said)  # a settled question still has something to say
        return () if note is None else (note,)
    target_id = view.target_ids[0]  # the first target, as an inquiry's study target is
    matters = min(view.stake / STAKE_DUE, 1.0)  # a due plan's question outranks a casual one's
    inquiry = None
    if snapshot.inquiry_offered:
        inquiry = _settle(view, snapshot, target_id, matters) or _know(view, snapshot, target_id, matters)
    note = _own_note(view, snapshot, said)
    return tuple(want for want in (inquiry, note) if want is not None)


def _settle(view: OwnView, snapshot: WantSnapshot, target_id: str, matters: float) -> Want | None:
    """A prediction whose own date has come, stronger the more overdue, or two alternatives she
    can't tell apart. She made the first commitment; the second is a question that isn't settled."""
    if view.prediction_due is not None and view.prediction_due <= snapshot.as_of:
        progress = min((snapshot.as_of - view.prediction_due) / SETTLE_OVERDUE_FULL, 1.0)
        strength = matters * (SETTLE_DUE_SHARE + (1.0 - SETTLE_DUE_SHARE) * progress)
        return _inquiry("settle", view, snapshot, target_id, strength)
    top = sorted(view.credences, reverse=True)[:2]
    if len(top) == 2 and top[0] - top[1] <= CONTRADICTION_GAP:
        return _inquiry("settle", view, snapshot, target_id, matters * view.uncertainty)
    return None


def _know(view: OwnView, snapshot: WantSnapshot, target_id: str, matters: float) -> Want | None:
    """A live question with high uncertainty she hasn't worked lately: uncertainty x staleness,
    both on her own clock."""
    if view.status not in LIVE_STATUSES:
        return None  # a parked or closed question stays asleep without something new
    stale = min(max((snapshot.as_of - view.last_worked_at) / KNOW_STALE_FULL, 0.0), 1.0)
    strength = matters * view.uncertainty * stale
    return None if strength <= 0.0 else _inquiry("know", view, snapshot, target_id, strength)


def _inquiry(kind: OwnKind, view: OwnView, snapshot: WantSnapshot, target_id: str, strength: float) -> Want:
    return Want(
        id=f"{kind}:{view.question_id}", kind=kind, target_id=target_id, action="inquire",
        priority_class=_own_class(target_id, snapshot), strength=min(strength, 1.0),
        question_id=view.question_id,
    )


def _own_note(view: OwnView, snapshot: WantSnapshot, said: set[str]) -> Want | None:
    """A finding of hers, or a prediction she settled, that bears on the person's live plan and that
    no result's share already says. A plan she has already spoken to (``due_noted``) is not news
    again, the same way a consumed result isn't."""
    if not (view.finding or view.resolved_prediction):
        return None
    plan = next((t for t in snapshot.targets if t.id in view.target_ids and t.kind == "commitment"), None)
    if plan is None or plan.status != "open" or plan.due_noted or plan.id in said:
        return None
    sources = int(view.finding) + int(view.resolved_prediction)
    return Want(
        id=f"share:{plan.id}", kind="share", target_id=plan.id, action="hold_note",
        priority_class=COMMITMENT_CLASS,
        strength=min(SHARE_BASE + SHARE_STEP * min(sources, SHARE_STEPS), 1.0),
    )


def _own_class(target_id: str, snapshot: WantSnapshot) -> str:
    target = next((t for t in snapshot.targets if t.id == target_id), None)
    return COMMITMENT_CLASS if target is not None and target.kind == "commitment" else THREAD_CLASS
