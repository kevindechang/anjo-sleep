"""The rule chooser: the spec §6a fallback behind the ``Chooser`` port (Task 1).

Until Task 7a's chooser is integrated, the night studies the open thread with the most unacted
change (``revision - acted_revision``) and forms no note: it never picks ``hold_note``. Like the
chooser it stands in for, it is pure (no clock, database, model call or environment read), and
every candidate that doesn't win gets an allowlisted reason code in the choice's receipt.

Candidates are sorted by priority class, then effective strength, then base strength, then want
id, so the choice never depends on the order wants arrive in.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from anjo.sleep.types import (
    Choice,
    ChoiceContext,
    Considered,
    ConsideredStatus,
    Opportunity,
    Policy,
    Target,
    Want,
    WantSnapshot,
)

THREAD_CLASS: Final = "active_thread"
STUDY: Final = "revisit_thread"

# The reason codes this chooser writes. A policy must allowlist every one of them.
UNKNOWN_CLASS = "unknown_priority_class"
NOT_OFFERED = "action_not_offered"
OVER_CAP = "candidate_limit"
NO_BUDGET = "budget"
LOST = "lost_to_target"
REASON_CODES = frozenset({UNKNOWN_CLASS, NOT_OFFERED, OVER_CAP, NO_BUDGET, LOST})
_REJECTIONS = frozenset({UNKNOWN_CLASS, OVER_CAP})


class RuleChooser:
    """The open thread with the largest unacted revision; never a note."""

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        """One ``understand`` want per open thread that changed since a study last acted on it."""
        return tuple(_thread_want(t) for t in snapshot.targets if _eligible(t))

    def choose(self, opportunity: Opportunity, wants: Sequence[Want], ctx: ChoiceContext) -> Choice:
        """The strongest offered want that the allowance can pay for, or nothing."""
        _require_reason_codes(ctx.policy)
        ranked = sorted(wants, key=lambda want: _order(want, ctx.policy))
        winner: Want | None = None
        offered = 0  # candidates inside the cap so far: offered wants of a known class
        considered: list[Considered] = []
        for want in ranked:
            reason = _held_reason(want, ctx, offered)
            if reason is None or reason == NO_BUDGET:
                offered += 1
            if reason is None and winner is not None:
                reason = LOST
            elif reason is None:
                winner = want
            considered.append(_considered(want, reason))
        return Choice(opportunity=opportunity, want=winner, considered=tuple(considered))


def _eligible(target: Target) -> bool:
    return target.kind == "thread" and target.status == "open" and target.unacted > 0


def _thread_want(target: Target) -> Want:
    # n / (n + 1): monotone in the unacted revisions and inside [0, 1), so sorting by strength
    # sorts by unacted revision.
    unacted = target.unacted
    return Want(
        id=f"understand:{target.id}",
        kind="understand",
        target_id=target.id,
        action=STUDY,
        priority_class=THREAD_CLASS,
        strength=unacted / (unacted + 1),
    )


def _require_reason_codes(policy: Policy) -> None:
    missing = REASON_CODES - policy.reason_codes
    if missing:
        raise ValueError(f"the policy's reason codes lack {sorted(missing)}")


def _order(want: Want, policy: Policy) -> tuple[int, float, float, str]:
    classes = policy.priority_classes
    rank = classes.index(want.priority_class) if want.priority_class in classes else len(classes)
    return (rank, -want.effective_strength, -want.strength, want.id)


def _held_reason(want: Want, ctx: ChoiceContext, offered: int) -> str | None:
    """Why ``want`` can't win, checked in order, or ``None`` when it can."""
    if want.priority_class not in ctx.policy.priority_classes:
        return UNKNOWN_CLASS
    if want.action != STUDY or not ctx.allows(want.action):
        return NOT_OFFERED
    if offered >= ctx.policy.max_candidates:
        return OVER_CAP
    if ctx.allowance.available_model_calls == 0:
        return NO_BUDGET
    return None


def _considered(want: Want, reason: str | None) -> Considered:
    status: ConsideredStatus
    if reason is None:
        status = "won"
    elif reason in _REJECTIONS:
        status = "rejected"
    else:
        status = "held_back"
    return Considered(
        want_id=want.id,
        kind=want.kind,
        target_id=want.target_id,
        action=want.action,
        strength=want.strength,
        weight=want.weight,
        status=status,
        reason=reason,
    )
