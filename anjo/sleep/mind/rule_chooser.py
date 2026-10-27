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

from anjo.sleep.mind.choose import rank_key, record_of, require_codes
from anjo.sleep.mind.policy import LOST, NO_BUDGET, NOT_OFFERED, OVER_CAP, UNKNOWN_CLASS
from anjo.sleep.types import (
    Choice,
    ChoiceContext,
    Considered,
    Opportunity,
    Target,
    Want,
    WantSnapshot,
)

THREAD_CLASS: Final = "active_thread"
STUDY: Final = "revisit_thread"

# The reason codes this chooser writes. A policy must allowlist every one of them.
REASON_CODES = frozenset({UNKNOWN_CLASS, NOT_OFFERED, OVER_CAP, NO_BUDGET, LOST})


class RuleChooser:
    """The open thread with the largest unacted revision; never a note."""

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        """One ``understand`` want per open thread that changed since a study last acted on it."""
        return tuple(_thread_want(t) for t in snapshot.targets if _eligible(t))

    def choose(self, opportunity: Opportunity, wants: Sequence[Want], ctx: ChoiceContext) -> Choice:
        """The strongest offered want that the allowance can pay for, or nothing."""
        require_codes(ctx.policy, REASON_CODES)
        ranked = sorted(wants, key=lambda want: rank_key(want, ctx.policy))
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
            considered.append(record_of(want, reason))
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
