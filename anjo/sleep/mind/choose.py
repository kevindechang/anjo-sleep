"""The chooser (Task 7a): one pure function that picks at most one want, and says why not the rest.

It keeps the core plan's order: priority class, then effective strength, then base strength, then
want id, so the choice never depends on the order wants arrive in. At most ``max_candidates`` are
considered; the rest are rejected with ``candidate_limit``. Picking nothing is a normal outcome.

For each want the first rule that applies decides, in this order:

1. its action isn't offered (``action_not_offered``, before any other hold);
2. its class isn't in the policy (``unknown_priority_class``);
3. it is past the candidate cap (``candidate_limit``);
4. it pulls less than its bar (``below_threshold``): ``away_min`` for a study or an inquiry,
   ``surface_min`` for a held note;
5. the night can't pay for it (``budget``);
6. a stronger want already won (``lost_to_target``).

No clock, database, model call or environment read: the same wants and context give the same
choice, in a live night or an accelerated one.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from anjo.sleep.mind.policy import (
    BELOW_THRESHOLD,
    LOST,
    NO_BUDGET,
    NOT_OFFERED,
    OVER_CAP,
    UNKNOWN_CLASS,
)
from anjo.sleep.mind.wants import derive_wants
from anjo.sleep.types import (
    Choice,
    ChoiceContext,
    Considered,
    ConsideredStatus,
    Opportunity,
    Policy,
    Want,
    WantSnapshot,
)

# Only a homework study (and an inquiry's research calls) spend Wonder; a note and an inquiry's
# own thinking stay available when it is spent.
WONDER_ACTIONS: Final = frozenset({"prepare_commitment", "revisit_thread"})
NOTE_ACTION: Final = "hold_note"
REJECTIONS: Final = frozenset({UNKNOWN_CLASS, OVER_CAP})
REQUIRED_CODES: Final = frozenset({UNKNOWN_CLASS, NOT_OFFERED, OVER_CAP, BELOW_THRESHOLD, NO_BUDGET, LOST})


class StateChooser:
    """Her chooser behind the ``Chooser`` port, in the place of ``RuleChooser``."""

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        return derive_wants(snapshot)

    def choose(self, opportunity: Opportunity, wants: Sequence[Want], ctx: ChoiceContext) -> Choice:
        return choose(opportunity, wants, ctx)


def choose(opportunity: Opportunity, wants: Sequence[Want], ctx: ChoiceContext) -> Choice:
    """The strongest offered want that clears its bar and the night's budget, or nothing."""
    require_codes(ctx.policy, REQUIRED_CODES)
    ranked = sorted(wants, key=lambda want: rank_key(want, ctx.policy))
    winner: Want | None = None
    in_cap = 0  # candidates inside the cap so far: offered wants of a known class
    considered: list[Considered] = []
    for want in ranked:
        reason = _held_reason(want, ctx, in_cap)
        if reason not in (NOT_OFFERED, UNKNOWN_CLASS, OVER_CAP):
            in_cap += 1
        if reason is None and winner is not None:
            reason = LOST
        elif reason is None:
            winner = want
        considered.append(record_of(want, reason))
    return Choice(opportunity=opportunity, want=winner, considered=tuple(considered))


def require_codes(policy: Policy, needed: frozenset[str]) -> None:
    missing = needed - policy.reason_codes
    if missing:
        raise ValueError(f"the policy's reason codes lack {sorted(missing)}")


def rank_key(want: Want, policy: Policy) -> tuple[int, float, float, str]:
    """Class, then effective strength, then base strength, then id; an unknown class sorts last."""
    classes = policy.priority_classes
    rank = classes.index(want.priority_class) if want.priority_class in classes else len(classes)
    return (rank, -want.effective_strength, -want.strength, want.id)


def record_of(want: Want, reason: str | None) -> Considered:
    status: ConsideredStatus
    if reason is None:
        status = "won"
    elif reason in REJECTIONS:
        status = "rejected"
    else:
        status = "held_back"
    return Considered(
        want_id=want.id, kind=want.kind, target_id=want.target_id, action=want.action,
        strength=want.strength, weight=want.weight, status=status, reason=reason,
    )


def _held_reason(want: Want, ctx: ChoiceContext, in_cap: int) -> str | None:
    """Why ``want`` can't win, checked in order, or ``None`` when it can."""
    if not ctx.allows(want.action):
        return NOT_OFFERED
    if want.priority_class not in ctx.policy.priority_classes:
        return UNKNOWN_CLASS
    if in_cap >= ctx.policy.max_candidates:
        return OVER_CAP
    bar = ctx.policy.surface_min if want.action == NOTE_ACTION else ctx.policy.away_min
    if want.effective_strength < bar:
        return BELOW_THRESHOLD
    allowance = ctx.allowance
    if allowance.actions_left == 0 or (want.action in WONDER_ACTIONS and allowance.available_model_calls == 0):
        return NO_BUDGET
    return None
