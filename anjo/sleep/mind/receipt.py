"""The receipt of one choice (Task 7a): ids, kinds, strengths, statuses, the outcome and
allowlisted reason codes, never text.

Kevin's receipts are never exported or published; synthetic people's are (spec §12.4). Either way
a receipt can't carry what she was thinking about: a reason that isn't an allowlisted code is
refused, and nothing here reads a target's label.
"""
from __future__ import annotations

from typing import Any, Final, Literal

from anjo.sleep.mind.policy import (
    BELOW_THRESHOLD,
    LOST,
    NO_BUDGET,
    NOT_OFFERED,
    NOTE_DECLINED,
    OVER_CAP,
    UNKNOWN_CLASS,
)
from anjo.sleep.types import Choice

Outcome = Literal["ran", "declined", "budget", "failed"]

RECEIPT_REASONS: Final = frozenset({
    UNKNOWN_CLASS, NOT_OFFERED, OVER_CAP, BELOW_THRESHOLD, NO_BUDGET, LOST, NOTE_DECLINED,
})
OUTCOMES: Final = frozenset({"ran", "declined", "budget", "failed"})
_OUTCOME_REASONS: Final[dict[str, str]] = {"declined": NOTE_DECLINED, "budget": NO_BUDGET}


def record(choice: Choice, outcome: Outcome | None) -> dict[str, Any]:
    """The receipt for ``choice``. ``outcome`` is what became of the winner: ``None`` when nothing
    won or nothing has run yet. A writer's decline is recorded as ``note_declined``."""
    if outcome is not None and (not isinstance(outcome, str) or outcome not in OUTCOMES):
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}, got a {type(outcome).__name__}")
    if outcome is not None and choice.want is None:
        raise ValueError("an outcome is recorded for a winner, and nothing won")
    if outcome == "declined" and choice.action != "hold_note":
        raise ValueError("only a held note can be declined")
    for candidate in choice.considered:
        if candidate.reason is not None and candidate.reason not in RECEIPT_REASONS:
            raise ValueError(f"reason {candidate.reason!r} is not an allowlisted code")
    return {
        "opportunity": choice.opportunity,
        "winner": choice.want.id if choice.want else None,
        "question_id": choice.question_id,
        "outcome": outcome,
        "outcome_reason": _OUTCOME_REASONS.get(outcome) if outcome else None,
        "candidates": [
            {
                "want_id": c.want_id,
                "kind": c.kind,
                "target_id": c.target_id,
                "action": c.action,
                "strength": c.strength,
                "weight": c.weight,
                "status": c.status,
                "reason": c.reason,
            }
            for c in choice.considered
        ],
    }
