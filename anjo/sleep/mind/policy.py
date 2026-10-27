"""The chooser's public default policy, and the reason codes it writes (Task 7a).

``PUBLIC_DEFAULT`` is the default for the standalone example (Task 15), where there is no Anjo
registry to read. It mirrors the registries ``anjo/sleep/app/policy.py`` builds the real policy
from (``cognitive_cycle.PRIORITY_CLASSES``, ``MAX_CANDIDATES``, ``REASON_CODES`` and
``workspace.SURFACE_MIN``), and a test in the app keeps the two equal. It is a stand-in, not the
source of truth: in the private app the registries win.

Phase 2 moves the slice's own codes into Anjo's registry; until then they live here.
"""
from __future__ import annotations

from typing import Final

from anjo.sleep.types import Policy

# The codes the chooser writes. A policy must allowlist every one of them.
UNKNOWN_CLASS: Final = "unknown_priority_class"
NOT_OFFERED: Final = "action_not_offered"
OVER_CAP: Final = "candidate_limit"
BELOW_THRESHOLD: Final = "below_threshold"
NO_BUDGET: Final = "budget"
LOST: Final = "lost_to_target"
NOTE_DECLINED: Final = "note_declined"

# The codes Anjo's registry doesn't have on main yet (spec Task 7a).
SLICE_CODES: Final = frozenset({NOTE_DECLINED, NOT_OFFERED, NO_BUDGET})

# A study or an inquiry needs at least this pull; the builder tunes it on fixtures. A held note
# needs ``surface_min`` (today's workspace.SURFACE_MIN).
AWAY_MIN: Final = 0.2

_HELD_BACK = frozenset({"quiet_hours", "budget", "paused_by_you", "below_threshold", "lost_to_target"})
_REJECTED = frozenset({
    "missing_identity", "wrong_event", "unknown_operation", "unknown_priority_class",
    "unknown_expected_change", "unknown_proposal_reason", "not_actionable", "invalid_score",
    "invalid_estimate", "inactive_person", "missing_target", "target_closed", "stale_target",
    "unknown_source", "unauthorized", "expired", "source_ref_limit", "dependency_limit",
    "candidate_limit", "duplicate_candidate",
})

PUBLIC_DEFAULT: Final = Policy(
    priority_classes=("user_commitment", "source_repair", "active_thread", "curiosity", "maintenance"),
    max_candidates=16,
    away_min=AWAY_MIN,
    surface_min=0.5,
    reason_codes=_HELD_BACK | _REJECTED | SLICE_CODES,
)
