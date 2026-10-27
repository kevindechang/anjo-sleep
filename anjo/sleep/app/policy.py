"""The chooser's ``Policy``, built from Anjo's registries (Task 7a).

The registries are read, never edited: ``cognitive_cycle.PRIORITY_CLASSES``, ``MAX_CANDIDATES``
and ``REASON_CODES``, and ``workspace.SURFACE_MIN``. The slice adds its own codes
(``note_declined``, ``action_not_offered``, ``budget``), which main's registry doesn't have yet;
autonomy Phase 2 moves them there.
"""
from __future__ import annotations

from anjo.core import cognitive_cycle, workspace
from anjo.sleep.mind.policy import AWAY_MIN, SLICE_CODES
from anjo.sleep.types import Policy


def build_policy() -> Policy:
    return Policy(
        priority_classes=tuple(cognitive_cycle.PRIORITY_CLASSES),
        max_candidates=cognitive_cycle.MAX_CANDIDATES,
        away_min=AWAY_MIN,
        surface_min=workspace.SURFACE_MIN,
        reason_codes=frozenset(cognitive_cycle.REASON_CODES) | SLICE_CODES,
    )
