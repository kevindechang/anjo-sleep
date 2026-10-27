# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The chooser's app adapters (Task 7a): the policy from Anjo's registries, and the intentions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from anjo.core import cognitive_cycle, workspace
from anjo.core import intentions as core_intentions
from anjo.sleep.app import intentions as ledger
from anjo.sleep.app.policy import build_policy
from anjo.sleep.mind.policy import PUBLIC_DEFAULT, SLICE_CODES
from anjo.sleep.ports import IntentionSource
from anjo.sleep.types import Commitment

NIGHT = datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)


def test_policy_from_registries_matches_anjo() -> None:
    policy = build_policy()
    assert policy.priority_classes == cognitive_cycle.PRIORITY_CLASSES
    assert policy.max_candidates == cognitive_cycle.MAX_CANDIDATES == 16
    assert policy.surface_min == workspace.SURFACE_MIN
    assert policy.reason_codes == cognitive_cycle.REASON_CODES | SLICE_CODES
    assert {"note_declined", "action_not_offered", "budget"} <= policy.reason_codes
    # the standalone example's public default must not drift from the registries it mirrors
    assert PUBLIC_DEFAULT == policy


def test_intentions_are_read_from_the_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"id": "i1", "text": "send the cut", "check_after": (NIGHT - timedelta(days=1)).isoformat()},
        {"id": "i2", "text": "call mum", "check_after": None},
        {"id": "i3", "text": "bad stamp", "check_after": "someday"},
    ]
    monkeypatch.setattr(core_intentions, "open_visible", lambda user_id: records)
    monkeypatch.setattr(ledger, "is_synthetic", lambda user_id: False)
    source = ledger.LedgerIntentions()
    assert isinstance(source, IntentionSource)
    assert source.intentions("kevin", NIGHT) == (
        Commitment("i1", "send the cut", NIGHT - timedelta(days=1)),
        Commitment("i2", "call mum", None),
        Commitment("i3", "bad stamp", None),
    )


def test_a_synthetic_person_has_no_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(user_id: str) -> list[dict[str, object]]:
        raise AssertionError("a synthetic person's ledger must not be read")

    monkeypatch.setattr(core_intentions, "open_visible", never)
    monkeypatch.setattr(ledger, "is_synthetic", lambda user_id: True)
    assert ledger.LedgerIntentions().intentions("synthetic-1", NIGHT) == ()
