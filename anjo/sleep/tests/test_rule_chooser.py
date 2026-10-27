"""RuleChooser, the §6a fallback: the open thread with the largest unacted revision, never a note."""
from __future__ import annotations

import dataclasses
import itertools
from datetime import date, datetime, timezone

import pytest

from anjo.sleep.mind.rule_chooser import REASON_CODES, RuleChooser
from anjo.sleep.ports import Chooser
from anjo.sleep.types import (
    Choice,
    ChoiceContext,
    NightAllowance,
    Policy,
    Target,
    TargetKind,
    TargetStatus,
    Want,
    WantSnapshot,
)

NIGHT = datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)
POLICY = Policy(
    priority_classes=("user_commitment", "source_repair", "active_thread", "curiosity"),
    max_candidates=16,
    away_min=0.2,
    surface_min=0.5,
    reason_codes=REASON_CODES | {"below_threshold"},
)
CTX = ChoiceContext(policy=POLICY, allowance=NightAllowance(actions_left=2, calls_left=8, usd_left=3.0))


def _target(
    target_id: str,
    revision: int,
    acted: int,
    *,
    status: TargetStatus = "open",
    kind: TargetKind = "thread",
) -> Target:
    return Target(
        id=target_id, person_id="synthetic-1", kind=kind, label=f"topic {target_id}",
        check_after=None, status=status, revision=revision, acted_revision=acted,
        confidence=None, last_mentioned=date(2026, 10, 27),
    )


def _night(*targets: Target, ctx: ChoiceContext = CTX) -> Choice:
    chooser = RuleChooser()
    snapshot = WantSnapshot(as_of=NIGHT, last_message_at=None, targets=targets, results=())
    return chooser.choose("away", chooser.derive(snapshot), ctx)


def _note_want(target_id: str = "t1") -> Want:
    return Want(
        id=f"share:{target_id}", kind="share", target_id=target_id, action="hold_note",
        priority_class="user_commitment", strength=1.0,
    )


def test_picks_the_open_thread_with_the_largest_unacted_revision() -> None:
    choice = _night(_target("t1", 5, 4), _target("t2", 3, 0), _target("t3", 9, 7))
    assert (choice.action, choice.target_id) == ("revisit_thread", "t2")
    losers = [c for c in choice.considered if c.status != "won"]
    assert {c.target_id for c in losers} == {"t1", "t3"}
    assert {c.reason for c in losers} == {"lost_to_target"}


def test_skips_closed_and_paused_targets() -> None:
    choice = _night(
        _target("closed", 10, 0, status="closed"),
        _target("paused", 8, 0, status="paused"),
        _target("open", 2, 1),
    )
    assert choice.target_id == "open"
    assert {c.target_id for c in choice.considered} == {"open"}


def test_skips_commitments() -> None:
    choice = _night(_target("c1", 9, 0, kind="commitment"), _target("t1", 2, 1))
    assert choice.target_id == "t1"


def test_unchanged_thread_wakes_nothing() -> None:
    choice = _night(_target("t1", 4, 4))
    assert choice.want is None and choice.action is None
    assert choice.considered == ()


def test_never_returns_hold_note() -> None:
    chooser = RuleChooser()
    thread = chooser.derive(
        WantSnapshot(as_of=NIGHT, last_message_at=None, targets=(_target("t2", 2, 1),), results=()),
    )
    choice = chooser.choose("away", (_note_want(), *thread), CTX)
    assert choice.action == "revisit_thread"
    note = next(c for c in choice.considered if c.action == "hold_note")
    assert (note.status, note.reason) == ("held_back", "action_not_offered")

    only_note = chooser.choose("away", (_note_want(),), CTX)
    assert only_note.want is None
    assert [c.reason for c in only_note.considered] == ["action_not_offered"]


def test_second_offer_for_a_note_chooses_nothing() -> None:
    second = dataclasses.replace(CTX, allowed_actions=frozenset({"hold_note"}))
    choice = _night(_target("t1", 3, 0), ctx=second)
    assert choice.want is None
    assert [c.reason for c in choice.considered] == ["action_not_offered"]


def test_no_model_calls_left_holds_the_study_back() -> None:
    spent = dataclasses.replace(CTX, allowance=NightAllowance(actions_left=2, calls_left=0, usd_left=3.0))
    choice = _night(_target("t1", 3, 0), ctx=spent)
    assert choice.want is None
    assert [(c.status, c.reason) for c in choice.considered] == [("held_back", "budget")]


def test_order_is_independent_of_arrival() -> None:
    chooser = RuleChooser()
    targets = (_target("a", 3, 1), _target("b", 3, 1), _target("c", 4, 3), _target("d", 1, 0))
    wants = chooser.derive(WantSnapshot(as_of=NIGHT, last_message_at=None, targets=targets, results=()))
    choices = {chooser.choose("away", order, CTX) for order in itertools.permutations(wants)}
    assert len(choices) == 1
    assert choices.pop().target_id == "a"  # a and b tie on 2 unacted; the id breaks it


def test_candidates_past_the_cap_are_rejected() -> None:
    capped = dataclasses.replace(CTX, policy=dataclasses.replace(POLICY, max_candidates=2))
    choice = _night(_target("t1", 3, 0), _target("t2", 2, 0), _target("t3", 1, 0), ctx=capped)
    assert choice.target_id == "t1"
    assert [(c.target_id, c.status, c.reason) for c in choice.considered] == [
        ("t1", "won", None),
        ("t2", "held_back", "lost_to_target"),
        ("t3", "rejected", "candidate_limit"),
    ]


def test_unknown_priority_class_is_rejected() -> None:
    odd = Want(
        id="understand:x", kind="understand", target_id="x", action="revisit_thread",
        priority_class="not_a_class", strength=1.0,
    )
    choice = RuleChooser().choose("away", (odd,), CTX)
    assert choice.want is None
    assert [(c.status, c.reason) for c in choice.considered] == [("rejected", "unknown_priority_class")]


def test_policy_without_the_reason_codes_is_refused() -> None:
    bare = dataclasses.replace(CTX, policy=dataclasses.replace(POLICY, reason_codes=frozenset()))
    with pytest.raises(ValueError, match="reason codes"):
        _night(_target("t1", 3, 0), ctx=bare)


def test_derived_strength_grows_with_unacted_revisions_inside_the_unit_scale() -> None:
    chooser = RuleChooser()
    targets = tuple(_target(f"t{n}", n, 0) for n in (1, 2, 5, 50))
    wants = chooser.derive(WantSnapshot(as_of=NIGHT, last_message_at=None, targets=targets, results=()))
    strengths = [w.strength for w in wants]
    assert strengths == sorted(strengths)
    assert all(0.0 < s < 1.0 for s in strengths)
    assert {w.kind for w in wants} == {"understand"}


def test_rule_chooser_satisfies_the_chooser_port() -> None:
    chooser: Chooser = RuleChooser()
    assert isinstance(chooser, Chooser)
