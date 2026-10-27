# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The shared types (spec §12.1) are frozen, hashable values that every task builds against."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from anjo.sleep import types as t

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

NOW = datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)
DAY = date(2026, 10, 27)

SHARED = (
    "Evidence", "Probe", "CheckpointRef", "GateResult", "NightResult", "Target", "Result",
    "Policy", "WantSnapshot", "Want", "Considered", "Choice", "ChoiceContext", "Expectation",
    "NightAllowance", "Commitment", "QuestionView",
)


def _policy() -> t.Policy:
    return t.Policy(
        priority_classes=("user_commitment", "active_thread"),
        max_candidates=16,
        away_min=0.2,
        surface_min=0.5,
        reason_codes=frozenset({"budget", "lost_to_target"}),
    )


def _target(**changes: Any) -> t.Target:
    base = t.Target(
        id="t1", person_id="synthetic-1", kind="thread", label="lens choice for scene 4",
        check_after=None, status="open", revision=3, acted_revision=1, confidence=0.4,
        last_mentioned=DAY,
    )
    return dataclasses.replace(base, **changes)


def _want(**changes: Any) -> t.Want:
    base = t.Want(
        id="understand:t1", kind="understand", target_id="t1", action="revisit_thread",
        priority_class="active_thread", strength=0.5,
    )
    return dataclasses.replace(base, **changes)


def _gate() -> t.GateResult:
    return t.GateResult(gate="general", passed=True, value=0.4, threshold=2.0)


def _checkpoint() -> t.CheckpointRef:
    return t.CheckpointRef(
        id="p-0a1b-n1", person_id="synthetic-1", night=DAY, uri="adapters/p-0a1b-n1",
        parent_id=None,
    )


def _choice() -> t.Choice:
    want = _want()
    won = t.Considered(
        want_id=want.id, kind=want.kind, target_id=want.target_id, action=want.action,
        strength=want.strength, weight=want.weight, status="won", reason=None,
    )
    return t.Choice(opportunity="away", want=want, considered=(won,))


def _night(**changes: Any) -> t.NightResult:
    base = t.NightResult(
        status="promoted", failed_at=None, checkpoint=_checkpoint(), gates={"general": _gate()},
        choices=(_choice(),), cost_usd=1.5, cost_parts={"training": 1.0, "extract": 0.5},
        log_path=Path("night.jsonl"),
    )
    return dataclasses.replace(base, **changes)


def _samples() -> dict[str, DataclassInstance]:
    allowance = t.NightAllowance(actions_left=2, calls_left=10, usd_left=3.0)
    question = t.QuestionView(
        id="q1", target_ids=("t1",), status="open", priority=1.2, continues=True,
    )
    commitment = t.Commitment(id="c1", label="shoot scene 4", check_after=NOW)
    result = t.Result(
        id="r1", target_id="t1", homework_id="h1", verified_sources=2, created_at=NOW,
        consumed=False,
    )
    return {
        "Evidence": t.Evidence(
            id="e1", person_id="synthetic-1", day=DAY, kind="preference",
            text="prefers natural light", source="turn:1", held_out=False, created_at=NOW,
            model="muse-spark-open", ancestry=frozenset({"muse-spark-open"}),
        ),
        "Probe": t.Probe(
            id="p1", person_id="synthetic-1", kind="taste_ab", prompt="Which shot?",
            options=("85 mm", "35 mm"), expected="85 mm", source_ids=("e1",),
        ),
        "CheckpointRef": _checkpoint(),
        "GateResult": _gate(),
        "NightResult": _night(),
        "Target": _target(),
        "Result": result,
        "Policy": _policy(),
        "WantSnapshot": t.WantSnapshot(
            as_of=NOW, last_message_at=None, targets=(_target(),), results=(result,),
            intentions=(commitment,), questions=(question,),
        ),
        "Want": _want(),
        "Considered": _choice().considered[0],
        "Choice": _choice(),
        "ChoiceContext": t.ChoiceContext(
            policy=_policy(), allowance=allowance, allowed_actions=frozenset({"hold_note"}),
        ),
        "Expectation": t.Expectation(
            id="x1", target_id="t1", made_at=NOW, horizon=NOW, p_needed=0.3, outcome=None,
        ),
        "NightAllowance": allowance,
        "Commitment": commitment,
        "QuestionView": question,
    }


def test_every_shared_type_is_defined_and_sampled() -> None:
    assert sorted(_samples()) == sorted(SHARED)
    for name in SHARED:
        assert dataclasses.is_dataclass(getattr(t, name)), name


@pytest.mark.parametrize("name", SHARED)
def test_shared_type_is_frozen_and_hashable(name: str) -> None:
    value = _samples()[name]
    assert type(value).__name__ == name
    first = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, first, None)
    copy = dataclasses.replace(value)
    assert copy == value
    assert hash(copy) == hash(value)


@pytest.mark.parametrize("name", SHARED)
def test_collections_inside_shared_types_are_immutable(name: str) -> None:
    value = _samples()[name]
    for f in dataclasses.fields(value):
        assert not isinstance(getattr(value, f.name), (list, set)), f"{name}.{f.name}"


def test_night_result_copies_its_mappings() -> None:
    gates = {"general": _gate()}
    night = _night(gates=gates)
    gates["safety"] = t.GateResult(gate="safety", passed=False, value=1.0, threshold=0.0)
    assert set(night.gates) == {"general"}
    assert dataclasses.asdict(night)["gates"]["general"]["passed"] is True


def test_night_cost_is_the_sum_of_its_parts() -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        _night(cost_usd=2.0)


def test_want_strength_stays_on_the_unit_scale() -> None:
    with pytest.raises(ValueError, match="strength"):
        _want(strength=1.5)
    assert _want(strength=0.5, weight=0.5).effective_strength == 0.25


def test_inquire_names_its_question_and_nothing_else_does() -> None:
    with pytest.raises(ValueError, match="question_id"):
        _want(action="inquire")
    with pytest.raises(ValueError, match="question_id"):
        _want(question_id="q1")
    assert _want(action="inquire", question_id="q1").question_id == "q1"


def test_target_revisions_and_confidence_are_checked() -> None:
    with pytest.raises(ValueError, match="acted_revision"):
        _target(revision=1, acted_revision=2)
    with pytest.raises(ValueError, match="confidence"):
        _target(confidence=1.2)


def test_considered_carries_a_reason_unless_it_won() -> None:
    with pytest.raises(ValueError, match="reason"):
        dataclasses.replace(_choice().considered[0], reason="budget")
    with pytest.raises(ValueError, match="reason"):
        dataclasses.replace(_choice().considered[0], status="held_back")


def test_allowance_caps_actions_and_gates_model_calls() -> None:
    with pytest.raises(ValueError, match="actions_left"):
        t.NightAllowance(actions_left=3, calls_left=1, usd_left=1.0)
    assert t.NightAllowance(actions_left=2, calls_left=5, usd_left=1.0).available_model_calls == 2
    assert t.NightAllowance(actions_left=2, calls_left=0, usd_left=1.0).available_model_calls == 0
    assert t.NightAllowance(actions_left=2, calls_left=5, usd_left=0.0).available_model_calls == 0


def test_choice_context_allows_every_action_unless_restricted() -> None:
    open_ctx = t.ChoiceContext(
        policy=_policy(), allowance=t.NightAllowance(actions_left=1, calls_left=1, usd_left=1.0),
    )
    assert open_ctx.allows("revisit_thread")
    second_offer = dataclasses.replace(open_ctx, allowed_actions=frozenset({"hold_note"}))
    assert second_offer.allows("hold_note")
    assert not second_offer.allows("revisit_thread")


def test_choice_exposes_its_action_or_nothing() -> None:
    chosen = _choice()
    assert (chosen.action, chosen.target_id, chosen.question_id) == ("revisit_thread", "t1", None)
    nothing = t.Choice(opportunity="away", want=None, considered=())
    assert (nothing.action, nothing.target_id, nothing.question_id) == (None, None, None)


def test_expectation_probability_is_checked() -> None:
    with pytest.raises(ValueError, match="p_needed"):
        t.Expectation(id="x", target_id="t", made_at=NOW, horizon=NOW, p_needed=1.1, outcome=None)
