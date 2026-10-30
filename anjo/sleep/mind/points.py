"""Her decision points (Task 16): the raw state at a night's two offers, recorded for replay.

On every night of a synthetic life the night records two points (``night.run``): the pre-study
offer (step 4) and the post-result offer (step 10). Each holds the raw input to ``snapshot.build``,
captured before any eligibility filter and before the note settles its target: every open target
with its revisions and confidence, every result, her workspace's questions with their priorities'
parts, the clock and the person's last message. With it go the context of the offer (policy,
allowance, offered actions) and the choice her chooser made there, so other deciders can be replayed
on exactly what she saw (``deciders``).

A point's record (``point_record``) holds ids, numbers, times and codes, never words: each target's
label and person id are left out, and a replay that needs a label looks it up by id.

Engine core, and pure: no clock, database, model call or environment read.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any, Final, Literal, get_args

from anjo.sleep.evidence.targets import (
    result_from_record,
    result_record,
    target_from_record,
    target_record,
)
from anjo.sleep.mind import snapshot
from anjo.sleep.mind.inquiry.types import Priority
from anjo.sleep.types import (
    Action,
    Choice,
    ChoiceContext,
    Commitment,
    Considered,
    NightAllowance,
    Policy,
    QuestionView,
    Result,
    Target,
    Want,
    WantSnapshot,
)

Phase = Literal["pre_study", "post_result"]
PHASES: Final = frozenset(get_args(Phase))
STUDY_ACTIONS: Final[frozenset[Action]] = frozenset({"prepare_commitment", "revisit_thread"})


@dataclass(frozen=True)
class QuestionState:
    """One question of her workspace as the chooser saw it, with its priority's parts (Task 17), so
    a replay can change one factor and recompute the value."""

    view: QuestionView
    parts: Priority

    def __post_init__(self) -> None:
        if self.parts.question_id != self.view.id:
            raise ValueError("a question's priority parts are its own")
        if not math.isclose(self.view.priority, self.parts.value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"question {self.view.id}'s priority is not the value of its parts")


@dataclass(frozen=True)
class RawSnapshot:
    """``snapshot.build``'s input, unfiltered."""

    targets: tuple[Target, ...]
    results: tuple[Result, ...]
    as_of: datetime
    last_message_at: datetime | None
    intentions: tuple[Commitment, ...] = ()
    questions: tuple[QuestionState, ...] = ()


@dataclass(frozen=True)
class DecisionPoint:
    """One offer of one night: what she saw, what the offer allowed, and her choice there. At the
    post-result point ``choice`` is ``None`` when no second offer was made (no result to say, or a
    note already chosen at step 4)."""

    night: date
    phase: Phase
    raw: RawSnapshot
    ctx: ChoiceContext
    choice: Choice | None


def build(raw: RawSnapshot, *, inquiry_offered: bool = True) -> WantSnapshot:
    """``snapshot.build`` over the raw input: the state her chooser derives wants from."""
    return replace(snapshot.build(
        list(raw.targets), list(raw.results), raw.as_of, raw.last_message_at,
        intentions=raw.intentions, questions=tuple(q.view for q in raw.questions),
    ), inquiry_offered=inquiry_offered)


def study_target(choice: Choice | None, raw: RawSnapshot) -> str | None:
    """The target a choice studies: a study's own target, or an inquiry's first target
    (``target_ids[0]``), so a question spanning targets gets no extra chances. A note or nothing
    studies nothing."""
    if choice is None or choice.want is None:
        return None
    if choice.action in STUDY_ACTIONS:
        return choice.target_id
    if choice.action == "inquire":
        question = next((q.view for q in raw.questions if q.view.id == choice.question_id), None)
        if question is None:
            raise ValueError(f"inquiry {choice.question_id} isn't in the snapshot")
        return question.target_ids[0]
    return None


# ── Records: ids, numbers, times and codes ────────────────────────────────────


def point_record(point: DecisionPoint) -> dict[str, Any]:
    raw = point.raw
    return {
        "night": point.night.isoformat(), "phase": point.phase,
        "as_of": raw.as_of.isoformat(), "last_message_at": _iso(raw.last_message_at),
        "targets": [target_record(replace(t, label="", person_id="")) for t in raw.targets],
        "results": [result_record(r) for r in raw.results],
        "intentions": [{"id": c.id, "check_after": _iso(c.check_after)} for c in raw.intentions],
        "questions": [_question_record(q) for q in raw.questions],
        "ctx": _ctx_record(point.ctx),
        "choice": None if point.choice is None else _choice_record(point.choice),
    }


def point_from_record(record: Mapping[str, Any]) -> DecisionPoint:
    phase = record["phase"]
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    raw = RawSnapshot(
        targets=tuple(target_from_record(t) for t in record["targets"]),
        results=tuple(result_from_record(r) for r in record["results"]),
        as_of=datetime.fromisoformat(record["as_of"]), last_message_at=_when(record["last_message_at"]),
        intentions=tuple(Commitment(c["id"], "", _when(c["check_after"])) for c in record["intentions"]),
        questions=tuple(_question_from(q) for q in record["questions"]),
    )
    choice = record["choice"]
    return DecisionPoint(
        night=date.fromisoformat(record["night"]), phase=phase, raw=raw, ctx=_ctx_from(record["ctx"]),
        choice=None if choice is None else _choice_from(choice),
    )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _when(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _question_record(state: QuestionState) -> dict[str, Any]:
    view, parts = state.view, state.parts
    return {"id": view.id, "target_ids": list(view.target_ids), "status": view.status, "priority": view.priority,
            "continues": view.continues, "stake": parts.stake, "u": parts.u, "a": parts.a, "ready": parts.ready}


def _question_from(record: Mapping[str, Any]) -> QuestionState:
    view = QuestionView(id=record["id"], target_ids=tuple(record["target_ids"]), status=record["status"],
                        priority=float(record["priority"]), continues=bool(record["continues"]))
    parts = Priority(record["id"], float(record["stake"]), float(record["u"]), float(record["a"]),
                     bool(record["ready"]))
    return QuestionState(view=view, parts=parts)


def _ctx_record(ctx: ChoiceContext) -> dict[str, Any]:
    policy, allowance = ctx.policy, ctx.allowance
    return {
        "policy": {"priority_classes": list(policy.priority_classes), "max_candidates": policy.max_candidates,
                   "away_min": policy.away_min, "surface_min": policy.surface_min,
                   "reason_codes": sorted(policy.reason_codes)},
        "allowance": {"actions_left": allowance.actions_left, "calls_left": allowance.calls_left,
                      "usd_left": allowance.usd_left},
        "allowed_actions": None if ctx.allowed_actions is None else sorted(ctx.allowed_actions),
    }


def _ctx_from(record: Mapping[str, Any]) -> ChoiceContext:
    policy, allowance, allowed = record["policy"], record["allowance"], record["allowed_actions"]
    return ChoiceContext(
        policy=Policy(tuple(policy["priority_classes"]), int(policy["max_candidates"]), float(policy["away_min"]),
                      float(policy["surface_min"]), frozenset(policy["reason_codes"])),
        allowance=NightAllowance(int(allowance["actions_left"]), int(allowance["calls_left"]),
                                 float(allowance["usd_left"])),
        allowed_actions=None if allowed is None else frozenset(allowed),
    )


def _want_record(want: Want) -> dict[str, Any]:
    return {"id": want.id, "kind": want.kind, "target_id": want.target_id, "action": want.action,
            "priority_class": want.priority_class, "strength": want.strength, "weight": want.weight,
            "question_id": want.question_id}


def _choice_record(choice: Choice) -> dict[str, Any]:
    return {
        "opportunity": choice.opportunity, "want": None if choice.want is None else _want_record(choice.want),
        "considered": [
            {"want_id": c.want_id, "kind": c.kind, "target_id": c.target_id, "action": c.action,
             "strength": c.strength, "weight": c.weight, "status": c.status, "reason": c.reason}
            for c in choice.considered
        ],
    }


def _choice_from(record: Mapping[str, Any]) -> Choice:
    want = record["want"]
    return Choice(
        opportunity=record["opportunity"],
        want=None if want is None else Want(**want),
        considered=tuple(Considered(**c) for c in record["considered"]),
    )
