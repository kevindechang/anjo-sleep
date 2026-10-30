# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's inquiry contract (Task 17): the workspace, the nine operations and the ports are frozen,
match the spec's names and numbers, round-trip through the thinker's JSON, and stay pure."""
from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, get_args

import pytest

from anjo.sleep import types as t
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.mind import snapshot
from anjo.sleep.mind.inquiry import ports
from anjo.sleep.mind.inquiry import types as it

NOW = datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)
DAY = date(2026, 10, 27)
INQUIRY = Path(it.__file__).parent
APD = INQUIRY.parent / "apd"


# ── Samples ───────────────────────────────────────────────────────────────────


def _question(**changes: Any) -> it.Question:
    base = it.Question(
        id="q1", target_ids=("t-scene4",), kind="contradiction", origin_evidence=("e-d2-1", "e-d5-3"),
        status="active", opened_at=NOW, subject="Scene 4's long-lens look against the small kitchen",
    )
    return dataclasses.replace(base, **changes)


def _set(question_id: str = "q1", credences: tuple[float, ...] = (0.5, 0.3, 0.2)) -> tuple[it.Hypothesis, ...]:
    *alternatives, catch_all = credences
    return (
        *(
            it.Hypothesis(id=f"{question_id}.h{n}", question_id=question_id, statement=f"alternative {n}", credence=c)
            for n, c in enumerate(alternatives, start=1)
        ),
        it.Hypothesis(id=it.catch_all_id(question_id), question_id=question_id, statement=it.CATCH_ALL, credence=catch_all),
    )


def _prediction(**changes: Any) -> it.Prediction:
    base = it.Prediction(
        id="p1", question_id="q1", claim="She names a place other than the kitchen",
        p_true_given={"q1.h1": 0.2, "q1.h2": 0.9, "q1.else": 0.5}, committed_at=NOW,
        horizon=NOW + timedelta(days=3), outcome=None, resolved_by=(),
    )
    return dataclasses.replace(base, **changes)


def _update() -> it.Update:
    return it.Update(
        question_id="q1", ref="e-d3-2", source="assessed_evidence", evidence_ids=("e-d3-2",),
        likelihoods={"q1.h1": 0.5, "q1.h2": 4.0, "q1.else": 1.0},
    )


def _finding(**changes: Any) -> it.Finding:
    base = it.Finding(
        id="f1", question_id="q1", statement="The 85 mm look doesn't fit the kitchen", p=0.5,
        cites=("e-d2-1", "e-d5-3"), kind="suspicion", hypothesis_id="q1.h1",
    )
    return dataclasses.replace(base, **changes)


def _ref(evidence_id: str = "e-d2-1") -> it.EvidenceRef:
    return it.EvidenceRef(id=evidence_id, day=DAY, created_at=NOW)


def _notice() -> it.NoticeOp:
    return it.NoticeOp(
        kind="contradiction", subject="Scene 4's lens against the kitchen", target_ids=("t-scene4",),
        origin_evidence=("e-d2-1", "e-d5-3"),
    )


def _start(**changes: Any) -> it.RunStart:
    base = it.RunStart(
        night=DAY, as_of=NOW, call="inquiry", rules=it.FULL_RULES, control="priority", focus="q1", budget=it.Budget(),
    )
    return dataclasses.replace(base, **changes)


def _history() -> tuple[it.Step, ...]:
    """A notice pass, then an inquiry run with a malformed reply and a rejected PREDICT."""
    return (
        it.Step(seq=0, at=NOW, entry=_start(call="notice", focus=None)),
        it.Step(seq=1, at=NOW, entry=it.Retrieved(None, "new_evidence", "", (_ref(), _ref("e-d5-3"))), tokens=300),
        it.Step(seq=2, at=NOW, entry=_notice(), tokens=40),
        it.Step(seq=3, at=NOW, entry=_start(budget=it.Budget(tokens_used=340))),
        it.Step(seq=4, at=NOW, entry=None, code="malformed", cycle=0, focus="q1", tokens=120),
        it.Step(
            seq=5, at=NOW, entry=it.PredictOp("q1", "x", {"q1.h1": 0.5, "q1.h2": 0.55, "q1.else": 0.5}, 2),
            code="k2_not_discriminating", cycle=1, focus="q1", tokens=150,
        ),
    )


def _workspace(**changes: Any) -> it.Workspace:
    base = it.Workspace(
        questions=(_question(),), hypotheses=_set(), predictions=(_prediction(),), updates=(_update(),),
        findings=(_finding(),), history=_history(),
    )
    return dataclasses.replace(base, **changes)


def _priority(**changes: Any) -> it.Priority:
    base = it.Priority(question_id="q1", stake=it.STAKE_NONE, u=0.5, a=it.A_WAITING, ready=False)
    return dataclasses.replace(base, **changes)


def _samples() -> dict[str, object]:
    reply = it.ModelReply(text='{"op": "PARK", "question_id": "q1"}', tokens=40, model="glimmer-fake")
    budget = it.Budget()
    decision = it.Decision(kind="continue", question_id="q1", reason=None, priorities=(_priority(),), budget=budget)
    event = it.WorkspaceEvent(step=_history()[2], before="0" * 64, after="1" * 64)
    turn = it.Turn(seq=2, prompt="rendered", reply=reply)
    ws = _workspace()
    return {
        "Question": _question(), "Hypothesis": _set()[0], "Prediction": _prediction(), "Update": _update(),
        "Finding": _finding(), "EvidenceRef": _ref(), "Retrieved": _history()[1].entry, "RunStart": _start(),
        "Edit": it.Edit("swap_lead", "q1", hypothesis_id="q1.h2"),
        "Answered": it.Answered("q1", "How long is an 85 mm lens's minimum distance?", "About 0.8 m"),
        "Resolved": it.Resolved("p1", "false", (_ref("e-d6-1"),)), "Step": _history()[2],
        "WorkspaceEvent": event, "Workspace": ws, "Applied": it.Applied(ws=ws, event=event),
        "Rejected": it.Rejected("k1_bucket"), "KernelRules": it.FULL_RULES, "Priority": _priority(),
        "Budget": budget, "Decision": decision, "ModelReply": reply,
        "ResolverReply": it.ResolverReply("true", 30),
        "Turn": turn,
        "InquiryRun": it.InquiryRun(ws=ws, events=(event,), decisions=(decision,), turns=(turn,), budget=budget),
        "Alternative": it.Alternative("She will shoot from the courtyard", 0.3),
        **{type(op).__name__: op for op in _ops()},
    }


def _ops() -> tuple[it.Op, ...]:
    return tuple(it.op_from_json(_example(cls)) for cls in it.OP_TYPES.values())


def _example(cls: type) -> dict[str, Any]:
    """The one JSON example in an operation's docstring."""
    lines = [line.strip() for line in (cls.__doc__ or "").splitlines() if line.strip().startswith('{"op"')]
    assert len(lines) == 1, f"{cls.__name__} documents exactly one JSON example"
    value = json.loads(lines[0])
    assert isinstance(value, dict)
    return value


# ── The spec's names and numbers ──────────────────────────────────────────────


def test_literals_match_the_spec() -> None:
    assert set(get_args(it.QuestionKind)) == {"contradiction", "gap", "opportunity", "anomaly"}
    assert set(get_args(it.PredictionOutcome)) == {"true", "false", "undetermined", "censored"}
    assert set(get_args(it.Verdict)) == {"true", "false", "undetermined"}
    assert set(get_args(it.UpdateSource)) == {"resolved_prediction", "assessed_evidence"}
    assert set(get_args(it.FindingKind)) == {"suspicion", "resolution", "retraction"}
    assert set(get_args(it.DecisionKind)) == {"continue", "investigate", "park", "stop"}
    assert set(get_args(it.Tool)) == {"evidence_search", "world_query", "research"}
    assert set(get_args(it.OpName)) == {
        "NOTICE", "HYPOTHESIZE", "PREDICT", "INVESTIGATE", "ASSESS", "CONCLUDE", "RETRACT", "PARK", "ABANDON",
    }
    assert set(it.OP_TYPES) == set(get_args(it.OpName))
    assert set(get_args(it.EditKind)) == {"swap_lead", "delete_prediction", "inject_hypothesis"}
    assert set(get_args(it.RunCall)) == {"notice", "inquiry"}


def test_question_status_is_the_choosers_status() -> None:
    assert get_args(it.QuestionStatus) == get_args(t.QuestionStatus)
    assert set(get_args(it.QuestionStatus)) == {"open", "active", "parked", "concluded", "retracted", "abandoned"}
    assert it.LIVE_STATUSES == snapshot.LIVE_QUESTION
    assert set(get_args(it.QuestionStatus)) == snapshot.LIVE_QUESTION | snapshot.CLOSED_QUESTION


def test_kernel_and_controller_numbers_are_the_specs() -> None:
    assert it.ASSESS_BUCKETS == (4.0, 2.0, 1.0, 0.5, 0.25)
    assert it.PROBABILITY_CLAMP == (0.05, 0.95)
    assert (it.MIN_SPREAD, it.RESOLVE_MIN, it.RETRACT_BELOW, it.SPLIT_MIN) == (0.3, 0.8, 0.5, 0.5)
    assert (it.STALL_LIMIT, it.MIN_ALTERNATIVES, it.MAX_ALTERNATIVES) == (3, 2, 4)
    assert (it.CYCLE_CAP, it.TOKEN_CAP, it.RECENT_OPS) == (12, 20_000, 6)
    assert (it.PRIORITY_FLOOR, it.U_FLOOR, it.A_READY, it.A_WAITING) == (0.2, 0.2, 1.0, 0.3)
    assert (it.STAKE_DUE, it.STAKE_NONE, it.STAKE_WINDOW) == (2.0, 1.0, timedelta(days=7))


def test_rejection_codes_cover_every_rule_that_rejects() -> None:
    codes = it.REJECT_CODES
    for prefix in ("k1_", "k2_", "k3_", "k4_", "domain_"):
        assert any(code.startswith(prefix) for code in codes), prefix
    assert {"hypothesize_once", "hypothesis_count", "malformed", "no_targets"} <= codes
    # The finding cap is APD's, per life (apd.types.FindingRegistry), so no wipe resets it and the
    # live product has none.
    assert "finding_cap" not in codes and not hasattr(it, "FINDING_CAP")
    with pytest.raises(ValueError):
        it.Rejected("stalled")  # type: ignore[arg-type]


# ── Records ───────────────────────────────────────────────────────────────────


def test_question_needs_targets_origin_and_subject() -> None:
    with pytest.raises(ValueError, match="target_ids"):
        _question(target_ids=())
    with pytest.raises(TypeError, match="target_ids"):
        _question(target_ids="t-scene4")
    with pytest.raises(ValueError, match="origin_evidence"):
        _question(origin_evidence=())
    with pytest.raises(ValueError, match="subject"):
        _question(subject=" ")
    with pytest.raises(ValueError, match="status"):
        _question(status="closed")
    assert _question(target_ids=["t-a", "t-b"]).target_ids == ("t-a", "t-b")  # a question may span targets


def test_catch_all_is_named_by_its_question() -> None:
    *alternatives, catch_all = _set()
    assert catch_all.is_catch_all and catch_all.id == it.catch_all_id("q1")
    assert not any(h.is_catch_all for h in alternatives)


def test_prediction_and_update_keep_read_only_copies_of_their_maps() -> None:
    given = {"q1.h1": 0.2, "q1.h2": 0.9, "q1.else": 0.5}
    prediction = _prediction(p_true_given=given)
    given["q1.h1"] = 0.99
    assert prediction.p_true_given["q1.h1"] == 0.2
    with pytest.raises(TypeError):
        prediction.p_true_given["q1.h1"] = 0.5  # type: ignore[index]
    assert prediction.pending
    with pytest.raises(ValueError, match="horizon"):
        _prediction(horizon=NOW)
    with pytest.raises(ValueError, match="positive"):
        dataclasses.replace(_update(), likelihoods={"q1.h1": 0.0})
    with pytest.raises(ValueError, match="evidence_ids"):
        dataclasses.replace(_update(), evidence_ids=())


def test_undetermined_stays_pending_until_the_horizon_censors_it() -> None:
    assert _prediction(outcome="undetermined").pending
    assert not any(_prediction(outcome=outcome).pending for outcome in ("true", "false", "censored"))


def test_finding_names_the_hypothesis_it_reports() -> None:
    with pytest.raises(ValueError, match="names the hypothesis"):
        _finding(hypothesis_id=None)
    assert _finding(kind="retraction", hypothesis_id=None).hypothesis_id is None
    with pytest.raises(ValueError, match="p"):
        _finding(p=1.2)


def test_workspace_holds_two_to_four_alternatives_and_one_catch_all() -> None:
    assert it.Workspace().seq == 0
    ws = _workspace()
    assert ws.seq == 6 and ws.question("q1") == _question() and ws.finding_of("q1") == _finding()
    assert len(ws.hypotheses_of("q1")) == 3
    for credences in ((0.8, 0.2), (0.2, 0.2, 0.2, 0.2, 0.1, 0.1)):  # one alternative; five
        with pytest.raises(ValueError, match="alternatives"):
            _workspace(hypotheses=_set(credences=credences), predictions=(), updates=(), findings=())
    with pytest.raises(ValueError, match="sum"):
        _workspace(hypotheses=_set(credences=(0.5, 0.3, 0.3)))
    no_catch_all = tuple(dataclasses.replace(h, id="q1.h9") if h.is_catch_all else h for h in _set())
    with pytest.raises(ValueError, match="catch-all"):
        _workspace(hypotheses=no_catch_all, predictions=(), updates=(), findings=())


def test_workspace_records_name_what_it_holds() -> None:
    with pytest.raises(ValueError, match="isn't held"):
        _workspace(findings=(_finding(question_id="q9"),))
    with pytest.raises(ValueError, match="appears twice"):
        _workspace(questions=(_question(), _question()))
    with pytest.raises(ValueError, match="one finding"):
        _workspace(findings=(_finding(), _finding(id="f2")))
    with pytest.raises(ValueError, match="exactly its question's"):
        _workspace(predictions=(_prediction(p_true_given={"q2.h1": 0.5}),))
    with pytest.raises(ValueError, match="outside its question"):
        _workspace(findings=(_finding(hypothesis_id="q1.h7"),))


def test_a_split_catch_all_leaves_every_prediction_naming_the_whole_set() -> None:
    """HYPOTHESIZE's split: each new hypothesis inherits the catch-all's committed p_true_given."""
    split = (*_set(credences=(0.5, 0.3, 0.1))[:-1], it.Hypothesis("q1.h3", "q1", "a third reading", 0.1),
             it.Hypothesis("q1.else", "q1", it.CATCH_ALL, 0.1))
    with pytest.raises(ValueError, match="exactly its question's"):
        _workspace(hypotheses=split)
    inherited = _prediction(p_true_given={**_prediction().p_true_given, "q1.h3": _prediction().p_true_given["q1.else"]})
    assert _workspace(hypotheses=split, predictions=(inherited,)).hypotheses_of("q1") == split


def test_history_starts_with_a_run_start_and_a_wipe_restarts_it() -> None:
    with pytest.raises(ValueError, match="starts with a RunStart"):
        _workspace(history=_history()[1:])
    with pytest.raises(ValueError, match="without a gap"):
        _workspace(history=(*_history()[:2], *_history()[3:]))
    later = tuple(dataclasses.replace(step, seq=step.seq + 7) for step in _history())
    with pytest.raises(ValueError, match="wiped it"):
        _workspace(history=later)
    wiped = (dataclasses.replace(later[0], entry=_start(call="notice", focus=None, wipe=True)), *later[1:])
    assert _workspace(history=wiped).seq == 13  # ids derive from log seqs, so a wipe can't reuse one
    with pytest.raises(ValueError, match="only its first step"):
        _workspace(history=(*_history()[:3], dataclasses.replace(_history()[3], entry=_start(wipe=True)), *_history()[4:]))


def test_steps_log_applied_rejected_and_malformed_operations() -> None:
    start, observed, notice, _, malformed, rejected = _history()
    assert notice.applied and not rejected.applied and not malformed.applied and not observed.applied
    assert not start.applied
    with pytest.raises(ValueError, match="malformed"):
        it.Step(seq=0, at=NOW, entry=None)
    with pytest.raises(TypeError, match="entry"):
        it.Step(seq=0, at=NOW, entry=_ref())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only an operation"):
        it.Step(seq=0, at=NOW, entry=observed.entry, code="k1_bucket")
    with pytest.raises(ValueError, match="only an operation"):
        it.Step(seq=0, at=NOW, entry=_start(), code="k1_bucket")
    with pytest.raises(ValueError, match="seq"):
        it.Step(seq=-1, at=NOW, entry=_notice())
    with pytest.raises(ValueError, match="belongs to no question"):
        it.Retrieved("q1", "new_evidence", "", ())
    with pytest.raises(ValueError, match="belongs to no question"):
        it.Retrieved(None, "evidence_search", "courtyard", ())


def test_cycle_steps_name_their_focus_and_marks_charge_nothing() -> None:
    """K5's stall count and a resume's budget are rebuilt from cycle, focus and tokens alone."""
    _, observed, _, _, malformed, rejected = _history()
    assert (malformed.cycle, malformed.focus, rejected.cycle) == (0, "q1", 1)
    assert observed.cycle is None and observed.tokens == 300  # the notice pass's tokens, no cycle
    with pytest.raises(ValueError, match="in focus"):
        it.Step(seq=0, at=NOW, entry=None, code="malformed", cycle=0)
    with pytest.raises(ValueError, match="cycle"):
        it.Step(seq=0, at=NOW, entry=None, code="malformed", cycle=-1, focus="q1")
    with pytest.raises(ValueError, match="tokens"):
        it.Step(seq=0, at=NOW, entry=_notice(), tokens=-5)
    for changes in ({"tokens": 10}, {"cycle": 0, "focus": "q1"}, {"focus": "q1"}):
        with pytest.raises(ValueError, match="a mark"):
            it.Step(seq=0, at=NOW, entry=_start(), **changes)


def test_run_start_records_what_a_fold_and_a_resume_need() -> None:
    start = _start(rules=it.NO_KERNEL, control="newest", wipe=True)
    assert (start.rules, start.control, start.wipe, start.budget) == (it.NO_KERNEL, "newest", True, it.Budget())
    with pytest.raises(ValueError, match="no focus"):
        _start(call="notice")
    with pytest.raises(ValueError, match="call"):
        _start(call="dream")
    with pytest.raises(ValueError, match="control"):
        _start(control="random")
    with pytest.raises(TypeError, match="KernelRules"):
        _start(rules=None)


def test_edit_names_exactly_what_it_changes() -> None:
    assert it.Edit("inject_hypothesis", "q1", statement="She means the rooftop").statement == "She means the rooftop"
    assert it.Edit("delete_prediction", "q1", prediction_id="p1").prediction_id == "p1"
    for kind, changes in (
        ("swap_lead", {}), ("swap_lead", {"hypothesis_id": "q1.h2", "prediction_id": "p1"}),
        ("delete_prediction", {"hypothesis_id": "q1.h2"}), ("inject_hypothesis", {}),
    ):
        with pytest.raises(ValueError, match="no other target"):
            it.Edit(kind, "q1", **changes)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="statement"):
        it.Edit("inject_hypothesis", "q1", statement=" ")
    with pytest.raises(ValueError, match="kind"):
        it.Edit("rewrite", "q1")  # type: ignore[arg-type]


def test_turns_are_indexed_to_logged_steps() -> None:
    reply = it.ModelReply(text="", tokens=12, model="glimmer-fake")  # a failed call: empty text, tokens kept
    event = it.WorkspaceEvent(step=_history()[4], before="0" * 64, after="1" * 64)
    run = it.InquiryRun(ws=_workspace(), events=(event,), decisions=(), turns=(it.Turn(4, "p", reply),), budget=it.Budget())
    assert run.turns[0].seq == run.events[0].step.seq
    with pytest.raises(ValueError, match="logged step"):
        dataclasses.replace(run, turns=(it.Turn(9, "p", reply),))


# ── The controller's values ───────────────────────────────────────────────────


def test_priority_is_stake_u_a_or_stake_when_ready() -> None:
    assert _priority(stake=it.STAKE_DUE, u=0.5, a=it.A_WAITING).value == pytest.approx(0.3)
    assert _priority(stake=it.STAKE_DUE, u=0.2, a=it.A_READY, ready=True).value == it.STAKE_DUE
    assert _priority(stake=it.STAKE_DUE, u=1.0, a=it.A_READY).value == 2.0  # QuestionView's ceiling
    for bad in ({"stake": 3.0}, {"u": 0.1}, {"a": 0.5}, {"ready": True}):
        with pytest.raises(ValueError):
            _priority(**bad)


def test_budget_spends_into_a_new_value_and_names_its_cap() -> None:
    budget = it.Budget()
    assert (budget.cycle_cap, budget.token_cap, budget.stop_reason) == (12, 20_000, None)
    spent = budget.spend(cycles=12, tokens=900)
    assert budget.cycles_used == 0 and spent.cycles_used == 12 and spent.stop_reason == "cycle_cap"
    assert budget.spend(tokens=20_000).stop_reason == "token_cap"
    assert (budget.tokens_left, it.Budget(tokens_used=19_500).tokens_left, budget.spend(tokens=21_000).tokens_left) == (
        20_000, 500, 0,
    )
    with pytest.raises(ValueError):
        it.Budget(tokens_used=-1)


def test_decision_names_its_question_and_reason() -> None:
    def decide(kind: Any, question_id: str | None, reason: Any) -> it.Decision:
        return it.Decision(kind=kind, question_id=question_id, reason=reason, priorities=(), budget=it.Budget())

    assert decide("stop", None, "below_floor").question_id is None
    assert decide("park", "q1", "stalled").reason == "stalled"
    assert decide("investigate", "q2", "stalled").kind == "investigate"
    for kind, question_id, reason in (
        ("stop", "q1", "cycle_cap"), ("continue", None, None), ("continue", "q1", "stalled"),
        ("park", "q1", None), ("stop", None, None), ("stop", None, "stalled"),
    ):
        with pytest.raises(ValueError):
            decide(kind, question_id, reason)


def test_kernel_rules_switch_off_k1_k2_and_k4_for_m_kernel_only() -> None:
    assert (it.FULL_RULES.k1, it.FULL_RULES.k2, it.FULL_RULES.k4) == (True, True, True)
    assert (it.NO_KERNEL.k1, it.NO_KERNEL.k2, it.NO_KERNEL.k4) == (False, False, False)


# ── Frozen values ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_samples()))
def test_every_contract_type_is_frozen_and_immutable(name: str) -> None:
    value = _samples()[name]
    assert type(value).__name__ == name
    assert dataclasses.is_dataclass(value) and not isinstance(value, type)
    first = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, first, None)
    for f in dataclasses.fields(value):
        assert not isinstance(getattr(value, f.name), (list, set, dict)), f"{name}.{f.name}"
    assert dataclasses.replace(value) == value
    hash(value)


# ── The operations' JSON ──────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", list(it.OP_TYPES.values()), ids=list(it.OP_TYPES))
def test_every_op_round_trips_through_its_documented_json(cls: type[it.Op]) -> None:
    data = _example(cls)
    op = it.op_from_json(data)
    assert type(op) is cls and op.name == data["op"]
    assert it.op_to_json(op) == data
    assert it.op_from_json(json.loads(json.dumps(it.op_to_json(op)))) == op


def test_m_kernel_assess_writes_credences_instead_of_likelihoods() -> None:
    op = it.AssessOp("q1", "e-d3-2", credences={"q1.h1": 0.7, "q1.h2": 0.2, "q1.else": 0.1})
    data = it.op_to_json(op)
    assert "likelihoods" not in data and data["credences"] == {"q1.h1": 0.7, "q1.h2": 0.2, "q1.else": 0.1}
    assert it.op_from_json(data) == op
    with pytest.raises(ValueError, match="exactly one"):
        it.AssessOp("q1", "e-d3-2")
    with pytest.raises(ValueError, match="exactly one"):
        it.AssessOp("q1", "e-d3-2", likelihoods={"q1.h1": 2.0}, credences={"q1.h1": 0.5})


def test_ops_copy_their_fields_into_immutable_values() -> None:
    given = {"q1.h1": 0.2, "q1.h2": 0.9}
    op = it.PredictOp("q1", "claim", given, 3)
    given["q1.h1"] = 0.8
    assert op.p_true_given["q1.h1"] == 0.2
    notice = it.NoticeOp("gap", "The shot list lacks a lens", ["t1"], ["e1"])  # type: ignore[arg-type]
    assert notice.target_ids == ("t1",) and notice.origin_evidence == ("e1",)
    # Rules are the kernel's: an op may carry no targets, and the kernel rejects it (no_targets).
    assert it.NoticeOp("gap", "The shot list lacks a lens", (), ()).target_ids == ()


def _with(name: str, **changes: Any) -> dict[str, Any]:
    return {**_example(it.OP_TYPES[name]), **changes}  # type: ignore[index]


NOT_OPS: tuple[tuple[str, object], ...] = (
    ("not an object", ["PARK", "q1"]),
    ("no op", {"question_id": "q1"}),
    ("unknown op", {"op": "WONDER", "question_id": "q1"}),
    ("missing field", {"op": "PARK"}),
    ("extra field", _with("PARK", why="tired")),
    ("empty id", _with("PARK", question_id="")),
    ("kind outside its literal", _with("NOTICE", kind="worry")),
    ("string for a list", _with("NOTICE", target_ids="t-scene4")),
    ("number in an id list", _with("CONCLUDE", cites=[3])),
    ("bool for a number", _with("PREDICT", horizon_days=True)),
    ("zero horizon", _with("PREDICT", horizon_days=0)),
    ("probability past 1", _with("PREDICT", p_true_given={"q1.h1": 1.5})),
    ("not a number", _with("PREDICT", p_true_given={"q1.h1": float("nan")})),
    ("hypothesis without a credence", _with("HYPOTHESIZE", hypotheses=[{"statement": "x"}])),
    ("non-positive likelihood", _with("ASSESS", likelihoods={"q1.h1": 0})),
    ("neither likelihoods nor credences", {"op": "ASSESS", "question_id": "q1", "evidence_id": "e1"}),
    ("both likelihoods and credences", _with("ASSESS", credences={"q1.h1": 0.5})),
    ("CONCLUDE can't retract", _with("CONCLUDE", kind="retraction")),
    ("unknown tool", _with("INVESTIGATE", tool="web")),
)


@pytest.mark.parametrize("data", [data for _, data in NOT_OPS], ids=[label for label, _ in NOT_OPS])
def test_strict_codec_refuses_what_isnt_one_operation(data: object) -> None:
    with pytest.raises(it.OpFormatError):
        it.op_from_json(data)


def test_codec_errors_never_repeat_the_models_text() -> None:
    with pytest.raises(it.OpFormatError) as caught:
        it.op_from_json(_with("NOTICE", kind="her private diagnosis"))
    with pytest.raises(it.OpFormatError) as keyed:
        it.op_from_json(_with("PREDICT", p_true_given={"her private diagnosis": 7.0}))
    assert "private" not in str(caught.value) and "private" not in str(keyed.value)


# ── Ports ─────────────────────────────────────────────────────────────────────

PORTS: dict[str, dict[str, tuple[str, ...]]] = {
    "Thinker": {"propose": ("rendered", "max_tokens")},
    "EvidenceSearch": {"search": ("query", "as_of", "k"), "get": ("ids", "as_of")},
    "WorldQuery": {"ask": ("text", "max_tokens")},
    "Research": {"research": ("question_id", "query")},
    "PredictionResolver": {"resolve": ("claim", "evidence")},
    "WorkspaceLog": {"append": ("event",), "events": ()},
}


@pytest.mark.parametrize("name", sorted(PORTS))
def test_port_is_a_checkable_protocol_with_its_methods(name: str) -> None:
    port = getattr(ports, name)
    assert Protocol in port.__mro__
    methods = {n for n, v in vars(port).items() if inspect.isfunction(v) and not n.startswith("_")}
    assert methods == set(PORTS[name])
    for method, params in PORTS[name].items():
        signature = inspect.signature(getattr(port, method))
        assert tuple(signature.parameters)[1:] == params, f"{name}.{method}"
        assert signature.return_annotation is not inspect.Signature.empty, f"{name}.{method}"
    stub = type(f"Fake{name}", (), {m: (lambda self, *a, **k: None) for m in PORTS[name]})
    assert isinstance(stub(), port) and not isinstance(object(), port)


def test_the_evidence_stores_search_fits_evidence_search() -> None:
    """``search`` is the store's as it is; ``get`` (fetch by id, time-masked) is 17-core's to add."""
    store = inspect.signature(JsonlEvidenceStore.search)
    port = inspect.signature(ports.EvidenceSearch.search)
    assert tuple(store.parameters) == tuple(port.parameters)


def test_research_is_absent_outside_the_live_night() -> None:
    fields = {f.name: f for f in dataclasses.fields(ports.InquiryPorts)}
    assert fields["research"].default is None
    assert [name for name, f in fields.items() if f.default is dataclasses.MISSING] == [
        "thinker", "search", "world", "resolver", "log",
    ]


# ── Purity and the boundary ───────────────────────────────────────────────────

PURE = ("types.py", "ops.py", "steps.py", "_check.py", "ports.py")
STDLIB_OK = frozenset({"__future__", "math", "collections.abc", "dataclasses", "datetime", "types", "typing"})
IMPURE_NAMES = frozenset({"now", "utcnow", "today", "getenv", "environ", "open", "connect", "urlopen", "time"})


def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.append(node.module or "")
    return names


@pytest.mark.parametrize("module", PURE)
def test_inquiry_contract_reads_no_clock_database_model_or_env(module: str) -> None:
    path = INQUIRY / module
    for name in _imports(path):
        assert name in STDLIB_OK or name.startswith("anjo.sleep.mind.inquiry.") or name == "anjo.sleep.types", (
            f"{module} imports {name}"
        )
    for node in ast.walk(ast.parse(path.read_text())):
        called = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        assert called not in IMPURE_NAMES, f"{module} uses {called}"


def test_importing_the_contracts_loads_only_the_engine_core() -> None:
    script = (
        "import json, sys\n"
        "import anjo.sleep.mind.inquiry.types, anjo.sleep.mind.inquiry.ports, anjo.sleep.mind.apd.types\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('anjo', 'torch', 'transformers', 'vllm', 'peft', 'sqlite3', 'httpx', 'requests'))))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60)
    loaded = json.loads(done.stdout)
    allowed = ("anjo.sleep.mind.inquiry", "anjo.sleep.mind.apd")
    assert all(
        name in ("anjo", "anjo.sleep", "anjo.sleep.types", "anjo.sleep.mind") or name.startswith(allowed)
        for name in loaded
    ), loaded


@pytest.mark.parametrize("package", [INQUIRY, APD], ids=["inquiry", "apd"])
def test_mind_packages_import_no_private_anjo(package: Path) -> None:
    """The import-linter contract ("the engine core imports no other anjo package"), checked here
    for every module the three Task 17 slots add under these packages."""
    for path in sorted(package.rglob("*.py")):
        for name in _imports(path):
            if name == "anjo" or name.startswith("anjo."):
                assert name.startswith("anjo.sleep") and not name.startswith("anjo.sleep.app"), (
                    f"{path.relative_to(package)} imports {name}"
                )
