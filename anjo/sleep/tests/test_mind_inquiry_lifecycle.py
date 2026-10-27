# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A question's whole life across nights (Task 17): a parked question wakes on new evidence on its
targets, a resolution can still be retracted, M-persist's wiped notice pass still sees earlier
evidence, a failed resolver call consumes nothing, a prediction's horizon is bounded, ids survive a
kill and resume, and the person's digest covers her workspace's records."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.targets import EventLogTargetStore
from anjo.sleep.mind.inquiry import controller, kernel, loop
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import JsonlWorkspaceLog, MemoryWorkspaceLog
from anjo.sleep.mind.inquiry.ports import InquiryPorts, WorkspaceLog
from anjo.sleep.tests.inquiry_fakes import (
    PERSON,
    FakeResolver,
    Kill,
    KillingLog,
    ScriptedThinker,
    Ws,
    commitment,
    day,
    ev,
    hypothesize,
    night,
    noon,
    notice_op,
    ports,
)
from anjo.sleep.types import Evidence, Target

SCENE = "t-scene4"
LOOK = ev("e-d2-1", "For scene 4 they want an 85 mm medium shot on full frame.", n=2, target=SCENE)
WINDOW = ev("e-d3-1", "They might shoot scene 4 through the window from the courtyard.", n=3, target=SCENE)
KITCHEN = ev("e-d5-1", "Scene 4 is in their 3 m kitchen, handheld.", n=5, target=SCENE)
LOCKED = ev("e-d6-1", "The courtyard is locked on Thursday.", n=6, target=SCENE)
OPEN = ev("e-d6-2", "The courtyard is open all week for the scene 4 shoot.", n=6, target=SCENE)
ELSEWHERE = ev("e-d6-3", "The festival entry form wants a logline.", n=6, target="t-festival")
ITEMS = (LOOK, WINDOW, KITCHEN, LOCKED, OPEN, ELSEWHERE)
TARGETS = (commitment(SCENE, "Shoot scene 4 on Thursday", due=noon(7), n=2),
           commitment("t-festival", "Festival entry", due=noon(20), n=2))
Q = "q1"  # the first applied operation of the life opens it
H = [f"{Q}.h2a", f"{Q}.h2b", f"{Q}.h2c", f"{Q}.else"]
NOTICED = [{"op": "NOTICE", "kind": "contradiction", "subject": "Scene 4's 85 mm medium shot against a 3 m kitchen",
            "target_ids": [SCENE], "origin_evidence": [LOOK.id, KITCHEN.id]}]
HYPOTHESIZE = {"op": "HYPOTHESIZE", "question_id": Q, "hypotheses": [
    {"statement": "The plan assumes room for an 85 mm medium shot that the kitchen lacks", "credence": 0.4},
    {"statement": "The 85 mm shot is taken through the window from the courtyard", "credence": 0.35},
    {"statement": "Scene 4 moves to a wider lens than 85 mm", "credence": 0.15}], "catch_all": 0.1}


def _predict(p: Sequence[float] = (0.6, 0.95, 0.6, 0.5), horizon: Any = 3) -> dict[str, Any]:
    return {"op": "PREDICT", "question_id": Q, "claim": "The courtyard is open for the scene 4 shoot",
            "p_true_given": dict(zip(H, p, strict=True)), "horizon_days": horizon}


def _search(query: str) -> dict[str, Any]:
    return {"op": "INVESTIGATE", "question_id": Q, "tool": "evidence_search", "query": query}


def _assess(item: Evidence, likelihoods: Sequence[float]) -> dict[str, Any]:
    return {"op": "ASSESS", "question_id": Q, "evidence_id": item.id, "likelihoods": dict(zip(H, likelihoods, strict=True))}


WORLD = {"op": "INVESTIGATE", "question_id": Q, "tool": "world_query", "query": "How far back does an 85 mm medium shot need?"}


class Failing(FakeResolver):
    """Fails its first ``failures`` calls, then answers like ``FakeResolver``."""

    def __init__(self, verdicts: dict[str, it.Verdict], failures: int) -> None:
        super().__init__(verdicts)
        self.failures = failures

    def resolve(self, claim: str, evidence: Sequence[Evidence]) -> it.ResolverReply:
        if self.failures > 0:
            self.failures -= 1
            self.calls.append((claim, tuple(evidence)))
            return it.ResolverReply("undetermined", 200, failed=True)
        return super().resolve(claim, evidence)


class Life:
    """One person's nights on one log, as the night glue runs them, with one thinker throughout."""

    def __init__(self, log: WorkspaceLog | None = None, resolver: FakeResolver | None = None,
                 wipe: bool = False) -> None:
        self.log = log if log is not None else MemoryWorkspaceLog()
        self.thinker = ScriptedThinker([])
        self.resolver = resolver or FakeResolver({"locked": "false", "open all week": "true"})
        self.wipe = wipe
        self.ws = it.Workspace()
        self.runs: list[it.InquiryRun] = []

    def ports(self) -> InquiryPorts:
        return ports(ITEMS, [], thinker=self.thinker, resolver=self.resolver, log=self.log)  # type: ignore[arg-type]

    def _keep(self, run: it.InquiryRun) -> it.InquiryRun:
        self.ws = run.ws
        self.runs.append(run)
        return run

    def notice(self, n: int, new: Sequence[Evidence], reply: Any, targets: Sequence[Target] = TARGETS) -> it.InquiryRun:
        self.thinker.script.append(reply)
        return self._keep(loop.notice(self.ws, new, self.ports(), it.Budget(), night=day(n), targets=targets,
                                      as_of=night(n), wipe=self.wipe))

    def inquire(self, n: int, script: list[Any], budget: it.Budget = it.Budget()) -> it.InquiryRun:
        self.thinker.script.extend(script)
        return self._keep(loop.run_inquiry(self.ws, self.ports(), budget, night=day(n), targets=TARGETS,
                                           as_of=night(n)))


def _entries(run: it.InquiryRun, kind: type) -> list[Any]:
    return [e.step.entry for e in run.events if isinstance(e.step.entry, kind)]


# ── 1. A parked question wakes on new evidence on its targets ─────────────────


def test_parked_question_wakes_on_new_evidence_on_its_targets() -> None:
    life = Life()
    life.notice(5, [LOOK, KITCHEN], NOTICED)
    life.inquire(5, [HYPOTHESIZE, _predict(), {"op": "PARK", "question_id": Q}])
    assert life.ws.question(Q).status == "parked"  # type: ignore[union-attr]

    quiet = life.notice(6, [ELSEWHERE], "[]")
    assert _entries(quiet, it.Wake) == [] and life.ws.question(Q).status == "parked"  # type: ignore[union-attr]

    woke = life.notice(6, [LOCKED], "[]")
    (wake,) = _entries(woke, it.Wake)
    assert (wake.question_id, [i.id for i in wake.items]) == (Q, [LOCKED.id])
    assert wake.items[0].target_id == SCENE and life.ws.question(Q).status == "active"  # type: ignore[union-attr]

    night6 = life.inquire(6, [WORLD])
    assert night6.decisions[0].question_id == Q and len(life.resolver.calls) == 1
    assert life.ws.predictions[0].outcome == "false"


def test_a_wake_needs_a_parked_question_and_an_item_on_its_targets() -> None:
    w = Ws()
    w.start(5)
    w.arrive(LOOK, KITCHEN)
    w.ok(notice_op([LOOK.id, KITCHEN.id]))
    question = w.ws.questions[0].id
    with pytest.raises(ValueError, match="parked"):
        w.step(it.Wake(question, (kernel_ref(LOCKED),)))
    w.ok(it.ParkOp(question))
    with pytest.raises(ValueError, match="target"):
        w.step(it.Wake(question, (kernel_ref(ELSEWHERE),)))
    w.ok(it.Wake(question, (kernel_ref(LOCKED),)))
    assert w.ws.question(question).status == "open"  # type: ignore[union-attr]
    assert kernel.wakes(w.ws, [kernel_ref(LOCKED)]) == ()


def kernel_ref(item: Evidence) -> it.EvidenceRef:
    return it.EvidenceRef(id=item.id, day=item.day, created_at=item.created_at, target_id=item.target_id)


# ── 2. A resolution can be retracted ──────────────────────────────────────────


def test_a_resolution_can_be_retracted() -> None:
    life = Life()
    life.notice(5, [LOOK, KITCHEN], NOTICED)
    life.inquire(5, [
        HYPOTHESIZE, _predict((0.05, 0.95, 0.5, 0.5)), _search("courtyard window"),
        _assess(WINDOW, (4, 0.25, 0.25, 0.25)),
        {"op": "CONCLUDE", "question_id": Q, "hypothesis_id": H[0], "kind": "resolution",
         "statement": "The kitchen can't hold an 85 mm medium shot", "cites": [LOOK.id, KITCHEN.id, WINDOW.id]},
    ])
    finding = life.ws.finding_of(Q)
    assert finding is not None and finding.kind == "resolution" and life.ws.question(Q).status == "concluded"  # type: ignore[union-attr]
    assert life.ws.predictions[0].pending
    assert controller.decide(life.ws, it.Budget(), focus=None, targets=TARGETS, as_of=night(5)).kind == "stop"

    life.notice(6, [OPEN], "[]")
    night6 = life.inquire(6, [
        {"op": "RETRACT", "question_id": Q, "statement": "The courtyard is open, so the shot needn't fit the kitchen",
         "cites": [OPEN.id]},
    ])
    assert night6.decisions[0].question_id == Q and life.ws.predictions[0].outcome == "true"
    assert kernel.retraction_ready(life.ws, Q) is False  # done: retracted
    retracted = life.ws.finding_of(Q)
    assert retracted is not None and (retracted.id, retracted.kind) == (finding.id, "retraction")
    assert life.ws.question(Q).status == "retracted"  # type: ignore[union-attr]


def test_a_concluded_question_takes_evidence_and_retracts_in_the_kernel() -> None:
    w = Ws()
    w.start(5)
    w.arrive(LOOK, KITCHEN)
    w.ok(notice_op([LOOK.id, KITCHEN.id]))
    question = w.ws.questions[0].id
    w.ok(hypothesize(question, 0.5, 0.3))
    ids = w.ids(question)
    w.found(question, WINDOW)
    w.ok(it.AssessOp(question, WINDOW.id, likelihoods=dict(zip(ids, (4.0, 0.25, 0.25), strict=True))))
    w.ok(it.ConcludeOp(question, ids[0], "resolution", "The kitchen can't hold it", (WINDOW.id,)))
    assert w.code(it.HypothesizeOp(question, (it.Alternative("x", 0.5), it.Alternative("y", 0.5)), 0.6)) == "not_live"
    w.found(question, OPEN, LOCKED)
    for item in (OPEN, LOCKED):
        w.ok(it.AssessOp(question, item.id, likelihoods=dict(zip(ids, (0.25, 4.0, 4.0), strict=True))))
    assert kernel.retraction_ready(w.ws, question)
    w.ok(it.RetractOp(question, "It fits after all", (OPEN.id,)))
    assert w.ws.finding_of(question).kind == "retraction"  # type: ignore[union-attr]


# ── 3. M-persist keeps the evidence ───────────────────────────────────────────


def test_wiped_notice_pass_still_sees_earlier_evidence() -> None:
    life = Life(wipe=True)
    life.notice(2, [LOOK], "[]")
    night5 = life.notice(5, [KITCHEN], NOTICED)
    prompt = life.thinker.prompts[-1]
    assert LOOK.text in prompt.split("## Earlier evidence you hold")[1].split("## Today's new evidence")[0]
    held = [e for e in _entries(night5, it.Retrieved) if e.tool == "held_evidence"]
    assert [[i.id for i in r.items] for r in held] == [[LOOK.id]]
    assert [e.step.code for e in night5.events if isinstance(e.step.entry, it.NoticeOp)] == [None]
    assert life.ws.questions[0].origin_evidence == (LOOK.id, KITCHEN.id)
    assert isinstance(life.ws.history[0].entry, it.RunStart) and life.ws.history[0].entry.wipe
    assert kernel.fold(life.log.events()) == life.ws


# ── 4. A failed resolver call consumes nothing ────────────────────────────────


def test_a_failed_resolver_call_consumes_no_evidence_and_is_retried() -> None:
    life = Life(resolver=Failing({"locked": "false"}, failures=1))
    life.notice(5, [LOOK, KITCHEN], NOTICED)
    life.inquire(5, [HYPOTHESIZE, _predict(), WORLD, WORLD, WORLD])
    life.notice(6, [LOCKED], "[]")
    night6 = life.inquire(6, [WORLD, WORLD], budget=it.Budget(cycle_cap=2))
    assert len(life.resolver.calls) == 2 and night6.resolver_tokens == 220
    resolved = _entries(night6, it.Resolved)
    assert [(r.outcome, [i.id for i in r.seen]) for r in resolved] == [("false", [LOCKED.id])]
    assert life.ws.predictions[0].outcome == "false"


def test_a_failed_resolver_call_at_the_horizon_doesnt_censor() -> None:
    life = Life(resolver=Failing({"locked": "false"}, failures=1))
    life.notice(5, [LOOK, KITCHEN], NOTICED)
    life.inquire(5, [HYPOTHESIZE, _predict(), WORLD, WORLD, WORLD])
    life.notice(6, [LOCKED], "[]")
    first = life.inquire(9, [], budget=it.Budget(cycle_cap=0))
    assert _entries(first, it.Resolved) == [] and life.ws.predictions[0].pending
    life.inquire(10, [], budget=it.Budget(cycle_cap=0))
    assert life.ws.predictions[0].outcome == "false"


# ── 5. A bounded horizon ──────────────────────────────────────────────────────


@pytest.mark.parametrize(("days", "code"), [(1, None), (30, None), (31, "malformed"), (3_000_000, "malformed")])
def test_a_prediction_horizon_is_bounded(days: int, code: it.RejectCode | None) -> None:
    w = Ws()
    w.start(5)
    w.arrive(LOOK, KITCHEN)
    w.ok(notice_op([LOOK.id, KITCHEN.id]))
    question = w.ws.questions[0].id
    w.ok(hypothesize(question, 0.5, 0.3))
    op = it.PredictOp(question, "claim", dict(zip(w.ids(question), (0.9, 0.1, 0.5), strict=True)), days)
    assert kernel.check(w.ws, op) == code


def test_a_huge_horizon_is_logged_malformed_not_raised() -> None:
    life = Life()
    life.notice(5, [LOOK, KITCHEN], NOTICED)
    run = life.inquire(5, [HYPOTHESIZE, _predict(horizon=3_000_000)], budget=it.Budget(cycle_cap=2))
    assert [e.step.code for e in run.events if isinstance(e.step.entry, it.PredictOp)] == ["malformed"]
    assert life.ws.predictions == ()


# ── 6. Ids from an op counter; the records digest ─────────────────────────────


def test_ids_count_applied_operations_only() -> None:
    w = Ws()
    w.start(5)
    w.arrive(LOOK, KITCHEN)
    w.ok(notice_op([LOOK.id, KITCHEN.id]))
    assert w.ws.questions[0].id == "q1"
    w.ok(it.PredictOp("q1", "claim", {"q1.h9a": 0.5}, 3), code="k1_hypotheses")
    w.start(6)
    w.ok(hypothesize("q1", 0.5, 0.3))
    assert w.ids("q1") == ["q1.h2a", "q1.h2b", "q1.else"] and w.ws.ops == 2
    w.start(7, wipe=True)
    assert w.ws.ops == 2  # a wipe keeps the counter, so ids stay unique across it


def test_kill_resume_records_digest_identical() -> None:
    script = [HYPOTHESIZE, _predict(), _search("courtyard window"), _assess(WINDOW, (0.5, 4, 1, 1)), WORLD, WORLD, WORLD]
    whole = Life()
    whole.notice(5, [LOOK, KITCHEN], NOTICED)
    uninterrupted = whole.inquire(5, list(script))
    for kill_at in [e.step.seq for e in uninterrupted.events]:
        log = KillingLog()
        killed = Life(log=log)
        killed.notice(5, [LOOK, KITCHEN], NOTICED)
        log.kill_after = kill_at
        with pytest.raises(Kill):
            killed.inquire(5, list(script))
        killed.ws = kernel.fold(log.events())
        killed._keep(loop.resume_inquiry(log.events(), killed.ports(), targets=TARGETS))
        assert kernel.records_digest(killed.ws) == kernel.records_digest(whole.ws), kill_at
        assert [h.id for h in killed.ws.hypotheses] == [h.id for h in whole.ws.hypotheses] == H
        assert [p.id for p in killed.ws.predictions] == [p.id for p in whole.ws.predictions]


def test_the_persons_digest_covers_her_workspace_records(tmp_path: Path) -> None:
    log = JsonlWorkspaceLog(tmp_path / "workspace.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    store = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD, workspace=log)
    bare = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    w = Ws()
    w.start(5)
    log.append(w.events[0])
    before = store.digest()
    assert before == store.replay_digest() and before != bare.digest()
    w.arrive(LOOK, KITCHEN)
    log.append(w.events[-1])
    assert store.digest() == before  # evidence arriving changes no record
    w.ok(notice_op([LOOK.id, KITCHEN.id]))
    log.append(w.events[-1])
    assert store.digest() != before and store.digest() == store.replay_digest()


# ── 7. The export folds under the rules the log recorded ──────────────────────


def test_the_export_folds_under_the_recorded_denylist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = JsonlWorkspaceLog(tmp_path / "workspace.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    life = Life(log=log)
    life.notice(5, [LOOK, KITCHEN], [{**NOTICED[0], "subject": "Scene 4's zebra dolly against a 3 m kitchen"}])
    start = life.runs[-1].events[0].step.entry
    assert isinstance(start, it.RunStart) and start.denylist == kernel.DENYLIST_VERSION

    newer = {**kernel.DENYLISTS[kernel.DENYLIST_VERSION], "traits": ("zebra*",)}
    monkeypatch.setattr(kernel, "DENYLISTS", {**kernel.DENYLISTS, "denylist-test": newer})
    monkeypatch.setattr(kernel, "DENYLIST_VERSION", "denylist-test")
    exported = log.export()["workspace"]
    assert isinstance(exported, dict) and exported["state"]["questions"][0]["id"] == Q
    assert life.notice(6, [LOCKED], [{**NOTICED[0], "subject": "Scene 4's zebra dolly again",
                                      "origin_evidence": [LOOK.id, LOCKED.id]}]).events[-1].step.code == "domain_denylist"


def test_an_unfoldable_log_still_exports_its_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = JsonlWorkspaceLog(tmp_path / "workspace.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    Life(log=log).notice(5, [LOOK, KITCHEN], NOTICED)
    monkeypatch.setattr(kernel, "DENYLISTS", {})  # the version the log recorded is gone
    exported = log.export()["workspace"]
    assert isinstance(exported, dict) and exported["state"] is None
    assert [e["step"]["seq"] for e in exported["events"]] == [e.step.seq for e in log.events()]
