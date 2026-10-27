"""MIND's signature demo on a fake thinker (Task 17): the film student's scene 4, planted in her
synthetic life, with a kill mid-inquiry and a clean resume.

Day 2 wants an 85 mm medium shot, day 3 floats the courtyard window, day 5 puts scene 4 in a 3 m
kitchen, day 6 locks the courtyard on Thursday. Her nights show the notice, three hypotheses, a
committed prediction, the window hit raising the courtyard, day 6 resolving the prediction false,
the resolution and a note anchored on Thursday's shoot with two fixes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from anjo.sleep.mind import snapshot
from anjo.sleep.mind.inquiry import findings, kernel, loop
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog
from anjo.sleep.tests.inquiry_fakes import (
    FakeResolver,
    Kill,
    KillingLog,
    ScriptedThinker,
    commitment,
    day,
    ev,
    night,
    noon,
    ports,
)
from anjo.sleep.types import Evidence

SCENE = "t-scene4"
LOOK = ev("e-d2-1", "For scene 4 I want the compressed long-lens look: an 85 mm medium shot on full frame.", n=2,
          target=SCENE)
WINDOW = ev("e-d3-1", "Maybe I shoot scene 4 through the window from the courtyard.", n=3, target=SCENE)
KITCHEN = ev("e-d5-1", "Scene 4 is in my 3 m kitchen, handheld.", n=5, target=SCENE)
LOCKED = ev("e-d6-1", "The courtyard is locked on Thursday.", n=6, target=SCENE)
ITEMS = (LOOK, WINDOW, KITCHEN, LOCKED)
THURSDAY = commitment(SCENE, "Shoot scene 4 on Thursday", due=noon(7), n=2)
TARGETS = (THURSDAY,)

Q = "q6"  # opened by the night-5 notice, the log's seventh step
H = [f"{Q}.h8a", f"{Q}.h8b", f"{Q}.h8c", f"{Q}.else"]  # kitchen lacks room, courtyard, wider lens, else
FIXES = ("Scene 4's 85 mm medium shot needs about 3 m the kitchen doesn't have, and the courtyard is locked on "
         "Thursday: shoot it on a 50 mm from the doorway, or keep the 85 mm and frame a close-up.")

NOTICE_5 = [{"op": "NOTICE", "kind": "contradiction", "subject": "Scene 4's 85 mm medium shot against a 3 m kitchen",
             "target_ids": [SCENE], "origin_evidence": [LOOK.id, KITCHEN.id]}]
NIGHT_5 = [
    {"op": "HYPOTHESIZE", "question_id": Q, "hypotheses": [
        {"statement": "The plan assumes room for an 85 mm medium shot that the kitchen lacks", "credence": 0.4},
        {"statement": "The 85 mm shot is taken through the window from the courtyard", "credence": 0.35},
        {"statement": "Scene 4 moves to a wider lens than 85 mm", "credence": 0.15}], "catch_all": 0.1},
    {"op": "PREDICT", "question_id": Q, "claim": "The courtyard is open for the scene 4 shoot",
     "p_true_given": dict(zip(H, (0.6, 0.95, 0.6, 0.5), strict=True)), "horizon_days": 3},
    {"op": "INVESTIGATE", "question_id": Q, "tool": "evidence_search", "query": "courtyard window"},
    {"op": "ASSESS", "question_id": Q, "evidence_id": WINDOW.id, "likelihoods": dict(zip(H, (0.5, 4, 1, 1), strict=True))},
    *[{"op": "INVESTIGATE", "question_id": Q, "tool": "world_query",
       "query": "How far back does an 85 mm medium shot on full frame need to be?"}] * 3,
]
NIGHT_6 = [
    {"op": "INVESTIGATE", "question_id": Q, "tool": "evidence_search", "query": "scene 4 kitchen 85 mm lens"},
    {"op": "ASSESS", "question_id": Q, "evidence_id": KITCHEN.id, "likelihoods": dict(zip(H, (4, 0.25, 0.5, 1), strict=True))},
    {"op": "ASSESS", "question_id": Q, "evidence_id": LOOK.id, "likelihoods": dict(zip(H, (2, 1, 0.25, 1), strict=True))},
    {"op": "CONCLUDE", "question_id": Q, "hypothesis_id": H[0], "kind": "resolution", "statement": FIXES,
     "cites": [LOOK.id, KITCHEN.id, LOCKED.id]},
]


class Life:
    """Her nights 2 to 6 on one log, as the night glue runs them: the notice pass, then inquiry."""

    def __init__(self, log: MemoryWorkspaceLog) -> None:
        self.log = log
        self.ws = it.Workspace()
        self.thinker = ScriptedThinker([])
        self.resolver = FakeResolver({"locked": "false"})
        self.runs: list[it.InquiryRun] = []

    def _ports(self) -> Any:
        return ports(ITEMS, [], thinker=self.thinker, resolver=self.resolver, log=self.log)

    def notice(self, n: int, new: Sequence[Evidence], reply: Any) -> None:
        self.thinker.script.append(reply)
        run = loop.notice(self.ws, new, self._ports(), it.Budget(), night=day(n), targets=TARGETS, as_of=night(n))
        self.ws = run.ws
        self.runs.append(run)

    def inquire(self, n: int, script: list[Any]) -> None:
        self.thinker.script.extend(script)
        run = loop.run_inquiry(self.ws, self._ports(), it.Budget(), night=day(n), targets=TARGETS, as_of=night(n))
        self.ws = run.ws
        self.runs.append(run)

    def resume(self) -> None:
        run = loop.resume_inquiry(self.log.events(), self._ports(), targets=TARGETS)
        self.ws = run.ws
        self.runs.append(run)


def _through_night_5(life: Life) -> None:
    life.notice(2, [LOOK], "[]")
    life.notice(3, [WINDOW], "[]")
    life.notice(5, [KITCHEN], NOTICE_5)


def _night_6(life: Life) -> None:
    life.notice(6, [LOCKED], "[]")
    life.inquire(6, list(NIGHT_6))


def _credences(ws: it.Workspace) -> list[float]:
    return [round(h.credence, 3) for h in ws.hypotheses_of(Q)]


def test_signature_demo_film_student_scene_4() -> None:
    life = Life(MemoryWorkspaceLog())
    _through_night_5(life)
    question = life.ws.question(Q)
    assert question is not None and (question.kind, question.target_ids) == ("contradiction", (SCENE,))  # 1. the notice
    assert question.origin_evidence == (LOOK.id, KITCHEN.id)

    life.inquire(5, list(NIGHT_5))
    night5 = life.runs[-1]
    assert [h.id for h in life.ws.hypotheses_of(Q)] == H  # 2. three hypotheses and the catch-all
    prediction = life.ws.predictions[0]
    assert (prediction.committed_at, prediction.outcome) == (night(5), None)  # 3. committed before day 6 exists
    assert _credences(life.ws) == [0.108, 0.757, 0.081, 0.054]  # 4. the window hit raises the courtyard
    assert [(d.kind, d.reason) for d in night5.decisions][-2:] == [("park", "stalled"), ("stop", "below_floor")]
    assert life.resolver.calls == []

    _night_6(life)
    (claim, shown), = life.resolver.calls
    assert claim == prediction.claim and [i.id for i in shown] == [LOCKED.id]
    assert life.ws.predictions[0].outcome == "false"  # 5. day 6 resolves it false
    resolved = next(u for u in life.ws.updates if u.source == "resolved_prediction")
    assert resolved.evidence_ids == (LOCKED.id,)
    finding = life.ws.finding_of(Q)
    assert finding is not None and (finding.kind, finding.hypothesis_id) == ("resolution", H[0])  # 6. the resolution
    assert finding.p == pytest.approx(0.895, abs=0.001) and life.ws.question(Q).status == "concluded"  # type: ignore[union-attr]

    result = findings.share_result(life.ws, Q, cited=(LOOK, KITCHEN, LOCKED), targets=TARGETS, results=(),
                                   as_of=night(6))
    assert result is not None and (result.basis, result.target_id, result.user_cites) == ("evidence", SCENE, 3)
    assert snapshot.build(list(TARGETS), [result], night(6), None).results == (result,)  # 7. the note on Thursday's shoot
    assert "50 mm" in finding.statement and "close-up" in finding.statement  # its two fixes
    assert findings.artifact(life.ws, Q, cited=(LOOK, KITCHEN, LOCKED), person_id="p", night=day(6), at=night(6),
                             model="glimmer") is None  # no source, so nothing to train on
    assert kernel.fold(life.log.events()) == life.ws


def test_signature_demo_survives_a_kill_mid_inquiry() -> None:
    whole = Life(MemoryWorkspaceLog())
    _through_night_5(whole)
    whole.inquire(5, list(NIGHT_5))
    predicted = next(e for e in whole.runs[-1].events if isinstance(e.step.entry, it.PredictOp))

    log = KillingLog()
    killed = Life(log)
    _through_night_5(killed)
    log.kill_after = predicted.step.seq
    with pytest.raises(Kill):
        killed.inquire(5, list(NIGHT_5))

    rebuilt = kernel.fold(log.events())
    assert kernel.digest(rebuilt) == predicted.after  # rebuilt from her log, same state
    killed.ws = rebuilt
    killed.resume()
    assert _credences(killed.ws) == _credences(whole.ws)
    assert [(p.claim, p.committed_at) for p in killed.ws.predictions] == [
        (p.claim, p.committed_at) for p in whole.ws.predictions
    ]
    assert kernel.fold(log.events()) == killed.ws

    _night_6(whole)
    _night_6(killed)
    ours, theirs = whole.ws.finding_of(Q), killed.ws.finding_of(Q)
    assert ours is not None and theirs is not None
    assert (theirs.kind, theirs.hypothesis_id, theirs.statement, round(theirs.p, 9)) == (
        ours.kind, ours.hypothesis_id, ours.statement, round(ours.p, 9))
