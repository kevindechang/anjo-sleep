# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The bridge from MIND's neural core to her workspace: an ignition opens a question through the
kernel, citing real events only, and with the core's switch off the night is today's."""
from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.mind.core import bridge
from anjo.sleep.mind.core.types import Decision, Ignition, Predictions, SlotPrediction
from anjo.sleep.mind.inquiry import codec, controller, kernel, loop
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog
from anjo.sleep.tests.inquiry_fakes import FakeScreen, Ws, commitment, day, ev, night, noon, ports
from anjo.sleep.types import Want, WantSnapshot

SHOOT = "t-shoot"
CAMERA = ev("e-d2-1", "The camera rental is confirmed for Thursday's shoot.", n=2, target=SHOOT)
CANCELLED = ev("e-d5-1", "The rental house cancelled on them again last week.", n=5, target=SHOOT)
TARGETS = (commitment(SHOOT, "Thursday's shoot", due=noon(8), n=2),)


def _held() -> tuple[Ws, MemoryWorkspaceLog]:
    w = Ws()
    w.start(5)
    w.arrive(CAMERA, CANCELLED)
    return w, MemoryWorkspaceLog(tuple(w.events))


def test_ignition_opens_a_question_through_the_kernel() -> None:
    w, log = _held()
    screen = FakeScreen()
    fired = Ignition(slot=2, other=None, kind="risk", score=0.91, cited_event_ids=(CANCELLED.id, "dreamt-7", CAMERA.id))
    done = bridge.ignite(w.ws, (fired,), log, targets=TARGETS, as_of=night(5), screen=screen)
    question = done.ws.questions[-1]
    assert question.origin_evidence == (CAMERA.id, CANCELLED.id)  # real items only, oldest first
    assert question.target_ids == (SHOOT,) and question.status == "open"
    assert screen.seen and screen.seen[0][1] == "question"
    assert kernel.fold(log.events()) == done.ws  # logged through the kernel, one step at a time
    assert done.events and all(e.step.applied for e in done.events)

    dreamt_only = Ignition(slot=3, other=None, kind="risk", score=0.95, cited_event_ids=("dreamt-1", "dreamt-2"))
    again = bridge.ignite(done.ws, (dreamt_only,), log, targets=TARGETS, as_of=night(5))
    assert again.ws == done.ws and again.events == ()
    assert again.dropped == ((dreamt_only, "nothing_real"),)


def test_a_collision_opens_no_question_unless_contradictions_are_switched_on() -> None:
    w, log = _held()
    collided = Ignition(slot=1, other=2, kind="collision", score=0.97, cited_event_ids=(CAMERA.id, CANCELLED.id))
    off = bridge.ignite(w.ws, (collided,), log, targets=TARGETS, as_of=night(5))
    assert off.ws == w.ws and off.events == () and log.events() == tuple(w.events)
    assert off.dropped == ((collided, "contradictions_off"),)
    assert not bridge.contradictions_on({}) and not bridge.contradictions_on({bridge.CONTRADICTIONS_ENV: "0"})

    assert bridge.contradictions_on({bridge.CONTRADICTIONS_ENV: "1"})
    on = bridge.ignite(w.ws, (collided,), log, targets=TARGETS, as_of=night(5), collisions=True)
    question = on.ws.questions[-1]
    assert question.kind == "contradiction" and question.origin_evidence == (CAMERA.id, CANCELLED.id)
    assert question.subject == "Whether the plans for Thursday's shoot agree with each other"


def _script() -> list[Any]:
    return [
        {"op": "HYPOTHESIZE", "question_id": "q1", "hypotheses": [
            {"statement": "The rental house cancels this booking too", "credence": 0.5},
            {"statement": "The booking holds this time", "credence": 0.4}], "catch_all": 0.1},
        {"op": "INVESTIGATE", "question_id": "q1", "tool": "evidence_search", "query": "rental cancelled"},
    ]


def _inquire(decider: loop.Decider | None) -> tuple[it.InquiryRun, MemoryWorkspaceLog]:
    w, _ = _held()
    w.ok(it.NoticeOp("anomaly", "Whether Thursday's camera booking holds", (SHOOT,), (CAMERA.id, CANCELLED.id)))
    log = MemoryWorkspaceLog(tuple(w.events))
    run_ports = ports((CAMERA, CANCELLED), _script(), log=log)
    if decider is None:
        return loop.run_inquiry(w.ws, run_ports, it.Budget(), night=day(5), targets=TARGETS, as_of=night(5)), log
    return loop.run_decided(decider, w.ws, run_ports, it.Budget(), night=day(5), targets=TARGETS, as_of=night(5)), log


def test_core_switch_off_is_todays_behavior() -> None:
    assert not bridge.core_on({}) and not bridge.core_on({bridge.CORE_ENV: "0"})
    assert bridge.core_on({bridge.CORE_ENV: "1"})

    view = bridge.CoreView(
        decision=Decision(speak=True, slot=0, p_speak=0.9),
        predictions=Predictions(speak=0.9, none=0.1, slots=(SlotPrediction(
            slot=0, usage=0.8, risk=0.9, hazard=(0.5,), missing=0.1, self_error=0.01, choice=0.9,
            cites=(CAMERA.id,)),)),
        targets_of={CAMERA.id: SHOOT},
    )
    inner = StateChooser()
    assert bridge.night_chooser({}, inner, lambda snapshot: view) is inner
    assert bridge.night_decide({}, view) is loop.CONTROLLER and loop.CONTROLLER.decide is controller.decide
    assert isinstance(bridge.night_chooser({bridge.CORE_ENV: "1"}, inner, lambda snapshot: view), bridge.CoreChooser)

    today, _ = _inquire(None)
    off, _ = _inquire(bridge.night_decide({}, view))
    assert off.events == today.events and off.decisions == today.decisions and off.ws == today.ws


def _starts(run: it.InquiryRun) -> list[it.RunStart]:
    return [e.step.entry for e in run.events if isinstance(e.step.entry, it.RunStart)]


def test_resume_keeps_the_core_decider() -> None:
    core = bridge.night_decide({bridge.CORE_ENV: "1"}, _view(True))
    assert core.name == bridge.CORE_DECIDER and isinstance(core.decide, bridge.CoreController)
    first, log = _inquire(core)
    assert [s.decider for s in _starts(first)] == [bridge.CORE_DECIDER]
    step = first.events[0].step
    assert codec.step_from_json(codec.step_to_json(step)) == step
    assert "decider" not in codec.step_to_json(_inquire(None)[0].events[0].step)["entry"]  # old logs, byte for byte
    with pytest.raises(ValueError, match="decider"):
        loop.resume_inquiry(log.events(), ports((CAMERA, CANCELLED), [], log=log), targets=TARGETS)
    resumed = loop.resume_inquiry(log.events(), ports((CAMERA, CANCELLED), [], log=log), targets=TARGETS, decider=core)
    assert [s.decider for s in _starts(resumed)] == [bridge.CORE_DECIDER]
    assert kernel.fold(log.events()) == resumed.ws

    today, today_log = _inquire(None)
    assert [s.decider for s in _starts(today)] == [None]  # the controller's logs read as they always did
    with pytest.raises(ValueError, match="decider"):
        loop.resume_inquiry(today_log.events(), ports((CAMERA, CANCELLED), [], log=today_log), targets=TARGETS,
                            decider=core)


def test_a_night_decides_by_its_dream_and_resumes_under_the_decider_its_run_names() -> None:
    view = _view(True)
    assert bridge.decider_for(None) is loop.CONTROLLER
    core = bridge.decider_for(view)
    assert core.name == bridge.CORE_DECIDER and core.decide == bridge.CoreController(view)
    assert bridge.resume_decider(None, view) is loop.CONTROLLER  # a controller's run runs on under it
    assert bridge.resume_decider(bridge.CORE_DECIDER, view).decide == bridge.CoreController(view)
    with pytest.raises(ValueError, match="core's view"):
        bridge.resume_decider(bridge.CORE_DECIDER, None)  # never under the controller in its place
    with pytest.raises(ValueError, match="unknown"):
        bridge.resume_decider("someone-else", view)


def test_core_controller_investigates_what_the_core_attends_to_first() -> None:
    w, _ = _held()
    other = "t-budget"
    w.ok(it.NoticeOp("anomaly", "Whether the budget holds", (other,), (CAMERA.id,)))  # older, same priority
    w.ok(it.NoticeOp("gap", "What Thursday's shoot still needs", (SHOOT,), (CANCELLED.id,)))
    targets = (*TARGETS, commitment(other, "the budget", due=noon(8), n=2))
    attends = bridge.CoreView(
        decision=Decision(speak=True, slot=1, p_speak=0.9),
        predictions=Predictions(speak=0.9, none=0.1, slots=(SlotPrediction(
            slot=1, usage=0.9, risk=0.8, hazard=(0.4,), missing=0.6, self_error=0.02, choice=0.9,
            cites=(CANCELLED.id,)),)),
    )
    today = controller.decide(w.ws, it.Budget(), focus=None, targets=targets, as_of=night(5))
    core = bridge.CoreController(attends)(w.ws, it.Budget(), focus=None, targets=targets, as_of=night(5))
    assert today.kind == core.kind == "investigate" and today.priorities == core.priorities
    assert today.question_id == w.ws.questions[0].id  # a tie goes to the older question
    assert core.question_id == w.ws.questions[1].id  # the core's slot cites the newer one's origin
    spent = it.Budget(cycles_used=it.CYCLE_CAP)
    assert bridge.CoreController(attends)(w.ws, spent, focus=None, targets=targets, as_of=night(5)).kind == "stop"


class _Inner:
    """A chooser whose wants are fixed: a held note on the shoot, a study of another thread."""

    def derive(self, snapshot: Any) -> tuple[Want, ...]:
        return (Want("w-note", "share", SHOOT, "hold_note", "share", 0.9),
                Want("w-study", "understand", "t-other", "revisit_thread", "understand", 0.7))

    def choose(self, opportunity: Any, wants: Any, ctx: Any) -> Any:
        return wants


def _view(speak: bool) -> bridge.CoreView:
    return bridge.CoreView(
        decision=Decision(speak=speak, slot=0 if speak else None, p_speak=0.9 if speak else 0.1),
        predictions=Predictions(speak=0.9 if speak else 0.1, none=0.1, slots=(SlotPrediction(
            slot=0, usage=0.8, risk=0.9, hazard=(0.5,), missing=0.1, self_error=0.01, choice=0.9,
            cites=(CAMERA.id,)),)),
        targets_of={CAMERA.id: SHOOT},
    )


def test_core_chooser_lets_a_note_surface_only_when_the_core_speaks() -> None:
    on = {bridge.CORE_ENV: "1"}
    snapshot = WantSnapshot(as_of=night(5), last_message_at=None, targets=TARGETS, results=())
    speaks = bridge.night_chooser(on, _Inner(), lambda snapshot: _view(True)).derive(snapshot)
    waits = bridge.night_chooser(on, _Inner(), lambda snapshot: _view(False)).derive(snapshot)
    assert [w.weight for w in speaks] == [1.0, 1.0]
    assert [w.weight for w in waits] == [0.0, 1.0]  # the study is the inner chooser's, unchanged


def test_the_bridge_imports_without_torch() -> None:
    code = ("import sys; import anjo.sleep.mind.core.bridge, anjo.sleep.mind.core.data, "
            "anjo.sleep.mind.core.findings, anjo.sleep.mind.core.toy, anjo.sleep.mind.core.state; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
