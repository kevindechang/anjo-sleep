# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's loop, controller, render and parser (Task 17), on fake ports: the notice pass, the
cycles, the blind resolver, the screen, K5, kill-resume, and a lenient reading of model text."""
from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pytest

from anjo.sleep.mind.inquiry import controller, kernel, loop, parse, render
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog
from anjo.sleep.mind.inquiry.ports import InquiryPorts, PredictionResolver
from anjo.sleep.tests.inquiry_fakes import (
    FakeResearch,
    FakeResolver,
    FakeScreen,
    FakeWorld,
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
E1 = ev("e-d2-1", "For scene 4 they want the compressed long-lens look: an 85 mm medium shot on full frame.", n=2, target=SCENE)
E3 = ev("e-d3-1", "They might shoot scene 4 through the window from the courtyard.", n=3, target=SCENE)
E2 = ev("e-d5-1", "Scene 4 is in their 3 m kitchen, handheld.", n=5, target=SCENE)
E6 = ev("e-d6-1", "The courtyard is locked on Thursday.", n=6, target=SCENE)
HELD = ev("e-d5-9", "HELD-OUT probe text about scene 4", n=5, target=SCENE, held_out=True)
ITEMS = (E1, E3, E2, E6, HELD)
TARGETS = (commitment(SCENE, "Scene 4 shoot on Thursday", due=noon(8), n=2),)
NOTICED = [{"op": "NOTICE", "kind": "contradiction", "subject": "Scene 4's 85 mm look against the 3 m kitchen",
            "target_ids": [SCENE], "origin_evidence": [E1.id, E2.id]}]


class Life:
    """One person's nights, all on one log, as the night glue would run them."""

    def __init__(self, items: Sequence[Evidence] = ITEMS, targets: Sequence[Target] = TARGETS) -> None:
        self.items, self.targets = items, targets
        self.log = MemoryWorkspaceLog()
        self.ws = it.Workspace()

    def ports(self, script: list[Any], **kw: Any) -> InquiryPorts:
        return ports(self.items, script, log=kw.pop("log", self.log), **kw)

    def notice(self, n: int, new: Sequence[Evidence], script: list[Any], **kw: Any) -> it.InquiryRun:
        run = loop.notice(self.ws, new, self.ports(script, **kw), it.Budget(), night=day(n), targets=self.targets,
                          as_of=night(n))
        self.ws = run.ws
        return run

    def inquire(self, n: int, script: list[Any], budget: it.Budget = it.Budget(), **kw: Any) -> it.InquiryRun:
        rules = kw.pop("rules", it.FULL_RULES)
        run = loop.run_inquiry(self.ws, self.ports(script, **kw), budget, night=day(n), targets=self.targets,
                               as_of=night(n), rules=rules)
        self.ws = run.ws
        return run


def _hyp(q: str = "q2", seq: int = 4) -> dict[str, Any]:
    return {"op": "HYPOTHESIZE", "question_id": q, "hypotheses": [
        {"statement": "The plan assumes room for an 85 mm medium shot that the kitchen lacks", "credence": 0.4},
        {"statement": "The 85 mm shot is taken through the window from the courtyard", "credence": 0.35},
        {"statement": "Scene 4 uses a wider lens than 85 mm", "credence": 0.15}], "catch_all": 0.1}


def _ids(q: str = "q2", seq: int = 4) -> list[str]:
    return [f"{q}.h{seq}a", f"{q}.h{seq}b", f"{q}.h{seq}c", f"{q}.else"]


def _predict(q: str = "q2", seq: int = 4) -> dict[str, Any]:
    return {"op": "PREDICT", "question_id": q, "claim": "The courtyard is open for the scene 4 shoot",
            "p_true_given": dict(zip(_ids(q, seq), (0.6, 0.95, 0.6, 0.5), strict=True)), "horizon_days": 3}


def _search(query: str, q: str = "q2") -> dict[str, Any]:
    return {"op": "INVESTIGATE", "question_id": q, "tool": "evidence_search", "query": query}


def _world(q: str = "q2") -> dict[str, Any]:
    return {"op": "INVESTIGATE", "question_id": q, "tool": "world_query",
            "query": "How much distance does an 85 mm medium shot need on full frame?"}


def _noticed_life() -> Life:
    """Nights 2 and 5 noticed: q2, a contradiction on the 85 mm look and the kitchen."""
    life = Life()
    life.notice(5, [E1, E3, E2], [NOTICED])
    return life


# ── The notice pass ───────────────────────────────────────────────────────────


def test_notice_pass_logs_its_start_the_new_items_and_each_notice() -> None:
    life = Life()
    screen = FakeScreen()
    thinker = ScriptedThinker([NOTICED], tokens=300)
    run = life.notice(5, [E1, E3, E2, HELD], [], thinker=thinker, screen=screen)
    kinds = [type(e.step.entry).__name__ for e in run.events]
    assert kinds == ["RunStart", "Retrieved", "NoticeOp"]
    arrived = run.events[1].step
    assert [i.id for i in arrived.entry.items] == [E1.id, E3.id, E2.id] and arrived.tokens == 300  # type: ignore[union-attr]
    assert run.events[2].step.tokens == 5 and screen.seen == [(NOTICED[0]["subject"], "question")]
    assert run.ws.questions[0].id == "q2" and run.budget == it.Budget(tokens_used=305)
    assert [t.seq for t in run.turns] == [1] and all(e.step.cycle is None for e in run.events)
    assert "HELD-OUT" not in thinker.prompts[0] and E1.text in thinker.prompts[0]
    assert life.log.events() == run.events


def test_notice_pass_logs_one_malformed_step_for_a_reply_that_isnt_notices() -> None:
    life = Life()
    run = life.notice(5, [E1, E2], ['{"op": "PARK", "question_id": "q1"}'])
    assert [e.step.code for e in run.events] == [None, None, "malformed"]
    assert life.notice(6, [E6], ["I see nothing worth a question: []"]).ws.questions == ()


def test_notice_pass_shows_earlier_items_on_the_same_target() -> None:
    life = Life()
    life.notice(2, [E1], ["[]"])
    thinker = ScriptedThinker([NOTICED])
    life.notice(5, [E2], [], thinker=thinker)
    earlier = thinker.prompts[0].split("## Earlier evidence you hold")[1].split("## Today's new evidence")
    assert E1.text in earlier[0] and E2.text in earlier[1]
    assert life.ws.questions[0].origin_evidence == (E1.id, E2.id)


# ── The resolver ──────────────────────────────────────────────────────────────


def test_pending_prediction_without_new_evidence_waits() -> None:
    life = _noticed_life()
    resolver = FakeResolver({"locked": "false"})
    life.inquire(5, [_hyp(), _predict(), _world(), _world(), _world()], resolver=resolver)
    prediction = life.ws.predictions[0]
    assert prediction.pending and resolver.calls == []
    assert controller.priority(life.ws, "q2", targets=TARGETS, as_of=night(5)).a == it.A_WAITING

    life.notice(6, [E6], ["[]"])
    assert controller.priority(life.ws, "q2", targets=TARGETS, as_of=night(6)).a == it.A_READY
    life.inquire(6, [_world(), _world()], resolver=resolver)
    assert len(resolver.calls) == 1 and life.ws.predictions[0].outcome == "false"


def test_undetermined_waits_for_new_items_then_the_horizon_censors_it() -> None:
    life = _noticed_life()
    life.inquire(5, [_hyp(), _predict(), _world(), _world(), _world()])
    life.notice(6, [E6], ["[]"])
    resolver = FakeResolver()
    life.inquire(6, [_world(), _world(), _world()], resolver=resolver)
    outcomes = [life.ws.predictions[0].outcome]
    life.inquire(9, [_world()], resolver=resolver)
    outcomes.append(life.ws.predictions[0].outcome)
    assert len(resolver.calls) == 1 and outcomes == ["undetermined", "censored"]


def test_prediction_committed_before_outcome_and_resolver_blind_to_hypotheses() -> None:
    assert list(inspect.signature(PredictionResolver.resolve).parameters) == ["self", "claim", "evidence"]
    life = _noticed_life()
    life.inquire(5, [_hyp(), _predict(), _world(), _world(), _world()])
    prediction = life.ws.predictions[0]
    assert (prediction.committed_at, prediction.outcome) == (night(5), None)
    life.notice(6, [E6], ["[]"])
    resolver = FakeResolver({"locked": "false"})
    life.inquire(6, [_world()], resolver=resolver)

    (claim, shown), = resolver.calls
    assert claim == prediction.claim and [item.id for item in shown] == [E6.id]
    assert all(item.created_at > prediction.committed_at for item in shown)
    statements = [h.statement for h in life.ws.hypotheses]
    assert not any(s in claim or any(s in item.text for item in shown) for s in statements)
    update = life.ws.updates[-1]
    assert (update.source, update.ref, update.evidence_ids) == ("resolved_prediction", prediction.id, (E6.id,))


# ── Tools and the screen ──────────────────────────────────────────────────────


def test_world_query_answer_is_rendered_and_never_moves_credence() -> None:
    life = _noticed_life()
    world = FakeWorld()
    thinker = ScriptedThinker([_hyp(), _world()])
    run = life.inquire(5, [], world=world, thinker=thinker, budget=it.Budget(cycle_cap=3))
    answered = [e.step for e in run.events if isinstance(e.step.entry, it.Answered)]
    assert len(answered) == 1 and answered[0].tokens == 30 and life.ws.updates == ()
    credences = [h.credence for h in life.ws.hypotheses]
    assert credences == pytest.approx([0.4, 0.35, 0.15, 0.1])
    assert world.answer in thinker.prompts[-1]


def test_research_query_carries_no_personal_terms() -> None:
    source = ev("e-web-1", "Telephoto lenses need distance for a medium shot.", n=5, kind="homework",
                model="wonder", ancestry=frozenset({"wonder"}), source="https://example.org/lenses")
    research = FakeResearch((source,))
    screen = FakeScreen(lambda text, purpose: purpose == "research_query" and ("Mira" in text or "Thursday" in text))
    life = _noticed_life()
    personal = {"op": "INVESTIGATE", "question_id": "q2", "tool": "research", "query": "Mira kitchen 85mm Thursday"}
    topical = {"op": "INVESTIGATE", "question_id": "q2", "tool": "research", "query": "85mm lens distance medium shot"}
    life.items = (*ITEMS, source)
    run = life.inquire(5, [_hyp(), personal, topical], research=research, screen=screen,
                       budget=it.Budget(cycle_cap=3))
    codes = [(type(e.step.entry).__name__, e.step.code) for e in run.events]
    assert ("InvestigateOp", "domain_screen") in codes
    assert research.queries == ["85mm lens distance medium shot"] and run.research_calls == 1
    assert ("research_query" in {purpose for _, purpose in screen.seen})
    assert kernel.research_sources(life.ws, "q2") == {source.id}

    without = _noticed_life()
    run = without.inquire(5, [_hyp(), topical], budget=it.Budget(cycle_cap=2))
    assert run.events[-1].step.code == "unavailable"


def test_person_diagnosis_question_refused_by_the_screen_behind_the_denylist() -> None:
    screen = FakeScreen(lambda text, purpose: "Dana" in text)
    life = Life()
    third_party = {**NOTICED[0], "subject": "Whether Dana will still lend the camera for scene 4"}
    diagnosis = {**NOTICED[0], "subject": "Whether they are anxious about scene 4"}
    run = life.notice(5, [E1, E2], [[third_party, diagnosis]], screen=screen)
    assert [e.step.code for e in run.events[2:]] == ["domain_screen", "domain_denylist"]
    assert run.events[2].step.tokens == 5 and len(screen.seen) == 1  # the denylist needs no screen call
    assert run.ws.questions == ()


# ── The controller ────────────────────────────────────────────────────────────


def test_controller_bootstraps_new_question() -> None:
    w = Ws()
    w.start(5)
    w.arrive(E1, E2, E3)
    w.ok(notice_op([E1.id, E2.id]))
    w.ok(hypothesize("q2", 0.5, 0.3))
    w.ok(it.PredictOp("q2", "claim", dict(zip(w.ids("q2"), (0.2, 0.8, 0.5), strict=True)), 3), at=night(5))
    old = controller.priority(w.ws, "q2", targets=(), as_of=night(5))
    assert (old.a, old.ready) == (it.A_WAITING, False)
    w.ok(notice_op([E3.id], kind="gap", subject="Scene 4's courtyard plan has no permission step", targets=("t-x",)))
    new_id = w.ws.questions[-1].id
    fresh = controller.priority(w.ws, new_id, targets=(), as_of=night(5))
    assert (fresh.u, fresh.a, fresh.value) == (1.0, it.A_READY, 1.0)
    decision = controller.decide(w.ws, it.Budget(), focus=None, targets=(), as_of=night(5))
    assert (decision.kind, decision.question_id) == ("investigate", new_id)
    assert [p.question_id for p in decision.priorities] == ["q2", new_id]

    alone = Ws()
    alone.start(5)
    alone.arrive(E3)
    alone.ok(notice_op([E3.id], kind="gap", subject="Scene 4's courtyard plan has no permission step"))
    decision = controller.decide(alone.ws, it.Budget(), focus=None, targets=(), as_of=night(5))
    assert (decision.kind, decision.question_id) == ("investigate", "q2")


def test_ready_to_resolve_question_keeps_priority() -> None:
    w = Ws()
    w.start(5)
    w.arrive(E1, E2, E3)
    w.ok(notice_op([E1.id, E2.id]))
    w.ok(hypothesize("q2", 0.6, 0.2))
    w.ok(it.PredictOp("q2", "claim", dict(zip(w.ids("q2"), (0.2, 0.8, 0.5), strict=True)), 3))
    w.found("q2", E3)
    w.ok(it.AssessOp("q2", E3.id, likelihoods=dict(zip(w.ids("q2"), (4.0, 0.25, 0.25), strict=True))))
    p = controller.priority(w.ws, "q2", targets=(), as_of=night(5))
    assert p.ready and p.u == it.U_FLOOR and p.value == it.STAKE_NONE  # stake x u x a would be 0.06
    due = controller.priority(w.ws, "q2", targets=TARGETS, as_of=night(5))
    assert due.value == it.STAKE_DUE
    decision = controller.decide(w.ws, it.Budget(), focus="q2", targets=(), as_of=night(5))
    assert (decision.kind, decision.question_id) == ("continue", "q2")

    w.ok(it.ConcludeOp("q2", w.ids("q2")[0], "suspicion", "The kitchen can't hold the shot", (E1.id,)))
    extra = ev("e-d6-2", "The courtyard key for scene 4 is with the caretaker.", n=6, target=SCENE)
    w.found("q2", E6, extra)
    for item in (E6, extra):
        w.ok(it.AssessOp("q2", item.id, likelihoods=dict(zip(w.ids("q2"), (0.25, 4.0, 4.0), strict=True))))
    assert kernel.retraction_ready(w.ws, "q2")
    retract = controller.priority(w.ws, "q2", targets=(), as_of=night(5))
    assert retract.ready and retract.value == it.STAKE_NONE


def test_progress_guard_counts_counter_evidence_as_progress() -> None:
    items = [ev(f"e-c{n}", f"courtyard note {n}", n=4) for n in range(4)]
    search = {"op": "INVESTIGATE", "question_id": "q2", "tool": "evidence_search", "query": "courtyard note"}

    def assess(item: Evidence) -> dict[str, Any]:
        return {"op": "ASSESS", "question_id": "q2", "evidence_id": item.id,
                "likelihoods": dict(zip(_ids(), (0.25, 4, 1, 1), strict=True))}

    counter = Life(items=(*ITEMS, *items))
    counter.notice(5, [E1, E2], [NOTICED])
    lead = {**_hyp(), "hypotheses": [{**h, "credence": c} for h, c in zip(_hyp()["hypotheses"], (0.85, 0.05, 0.05), strict=True)],
            "catch_all": 0.05}
    run = counter.inquire(5, [lead, search, assess(items[0]), _world(), _world()], budget=it.Budget(cycle_cap=5))
    leading = [it.Hypothesis(h, "q2", "s", c) for h, c in zip(_ids(), (0.85, 0.05, 0.05, 0.05), strict=True)]
    assert controller.uncertainty(counter.ws.hypotheses_of("q2")) > controller.uncertainty(leading)  # entropy rose
    assert all(d.kind != "park" for d in run.decisions) and run.decisions[-1].reason == "cycle_cap"
    assert not controller.stalled(counter.ws, "q2")

    idle = _noticed_life()
    run = idle.inquire(5, [_hyp(), _world(), _world(), _world(), _world()])
    kinds = [(d.kind, d.reason) for d in run.decisions]
    assert ("park", "stalled") in kinds and kinds[-1] == ("stop", "below_floor")
    assert idle.ws.question("q2").status == "active"  # type: ignore[union-attr]
    assert sum(1 for t in run.turns) == 4  # the fifth reply was never asked for

    tomorrow = idle.inquire(6, [_world()], budget=it.Budget(cycle_cap=1))
    assert tomorrow.decisions[0].kind == "investigate"  # a new night counts afresh


# ── Budget, rendering and the fold ────────────────────────────────────────────


def test_every_capped_call_gets_the_tokens_left_and_a_failed_call_is_charged() -> None:
    class Recording(ScriptedThinker):
        def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
            caps.append(max_tokens)
            return it.ModelReply(text="", tokens=400, model="glimmer-fake")

    caps: list[int] = []
    life = _noticed_life()
    run = life.inquire(5, [], thinker=Recording([]), budget=it.Budget(tokens_used=19_000))
    assert caps == [1000, 600, 200] and run.decisions[-1].reason == "token_cap"
    assert [e.step.code for e in run.events[1:]] == ["malformed"] * 3 and run.budget.tokens_used == 20_200


def test_render_never_includes_conversation() -> None:
    assert list(inspect.signature(render.render).parameters) == ["ws", "focus", "evidence"]
    assert list(inspect.signature(render.render_notice).parameters) == ["ws", "new", "earlier", "targets"]
    conversation = "user: I keep thinking about the CONVERSATION-ONLY detail of my week"
    life = _noticed_life()
    thinker = ScriptedThinker([_hyp(), _search("scene 4 courtyard window")])
    life.inquire(5, [], thinker=thinker, budget=it.Budget(cycle_cap=3))
    assert thinker.prompts and all("CONVERSATION-ONLY" not in p for p in thinker.prompts)
    assert conversation not in render.render(life.ws, "q2", [E1, E2, E3])
    last = thinker.prompts[-1]
    assert "credences are heuristic" in last and E3.text in last and "search hit, not yet assessed" in last
    assert "q2.h4a: 0.40" in last


def test_a_run_appends_every_event_as_it_happens_and_folds_back() -> None:
    life = _noticed_life()
    run = life.inquire(5, [_hyp(), _predict(), _search("courtyard window"), "not json at all", _world()])
    assert life.log.events()[-len(run.events):] == run.events
    assert kernel.fold(life.log.events()) == run.ws
    it.check_turns(run.turns, run.events)
    assert run.events[0].step.entry.call == "inquiry"  # type: ignore[union-attr]
    cycles = [e.step.cycle for e in run.events if e.step.cycle is not None]
    assert cycles == sorted(cycles) and cycles[0] == 0


def test_kill_resume_digest_identical() -> None:
    script = [_hyp(), _predict(), _search("courtyard window"),
              {"op": "ASSESS", "question_id": "q2", "evidence_id": E3.id,
               "likelihoods": dict(zip(_ids(), (0.5, 4, 1, 1), strict=True))},
              _world(), _world(), _world()]
    whole = _noticed_life()
    uninterrupted = whole.inquire(5, list(script))

    killed = _noticed_life()
    kill_at = uninterrupted.events[2].step.seq  # right after the PREDICT is logged
    killing = KillingLog(None)
    for event in killed.log.events():
        killing.append(event)
    killing.kill_after = kill_at
    killed.log = killing
    thinker = ScriptedThinker(list(script))
    with pytest.raises(Kill):
        killed.inquire(5, [], thinker=thinker)

    rebuilt = kernel.fold(killing.events())
    assert kernel.digest(rebuilt) == uninterrupted.events[2].after
    resumed = loop.resume_inquiry(killing.events(), killed.ports([], thinker=thinker), targets=TARGETS)
    assert loop.resume(killing.events()[: kill_at + 1]).budget.cycles_used == 2
    assert [(h.id, round(h.credence, 12)) for h in resumed.ws.hypotheses] == [
        (h.id, round(h.credence, 12)) for h in uninterrupted.ws.hypotheses
    ]
    assert [p.claim for p in resumed.ws.predictions] == [p.claim for p in uninterrupted.ws.predictions]
    assert kernel.fold(killing.events()) == resumed.ws


# ── The parser ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    'Sure. ```json\n{"op": "park", "question_id": "q2"}\n```',
    '{"op": "PARK", "question_id": "q2", "reason": "nothing new"}',
    'I will park it: [{"op": "PARK", "question_id": "q2"}]',
    '{"thought": 1} then {"op": " Park ", "question_id": "q2"}',
])
def test_parser_is_lenient_about_wrapping_case_and_extra_keys(text: str) -> None:
    assert parse.parse_op(text) == it.ParkOp("q2")


def test_parser_coerces_what_is_safe() -> None:
    op = parse.parse_op(
        '{"op": "PREDICT", "question_id": "q2", "claim": "c", "p_true_given": {"a": "0.2", "b": 0.9}, "horizon_days": 3.0}'
    )
    assert op == it.PredictOp("q2", "c", {"a": 0.2, "b": 0.9}, 3)
    conclude = parse.parse_op(
        '{"op": "CONCLUDE", "question_id": "q2", "hypothesis_id": "h", "kind": "suspicion", "statement": "s", "cites": "e-1"}'
    )
    assert isinstance(conclude, it.ConcludeOp) and conclude.cites == ("e-1",)


@pytest.mark.parametrize("text", [
    "", "no json here", '{"op": "FLY"}', '{"op": "PREDICT", "question_id": "q2"}',
    '{"op": "ASSESS", "question_id": "q2", "evidence_id": "e", "likelihoods": {"a": 1}, "credences": {"a": 1}}',
    '{"op": "PREDICT", "question_id": "q2", "claim": "c", "p_true_given": {"a": 0.2}, "horizon_days": 1.5}',
])
def test_parser_rejects_what_isnt_one_operation(text: str) -> None:
    assert parse.parse_op(text) == it.Rejected("malformed")


def test_notice_parser_wants_an_array_of_notices() -> None:
    one = '{"op": "NOTICE", "kind": "gap", "subject": "s", "target_ids": "t", "origin_evidence": ["e"]}'
    assert parse.parse_notices(f"Found: [{one}]") == (it.NoticeOp("gap", "s", ("t",), ("e",)),)
    assert parse.parse_notices(one) == (it.NoticeOp("gap", "s", ("t",), ("e",)),)
    assert parse.parse_notices("[]") == ()
    assert parse.parse_notices('[{"op": "PARK", "question_id": "q2"}]') == it.Rejected("malformed")
    assert parse.parse_notices("nothing") == it.Rejected("malformed")
