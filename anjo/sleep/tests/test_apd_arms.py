# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The APD arms (Task 17-eval): every arm runs each night of a life on the same thinker, brief,
tools and budget, reports through one capped registry, and sees nothing after its night."""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
from collections.abc import Callable
from datetime import timedelta
from typing import get_args

import pytest

from anjo.sleep.mind.apd import arms
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.ports import InquiryPorts
from anjo.sleep.tests import apd_fakes as fx

AB: tuple[apd.ArmName, ...] = ("A", "B")
MIND: tuple[apd.ArmName, ...] = ("M", "M-persist", "M-kernel", "M-control")
_STEP_LINE = re.compile(r"^- ", re.MULTILINE)


def _steps_tonight(prompt: str) -> int:
    section = prompt.split(arms.B_STEPS_HEADING, 1)[1].split("\n## ", 1)[0]
    return len(_STEP_LINE.findall(section))


def b_reply(prompt: str) -> str:
    """Search, ask the world, report, then stop."""
    script = (
        {"notebook": f"notes after {_steps_tonight(prompt)} steps", "action": {"tool": "evidence_search", "query": "plan scene"}},
        {"action": {"tool": "world_query", "text": "How close can an 85 mm lens focus?"}},
        {"action": {"tool": "report_finding", "problem": "Scene 4's plan misses a step", "conclusion": "It needs a permit",
                    "p": 0.4, "cites": ["e-01-1"], "kind": "suspicion"}},
        {"action": {"tool": "stop"}},
    )
    return json.dumps(script[min(_steps_tonight(prompt), len(script) - 1)])


def a_reply(prompt: str) -> str:
    return "One gap stands out.\n" + json.dumps([
        {"problem": "Scene 4's plan misses a step", "conclusion": "It needs a permit", "p": 0.4, "cites": ["e-01-1"],
         "kind": "suspicion"},
    ])


def shared_reply(prompt: str) -> str:
    if "NOTICE PASS" in prompt or "\nCYCLE " in prompt:
        return fx.mind_reply(prompt)
    if arms.A_EVIDENCE_HEADING in prompt:
        return a_reply(prompt)
    return b_reply(prompt)


def _run(
    arm: apd.ArmName, life: apd.Life, reply: Callable[[str], str] = shared_reply, *, tokens: int = 100,
) -> tuple[apd.ArmRun, fx.ScriptThinker, fx.World, fx.FakeCore]:
    thinker = fx.ScriptThinker(reply, tokens=tokens)
    world, fake = fx.World(), fx.FakeCore()
    return arms.run_arm(arm, life, fx.models(thinker, world), core=fake.core()), thinker, world, fake


# ── The spec's tests ──────────────────────────────────────────────────────────


def test_arms_share_budget_brief_and_tools() -> None:
    life = fx.life(3, split="held_out")
    brief = arms.load_brief()
    for tool in arms.APD_TOOLS:
        assert f"`{tool}`" in brief  # the brief names every shared tool
    assert f"at most {apd.FINDING_CAP} distinct findings" in brief
    assert arms.APD_TOOLS == {"evidence_search", "world_query", "report_finding"}
    for arm in (*AB, *MIND):
        run, thinker, world, fake = _run(arm, life)
        assert thinker.calls, arm
        assert all(prompt.startswith(brief + "\n\n") for prompt, _ in thinker.calls), arm
        assert all(0 < cap <= it.TOKEN_CAP for _, cap in thinker.calls), arm
        assert len(run.tokens) == len(life.days) and all(0 < n <= it.TOKEN_CAP for n in run.tokens), arm
        if arm in MIND:
            notices = [c for c in fake.calls if c["call"] == "notice"]
            assert [c["budget"] for c in notices] == [it.Budget()] * len(life.days)  # fresh each night
            for call in fake.calls:
                ports = call["ports"]
                assert isinstance(ports, InquiryPorts)
                assert ports.world is world and ports.research is None
                assert isinstance(ports.search, arms.LifeSearch) and ports.search.evidence == life.evidence
    b_run, b_thinker, b_world, _ = _run("B", life)
    assert b_world.calls and all(cap <= it.TOKEN_CAP for _, cap in b_world.calls)  # the same world_query
    for tool in (*arms.APD_TOOLS, "stop"):
        assert f'"tool": "{tool}"' in b_thinker.calls[0][0]


def test_findings_capped_updates_in_place_free() -> None:
    def ten_new_then_update(prompt: str) -> str:
        if "## Your findings so far\nnone" in prompt:
            return json.dumps([
                {"problem": f"problem {n}", "conclusion": "c", "p": 0.3, "cites": ["e-01-1"], "kind": "suspicion"}
                for n in range(10)
            ])
        return json.dumps([
            {"finding_id": "f1", "problem": "problem 0", "conclusion": "settled", "p": 0.9, "cites": ["e-02-1"],
             "kind": "resolution"},
            {"problem": "one more", "conclusion": "c", "p": 0.3, "cites": ["e-01-1"], "kind": "suspicion"},
        ])

    run, *_ = _run("A", fx.life(2, split="held_out"), ten_new_then_update)
    ids = {r.finding_id for r in run.reports}
    assert len(ids) == apd.FINDING_CAP and len(run.reports) == apd.FINDING_CAP + 1  # the update is free
    assert run.reports[-1].finding_id == "f1" and run.reports[-1].kind == "resolution"
    assert "one more" not in {r.problem for r in run.reports}  # past the cap: recorded nowhere

    # MIND reports each workspace finding under its own id, and M-persist's wipes don't reset the cap.
    persist, *_ = _run("M-persist", fx.life(10, split="held_out"))
    assert len({r.finding_id for r in persist.reports}) == apd.FINDING_CAP
    assert all(r.finding_id.startswith("f") for r in persist.reports)


# ── report_finding ────────────────────────────────────────────────────────────


def test_report_finding_refuses_unknown_ids_and_drops_cites_the_arm_could_not_see() -> None:
    call = arms.FindingCall("p", "c", 0.5, ("e-01-1", "e-09-1"), "suspicion")
    filed = arms.report_finding(apd.FindingRegistry(), call, day=fx.DAY0, citable=frozenset({"e-01-1"}))
    assert filed.report is not None and filed.report.cites == ("e-01-1",) and filed.report.finding_id == "f1"
    stale = arms.report_finding(filed.registry, dataclasses.replace(call, finding_id="f7"), day=fx.DAY0,
                                citable=frozenset())
    assert stale.report is None and stale.registry == filed.registry and "f7" in stale.note
    bad = arms.report_finding(filed.registry, dataclasses.replace(call, p=1.5), day=fx.DAY0, citable=frozenset())
    assert bad.report is None and bad.registry == filed.registry
    own = arms.report_finding(apd.FindingRegistry(), call, day=fx.DAY0, citable=frozenset(), new_id="f42")
    assert own.report is not None and own.report.finding_id == "f42"


def test_finding_calls_parse_strictly() -> None:
    good = {"problem": "p", "conclusion": "c", "p": 1, "cites": ["e1"], "kind": "retraction", "finding_id": "f2"}
    assert arms.finding_call(good) == arms.FindingCall("p", "c", 1.0, ("e1",), "retraction", "f2")
    broken: list[object] = [{**good, "p": True}, {**good, "kind": "hunch"}, {**good, "cites": "e1"}, {**good, "problem": 3}, []]
    assert all(arms.finding_call(case) is None for case in broken)


# ── The arms ──────────────────────────────────────────────────────────────────


def test_mind_arms_differ_only_in_rules_control_and_wipe() -> None:
    life = fx.life(2, split="held_out")
    expected: dict[apd.ArmName, tuple[it.KernelRules, str, bool]] = {
        "M": (it.FULL_RULES, "priority", False), "M-persist": (it.FULL_RULES, "priority", True),
        "M-kernel": (it.NO_KERNEL, "priority", False), "M-control": (it.FULL_RULES, "newest", False),
        "M+replay": (it.FULL_RULES, "priority", False), "M+shuffled": (it.FULL_RULES, "priority", False),
    }
    for arm, (rules, control, wipe) in expected.items():
        _, _, _, fake = _run(arm, life)
        notices = [c for c in fake.calls if c["call"] == "notice"]
        inquiries = [c for c in fake.calls if c["call"] == "inquiry"]
        assert [c["wipe"] for c in notices] == [wipe] * 2, arm  # a wipe on each night's first call only
        assert all(c["wipe"] is False and c["focus"] is None for c in inquiries), arm
        assert all(c["rules"] == rules and c["control"] == control for c in inquiries), arm


def test_each_night_gets_its_own_targets_evidence_and_clock() -> None:
    base = fx.life(3, split="held_out")
    due = dataclasses.replace(fx.target(base.days[1], "t-thursday"), kind="commitment")
    life = dataclasses.replace(base, targets={day: (fx.target(fx.DAY0), *((due,) if day >= base.days[1] else ()))
                                              for day in base.days})
    _, _, _, fake = _run("M", life)
    for call in fake.calls:
        night = call["night"]
        assert call["targets"] == life.targets[night]  # type: ignore[index]
        assert call["as_of"] == arms.night_as_of(night)  # type: ignore[arg-type]
        if call["call"] == "notice":
            assert call["new"] == tuple(e.id for e in life.evidence if e.day == night)
    _, a_thinker, _, _ = _run("A", life)
    first_night = a_thinker.calls[0][0]
    assert "t-thursday" not in first_night and "e-02-1" not in first_night and "e-01-2" in first_night
    assert "t-thursday" in a_thinker.calls[1][0] and "e-02-1" in a_thinker.calls[1][0]


def test_mind_arm_keeps_a_foldable_log_with_its_turns() -> None:
    run, thinker, _, fake = _run("M", fx.life(2))
    assert fx.fold(run.events) == fx.fold(run.events) and fx.digest(fx.fold(run.events)) == run.events[-1].after
    assert [e.step.seq for e in run.events] == list(range(len(run.events)))
    assert len(run.turns) == len(thinker.calls)  # every thinker call is a turn, indexed to its step
    assert all(isinstance(run.events[turn.seq].step.entry, (it.Retrieved, *it.OP_CLASSES)) for turn in run.turns)
    assert run.reports and set(run.contexts) == set(fx.life(2).days)
    assert run.reports[0].problem.startswith("The plan in") and run.reports[0].kind == "resolution"
    assert run.reports[0].finding_id in run.contexts[fx.DAY0]


def test_a_reflects_once_a_night_with_everything_said_so_far() -> None:
    life = fx.life(3, split="held_out")
    run, thinker, world, _ = _run("A", life)
    assert len(thinker.calls) == 3 and not world.calls
    assert all(cap == it.TOKEN_CAP for _, cap in thinker.calls)
    last = thinker.calls[-1][0]
    assert all(e.id in last for e in life.evidence)
    assert run.contexts[life.days[0]].startswith("One gap stands out")  # A's reflections
    assert [r.finding_id for r in run.reports] == ["f1", "f2", "f3"]  # no finding_id: a new finding each night


def test_b_keeps_its_notebook_across_nights_and_charges_world_queries() -> None:
    life = fx.life(2)
    run, thinker, world, _ = _run("B", life, tokens=100)
    assert run.contexts[life.days[0]] == "notes after 0 steps"
    second_night = [p for p, _ in thinker.calls if f"Tonight: {life.days[1]}" in p][0]
    assert "notes after 0 steps" in second_night  # the notebook persists
    assert run.tokens == (4 * 100 + 50,) * 2  # four thinker calls and one world_query a night
    assert "e-02-1" not in "".join(p for p, _ in thinker.calls if f"Tonight: {life.days[0]}" in p)
    assert [r.kind for r in run.reports] == ["suspicion", "suspicion"]


def test_b_gets_one_opening_call_then_the_cycle_cap() -> None:
    def forever(prompt: str) -> str:
        return json.dumps({"action": {"tool": "evidence_search", "query": "kitchen"}})

    run, thinker, _, _ = _run("B", fx.life(2), forever, tokens=100)
    assert len(thinker.calls) == 2 * (it.CYCLE_CAP + 1)  # like MIND's notice pass, then 12 cycles
    run, thinker, _, _ = _run("B", fx.life(2), forever, tokens=8000)
    assert run.tokens == (it.TOKEN_CAP, it.TOKEN_CAP) and len(thinker.calls) == 6  # the token cap is hard
    assert [cap for _, cap in thinker.calls[:3]] == [20_000, 12_000, 4_000]


def test_b_malformed_reply_costs_its_cycle() -> None:
    run, thinker, _, _ = _run("B", fx.life(1), lambda prompt: "no json here", tokens=10)
    assert len(thinker.calls) == it.CYCLE_CAP + 1 and run.tokens == (10 * (it.CYCLE_CAP + 1),)
    assert "no action" in thinker.calls[-1][0]


def test_development_lives_run_only_b_and_m() -> None:
    life = fx.life(1)
    for arm in get_args(apd.ArmName):
        if arm in apd.DEVELOPMENT_ARMS:
            assert _run(arm, life)[0].arm == arm
            continue
        with pytest.raises(ValueError, match="development lives run"):
            _run(arm, life)


def test_mind_core_matches_the_frozen_signatures() -> None:
    """Skipped until 17-core's kernel and loop land; then the lazily imported functions must take
    exactly the frozen parameters the harness calls them with."""
    pytest.importorskip("anjo.sleep.mind.inquiry.loop")
    core = arms.mind_core()
    for real, frozen in ((core.notice, arms.NoticeFn), (core.run_inquiry, arms.InquiryFn), (core.apply, arms.ApplyFn)):
        wanted = list(inspect.signature(frozen.__call__).parameters.values())[1:]
        got = list(inspect.signature(real).parameters.values())
        assert [(p.name, p.kind, p.default) for p in got] == [(p.name, p.kind, p.default) for p in wanted], real
    assert callable(core.fold) and callable(core.digest)


# ── The life's own tools ──────────────────────────────────────────────────────


def test_life_search_is_time_masked_and_skips_held_out_items() -> None:
    life = fx.life(3)
    hidden = fx.evidence(fx.DAY0, 9, "plan held out", held_out=True)
    search = arms.LifeSearch((*life.evidence, hidden))
    night1 = arms.night_as_of(fx.DAY0)
    assert {e.id for e in search.search("plan", night1, 10)} == {"e-01-1"}
    assert [e.id for e in search.search("plan", arms.night_as_of(life.days[2]), 2)] == ["e-03-1", "e-02-1"]
    assert search.get(["e-02-1", "e-01-2", "e-01-9", "nope"], night1) == (life.evidence[1],)
    assert night1.tzinfo is not None and night1.date() == fx.DAY0 and night1 + timedelta(microseconds=1) > night1


def test_a_life_stamped_outside_its_days_is_refused() -> None:
    life = fx.life(2)
    late = dataclasses.replace(life.evidence[0], created_at=life.evidence[0].created_at + timedelta(hours=6))
    with pytest.raises(ValueError, match="within its own day"):
        _run("M", dataclasses.replace(life, evidence=(late, *life.evidence[1:])))
    early = dataclasses.replace(life.evidence[2], created_at=life.evidence[0].created_at)
    with pytest.raises(ValueError, match="within its own day"):
        _run("B", dataclasses.replace(life, evidence=(*life.evidence[:2], early, *life.evidence[3:])))


def test_memory_log_refuses_a_gap_in_seq() -> None:
    run, *_ = _run("M", fx.life(1))
    log = arms.MemoryLog(run.events[:3])
    assert log.events() == run.events[:3]
    with pytest.raises(ValueError, match="next"):
        log.append(run.events[4])
