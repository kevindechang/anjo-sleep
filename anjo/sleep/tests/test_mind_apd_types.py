# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The APD pilot's contract (Task 17): lives, planted problems, arm runs and scores are frozen,
carry the pre-registered numbers and refuse what the protocol rules out."""
from __future__ import annotations

import ast
import dataclasses
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_args

import pytest

from anjo.sleep import types as t
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.inquiry import types as it

DAY0 = date(2026, 10, 1)
NOW = datetime(2026, 10, 1, 21, 0, tzinfo=timezone.utc)
DAYS = tuple(DAY0 + timedelta(days=n) for n in range(apd.LIFE_DAYS))


def _evidence(evidence_id: str = "e-d1-1", **changes: Any) -> t.Evidence:
    base = t.Evidence(
        id=evidence_id, person_id="life-1", day=DAY0, kind="lesson", text="I want the 85 mm for scene 4.",
        source="session:d1", held_out=False, created_at=NOW, model="spark-open-fake", ancestry=frozenset(),
    )
    return dataclasses.replace(base, **changes)


def _target(**changes: Any) -> t.Target:
    base = t.Target(
        id="t-scene4", person_id="life-1", kind="thread", label="Scene 4's lens", check_after=None,
        status="open", revision=1, acted_revision=0, confidence=None, last_mentioned=DAY0,
    )
    return dataclasses.replace(base, **changes)


def _life(**changes: Any) -> apd.Life:
    base = apd.Life(
        id="life-1", field="indie film", seed=7, split="development", days=DAYS,
        sessions=(apd.Session(DAY0, "We talked about the kitchen set.", "spark-open-fake"),),
        evidence=(_evidence(),), targets={day: (_target(),) for day in DAYS},
    )
    return dataclasses.replace(base, **changes)


def _item(**changes: Any) -> apd.PlantedItem:
    base = apd.PlantedItem(
        id="i1", family="contradiction", template="lens-vs-room", topic="scene 4's shoot", surface_day=DAYS[7],
        origin_ids=("e-d2-1", "e-d5-3"), disambiguating_ids=("e-d4-2",),
        resolution="The 85 mm can't fit the kitchen; the scene moves to the courtyard.",
    )
    return dataclasses.replace(base, **changes)


def _report(finding_id: str = "f1", seq: int = 0, **changes: Any) -> apd.FindingReport:
    base = apd.FindingReport(
        finding_id=finding_id, day=DAYS[5], seq=seq, problem="The lens doesn't fit the room",
        conclusion="Scene 4 can't be shot on the 85 mm in the kitchen", p=0.5, cites=("e-d2-1",),
        kind="suspicion",
    )
    return dataclasses.replace(base, **changes)


def _run(**changes: Any) -> apd.ArmRun:
    base = apd.ArmRun(arm="B", life_id="life-1", reports=(_report(),), tokens=(900, 1200), contexts={DAYS[0]: "notes"})
    return dataclasses.replace(base, **changes)


def _event() -> it.WorkspaceEvent:
    step = it.Step(seq=0, at=NOW, entry=it.ParkOp("q1"), code="unknown_ref", cycle=0, focus="q1", tokens=30)
    return it.WorkspaceEvent(step=step, before="0" * 64, after="0" * 64)


def _turn(seq: int = 0) -> it.Turn:
    return it.Turn(seq=seq, prompt="rendered", reply=it.ModelReply(text='{"op": "PARK"}', tokens=30, model="fake"))


class _Port:
    def propose(self, rendered: str, max_tokens: int) -> it.ModelReply:
        return it.ModelReply(text=rendered, tokens=1, model="fake")

    def ask(self, text: str, max_tokens: int) -> it.ModelReply:
        return it.ModelReply(text=text, tokens=1, model="fake")

    def resolve(self, claim: str, evidence: Any) -> it.ResolverReply:
        return it.ResolverReply(verdict="undetermined", tokens=1)


def _samples() -> dict[str, object]:
    port = _Port()
    return {
        "Session": _life().sessions[0], "PlantedItem": _item(), "ProblemFile": apd.ProblemFile("life-1", (_item(),)),
        "Life": _life(), "FindingReport": _report(), "ArmModels": apd.ArmModels(port, port, port),
        "ArmRun": _run(), "ItemScore": apd.ItemScore("i1", "gap", "f1", apd.SUSPECTED),
        "UnmatchedScore": apd.UnmatchedScore("f2", apd.WRONG),
        "LifeScore": apd.LifeScore("life-1", "M", "audited", (), ()),
        "SignTest": apd.SignTest(wins=9, losses=3, ties=4, p=0.073), "FindingRegistry": apd.FindingRegistry((_report(),)),
        "GateVerdict": apd.GateVerdict("script_audit", None, passed=True),
    }


# ── The pre-registered names and numbers ──────────────────────────────────────


def test_families_arms_and_gates_match_the_protocol() -> None:
    assert apd.FAMILIES == {"contradiction", "gap", "opportunity", "reversal", "decoy"}
    assert apd.PROBLEM_FAMILIES == apd.FAMILIES - {"reversal", "decoy"}
    assert apd.ARMS == {"A", "B", "M", "M-persist", "M-kernel", "M-control", "M+replay", "M+shuffled"}
    assert apd.DEVELOPMENT_ARMS == {"B", "M"} and apd.HELD_OUT_ONLY_ARMS <= apd.ARMS
    assert apd.WORKSPACE_ARMS == apd.ARMS - {"A", "B"}
    assert set(get_args(apd.Split)) == {"development", "held_out"}
    assert set(get_args(apd.JudgeView)) == {"audited", "gemini"}
    assert set(get_args(apd.ApdGate)) == {"leak", "discoverability", "script_audit"}
    assert set(get_args(apd.EditKind)) == {"swap_lead", "delete_prediction", "inject_hypothesis"}


def test_protocol_numbers_are_the_pre_registered_ones() -> None:
    assert (apd.LIFE_DAYS, apd.MAX_PREVALENCE, apd.HALF_GAP_DAYS) == (10, 6, 3)
    assert (apd.DEVELOPMENT_LIVES, apd.HELD_OUT_LIVES, apd.HELD_OUT_LIVES_CUT, apd.EMPTY_HELD_OUT_LIVES) == (8, 16, 12, 2)
    assert (apd.WIN_P, apd.INTERVENTION_SNAPSHOTS, apd.KILL_RESUME_LIVES) == (0.05, 30, 3)
    assert apd.FINDING_CAP == 8  # per arm and life, the registry's; the inquiry kernel has no cap
    assert apd.Edit is it.Edit and apd.EditKind is it.EditKind  # an edit is a logged inquiry step


def test_scoring_points_follow_each_family() -> None:
    assert (apd.RESOLVED, apd.SUSPECTED, apd.WRONG, apd.NOTHING) == (1.0, 0.25, -0.5, 0.0)
    assert set(apd.ITEM_POINTS) == apd.FAMILIES
    for family in apd.PROBLEM_FAMILIES:
        assert apd.ITEM_POINTS[family] == {1.0, 0.25, -0.5, 0.0}
    assert apd.ITEM_POINTS["reversal"] == {1.0, -0.5, 0.0}  # dismissed, left standing, or never matched
    assert apd.ITEM_POINTS["decoy"] == {-0.5, 0.0}  # a decoy can only cost
    assert apd.UNMATCHED_POINTS == {-0.5, 0.0}
    with pytest.raises(TypeError):
        apd.ITEM_POINTS["decoy"] = frozenset({1.0})  # type: ignore[index]


# ── Lives and planted problems ────────────────────────────────────────────────


def test_planted_item_keeps_origin_and_disambiguation_apart() -> None:
    with pytest.raises(ValueError, match="two halves"):
        _item(origin_ids=("e-d2-1",))
    assert _item(family="gap", origin_ids=("e-d2-1",)).origin_ids == ("e-d2-1",)
    with pytest.raises(ValueError, match="away from the origin"):
        _item(disambiguating_ids=("e-d5-3",))
    with pytest.raises(ValueError, match="origin_ids"):
        _item(family="gap", origin_ids=())
    with pytest.raises(ValueError, match="disambiguating_ids"):
        _item(disambiguating_ids=())
    with pytest.raises(TypeError, match="origin_ids"):
        _item(origin_ids="e-d2-1")
    with pytest.raises(ValueError, match="family"):
        _item(family="riddle")


def test_planted_item_carries_a_neutral_topic_for_the_discoverability_gate() -> None:
    assert _item().topic == "scene 4's shoot"
    with pytest.raises(ValueError, match="topic"):
        _item(topic="")
    with pytest.raises(ValueError, match="neutral"):
        _item(topic="lens-vs-room")
    with pytest.raises(ValueError, match="neutral"):
        _item(topic=_item().resolution)


def test_problem_file_holds_up_to_six_distinct_items() -> None:
    items = tuple(_item(id=f"i{n}") for n in range(apd.MAX_PREVALENCE))
    assert len(apd.ProblemFile("life-1", items).items) == 6
    assert apd.ProblemFile("life-1", ()).items == ()  # an empty held-out life
    with pytest.raises(ValueError, match="at most 6"):
        apd.ProblemFile("life-1", (*items, _item(id="i9")))
    with pytest.raises(ValueError, match="twice"):
        apd.ProblemFile("life-1", (_item(), _item()))
    with pytest.raises(TypeError, match="items"):
        apd.ProblemFile("life-1", ("i1",))  # type: ignore[arg-type]


def test_life_is_consecutive_days_with_its_own_model_written_evidence() -> None:
    assert _life().days == DAYS
    with pytest.raises(ValueError, match="consecutive"):
        _life(days=(DAYS[0], DAYS[2]))
    with pytest.raises(ValueError, match="at least one day"):
        _life(days=(), sessions=(), evidence=(), targets={})
    with pytest.raises(ValueError, match="doesn't record the model"):
        _life(evidence=(_evidence(model=None),))
    with pytest.raises(ValueError, match="isn't this life's"):
        _life(evidence=(_evidence(person_id="life-2"),))
    with pytest.raises(ValueError, match="isn't this life's"):
        _life(evidence=(_evidence(day=DAYS[-1] + timedelta(days=1)),))
    with pytest.raises(ValueError, match="this life's"):
        _life(targets={day: (_target(person_id="life-2"),) for day in DAYS})
    with pytest.raises(ValueError, match="session"):
        _life(sessions=(apd.Session(DAY0 - timedelta(days=1), "before the life", "spark-open-fake"),))
    with pytest.raises(ValueError, match="split"):
        _life(split="test")


# ── What an arm reports ───────────────────────────────────────────────────────


def test_targets_are_dated_so_a_night_never_sees_a_later_plan() -> None:
    """Stake reads targets[night]; a commitment first mentioned on day 7 is absent before it."""
    due = _target(id="t-thursday", kind="commitment", check_after=NOW + timedelta(days=8), last_mentioned=DAYS[6])
    targets = {day: (_target(), *((due,) if day >= DAYS[6] else ())) for day in DAYS}
    life = _life(targets=targets)
    assert life.targets[DAYS[2]] == (_target(),) and life.targets[DAYS[6]] == (_target(), due)
    with pytest.raises(ValueError, match="mentioned by that day"):
        _life(targets={day: (_target(), due) for day in DAYS})
    with pytest.raises(ValueError, match="every day"):
        _life(targets={DAYS[0]: (_target(),)})
    with pytest.raises(TypeError, match="Target"):
        _life(targets={day: ("t-scene4",) for day in DAYS})
    with pytest.raises(TypeError):
        life.targets[DAYS[0]] = ()  # type: ignore[index]


def test_finding_report_checks_its_fields() -> None:
    assert _report(kind="retraction", cites=()).cites == ()
    with pytest.raises(ValueError, match="p"):
        _report(p=1.5)
    with pytest.raises(ValueError, match="kind"):
        _report(kind="hunch")
    with pytest.raises(ValueError, match="seq"):
        _report(seq=-1)


def test_registry_caps_distinct_findings_per_life_for_every_arm() -> None:
    """The cap lives in report_finding's registry, across nights, so M-persist's wipes can't reset it."""
    registry = apd.FindingRegistry()
    for n in range(apd.FINDING_CAP):
        assert registry.accepts(f"f{n}")
        registry = registry.add(_report(f"f{n}", n))
    assert not registry.accepts("f8") and registry.accepts("f0")
    updated = registry.add(_report("f0", 8, kind="resolution", p=0.9))
    assert len(updated.reports) == 9 and updated.finding_ids == registry.finding_ids
    with pytest.raises(ValueError, match="at most 8 distinct"):
        registry.add(_report("f8", 9))
    with pytest.raises(ValueError, match="seq order"):
        registry.add(_report("f0", 3))
    assert len(registry.reports) == 8  # add returns a new registry


def test_arm_run_caps_distinct_findings_but_updates_in_place_are_free() -> None:
    eight = tuple(_report(f"f{n}", n) for n in range(apd.FINDING_CAP))
    updates = tuple(_report("f0", 8 + n, kind="resolution", p=0.9) for n in range(5))
    assert len(_run(reports=eight + updates).reports) == 13
    with pytest.raises(ValueError, match="at most 8 distinct"):
        _run(reports=(*eight, _report("f8", 8)))
    with pytest.raises(ValueError, match="seq order"):
        _run(reports=(_report("f1", 2), _report("f2", 1)))
    with pytest.raises(ValueError, match="seq order"):
        _run(reports=(_report("f1", 1), _report("f2", 1)))


def test_only_mind_arms_carry_a_workspace_log_and_its_turns() -> None:
    run = _run(arm="M", events=(_event(),), turns=(_turn(),))
    assert run.events == (_event(),) and run.turns == (_turn(),)
    with pytest.raises(ValueError, match="keeps no workspace"):
        _run(arm="B", events=(_event(),))
    with pytest.raises(ValueError, match="keeps no workspace"):
        _run(arm="A", turns=(_turn(),))
    with pytest.raises(ValueError, match="logged step"):
        _run(arm="M-persist", events=(_event(),), turns=(_turn(5),))
    with pytest.raises(ValueError, match="arm"):
        _run(arm="M-persistent")
    with pytest.raises(ValueError, match="tokens"):
        _run(tokens=(-1,))


def test_arm_run_copies_its_contexts_read_only() -> None:
    contexts = {DAYS[0]: "notes"}
    run = _run(contexts=contexts)
    contexts[DAYS[1]] = "more"
    assert dict(run.contexts) == {DAYS[0]: "notes"}
    with pytest.raises(TypeError):
        run.contexts[DAYS[1]] = "more"  # type: ignore[index]


# ── Scores ────────────────────────────────────────────────────────────────────


def test_item_scores_stay_within_their_family() -> None:
    assert apd.ItemScore("i1", "reversal", "f1", apd.RESOLVED).points == 1.0
    assert apd.ItemScore("i1", "decoy", None, apd.NOTHING).canonical is None
    for family, points in (("decoy", apd.RESOLVED), ("reversal", apd.SUSPECTED), ("gap", 0.5)):
        with pytest.raises(ValueError, match="scores one of"):
            apd.ItemScore("i1", family, "f1", points)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nothing matched scores 0"):
        apd.ItemScore("i1", "gap", None, apd.WRONG)
    with pytest.raises(ValueError, match="scores one of"):
        apd.UnmatchedScore("f9", apd.RESOLVED)


def test_life_score_totals_items_and_unmatched_findings() -> None:
    score = apd.LifeScore(
        "life-1", "M", "gemini",
        items=(apd.ItemScore("i1", "contradiction", "f1", apd.RESOLVED), apd.ItemScore("i2", "gap", "f2", apd.SUSPECTED),
               apd.ItemScore("i3", "decoy", "f3", apd.WRONG)),
        unmatched=(apd.UnmatchedScore("f4", apd.WRONG), apd.UnmatchedScore("f5", apd.NOTHING)),
    )
    assert score.total == pytest.approx(0.25)
    assert apd.LifeScore("life-1", "B", "audited", (), ()).total == 0.0
    with pytest.raises(ValueError, match="scored once"):
        apd.LifeScore("life-1", "M", "audited", (score.items[0], score.items[0]), ())
    with pytest.raises(ValueError, match="view"):
        apd.LifeScore("life-1", "M", "claude", (), ())  # type: ignore[arg-type]


def test_sign_test_reports_lives_and_the_estimate() -> None:
    result = apd.SignTest(wins=9, losses=3, ties=4, p=0.073)
    assert result.lives == 16 and result.estimate == 0.75
    assert apd.SignTest(wins=0, losses=0, ties=16, p=1.0).estimate is None
    with pytest.raises(ValueError, match="wins"):
        apd.SignTest(wins=-1, losses=0, ties=0, p=1.0)
    with pytest.raises(ValueError, match="p"):
        apd.SignTest(wins=1, losses=0, ties=0, p=1.5)


# ── Gates ─────────────────────────────────────────────────────────────────────


def test_script_audit_judges_the_life_and_other_gates_an_item() -> None:
    assert apd.GateVerdict("leak", "i1", passed=False, reason="names the problem early").item_id == "i1"
    with pytest.raises(ValueError, match="script audit"):
        apd.GateVerdict("script_audit", "i1", passed=True)
    with pytest.raises(ValueError, match="script audit"):
        apd.GateVerdict("discoverability", None, passed=True)


# ── Frozen and pure ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_samples()))
def test_every_apd_type_is_frozen_and_immutable(name: str) -> None:
    value = _samples()[name]
    assert type(value).__name__ == name
    assert dataclasses.is_dataclass(value) and not isinstance(value, type)
    first = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, first, None)
    for f in dataclasses.fields(value):
        assert not isinstance(getattr(value, f.name), (list, set, dict)), f"{name}.{f.name}"


def test_sequences_are_copied_into_tuples() -> None:
    item = _item(origin_ids=["e-d2-1", "e-d5-3"])
    life = _life(
        days=list(DAYS), sessions=list(_life().sessions), evidence=[_evidence()],
        targets={day: [_target()] for day in DAYS},
    )
    assert item.origin_ids == ("e-d2-1", "e-d5-3")
    assert all(isinstance(getattr(life, name), tuple) for name in ("days", "sessions", "evidence"))
    assert all(isinstance(held, tuple) for held in life.targets.values())


def test_apd_contract_imports_only_the_engine_core() -> None:
    allowed = {"__future__", "math", "collections.abc", "dataclasses", "datetime", "types", "typing", "anjo.sleep.types"}
    tree = ast.parse(Path(apd.__file__).read_text())
    for node in ast.walk(tree):
        modules = (
            [alias.name for alias in node.names] if isinstance(node, ast.Import)
            else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        )
        for module in modules:
            assert module in allowed or module.startswith("anjo.sleep.mind.inquiry."), module
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "utcnow", "today", "getenv", "environ"}, node.attr
