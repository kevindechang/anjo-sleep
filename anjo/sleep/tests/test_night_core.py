# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's core in the night, the one system: step 2c dreams over the day, the inquiry decides by
the dream and a killed run resumes under the decider it names, practice rehearses the dream's
picks, tonight's rows carry their evidence's surprise as a loss weight, and the night keeps its unit.
A fake core here; the core's own night is ``test_mind_core_night.py``. No model, no network."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind.core import bridge
from anjo.sleep.mind.core.types import Decision, Predictions
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.night.deps import Dreamt, MindCall, NightSettings
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.tests.inquiry_fakes import Kill
from anjo.sleep.tests.mind_night_fakes import LifeThinker, Researches, mind_night, workspace
from anjo.sleep.tests.night_fakes import NIGHT, World, step, steps, summary, thread
from anjo.sleep.types import Evidence, NightResult

UID = "p1"
STATE = "a" * 16
ON = NightSettings(token_cap=16384, min_new_examples=1, mind_inquiry=True)
WAITS = bridge.CoreView(decision=Decision(speak=False, slot=None, p_speak=0.1),
                        predictions=Predictions(speak=0.1, none=0.9, slots=()))


@dataclass
class FakeCore:
    """Dreams the day (item ``n`` weighs ``n + 2``), picks its first ``picks`` items and keeps each unit."""

    view: bridge.CoreView | None = WAITS
    picks: int = 1
    keep_fails: bool = False
    people: frozenset[str] = frozenset({UID})
    days: list[list[str]] = field(default_factory=list)
    kept: list[str | None] = field(default_factory=list)

    def serves(self, user_id: str) -> bool:
        return user_id in self.people

    def dream(self, call: MindCall, evidence: Sequence[Evidence]) -> Dreamt:
        day = list(evidence)
        self.days.append([item.id for item in day])
        picked = ("not-trainable",) if self.picks < 0 else tuple(item.id for item in day[:self.picks])
        return Dreamt(state_id=STATE, rehearse=picked, weights={item.id: n + 2.0 for n, item in enumerate(day)},
                      view=self.view, receipt={"state": STATE, "picked": list(picked)})

    def keep(self, call: MindCall, dreamt: Dreamt, *, lora: str | None, projector: str | None) -> dict[str, object]:
        if self.keep_fails:
            raise OSError("the disk is full")
        self.kept.append(lora)
        return {"version": len(self.kept), "state": dreamt.state_id, "lora": lora}


@dataclass
class Seeds:
    """The night's practice, keeping the evidence each run was seeded on."""

    world: World
    seeded: list[list[str]] = field(default_factory=list)

    def __call__(self, chat: Any, night: date) -> Any:
        runner, seeded = self.world.practice(chat, night), self.seeded

        class Recording:
            max_spark_calls = runner.max_spark_calls

            def run(self, uid: str, night: date, evidence: Sequence[Evidence], prev: Any, *, at: Any) -> Any:
                seeded.append([item.id for item in evidence])
                return runner.run(uid, night, evidence, prev, at=at)
        return Recording()


@dataclass
class Rows:
    world: World
    data: list[str] = field(default_factory=list)

    def train(self, user_id: str, night: date, prev: Any, data: Any, name: str) -> Any:
        self.data.append(data.to_jsonl())
        return self.world.trainer.train(user_id, night, prev, data, name)


def _world(root: Path) -> World:
    world = World(root)
    world.targets(UID).upsert(thread(UID))
    world.plan(UID, NIGHT)
    return world


def _night(world: World, **deps: Any) -> NightResult:
    return run_night_batch_with(world.deps(**deps), {UID: NIGHT})[UID]


def test_a_night_with_the_core_dreams_rehearses_its_picks_marks_its_rows_and_keeps_its_unit(tmp_path: Path) -> None:
    world, core = _world(tmp_path), FakeCore()
    seeds, rows = Seeds(world), Rows(world)
    result = _night(world, core=core, practice=seeds, trainer=rows)
    names = [record["step"] for record in steps(result.log_path)]
    assert names[names.index("notice") + 1] == "dream"  # 2c: after the notice pass, before the choice
    day = [f"{UID}-{NIGHT.isoformat()}-e{i}" for i in range(3)]
    assert core.days == [day] and step(result.log_path, "dream")["picked"] == day[:1]
    assert seeds.seeded == [day[:1]]  # practice rehearses what the dream picked
    surprise = {item_id: n + 2.0 for n, item_id in enumerate(day)}
    made = {t.id: [surprise[i] for i in t.f_ids if i in surprise] for t in world.transcripts(UID).for_night(NIGHT)}
    weights = {row["id"]: row.get("weight", 1.0) for row in map(json.loads, rows.data[0].splitlines())}
    assert weights and all(w == (sum(made[i]) / len(made[i]) if made[i] else 1.0) for i, w in weights.items())
    assert step(result.log_path, "dataset")["mind_weighted"] == sum(w != 1.0 for w in weights.values()) > 0
    assert step(result.log_path, "train")["surprise_weights"] == "not applied (the training job did not report)"
    serving = world.registry.last_promoted(UID)
    assert serving is not None and core.kept == [serving.id]
    assert summary(result.log_path)["mind_unit"] == {"version": 1, "state": STATE, "lora": serving.id}


def test_the_core_dreams_only_for_the_people_it_serves(tmp_path: Path) -> None:
    world, core, other = _world(tmp_path), FakeCore(), "p2"
    world.targets(other).upsert(thread(other))
    world.plan(other, NIGHT)
    results = run_night_batch_with(world.deps(core=core), {UID: NIGHT, other: NIGHT})
    assert "dream" in [r["step"] for r in steps(results[UID].log_path)]
    assert "dream" not in [r["step"] for r in steps(results[other].log_path)]  # never another person
    assert "mind_unit" not in summary(results[other].log_path) and "mind_weighted" not in step(
        results[other].log_path, "dataset")
    assert len(core.days) == 1 and len(core.kept) == 1


@pytest.mark.parametrize("picks", [0, -1], ids=["nothing picked", "nothing trainable picked"])
def test_a_dream_with_nothing_to_rehearse_seeds_practice_as_before(tmp_path: Path, picks: int) -> None:
    world = _world(tmp_path)
    seeds = Seeds(world)
    _night(world, core=FakeCore(picks=picks), practice=seeds)
    assert seeds.seeded == [[f"{UID}-{NIGHT.isoformat()}-e{i}" for i in range(3)]]


def test_a_person_deleted_during_the_night_gets_no_unit(tmp_path: Path) -> None:
    world, core = _world(tmp_path), FakeCore()
    finished = {"yes": False}

    def cancel(uid: str) -> bool:
        return finished["yes"]

    class Finishing(Rows):
        def train(self, user_id: str, night: date, prev: Any, data: Any, name: str) -> Any:
            finished["yes"] = True  # the delete lands while the adapter trains
            return super().train(user_id, night, prev, data, name)

    result = _night(world, core=core, trainer=Finishing(world), cancel=cancel, purge=lambda uid: None)
    assert result.status == "failed" and core.kept == []


def test_a_unit_that_fails_to_keep_is_recorded_and_the_night_still_ends(tmp_path: Path) -> None:
    world = _world(tmp_path)
    result = _night(world, core=FakeCore(keep_fails=True))
    record = summary(result.log_path)
    assert record["mind_unit"] is None and record["mind_unit_error"] == "OSError"
    assert result.status == "promoted"


def _starts(world: World) -> list[str | None]:
    return [e.step.entry.decider for e in workspace(world, UID).events()
            if isinstance(e.step.entry, it.RunStart) and e.step.entry.call == "inquiry"]


def _killing(found: Researches, killed: list[bool]) -> Any:
    def research(call: MindCall) -> Any:
        port = found(call)

        class Once:
            def research(self, question_id: str, query: str) -> Any:
                if not killed:
                    killed.append(True)
                    raise Kill()
                return port.research(question_id, query)
        return Once()
    return research


def _killed(root: Path) -> tuple[World, Any, LifeThinker]:
    """A night whose core-decided inquiry is killed mid tool call."""
    world, found = _world(root), Researches()
    research = _killing(found, [])
    thinker = LifeThinker(targets=(f"{UID}-t1",), tool="research", queries=["sorting line pilots"])
    mind = mind_night(world, thinker, research=research)
    with pytest.raises(Kill):
        _night(world, settings=ON, inquiry=mind, notice=mind, core=FakeCore())
    assert _starts(world) == [bridge.CORE_DECIDER]  # the dream's view decided the inquiry
    return world, research, thinker


def test_a_killed_core_decided_inquiry_resumes_under_the_core(tmp_path: Path) -> None:
    world, research, thinker = _killed(tmp_path)
    mind = mind_night(world, thinker, research=research)
    rerun = _night(world, settings=ON, inquiry=mind, notice=mind, core=FakeCore())
    inquiry = step(rerun.log_path, "choose")["inquiry"]
    assert inquiry["resumed"] and inquiry["ops"]["CONCLUDE"] == 1
    assert _starts(world) == [bridge.CORE_DECIDER, bridge.CORE_DECIDER]


def test_a_core_decided_run_never_resumes_under_the_controller(tmp_path: Path) -> None:
    world, research, thinker = _killed(tmp_path)
    mind = mind_night(world, thinker, research=research)
    rerun = _night(world, settings=ON, inquiry=mind, notice=mind)  # the core switched off since
    inquiry = step(rerun.log_path, "choose")["inquiry"]
    assert inquiry["stopped"] == "failed" and inquiry["error"] == "ValueError"
    assert _starts(world) == [bridge.CORE_DECIDER]


def test_without_a_dream_the_inquiry_is_the_controllers(tmp_path: Path) -> None:
    world = _world(tmp_path)
    mind = mind_night(world, LifeThinker(targets=(f"{UID}-t1",)))
    _night(world, settings=ON, inquiry=mind, notice=mind, core=FakeCore(view=None))
    assert _starts(world) == [None]
