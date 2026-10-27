# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Homework in the night (Task 7): a study action her chooser picks runs ``do_homework`` on that
target and commits a ``research`` result. The artifact is evidence with Wonder in its ancestry, so
while hosted outputs can't train it stays out of every training input (the training brief, SFT
recall, practice and its F, the dataset) and still reaches her through the served brief, recall and
the note. Above the rolling $3-a-month Wonder cap no homework runs, and the night says why."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.homework.agent import WonderHomework
from anjo.sleep.meta_api import PINNED_MODELS
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.stops import BudgetStop
from anjo.sleep.tests.homework_fakes import (
    GUIDE_QUOTE,
    STUDY_QUOTE,
    STUDY_URL,
    FakeSearch,
    FakeWonder,
    research_script,
    study_web,
)
from anjo.sleep.tests.night_fakes import NIGHT, Trainer, World, own, step, summary, thread, window
from anjo.sleep.types import NightResult

WONDER_ID = PINNED_MODELS["muse-wonder"]


class KeepingTrainer(Trainer):
    """The night's trainer fake, keeping the rows it was given."""

    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.rows: list[str] = []

    def train(self, user_id: str, night: date, prev: Any, data: Any, name: str) -> Any:
        self.rows.append(data.to_jsonl())
        return super().train(user_id, night, prev, data, name)


def _world(tmp_path: Path) -> tuple[World, FakeWonder]:
    world = World(tmp_path)
    world.planned[("p1", NIGHT)] = [
        replace(own("p1", NIGHT, 0, "I can't get the sorter to see black film at all."), target_id="p1-t1"),
        own("p1", NIGHT, 1), own("p1", NIGHT, 2),
    ]
    world.targets("p1").upsert(thread("p1", "sorting black film"))
    return world, FakeWonder(research_script())


def _homework(world: World, wonder: FakeWonder) -> WonderHomework:
    return WonderHomework(wonder=wonder, search=FakeSearch(), fetcher=study_web().fetcher(), evidence=world.evidence)


def _one(world: World, **deps: Any) -> NightResult:
    return run_night_batch_with(world.deps(**deps), {"p1": NIGHT})["p1"]


def test_study_action_runs_homework_and_records_result(tmp_path: Path) -> None:
    world, wonder = _world(tmp_path)
    result = _one(world, homework=_homework(world, wonder))
    choose = step(result.log_path, "choose")
    assert choose["action"] == "revisit_thread" and choose["target_id"] == "p1-t1"
    assert {"revisit_thread", "prepare_commitment"} <= set(choose["offered"])  # study actions are back
    assert choose["ports"]["homework"] == "available"
    assert "sorting black film" in json.dumps(wonder.calls[0]["messages"])
    [found] = world.targets("p1").results()
    assert found.basis == "research" and found.verified_sources == 2 and found.target_id == "p1-t1"
    [artifact] = [e for e in world.evidence("p1").stored(NIGHT) if e.kind == "homework"]
    assert found.homework_id == artifact.id and WONDER_ID in (artifact.ancestry or ())
    assert choose["study"]["result"] == found.id and choose["study"]["verified_sources"] == 2
    assert choose["study"]["wonder_calls"] == 3 and choose["receipt"]["outcome"] == "ran"
    [target] = world.targets("p1").open_targets(window(NIGHT)[1])
    assert target.acted_revision == target.revision
    assert summary(result.log_path)["wonder_calls"] == 3


def test_homework_never_reaches_training_inputs(tmp_path: Path) -> None:
    world, wonder = _world(tmp_path)
    trainer = KeepingTrainer(world)
    result = _one(world, homework=_homework(world, wonder), trainer=trainer)
    assert result.status == "promoted"
    [artifact] = [e for e in world.evidence("p1").stored(NIGHT) if e.kind == "homework"]
    marks = (STUDY_QUOTE, GUIDE_QUOTE, STUDY_URL)
    # the trainable view, so extraction's trainable items and the training brief never hold it
    assert artifact.id not in {e.id for e in world.evidence("p1").trainable(NIGHT)}
    briefs = [p for p in world.spark.prompts if p.startswith("# brief_v1")]
    served = [p for p in briefs if "INCLUDE_HOSTED: true" in p]
    training = [p for p in briefs if "INCLUDE_HOSTED: false" in p]
    assert served and training
    assert any(STUDY_QUOTE in p for p in served) and not any(m in p for p in training for m in marks)
    # practice: no homework in the scene prompt or in F, and none in any pre-practice C
    scenes = [p for p in world.spark.prompts if p.startswith("# scenes_v1")]
    assert scenes and not any(m in p for p in scenes for m in marks)
    assert step(result.log_path, "practice")["homework"] is None
    transcripts = world.transcripts("p1").for_night(NIGHT)
    assert transcripts and not any(m in (t.context or "") + t.reply for t in transcripts for m in marks)
    # SFT recall and the dataset rows
    sft = Bm25Recall(world.evidence, until=NIGHT).recall("p1", "optical sorters black film", 10, 4000, mode="sft")
    assert artifact.id not in {e.id for e in sft}
    assert trainer.rows and not any(m in row for row in trainer.rows for m in marks)
    # ... and it still reaches her: served recall, the served brief (above) and the note
    served_recall = Bm25Recall(world.evidence, until=NIGHT).recall("p1", "optical sorters black film", 10, 4000)
    assert artifact.id in {e.id for e in served_recall}
    [(_, note, _)] = world.notifier.queued
    assert STUDY_URL in note
    assert any(STUDY_QUOTE in fact.text for fact in world.writer.calls[0][1])


def test_homework_stops_at_monthly_cap(tmp_path: Path) -> None:
    world, wonder = _world(tmp_path)
    spent = world.logs("p1") / "2026-10-20"
    spent.mkdir(parents=True)
    (spent / "summary.json").write_text(json.dumps({
        "night_end": "2026-10-21T00:00:00+00:00", "wonder_usd": 3.0, "wonder_calls": 9, "paid_actions": 1}))
    result = _one(world, homework=_homework(world, wonder))
    assert wonder.calls == []  # no homework ran
    choose = step(result.log_path, "choose")
    assert choose["action"] is None and choose["wonder_month"] == {"cap_usd": 3.0, "spent_usd": 3.0, "left_usd": 0.0}
    assert "monthly" in choose["ports"]["homework"]
    reasons = {c["want_id"]: c["reason"] for c in choose["receipt"]["candidates"]}
    assert reasons == {"understand:p1-t1": "budget"}
    assert world.targets("p1").results() == []
    assert result.status == "promoted"  # practice and training still run


def test_a_cap_spent_more_than_a_month_ago_is_free_again(tmp_path: Path) -> None:
    world, wonder = _world(tmp_path)
    old = world.logs("p1") / "2026-09-01"
    old.mkdir(parents=True)
    (old / "summary.json").write_text(json.dumps({
        "night_end": "2026-09-02T00:00:00+00:00", "wonder_usd": 3.0, "paid_actions": 1}))
    result = _one(world, homework=_homework(world, wonder))
    assert step(result.log_path, "choose")["wonder_month"]["left_usd"] == 3.0
    assert len(wonder.calls) == 3


def test_a_month_with_less_than_one_call_left_stops_the_study_without_spending_an_action(tmp_path: Path) -> None:
    world, wonder = _world(tmp_path)
    spent = world.logs("p1") / "2026-10-20"
    spent.mkdir(parents=True)
    (spent / "summary.json").write_text(json.dumps({
        "night_end": "2026-10-21T00:00:00+00:00", "wonder_usd": 2.8, "paid_actions": 1}))
    result = _one(world, homework=_homework(world, wonder))
    assert wonder.calls == []  # $0.20 left can't hold one call's worst case
    choose = step(result.log_path, "choose")
    assert choose["study"]["stopped"] == "budget" and choose["receipt"]["outcome"] == "budget"
    assert summary(result.log_path)["paid_actions"] == 0


class _Failing:
    """A homework port whose study debits one call and then fails with ``error``."""

    available = True

    def __init__(self, error: Exception) -> None:
        self.error = error

    def study(self, user_id: str, target: Any, night: date, meter: Any, **kw: Any) -> Any:
        meter.debit(0.1)
        raise self.error


@pytest.mark.parametrize("error", [
    ModelApiError("Meta Model API chat with muse-wonder-1.0 returned HTTP 503"),
    BudgetStop("meta_usd can't hold the worst case"),
    ValueError("a homework request can't fit inside its input bound"),
    LookupError("unknown encoding: bogus"),
])
def test_a_homework_failure_stays_in_its_step(tmp_path: Path, error: Exception) -> None:
    world, _ = _world(tmp_path)
    result = _one(world, homework=_Failing(error))
    assert result.status == "promoted"  # brief, practice, training and promotion still ran
    choose = step(result.log_path, "choose")
    assert choose["study"]["stopped"] == "failed" and choose["study"]["error"] == type(error).__name__
    assert choose["receipt"]["outcome"] == "failed"
    [target] = world.targets("p1").open_targets(window(NIGHT)[1])
    assert target.acted_revision != target.revision  # the want stays
    assert world.targets("p1").results() == []
    assert summary(result.log_path)["wonder_usd"] == pytest.approx(0.1)  # the debit is kept


def test_a_wonder_error_mid_study_keeps_the_night_and_its_debit(tmp_path: Path) -> None:
    world, _ = _world(tmp_path)
    wonder = FakeWonder(research_script(), fail_at=2)
    result = _one(world, homework=_homework(world, wonder))
    assert result.status == "promoted"
    choose = step(result.log_path, "choose")
    assert choose["study"]["stopped"] == "failed" and choose["study"]["error"] == "ModelApiError"
    assert summary(result.log_path)["wonder_calls"] == 2
