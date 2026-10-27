# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Minimal practice (Task 5): cue-free scenes from the visible part of a life, F without practice
or hosted items, and each example's input a pre-practice snapshot that never holds its target.
Spark Open and Glimmer are fakes."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief_store import JsonlBriefStore, TrainingBriefs
from anjo.sleep.context.builder import RecallSettings
from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.context.training import TrainingContext
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.model_api import ChatResult
from anjo.sleep.practice.run import CUES, MinimalPractice, PracticeError, has_cue, practice_f
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.tests.sleep_fakes import WONDER, brief_text, ev

NIGHT = date(2026, 10, 28)
AT = datetime(2026, 10, 28, 23, 30, tzinfo=timezone.utc)


def _result(text: str) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model="m", input_tokens=1, output_tokens=1, finish_reason="stop")


class SparkScenes:
    def __init__(self, scenes: list[str]) -> None:
        self.scenes = scenes
        self.prompts: list[str] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        assert model == SPARK_OPEN
        self.prompts.append(messages[-1]["content"])
        return _result(json.dumps({"scenes": self.scenes}))


class Glimmer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        self.calls.append({"adapter": adapter, "messages": messages, "seed": seed})
        return _result(f"TARGET-REPLY-{len(self.calls)} to: {messages[-1]['content'][:20]}")


def _scenes(n: int = 8) -> list[str]:
    return [f"I have a new sorting-line brief, number {i}. Where would you start?" for i in range(n)]


def _evidence() -> list[Any]:
    return [
        ev("own", "I run a recycling startup and I hate jargon.", minutes=1),
        ev("pr-old", "They asked: x\nShe answered: y", kind="practice", ancestry=frozenset({SPARK_OPEN, GLIMMER}), minutes=2),
        ev("hw", "A paper found that optical sorters miss black plastic.", kind="homework",
           ancestry=frozenset({WONDER}), minutes=3),
        ev("unknown", "an item of unknown ancestry", ancestry=None, minutes=4),
        ev("held", "a held-out preference", held_out=True, minutes=5),
    ]


def _practice(spark: SparkScenes, glimmer: Glimmer, snapshot: Any = None) -> MinimalPractice:
    return MinimalPractice(chat=spark, client=glimmer, seed=7,
                           snapshot=snapshot or (lambda uid, message: ("C-SNAPSHOT", frozenset({SPARK_OPEN}))))


def test_practice_f_excludes_practice_and_hosted_items() -> None:
    f = practice_f(_evidence(), hosted_in_training=False)
    assert [item.id for item in f] == ["own"]
    spark, glimmer = SparkScenes(_scenes()), Glimmer()
    run = _practice(spark, glimmer).run("p1", NIGHT, _evidence(), None, at=AT)
    assert run.f_ids == ("own",)
    system = glimmer.calls[0]["messages"][0]["content"]
    assert "jargon" in system and "optical sorters" not in system and "She answered" not in system
    assert "unknown ancestry" not in system and "held-out" not in system
    assert "optical sorters" not in spark.prompts[0]


def test_practice_f_is_newest_first_inside_its_cap() -> None:
    items = [ev(f"e{i}", "word " * 40, minutes=i) for i in range(5)]
    kept = practice_f(items, hosted_in_training=False, cap_tokens=160)
    assert [item.id for item in kept] == ["e4", "e3", "e2"]


def test_practice_prompts_are_cue_free() -> None:
    cued = ["As we discussed, can you redo the sorting plan?", "Like I said last time, the pitch is long."]
    spark, glimmer = SparkScenes(_scenes(6) + cued), Glimmer()
    run = _practice(spark, glimmer).run("p1", NIGHT, _evidence(), None, at=AT)
    assert len(run.transcripts) == 6 and run.dropped_cued == 2
    assert not any(has_cue(t.message) for t in run.transcripts)
    assert all(has_cue(f"well, {cue} the plan") for cue in CUES)
    assert "Never refer to an earlier conversation" in spark.prompts[0]


def test_unreadable_scenes_fail_practice() -> None:
    class Broken(SparkScenes):
        def __call__(self, *a: Any, **kw: Any) -> ChatResult:
            return _result("not json")

    with pytest.raises(PracticeError):
        _practice(Broken([]), Glimmer()).run("p1", NIGHT, _evidence(), None, at=AT)


def test_homework_never_reaches_practice_while_hosted_cannot_train() -> None:
    with pytest.raises(ValueError):
        _practice(SparkScenes(_scenes()), Glimmer()).run("p1", NIGHT, _evidence(), object(), at=AT)


def test_transcripts_carry_cleared_ancestry_and_seeded_replies() -> None:
    glimmer = Glimmer()
    run = _practice(SparkScenes(_scenes()), glimmer).run("p1", NIGHT, _evidence(), None, at=AT)
    assert all(t.ancestry == frozenset({SPARK_OPEN, GLIMMER}) for t in run.transcripts)
    again = Glimmer()
    _practice(SparkScenes(_scenes()), again).run("p1", NIGHT, _evidence(), None, at=AT)
    assert [c["seed"] for c in glimmer.calls] == [c["seed"] for c in again.calls]
    assert len({c["seed"] for c in glimmer.calls}) == 8
    assert all(c["adapter"] is None for c in glimmer.calls)  # the target is base Glimmer with F


class _Chat:
    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        return _result(brief_text("PRE-PRACTICE BRIEF"))


def _training_context(tmp_path: Path) -> tuple[TrainingContext, JsonlEvidenceStore, JsonlBriefStore]:
    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    briefs = JsonlBriefStore(tmp_path / "briefs.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)
    recall = Bm25Recall(lambda uid: store, until=NIGHT)
    return TrainingContext(TrainingBriefs(lambda uid: briefs), recall, RecallSettings(k=6, budget_tokens=1200)), store, briefs


def test_sft_uses_pre_practice_brief(tmp_path: Path) -> None:
    context, store, briefs = _training_context(tmp_path)
    items = [ev("own", "I run a recycling startup and the sorting brief is new.")]
    store.add(items)
    briefs.fold(NIGHT, items, at=AT, hosted_in_training=False, chat=_Chat())
    run = _practice(SparkScenes(_scenes()), Glimmer(), snapshot=context.snapshot).run("p1", NIGHT, items, None, at=AT)
    store.add([t.evidence() for t in run.transcripts])
    first = run.transcripts[0]
    assert first.context is not None and "PRE-PRACTICE BRIEF" in first.context
    assert first.ancestry == frozenset({SPARK_OPEN, GLIMMER})
    # Practice items are stored now; a fresh snapshot still recalls none of them (sft mode).
    after, _ = context.snapshot("p1", first.message)
    assert after is not None and "TARGET-REPLY" not in after


def test_sft_context_never_contains_its_target(tmp_path: Path) -> None:
    context, store, briefs = _training_context(tmp_path)
    items = [ev("own", "sorting brief recycling startup")]
    store.add(items)
    briefs.fold(NIGHT, items, at=AT, hosted_in_training=False, chat=_Chat())
    run = _practice(SparkScenes(_scenes()), Glimmer(), snapshot=context.snapshot).run("p1", NIGHT, items, None, at=AT)
    for transcript in run.transcripts:
        assert transcript.context is not None
        assert transcript.reply not in transcript.context
        assert "TARGET-REPLY" not in transcript.context
