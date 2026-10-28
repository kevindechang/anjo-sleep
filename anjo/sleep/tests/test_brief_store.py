# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The brief log (Task 5, step 6): both briefs folded nightly, each with its ancestry, so a cleared
reply built on the training brief can train."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief_store import JsonlBriefStore, ServedBriefs, TrainingBriefs
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.model_api import ChatResult
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.tests.sleep_fakes import WONDER, brief_text, ev

NIGHT = date(2026, 10, 28)
AT = datetime(2026, 10, 28, 23, 0, tzinfo=timezone.utc)


class _Chat:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
                 max_tokens: int | None = None, seed: int | None = None) -> ChatResult:
        self.calls += 1
        return ChatResult(text=brief_text(f"fold {self.calls}"), tool_calls=(), model=SPARK_OPEN,
                          input_tokens=10, output_tokens=10, finish_reason="stop")


def _store(tmp_path: Path) -> JsonlBriefStore:
    return JsonlBriefStore(tmp_path / "briefs.jsonl", "p1", codec=PLAIN, guard=NO_GUARD)


def test_training_brief_ancestry_is_its_sources_plus_the_compiler(tmp_path: Path) -> None:
    store = _store(tmp_path)
    own = ev("e1", ancestry=frozenset())
    practiced = ev("e2", kind="practice", ancestry=frozenset({SPARK_OPEN, GLIMMER}))
    hosted = ev("e3", kind="homework", ancestry=frozenset({WONDER}))
    store.fold(NIGHT, [own, practiced, hosted], at=AT, hosted_in_training=False, chat=_Chat())
    training = TrainingBriefs(lambda uid: store).served("p1")
    served = ServedBriefs(lambda uid: store).served("p1")
    assert training is not None and served is not None
    assert training.ancestry == frozenset({SPARK_OPEN, GLIMMER})  # hosted homework never folds in
    assert served.ancestry == frozenset({SPARK_OPEN, GLIMMER, WONDER})
    assert training.compiled_at == AT


def test_an_unknown_source_makes_the_served_brief_unknown_but_never_the_training_one(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.fold(NIGHT, [ev("e1", ancestry=None), ev("e2")], at=AT, hosted_in_training=False, chat=_Chat())
    latest = store.latest()
    assert latest is not None
    assert latest.served_ancestry is None
    assert latest.training_ancestry == frozenset({SPARK_OPEN})


def test_a_night_with_nothing_trainable_keeps_the_previous_training_brief(tmp_path: Path) -> None:
    store, chat = _store(tmp_path), _Chat()
    first = store.fold(NIGHT, [], at=AT, hosted_in_training=False, chat=chat)
    assert chat.calls == 0 and first.training_ancestry == frozenset()
    store.fold(date(2026, 10, 29), [ev("e1", kind="homework", ancestry=frozenset({WONDER}))],
               at=AT, hosted_in_training=False, chat=chat)
    latest = store.latest()
    assert latest is not None
    assert latest.training == first.training and latest.training_ancestry == frozenset()


def test_ancestry_accumulates_and_folded_items_are_not_folded_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    one, two = ev("e1", ancestry=frozenset({GLIMMER})), ev("e2", minutes=5)
    store.fold(NIGHT, [one], at=AT, hosted_in_training=False, chat=_Chat())
    assert [item.id for item in store.unfolded([one, two])] == ["e2"]
    store.fold(date(2026, 10, 29), [two], at=AT, hosted_in_training=False, chat=_Chat())
    latest = store.latest()
    assert latest is not None and latest.training_ancestry == frozenset({GLIMMER, SPARK_OPEN})
    assert store.unfolded([one, two]) == []
