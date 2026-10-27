# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The evidence store (Task 3): an append-only JSONL file per person, read through the port."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep import ports
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.evidence.jsonl import LogError
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.meta_api import PINNED_MODELS
from anjo.sleep.provenance import GLIMMER, may_train
from anjo.sleep.types import Evidence

PERSON = "synthetic-film-student"
DAY = date(2026, 10, 28)
NOW = datetime(2026, 10, 29, 2, 0, tzinfo=timezone.utc)


def _item(**changes: Any) -> Evidence:
    base = Evidence(
        id="e-1", person_id=PERSON, day=DAY, kind="preference", text="They prefer long takes.",
        source="message:1", held_out=False, created_at=NOW, model=SPARK_OPEN,
        ancestry=frozenset({SPARK_OPEN}),
    )
    return dataclasses.replace(base, **changes)


def _store(tmp_path: Path, **kwargs: Any) -> JsonlEvidenceStore:
    return JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON, **kwargs)


def test_the_store_satisfies_its_port(tmp_path: Path) -> None:
    assert isinstance(_store(tmp_path), ports.EvidenceStore)


def test_trainable_never_returns_held_out(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_item(id="e-kept"), _item(id="e-held", held_out=True)])

    assert [item.id for item in store.trainable(DAY)] == ["e-kept"]
    assert [item.id for item in store.stored(DAY)] == ["e-kept"]
    assert [item.id for item in store.held_out(DAY)] == ["e-held"]
    assert {item.id for item in store.for_day(DAY)} == {"e-kept", "e-held"}


def test_trainable_and_stored_stop_at_until(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_item(id="e-1"), _item(id="e-2", day=DAY + timedelta(days=1))])

    assert [item.id for item in store.trainable(DAY)] == ["e-1"]
    assert [item.id for item in store.stored(DAY)] == ["e-1"]
    assert [item.id for item in store.for_day(DAY + timedelta(days=1))] == ["e-2"]


def test_trainable_refuses_unknown_and_uncleared_ancestry(tmp_path: Path) -> None:
    hosted = next(iter(PINNED_MODELS.values()))
    items = [
        _item(id="e-own", model=None, ancestry=frozenset()),
        _item(id="e-spark-open", ancestry=frozenset({SPARK_OPEN})),
        _item(id="e-glimmer", ancestry=frozenset({GLIMMER, SPARK_OPEN})),
        _item(id="e-unknown", ancestry=None),
        _item(id="e-hosted", kind="homework", ancestry=frozenset({hosted})),
        _item(id="e-frontier", ancestry=frozenset({"claude-opus"})),
    ]
    cleared_only = _store(tmp_path)
    cleared_only.add(items)
    with_hosted = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON, hosted_in_training=True)

    assert [item.id for item in cleared_only.trainable(DAY)] == ["e-own", "e-spark-open", "e-glimmer"]
    assert [item.id for item in with_hosted.trainable(DAY)] == [
        "e-own", "e-spark-open", "e-glimmer", "e-hosted",
    ]
    # Stored context still carries every non-held-out item, homework included (arm 2F).
    assert len(cleared_only.stored(DAY)) == len(items)


def test_may_train_is_an_allowlist() -> None:
    assert may_train(frozenset(), hosted_in_training=False)
    assert not may_train(None, hosted_in_training=True)
    assert not may_train(frozenset({"gemini-3.8-flash"}), hosted_in_training=True)


def test_add_stores_an_id_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_item(id="e-1"), _item(id="e-1", text="A second copy in one batch.")])
    store.add([_item(id="e-1", text="The same id on a later night.")])

    assert [item.text for item in store.for_day(DAY)] == ["They prefer long takes."]
    assert len((tmp_path / "evidence.jsonl").read_text().splitlines()) == 1


def test_add_refuses_another_persons_items(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="person"):
        store.add([_item(person_id="someone-else")])
    assert store.for_day(DAY) == []


def test_items_round_trip_with_their_provenance(tmp_path: Path) -> None:
    item = _item(target_id="t-abc", ancestry=frozenset({SPARK_OPEN, GLIMMER}))
    _store(tmp_path).add([item])

    assert _store(tmp_path).for_day(DAY) == [item]


def test_search_ranks_by_overlap_and_respects_as_of(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([
        _item(id="e-takes", text="They prefer long takes in documentary work."),
        _item(id="e-sound", text="They record sound on set."),
        _item(id="e-later", text="Long takes again.", created_at=NOW + timedelta(days=1)),
        _item(id="e-held", text="Long takes, held out.", held_out=True),
    ])

    assert [item.id for item in store.search("long takes", NOW, k=5)] == ["e-takes"]
    # Equal overlap: the newer item first.
    assert [item.id for item in store.search("long takes", NOW + timedelta(days=1), k=2)] == [
        "e-later", "e-takes",
    ]
    assert store.search("nothing matches", NOW, k=5) == []


def test_a_torn_last_line_is_ignored_and_repaired(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_item(id="e-1")])
    path = tmp_path / "evidence.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"id": "e-torn"')  # a write killed mid-line

    assert [item.id for item in store.for_day(DAY)] == ["e-1"]
    store.add([_item(id="e-2")])
    assert [item.id for item in store.for_day(DAY)] == ["e-1", "e-2"]
    assert path.read_text().endswith("\n")


def test_an_unreadable_line_refuses_the_log(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    path.write_text("not json\n")
    with pytest.raises(LogError, match="line 1"):
        _store(tmp_path).for_day(DAY)


def test_export_lists_every_item(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add([_item(id="e-1"), _item(id="e-2", held_out=True, ancestry=None)])

    exported: dict[str, Any] = store.export()
    assert [record["id"] for record in exported["evidence"]] == ["e-1", "e-2"]
    assert exported["evidence"][0]["ancestry"] == [SPARK_OPEN]
    assert exported["evidence"][1]["ancestry"] is None
