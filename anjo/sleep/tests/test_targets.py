# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Targets (Task 3): an append-only event log per person, and the table is its fold."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep import ports
from anjo.sleep.evidence.jsonl import read_records, PLAIN
from anjo.sleep.evidence.targets import (
    CONFIDENCE_STEP,
    EventLogTargetStore,
    Mention,
    TargetStoreError,
    fold,
    target_id_for,
)
from anjo.sleep.types import Evidence, Expectation, Result, Target

PERSON = "synthetic-film-student"
DAY = date(2026, 10, 28)
NOW = datetime(2026, 10, 29, 2, 0, tzinfo=timezone.utc)


def _store(tmp_path: Path) -> EventLogTargetStore:
    return EventLogTargetStore(tmp_path / "targets.jsonl", PERSON)


def _thread(**changes: Any) -> Target:
    base = Target(
        id="t-edit", person_id=PERSON, kind="thread", label="the edit of the short", check_after=None,
        status="open", revision=1, acted_revision=0, confidence=0.4, last_mentioned=DAY,
    )
    return dataclasses.replace(base, **changes)


def _commitment(**changes: Any) -> Target:
    base = Target(
        id="t-festival", person_id=PERSON, kind="commitment", label="festival submission",
        check_after=NOW + timedelta(days=3), status="open", revision=1, acted_revision=0,
        confidence=None, last_mentioned=DAY,
    )
    return dataclasses.replace(base, **changes)


def _result(**changes: Any) -> Result:
    base = Result(
        id="r-1", target_id="t-edit", homework_id="hw-1", verified_sources=0, created_at=NOW,
        consumed=False,
    )
    return dataclasses.replace(base, **changes)


def _evidence(**changes: Any) -> Evidence:
    base = Evidence(
        id="e-1", person_id=PERSON, day=DAY, kind="lesson", text="They are cutting the short.",
        source="message:1", held_out=False, created_at=NOW, target_id="t-edit",
    )
    return dataclasses.replace(base, **changes)


def _mention(**changes: Any) -> Mention:
    base = Mention(
        target_id="t-edit", kind="thread", label="the edit of the short", check_after=None,
        confidence=0.4,
    )
    return dataclasses.replace(base, **changes)


def _exercise(store: EventLogTargetStore) -> None:
    """One of every mutation."""
    store.upsert(_thread())
    store.upsert(_commitment())
    store.record_mention(_mention(), _evidence(id="e-2", day=DAY + timedelta(days=1)))
    store.mark_acted("t-edit", 1)
    store.add_result(_result(verified_sources=2))
    store.record_expectation(Expectation(
        id="x-1", target_id="t-festival", made_at=NOW, horizon=NOW + timedelta(days=7), p_needed=0.6,
    ))
    store.resolve_expectation("x-1", "needed")
    store.settle("t-festival")


def test_the_store_satisfies_its_port(tmp_path: Path) -> None:
    assert isinstance(_store(tmp_path), ports.TargetStore)


def test_target_ids_are_stable_for_the_same_label() -> None:
    first = target_id_for(PERSON, "thread", "The edit of  the short")
    assert first == target_id_for(PERSON, "thread", "the edit of the short ")
    assert first != target_id_for(PERSON, "commitment", "the edit of the short")
    assert first != target_id_for("another-person", "thread", "the edit of the short")


def test_new_mention_raises_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.record_mention(_mention(), _evidence(id="e-1"))
    created = store.get("t-edit")
    assert created is not None and (created.revision, created.acted_revision) == (1, 0)

    later = DAY + timedelta(days=2)
    new_date = NOW + timedelta(days=9)
    assert store.record_mention(
        _mention(check_after=new_date), _evidence(id="e-2", kind="correction", day=later),
    )
    raised = store.get("t-edit")
    assert raised is not None
    assert (raised.revision, raised.last_mentioned) == (2, later)
    # The same item never counts twice, on a re-run of the same night or later.
    assert not store.record_mention(_mention(), _evidence(id="e-2", kind="correction", day=later))
    assert store.get("t-edit") == raised


def test_a_new_date_moves_a_commitment(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_commitment())
    moved = NOW + timedelta(days=10)
    store.record_mention(
        _mention(target_id="t-festival", kind="commitment", label="festival", check_after=moved,
                 confidence=None),
        _evidence(id="e-date", target_id="t-festival"),
    )
    target = store.get("t-festival")
    assert target is not None
    assert (target.check_after, target.revision, target.label) == (moved, 2, "festival submission")


def test_homework_result_never_raises_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_thread())
    store.add_result(_result(verified_sources=3))
    store.add_result(_result(id="r-2", homework_id=None, basis="evidence", user_cites=2))

    target = store.get("t-edit")
    assert target is not None and target.revision == 1
    for kind in ("homework", "practice"):
        with pytest.raises(TargetStoreError, match="person's own"):
            store.record_mention(_mention(), _evidence(id=f"e-{kind}", kind=kind))
    with pytest.raises(TargetStoreError, match="held out"):
        store.record_mention(_mention(), _evidence(id="e-held", kind="preference", held_out=True))
    assert store.get("t-edit") == target


def test_mark_acted_records_consumed_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_thread(revision=3))
    store.mark_acted("t-edit", 2)

    target = store.get("t-edit")
    assert target is not None and (target.acted_revision, target.unacted) == (2, 1)
    with pytest.raises(TargetStoreError, match="beyond"):
        store.mark_acted("t-edit", 4)
    with pytest.raises(TargetStoreError, match="back"):
        store.mark_acted("t-edit", 1)
    with pytest.raises(TargetStoreError, match="unknown"):
        store.mark_acted("t-missing", 1)
    # A later upsert keeps what the night consumed.
    store.upsert(_thread(revision=4, acted_revision=0))
    kept = store.get("t-edit")
    assert kept is not None and (kept.revision, kept.acted_revision) == (4, 2)


def test_upsert_never_lowers_revision_or_changes_kind(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_thread(revision=2))
    with pytest.raises(TargetStoreError, match="lower"):
        store.upsert(_thread(revision=1))
    with pytest.raises(TargetStoreError, match="kind"):
        store.upsert(_commitment(id="t-edit", revision=2))
    with pytest.raises(TargetStoreError, match="person"):
        store.upsert(_thread(id="t-other", person_id="someone-else"))


def test_confidence_rises_only_on_a_verified_source(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_thread(confidence=0.4))
    store.add_result(_result(id="r-unverified", verified_sources=0))
    store.add_result(_result(id="r-evidence", homework_id=None, basis="evidence", verified_sources=2))
    unchanged = store.get("t-edit")
    assert unchanged is not None and unchanged.confidence == 0.4

    store.add_result(_result(id="r-verified", verified_sources=1))
    risen = store.get("t-edit")
    assert risen is not None and risen.confidence == pytest.approx(0.4 + CONFIDENCE_STEP)

    store.upsert(_thread(id="t-sure", confidence=0.95))
    store.add_result(_result(id="r-cap", target_id="t-sure", verified_sources=4))
    capped = store.get("t-sure")
    assert capped is not None and capped.confidence == 1.0


def test_settle_consumes_results_and_notes_a_commitment(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_thread())
    store.upsert(_commitment())
    store.add_result(_result(id="r-thread"))
    store.add_result(_result(id="r-festival", target_id="t-festival"))

    store.settle("t-festival")

    consumed = {result.id: result.consumed for result in store.results()}
    assert consumed == {"r-thread": False, "r-festival": True}
    festival, thread = store.get("t-festival"), store.get("t-edit")
    assert festival is not None and festival.due_noted
    assert thread is not None and not thread.due_noted
    with pytest.raises(TargetStoreError, match="unknown"):
        store.settle("t-missing")


def test_results_need_a_known_target_and_a_new_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_thread())
    store.add_result(_result())
    with pytest.raises(TargetStoreError, match="already"):
        store.add_result(_result())
    with pytest.raises(TargetStoreError, match="unknown"):
        store.add_result(_result(id="r-2", target_id="t-missing"))


def test_expectations_are_recorded_and_resolved_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(_commitment())
    expectation = Expectation(
        id="x-1", target_id="t-festival", made_at=NOW, horizon=NOW + timedelta(days=7), p_needed=0.3,
    )
    store.record_expectation(expectation)
    store.resolve_expectation("x-1", "not_needed")

    assert store.expectations() == [dataclasses.replace(expectation, outcome="not_needed")]
    with pytest.raises(TargetStoreError, match="resolved"):
        store.resolve_expectation("x-1", "needed")
    with pytest.raises(TargetStoreError, match="unknown"):
        store.resolve_expectation("x-missing", "needed")


def test_accelerated_targets_use_virtual_time(tmp_path: Path) -> None:
    """Seven simulated days on one calendar date: open_targets reads only the clock it's given."""
    store = _store(tmp_path)
    virtual = [DAY + timedelta(days=offset) for offset in range(7)]
    for offset, day in enumerate(virtual):
        store.record_mention(
            _mention(target_id=f"t-{offset}", label=f"topic {offset}"),
            _evidence(id=f"e-{offset}", day=day, target_id=f"t-{offset}"),
        )
    store.upsert(dataclasses.replace(store.get("t-1") or _thread(), status="paused"))

    def open_on(day: date) -> list[str]:
        as_of = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        return [target.id for target in store.open_targets(as_of)]

    assert open_on(virtual[0]) == ["t-0"]
    assert open_on(virtual[2]) == ["t-0", "t-2"]
    assert open_on(virtual[6]) == ["t-0", "t-2", "t-3", "t-4", "t-5", "t-6"]


def test_target_table_is_a_fold_of_its_log(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _exercise(store)

    events = read_records(tmp_path / "targets.jsonl", PLAIN)
    assert [event["op"] for event in events] == [
        "upsert", "upsert", "upsert", "mark_acted", "add_result", "record_expectation",
        "resolve_expectation", "settle",
    ]
    assert [event["seq"] for event in events] == list(range(len(events)))
    state = fold(events)
    assert sorted(state.targets.values(), key=lambda target: target.id) == [
        store.get("t-edit"), store.get("t-festival"),
    ]
    assert list(state.results) == store.results()
    assert list(state.expectations) == store.expectations()
    # A fresh reader of the same file sees the same table.
    assert _store(tmp_path).export() == store.export()


def test_a_log_with_a_gap_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _exercise(store)
    events = read_records(tmp_path / "targets.jsonl", PLAIN)
    with pytest.raises(TargetStoreError, match="seq"):
        fold([events[0], events[2]])


def test_replay_digest_equals_live_digest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    empty = store.digest()
    assert store.replay_digest() == empty

    _exercise(store)

    assert store.digest() != empty
    assert store.replay_digest() == store.digest()
    assert _store(tmp_path).replay_digest() == store.digest()


def test_a_second_writer_on_the_same_log_is_caught_up(tmp_path: Path) -> None:
    first, second = _store(tmp_path), _store(tmp_path)
    first.upsert(_thread())
    second.add_result(_result())  # second has never read the log: it catches up before writing
    first.settle("t-edit")

    assert first.results() == second.results()
    assert [event["seq"] for event in read_records(tmp_path / "targets.jsonl", PLAIN)] == [0, 1, 2]
    assert first.digest() == second.digest() == first.replay_digest()


def test_export_carries_targets_results_and_expectations(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _exercise(store)

    exported: dict[str, Any] = store.export()
    assert [target["id"] for target in exported["targets"]] == ["t-edit", "t-festival"]
    assert exported["targets"][0]["first_seen"] == DAY.isoformat()
    assert [result["id"] for result in exported["results"]] == ["r-1"]
    assert exported["expectations"][0]["outcome"] == "needed"
