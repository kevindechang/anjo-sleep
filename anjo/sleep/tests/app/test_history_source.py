# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The app's evidence source (Task 3): the person's turns from Anjo's history, their own stored
corrections and preferences, and a synthetic person's cleared sessions only."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from anjo.sleep import ports
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.model_api import ChatResult

REAL = "person-real"
SYNTHETIC = "synthetic-film-student"
DAY = date(2026, 10, 28)
START = datetime(2026, 10, 28, tzinfo=timezone.utc)
END = START + timedelta(days=1)
WINDOW = (START, END)


@dataclass
class EchoSparkOpen:
    """Returns one lesson per message it is sent, so a test sees exactly what was read."""

    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]]) -> ChatResult:
        self.calls.append(messages)
        sent = json.loads(messages[-1]["content"])["messages"]
        items = [
            {"turn": m["turn"], "kind": "lesson", "text": f"They said: {m['text']}", "target": None}
            for m in sent
        ]
        return ChatResult(text=json.dumps({"items": items}), tool_calls=(), model=SPARK_OPEN,
                          input_tokens=10, output_tokens=5, finish_reason="stop")

    def sent(self) -> str:
        return json.dumps(self.calls)


def _message(user_id: str, role: str, content: str, *, at: datetime, session_id: str = "s-1") -> int:
    from anjo.core.crypto import encrypt_db
    from anjo.core.db import get_db

    db = get_db()
    cursor = db.execute(
        "INSERT INTO messages (user_id, role, content, created_at, session_id, turn_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, role, encrypt_db(content), at.isoformat(), session_id, uuid.uuid4().hex),
    )
    db.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _source(chat: EchoSparkOpen) -> Any:
    from anjo.sleep.app.history_source import HistorySource

    return HistorySource(chat=chat)


def test_the_source_satisfies_its_port() -> None:
    assert isinstance(_source(EchoSparkOpen()), ports.EvidenceSource)


def test_extraction_reads_only_the_persons_turns() -> None:
    noon = START + timedelta(hours=12)
    _message(REAL, "user", "I edit on paper first.", at=noon)
    _message(REAL, "assistant", "HER-REPLY carrying memory", at=noon + timedelta(seconds=1))
    chat = EchoSparkOpen()

    items = _source(chat).extract_day(REAL, DAY, WINDOW)

    assert "I edit on paper first." in chat.sent() and "HER-REPLY" not in chat.sent()
    assert [item.text for item in items] == ["They said: I edit on paper first."]
    # A real person's words have no ancestor; the item's is the model that extracted it.
    assert items[0].ancestry == frozenset({SPARK_OPEN}) and items[0].model == SPARK_OPEN


def test_synthetic_extraction_reads_only_cleared_sessions() -> None:
    from anjo.sleep.app import cleared_sessions

    noon = START + timedelta(hours=12)
    cleared_sessions.register_synthetic(SYNTHETIC, at=START)
    _message(SYNTHETIC, "user", "simulated day words", at=noon, session_id="s-sim")
    _message(SYNTHETIC, "user", "FILMED-OPEN words", at=noon, session_id="s-film")
    chat = EchoSparkOpen()

    assert _source(chat).extract_day(SYNTHETIC, DAY, WINDOW) == []
    assert chat.calls == []  # no cleared session yet: nothing is read

    cleared_sessions.mark_cleared(SYNTHETIC, "s-sim", at=noon)
    items = _source(chat).extract_day(SYNTHETIC, DAY, WINDOW)

    assert "simulated day words" in chat.sent() and "FILMED-OPEN" not in chat.sent()
    assert [item.source.startswith("message:") for item in items] == [True]
    assert items[0].ancestry == frozenset({SPARK_OPEN})


def test_cleared_flag_refused_for_real_people() -> None:
    from anjo.sleep.app import cleared_sessions

    with pytest.raises(cleared_sessions.NotSynthetic):
        cleared_sessions.mark_cleared(REAL, "s-1", at=START)
    assert cleared_sessions.cleared_sessions(REAL) == frozenset()
    assert not cleared_sessions.is_synthetic(REAL)


def test_cleared_sessions_are_recorded_once() -> None:
    from anjo.sleep.app import cleared_sessions

    cleared_sessions.register_synthetic(SYNTHETIC, at=START)
    cleared_sessions.register_synthetic(SYNTHETIC, at=END)
    cleared_sessions.mark_cleared(SYNTHETIC, "s-1", at=START)
    cleared_sessions.mark_cleared(SYNTHETIC, "s-1", at=END)

    assert cleared_sessions.cleared_sessions(SYNTHETIC) == frozenset({"s-1"})
    assert [record["kind"] for record in cleared_sessions.export_sessions(SYNTHETIC)] == [
        "synthetic", "cleared",
    ]


def test_extract_skips_forgotten_turns() -> None:
    from anjo.core.forgotten_sessions import mark_session_forgotten

    noon = START + timedelta(hours=12)
    _message(REAL, "user", "kept words", at=noon, session_id="s-kept")
    _message(REAL, "user", "FORGOTTEN words", at=noon, session_id="s-gone")
    mark_session_forgotten(REAL, "s-gone")
    chat = EchoSparkOpen()

    _source(chat).extract_day(REAL, DAY, WINDOW)

    assert "kept words" in chat.sent() and "FORGOTTEN" not in chat.sent()


def test_extract_reads_only_its_window() -> None:
    _message(REAL, "user", "BEFORE the window", at=START - timedelta(microseconds=1))
    _message(REAL, "user", "at the start", at=START)
    _message(REAL, "user", "late that night", at=END - timedelta(microseconds=1))
    _message(REAL, "user", "AT-THE-END belongs to the next day", at=END)
    chat = EchoSparkOpen()

    items = _source(chat).extract_day(REAL, DAY, WINDOW)

    assert [item.text for item in items] == ["They said: at the start", "They said: late that night"]
    assert "BEFORE" not in chat.sent() and "AT-THE-END" not in chat.sent()
    assert all(item.day == DAY and item.created_at == END for item in items)


def test_an_accelerated_window_is_its_own_day() -> None:
    """Two simulated days inside one calendar hour stay two days."""
    first = (START, START + timedelta(minutes=20))
    second = (START + timedelta(minutes=20), START + timedelta(minutes=40))
    _message(REAL, "user", "virtual day one", at=START + timedelta(minutes=5))
    _message(REAL, "user", "virtual day two", at=START + timedelta(minutes=25))
    source = _source(EchoSparkOpen())

    day_one = source.extract_day(REAL, date(2026, 11, 1), first)
    day_two = source.extract_day(REAL, date(2026, 11, 2), second)

    assert [(item.day, item.text) for item in day_one + day_two] == [
        (date(2026, 11, 1), "They said: virtual day one"),
        (date(2026, 11, 2), "They said: virtual day two"),
    ]


def _correction_row(user_id: str, evidence_id: str, quote: str) -> None:
    from anjo.core.crypto import encrypt_db
    from anjo.core.db import get_db

    payload = {"patch": {"field": "verbosity", "value": "short", "scope": "ongoing",
                         "operation": "set"}, "quote": quote}
    db = get_db()
    db.execute(
        "INSERT INTO learning_evidence (user_id, evidence_id, event_key, relationship_epoch, "
        "consent_generation, revocation_generation, kind, payload_encrypted, created_at) "
        "VALUES (?, ?, ?, 0, 0, 0, 'explicit_preference', ?, ?)",
        (user_id, evidence_id, f"key-{evidence_id}", encrypt_db(json.dumps(payload)), START.isoformat()),
    )
    db.commit()


def test_imported_items_enter_once_across_nights(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core import learning_evidence, personal_preferences
    from anjo.sleep.app.storage import evidence_store

    _correction_row(REAL, "ev-eligible", "keep it short, please")
    _correction_row(REAL, "ev-withheld", "WITHHELD correction")
    seen_rows: list[str] = []

    def eligible(row: Any) -> bool:
        seen_rows.append(row["evidence_id"])
        return bool(row["evidence_id"] == "ev-eligible")

    monkeypatch.setattr(learning_evidence, "training_eligible", eligible)
    monkeypatch.setattr(personal_preferences, "list_explicit_preferences", lambda user_id, **_: [
        {"preference_id": "p-1", "field": "humor", "value": "dry", "scope": "ongoing",
         "applies_now": True, "created_at": START.isoformat(), "receipt": "dry humor works for me"},
        {"preference_id": "p-2", "field": "pace", "value": "slow", "scope": "ongoing",
         "applies_now": True, "created_at": START.isoformat(), "receipt": ""},
    ])
    source = _source(EchoSparkOpen())

    night_1 = source.extract_day(REAL, DAY, WINDOW)
    night_2 = source.extract_day(REAL, DAY + timedelta(days=1), (END, END + timedelta(days=1)))

    assert sorted(seen_rows) == ["ev-eligible", "ev-eligible", "ev-withheld", "ev-withheld"]
    assert sorted(item.source for item in night_1) == [
        "correction:ev-eligible", "preference:p-1", "preference:p-2",
    ]
    assert night_2 == []
    stored = {item.source: item for item in evidence_store(REAL).for_day(DAY)}
    assert stored["correction:ev-eligible"].text == "keep it short, please"
    assert stored["correction:ev-eligible"].ancestry == frozenset()
    assert stored["preference:p-1"].model is None and stored["preference:p-1"].ancestry == frozenset()
    # Without the person's quote the item is Anjo's parse of it: unknown ancestry, never trains.
    assert stored["preference:p-2"].ancestry is None


def test_a_synthetic_person_imports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core import personal_preferences
    from anjo.sleep.app import cleared_sessions

    cleared_sessions.register_synthetic(SYNTHETIC, at=START)
    monkeypatch.setattr(personal_preferences, "list_explicit_preferences", lambda user_id, **_: [
        {"preference_id": "p-film", "field": "humor", "value": "dry", "scope": "ongoing",
         "applies_now": True, "created_at": START.isoformat(), "receipt": "said in a filmed open"},
    ])

    assert _source(EchoSparkOpen()).extract_day(SYNTHETIC, DAY, WINDOW) == []


def test_extraction_records_target_mentions() -> None:
    from anjo.sleep.app.storage import target_store

    _message(REAL, "user", "I'll lock the cut by Friday.", at=START + timedelta(hours=9))

    def plan(messages: list[dict[str, Any]]) -> ChatResult:
        item = {"turn": 0, "kind": "lesson", "text": "They will lock the cut.", "target": {
            "kind": "commitment", "label": "lock the cut", "check_after": "2026-10-30",
            "existing_id": None, "confidence": None}}
        return ChatResult(text=json.dumps({"items": [item]}), tool_calls=(), model=SPARK_OPEN,
                          input_tokens=1, output_tokens=1, finish_reason="stop")

    from anjo.sleep.app.history_source import HistorySource

    [item] = HistorySource(chat=plan).extract_day(REAL, DAY, WINDOW)
    [target] = target_store(REAL).open_targets(END)

    assert item.target_id == target.id and target.kind == "commitment"
    assert target.check_after == datetime(2026, 10, 30, tzinfo=timezone.utc)


def test_extraction_needs_the_sleep_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app.flags import SleepHostOff

    monkeypatch.delenv("ANJO_SLEEP_HOST")
    chat = EchoSparkOpen()
    with pytest.raises(SleepHostOff):
        _source(chat).extract_day(REAL, DAY, WINDOW)
    assert chat.calls == []


def test_history_between_refuses_a_naive_window() -> None:
    from anjo.core.history import get_history_between

    with pytest.raises(ValueError, match="timezone"):
        get_history_between(REAL, START.replace(tzinfo=None), END)


def test_history_between_returns_ids_times_and_sessions_oldest_first() -> None:
    from anjo.core.history import get_history_between

    first = _message(REAL, "user", "one", at=START + timedelta(hours=1), session_id="s-a")
    second = _message(REAL, "assistant", "two", at=START + timedelta(hours=2), session_id="s-b")

    rows = get_history_between(REAL, START, END)
    assert [(row["id"], row["role"], row["content"], row["session_id"]) for row in rows] == [
        (first, "user", "one", "s-a"), (second, "assistant", "two", "s-b"),
    ]
    assert get_history_between(REAL, START, END, session_ids=frozenset({"s-b"}))[0]["id"] == second
    assert get_history_between("someone-else", START, END) == []
