# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Context C and recall on private Anjo (Task 2): memory, preferences since the brief, MiniLM."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from anjo.sleep import ports
from anjo.sleep.context.builder import ComposedContextBuilder, RecallSettings, StoredBrief
from anjo.sleep.tests.sleep_fakes import LeakyStore, StoreSpy, ev, probe

PERSON = "person-real"
DAY = date(2026, 10, 28)
T0 = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)


def _vector(text: str) -> list[float]:
    """A tiny stand-in for MiniLM: a unit vector over three topics."""
    parts = [float("ferry" in text), float("sound" in text), float("cut" in text)]
    norm = sum(p * p for p in parts) ** 0.5 or 1.0
    return [p / norm for p in parts]


def _correction_row(evidence_id: str, quote: str, at: datetime) -> None:
    from anjo.core.crypto import encrypt_db
    from anjo.core.db import get_db

    payload = {"patch": {"field": "verbosity", "value": "short", "scope": "ongoing",
                         "operation": "set"}, "quote": quote}
    db = get_db()
    db.execute(
        "INSERT INTO learning_evidence (user_id, evidence_id, event_key, relationship_epoch, "
        "consent_generation, revocation_generation, kind, payload_encrypted, created_at) "
        "VALUES (?, ?, ?, 0, 0, 0, 'explicit_preference', ?, ?)",
        (PERSON, evidence_id, f"key-{evidence_id}", encrypt_db(json.dumps(payload)), at.isoformat()),
    )
    db.commit()


class _Brief:
    def served(self, user_id: str) -> StoredBrief:
        return StoredBrief("brief text", T0)


def _builder(
    monkeypatch: pytest.MonkeyPatch, prefs: list[dict[str, Any]], eligible: set[str],
) -> ComposedContextBuilder:
    from anjo.core import learning_evidence, personal_preferences
    from anjo.memory import long_term
    from anjo.sleep.app.context_builder import anjo_context_builder

    monkeypatch.setattr(personal_preferences, "list_explicit_preferences", lambda user_id, **_: prefs)
    monkeypatch.setattr(learning_evidence, "training_eligible", lambda row: row["evidence_id"] in eligible)
    monkeypatch.setattr(long_term, "query_memories", lambda message, user_id, n_results=4: [(0.9, "the ferry short")])
    store = StoreSpy([ev("a", "ferry deck long take", minutes=1), ev("b", "sound mix", minutes=2)])
    return anjo_context_builder(
        _Brief(), until=DAY, settings=RecallSettings(k=1, budget_tokens=1000),
        evidence=lambda user_id: store, embed=_vector,
    )


def test_the_builder_is_a_context_builder_port(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(_builder(monkeypatch, [], set()), ports.ContextBuilder)


def test_same_day_tone_request_reaches_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _correction_row("ev-new", "be blunt with me", T0 + timedelta(hours=2))
    _correction_row("ev-old", "already in the brief", T0 - timedelta(hours=2))
    _correction_row("ev-withheld", "WITHHELD", T0 + timedelta(hours=3))
    prefs = [
        {"preference_id": "p-1", "field": "humor", "value": "dry", "scope": "ongoing", "applies_now": True,
         "created_at": (T0 + timedelta(hours=1)).isoformat(), "receipt": "dry humor works for me"},
        {"preference_id": "p-2", "field": "pace", "value": "slow", "scope": "ongoing", "applies_now": True,
         "created_at": (T0 - timedelta(hours=1)).isoformat(), "receipt": "folded already"},
    ]
    builder = _builder(monkeypatch, prefs, {"ev-new", "ev-old"})
    text = builder.build(PERSON, probe(person=PERSON, prompt="how should the cut come on the ferry"))
    assert "be blunt with me" in text and "dry humor works for me" in text
    assert "already in the brief" not in text and "folded already" not in text
    assert "WITHHELD" not in text
    assert text.index("dry humor") < text.index("be blunt")  # oldest first
    assert "- the ferry short" in text and "brief text" in text


def test_a_preference_without_a_quote_reads_as_field_and_value(monkeypatch: pytest.MonkeyPatch) -> None:
    prefs = [{"preference_id": "p-1", "field": "pace", "value": "slow", "scope": "ongoing",
              "applies_now": True, "created_at": (T0 + timedelta(hours=1)).isoformat(), "receipt": ""}]
    assert "pace: slow" in _builder(monkeypatch, prefs, set()).build(PERSON, probe(person=PERSON))


def test_minilm_recall_ranks_by_cosine_and_never_returns_held_out() -> None:
    from anjo.sleep.app.recall import MiniLmRecall

    store = LeakyStore([
        ev("a", "ferry deck long take", minutes=1), ev("b", "sound mix", minutes=2),
        ev("held", "ferry ferry ferry", held_out=True, minutes=3),
    ])
    recall = MiniLmRecall(lambda u: store, until=DAY, embed=_vector)
    got = recall.recall(PERSON, "ferry question", k=5, budget_tokens=1000)
    assert [e.id for e in got] == ["a", "b"]


def test_minilm_recall_respects_the_token_budget_and_sft_mode() -> None:
    from anjo.sleep.app.recall import MiniLmRecall
    from anjo.sleep.context.recall_common import estimate_tokens
    from anjo.sleep.tests.sleep_fakes import SPARK

    items = [ev(f"i{n}", "ferry " + "word " * 40, minutes=n) for n in range(6)]
    items += [ev("prac", "ferry practice", kind="practice", ancestry=frozenset({SPARK}), minutes=9),
              ev("unk", "ferry unknown", ancestry=None, minutes=10)]
    store = StoreSpy(items)
    recall = MiniLmRecall(lambda u: store, until=DAY, embed=_vector)
    one = estimate_tokens(items[0].text)
    got = recall.recall(PERSON, "ferry", k=10, budget_tokens=one * 2)
    assert got and sum(estimate_tokens(e.text) for e in got) <= one * 2
    sft = {e.id for e in recall.recall(PERSON, "ferry", k=20, budget_tokens=10_000, mode="sft")}
    assert "prac" not in sft and "unk" not in sft and "i0" in sft


def test_the_builder_refuses_to_run_off_the_sleep_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app.context_builder import anjo_context_builder
    from anjo.sleep.app.flags import SleepHostOff

    monkeypatch.delenv("ANJO_SLEEP_HOST")
    with pytest.raises(SleepHostOff):
        anjo_context_builder(_Brief(), until=DAY, settings=RecallSettings(k=1, budget_tokens=10))
