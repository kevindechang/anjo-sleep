# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Evidence extraction (Task 3): Spark Open reads the person's own turns; a fake plays it here."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.evidence.extract import (
    PROMPT_PATH,
    ExtractionError,
    Imported,
    Turn,
    extract,
    held_out_count,
    ingest,
)
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.evidence.targets import EventLogTargetStore, target_id_for
from anjo.sleep.model_api import ChatResult
from anjo.sleep.types import Target

PERSON = "synthetic-film-student"
DAY = date(2026, 10, 28)
NOW = datetime(2026, 10, 29, 2, 0, tzinfo=timezone.utc)
SIMULATED = frozenset({SPARK_OPEN})


@dataclass
class FakeSparkOpen:
    """Answers with ``items`` and records every request."""

    items: list[dict[str, Any]] = field(default_factory=list)
    model: str = SPARK_OPEN
    reply: str | None = None
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]]) -> ChatResult:
        self.calls.append(messages)
        text = self.reply if self.reply is not None else json.dumps({"items": self.items})
        return ChatResult(
            text=text, tool_calls=(), model=self.model, input_tokens=100, output_tokens=20,
            finish_reason="stop",
        )

    def sent(self) -> str:
        return json.dumps(self.calls)


def _turns(*texts: str, ancestry: frozenset[str] | None = SIMULATED) -> list[Turn]:
    roles = ("user", "assistant")
    return [
        Turn(source=f"message:{n}", role=roles[n % 2], text=text, ancestry=ancestry)
        for n, text in enumerate(texts)
    ]


def _item(turn: int, kind: str, text: str, target: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"turn": turn, "kind": kind, "text": text, "target": target}


def _run(
    chat: FakeSparkOpen,
    turns: Sequence[Turn],
    *,
    imported: Sequence[Imported] = (),
    targets: Sequence[Target] = (),
    known_ids: frozenset[str] = frozenset(),
    day: date = DAY,
) -> Any:
    return extract(
        PERSON, day, turns, imported=imported, targets=targets, known_ids=known_ids, now=NOW,
        chat=chat,
    )


def test_the_prompt_is_versioned_with_the_code() -> None:
    assert PROMPT_PATH.name == "extract_v1.md"
    assert "JSON" in PROMPT_PATH.read_text(encoding="utf-8")


def test_extraction_reads_only_the_persons_turns() -> None:
    chat = FakeSparkOpen(items=[_item(0, "preference", "They prefer long takes.")])
    turns = _turns(
        "I like long takes.", "ASSISTANT-WORDS from Anjo's memory", "And natural light.",
    )

    extraction = _run(chat, turns)

    assert len(chat.calls) == 1
    sent = chat.sent()
    assert "I like long takes." in sent and "And natural light." in sent
    assert "ASSISTANT-WORDS" not in sent
    assert [item.source for item in extraction.items] == ["message:0"]


def test_the_extraction_call_is_spark_open_at_temperature_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep import baseten_api
    from anjo.sleep.evidence import extract as extract_module

    seen: dict[str, Any] = {}

    def fake_chat(model: str, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                  temperature: float | None = None) -> ChatResult:
        seen.update(model=model, tools=tools, temperature=temperature)
        return ChatResult(text='{"items": []}', tool_calls=(), model=model, input_tokens=1,
                          output_tokens=1, finish_reason="stop")

    monkeypatch.setattr(baseten_api, "chat", fake_chat)
    extract_module.spark_open([{"role": "user", "content": "x"}])
    assert seen == {"model": SPARK_OPEN, "tools": None, "temperature": 0.0}


def test_evidence_records_producing_model() -> None:
    chat = FakeSparkOpen(items=[_item(0, "lesson", "They shoot on 16mm.")])
    own = Imported(source="preference:p-1", kind="preference", text="no voiceover", ancestry=frozenset())

    extraction = _run(chat, _turns("I shoot on 16mm."), imported=[own])

    by_source = {item.source: item for item in extraction.items}
    assert by_source["message:0"].model == SPARK_OPEN
    assert by_source["preference:p-1"].model is None
    assert all(item.day == DAY and item.created_at == NOW for item in extraction.items)
    assert (extraction.input_tokens, extraction.output_tokens) == (100, 20)


def test_ancestry_is_transitive_and_unknown_is_refused(tmp_path: Path) -> None:
    hosted = "muse-wonder-1.0"
    chat = FakeSparkOpen(items=[_item(0, "lesson", "A real person's lesson.")])
    real = _run(chat, _turns("real words", ancestry=frozenset())).items[0]
    simulated = _run(chat, _turns("simulated words")).items[0]
    derived = _run(chat, _turns("words a hosted model wrote", ancestry=frozenset({hosted}))).items[0]
    unknown = _run(chat, _turns("words of unknown origin", ancestry=None)).items[0]

    assert real.ancestry == frozenset({SPARK_OPEN})
    assert simulated.ancestry == frozenset({SPARK_OPEN})
    assert derived.ancestry == frozenset({SPARK_OPEN, hosted})
    assert unknown.ancestry is None

    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON)
    store.add([
        real,
        _with_id(simulated, "e-sim"),
        _with_id(derived, "e-derived"),
        _with_id(unknown, "e-unknown"),
    ])
    assert [item.id for item in store.trainable(DAY)] == [real.id, "e-sim"]


def _with_id(item: Any, new_id: str) -> Any:
    from dataclasses import replace

    return replace(item, id=new_id)


def test_extract_holds_out_twenty_percent_of_preferences() -> None:
    items = [_item(0, "preference", f"They like film stock number {n}.") for n in range(10)]
    items.append(_item(0, "lesson", "They teach a night class."))
    chat = FakeSparkOpen(items=items)

    first = _run(chat, _turns("many preferences"))
    again = _run(chat, _turns("many preferences"))
    other_day = _run(chat, _turns("many preferences"), day=date(2026, 10, 30))

    held = [item for item in first.items if item.held_out]
    assert len(held) == 2 and all(item.kind == "preference" for item in held)
    assert [item.held_out for item in again.items] == [item.held_out for item in first.items]
    assert sum(item.held_out for item in other_day.items) == 2
    assert [held_out_count(n) for n in (0, 1, 2, 3, 5, 8, 10)] == [0, 0, 0, 1, 1, 2, 2]


def test_held_out_items_never_touch_targets() -> None:
    target = {"kind": "thread", "label": "color grading", "check_after": None, "existing_id": None,
              "confidence": 0.3}
    items = [_item(0, "preference", f"They like grade {n}.", target) for n in range(5)]
    extraction = _run(FakeSparkOpen(items=items), _turns("grades"))

    held = [item for item in extraction.items if item.held_out]
    assert len(held) == 1 and held[0].target_id is None
    assert held[0].id not in {evidence_id for evidence_id, _ in extraction.mentions}
    assert len(extraction.mentions) == 4


def test_a_night_with_held_out_mentions_ingests_cleanly(tmp_path: Path) -> None:
    """The held-out guard in record_mention never fires on extraction's own output."""
    evidence = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON)
    targets = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON)
    topic = {"kind": "thread", "label": "color grading", "check_after": None, "existing_id": None,
             "confidence": 0.3}
    items = [_item(0, "preference", f"They like grade {n}.", topic) for n in range(10)]

    added = ingest(_run(FakeSparkOpen(items=items), _turns("grades")), evidence, targets)

    held = [item for item in added if item.held_out]
    assert len(held) == 2 and all(item.target_id is None for item in held)
    assert [item.id for item in evidence.held_out(DAY)] == [item.id for item in held]
    [target] = targets.open_targets(NOW)
    assert target.revision == 8  # created by the first kept mention, raised by the other seven


def test_target_ids_stable_across_nights(tmp_path: Path) -> None:
    evidence = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON)
    targets = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON)
    plan = {"kind": "commitment", "label": "Festival submission", "check_after": "2026-11-03",
            "existing_id": None, "confidence": None}

    night_1 = _run(FakeSparkOpen(items=[_item(0, "lesson", "They will submit.", plan)]),
                   _turns("I'll submit to the festival by Nov 3."))
    ingest(night_1, evidence, targets)
    [created] = targets.open_targets(NOW)
    assert created.id == target_id_for(PERSON, "commitment", "festival submission")
    assert created.check_after == datetime(2026, 11, 3, tzinfo=timezone.utc)

    # Night 2: the model matches the mention to the open target it was shown.
    matched = {**plan, "label": "submitting the film", "existing_id": created.id}
    night_2 = _run(FakeSparkOpen(items=[_item(0, "lesson", "Still submitting.", matched)]),
                   _turns("Still on for the festival."), targets=[created], day=date(2026, 10, 29))
    ingest(night_2, evidence, targets)
    # Night 3: no id from the model, but the same plan by name.
    night_3 = _run(FakeSparkOpen(items=[_item(0, "lesson", "Submission is close.", plan)]),
                   _turns("Festival submission is close."), day=date(2026, 10, 30))
    ingest(night_3, evidence, targets)

    [same] = targets.open_targets(NOW)
    assert same.id == created.id and same.revision == 3
    assert {item.target_id for item in evidence.stored(date(2026, 10, 30))} == {created.id}


def test_an_unknown_or_mismatched_existing_id_is_not_trusted() -> None:
    known = Target(
        id="t-known", person_id=PERSON, kind="thread", label="sound design", check_after=None,
        status="open", revision=1, acted_revision=0, confidence=0.5, last_mentioned=DAY,
    )
    wrong_kind = {"kind": "commitment", "label": "mix by Friday", "check_after": "2026-10-31",
                  "existing_id": "t-known", "confidence": None}
    invented = {"kind": "thread", "label": "lenses", "check_after": None,
                "existing_id": "t-invented", "confidence": 0.2}
    extraction = _run(
        FakeSparkOpen(items=[_item(0, "lesson", "Mix due.", wrong_kind),
                             _item(0, "lesson", "Lenses.", invented)]),
        _turns("mix by Friday, and lenses"), targets=[known],
    )

    assert [mention.target_id for _, mention in extraction.mentions] == [
        target_id_for(PERSON, "commitment", "mix by Friday"),
        target_id_for(PERSON, "thread", "lenses"),
    ]


def test_imported_items_enter_once() -> None:
    own = Imported(source="correction:c-1", kind="correction", text="not that", ancestry=frozenset())
    first = _run(FakeSparkOpen(), [], imported=[own])
    second = _run(FakeSparkOpen(), [], imported=[own], known_ids=frozenset(i.id for i in first.items),
                  day=date(2026, 10, 29))

    assert [item.source for item in first.items] == ["correction:c-1"]
    assert second.items == ()


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    evidence = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON)
    targets = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON)
    topic = {"kind": "thread", "label": "pacing", "check_after": None, "existing_id": None,
             "confidence": 0.4}
    extraction = _run(FakeSparkOpen(items=[_item(0, "lesson", "Pacing drags.", topic)]),
                      _turns("the pacing drags"))

    added = ingest(extraction, evidence, targets)
    again = ingest(extraction, evidence, targets)

    assert len(added) == 1 and again == []
    [target] = targets.open_targets(NOW)
    assert target.revision == 1


def test_an_empty_day_makes_no_model_call() -> None:
    chat = FakeSparkOpen()
    extraction = _run(chat, _turns())
    assert chat.calls == [] and extraction.items == ()
    only_her = [Turn(source="message:1", role="assistant", text="hi", ancestry=SIMULATED)]
    assert _run(chat, only_her).items == () and chat.calls == []


@pytest.mark.parametrize("reply", ["not json", '{"items": "none"}', "[]"])
def test_a_reply_that_is_not_the_asked_json_is_refused(reply: str) -> None:
    with pytest.raises(ExtractionError):
        _run(FakeSparkOpen(reply=reply), _turns("something"))


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        ({"turn": 7, "kind": "lesson", "text": "x"}, "turn 7"),
        ({"turn": 0, "kind": "homework", "text": "x"}, "kind 'homework'"),
        ({"turn": 0, "kind": "lesson", "text": " "}, "empty text"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": {
            "kind": "commitment", "label": "a plan", "check_after": None}}, "needs the date"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": {
            "kind": "thread", "label": "t", "check_after": None, "confidence": 1.5}}, "confidence"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": {
            "kind": "commitment", "label": "a plan", "check_after": "soon"}}, "isoformat"),
    ],
)
def test_an_item_that_breaks_the_contract_is_listed_not_kept(item: dict[str, Any], reason: str) -> None:
    good = _item(0, "lesson", "They cut on paper first.")
    extraction = _run(FakeSparkOpen(items=[item, good]), _turns("something"))

    assert [kept.text for kept in extraction.items] == ["They cut on paper first."]
    [refusal] = extraction.refused
    assert refusal.startswith("item 0:") and reason in refusal


def test_a_fenced_json_reply_is_read() -> None:
    reply = '```json\n{"items": [{"turn": 0, "kind": "tone_request", "text": "Be brief.", "target": null}]}\n```'
    extraction = _run(FakeSparkOpen(reply=reply), _turns("be brief"))
    assert [item.kind for item in extraction.items] == ["tone_request"]
