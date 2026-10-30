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
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
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
    model: Any = SPARK_OPEN  # Any: a test passes None, as a provider might
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
                  temperature: float | None = None, retries: Any = None) -> ChatResult:
        seen.update(model=model, tools=tools, temperature=temperature, timeout_s=timeout_s, retries=retries)
        return ChatResult(text='{"items": []}', tool_calls=(), model=model, input_tokens=1,
                          output_tokens=1, finish_reason="stop")

    monkeypatch.setattr(baseten_api, "chat", fake_chat)
    extract_module.spark_open([{"role": "user", "content": "x"}])
    retries = seen.pop("retries")
    assert seen == {"model": SPARK_OPEN, "tools": None, "temperature": 0.0, "timeout_s": 300.0}
    # One lost reply there fails the whole night (2026-10-30), so it is sent once more.
    assert retries is extract_module.EXTRACT_RETRIES
    assert extract_module.TIMEOUT_S == 300.0
    assert (retries.connect_attempts, retries.reply_attempts) == (3, 2)


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

    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
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
    evidence = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    targets = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
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
    evidence = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    targets = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
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
    evidence = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    targets = EventLogTargetStore(tmp_path / "targets.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
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


def _thread(confidence: Any) -> dict[str, Any]:
    return {"kind": "thread", "label": "t", "check_after": None, "confidence": confidence}


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        ("INJECTED words", "not an object"),
        ({"turn": 0, "text": "x"}, "missing field"),
        ({"turn": "INJECTED turn", "kind": "lesson", "text": "x"}, "turn out of range"),
        ({"turn": 7, "kind": "lesson", "text": "x"}, "turn out of range"),
        ({"turn": 0, "kind": "INJECTED kind", "text": "x"}, "kind not allowed"),
        ({"turn": 0, "kind": ["INJECTED"], "text": "x"}, "kind not allowed"),
        ({"turn": 0, "kind": "lesson", "text": " "}, "empty text"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": "INJECTED"}, "target not an object"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": {
            "kind": "INJECTED kind", "label": "a plan"}}, "target kind not allowed"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": {
            "kind": "commitment", "label": "a plan", "check_after": None}}, "commitment without a date"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": {
            "kind": "thread", "label": "t", "check_after": "2026-11-01"}}, "thread with a date"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": _thread("INJECTED")}, "confidence not a number"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": _thread(1.5)}, "confidence outside 0 to 1"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": _thread(-0.1)}, "confidence outside 0 to 1"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": _thread(float("nan"))},
         "confidence outside 0 to 1"),
        ({"turn": 0, "kind": "lesson", "text": "x", "target": {
            "kind": "commitment", "label": "a plan", "check_after": "INJECTED date"}},
         "check_after is not a date"),
    ],
)
def test_an_item_that_breaks_the_contract_is_listed_not_kept(item: Any, reason: str) -> None:
    good = _item(0, "lesson", "They cut on paper first.")
    extraction = _run(FakeSparkOpen(items=[item, good]), _turns("something"))

    assert [kept.text for kept in extraction.items] == ["They cut on paper first."]
    # A refusal is a fixed category; nothing the model wrote reaches it, or the logs.
    assert extraction.refused == (f"item 0: {reason}",)


def test_an_items_ancestry_covers_every_turn_the_call_read() -> None:
    hosted = "muse-wonder-1.0"
    real = Turn(source="message:0", role="user", text="real words", ancestry=frozenset())
    derived = Turn(source="message:2", role="user", text="a hosted draft", ancestry=frozenset({hosted}))
    unknown = Turn(source="message:4", role="user", text="pasted", ancestry=None)
    reply = Turn(source="message:1", role="assistant", text="her reply", ancestry=None)
    chat = FakeSparkOpen(items=[_item(0, "lesson", "From the real turn.")])

    [mixed] = _run(chat, [real, reply, derived]).items
    [poisoned] = _run(chat, [real, derived, unknown]).items

    # Her reply isn't read, so its ancestry doesn't count; every own turn's does.
    assert mixed.ancestry == frozenset({SPARK_OPEN, hosted})
    assert poisoned.ancestry is None


@pytest.mark.parametrize("model", [None, "", "another-model"])
def test_a_reply_from_an_unknown_model_is_refused(model: Any) -> None:
    with pytest.raises(ExtractionError, match="model"):
        _run(FakeSparkOpen(items=[_item(0, "lesson", "x")], model=model), _turns("something"))


def test_ids_come_from_the_turn_not_the_models_words() -> None:
    first = _run(FakeSparkOpen(items=[
        _item(0, "lesson", "They cut on paper."), _item(0, "lesson", "They log every take."),
        _item(0, "preference", "They like 16mm."),
    ]), _turns("words"))
    reworded = _run(FakeSparkOpen(items=[
        _item(0, "lesson", "Paper edits first."), _item(0, "lesson", "Every take is logged."),
        _item(0, "preference", "16mm, please."),
    ]), _turns("words"))

    assert len({item.id for item in first.items}) == 3
    assert [item.id for item in reworded.items] == [item.id for item in first.items]


@pytest.mark.parametrize("ancestry", ["Spark Open", {SPARK_OPEN}, [SPARK_OPEN], frozenset({1}), frozenset({""})])
def test_imported_and_turn_ancestry_must_be_a_frozenset_of_names_or_none(ancestry: Any) -> None:
    with pytest.raises(TypeError, match="ancestry"):
        Imported(source="preference:p-1", kind="preference", text="x", ancestry=ancestry)
    with pytest.raises(TypeError, match="ancestry"):
        Turn(source="message:0", role="user", text="x", ancestry=ancestry)


def test_imported_ancestry_is_kept_as_given() -> None:
    hosted = "muse-wonder-1.0"
    own = Imported(source="correction:c-1", kind="correction", text="not that", ancestry=frozenset())
    parsed = Imported(source="preference:p-1", kind="preference", text="pace: slow", ancestry=None)
    drafted = Imported(source="preference:p-2", kind="preference", text="x", ancestry=frozenset({hosted}))

    items = {item.source: item for item in _run(FakeSparkOpen(), [], imported=[own, parsed, drafted]).items}

    assert items["correction:c-1"].ancestry == frozenset()
    assert items["preference:p-1"].ancestry is None
    assert items["preference:p-2"].ancestry == frozenset({hosted})


def test_the_extraction_log_keeps_the_first_commit_per_window_and_input(tmp_path: Path) -> None:
    from datetime import timedelta

    from anjo.sleep.evidence.extractions import ExtractionLog, read_digest

    log = ExtractionLog(tmp_path / "extractions.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    window = (NOW - timedelta(days=1), NOW)
    same_instant = (window[0].astimezone(timezone(timedelta(hours=8))), window[1])
    read = ["message:1", "message:3"]

    assert log.committed(DAY, window, read) is None
    log.commit(DAY, window, read, ["e-1", "e-2"])
    log.commit(DAY, same_instant, list(reversed(read)), ["e-3"])

    assert log.committed(DAY, same_instant, list(reversed(read))) == ("e-1", "e-2")
    # A newly cleared (or forgotten) turn is a new input: the day is read again.
    assert log.committed(DAY, window, [*read, "message:5"]) is None
    assert log.committed(DAY, (window[0], NOW + timedelta(hours=1)), read) is None
    assert log.committed(date(2026, 10, 29), window, read) is None
    assert log.export() == {"extractions": [{
        "day": "2026-10-28", "start": "2026-10-28T02:00:00+00:00", "end": "2026-10-29T02:00:00+00:00",
        "read": read_digest(read), "item_ids": ["e-1", "e-2"],
    }]}
    with pytest.raises(ValueError, match="aware"):
        log.committed(DAY, (datetime(2026, 10, 28), NOW), read)


def test_a_fenced_json_reply_is_read() -> None:
    reply = '```json\n{"items": [{"turn": 0, "kind": "tone_request", "text": "Be brief.", "target": null}]}\n```'
    extraction = _run(FakeSparkOpen(reply=reply), _turns("be brief"))
    assert [item.kind for item in extraction.items] == ["tone_request"]
