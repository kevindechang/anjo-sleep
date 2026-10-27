"""Evidence extraction for one day (Task 3).

Spark Open reads the person's own turns inside the day's window with ``prompts/extract_v1.md``
at temperature 0 and lists what Anjo can learn from them, with any target each item mentions: a
plan the person dated, or a topic she was lost on or that stays open. Her own turns are never
sent: they carry Anjo's untagged memory in their context (§3).

- **Provenance:** an extracted item's ``model`` is the model that answered, and its ancestry is
  that model plus the ancestry of the turn it came from: empty for a real person's words,
  {Spark Open} for a simulated person's, unknown (``None``) when the turn's is unknown.
- **Imported items:** the person's own stored corrections and preferences, read by the app.
  Their ids come from their source alone, so each enters the store once across nights.
- **Hold-out:** 20% of each day's new preferences (rounded half up) are held out for the sealed
  tests, chosen by a hash seeded with the person and the day. A held-out item never names a
  target, so nothing held out reaches her through the chooser.
- **Refusals:** a reply that isn't the JSON asked for raises ``ExtractionError``. A single item
  that breaks the contract is left out and listed in ``Extraction.refused``, never dropped
  silently.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, get_args

from anjo.sleep import baseten_api
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.evidence.targets import (
    PERSON_KINDS,
    EventLogTargetStore,
    Mention,
    target_id_for,
)
from anjo.sleep.model_api import ChatResult
from anjo.sleep.types import Evidence, EvidenceKind, Target, TargetKind

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "extract_v1.md"
TIMEOUT_S: Final = 120.0
_TARGET_KINDS = frozenset(get_args(TargetKind))

ChatFn = Callable[[list[dict[str, Any]]], ChatResult]


def spark_open(messages: list[dict[str, Any]]) -> ChatResult:
    """The extraction call: Spark Open on Baseten at temperature 0, no tools."""
    return baseten_api.chat(SPARK_OPEN, messages, None, TIMEOUT_S, temperature=0.0)


class ExtractionError(ValueError):
    """The model's reply isn't the JSON the prompt asks for."""


@dataclass(frozen=True)
class Turn:
    """One stored turn. ``ancestry`` is the turn text's own: empty for a real person's words."""

    source: str
    role: str
    text: str
    ancestry: frozenset[str] | None


@dataclass(frozen=True)
class Imported:
    """One of the person's own stored corrections or preferences, in their words."""

    source: str
    kind: EvidenceKind
    text: str
    ancestry: frozenset[str] | None


@dataclass(frozen=True)
class Extraction:
    """A day's new items, the target each mentions (by evidence id), and the call's tokens."""

    items: tuple[Evidence, ...]
    mentions: tuple[tuple[str, Mention], ...]
    input_tokens: int
    output_tokens: int
    refused: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Found:
    turn: int
    kind: EvidenceKind
    text: str
    target: Mapping[str, Any] | None


def held_out_count(preferences: int) -> int:
    """20% of a day's new preferences, rounded half up."""
    return (2 * preferences + 5) // 10


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def extract(
    person_id: str,
    day: date,
    turns: Sequence[Turn],
    *,
    imported: Sequence[Imported],
    targets: Sequence[Target],
    known_ids: frozenset[str],
    now: datetime,
    chat: ChatFn = spark_open,
) -> Extraction:
    """The day's new evidence, stamped with ``day`` and created at ``now`` (the night's clock)."""
    own = [turn for turn in turns if turn.role == "user"]
    found: list[tuple[Evidence, Mention | None]] = []
    refused: list[str] = []
    input_tokens = output_tokens = 0
    if own:
        reply = chat(_messages(day, own, targets))
        input_tokens, output_tokens = reply.input_tokens, reply.output_tokens
        shown = {target.id: target for target in targets}
        for index, raw in enumerate(_items(reply.text)):
            try:
                hit = _found(raw, len(own))
                mention = None if hit.target is None else _mention(hit.target, person_id, shown)
            except (KeyError, TypeError, ValueError) as exc:
                refused.append(f"item {index}: {exc}")
                continue
            turn = own[hit.turn]
            ancestry = None if turn.ancestry is None else turn.ancestry | {reply.model}
            found.append((Evidence(
                id="e-" + _hash(person_id, day.isoformat(), turn.source, hit.kind, hit.text)[:20],
                person_id=person_id, day=day, kind=hit.kind, text=hit.text, source=turn.source,
                held_out=False, created_at=now, model=reply.model, ancestry=ancestry,
            ), mention))
    for item in imported:
        found.append((Evidence(
            id="e-" + _hash(person_id, "imported", item.source)[:20], person_id=person_id, day=day,
            kind=item.kind, text=item.text, source=item.source, held_out=False, created_at=now,
            model=None, ancestry=item.ancestry,
        ), None))
    return _finish(person_id, day, found, known_ids, input_tokens, output_tokens, refused)


def _finish(
    person_id: str,
    day: date,
    found: list[tuple[Evidence, Mention | None]],
    known_ids: frozenset[str],
    input_tokens: int,
    output_tokens: int,
    refused: list[str],
) -> Extraction:
    seen = set(known_ids)
    fresh: list[tuple[Evidence, Mention | None]] = []
    for item, mention in found:
        if item.id not in seen:
            seen.add(item.id)
            fresh.append((item, mention))
    preferences = sorted(
        (item.id for item, _ in fresh if item.kind == "preference"),
        key=lambda item_id: _hash(person_id, day.isoformat(), item_id),
    )
    held = frozenset(preferences[: held_out_count(len(preferences))])
    items: list[Evidence] = []
    mentions: list[tuple[str, Mention]] = []
    for item, mention in fresh:
        if item.id in held:
            items.append(replace(item, held_out=True))
            continue
        if mention is not None and item.kind in PERSON_KINDS:
            item = replace(item, target_id=mention.target_id)
            mentions.append((item.id, mention))
        items.append(item)
    return Extraction(tuple(items), tuple(mentions), input_tokens, output_tokens, tuple(refused))


def _messages(day: date, own: Sequence[Turn], targets: Sequence[Target]) -> list[dict[str, Any]]:
    payload = {
        "today": day.isoformat(),
        "open_targets": [
            {
                "id": target.id, "kind": target.kind, "label": target.label,
                "check_after": None if target.check_after is None else target.check_after.date().isoformat(),
            }
            for target in targets
        ],
        "messages": [{"turn": index, "text": turn.text} for index, turn in enumerate(own)],
    }
    return [
        {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _items(text: str) -> list[Any]:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        payload = json.loads(body)
    except ValueError:
        raise ExtractionError("the extraction reply is not JSON") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ExtractionError("the extraction reply has no items list")
    return list(payload["items"])


def _found(raw: object, turn_count: int) -> _Found:
    if not isinstance(raw, dict):
        raise TypeError("not an object")
    turn, kind, text, target = raw["turn"], raw["kind"], raw["text"], raw.get("target")
    if not isinstance(turn, int) or isinstance(turn, bool) or not 0 <= turn < turn_count:
        raise ValueError(f"turn {turn!r} is not one of the {turn_count} messages")
    if kind not in PERSON_KINDS:
        raise ValueError(f"kind {kind!r} is not one the person's turns can give")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty text")
    if target is not None and not isinstance(target, dict):
        raise TypeError("target is not an object")
    return _Found(turn=turn, kind=kind, text=text.strip(), target=target)


def _mention(raw: Mapping[str, Any], person_id: str, shown: Mapping[str, Target]) -> Mention:
    kind, label, check_after = raw["kind"], raw["label"], raw.get("check_after")
    if kind not in _TARGET_KINDS:
        raise ValueError(f"target kind {kind!r}")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("target without a label")
    if kind == "commitment" and check_after is None:
        raise ValueError("a commitment needs the date the person gave")
    if kind == "thread" and check_after is not None:
        raise ValueError("a thread has no date")
    confidence = raw.get("confidence") if kind == "thread" else None
    if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)):
        raise TypeError("confidence is not a number")
    existing = shown.get(raw.get("existing_id") or "")
    matched = existing is not None and existing.kind == kind
    return Mention(  # its own checks raise TargetStoreError, a ValueError
        target_id=existing.id if existing is not None and matched else target_id_for(person_id, kind, label),
        kind=kind, label=label.strip(), check_after=_date(check_after),
        confidence=None if confidence is None else float(confidence),
    )


def _date(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("check_after is not a string")
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def ingest(
    extraction: Extraction, evidence: JsonlEvidenceStore, targets: EventLogTargetStore,
) -> list[Evidence]:
    """Record a day's extraction: count each new item's mention, then store the new items.
    Mentions go first and count each evidence id once, so a re-run after a crash between the
    two steps neither loses an item nor raises a revision twice. Returns the items it added."""
    known = evidence.ids()
    fresh = [item for item in extraction.items if item.id not in known]
    by_id = {item.id: item for item in fresh}
    for evidence_id, mention in extraction.mentions:
        item = by_id.get(evidence_id)
        if item is not None:
            targets.record_mention(mention, item)
    evidence.add(fresh)
    return fresh
