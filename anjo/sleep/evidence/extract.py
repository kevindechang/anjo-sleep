"""Evidence extraction for one day (Task 3).

Spark Open reads the person's own turns inside the day's window with ``prompts/extract_v1.md``
at temperature 0 and lists what Anjo can learn from them, with any target each item mentions: a
plan the person dated, or a topic she was lost on or that stays open. Her own turns are never
sent: they carry Anjo's untagged memory in their context (§3).

- **Provenance:** an extracted item's ``model`` is Spark Open's id on the provider
  ``ANJO_SLEEP_SPARK`` selects (Baseten's by default; APD's lives pass their provider's), and a
  reply that names any other id is refused. Its ancestry is Spark Open plus the ancestry of every
  turn the call read, since the model saw them all: empty for a real person's words, {Spark Open}
  for a simulated person's, unknown (``None``) when any read turn's is unknown.
- **Ids:** an extracted item's id comes from the person, the day, its source turn, its kind and
  its place among that turn's items of that kind, never from the model's wording, so a re-run
  whose reply words an item differently names the same item. ``ExtractionLog`` makes a committed
  day's re-run skip the model altogether.
- **Imported items:** the person's own stored corrections and preferences, read by the app, with
  the ancestry it gives (empty for the person's own words). Their ids come from their source
  alone, so each enters the store once across nights.
- **Hold-out:** 20% of each day's new preferences (rounded half up) are held out for the sealed
  tests, chosen by a hash seeded with the person and the day. Extraction strips the target from
  an item it holds out and emits no mention for it, so nothing held out reaches her through the
  chooser. ``record_mention`` still refuses a held-out item, but only as a guard: on
  extraction's own output it never fires.
- **Refusals:** a reply that isn't the JSON asked for raises ``ExtractionError``. A single item
  that breaks the contract is left out and listed in ``Extraction.refused`` as ``item <index>:
  <category>``, never dropped silently. The category is fixed text, so nothing the model wrote
  (which can carry the person's words) reaches a log through it.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, TypeGuard, get_args

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


def _is_target_kind(value: object) -> TypeGuard[TargetKind]:
    return isinstance(value, str) and value in _TARGET_KINDS

ChatFn = Callable[[list[dict[str, Any]]], ChatResult]


def spark_open(messages: list[dict[str, Any]]) -> ChatResult:
    """The extraction call: Spark Open at temperature 0, no tools, through ``baseten_api.chat``'s
    provider switch."""
    return baseten_api.chat(SPARK_OPEN, messages, None, TIMEOUT_S, temperature=0.0)


def _accepted_models(configured: str) -> frozenset[str]:
    """The ids a reply may name: the one ``chat`` is configured with, and the one
    ``ANJO_SLEEP_SPARK`` selects. Both are the cleared Spark Open (``provenance``)."""
    from anjo.sleep import spark_provider  # imported here: spark_provider imports this package

    return frozenset({configured, spark_provider.current().model})


def _selected_model() -> str:
    """The Spark Open id the provider ``ANJO_SLEEP_SPARK`` selects serves (Baseten's by default)."""
    from anjo.sleep import spark_provider  # imported here: spark_provider imports this package

    return spark_provider.current().model


class ExtractionError(ValueError):
    """The model's reply isn't the JSON the prompt asks for."""


def _check_ancestry(ancestry: object) -> None:
    """None, or a frozenset of model names. A string or a mutable set is refused."""
    if ancestry is None:
        return
    if not isinstance(ancestry, frozenset) or not all(isinstance(name, str) and name for name in ancestry):
        raise TypeError("ancestry must be a frozenset of model names, or None")


@dataclass(frozen=True)
class Turn:
    """One stored turn. ``ancestry`` is the turn text's own: empty for a real person's words."""

    source: str
    role: str
    text: str
    ancestry: frozenset[str] | None

    def __post_init__(self) -> None:
        _check_ancestry(self.ancestry)


@dataclass(frozen=True)
class Imported:
    """One of the person's stored corrections or preferences. ``ancestry`` is empty only when
    ``text`` is the person's own words; Anjo's parse of them is unknown (``None``)."""

    source: str
    kind: EvidenceKind
    text: str
    ancestry: frozenset[str] | None

    def __post_init__(self) -> None:
        _check_ancestry(self.ancestry)


@dataclass(frozen=True)
class Extraction:
    """A day's new items, the target each mentions (by evidence id), and the call's tokens."""

    items: tuple[Evidence, ...]
    mentions: tuple[tuple[str, Mention], ...]
    input_tokens: int
    output_tokens: int
    refused: tuple[str, ...] = ()


class _Refused(ValueError):
    """One item's broken rule. The message is a fixed category, never a value from the reply."""


@dataclass(frozen=True)
class _Found:
    turn: int
    kind: EvidenceKind
    text: str
    target: Mapping[str, Any] | None


def own_turns(turns: Sequence[Turn]) -> list[Turn]:
    """The turns extraction reads: the person's own, never hers."""
    return [turn for turn in turns if turn.role == "user"]


def held_out_count(preferences: int) -> int:
    """20% of a day's new preferences, rounded half up."""
    return (2 * preferences + 5) // 10


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _reason(exc: Exception) -> str:
    """The refusal category: fixed text for a rule, the exception type otherwise."""
    if isinstance(exc, _Refused):
        return str(exc)
    if isinstance(exc, KeyError):
        return "missing field"
    return type(exc).__name__


def _read_ancestry(own: Sequence[Turn]) -> frozenset[str] | None:
    """The ancestry of everything the call read: unknown if any turn's is."""
    names: set[str] = set()
    for turn in own:
        if turn.ancestry is None:
            return None
        names |= turn.ancestry
    return frozenset(names)


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
    model: str | None = None,
) -> Extraction:
    """The day's new evidence, stamped with ``day`` and created at ``now`` (the night's clock).
    ``model`` is the Spark Open id ``chat`` serves: the provider ``ANJO_SLEEP_SPARK`` selects when
    not given, the experiment's provider's for APD's lives. A reply naming either the configured id
    or the selected one is accepted: both are the same cleared model (``provenance``)."""
    served = _selected_model() if model is None else model
    accepted = _accepted_models(served)
    own = own_turns(turns)
    found: list[tuple[Evidence, Mention | None]] = []
    refused: list[str] = []
    input_tokens = output_tokens = 0
    if own:
        reply = chat(_messages(day, own, targets))
        if reply.model not in accepted:
            raise ExtractionError("the extraction reply doesn't name the extraction model")
        input_tokens, output_tokens = reply.input_tokens, reply.output_tokens
        found, refused = _read_reply(person_id, day, own, targets, now, reply.text, served)
    for item in imported:
        found.append((Evidence(
            id="e-" + _hash(person_id, "imported", item.source)[:20], person_id=person_id, day=day,
            kind=item.kind, text=item.text, source=item.source, held_out=False, created_at=now,
            model=None, ancestry=item.ancestry,
        ), None))
    return _finish(person_id, day, found, known_ids, input_tokens, output_tokens, refused)


def _read_reply(
    person_id: str,
    day: date,
    own: Sequence[Turn],
    targets: Sequence[Target],
    now: datetime,
    text: str,
    model: str,
) -> tuple[list[tuple[Evidence, Mention | None]], list[str]]:
    shown = {target.id: target for target in targets}
    read = _read_ancestry(own)
    ancestry = None if read is None else read | {model}
    found: list[tuple[Evidence, Mention | None]] = []
    refused: list[str] = []
    ordinals: dict[tuple[str, str], int] = {}
    for index, raw in enumerate(_items(text)):
        try:
            hit = _found(raw, len(own))
            mention = None if hit.target is None else _mention(hit.target, person_id, shown)
        except (KeyError, TypeError, ValueError) as exc:
            refused.append(f"item {index}: {_reason(exc)}")
            continue
        turn = own[hit.turn]
        ordinal = ordinals.get((turn.source, hit.kind), 0)
        ordinals[(turn.source, hit.kind)] = ordinal + 1
        found.append((Evidence(
            id="e-" + _hash(person_id, day.isoformat(), turn.source, hit.kind, str(ordinal))[:20],
            person_id=person_id, day=day, kind=hit.kind, text=hit.text, source=turn.source,
            held_out=False, created_at=now, model=model, ancestry=ancestry,
        ), mention))
    return found, refused


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
            items.append(replace(item, held_out=True, target_id=None))  # and no mention
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
        raise _Refused("not an object")
    turn, kind, text, target = raw["turn"], raw["kind"], raw["text"], raw.get("target")
    if not isinstance(turn, int) or isinstance(turn, bool) or not 0 <= turn < turn_count:
        raise _Refused("turn out of range")
    if not isinstance(kind, str) or kind not in PERSON_KINDS:
        raise _Refused("kind not allowed")
    if not isinstance(text, str) or not text.strip():
        raise _Refused("empty text")
    if target is not None and not isinstance(target, dict):
        raise _Refused("target not an object")
    return _Found(turn=turn, kind=kind, text=text.strip(), target=target)


def _mention(raw: Mapping[str, Any], person_id: str, shown: Mapping[str, Target]) -> Mention:
    kind, label, check_after = raw["kind"], raw["label"], raw.get("check_after")
    if not _is_target_kind(kind):
        raise _Refused("target kind not allowed")
    if not isinstance(label, str) or not label.strip():
        raise _Refused("target without a label")
    if kind == "commitment" and check_after is None:
        raise _Refused("commitment without a date")
    if kind == "thread" and check_after is not None:
        raise _Refused("thread with a date")
    confidence = raw.get("confidence") if kind == "thread" else None
    if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)):
        raise _Refused("confidence not a number")
    if confidence is not None and not (math.isfinite(confidence) and 0.0 <= confidence <= 1.0):
        raise _Refused("confidence outside 0 to 1")
    existing_id = raw.get("existing_id")
    existing = shown.get(existing_id) if isinstance(existing_id, str) else None
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
        raise _Refused("check_after is not a date")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise _Refused("check_after is not a date") from None
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
