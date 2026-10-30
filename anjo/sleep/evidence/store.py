"""One person's evidence as an append-only JSON Lines file (Task 3, the ``EvidenceStore`` port).

Each item is stored once, by id: an exact copy is skipped, and the same id with other content is
refused. Held-out items are kept for the sealed tests and never leave
through ``trainable``, ``stored`` or ``search``. ``trainable`` also refuses any item whose
ancestry isn't cleared (``anjo.sleep.provenance``). The codec and the writer guard are always
named: the app passes encryption and the account's guard, tests and the public example pass
``PLAIN`` and ``NO_GUARD``.

**The comparison's agent notes never leave through the ordinary readers.** An ``agent_note``
(``AGENT_NOTE``: the baseline arm's nightly note, ``agent-`` ids) is a model's note about the
person, not the person's own evidence, and it is read only by the morning arm 2A (``agent_notes``).
Its ancestry is Spark Open, which ``may_train`` accepts, so the kind is what keeps it out of
training and out of every other reader: ``trainable``, ``stored``, ``for_day`` and ``search`` leave
it out, and so every brief, practice seed and inquiry that reads through them does too.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, get_args

from anjo.sleep.evidence.jsonl import Codec, Guard, LogError, append_records, read_records
from anjo.sleep.provenance import may_train
from anjo.sleep.types import Evidence, EvidenceKind

AGENT_NOTE: Final = "agent_note"  # the baseline arm's nightly note (arm 2A reads it, and only it)
_KINDS = frozenset(get_args(EvidenceKind))
_QUIET_KINDS: Final[frozenset[str]] = frozenset({AGENT_NOTE})  # what the ordinary readers leave out
_WORD = re.compile(r"\w+")


class EvidenceConflict(ValueError):
    """An item whose id is stored with other content. The message names the id, never the text."""


def evidence_record(item: Evidence) -> dict[str, Any]:
    return {
        "id": item.id,
        "person_id": item.person_id,
        "day": item.day.isoformat(),
        "kind": item.kind,
        "text": item.text,
        "source": item.source,
        "held_out": item.held_out,
        "created_at": item.created_at.isoformat(),
        "model": item.model,
        "ancestry": None if item.ancestry is None else sorted(item.ancestry),
        "target_id": item.target_id,
    }


def evidence_from_record(record: Mapping[str, Any]) -> Evidence:
    """The item a stored record holds; ``LogError`` when the record isn't one. The message
    never repeats a stored value."""
    try:
        kind = record["kind"]
        ancestry = _optional_names(record["ancestry"])
        item = Evidence(
            id=_text(record["id"]),
            person_id=_text(record["person_id"]),
            day=date.fromisoformat(record["day"]),
            kind=kind,
            text=_text(record["text"]),
            source=_text(record["source"]),
            held_out=_flag(record["held_out"]),
            created_at=datetime.fromisoformat(record["created_at"]),
            model=_optional_text(record["model"]),
            ancestry=ancestry,
            target_id=_optional_text(record["target_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LogError(f"not an evidence record ({type(exc).__name__})") from None
    if kind not in _KINDS:
        raise LogError("unknown evidence kind")
    return item


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _optional_names(value: object) -> frozenset[str] | None:
    """A list of names, or None. A bare string is refused: it would split into characters."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("expected a list of names")
    return frozenset(_text(name) for name in value)


def _flag(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("expected a boolean")
    return value


def _words(text: str) -> frozenset[str]:
    return frozenset(word.casefold() for word in _WORD.findall(text))


class JsonlEvidenceStore:
    """The ``EvidenceStore`` port over one file. ``guard`` serializes writers across processes."""

    def __init__(
        self,
        path: Path,
        person_id: str,
        *,
        codec: Codec,
        guard: Guard,
        hosted_in_training: bool = False,
    ) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard
        self.hosted_in_training = hosted_in_training

    def _items(self) -> list[Evidence]:
        items: list[Evidence] = []
        for number, record in enumerate(read_records(self.path, self._codec), start=1):
            try:
                items.append(evidence_from_record(record))
            except LogError as exc:
                raise LogError(f"{self.path.name} line {number}: {exc}") from None
        return items

    def ids(self) -> frozenset[str]:
        return frozenset(item.id for item in self._items())

    def add(self, items: list[Evidence]) -> None:
        """Store new items and skip exact copies of stored ones. An id stored (or earlier in the
        batch) with other content raises ``EvidenceConflict``, and nothing from the batch is
        written."""
        foreign = sorted({item.id for item in items if item.person_id != self.person_id})
        if foreign:
            raise ValueError(f"evidence for another person: {', '.join(foreign)}")
        with self._guard():
            known = {item.id: evidence_record(item) for item in self._items()}
            fresh: list[dict[str, Any]] = []
            for item in items:
                record = evidence_record(item)
                stored = known.get(item.id)
                if stored is None:
                    known[item.id] = record
                    fresh.append(record)
                elif stored != record:
                    raise EvidenceConflict(f"evidence {item.id} is already stored with other content")
            append_records(self.path, self._codec, fresh)

    def for_day(self, day: date) -> list[Evidence]:
        """Every item stamped with ``day``, held-out ones included, but never an agent note."""
        return [item for item in self._items() if item.day == day and item.kind not in _QUIET_KINDS]

    def trainable(self, until: date) -> list[Evidence]:
        return [
            item for item in self.stored(until)
            if item.kind not in _QUIET_KINDS
            and may_train(item.ancestry, hosted_in_training=self.hosted_in_training)
        ]

    def stored(self, until: date) -> list[Evidence]:
        """Every non-held-out item through ``until``, for served context and arm 2F; the baseline's
        notes are left out, since only arm 2A reads them (``agent_notes``)."""
        return [
            item for item in self._items()
            if item.day <= until and not item.held_out and item.kind not in _QUIET_KINDS
        ]

    def held_out(self, day: date) -> list[Evidence]:
        return [item for item in self.for_day(day) if item.held_out]

    def agent_notes(self, until: date) -> list[Evidence]:
        """The baseline's notes through ``until``, oldest first: arm 2A's one source for them."""
        return [
            item for item in self._items()
            if item.kind == AGENT_NOTE and item.day <= until and not item.held_out
        ]

    def search(self, query: str, as_of: datetime, k: int) -> list[Evidence]:
        """Lexical overlap with ``query``; equal overlap puts the newer item first."""
        wanted = _words(query)
        scored = [
            (len(wanted & _words(item.text)), item)
            for item in self._items()
            if not item.held_out and item.created_at <= as_of and item.kind not in _QUIET_KINDS
        ]
        ranked = sorted(
            ((score, item) for score, item in scored if score > 0),
            key=lambda pair: (-pair[0], -pair[1].created_at.timestamp(), pair[1].id),
        )
        return [item for _, item in ranked[: max(k, 0)]]

    def get(self, ids: Sequence[str], as_of: datetime) -> tuple[Evidence, ...]:
        """The items with these ids created at or before ``as_of``, in the order asked (MIND's
        ``EvidenceSearch.get``, Task 17). An id it doesn't hold, a later item and a held-out item
        are left out, so a resumed inquiry fetches exactly what the first run saw."""
        held = {item.id: item for item in self._items() if not item.held_out and item.created_at <= as_of}
        return tuple(held[item_id] for item_id in ids if item_id in held)

    def export(self) -> dict[str, object]:
        return {"evidence": [evidence_record(item) for item in self._items()]}
