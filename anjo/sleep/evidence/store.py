"""One person's evidence as an append-only JSON Lines file (Task 3, the ``EvidenceStore`` port).

Each item is stored once, by id. Held-out items are kept for the sealed tests and never leave
through ``trainable``, ``stored`` or ``search``. ``trainable`` also refuses any item whose
ancestry isn't cleared (``anjo.sleep.provenance``). The app wraps this store with encryption and
the account's writer guard; the public example uses it as is.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, get_args

from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN, Codec, Guard, LogError, append_records, read_records
from anjo.sleep.provenance import may_train
from anjo.sleep.types import Evidence, EvidenceKind

_KINDS = frozenset(get_args(EvidenceKind))
_WORD = re.compile(r"\w+")


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
    """The item a stored record holds; ``LogError`` when the record isn't one."""
    try:
        kind = record["kind"]
        ancestry = record["ancestry"]
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
            ancestry=None if ancestry is None else frozenset(_text(name) for name in ancestry),
            target_id=_optional_text(record["target_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LogError(f"not an evidence record ({type(exc).__name__})") from None
    if kind not in _KINDS:
        raise LogError(f"unknown evidence kind {kind!r}")
    return item


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


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
        codec: Codec = PLAIN,
        guard: Guard = NO_GUARD,
        hosted_in_training: bool = False,
    ) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard
        self.hosted_in_training = hosted_in_training

    def _items(self) -> list[Evidence]:
        return [evidence_from_record(record) for record in read_records(self.path, self._codec)]

    def ids(self) -> frozenset[str]:
        return frozenset(item.id for item in self._items())

    def add(self, items: list[Evidence]) -> None:
        foreign = sorted({item.id for item in items if item.person_id != self.person_id})
        if foreign:
            raise ValueError(f"evidence for another person: {', '.join(foreign)}")
        with self._guard():
            known = set(self.ids())
            fresh: list[Evidence] = []
            for item in items:
                if item.id not in known:
                    known.add(item.id)
                    fresh.append(item)
            append_records(self.path, self._codec, [evidence_record(item) for item in fresh])

    def for_day(self, day: date) -> list[Evidence]:
        return [item for item in self._items() if item.day == day]

    def trainable(self, until: date) -> list[Evidence]:
        return [
            item for item in self.stored(until)
            if may_train(item.ancestry, hosted_in_training=self.hosted_in_training)
        ]

    def stored(self, until: date) -> list[Evidence]:
        return [item for item in self._items() if item.day <= until and not item.held_out]

    def held_out(self, day: date) -> list[Evidence]:
        return [item for item in self.for_day(day) if item.held_out]

    def search(self, query: str, as_of: datetime, k: int) -> list[Evidence]:
        """Lexical overlap with ``query``; equal overlap puts the newer item first."""
        wanted = _words(query)
        scored = [
            (len(wanted & _words(item.text)), item)
            for item in self._items()
            if not item.held_out and item.created_at <= as_of
        ]
        ranked = sorted(
            ((score, item) for score, item in scored if score > 0),
            key=lambda pair: (-pair[0], -pair[1].created_at.timestamp(), pair[1].id),
        )
        return [item for _, item in ranked[: max(k, 0)]]

    def export(self) -> dict[str, object]:
        return {"evidence": [evidence_record(item) for item in self._items()]}
