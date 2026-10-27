"""Which days a person's extraction has committed (Task 3), one line per day, window and input.

Each line holds the ids of the items that extraction stored. The app commits the line after the
items and their mentions are stored, so a re-run of a committed day reads those items back
without calling the model, even though a new reply would word them differently. A run cut short
before its line calls the model again; the items' ids come from the turns they were read from,
so what the first run stored is skipped. A window is keyed by its instants in UTC, and the input
by a digest of the turns the call read: a session cleared after a night, or a turn forgotten,
makes a new input, and that day is read again.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from anjo.sleep.evidence.jsonl import Codec, Guard, LogError, append_records, read_records

Window = tuple[datetime, datetime]
Key = tuple[str, str, str, str]


def read_digest(read: Sequence[str]) -> str:
    """The input's digest: the sources of the turns the call read, in any order."""
    return hashlib.sha256("\x1f".join(sorted(read)).encode()).hexdigest()


def _instant(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("an extraction window needs aware datetimes")
    return moment.astimezone(timezone.utc).isoformat()


def _key(day: date, window: Window, read: Sequence[str]) -> Key:
    start, end = window
    return day.isoformat(), _instant(start), _instant(end), read_digest(read)


def _entry(record: Mapping[str, Any]) -> tuple[Key, tuple[str, ...]]:
    key = (record.get("day"), record.get("start"), record.get("end"), record.get("read"))
    ids = record.get("item_ids")
    day, start, end, digest = key
    if not (
        isinstance(day, str) and isinstance(start, str) and isinstance(end, str)
        and isinstance(digest, str)
        and isinstance(ids, list) and all(isinstance(item_id, str) for item_id in ids)
    ):
        raise LogError("not an extraction record")
    return (day, start, end, digest), tuple(ids)


def _record(key: Key, ids: Sequence[str]) -> dict[str, object]:
    return {"day": key[0], "start": key[1], "end": key[2], "read": key[3], "item_ids": list(ids)}


class ExtractionLog:
    """The committed extractions of one person. ``guard`` serializes writers across processes."""

    def __init__(self, path: Path, person_id: str, *, codec: Codec, guard: Guard) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard

    def _entries(self) -> dict[Key, tuple[str, ...]]:
        entries: dict[Key, tuple[str, ...]] = {}
        for number, record in enumerate(read_records(self.path, self._codec), start=1):
            try:
                key, ids = _entry(record)
            except LogError as exc:
                raise LogError(f"{self.path.name} line {number}: {exc}") from None
            entries.setdefault(key, ids)
        return entries

    def committed(self, day: date, window: Window, read: Sequence[str]) -> tuple[str, ...] | None:
        """The ids the committed extraction of ``day`` over ``window``, reading the turns
        ``read``, stored; None when that extraction isn't committed."""
        return self._entries().get(_key(day, window, read))

    def commit(self, day: date, window: Window, read: Sequence[str], item_ids: Sequence[str]) -> None:
        """Record the extraction as committed. One already committed keeps its first line."""
        key = _key(day, window, read)
        with self._guard():
            if key in self._entries():
                return
            append_records(self.path, self._codec, [_record(key, item_ids)])

    def export(self) -> dict[str, object]:
        return {"extractions": [_record(key, ids) for key, ids in self._entries().items()]}
