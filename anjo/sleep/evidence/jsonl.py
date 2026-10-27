"""Append-only JSON Lines files, one record per line, each line passed through a codec.

The app's codec encrypts every line with the same at-rest encryption as chat history; the plain
codec is for tests and the public example. A line is only ever appended. A write killed
mid-line leaves a torn last line: readers ignore it, and the next append cuts it off first.
Any other line that can't be read refuses the whole file, so nothing is skipped silently.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Protocol

Guard = Callable[[], AbstractContextManager[object]]
NO_GUARD: Guard = nullcontext
_NEWLINE = b"\n"


class Codec(Protocol):
    def encode(self, text: str) -> str:
        """One line's stored form; it must not contain a newline."""

    def decode(self, line: str) -> str:
        """The text a stored line holds; raises ``ValueError`` when it can't be read."""


class PlainCodec:
    def encode(self, text: str) -> str:
        return text

    def decode(self, line: str) -> str:
        return line


PLAIN = PlainCodec()


class LogError(ValueError):
    """A stored line that can't be read. The file is refused rather than read in part."""


def canonical(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_records(path: Path, codec: Codec) -> list[dict[str, Any]]:
    """Every complete record in order; a torn last line is left out."""
    if not path.exists():
        return []
    complete, _, _torn = path.read_bytes().rpartition(_NEWLINE)
    if not complete:
        return []
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(complete.split(_NEWLINE), start=1):
        try:
            value = json.loads(codec.decode(raw.decode("utf-8")))
        except ValueError:
            raise LogError(f"{path.name} line {number} can't be read") from None
        if not isinstance(value, dict):
            raise LogError(f"{path.name} line {number} is not a record")
        records.append(value)
    return records


def append_records(path: Path, codec: Codec, records: Iterable[Mapping[str, Any]]) -> None:
    """Append records as whole lines. The caller holds the file's writer guard."""
    lines = [codec.encode(canonical(record)) for record in records]
    if any("\n" in line for line in lines):
        raise ValueError("an encoded record must fit on one line")
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _cut_torn_tail(path)
    data = "".join(line + "\n" for line in lines).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _cut_torn_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == _NEWLINE:
            return
        handle.seek(0)
        handle.truncate(handle.read().rfind(_NEWLINE) + 1)
