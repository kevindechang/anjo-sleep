"""One person's mind on disk: the one state of the one system, versioned a night at a time.

MIND and Sleep are one system, so a night's result is one unit: the core state she wakes with,
the thought channel's projector and reader version she is read through, and the adapter (her
LoRA) that serves her. Saving them together is what lets a night be replayed or rolled back whole.

Under ``<data dir>/sleep/mind_state/<person>/``:

- ``states/<id>.npz``: each core state's bytes (``state.save``), content-addressed: the id is the
  first 16 hex characters of the bytes' SHA-256 (``state_id``), checked on every read. It is the
  thought channel's ``state_sha``, so a training row's ``core`` snapshot names its state by this id.
- ``units.jsonl``: one line per version, appended and synced: the night, the state's id, the shared
  core's model id (``Core.model_id``, the snapshot's ``core_model``), the projector version, the
  adapter, the version it continued from and, for a rollback, the version it rolled back from. Ids
  and versions only, never words.

A night starts from ``before(night)``, the newest unit of an earlier night, so a rerun of a night
never observes that night's events twice. ``rollback(version)`` appends a copy of an earlier unit,
so the history keeps both. A delete removes the person's whole folder (``erase``; the app's
ownership runs it). Torch-free: only bytes, ids and dates.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

STORE_DIR: Final = "mind_state"  # under <data dir>/sleep/
STATES: Final = "states"
UNITS: Final = "units.jsonl"
_PERSON = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")  # every account id ownership accepts, never "." or ".."
_ID = re.compile(r"[0-9a-f]{16}")
ID_HEX: Final = 16


class StoreError(ValueError):
    """A state or unit the store can't give back as it was kept."""


@dataclass(frozen=True)
class Unit:
    version: int
    night: date
    state: str  # the core state's id
    model: str  # the shared core's model id
    projector: str | None  # the thought channel's projector and reader version, None before it serves
    lora: str | None  # the adapter that serves her after this night
    parent: int | None  # the version this night continued from
    rolled_back_from: int | None = None


def state_id(blob: bytes) -> str:
    """A saved state's id: ``state_id(state.save(s))`` is the thought channel's ``state_sha(s)``."""
    return hashlib.sha256(blob).hexdigest()[:ID_HEX]


def unit_record(unit: Unit) -> dict[str, Any]:
    return {**asdict(unit), "night": unit.night.isoformat()}


def _unit(row: dict[str, Any]) -> Unit:
    return Unit(**{**row, "night": date.fromisoformat(row["night"])})


def _write(path: Path, blob: bytes) -> None:
    pending = path.with_name(f".{path.name}.tmp")
    with open(pending, "wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


class MindStore:
    """One person's units and core states."""

    def __init__(self, root: Path, person: str) -> None:
        if not isinstance(person, str) or not _PERSON.fullmatch(person):
            raise ValueError("a person's folder is a plain name")
        self.folder = root / person

    # ── states ───────────────────────────────────────────────────────────────

    def put_state(self, blob: bytes) -> str:
        held = state_id(blob)
        path = self.folder / STATES / f"{held}.npz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, blob)
        return held

    def state(self, held: str) -> bytes:
        if not isinstance(held, str) or not _ID.fullmatch(held) or not (path := self.folder / STATES / f"{held}.npz").is_file():
            raise StoreError(f"unknown core state {held!r}")
        blob = path.read_bytes()
        if state_id(blob) != held:
            raise StoreError(f"core state {held} doesn't match its id")
        return blob

    # ── units ────────────────────────────────────────────────────────────────

    def units(self) -> tuple[Unit, ...]:
        path = self.folder / UNITS
        if not path.is_file():
            return ()
        return tuple(_unit(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line)

    def head(self) -> Unit | None:
        units = self.units()
        return units[-1] if units else None

    def before(self, night: date) -> Unit | None:
        """The newest unit of a night before ``night``: what that night starts from."""
        return next((u for u in reversed(self.units()) if u.night < night), None)

    def _append(self, unit: Unit) -> Unit:
        self.folder.mkdir(parents=True, exist_ok=True)
        with open(self.folder / UNITS, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(unit_record(unit), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return unit

    def keep(self, night: date, state: str, *, model: str, projector: str | None, lora: str | None) -> Unit:
        """Tonight's unit, continuing from the night before it."""
        self.state(state)  # kept first, whole
        start = self.before(night)
        head = self.head()
        version = 1 if head is None else head.version + 1
        return self._append(Unit(version, night, state, model, projector, lora,
                                 None if start is None else start.version))

    def rollback(self, version: int) -> Unit:
        """Her mind as it was at ``version``, kept as the newest unit."""
        units = self.units()
        found = next((u for u in units if u.version == version), None)
        if found is None:
            raise StoreError(f"no version {version} to roll back to")
        return self._append(Unit(units[-1].version + 1, found.night, found.state, found.model, found.projector,
                                 found.lora, found.parent, rolled_back_from=units[-1].version))

    def erase(self) -> bool:
        if not self.folder.exists():
            return False
        shutil.rmtree(self.folder)
        return True
