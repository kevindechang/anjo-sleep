"""Each person's promoted checkpoints (Task 5): the pointer serving reads, and every adapter kept.

A person's folder under the registry root holds three files, all plain JSON (they hold adapter
names and dates, never a person's words):

- ``promoted.jsonl``: every checkpoint ever promoted, oldest first. Nothing is removed, so undo
  and the night slider (§2.8) can serve any of them.
- ``current.json``: the pointer to the checkpoint that serves. A promotion appends to the history
  first, then writes a new pointer file and renames it over the old one, so a promotion cut short
  at any point leaves the previous checkpoint serving (RF1).
- ``rolled_back.jsonl``: every checkpoint a rollback took out of service, and the one it went
  back to. History and ``previous`` skip them, so undo never lands on a night that was undone.
- ``trained.jsonl``: every adapter name a night started training under, promoted or not, so a
  name is never reused and each night's adapter number counts the nights that trained.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.evidence.jsonl import PLAIN, LogError, append_records, read_records
from anjo.sleep.types import CheckpointRef

ROOT_DIR: Final = "checkpoints"
POINTER: Final = "current.json"
PROMOTED: Final = "promoted.jsonl"
TRAINED: Final = "trained.jsonl"
ROLLED_BACK: Final = "rolled_back.jsonl"
DATA_DIR_ENV: Final = "ANJO_DATA_DIR"
_USER = re.compile(r"[A-Za-z0-9_-]{1,128}")


class CheckpointError(ValueError):
    """A promotion or record the registry refuses, or a file it can't read."""


class NoCheckpoint(LookupError):
    """The person has no promoted checkpoint yet: the base model serves."""


def ref_record(ckpt: CheckpointRef) -> dict[str, Any]:
    return {
        "id": ckpt.id, "person_id": ckpt.person_id, "night": ckpt.night.isoformat(),
        "uri": ckpt.uri, "parent_id": ckpt.parent_id, "rendering": ckpt.rendering,
    }


def ref_from_record(record: Mapping[str, Any]) -> CheckpointRef:
    try:
        parent = record["parent_id"]
        return CheckpointRef(
            id=str(record["id"]), person_id=str(record["person_id"]),
            night=date.fromisoformat(record["night"]), uri=str(record["uri"]),
            parent_id=None if parent is None else str(parent),
            rendering=None if record.get("rendering") is None else str(record["rendering"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"not a checkpoint record ({type(error).__name__})") from None


class CheckpointRegistry:
    """The registry under ``root``: one folder per person."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not _USER.fullmatch(user_id):
            raise CheckpointError("a user id is a plain name")
        return self.root / user_id

    def _promoted(self, user_id: str) -> list[CheckpointRef]:
        try:
            records = read_records(self._dir(user_id) / PROMOTED, PLAIN)
        except LogError as error:
            raise CheckpointError(str(error)) from None
        return [ref_from_record(record) for record in records]

    def promote(self, user_id: str, ckpt: CheckpointRef) -> None:
        """Make ``ckpt`` the one that serves: history first, then an atomic pointer swap."""
        if ckpt.person_id != user_id:
            raise CheckpointError("a person is served only their own checkpoint")
        folder = self._dir(user_id)
        if ckpt.id not in {stored.id for stored in self._promoted(user_id)}:
            append_records(folder / PROMOTED, PLAIN, [ref_record(ckpt)])
        self._point(folder, ckpt)

    def rollback(self, user_id: str, to: CheckpointRef) -> None:
        """Serve ``to`` again, an earlier checkpoint in the history, and record the move: every
        checkpoint after it in the history is rolled back."""
        history = self.history(user_id)
        ids = [ckpt.id for ckpt in history]
        if to.id not in ids[:-1]:
            raise CheckpointError("a rollback goes back to an earlier checkpoint that served")
        undone = ids[ids.index(to.id) + 1:]
        append_records(self._dir(user_id) / ROLLED_BACK, PLAIN, [{"id": item, "to": to.id} for item in undone])
        self._point(self._dir(user_id), history[ids.index(to.id)])

    def _rolled_back(self, user_id: str) -> frozenset[str]:
        try:
            return frozenset(str(record["id"]) for record in read_records(self._dir(user_id) / ROLLED_BACK, PLAIN))
        except (LogError, KeyError, TypeError) as error:
            raise CheckpointError(f"{ROLLED_BACK} can't be read ({type(error).__name__})") from None

    def _point(self, folder: Path, ckpt: CheckpointRef) -> None:
        pending = folder / f".{POINTER}.{os.getpid()}.tmp"
        with open(pending, "w", encoding="utf-8") as handle:
            json.dump(ref_record(ckpt), handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(pending, folder / POINTER)
        finally:
            pending.unlink(missing_ok=True)

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        """The checkpoint that serves, or None while the base model does (``AdapterRegistry``)."""
        pointer = self._dir(person_id) / POINTER
        if not pointer.exists():
            return None
        try:
            record = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CheckpointError(f"{POINTER} can't be read ({type(error).__name__})") from None
        if not isinstance(record, dict):
            raise CheckpointError(f"{POINTER} is not a record")
        return ref_from_record(record)

    def current(self, user_id: str) -> CheckpointRef:
        found = self.last_promoted(user_id)
        if found is None:
            raise NoCheckpoint("no night was promoted yet; the base model serves")
        return found

    def history(self, user_id: str) -> list[CheckpointRef]:
        """Every promoted checkpoint up to the one serving, oldest first, without those a rollback
        took out of service."""
        current = self.last_promoted(user_id)
        if current is None:
            return []
        undone = self._rolled_back(user_id) - {current.id}
        promoted = [ckpt for ckpt in self._promoted(user_id) if ckpt.id not in undone]
        ids = [ckpt.id for ckpt in promoted]
        return promoted[: ids.index(current.id) + 1] if current.id in ids else [*promoted, current]

    def previous(self, user_id: str) -> CheckpointRef | None:
        history = self.history(user_id)
        return history[-2] if len(history) >= 2 else None

    def record_trained(self, user_id: str, name: str, night: date) -> None:
        """Record that a night started training under ``name``; a name is taken once."""
        if name in self.trained_names(user_id):
            raise CheckpointError(f"adapter name {name} was already trained; a name is never reused")
        append_records(self._dir(user_id) / TRAINED, PLAIN, [{"name": name, "night": night.isoformat()}])

    def _trained(self, user_id: str) -> list[tuple[str, date]]:
        try:
            records = read_records(self._dir(user_id) / TRAINED, PLAIN)
            return [(str(record["name"]), date.fromisoformat(record["night"])) for record in records]
        except (LogError, KeyError, TypeError, ValueError) as error:
            raise CheckpointError(f"{TRAINED} can't be read ({type(error).__name__})") from None

    def trained_names(self, user_id: str) -> frozenset[str]:
        return frozenset(name for name, _ in self._trained(user_id))

    def trained_nights(self, user_id: str) -> frozenset[date]:
        return frozenset(night for _, night in self._trained(user_id))


def default_registry(environ: Mapping[str, str] | None = None) -> CheckpointRegistry:
    """The registry in the data dir: ``$ANJO_DATA_DIR/sleep/checkpoints``."""
    env = os.environ if environ is None else environ
    root = env.get(DATA_DIR_ENV, "").strip()
    if not root:
        raise CheckpointError(f"{DATA_DIR_ENV} names the data dir the registry lives in")
    return CheckpointRegistry(Path(root) / "sleep" / ROOT_DIR)


def promote(user_id: str, ckpt: CheckpointRef) -> None:
    default_registry().promote(user_id, ckpt)


def current(user_id: str) -> CheckpointRef:
    return default_registry().current(user_id)


def previous(user_id: str) -> CheckpointRef | None:
    return default_registry().previous(user_id)


def history(user_id: str) -> list[CheckpointRef]:
    return default_registry().history(user_id)
