"""The cleanup inventory (Task 9): what a delete still has to remove remotely, kept on disk until it
has. One small JSON file per person, named by their opaque account id, holding adapter names (hashes)
and nothing else. It lives outside the person's folder, so account erasure leaves it, and a retried
delete or the night's start reads it."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Final

_USER: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")


class FileInventory:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not _USER.fullmatch(user_id):
            raise ValueError("a user id is a plain name")
        return self.root / f"{user_id}.json"

    def read(self, user_id: str) -> frozenset[str]:
        path = self._path(user_id)
        if not path.exists():
            return frozenset()
        try:
            return frozenset(str(name) for name in json.loads(path.read_text(encoding="utf-8"))["adapters"])
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("the cleanup inventory can't be read") from None

    def write(self, user_id: str, adapters: frozenset[str]) -> None:
        """Record ``adapters`` (added to any already there), atomically."""
        path = self._path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(pending, "w", encoding="utf-8") as handle:
            json.dump({"adapters": sorted(adapters | self.read(user_id))}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)

    def clear(self, user_id: str) -> None:
        self._path(user_id).unlink(missing_ok=True)

    def pending(self) -> list[str]:
        """Every person with work outstanding."""
        return sorted(p.stem for p in self.root.glob("*.json")) if self.root.is_dir() else []
