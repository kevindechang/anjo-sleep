"""Deletion tombstones (Task 9): an id the forced-id list must never again hold.

The list lives in a deploy setting, which code does not edit. A delete leaves a tombstone (an empty file
named by the opaque id) in the sleep data dir, and the effective list is always the setting minus the
tombstones: applied at startup and at every read, so a restart cannot bring the entry back.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_USER: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")


class Tombstones:
    def __init__(self, root: Path) -> None:
        self.root = root

    def add(self, user_id: str) -> None:
        if not isinstance(user_id, str) or not _USER.fullmatch(user_id):
            raise ValueError("a user id is a plain name")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / user_id).touch()

    def all(self) -> frozenset[str]:
        return frozenset(p.name for p in self.root.iterdir()) if self.root.is_dir() else frozenset()

    def without(self, forced: str) -> str:
        """The comma list ``forced`` minus every tombstoned id."""
        dead = self.all()
        return ",".join(part for part in (p.strip() for p in forced.split(",")) if part and part not in dead)
