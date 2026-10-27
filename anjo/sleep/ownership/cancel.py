"""The cancel flag (Task 9, RF2): one counter per person, in a file that outlives the delete.

``set`` bumps the person's counter. A night reads it through ``watcher()``: the first look at a
person is the baseline, and any later change means that person was cancelled while the night ran.
A night started after the delete sees the new counter as its baseline, so a person who deletes
their data and stays is not locked out. The file holds a number and nothing else.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

_USER: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")


class CancelFlagError(ValueError):
    """A user id that is not a plain name, or a counter file that can't be read."""


class EpochFlag:
    """The flags under ``root``, one small file per person."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not _USER.fullmatch(user_id):
            raise CancelFlagError("a user id is a plain name")
        return self.root / user_id

    def epoch(self, user_id: str) -> int:
        path = self._path(user_id)
        if not path.exists():
            return 0
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            raise CancelFlagError("the cancel flag can't be read") from None

    def set(self, user_id: str) -> None:
        """Cancel every night that is running for ``user_id``: bump the counter, atomically."""
        path = self._path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(pending, "w", encoding="utf-8") as handle:
            handle.write(str(self.epoch(user_id) + 1))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)

    def watcher(self) -> Callable[[str], bool]:
        """A night's ``cancel`` check: True once the person's counter moved since the first look."""
        seen: dict[str, int] = {}

        def cancelled(user_id: str) -> bool:
            now = self.epoch(user_id)  # one read: the baseline and the comparison can't straddle a delete
            return now != seen.setdefault(user_id, now)

        return cancelled
