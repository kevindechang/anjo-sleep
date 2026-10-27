"""The ``sleep`` section of the account export (Task 3).

It carries everything under ``data/users/<id>/sleep/``: the evidence, the target log's table
(targets, results, expectations and the evidence ids it counted) and the cleared-session records.
The inventory names the whole folder (``sleep/**``), so a file this section doesn't know fails
the export instead of being left out silently; a task that adds a sleep file adds it here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from anjo.core.export_inventory import TEMPORARY_FILE, classify_file
from anjo.sleep.app.cleared_sessions import export_sessions
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import (
    CLEARED_FILE,
    EVIDENCE_FILE,
    SLEEP_DIR,
    TARGETS_FILE,
    evidence_store,
    sleep_dir,
    target_store,
)

KNOWN_FILES = frozenset({EVIDENCE_FILE, TARGETS_FILE, CLEARED_FILE})


def _check_folder(folder: Path) -> None:
    """Refuse a file the section doesn't carry, or any entry that isn't a regular file or a
    folder. A temporary write file is skipped, as the inventory skips it (``TEMPORARY_FILE``)."""
    unknown: list[str] = []
    for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
        name = path.relative_to(folder).as_posix()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"sleep entry {name} is not a regular file")
        if path.is_dir() or classify_file(f"{SLEEP_DIR}/{name}") == TEMPORARY_FILE:
            continue
        if name not in KNOWN_FILES:
            unknown.append(name)
    if unknown:
        raise ValueError(f"sleep files the export doesn't carry: {', '.join(unknown)}")


def export_sleep(user_id: str) -> dict[str, Any]:
    require_sleep_host()
    _check_folder(sleep_dir(user_id))
    return {
        **evidence_store(user_id).export(),
        **target_store(user_id).export(),
        "cleared_sessions": export_sessions(user_id),
    }
