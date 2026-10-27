"""The ``sleep`` section of the account export (Task 3).

It carries everything under ``data/users/<id>/sleep/``: the evidence, the target log's table
(targets, results, expectations and the evidence ids it counted), the cleared-session records
and each served reply's ancestry (Task 4).
The inventory names the whole folder (``sleep/**``), so a file this section doesn't know fails
the export instead of being left out silently; a task that adds a sleep file adds it here.
"""
from __future__ import annotations

from typing import Any

from anjo.sleep.app.ancestry import ANCESTRY_FILE, export_ancestry
from anjo.sleep.app.cleared_sessions import export_sessions
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import CLEARED_FILE, EVIDENCE_FILE, TARGETS_FILE, evidence_store, sleep_dir, target_store

KNOWN_FILES = frozenset({EVIDENCE_FILE, TARGETS_FILE, CLEARED_FILE, ANCESTRY_FILE})


def export_sleep(user_id: str) -> dict[str, Any]:
    require_sleep_host()
    folder = sleep_dir(user_id)
    unknown = sorted(
        path.relative_to(folder).as_posix() for path in folder.rglob("*")
        if path.is_file() and path.relative_to(folder).as_posix() not in KNOWN_FILES
    ) if folder.is_dir() else []
    if unknown:
        raise ValueError(f"sleep files the export doesn't carry: {', '.join(unknown)}")
    return {
        **evidence_store(user_id).export(),
        **target_store(user_id).export(),
        "cleared_sessions": export_sessions(user_id),
        "reply_ancestry": export_ancestry(user_id),
    }
