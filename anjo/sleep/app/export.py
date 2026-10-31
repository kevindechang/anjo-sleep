"""The ``sleep`` section of the account export (Task 3).

It carries everything under ``data/users/<id>/sleep/``: the evidence, the target log's table
(targets, results, expectations and the evidence ids it counted), the committed extractions, the
cleared-session records, each served reply's ancestry (Task 4), the briefs, practice transcripts and
the harness strategy list (Task 5), and MIND's workspace log with its folded records (Task 17).
The inventory names the whole folder (``sleep/**``), so a file this section doesn't know fails
the export instead of being left out silently; a task that adds a sleep file adds it here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from anjo.core.export_inventory import TEMPORARY_FILE, classify_file
from anjo.sleep.app.ancestry import ANCESTRY_FILE, export_ancestry
from anjo.sleep.app.cleared_sessions import export_sessions
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import (
    BRIEFS_FILE,
    CLEARED_FILE,
    EVIDENCE_FILE,
    EXTRACTIONS_FILE,
    PRACTICE_FILE,
    SLEEP_DIR,
    STRATEGIES_FILE,
    TARGETS_FILE,
    WORKSPACE_FILE,
    brief_store,
    evidence_store,
    extraction_log,
    sleep_dir,
    strategy_store,
    target_store,
    transcript_store,
    workspace_log,
)
from anjo.sleep.harness import harness_on

KNOWN_FILES = frozenset({
    EVIDENCE_FILE, TARGETS_FILE, EXTRACTIONS_FILE, CLEARED_FILE, ANCESTRY_FILE, BRIEFS_FILE, PRACTICE_FILE,
    STRATEGIES_FILE, WORKSPACE_FILE,
})


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
    strategies = strategy_store(user_id)
    return {
        **evidence_store(user_id).export(),
        **target_store(user_id).export(),
        **extraction_log(user_id).export(),
        **brief_store(user_id).export(),
        **transcript_store(user_id).export(),
        # Off with no list ever promoted: the section is today's, without an empty key. A list that
        # exists is the person's and exports even when the switch is off.
        **({} if not harness_on() and not strategies.path.exists() else strategies.export()),
        **workspace_log(user_id).export(),
        "cleared_sessions": export_sessions(user_id),
        "reply_ancestry": export_ancestry(user_id),
    }
