"""Synthetic people and their cleared sessions (Tasks 2, 3 and 4).

A cleared session is one the persona simulator talks in while hosted outputs can't train
(Task 2). Extraction reads a synthetic person's turns only from cleared sessions, so filmed opens
and anything else outside the simulated days never become evidence. Only a synthetic person can
have one.

The contract the dev API writes (Task 4):

- ``register_synthetic(user_id, at=...)`` once, when a synthetic person's account is made and
  before it stores any message; an account with messages is refused (``NotSynthetic``).
- ``mark_cleared(user_id, session_id, at=...)`` when a request carries the cleared flag, with the
  session id the turn is stored under (``messages.session_id``). It raises ``NotSynthetic`` for
  anyone else, which the API answers with a refusal.
- Stored in ``data/users/<id>/sleep/cleared_sessions.jsonl``, one encrypted record per line:
  ``{"kind": "synthetic", "at": <iso>}`` or ``{"kind": "cleared", "session_id": <id>, "at": <iso>}``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import CLEARED_FILE, CODEC, sleep_dir, writer_guard
from anjo.sleep.evidence.jsonl import LogError, append_records, read_records


class NotSynthetic(ValueError):
    """The cleared flag was sent for a person who isn't synthetic."""


def _path(user_id: str) -> Path:
    return sleep_dir(user_id) / CLEARED_FILE


def _records(user_id: str) -> list[dict[str, Any]]:
    records = read_records(_path(user_id), CODEC)
    for record in records:
        kind = record.get("kind")
        if kind not in ("synthetic", "cleared") or (kind == "cleared" and not isinstance(record.get("session_id"), str)):
            raise LogError(f"{CLEARED_FILE} holds a record that is not one of its kinds")
    return records


def is_synthetic(user_id: str) -> bool:
    return any(record["kind"] == "synthetic" for record in _records(user_id))


def cleared_sessions(user_id: str) -> frozenset[str]:
    return frozenset(record["session_id"] for record in _records(user_id) if record["kind"] == "cleared")


def _has_stored_messages(user_id: str) -> bool:
    """Any stored turn at all, forgotten ones included."""
    from anjo.core.db import get_db

    return get_db().execute("SELECT 1 FROM messages WHERE user_id = ? LIMIT 1", (user_id,)).fetchone() is not None


def register_synthetic(user_id: str, *, at: datetime) -> None:
    """Mark a new account synthetic. An account that already has stored messages is refused:
    marking a real person synthetic would let their turns into extraction as simulated ones."""
    require_sleep_host()
    with writer_guard(user_id)():
        if is_synthetic(user_id):
            return
        if _has_stored_messages(user_id):
            raise NotSynthetic("an account with stored messages can't be registered as synthetic")
        append_records(_path(user_id), CODEC, [{"kind": "synthetic", "at": at.isoformat()}])


def mark_cleared(user_id: str, session_id: str, *, at: datetime) -> None:
    require_sleep_host()
    if not session_id:
        raise ValueError("a cleared session needs its session id")
    with writer_guard(user_id)():
        if not is_synthetic(user_id):
            raise NotSynthetic("only a synthetic person's session can be cleared")
        if session_id not in cleared_sessions(user_id):
            append_records(
                _path(user_id), CODEC,
                [{"kind": "cleared", "session_id": session_id, "at": at.isoformat()}],
            )


def export_sessions(user_id: str) -> list[dict[str, Any]]:
    return _records(user_id)
