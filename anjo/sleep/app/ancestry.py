"""Each served reply's ancestry, keyed by turn id (Task 4, spec §3).

A flagged turn records two things here, in ``data/users/<id>/sleep/reply_ancestry.jsonl``,
encrypted like the rest of the person's sleep files:

- ``context``, when the prompt is built: each context part's source and ancestry, and whether
  the session was cleared. Anjo's memory has no recorded source, so a part carrying it has
  unknown ancestry (``None``).
- ``served``, once Glimmer's first word is shown, or a tool round of its ends cleanly: Glimmer,
  under the adapter it was sent. A call that breaks off before then records nothing.

``reply_ancestry`` folds them. A turn with a context record and no served record was answered by
the frontier fallback (RF3); its ancestry is unknown, since another model wrote it. A reply Glimmer
served derives from Glimmer and from everything its context derives from. Tasks 3 and 5 read it.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import CODEC, sleep_dir, writer_guard
from anjo.sleep.evidence.jsonl import LogError, append_records, read_records
from anjo.sleep.provenance import GLIMMER

ANCESTRY_FILE: Final = "reply_ancestry.jsonl"
SERVED_BY_SLEEP: Final = "sleep_modal"
SERVED_BY_FALLBACK: Final = "fallback"
ServedBy = Literal["sleep_modal", "fallback"]


@dataclass(frozen=True)
class ContextPart:
    """One part of a turn's context: where it came from, and what its words derive from (empty
    for the person's own words, ``None`` when unknown)."""

    source: str
    ancestry: frozenset[str] | None


@dataclass(frozen=True)
class ReplyAncestry:
    turn_id: str
    session_id: str | None
    cleared: bool
    served_by: ServedBy
    model: str | None
    context: frozenset[str] | None
    reply: frozenset[str] | None


def union(parts: Iterable[frozenset[str] | None]) -> frozenset[str] | None:
    """The ancestry of text built from these parts: unknown when any part's is."""
    total: frozenset[str] = frozenset()
    for ancestry in parts:
        if ancestry is None:
            return None
        total |= ancestry
    return total


def _path(user_id: str) -> Path:
    return sleep_dir(user_id) / ANCESTRY_FILE


def _encoded(ancestry: frozenset[str] | None) -> list[str] | None:
    return None if ancestry is None else sorted(ancestry)


def _decoded(value: object) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LogError(f"{ANCESTRY_FILE} holds an ancestry that is not a list of model ids")
    return frozenset(value)


def _append(user_id: str, record: dict[str, Any]) -> None:
    require_sleep_host()
    with writer_guard(user_id)():
        append_records(_path(user_id), CODEC, [record])


def record_context(
    user_id: str, turn_id: str, session_id: str | None, *, cleared: bool,
    parts: Iterable[ContextPart],
) -> None:
    _append(user_id, {
        "kind": "context", "turn_id": turn_id, "session_id": session_id, "cleared": cleared,
        "parts": [
            {"source": part.source, "ancestry": _encoded(part.ancestry)} for part in parts
        ],
        "at": datetime.now(timezone.utc).isoformat(),
    })


def record_served(user_id: str, turn_id: str, *, model: str) -> None:
    _append(user_id, {
        "kind": "served", "turn_id": turn_id, "by": SERVED_BY_SLEEP, "model": model,
        "at": datetime.now(timezone.utc).isoformat(),
    })


def _records(user_id: str) -> list[dict[str, Any]]:
    records = read_records(_path(user_id), CODEC)
    for record in records:
        if record.get("kind") not in ("context", "served") or not isinstance(record.get("turn_id"), str):
            raise LogError(f"{ANCESTRY_FILE} holds a record that is not one of its kinds")
    return records


def reply_ancestries(user_id: str) -> dict[str, ReplyAncestry]:
    """Every recorded turn's ancestry, by turn id."""
    by_turn: dict[str, list[dict[str, Any]]] = {}
    for record in _records(user_id):
        by_turn.setdefault(record["turn_id"], []).append(record)
    folded = {turn_id: _fold(turn_id, records) for turn_id, records in by_turn.items()}
    return {turn_id: ancestry for turn_id, ancestry in folded.items() if ancestry is not None}


def reply_ancestry(user_id: str, turn_id: str) -> ReplyAncestry | None:
    """The turn's ancestry, or None when no flagged prompt was built for it."""
    return reply_ancestries(user_id).get(turn_id)


def _fold(turn_id: str, mine: list[dict[str, Any]]) -> ReplyAncestry | None:
    contexts = [record for record in mine if record["kind"] == "context"]
    if not contexts:
        return None
    context = contexts[-1]
    parts = context.get("parts")
    if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
        raise LogError(f"{ANCESTRY_FILE} holds a context record without its parts")
    context_ancestry = union(_decoded(part.get("ancestry")) for part in parts)
    served = [record for record in mine if record["kind"] == "served"]
    model = served[-1].get("model") if served else None
    by: ServedBy = SERVED_BY_SLEEP if served else SERVED_BY_FALLBACK
    reply = None if context_ancestry is None or not served else context_ancestry | {GLIMMER}
    return ReplyAncestry(
        turn_id=turn_id, session_id=context.get("session_id"), cleared=context.get("cleared") is True,
        served_by=by, model=model if isinstance(model, str) else None,
        context=context_ancestry, reply=reply,
    )


def export_ancestry(user_id: str) -> list[dict[str, Any]]:
    return _records(user_id)
