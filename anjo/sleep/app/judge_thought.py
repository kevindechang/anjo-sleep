"""Her own line of thought, for the judge it is about: ``GET /api/sleep/me/thought`` and the person's
two actions on it, ``POST /api/sleep/me/thought/{id}/dispute`` and ``.../hide`` (``judge_api``).

The view and the actions are the engine's (``anjo.sleep.mind.thought``: ``thought_view``, ``dispute``,
``hide``), pure and with no model call. This module only reads and stores what is the person's own:

- **her workspace log** (``storage.workspace_log``), read whole, as the engine scans it;
- **their marks** (``<sleep folder>/thought_marks.jsonl``, one encrypted record a line, erased with
  the folder): what they disputed or hid. A mark never edits her log, which is append-only;
- **their visits** (``<sleep folder>/thought_visit.json``, encrypted): the morning note covers what
  happened since their last visit, and a visit is one stretch of calls with no gap over
  ``VISIT_GAP``, so the page's polling doesn't empty the note;
- **a dispute's note**, also kept as their own words in their evidence (a ``correction``, ancestry
  empty), so the next night reads it through the normal path.

Only an item the view shows them (a hunch, a settled bet, or the question it asks) can be disputed
or hidden; anything else is 404, as the engine's own refusal (``ValueError``) is.

Until the engine lands, ``engine()`` is None and the routes answer 503: the import is guarded, so
this branch merges before the engine's.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import logging
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

ENGINE_MODULE: Final = "anjo.sleep.mind.thought"
MARKS_FILE: Final = "thought_marks.jsonl"
VISIT_FILE: Final = "thought_visit.json"
VISIT_GAP: Final = timedelta(minutes=30)
NOTE_MAX: Final = 1000
DISPUTE_KIND: Final = "correction"  # the person's own words, setting her straight

_engine: list[Any] = []


class ThoughtUnavailable(Exception):
    """The engine isn't on this branch yet."""


class UnknownItem(ValueError):
    """Not an item the person's view shows."""


def engine() -> Any | None:
    """The engine module, or None until it lands. Tests set a fake with ``set_engine``."""
    if _engine:
        return _engine[0]
    try:
        return importlib.import_module(ENGINE_MODULE)
    except ModuleNotFoundError as missing:
        if missing.name != ENGINE_MODULE:
            raise
        return None


def set_engine(made: Any | None) -> None:
    _engine[:] = [] if made is None else [made]


def _required() -> Any:
    found = engine()
    if found is None:
        raise ThoughtUnavailable(ENGINE_MODULE)
    return found


def _sleep_file(user_id: str, name: str) -> Path:
    from anjo.sleep.app.storage import sleep_dir

    path: Path = sleep_dir(user_id) / name
    return path


# ── Marks ─────────────────────────────────────────────────────────────────────


def _record(mark: Any) -> dict[str, Any]:
    """A mark as one JSON record, whatever shape the engine returns it in."""
    if dataclasses.is_dataclass(mark) and not isinstance(mark, type):
        raw: Mapping[str, Any] = dataclasses.asdict(mark)
    elif isinstance(mark, Mapping):
        raw = mark
    else:
        raise TypeError("a mark is a record")
    return {key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in raw.items() if value is not None}


def _marks_for(found: Any, rows: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Stored records as the engine takes them: its own reader if it has one, its ``Mark`` if that
    is a dataclass, else the records as they are."""
    rows = list(rows)
    if hasattr(found, "mark_from_record"):
        return [found.mark_from_record(row) for row in rows]
    mark = getattr(found, "Mark", None)
    if mark is not None and dataclasses.is_dataclass(mark):
        return [mark(**{k: datetime.fromisoformat(v) if k == "at" else v for k, v in row.items()}) for row in rows]
    return rows


def load_marks(user_id: str) -> list[dict[str, Any]]:
    from anjo.sleep.app.storage import CODEC
    from anjo.sleep.evidence.jsonl import read_records

    rows: list[dict[str, Any]] = read_records(_sleep_file(user_id, MARKS_FILE), CODEC)
    return rows


def _add_mark(user_id: str, mark: Any) -> dict[str, Any]:
    from anjo.sleep.app.storage import CODEC, writer_guard
    from anjo.sleep.evidence.jsonl import append_records

    record = _record(mark)
    with writer_guard(user_id)():
        append_records(_sleep_file(user_id, MARKS_FILE), CODEC, [record])
    return record


# ── Visits ────────────────────────────────────────────────────────────────────


def _read_visit(user_id: str) -> dict[str, Any]:
    from anjo.sleep.app.storage import CODEC

    try:
        data = json.loads(CODEC.decode(_sleep_file(user_id, VISIT_FILE).read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError) as error:
        logger.error("a judge's visit record can't be read: %s", type(error).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def _when(raw: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def visit(user_id: str, now: datetime) -> datetime | None:
    """Their last visit before this one (the morning note's start), and this call recorded."""
    from anjo.sleep.app.storage import CODEC, writer_guard

    held = _read_visit(user_id)
    last, since = _when(held.get("last")), _when(held.get("since"))
    if last is None or now - last > VISIT_GAP:
        since = last  # a new visit: the note covers everything since the last one ended
    path = _sleep_file(user_id, VISIT_FILE)
    record = {"since": None if since is None else since.isoformat(), "last": now.isoformat()}
    with writer_guard(user_id)():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.part")
        part.write_text(CODEC.encode(json.dumps(record, sort_keys=True)), encoding="utf-8")
        os.replace(part, path)
    return since


# ── The view and the actions ──────────────────────────────────────────────────


def _events(user_id: str) -> tuple[Any, ...]:
    from anjo.sleep.app.storage import workspace_log

    return tuple(workspace_log(user_id).events())


def _view(found: Any, user_id: str, now: datetime, since: datetime | None) -> dict[str, Any]:
    marks = _marks_for(found, load_marks(user_id))
    view: dict[str, Any] = found.thought_view(_events(user_id), marks, now=now, since=since)
    return view


def view(user_id: str, now: datetime | None = None) -> dict[str, Any]:
    found = _required()
    at = now or datetime.now(timezone.utc)
    return _view(found, user_id, at, visit(user_id, at))


def _shown(view_: Mapping[str, Any]) -> frozenset[str]:
    ids = {str(item.get("id")) for key in ("hunches", "settled") for item in view_.get(key) or ()
           if isinstance(item, Mapping)}
    question = view_.get("question")
    if isinstance(question, Mapping) and question.get("question_id"):
        ids.add(str(question["question_id"]))
    return frozenset(ids)


def _act(user_id: str, item_id: str, make: Any, now: datetime | None) -> tuple[dict[str, Any], datetime]:
    found = _required()
    at = now or datetime.now(timezone.utc)
    if item_id not in _shown(_view(found, user_id, at, None)):
        raise UnknownItem(item_id)
    return _add_mark(user_id, make(found, at)), at


def _note_item(user_id: str, item_id: str, note: str, at: datetime) -> Any:
    from anjo.sleep.types import Evidence

    digest = hashlib.sha256("\x1f".join((user_id, "thought-dispute", item_id, at.isoformat(), note)).encode())
    return Evidence(id=digest.hexdigest(), person_id=user_id, day=at.date(), kind=DISPUTE_KIND, text=note,
                    source=f"thought:dispute:{item_id}", held_out=False, created_at=at, model=None,
                    ancestry=frozenset(), target_id=None)


def dispute(user_id: str, item_id: str, note: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Mark ``item_id`` disputed; a note is also kept as their own words in their evidence."""
    words = (note or "").strip()[:NOTE_MAX] or None
    record, at = _act(user_id, item_id, lambda found, at: found.dispute(item_id, at=at, note=words), now)
    if words is not None:
        from anjo.sleep.app.storage import evidence_store

        evidence_store(user_id).add([_note_item(user_id, item_id, words, at)])
    return record


def hide(user_id: str, item_id: str, now: datetime | None = None) -> dict[str, Any]:
    """Hide ``item_id`` from their view. Her workspace keeps it: this is not a forget."""
    return _act(user_id, item_id, lambda found, at: found.hide(item_id, at=at), now)[0]
