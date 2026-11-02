"""Her own line of thought, for the judge it is about: ``GET /api/sleep/me/thought`` and the person's
two actions on it, ``POST /api/sleep/me/thought/{id}/dispute`` and ``.../hide`` (``judge_api``).

The view and the actions are the engine's (``anjo.sleep.mind.thought``: ``thought_view``, ``dispute``,
``hide``), pure and with no model call. This module only reads and stores what is the person's own:

- **her workspace log** (``storage.workspace_log``), read whole, as the engine scans it;
- **their marks** (``<sleep folder>/thought_marks.jsonl``, one encrypted record a line, erased with
  the folder): what they disputed or hid. A mark never edits her log, which is append-only;
- **their visits** (``<sleep folder>/thought_visit.json``, encrypted): the morning note covers what
  happened since their last visit. A visit is one stretch of calls with no gap over ``VISIT_GAP``,
  so the page's polling doesn't empty the note, and a night logged since the last call starts a new
  one: a judge who sleeps and wakes within the gap still gets the night's note (``visit``);
- **a dispute's note**, also kept as their own words in their evidence (a ``correction``, ancestry
  empty), so the next night reads it through the normal path.

Only an item in her log (a hunch, or the question it is about) can be disputed or hidden; any other
id is 404, as the engine refuses it (``ValueError``). A log or marks file that can't be read whole,
or whose digests don't replay, is ``LogUnreadable`` (500): never a view of part of it.

**Marks:** each is stored as the engine's own ``Mark.to_json()`` and passed back through
``Mark.from_json``, which checks it. A note is at most ``thought.MAX_NOTE`` characters, the engine's
own limit: a longer one is ``NoteTooLong`` (422), never cut and never a 404.

**The clock:** the view's ``now`` is the night clock (``night_clock``): the wall clock, or the last
step her log holds when a night logged ahead of it, so nothing she logged is left out.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.mind import thought

logger = logging.getLogger(__name__)

MARKS_FILE: Final = "thought_marks.jsonl"
VISIT_FILE: Final = "thought_visit.json"
VISIT_GAP: Final = timedelta(minutes=30)
BEFORE_NIGHT: Final = timedelta(microseconds=1)  # a note from a night's start includes its own first steps
DISPUTE_KIND: Final = "correction"  # the person's own words, setting her straight


class LogUnreadable(Exception):
    """Her log or their marks can't be read whole, or her log's digests don't replay (500)."""


class UnknownItem(ValueError):
    """No hunch or question with that id in her log (404)."""


class NoteTooLong(Exception):
    """A dispute's note over ``thought.MAX_NOTE`` characters (422)."""


def _sleep_file(user_id: str, name: str) -> Path:
    from anjo.sleep.app.storage import sleep_dir

    path: Path = sleep_dir(user_id) / name
    return path


# ── Marks ─────────────────────────────────────────────────────────────────────


def load_marks(user_id: str) -> list[dict[str, Any]]:
    from anjo.sleep.app.storage import CODEC
    from anjo.sleep.evidence.jsonl import read_records

    rows: list[dict[str, Any]] = read_records(_sleep_file(user_id, MARKS_FILE), CODEC)
    return rows


def _add_mark(user_id: str, mark: thought.Mark) -> dict[str, Any]:
    from anjo.sleep.app.storage import CODEC, writer_guard
    from anjo.sleep.evidence.jsonl import append_records

    record: dict[str, Any] = mark.to_json()
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


def _night_start(events: Sequence[Any], logged: int) -> datetime | None:
    """When the first night logged after the first ``logged`` events began. A live pass between
    nights (``night.mind.LIVE_DECIDER``) runs after every reply and is no night."""
    from anjo.sleep.mind.inquiry.types import RunStart
    from anjo.sleep.night.mind import LIVE_DECIDER

    for event in events[logged:]:
        entry = event.step.entry
        if isinstance(entry, RunStart) and entry.decider != LIVE_DECIDER:
            at: datetime = event.step.at
            return at
    return None


def visit(user_id: str, now: datetime, clock: datetime | None = None,
          events: Sequence[Any] = ()) -> datetime | None:
    """Their last visit before this one (the morning note's start), and this call recorded.

    ``now`` is the wall clock, which times the gap; ``clock`` the night clock the view read (``now``
    when ``None``), and the next visit's note starts where this one's clock ended. ``events`` is her
    log as this call read it: a night logged since the last call starts a new visit whose note
    starts at that night's start, or at the last visit's clock when that is earlier."""
    from anjo.sleep.app.storage import CODEC, writer_guard

    held = _read_visit(user_id)
    last, since = _when(held.get("last")), _when(held.get("since"))
    before = _when(held.get("seen")) or last
    logged = held.get("logged")
    logged = logged if isinstance(logged, int) and 0 <= logged <= len(events) else len(events)
    night = _night_start(events, logged) if before is not None else None
    if night is not None and before is not None:
        since = min(before, night - BEFORE_NIGHT)
    elif last is None or now - last > VISIT_GAP:
        since = before  # a new visit: the note covers everything since the last one
    path = _sleep_file(user_id, VISIT_FILE)
    record = {"since": None if since is None else since.isoformat(), "last": now.isoformat(),
              "seen": (clock or now).isoformat(), "logged": len(events)}
    with writer_guard(user_id)():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.part")
        part.write_text(CODEC.encode(json.dumps(record, sort_keys=True)), encoding="utf-8")
        os.replace(part, path)
    return since


# ── The view and the actions ──────────────────────────────────────────────────


def _read(user_id: str) -> tuple[tuple[Any, ...], list[thought.Mark]]:
    """Her log and their marks, each read whole and checked, or ``LogUnreadable``."""
    from anjo.sleep.app.storage import workspace_log

    try:
        events = tuple(workspace_log(user_id).events())
        marks = [thought.Mark.from_json(row) for row in load_marks(user_id)]
    except ValueError as error:
        logger.error("a judge's thought log can't be read: %s", type(error).__name__)
        raise LogUnreadable(type(error).__name__) from None
    return events, marks


def night_clock(events: Sequence[Any], wall: datetime) -> datetime:
    """MIND's clock for the view: the wall clock, or her last logged step if a night logged ahead."""
    logged: list[datetime] = [event.step.at for event in events]
    return max([wall, *logged])


def _view(events: Sequence[Any], marks: Sequence[thought.Mark], now: datetime,
          since: datetime | None) -> dict[str, Any]:
    try:
        shown: dict[str, Any] = thought.thought_view(events, marks, now=now, since=since)
    except ValueError as error:  # a digest that doesn't replay: a tampered or broken log
        logger.error("a judge's thought log doesn't replay: %s", type(error).__name__)
        raise LogUnreadable(type(error).__name__) from None
    return shown


def view(user_id: str, now: datetime | None = None) -> dict[str, Any]:
    wall = now or datetime.now(timezone.utc)
    events, marks = _read(user_id)
    clock = night_clock(events, wall)
    return _view(events, marks, clock, visit(user_id, wall, clock, events))


def _act(user_id: str, item_id: str, kind: thought.MarkKind, note: str | None,
         now: datetime | None) -> tuple[dict[str, Any], datetime]:
    at = now or datetime.now(timezone.utc)
    events, marks = _read(user_id)
    _view(events, marks, night_clock(events, at), None)  # a log that doesn't replay is a 500, never a 404
    try:
        mark = (thought.dispute(events, item_id, at=at, note=note) if kind == "dispute"
                else thought.hide(events, item_id, at=at))
    except ValueError:
        raise UnknownItem(item_id) from None
    return _add_mark(user_id, mark), at


def _note_item(user_id: str, item_id: str, note: str, at: datetime) -> Any:
    from anjo.sleep.types import Evidence

    digest = hashlib.sha256("\x1f".join((user_id, "thought-dispute", item_id, at.isoformat(), note)).encode())
    return Evidence(id=digest.hexdigest(), person_id=user_id, day=at.date(), kind=DISPUTE_KIND, text=note,
                    source=f"thought:dispute:{item_id}", held_out=False, created_at=at, model=None,
                    ancestry=frozenset(), target_id=None)


def dispute(user_id: str, item_id: str, note: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Mark ``item_id`` disputed; a note is also kept as their own words in their evidence."""
    words = (note or "").strip() or None
    if words is not None and len(words) > thought.MAX_NOTE:
        raise NoteTooLong(len(words))
    record, at = _act(user_id, item_id, "dispute", words, now)
    if words is not None:
        from anjo.sleep.app.storage import evidence_store

        evidence_store(user_id).add([_note_item(user_id, item_id, words, at)])
    return record


def hide(user_id: str, item_id: str, now: datetime | None = None) -> dict[str, Any]:
    """Hide ``item_id`` from their view. Her workspace keeps it: this is not a forget."""
    return _act(user_id, item_id, "hide", None, now)[0]
