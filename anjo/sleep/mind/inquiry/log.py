"""Her workspace log, the ``WorkspaceLog`` port (Task 17): on Task 3's JSON Lines machinery for a
person, or in memory for an APD run.

Each event is one record (``codec.event_to_json``), appended under the file's lock with the codec
and writer guard the caller names: the app passes the account's encryption and guard
(``anjo.sleep.app.storage.workspace_log``), tests and the public example ``PLAIN`` and
``NO_GUARD``. An append is refused unless its step's seq is the log's next and its ``before`` is
the log's last ``after``, so the log stays one unbroken chain. The live digest is the last event's
``after``; ``replay_digest`` folds the whole log from scratch, and the two must agree.
``records_digest`` covers her records alone, without the run history: the part of the person's
digest a resumed night must reproduce.

The export folds under the denylist each run recorded; a log no longer foldable (a version this
build doesn't know) still exports its events, with ``state`` None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from anjo.sleep.evidence.jsonl import Codec, Guard, LogError, append_records, read_records
from anjo.sleep.mind.inquiry.codec import CodecError, event_from_json, event_to_json, records_document
from anjo.sleep.mind.inquiry.kernel import digest, fold, records_digest
from anjo.sleep.mind.inquiry.types import Workspace, WorkspaceEvent

EMPTY_DIGEST: Final = digest(Workspace())


class WorkspaceLogError(ValueError):
    """An append that doesn't continue the log."""


def _check_next(events: tuple[WorkspaceEvent, ...], event: WorkspaceEvent) -> None:
    if event.step.seq != len(events):
        raise WorkspaceLogError(f"event {event.step.seq} where {len(events)} comes next")
    last = events[-1].after if events else EMPTY_DIGEST
    if event.before != last:
        raise WorkspaceLogError(f"event {event.step.seq} doesn't continue the log's last digest")


def _export(events: tuple[WorkspaceEvent, ...]) -> dict[str, object]:
    try:
        state: dict[str, object] | None = records_document(fold(events))
    except ValueError:
        state = None  # her events are still hers to take
    return {"workspace": {"events": [event_to_json(e) for e in events], "state": state}}


class MemoryWorkspaceLog:
    """An in-memory log, for an APD run and the tests."""

    def __init__(self, events: tuple[WorkspaceEvent, ...] = ()) -> None:
        self._events: tuple[WorkspaceEvent, ...] = ()
        for event in events:
            self.append(event)

    def append(self, event: WorkspaceEvent) -> None:
        _check_next(self._events, event)
        self._events = (*self._events, event)

    def events(self) -> tuple[WorkspaceEvent, ...]:
        return self._events

    def records_digest(self) -> str:
        return records_digest(fold(self._events))


class JsonlWorkspaceLog:
    """One person's workspace log over one file. ``guard`` serializes writers across processes."""

    def __init__(self, path: Path, person_id: str, *, codec: Codec, guard: Guard) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard

    def events(self) -> tuple[WorkspaceEvent, ...]:
        out: list[WorkspaceEvent] = []
        for number, record in enumerate(read_records(self.path, self._codec), start=1):
            try:
                out.append(event_from_json(record))
            except CodecError as exc:
                raise LogError(f"{self.path.name} line {number}: {exc}") from None
        return tuple(out)

    def append(self, event: WorkspaceEvent) -> None:
        record: dict[str, Any] = event_to_json(event)
        with self._guard():
            _check_next(self.events(), event)
            append_records(self.path, self._codec, [record])

    def workspace(self) -> Workspace:
        return fold(self.events())

    def digest(self) -> str:
        """The live digest: the last event's ``after``."""
        events = self.events()
        return events[-1].after if events else EMPTY_DIGEST

    def replay_digest(self) -> str:
        """The digest of the log folded from scratch; it must equal ``digest``."""
        return digest(fold(self.events()))

    def records_digest(self) -> str:
        """The digest of her records alone, folded from scratch (``kernel.records_digest``)."""
        return records_digest(fold(self.events()))

    def export(self) -> dict[str, object]:
        return _export(self.events())
