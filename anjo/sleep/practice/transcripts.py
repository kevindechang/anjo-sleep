"""Practice transcripts and where they are kept (Task 5).

A practice transcript is one scene: the person's message (Spark Open playing them), her reply
(the training target: Glimmer with context F, or under full practice the critic's best candidate)
and the snapshot of context C taken before tonight's practice, which becomes the example's input.
Full practice (Task 7) also keeps where the target came from (``target``: ``glimmer`` or
``teacher``) and the critic's worst candidate (``rejected``), for the night's preference pairs.
The night stores each transcript twice:

- as a ``practice`` evidence item, readable in her served context and folded into the next brief;
- here, in the person's transcript log, with the immutable C snapshot and the scene's parts, so
  the dataset never rebuilds an input after the target exists.

The log is an append-only JSON Lines file. The codec and the writer guard are always named: the
app passes encryption and the account's guard, tests and the public example pass ``PLAIN``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

from anjo.sleep.evidence.jsonl import Codec, Guard, LogError, append_records, read_records
from anjo.sleep.types import Evidence

Candidate = Literal["glimmer", "teacher"]
CANDIDATES: tuple[Candidate, ...] = get_args(Candidate)


@dataclass(frozen=True)
class PracticeTranscript:
    """One practice scene. ``context`` is C before tonight's practice (None when it was empty),
    ``f_ids`` the evidence the reply's context F held, and ``ancestry`` every model the scene's
    text derives from: Spark Open, Glimmer and everything F and C derive from."""

    id: str
    person_id: str
    night: date
    scene: int
    message: str
    reply: str
    context: str | None
    f_ids: tuple[str, ...]
    ancestry: frozenset[str] | None
    created_at: datetime
    target: Candidate = "glimmer"
    rejected: str | None = None
    rejected_from: Candidate | None = None
    core: str | None = None  # the core state the reply was made under (``night.core_snapshot``, as JSON), channel on

    def evidence(self) -> Evidence:
        """The transcript as a ``practice`` evidence item."""
        return Evidence(
            id=self.id, person_id=self.person_id, day=self.night, kind="practice",
            text=f"They asked: {self.message}\nShe answered: {self.reply}",
            source=f"practice:{self.night.isoformat()}:{self.scene}", held_out=False,
            created_at=self.created_at, model=None, ancestry=self.ancestry,
        )


def transcript_record(item: PracticeTranscript) -> dict[str, Any]:
    return {
        "id": item.id, "person_id": item.person_id, "night": item.night.isoformat(),
        "scene": item.scene, "message": item.message, "reply": item.reply, "context": item.context,
        "f_ids": list(item.f_ids),
        "ancestry": None if item.ancestry is None else sorted(item.ancestry),
        "created_at": item.created_at.isoformat(),
        "target": item.target, "rejected": item.rejected, "rejected_from": item.rejected_from,
        **({} if item.core is None else {"core": item.core}),
    }


def _candidate(value: object, *, optional: bool) -> Any:
    if (value is None and optional) or value in CANDIDATES:
        return value
    raise ValueError("not a practice candidate")


def transcript_from_record(record: Mapping[str, Any]) -> PracticeTranscript:
    try:
        ancestry = record["ancestry"]
        context = record["context"]
        rejected = record.get("rejected")
        scene = record["scene"]
        if isinstance(scene, bool) or not isinstance(scene, int):
            raise TypeError("scene")
        return PracticeTranscript(
            id=str(record["id"]), person_id=str(record["person_id"]),
            night=date.fromisoformat(record["night"]), scene=scene,
            message=str(record["message"]), reply=str(record["reply"]),
            context=None if context is None else str(context),
            f_ids=tuple(str(item) for item in record["f_ids"]),
            ancestry=None if ancestry is None else frozenset(str(name) for name in ancestry),
            created_at=datetime.fromisoformat(record["created_at"]),
            target=_candidate(record.get("target", "glimmer"), optional=False),
            rejected=None if rejected is None else str(rejected),
            rejected_from=_candidate(record.get("rejected_from"), optional=True),
            core=None if record.get("core") is None else str(record["core"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LogError(f"not a practice transcript ({type(error).__name__})") from None


class TranscriptStore(Protocol):
    def add(self, items: Sequence[PracticeTranscript]) -> None:
        """Keep each transcript once, by id."""

    def get(self, item_id: str) -> PracticeTranscript | None:
        """The transcript stored under ``item_id``, or None."""

    def for_night(self, night: date) -> list[PracticeTranscript]:
        """The transcripts practised on ``night``, in scene order."""


class JsonlTranscriptStore:
    """One person's transcript log over one file."""

    def __init__(self, path: Path, person_id: str, *, codec: Codec, guard: Guard) -> None:
        self.path = path
        self.person_id = person_id
        self._codec = codec
        self._guard = guard

    def _all(self) -> dict[str, PracticeTranscript]:
        return {
            item.id: item
            for item in (transcript_from_record(record) for record in read_records(self.path, self._codec))
        }

    def add(self, items: Sequence[PracticeTranscript]) -> None:
        if any(item.person_id != self.person_id for item in items):
            raise ValueError("a transcript for another person")
        with self._guard():
            known = self._all()
            fresh = [transcript_record(item) for item in items if item.id not in known]
            append_records(self.path, self._codec, fresh)

    def get(self, item_id: str) -> PracticeTranscript | None:
        return self._all().get(item_id)

    def for_night(self, night: date) -> list[PracticeTranscript]:
        return sorted((item for item in self._all().values() if item.night == night), key=lambda item: item.scene)

    def export(self) -> dict[str, object]:
        return {"practice": [transcript_record(item) for item in self._all().values()]}
