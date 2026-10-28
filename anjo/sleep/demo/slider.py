"""The night slider's stops (Task 13a, full step): her weights night by night, notes off.

The stops are night 0 (the base model), the nights nearest 3 and 7 that she reached, and her latest
night, each with its real number. A night's number counts the distinct nights she has run, from 1;
a night run again on the same date is one night, and its last run's status stands.

Each stop answers with the checkpoint serving after that night: the last one promoted on or before
it. A rejected, ``nothing_new``, napped or failed night has no checkpoint of its own, so its stop shows the
last promoted one, from an earlier night, and says which (``from_night``, ``fallback``); with none
promoted yet, that is the base model. A checkpoint a rollback took out of service is not in the
registry's history, so it is never a stop's.

Every stop is notes off: arm 3 on its checkpoint, arm 1 at night 0. The results it shows are
labeled development (spec §2.8). A stop whose checkpoint was trained under another rendering than
the one serving now is ``available=False``: it answers nothing, pre-generated or live, and never
falls back to another model (``VllmClient`` would refuse it too).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

from anjo.sleep.types import CheckpointRef

DEVELOPMENT: Final = "development"
STOP_NIGHTS: Final = (3, 7)  # besides night 0 and her latest


@dataclass(frozen=True)
class NightMark:
    """One night she ran: its date and its status (``NightStatus``)."""

    night: date
    status: str


@dataclass(frozen=True)
class Stop:
    """One stop. ``checkpoint`` None is the base model. ``from_night`` is the number of the night
    whose checkpoint answers (0 for the base); ``status`` is the stop's own night's, None at 0."""

    key: str
    night: int
    checkpoint: CheckpointRef | None
    from_night: int
    status: str | None
    latest: bool
    available: bool

    @property
    def fallback(self) -> bool:
        """The stop's own night promoted nothing, so an earlier night's checkpoint answers."""
        return self.night != self.from_night


class Pinned:
    """An ``AdapterRegistry`` serving one checkpoint for one person: arm 3 at a stop."""

    def __init__(self, person_id: str, checkpoint: CheckpointRef | None) -> None:
        self._person = person_id
        self._checkpoint = checkpoint

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        return self._checkpoint if person_id == self._person else None


def _numbered(nights: Sequence[NightMark]) -> list[NightMark]:
    """One mark per date, in date order, each carrying the last run's status."""
    last = {mark.night: mark for mark in nights}
    return [last[day] for day in sorted(last)]


def _number(marks: Sequence[NightMark], on: date) -> int:
    """The number of the night on ``on``: how many nights ran up to it."""
    return sum(1 for mark in marks if mark.night <= on)


def _serving(history: Sequence[CheckpointRef], on: date) -> CheckpointRef | None:
    found = [ckpt for ckpt in history if ckpt.night <= on]
    return found[-1] if found else None


def plan_stops(nights: Sequence[NightMark], history: Sequence[CheckpointRef], *, rendering: str) -> tuple[Stop, ...]:
    """The slider's stops, night 0 first. ``history`` is the registry's promoted checkpoints,
    oldest first; ``rendering`` the one serving now."""
    marks = _numbered(nights)
    wanted = sorted({n for n in STOP_NIGHTS if n <= len(marks)} | ({len(marks)} if marks else set()))
    stops = [Stop("n0", 0, None, 0, None, False, True)]
    for n in wanted:
        mark = marks[n - 1]
        ckpt = _serving(history, mark.night)
        stops.append(Stop(
            key=f"n{n}", night=n, checkpoint=ckpt, from_night=0 if ckpt is None else _number(marks, ckpt.night),
            status=mark.status, latest=n == len(marks), available=ckpt is None or ckpt.rendering == rendering,
        ))
    return tuple(stops)


def latest_stop(stops: Sequence[Stop]) -> Stop | None:
    """Her latest night's stop; None before her first night."""
    return next((stop for stop in stops if stop.latest), None)


def find(stops: Sequence[Stop], key: str) -> Stop | None:
    return next((stop for stop in stops if stop.key == key), None)
