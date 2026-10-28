"""MIND's neural core: the values it reads and returns. Pure and torch-free.

An event is one evidence item of a life, or one turn or finding of a real person, read as a
vector from Glimmer's own hidden states (``EventLatent``: the latents module's record, kept in
``event`` so nothing here loads the encoder, the one the core reads; a ``LabelledStream`` checks each event's vector is one finite row and its time aware). The core
binds events into slots, one per open thread, and answers per slot: will this thread fail within
``HORIZON_DAYS`` days, is a prerequisite missing, and how wrong is its own forecast likely to be.
Training reads ``ThreadLabel`` rows: which events make up a thread, from when its problem is
knowable and when it surfaced. Imagined states never become evidence: an ``Ignition`` cites only
ids of events the core observed.

The package: ``event``, ``types``, ``state``, ``store``, ``curriculum``, ``toy``, ``data``, ``findings``
and ``bridge`` are torch-free, and all but ``toy`` and ``data`` load no training code either, so the
night and the app import the bridge; ``net``, ``api``, ``train``, ``evaluate`` and the
CLI (``python -m anjo.sleep.mind.core``) need torch, on the CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal, get_args

import numpy as np

from anjo.sleep.mind.core.event import EventLatent as EventLatent  # the one event record

HORIZON_DAYS: Final = 7  # the hazard heads look this many days ahead
KINDS: Final = (
    "lesson", "correction", "preference", "tone_request", "homework", "practice", "turn", "finding", "other",
)
IgnitionKind = Literal["risk", "missing", "collision"]
IGNITION_KINDS: Final = frozenset(get_args(IgnitionKind))


def kind_index(kind: str) -> int:
    """The kind's row in ``KINDS``; a kind the core doesn't know reads as ``other``."""
    return KINDS.index(kind) if kind in KINDS else KINDS.index("other")


def days_of(t: datetime) -> float:
    """``t`` on the core's clock: days since the POSIX epoch."""
    if t.tzinfo is None:
        raise ValueError("the core's times are timezone-aware")
    return t.timestamp() / 86_400.0


@dataclass(frozen=True)
class CoreConfig:
    """The shared core's shape. ``fast`` off is the no-fast-weights ablation (N-nofast);
    ``memory`` off resets the state before every event: the no-memory baseline. ``stance`` reads
    the fast memory by topic, whatever the stance, and gives each slot update the product of the
    recalled stance and the event's own (``net``): the stance probe for contradictions."""

    dim_in: int
    width: int = 64
    slots: int = 16
    horizon: int = HORIZON_DAYS
    fast: bool = True
    memory: bool = True
    theta_max: float = 0.5  # the fast memory's largest step: 2·theta <= 1, so a step never overshoots
    free_below: float = 0.05  # a slot whose usage fell below this is free (see ``net``)
    cite_min: float = 0.2  # an event is cited by a slot it wrote into at least this much
    cite_keep: int = 8  # cited events a slot keeps, the strongest writes first
    dream_gap_days: float = 0.25  # an imagined event's distance from the one before
    stance: bool = False

    def __post_init__(self) -> None:
        for name in ("dim_in", "width", "slots", "horizon", "cite_keep"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 < self.theta_max <= 0.5:
            raise ValueError("theta_max is in (0, 0.5]")


@dataclass(frozen=True)
class ThreadLabel:
    """A labelled thread of one life: what training and the offline eval read.

    ``event_ids`` are the events that make up the thread (training binds them to one slot);
    ``origin_ids`` are the ids an intervention must cite to catch it (APD's match rule). A failing
    thread (``failed``) is a problem: knowable from ``knowable_at`` and raised by the person at
    ``outcome_at``. A thread that doesn't fail (a decoy, a reversal, a benign plan) is never worth
    an intervention. ``missing`` marks a missing prerequisite (a gap). ``template`` is the planted
    item's template, when it has one: training holds lives back by it.
    """

    id: str
    family: str
    event_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    knowable_at: datetime
    outcome_at: datetime
    failed: bool
    missing: bool = False
    template: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.event_ids or not self.origin_ids:
            raise ValueError("a label names its id, its events and its origin")
        if self.outcome_at < self.knowable_at:
            raise ValueError(f"label {self.id} surfaces before it is knowable")


def _check_event(event: EventLatent) -> None:
    if not event.event_id or not event.life_or_person:
        raise ValueError("an event names its id and its life or person")
    days_of(event.t)
    vec = np.asarray(event.vec)
    if vec.ndim != 1 or vec.size == 0 or not np.all(np.isfinite(vec)):
        raise ValueError(f"event {event.event_id}'s vector is one finite row")


@dataclass(frozen=True)
class LabelledStream:
    """One life's events in time order, its days (the person-days it counts for) and its labels."""

    life_id: str
    days: tuple[date, ...]
    events: tuple[EventLatent, ...]
    labels: tuple[ThreadLabel, ...]

    def __post_init__(self) -> None:
        for event in self.events:
            _check_event(event)
        if any(b.t < a.t for a, b in zip(self.events, self.events[1:])):
            raise ValueError(f"life {self.life_id}'s events run in time order")
        ids = [e.event_id for e in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError(f"an event id appears twice in life {self.life_id}")
        held = frozenset(ids)
        for label in self.labels:
            if not set(label.event_ids) <= held:
                raise ValueError(f"label {label.id} names an event life {self.life_id} doesn't hold")
        if any(e.t.date() not in frozenset(self.days) for e in self.events):
            raise ValueError(f"an event of life {self.life_id} falls outside its days")


# ── What the core answers ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class SlotPrediction:
    """One open slot: its risk of failing within the horizon (and the daily hazard behind it), a
    missing prerequisite, the core's predicted squared error of its own risk, its share of the
    choice, and the real events it cites."""

    slot: int
    usage: float
    risk: float
    hazard: tuple[float, ...]
    missing: float
    self_error: float
    choice: float
    cites: tuple[str, ...]


@dataclass(frozen=True)
class Predictions:
    speak: float
    none: float
    slots: tuple[SlotPrediction, ...]

    def slot(self, index: int) -> SlotPrediction | None:
        return next((s for s in self.slots if s.slot == index), None)


@dataclass(frozen=True)
class Ignition:
    """The core writing into the workspace: a slot's risk (or a missing step) crossed the ignition
    line in its dreams, or two slots' futures collided. ``cited_event_ids`` are real events only."""

    slot: int
    other: int | None
    kind: IgnitionKind
    score: float
    cited_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in IGNITION_KINDS:
            raise ValueError(f"an ignition is one of {sorted(IGNITION_KINDS)}")
        if (self.kind == "collision") != (self.other is not None):
            raise ValueError("a collision names two slots; any other ignition one")
        if not math.isfinite(self.score) or not self.cited_event_ids:
            raise ValueError("an ignition has a finite score and cites real events")


@dataclass(frozen=True)
class Decision:
    """Speak or wait, the slot to raise, and the ignitions to write into the workspace."""

    speak: bool
    slot: int | None
    p_speak: float
    ops: tuple[Ignition, ...] = ()

    def __post_init__(self) -> None:
        if self.speak and self.slot is None:
            raise ValueError("speaking names the slot it raises")
