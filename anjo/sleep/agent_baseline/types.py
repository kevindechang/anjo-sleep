"""What the baseline arm reads and leaves (arm 2A): one night's call, its outcome, and its port.

The values are frozen and hold no text but the brief it was given: a night's call names the person,
the night's clock, the store and the meters it spends from, the step's hold key and the cancel flag.
``AgentBaseline`` is the one method the night calls (``night.run``), so a fake stands in for the arm
in tests and the night never imports the model call.

Nothing here reads the wall clock, and every time is on the night's clock.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Protocol

from anjo.sleep.mind.inquiry.ports import Research
from anjo.sleep.night.meter import SpendMeter, WonderMeter
from anjo.sleep.ports import EvidenceStore

AgentOp = Literal["recall", "research", "note", "done"]


@dataclass(frozen=True)
class AgentStep:
    """One operation the baseline's model asked for: a tool call, a note, or its own stop.

    ``query`` is set for the two tools and ``text``/``cites`` for a note, so a step that needs
    neither (``done``) carries nothing.
    """

    op: AgentOp
    query: str | None = None
    text: str | None = None
    cites: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentCall:
    """One person's night as the baseline reads it: the store and brief it starts from, the night's
    clock (``as_of``) and its records' stamp (``at``), the meters its calls spend from, the step's
    hold key, and the cancel flag, read before each write (RF2). ``brief`` is the served brief as of
    last night, or None before her first fold."""

    user_id: str
    night: date
    as_of: datetime
    at: datetime
    store: EvidenceStore
    wonder: WonderMeter
    spend: SpendMeter
    job: str
    brief: str | None = None
    cancelled: Callable[[], bool] = field(default=lambda: False)


@dataclass(frozen=True)
class AgentOutcome:
    """What one pass left: the notes it wrote, as evidence ids, and content-free counts.

    ``stopped`` says why the pass ended: ``done`` (its own stop, or its note cap), ``token_cap``,
    ``step_cap``, ``malformed`` (a reply it couldn't read), ``budget`` (its hold found no room) or
    ``failed`` (a failure inside the pass, which never fails the night)."""

    notes: tuple[str, ...] = ()
    tokens_used: int = 0
    steps: int = 0
    calls: int = 0
    usd: float = 0.0
    stopped: str | None = None
    receipt: Mapping[str, object] = field(default_factory=dict)


class AgentBaseline(Protocol):
    """The night's baseline step (arm 2A): one stateless pass over one person's evidence and brief."""

    def night(self, call: AgentCall) -> AgentOutcome: ...


ResearchFor = Callable[[AgentCall], Research]
