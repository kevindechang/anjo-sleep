"""Replay's records and registered settings (Task 17b). Pure: frozen dataclasses and constants.

Replay trains how she investigates, on synthetic lives only. A **start state** is the workspace
at the end of one night of a development life's M run, once for each open or active question with
at least two hypotheses; its **frozen proposition** is that question's hypothesis set plus the
catch-all, shared by every branch. A **branch** runs one night's inquiry from the start state with
every tool masked to that night, and ends with a credence vector over the frozen set. Spark Open
**labels** each start state once, from the evidence after it; the branches are scored against
that one label.

The settings below are registered (Task 17's protocol): ``REPLAY_K`` and ``BRANCH_TEMPERATURE``
are the spec's; ``TOKEN_PENALTY`` (λ) is set on development. Its default makes a night spent to
the 20,000-token cap cost 0.2, a tenth of the Brier score's range of 2, so tokens only decide
between branches whose beliefs end about equally right.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.mind.inquiry.types import Turn, WorkspaceEvent
from anjo.sleep.types import Target

REPLAY_K: Final = 4  # branches per start state
BRANCH_TEMPERATURE: Final = 0.8
TOKEN_PENALTY: Final = 0.01  # λ: score points per 1,000 tokens, registered on development
UNDETERMINED: Final = "undetermined"
STOPPED: Final = "stop"  # vLLM's finish_reason for a reply that ended at a stop token
APD_LIFE_ID: Final = re.compile(r"^development-[a-z0-9]+(?:-[a-z0-9]+)*-[0-9]+$")


class ReplayRefused(PermissionError):
    """Replay was handed something it never trains on: a life that isn't a synthetic development
    life, or a run that isn't that life's M run."""


def check_synthetic(life: Life) -> None:
    """Refuse a life that isn't a synthetic development life: an APD development id (``Life``
    already holds every item and target to its own id), with every session and item Spark Open's
    writing."""
    if life.split != "development" or not APD_LIFE_ID.match(life.id):
        raise ReplayRefused(f"replay runs on synthetic development lives only, not {life.id!r}")
    if any(session.model != SPARK_OPEN for session in life.sessions) or any(
        item.model != SPARK_OPEN for item in life.evidence
    ):
        raise ReplayRefused(f"life {life.id} holds writing Spark Open didn't produce")


@dataclass(frozen=True)
class StartState:
    """One start state: the life's log up to the end of ``night`` (``events``), the question it
    branches on, and that question's frozen hypothesis set, ``(id, statement)`` pairs with the
    catch-all last. ``as_of`` is the night's tool mask; ``ancestry`` every model whose text the
    workspace holds (the life's evidence and the M run's thinker turns)."""

    id: str
    life_id: str
    night: date
    as_of: datetime
    question_id: str
    hypotheses: tuple[tuple[str, str], ...]
    events: tuple[WorkspaceEvent, ...]
    targets: tuple[Target, ...]
    ancestry: frozenset[str] | None

    @property
    def frozen_ids(self) -> tuple[str, ...]:
        return tuple(hypothesis_id for hypothesis_id, _ in self.hypotheses)


@dataclass(frozen=True)
class Label:
    """Spark Open's label for one start state: the frozen hypothesis that holds, or ``None``
    (undetermined). ``model`` is the model the API reported writing it, as reported (``""`` when no
    call answered). ``failed`` is a call that failed or answered out of form, read as undetermined."""

    start_id: str
    holds: str | None
    model: str
    tokens: int = 0
    failed: bool = False

    @property
    def usable(self) -> bool:
        """Whether its start state may branch and train: determined, and written by Spark Open by
        the API's own report. Any other model, or none reported, isn't cleared."""
        return self.holds is not None and self.model == SPARK_OPEN


@dataclass(frozen=True)
class CallEnd:
    """How one thinker call ended: vLLM's ``finish_reason`` (``STOPPED`` at a stop token,
    ``"length"`` cut at its token cap), or ``failed`` (no reply: empty text charged its cap)."""

    finish: str | None
    failed: bool = False


@dataclass(frozen=True)
class Branch:
    """One branch's end: its credence vector over the frozen set (a hypothesis the branch added
    is counted in the catch-all), the tokens it spent, its thinker turns on the start state's
    question (its trajectory), and every model whose text its prompts held. ``ends`` is how each
    of those turns' calls ended, ``applied`` the turns whose operation the kernel applied,
    ``failed`` the branch's failed thinker calls and ``day`` the date its prompts were rendered
    on. ``complete``: no call failed, at least one turn is a target, and the question still has
    hypotheses at the end."""

    start_id: str
    index: int
    seed: int
    credences: Mapping[str, float] = field(hash=False)
    tokens: int
    turns: tuple[Turn, ...]
    complete: bool
    ancestry: frozenset[str] | None
    ends: Mapping[int, CallEnd] = field(default_factory=dict, hash=False)
    applied: frozenset[int] = frozenset()
    failed: int = 0
    day: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "credences", MappingProxyType(dict(self.credences)))
        object.__setattr__(self, "ends", MappingProxyType(dict(self.ends)))

    @property
    def targets(self) -> tuple[Turn, ...]:
        """The turns a dataset trains on: those the kernel applied whose reply ended at a stop
        token. A rejected or malformed turn, or a reply cut at its token cap, never is (an earlier
        one may still shape a later prompt)."""
        return tuple(
            turn for turn in self.turns
            if turn.seq in self.applied and (end := self.ends.get(turn.seq)) is not None
            and end.finish == STOPPED and not end.failed
        )


@dataclass(frozen=True)
class Scored:
    """A branch with its score against its start state's label."""

    branch: Branch
    brier: float
    score: float


class LabelRewrite(ValueError):
    """A second label for a start state: each is labelled once, and every branch shares it."""


@dataclass(frozen=True)
class LabelBook:
    """The labels, one per start state, written once: ``with_label`` returns a new book and
    refuses a start state already labelled."""

    labels: Mapping[str, Label] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))

    def with_label(self, label: Label) -> LabelBook:
        if label.start_id in self.labels:
            raise LabelRewrite(f"start state {label.start_id} is already labelled")
        return LabelBook({**self.labels, label.start_id: label})

    def holds(self, start_id: str) -> str | None:
        """The start state's label (``None``: undetermined); a start state never labelled is an
        error, since no branch runs before its label is written."""
        if start_id not in self.labels:
            raise KeyError(f"start state {start_id} has no label")
        return self.labels[start_id].holds
