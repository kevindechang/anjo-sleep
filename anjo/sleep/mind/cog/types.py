"""The contract of MIND's cognition sprint (2026-11-02): a world, a predictor, an inquirer.

A ``Life`` is one simulated person's days, generated from a hidden state the predictor never
sees. Each day the predictor ``observe``s that day's events, may ask the person one ``Question``
(the answer arrives as an ``answer`` event), ``predict``s the day's queries about later days, and
then ``end_day`` runs (the night: any offline update). A query asked on day ``d`` is about a day
after ``d``; no event of day ``d`` carries its outcome. ``Life.resolve`` gives the outcome and two
references: ``oracle_p``, the Bayes filter that knows the person's parameters but not the hidden
state (the honest ceiling), and ``state_p``, which also knows the hidden state (an upper bound).

Everything here is pure and frozen: no clock, file, model or network.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

EventKind = Literal["did", "skipped", "plan", "cancel", "mood", "mention", "message", "answer"]
EVENT_KINDS: Final[tuple[EventKind, ...]] = (
    "did", "skipped", "plan", "cancel", "mood", "mention", "message", "answer",
)
# Which queries need more than a per-person count to answer well. Fixed by how the world made
# them, before any predictor runs: "count" outcomes depend only on the person's fixed habits;
# "hidden" outcomes also depend on a hidden state or a hidden fact the person never states.
QueryFamily = Literal["count", "hidden"]
QUERY_FAMILIES: Final[tuple[QueryFamily, ...]] = ("count", "hidden")


@dataclass(frozen=True)
class Event:
    day: int
    kind: EventKind
    subject: str  # an activity, a person in their life, a topic, or a fact id
    text: str  # the event as the person's own words (templated or paraphrased)
    tokens: tuple[str, ...]  # the same event as structured tokens; holds no more than ``text``
    value: float | None = None  # e.g. a mood level, or the day a plan is for


@dataclass(frozen=True)
class Query:
    qid: str
    asked_day: int
    target_day: int  # > asked_day
    subject: str
    text: str  # e.g. "Will they go on their run on Thursday?"
    tokens: tuple[str, ...]
    family: QueryFamily


@dataclass(frozen=True)
class Resolution:
    qid: str
    outcome: bool
    oracle_p: float  # Bayes filter with the true parameters, hidden state inferred
    state_p: float  # true parameters and the true hidden state


@dataclass(frozen=True)
class Question:
    question_id: str
    day: int
    subject: str
    text: str  # e.g. "How is work going this week?"
    tokens: tuple[str, ...]
    cost: float  # the interruption cost, in the same units as the inquiry score


@dataclass(frozen=True)
class LifeMeta:
    """What a predictor may know before day 0: names, not habits."""
    person_id: str
    n_days: int
    start_weekday: int  # 0 = Monday
    activities: tuple[str, ...]
    people: tuple[str, ...]
    topics: tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    qid: str
    p: float  # probability the query's outcome is True, in (0, 1)
    u: float | None = None  # epistemic uncertainty on any monotone scale; None if not modelled


@dataclass(frozen=True)
class Truth:
    """For scoring only; never shown to a predictor or inquirer."""
    shift_days: tuple[int, ...]  # days a regime shift took effect
    hidden_facts: Mapping[str, object] = field(default_factory=dict)
    params: Mapping[str, object] = field(default_factory=dict)


class Life(Protocol):
    meta: LifeMeta

    def events(self, day: int) -> tuple[Event, ...]: ...
    def queries(self, day: int) -> tuple[Query, ...]: ...
    def questions(self, day: int) -> tuple[Question, ...]: ...
    def answer(self, question_id: str) -> Event: ...
    def resolve(self, qid: str) -> Resolution: ...
    def truth(self) -> Truth: ...


class Predictor(Protocol):
    name: str

    def begin(self, meta: LifeMeta) -> None: ...
    def observe(self, day: int, events: Sequence[Event]) -> None: ...
    def predict(self, queries: Sequence[Query]) -> Sequence[Prediction]: ...
    def end_day(self, day: int) -> None: ...


class Inquirer(Protocol):
    name: str

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        """A question id to ask now, or None to stay silent."""
        ...
