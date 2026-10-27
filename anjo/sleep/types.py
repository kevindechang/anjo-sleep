"""Shared types for Anjo Sleep (spec §12.1): the values every task builds against.

Each is a frozen dataclass, so tasks can build against one contract in parallel. Collections
inside them are tuples and frozensets. The two mappings on ``NightResult`` are copied when it is
built, typed read-only and left out of its hash. Every time here is on the night's clock, which
is virtual in accelerated runs; nothing in this module reads the wall clock.

Smaller types that one task names (``Source``, ``Change``, ``HeldNote``) live in that task's own
module. ``Commitment`` and ``QuestionView`` are here because ``WantSnapshot`` and the
``IntentionSource`` port carry them.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

EvidenceKind = Literal["lesson", "correction", "preference", "tone_request", "homework", "practice"]
ProbeKind = Literal["tone_hold", "taste_ab", "vocab", "pushback", "new_work"]
GateName = Literal["general", "collapse", "safety"]
NightStatus = Literal["promoted", "rejected", "nothing_new", "failed"]
CostPart = Literal["extract", "training", "practice", "gates", "brief", "mind"]
TargetKind = Literal["commitment", "thread"]
TargetStatus = Literal["open", "paused", "closed"]
ResultBasis = Literal["research", "evidence"]
QuestionStatus = Literal["open", "active", "parked", "concluded", "retracted", "abandoned"]
WantKind = Literal["follow_up", "understand", "share"]
Action = Literal["prepare_commitment", "revisit_thread", "inquire", "hold_note"]
Opportunity = Literal["away"]
ConsideredStatus = Literal["won", "held_back", "rejected"]
ExpectationOutcome = Literal["needed", "not_needed", "censored"]

MAX_NIGHT_ACTIONS = 2  # one study and one note a night (Task 5)


def _check_unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


# ── Evidence and tests (Tasks 2 and 3) ────────────────────────────────────────


@dataclass(frozen=True)
class Evidence:
    """One item the night can learn from (Task 3).

    ``model`` names what wrote the text (``None`` for the person's own words). ``ancestry`` is
    every model whose output the item derives from, transitively: empty for the person's own
    words, ``None`` when unknown, which training refuses. ``target_id`` links the item to the
    target it mentions, if any.
    """

    id: str
    person_id: str
    day: date
    kind: EvidenceKind
    text: str
    source: str
    held_out: bool
    created_at: datetime
    model: str | None = None
    ancestry: frozenset[str] | None = None
    target_id: str | None = None


@dataclass(frozen=True)
class Probe:
    """One sealed test item (Task 2). ``options`` is the label pair for ``taste_ab`` only."""

    id: str
    person_id: str
    kind: ProbeKind
    prompt: str
    options: tuple[str, str] | None
    expected: str | None
    source_ids: tuple[str, ...]


# ── Checkpoints, gates and the night (Tasks 4 and 5) ──────────────────────────


@dataclass(frozen=True)
class CheckpointRef:
    """A trained adapter, or a merged checkpoint, for one person and night.

    ``id`` is the adapter name (``p-<hash>-n<night>``), ``uri`` where the GPU runner stored it,
    and ``parent_id`` the checkpoint it trained on top of (``None`` for a clean slate).
    """

    id: str
    person_id: str
    night: date
    uri: str
    parent_id: str | None


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict on a checkpoint (Task 5).

    ``value`` is the number the gate compares with ``threshold``: points of general accuracy
    lost, the collapse rate above base, or new safety trips. ``reason`` says why it failed.
    """

    gate: GateName
    passed: bool
    value: float
    threshold: float
    reason: str | None = None


@dataclass(frozen=True)
class NightAllowance:
    """What the night may still spend (Task 5).

    ``actions_left`` is at most two (one study, one note) and never more than the person's
    rolling background allowance. ``calls_left`` and ``usd_left`` are Wonder's calls and dollars.
    """

    actions_left: int
    calls_left: int
    usd_left: float

    def __post_init__(self) -> None:
        if not 0 <= self.actions_left <= MAX_NIGHT_ACTIONS:
            raise ValueError(f"actions_left must be in [0, {MAX_NIGHT_ACTIONS}], got {self.actions_left}")
        if self.calls_left < 0 or self.usd_left < 0:
            raise ValueError("calls_left and usd_left must not be negative")

    @property
    def available_model_calls(self) -> int:
        """Actions a Wonder-backed choice may take: ``actions_left`` while Wonder's calls and
        dollars remain, else 0. ``inquire`` and ``hold_note`` don't need it."""
        return self.actions_left if self.calls_left > 0 and self.usd_left > 0 else 0


# ── Targets and results (Task 3) ──────────────────────────────────────────────


@dataclass(frozen=True)
class Target:
    """Something open the chooser can act on (Task 3).

    A ``commitment`` is a plan the person dated (their date is ``check_after``). A ``thread`` is
    a topic she was lost on or that stays unresolved. Only the person's own evidence raises
    ``revision``. ``acted_revision`` is the revision a study last consumed. ``confidence`` is her
    heuristic confidence that she understood a thread; it never describes the person.
    ``due_noted`` is set once a commitment's due note is held or declined.
    """

    id: str
    person_id: str
    kind: TargetKind
    label: str
    check_after: datetime | None
    status: TargetStatus
    revision: int
    acted_revision: int
    confidence: float | None
    last_mentioned: date
    due_noted: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.acted_revision <= self.revision:
            raise ValueError(
                f"acted_revision must be in [0, revision], got {self.acted_revision} > {self.revision}"
            )
        if self.confidence is not None:
            _check_unit("confidence", self.confidence)

    @property
    def unacted(self) -> int:
        """Revisions the person added since a study last acted on this target."""
        return self.revision - self.acted_revision


@dataclass(frozen=True)
class Result:
    """A committed study result on a target (Task 3).

    ``homework_id`` names the homework behind a ``research`` result and is ``None`` for an
    ``evidence`` result, which comes from one of her findings (Task 17). ``user_cites`` counts the
    person's own items it cites.
    """

    id: str
    target_id: str
    homework_id: str | None
    verified_sources: int
    created_at: datetime
    consumed: bool
    basis: ResultBasis = "research"
    user_cites: int = 0


# ── Her chooser (Tasks 7a, 16 and 17) ─────────────────────────────────────────


@dataclass(frozen=True)
class Policy:
    """The chooser's registries as one value (Task 7a). The app builds it from Anjo's registries;
    the standalone example uses a labeled public default."""

    priority_classes: tuple[str, ...]
    max_candidates: int
    away_min: float
    surface_min: float
    reason_codes: frozenset[str]


@dataclass(frozen=True)
class Commitment:
    """One open entry of the person's intentions ledger (``IntentionSource``)."""

    id: str
    label: str
    check_after: datetime | None


@dataclass(frozen=True)
class QuestionView:
    """What the chooser sees of one question in her workspace (Task 17)."""

    id: str
    target_ids: tuple[str, ...]
    status: QuestionStatus
    priority: float
    continues: bool


@dataclass(frozen=True)
class WantSnapshot:
    """The state ``derive`` reads: built by ``snapshot.build`` from the stores (Task 7a).

    ``last_message_at`` is the person's last message on the night's clock, ``None`` if they have
    never written.
    """

    as_of: datetime
    last_message_at: datetime | None
    targets: tuple[Target, ...]
    results: tuple[Result, ...]
    intentions: tuple[Commitment, ...] = ()
    questions: tuple[QuestionView, ...] = ()


@dataclass(frozen=True)
class Want:
    """A typed pull toward one target, computed from state and never stored (Task 7a).

    ``strength`` is on [0, 1]; ``weight`` multiplies it into the effective strength (1.0 unless a
    said setting or a learned weight applies). ``action`` is what it would do at the ``away``
    opportunity; ``question_id`` is set for ``inquire`` and only for it.
    """

    id: str
    kind: WantKind
    target_id: str
    action: Action
    priority_class: str
    strength: float
    weight: float = 1.0
    question_id: str | None = None

    def __post_init__(self) -> None:
        _check_unit("strength", self.strength)
        if self.weight < 0:
            raise ValueError(f"weight must not be negative, got {self.weight}")
        if (self.action == "inquire") != (self.question_id is not None):
            raise ValueError("question_id is set for an inquire action and only for it")

    @property
    def effective_strength(self) -> float:
        return self.strength * self.weight


@dataclass(frozen=True)
class Considered:
    """One candidate in a choice's receipt: ids, kind, strengths, status and an allowlisted
    reason code (``None`` only for the winner). It never holds text."""

    want_id: str
    kind: WantKind
    target_id: str
    action: Action
    strength: float
    weight: float
    status: ConsideredStatus
    reason: str | None

    def __post_init__(self) -> None:
        if (self.status == "won") != (self.reason is None):
            raise ValueError("a reason is recorded for every candidate that did not win, and only then")


@dataclass(frozen=True)
class Choice:
    """The chooser's output at one opportunity: one want and its action, or nothing, with every
    candidate it considered."""

    opportunity: Opportunity
    want: Want | None
    considered: tuple[Considered, ...]

    @property
    def action(self) -> Action | None:
        return self.want.action if self.want else None

    @property
    def target_id(self) -> str | None:
        return self.want.target_id if self.want else None

    @property
    def question_id(self) -> str | None:
        return self.want.question_id if self.want else None


@dataclass(frozen=True)
class ChoiceContext:
    """What ``choose`` gets besides the wants. ``allowed_actions`` of ``None`` allows every
    action; the night's second offer passes ``{"hold_note"}`` (Task 5, step 10)."""

    policy: Policy
    allowance: NightAllowance
    allowed_actions: frozenset[str] | None = None

    def allows(self, action: str) -> bool:
        return self.allowed_actions is None or action in self.allowed_actions


@dataclass(frozen=True)
class Expectation:
    """How likely she thinks the person is to need a study before ``horizon`` (Task 16). It is
    about the usefulness of her own action, never a prediction about the person."""

    id: str
    target_id: str
    made_at: datetime
    horizon: datetime
    p_needed: float
    outcome: ExpectationOutcome | None = None

    def __post_init__(self) -> None:
        _check_unit("p_needed", self.p_needed)


# ── The night's result (Task 5) ───────────────────────────────────────────────


@dataclass(frozen=True)
class NightResult:
    """What one person's night did. ``cost_usd`` is the sum of ``cost_parts``."""

    status: NightStatus
    failed_at: str | None
    checkpoint: CheckpointRef | None
    gates: Mapping[str, GateResult] = field(hash=False)
    choices: tuple[Choice, ...]
    cost_usd: float
    cost_parts: Mapping[CostPart, float] = field(hash=False)
    log_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", dict(self.gates))
        object.__setattr__(self, "cost_parts", dict(self.cost_parts))
        total = math.fsum(self.cost_parts.values())
        if not math.isclose(self.cost_usd, total, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"cost_usd {self.cost_usd} is not the sum of cost_parts ({total})")
