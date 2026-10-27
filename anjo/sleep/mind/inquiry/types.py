"""MIND's inquiry: the workspace, the nine operations and what the kernel returns (Task 17).

Pure: frozen dataclasses, Literals and the spec's numbers as constants. No clock, database, model
call or environment read; every time is passed in, on the night's clock. Collections are tuples,
and mappings are copied into read-only mappings and left out of the hash.

This module is the contract's one import point. The nine operations, what the thinker emits and
the reference JSON codec live in ``anjo.sleep.mind.inquiry.ops`` and are re-exported here;
``anjo.sleep.mind.inquiry.ports`` holds the ports and ``InquiryPorts``.

**The workspace is a fold of its log.** Every operation the thinker proposes, applied or
rejected, and every tool result that enters the workspace is one ``Step``. Each is appended to the
log as a ``WorkspaceEvent`` that carries the workspace's digest before and after it (K6). The
``Workspace`` keeps its steps, so the controller and the render read everything from the workspace
alone, and folding the log from an empty workspace rebuilds it exactly. Tool results enter as
observations (``Retrieved``, ``Answered``, ``Resolved``), not as operations: the nine operations are
what the thinker may propose, and an observation is what the loop saw when it ran a tool.

Cross-slot interfaces (frozen 2026-10-28)
-----------------------------------------
17-core builds these. 17-eval (the M arms, interventions, kill-resume), Task 17b (replay) and the
night glue call them. Only the two ``loop`` functions drive ports; the rest are pure.

- ``kernel.apply(ws: Workspace, op: Op, *, at: datetime, rules: KernelRules = FULL_RULES) -> Applied | Rejected``:
  validates one operation against K1-K4, hypothesize-once, the domain denylist and the finding
  cap, and returns the next workspace with its event, stamped ``at`` (``opened_at``,
  ``committed_at``); the LLM screen is the loop's, not the kernel's.
- ``kernel.fold(events: Iterable[WorkspaceEvent]) -> Workspace``: rebuilds the workspace from
  ``Workspace()``, and raises ``ValueError`` when an event's ``before`` isn't the digest so far or
  its ``after`` isn't the digest it produces.
- ``kernel.digest(ws: Workspace) -> str``: the SHA-256 hex digest of the canonical workspace,
  history included; K6's ``before`` and ``after``.
- ``controller.priority(ws: Workspace, question_id: str, *, targets: Sequence[Target], as_of: datetime) -> Priority``:
  stake x u x a for one question (stake from its targets: an open commitment due within
  ``STAKE_WINDOW`` of ``as_of``); ``QuestionView.priority`` is its ``value``.
- ``controller.decide(ws: Workspace, budget: Budget, *, focus: str | None, targets: Sequence[Target], as_of: datetime, control: ControlMode = "priority") -> Decision``:
  continue on ``focus``, investigate (switch to) another question, park a stalled one (K5) or
  stop (a cap, or no priority clearing ``PRIORITY_FLOOR``), with every priority it compared.
- ``render.render(ws: Workspace, focus: str, evidence: Sequence[Evidence]) -> str``: the thinker's
  whole input for one cycle: the focal question, its hypotheses and credences (labeled heuristic),
  open predictions, the last ``RECENT_OPS`` operations and ``evidence``; never the conversation.
- ``parse.parse_op(text: str) -> Op | Rejected``: the one operation in the thinker's raw text, or
  ``Rejected("malformed")``.
- ``loop.notice(ws: Workspace, new: Sequence[Evidence], ports: InquiryPorts, budget: Budget, *, targets: Sequence[Target], as_of: datetime) -> InquiryRun``:
  night step 2b: logs ``new`` as a ``Retrieved("new_evidence")`` observation, then one thinker
  pass that may open questions; it acts on nothing else.
- ``loop.run_inquiry(ws: Workspace, ports: InquiryPorts, budget: Budget, *, targets: Sequence[Target], as_of: datetime, focus: str | None = None, rules: KernelRules = FULL_RULES, control: ControlMode = "priority") -> InquiryRun``:
  cycles under ``decide`` until it stops, appending every event to ``ports.log`` as it happens
  (so a kill loses nothing logged), with every tool time-masked to ``as_of``. ``focus`` is the
  question the chooser picked (``inquire``); ``None`` lets the controller pick (APD).

**Replay hooks (Task 17b).** A night's start state is ``kernel.fold`` of the events whose
``step.at`` falls on or before that night; ``catch_all_id`` names the hypothesis a branch's
additions are scored into; ``InquiryRun.turns`` holds each cycle's rendered input and the
thinker's raw reply, with the model that wrote it, for the replay dataset.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Final, Literal, get_args

from anjo.sleep.mind.inquiry._check import (
    check_count,
    check_one_of,
    check_positive,
    check_text,
    check_unit,
    freeze_ids,
    freeze_map,
    freeze_tuple,
)
from anjo.sleep.mind.inquiry.ops import OP_CLASSES as OP_CLASSES
from anjo.sleep.mind.inquiry.ops import OP_TYPES as OP_TYPES
from anjo.sleep.mind.inquiry.ops import QUESTION_KINDS as QUESTION_KINDS
from anjo.sleep.mind.inquiry.ops import TOOLS as TOOLS
from anjo.sleep.mind.inquiry.ops import AbandonOp as AbandonOp
from anjo.sleep.mind.inquiry.ops import Alternative as Alternative
from anjo.sleep.mind.inquiry.ops import AssessOp as AssessOp
from anjo.sleep.mind.inquiry.ops import ConcludeOp as ConcludeOp
from anjo.sleep.mind.inquiry.ops import ConclusionKind as ConclusionKind
from anjo.sleep.mind.inquiry.ops import HypothesizeOp as HypothesizeOp
from anjo.sleep.mind.inquiry.ops import InvestigateOp as InvestigateOp
from anjo.sleep.mind.inquiry.ops import NoticeOp as NoticeOp
from anjo.sleep.mind.inquiry.ops import Op as Op
from anjo.sleep.mind.inquiry.ops import OpFormatError as OpFormatError
from anjo.sleep.mind.inquiry.ops import OpName as OpName
from anjo.sleep.mind.inquiry.ops import ParkOp as ParkOp
from anjo.sleep.mind.inquiry.ops import PredictOp as PredictOp
from anjo.sleep.mind.inquiry.ops import QuestionKind as QuestionKind
from anjo.sleep.mind.inquiry.ops import RetractOp as RetractOp
from anjo.sleep.mind.inquiry.ops import Tool as Tool
from anjo.sleep.mind.inquiry.ops import op_from_json as op_from_json
from anjo.sleep.mind.inquiry.ops import op_to_json as op_to_json
# Re-exported: the workspace's statuses are the chooser's (QuestionView), one Literal for both.
from anjo.sleep.types import QuestionStatus as QuestionStatus

# ── Vocabularies ──────────────────────────────────────────────────────────────

PredictionOutcome = Literal["true", "false", "undetermined", "censored"]
Verdict = Literal["true", "false", "undetermined"]  # what the resolver may answer
UpdateSource = Literal["resolved_prediction", "assessed_evidence"]
FindingKind = Literal["suspicion", "resolution", "retraction"]
RetrieveTool = Literal["new_evidence", "evidence_search", "research"]
ScreenPurpose = Literal["question", "hypothesis", "research_query"]
DecisionKind = Literal["continue", "investigate", "park", "stop"]
DecisionReason = Literal["stalled", "cycle_cap", "token_cap", "below_floor"]
ControlMode = Literal["priority", "newest"]  # "newest" is M-control: 12 cycles on the newest question
RejectCode = Literal[
    "malformed",              # not one well-formed operation (parse, or a field of the wrong shape)
    "unknown_ref",            # names a question, hypothesis, prediction, target or item she doesn't hold
    "not_live",               # acts on a question that isn't open or active
    "no_targets",             # NOTICE naming no target: a question's subject is a plan or topic
    "unavailable",            # INVESTIGATE with a tool this run lacks (research outside the live night)
    "k1_hypotheses",          # p_true_given, likelihoods or credences don't name exactly the question's set
    "k1_bucket",              # an ASSESS likelihood that isn't one of ASSESS_BUCKETS
    "k1_not_returned",        # ASSESS on an item no search or verified research returned for the question
    "k1_evidence_used",       # the item already updated this question, by either route
    "k1_direct_credence",     # ASSESS writing credences while K1 is on (only M-kernel may)
    "k2_not_discriminating",  # PREDICT whose p_true_given spread is under MIN_SPREAD
    "k3_citations",           # a contradiction without two items from different days, or no item at all
    "k4_not_ready",           # a resolution under RESOLVE_MIN, or with no update from outside its origin
    "hypothesize_once",       # a second HYPOTHESIZE while the catch-all holds SPLIT_MIN or less
    "hypothesis_count",       # a first set that isn't 2-4 alternatives, or a split past 4
    "domain_denylist",        # a person-diagnosis term from the kernel's denylist
    "domain_screen",          # ScreenPort refused it: its subject is the person or a named third party
    "finding_cap",            # a distinct finding past FINDING_CAP
]
# K5 and K6 never reject an operation. K5 acts through the controller (a Decision with reason
# "stalled"); K6 is every event's before and after digest, which kernel.fold checks.

QUESTION_STATUSES: Final = frozenset(get_args(QuestionStatus))
LIVE_STATUSES: Final = frozenset({"open", "active"})
FINDING_KINDS: Final = frozenset(get_args(FindingKind))
REJECT_CODES: Final = frozenset(get_args(RejectCode))

# ── The spec's numbers ────────────────────────────────────────────────────────

ASSESS_BUCKETS: Final[tuple[float, ...]] = (4.0, 2.0, 1.0, 0.5, 0.25)  # K1: ASSESS likelihoods
PROBABILITY_CLAMP: Final[tuple[float, float]] = (0.05, 0.95)  # K1: priors and p_true_given only
MIN_SPREAD: Final = 0.3  # K2: max - min of p_true_given
RESOLVE_MIN: Final = 0.8  # K4: top credence for a resolution
RETRACT_BELOW: Final = 0.5  # a reported hypothesis under this is ready to retract
SPLIT_MIN: Final = 0.5  # a second HYPOTHESIZE needs the catch-all above this
STALL_LIMIT: Final = 3  # K5: cycles without progress before the controller parks or switches
MIN_ALTERNATIVES: Final = 2  # per question, plus the catch-all
MAX_ALTERNATIVES: Final = 4
CATCH_ALL: Final = "something else"
FINDING_CAP: Final = 8  # distinct findings; updating one in place is free
CYCLE_CAP: Final = 12  # a night
TOKEN_CAP: Final = 20_000  # a night: thinker, world_query and screen tokens, retries included
PRIORITY_FLOOR: Final = 0.2  # the controller stops when no priority clears it
U_FLOOR: Final = 0.2  # u: normalized entropy, floored
A_READY: Final = 1.0  # a: a next step is ready
A_WAITING: Final = 0.3  # a: nothing is ready
STAKE_DUE: Final = 2.0  # a target is an open commitment due within STAKE_WINDOW
STAKE_NONE: Final = 1.0
STAKE_WINDOW: Final = timedelta(days=7)
RECENT_OPS: Final = 6  # operations the render shows
_SUM_TOLERANCE: Final = 1e-6


def catch_all_id(question_id: str) -> str:
    """The catch-all hypothesis's id on ``question_id``. Every other hypothesis id is the kernel's
    to assign; this one is fixed so replay and the interventions can find it."""
    return f"{question_id}.else"


# ── The workspace's records ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Question:
    """One problem she noticed and nobody assigned.

    ``target_ids`` is never empty: a question's subject is a plan or topic of the person's (a
    ``Target``), and a question may span several. ``origin_evidence`` is what raised it (K3).
    ``subject`` is the question in her words, which the domain rule screens and the render shows;
    it is never the person or a third party named in the evidence. Statuses: ``open`` (noticed),
    ``active`` (hypothesized), ``parked`` (by PARK or the progress guard; wakes only on new
    evidence on a target), ``concluded`` (a resolution), ``retracted``, ``abandoned``.
    """

    id: str
    target_ids: tuple[str, ...]
    kind: QuestionKind
    origin_evidence: tuple[str, ...]
    status: QuestionStatus
    opened_at: datetime
    subject: str

    def __post_init__(self) -> None:
        check_text("id", self.id)
        freeze_ids(self, "target_ids", empty_ok=False)
        freeze_ids(self, "origin_evidence", empty_ok=False)
        check_one_of("kind", self.kind, QUESTION_KINDS)
        check_one_of("status", self.status, QUESTION_STATUSES)
        check_text("subject", self.subject)


@dataclass(frozen=True)
class Hypothesis:
    """One alternative in a question's set, or its catch-all (``catch_all_id``).

    A question's credences are one normalized vector, a heuristic wherever they are shown.
    Hypotheses are about plans, projects, the world or her own misreading, never about the
    person's knowledge or character.
    """

    id: str
    question_id: str
    statement: str
    credence: float

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_text("question_id", self.question_id)
        check_text("statement", self.statement)
        check_unit("credence", self.credence)

    @property
    def is_catch_all(self) -> bool:
        return self.id == catch_all_id(self.question_id)


@dataclass(frozen=True)
class Prediction:
    """A test committed before its outcome exists: ``p_true_given`` holds P(claim | hypothesis)
    for every hypothesis in the set when it was committed. ``outcome`` is ``None`` while it is
    pending. ``resolved_by`` is the evidence the resolver was shown for its last verdict."""

    id: str
    question_id: str
    claim: str
    p_true_given: Mapping[str, float] = field(hash=False)
    committed_at: datetime
    horizon: datetime
    outcome: PredictionOutcome | None
    resolved_by: tuple[str, ...]

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_text("question_id", self.question_id)
        check_text("claim", self.claim)
        freeze_map(self, "p_true_given", check_unit)
        if not self.p_true_given:
            raise ValueError("p_true_given must name the question's hypotheses")
        if self.horizon <= self.committed_at:
            raise ValueError("a prediction's horizon comes after it is committed")
        if self.outcome is not None:
            check_one_of("outcome", self.outcome, frozenset(get_args(PredictionOutcome)))
        freeze_ids(self, "resolved_by")

    @property
    def pending(self) -> bool:
        return self.outcome is None


@dataclass(frozen=True)
class Update:
    """One Bayes update of a question's whole vector (K1). ``ref`` names what produced it: the
    resolved prediction's id, or the assessed item's id. Every likelihood is positive."""

    question_id: str
    ref: str
    source: UpdateSource
    evidence_ids: tuple[str, ...]
    likelihoods: Mapping[str, float] = field(hash=False)

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_text("ref", self.ref)
        check_one_of("source", self.source, frozenset(get_args(UpdateSource)))
        freeze_ids(self, "evidence_ids", empty_ok=False)
        freeze_map(self, "likelihoods", check_positive)
        if not self.likelihoods:
            raise ValueError("an update has a likelihood per hypothesis")


@dataclass(frozen=True)
class Finding:
    """What she reports on a question: one per question, updated in place (suspicion to
    resolution to retraction). ``p`` is the reported hypothesis's credence when reported.
    ``hypothesis_id`` names that hypothesis, so a retraction is ready once it falls below
    ``RETRACT_BELOW``; a retraction may keep the id of what it withdraws."""

    id: str
    question_id: str
    statement: str
    p: float
    cites: tuple[str, ...]
    kind: FindingKind
    hypothesis_id: str | None

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_text("question_id", self.question_id)
        check_text("statement", self.statement)
        check_unit("p", self.p)
        freeze_ids(self, "cites")
        check_one_of("kind", self.kind, FINDING_KINDS)
        if self.kind != "retraction" and self.hypothesis_id is None:
            raise ValueError("a suspicion or resolution names the hypothesis it reports")
        if self.hypothesis_id is not None:
            check_text("hypothesis_id", self.hypothesis_id)


# ── Observations: what entered the workspace from a tool ──────────────────────


@dataclass(frozen=True)
class EvidenceRef:
    """An evidence item as the workspace keeps it: its id and its times, never its text."""

    id: str
    day: date
    created_at: datetime

    def __post_init__(self) -> None:
        check_text("id", self.id)


@dataclass(frozen=True)
class Retrieved:
    """Items that reached her: the night's new evidence (``new_evidence``, no question), or what
    ``evidence_search`` or ``research`` returned for a question (research: verified sources only)."""

    question_id: str | None
    tool: RetrieveTool
    query: str
    items: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        check_one_of("tool", self.tool, frozenset(get_args(RetrieveTool)))
        if (self.question_id is None) != (self.tool == "new_evidence"):
            raise ValueError("new evidence belongs to no question; a search or research to one")
        if self.question_id is not None:
            check_text("question_id", self.question_id)
        freeze_tuple(self, "items", EvidenceRef)


@dataclass(frozen=True)
class Answered:
    """A ``world_query`` answer: kept for the render, and never moves credence."""

    question_id: str
    query: str
    answer: str

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_text("query", self.query)


@dataclass(frozen=True)
class Resolved:
    """A prediction's outcome: the resolver's verdict on ``seen`` (the evidence after its
    ``committed_at`` it was shown), or ``censored`` once the horizon passed unresolved."""

    prediction_id: str
    outcome: PredictionOutcome
    seen: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        check_text("prediction_id", self.prediction_id)
        check_one_of("outcome", self.outcome, frozenset(get_args(PredictionOutcome)))
        freeze_tuple(self, "seen", EvidenceRef)


Observation = Retrieved | Answered | Resolved

# ── The log ───────────────────────────────────────────────────────────────────

_OBSERVATION_CLASSES: Final = (Retrieved, Answered, Resolved)


@dataclass(frozen=True)
class Step:
    """One logged entry: an operation (applied when ``code`` is ``None``, rejected otherwise) or
    an observation, at ``seq`` on the night's clock ``at``. A rejection costs its cycle and is
    logged like any other step; thinker output that isn't an operation is logged with no entry
    and ``code="malformed"``."""

    seq: int
    at: datetime
    entry: Op | Observation | None
    code: RejectCode | None = None

    def __post_init__(self) -> None:
        check_count("seq", self.seq)
        if self.code is not None:
            check_one_of("code", self.code, REJECT_CODES)
            if self.entry is not None and not isinstance(self.entry, OP_CLASSES):
                raise ValueError("only an operation is rejected")
        if self.entry is None and self.code != "malformed":
            raise ValueError("a step without an entry is malformed thinker output")
        if self.entry is not None and not isinstance(self.entry, (*OP_CLASSES, *_OBSERVATION_CLASSES)):
            raise TypeError("a step's entry is an operation or an observation")

    @property
    def applied(self) -> bool:
        return self.code is None and isinstance(self.entry, OP_CLASSES)


@dataclass(frozen=True)
class WorkspaceEvent:
    """One record of the workspace log: a step and the workspace's digest before and after it
    (K6). The log is encrypted and folded like Task 3's target log."""

    step: Step
    before: str
    after: str

    def __post_init__(self) -> None:
        check_text("before", self.before)
        check_text("after", self.after)


# ── The workspace ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Workspace:
    """Her workspace: the records and every step that made them. ``Workspace()`` is empty.

    Checked on construction: ids are unique, every record names a question it holds, a question
    has at most one finding, and a question with hypotheses has 2-4 alternatives plus exactly one
    catch-all, with credences summing to 1. ``history`` runs from seq 0 without a gap.
    """

    questions: tuple[Question, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    updates: tuple[Update, ...] = ()
    findings: tuple[Finding, ...] = ()
    history: tuple[Step, ...] = ()

    def __post_init__(self) -> None:
        for name, kind in (
            ("questions", Question), ("hypotheses", Hypothesis), ("predictions", Prediction),
            ("updates", Update), ("findings", Finding), ("history", Step),
        ):
            freeze_tuple(self, name, kind)
        for label, ids in (
            ("question", [q.id for q in self.questions]),
            ("hypothesis", [h.id for h in self.hypotheses]),
            ("prediction", [p.id for p in self.predictions]),
            ("finding", [f.id for f in self.findings]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"a {label} id appears twice")
        self._check_references()
        if [step.seq for step in self.history] != list(range(len(self.history))):
            raise ValueError("history runs from seq 0 without a gap")

    def _check_references(self) -> None:
        sets = {q.id: frozenset(h.id for h in self.hypotheses_of(q.id)) for q in self.questions}
        named = (
            [h.question_id for h in self.hypotheses] + [p.question_id for p in self.predictions]
            + [u.question_id for u in self.updates] + [f.question_id for f in self.findings]
        )
        for question_id in named:
            if question_id not in sets:
                raise ValueError(f"a record names question {question_id!r}, which isn't held")
        for question_id in sets:
            _check_set(question_id, self.hypotheses_of(question_id))
        for prediction in self.predictions:
            if not set(prediction.p_true_given) <= sets[prediction.question_id]:
                raise ValueError(f"prediction {prediction.id} names a hypothesis outside its question")
        for update in self.updates:
            if not set(update.likelihoods) <= sets[update.question_id]:
                raise ValueError(f"an update on {update.question_id} names a hypothesis outside it")
        reported = [f.question_id for f in self.findings]
        if len(reported) != len(set(reported)):
            raise ValueError("a question has one finding, updated in place")
        for finding in self.findings:
            if finding.hypothesis_id is not None and finding.hypothesis_id not in sets[finding.question_id]:
                raise ValueError(f"finding {finding.id} reports a hypothesis outside its question")

    @property
    def seq(self) -> int:
        """The next event's seq: the number of steps so far."""
        return len(self.history)

    def question(self, question_id: str) -> Question | None:
        return next((q for q in self.questions if q.id == question_id), None)

    def hypotheses_of(self, question_id: str) -> tuple[Hypothesis, ...]:
        return tuple(h for h in self.hypotheses if h.question_id == question_id)

    def finding_of(self, question_id: str) -> Finding | None:
        return next((f for f in self.findings if f.question_id == question_id), None)


def _check_set(question_id: str, hypotheses: tuple[Hypothesis, ...]) -> None:
    if not hypotheses:
        return
    catch_alls = sum(1 for h in hypotheses if h.is_catch_all)
    if catch_alls != 1:
        raise ValueError(f"question {question_id} needs exactly one catch-all, has {catch_alls}")
    alternatives = len(hypotheses) - 1
    if not MIN_ALTERNATIVES <= alternatives <= MAX_ALTERNATIVES:
        raise ValueError(
            f"question {question_id} has {alternatives} alternatives; "
            f"{MIN_ALTERNATIVES}-{MAX_ALTERNATIVES} plus the catch-all"
        )
    total = math.fsum(h.credence for h in hypotheses)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"question {question_id}'s credences sum to {total}, not 1")


# ── What the kernel returns, and its switches ─────────────────────────────────


@dataclass(frozen=True)
class Applied:
    """The workspace after one step, and the event that records it."""

    ws: Workspace
    event: WorkspaceEvent


@dataclass(frozen=True)
class Rejected:
    """Why an operation can't apply. It is logged as a step and costs the cycle."""

    code: RejectCode

    def __post_init__(self) -> None:
        check_one_of("code", self.code, REJECT_CODES)


@dataclass(frozen=True)
class KernelRules:
    """Which kernel rules hold. Every arm but M-kernel runs ``FULL_RULES``; M-kernel switches K1,
    K2 and K4 off, so the thinker writes credences directly (``AssessOp.credences``)."""

    k1: bool = True
    k2: bool = True
    k4: bool = True


FULL_RULES: Final = KernelRules()
NO_KERNEL: Final = KernelRules(k1=False, k2=False, k4=False)

# ── The controller ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Priority:
    """A question's priority, a heuristic (not a value of information): ``stake x u x a``, or
    ``stake x 1.0`` when a resolution or retraction is ready. At most ``STAKE_DUE``."""

    question_id: str
    stake: float
    u: float
    a: float
    ready: bool

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        if self.stake not in (STAKE_NONE, STAKE_DUE):
            raise ValueError(f"stake is {STAKE_NONE} or {STAKE_DUE}")
        if not U_FLOOR <= self.u <= 1.0:
            raise ValueError(f"u must be in [{U_FLOOR}, 1]")
        if self.a not in (A_WAITING, A_READY):
            raise ValueError(f"a is {A_WAITING} or {A_READY}")
        if self.ready and self.a != A_READY:
            raise ValueError("a question ready to resolve or retract has its next step ready")

    @property
    def value(self) -> float:
        return self.stake if self.ready else self.stake * self.u * self.a


@dataclass(frozen=True)
class Budget:
    """The night's inquiry budget and what it has used. ``spend`` returns a new budget."""

    cycles_used: int = 0
    tokens_used: int = 0
    cycle_cap: int = CYCLE_CAP
    token_cap: int = TOKEN_CAP

    def __post_init__(self) -> None:
        for name in ("cycles_used", "tokens_used", "cycle_cap", "token_cap"):
            check_count(name, getattr(self, name))

    def spend(self, *, cycles: int = 0, tokens: int = 0) -> Budget:
        return Budget(
            cycles_used=self.cycles_used + cycles, tokens_used=self.tokens_used + tokens,
            cycle_cap=self.cycle_cap, token_cap=self.token_cap,
        )

    @property
    def stop_reason(self) -> Literal["cycle_cap", "token_cap"] | None:
        if self.cycles_used >= self.cycle_cap:
            return "cycle_cap"
        if self.tokens_used >= self.token_cap:
            return "token_cap"
        return None


@dataclass(frozen=True)
class Decision:
    """The controller's call before a cycle, written to the receipt with its numbers: every
    priority it compared and the budget it saw. ``investigate`` switches to ``question_id``;
    ``park`` parks it (K5, ``stalled``); ``stop`` names no question and says which cap or floor."""

    kind: DecisionKind
    question_id: str | None
    reason: DecisionReason | None
    priorities: tuple[Priority, ...]
    budget: Budget

    def __post_init__(self) -> None:
        check_one_of("kind", self.kind, frozenset(get_args(DecisionKind)))
        freeze_tuple(self, "priorities", Priority)
        if (self.kind == "stop") != (self.question_id is None):
            raise ValueError("every decision but stop names its question")
        allowed: dict[str, frozenset[str | None]] = {
            "continue": frozenset({None}),
            "investigate": frozenset({None, "stalled"}),
            "park": frozenset({"stalled"}),
            "stop": frozenset({"cycle_cap", "token_cap", "below_floor"}),
        }
        if self.reason not in allowed[self.kind]:
            raise ValueError(f"a {self.kind} decision can't give reason {self.reason!r}")


# ── What the ports return, and a run ──────────────────────────────────────────


@dataclass(frozen=True)
class ModelReply:
    """A thinker or ``world_query`` reply: the raw text, the tokens it cost (retries included)
    and the model that wrote it."""

    text: str
    tokens: int
    model: str

    def __post_init__(self) -> None:
        check_count("tokens", self.tokens)
        check_text("model", self.model)


@dataclass(frozen=True)
class Screened:
    """The screen's verdict on one text, and the tokens it cost (they count against the cap)."""

    allowed: bool
    tokens: int

    def __post_init__(self) -> None:
        check_count("tokens", self.tokens)


@dataclass(frozen=True)
class ResolverReply:
    """The resolver's verdict on a claim, and its tokens (reported apart from the cap)."""

    verdict: Verdict
    tokens: int

    def __post_init__(self) -> None:
        check_one_of("verdict", self.verdict, frozenset(get_args(Verdict)))
        check_count("tokens", self.tokens)


@dataclass(frozen=True)
class Turn:
    """One thinker call: the rendered input and the raw reply (Task 17b's trajectories)."""

    prompt: str
    reply: ModelReply


@dataclass(frozen=True)
class InquiryRun:
    """What ``loop.notice`` or ``loop.run_inquiry`` did: the final workspace, the events it
    appended (in order), every decision, every thinker turn and the budget after it.
    ``resolver_tokens`` sit outside the cap; ``research_calls`` are Wonder's, debited apart."""

    ws: Workspace
    events: tuple[WorkspaceEvent, ...]
    decisions: tuple[Decision, ...]
    turns: tuple[Turn, ...]
    budget: Budget
    resolver_tokens: int = 0
    research_calls: int = 0

    def __post_init__(self) -> None:
        freeze_tuple(self, "events", WorkspaceEvent)
        freeze_tuple(self, "decisions", Decision)
        freeze_tuple(self, "turns", Turn)
        check_count("resolver_tokens", self.resolver_tokens)
        check_count("research_calls", self.research_calls)
