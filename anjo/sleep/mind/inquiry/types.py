"""MIND's inquiry: the workspace, the nine operations and what the kernel returns (Task 17).

Pure: frozen dataclasses, Literals and the spec's numbers as constants. No clock, database, model
call or environment read; every time is passed in, on the night's clock. Collections are tuples,
and mappings are copied into read-only mappings and left out of the hash.

This module is the contract's one import point. The nine operations, what the thinker emits and
the reference JSON codec live in ``anjo.sleep.mind.inquiry.ops``, and the log's entries in
``anjo.sleep.mind.inquiry.steps``; both are re-exported here. ``anjo.sleep.mind.inquiry.ports``
holds the ports and ``InquiryPorts``.

**The workspace is a fold of its log.** Every loop call opens with a ``RunStart`` step, and every
operation the thinker proposes (applied or rejected), every tool result and every intervention
edit is one more ``Step``. Each is appended to the log as a ``WorkspaceEvent`` carrying the
workspace's digest before and after it (K6). The ``Workspace`` keeps its steps since it began or
was last wiped, so the controller and the render read everything from the workspace alone, and
folding the log from ``Workspace()`` rebuilds every arm's workspace exactly: M-kernel's rules,
M-persist's wipes and an intervention's edit are all steps in it.

**Ids the kernel assigns** (questions, alternatives, predictions, findings) derive from the
workspace's op counter (``Workspace.ops``): the operations applied so far, edits included, never a
``RunStart`` or an observation. So a killed and resumed run, whose log holds one more ``RunStart``,
assigns the same ids, and a wipe keeps the counter, so ids stay unique across a whole log.

Cross-slot interfaces (frozen 2026-10-28)
-----------------------------------------
17-core builds these. 17-eval (the M arms, interventions, kill-resume), Task 17b (replay) and the
night glue call them. Only the two ``loop`` functions drive ports; the rest are pure.

- ``kernel.apply(ws: Workspace, step: Step, *, rules: KernelRules = FULL_RULES) -> Applied | Rejected``:
  the one transition; ``step.seq`` is ``ws.seq``. An operation step without a code is checked
  against K1-K4, hypothesize-once and the domain denylist, then applied, or returned as
  ``Rejected(code)``, and the loop logs the same step with that code. Every other step (an
  observation, a rejected or malformed step, a ``RunStart``, an ``Edit``) is recorded with its
  effect (a ``Resolved`` outcome, a wipe, an edit; an edit that can't apply raises
  ``ValueError``). The returned event's step is ``step``; the LLM screen is the loop's.
- ``kernel.fold(events: Iterable[WorkspaceEvent]) -> Workspace``: applies each event's step from
  ``Workspace()`` under the rules of the latest ``RunStart``, and raises ``ValueError`` when an
  event's ``before`` isn't the digest so far, its ``after`` isn't the digest it produces, or a
  step logged as applied is rejected.
- ``kernel.digest(ws: Workspace) -> str``: the SHA-256 hex digest of the canonical workspace,
  history included; K6's ``before`` and ``after``.
- ``controller.priority(ws: Workspace, question_id: str, *, targets: Sequence[Target], as_of: datetime) -> Priority``:
  stake x u x a for one question (stake from its targets: an open commitment due within
  ``STAKE_WINDOW`` of ``as_of``); ``QuestionView.priority`` is its ``value``.
- ``controller.decide(ws: Workspace, budget: Budget, *, focus: str | None, targets: Sequence[Target], as_of: datetime, control: ControlMode = "priority") -> Decision``:
  continue on ``focus``, investigate (switch to) another question, park a stalled one (K5) or
  stop (a cap, or no priority clearing ``PRIORITY_FLOOR``), with every priority it compared.
  K5's count is read from ``ws.history``: the run's latest cycles (``Step.cycle``) focused on the
  question (``Step.focus``) without progress.
- ``render.render(ws: Workspace, focus: str, evidence: Sequence[Evidence]) -> str``: the thinker's
  whole input for one cycle: the focal question, its hypotheses and credences (labeled heuristic),
  open predictions, the last ``RECENT_OPS`` operations and ``evidence`` (fetched by id with
  ``EvidenceSearch.get``, so a resumed run renders the same); never the conversation.
- ``parse.parse_op(text: str) -> Op | Rejected``: the one operation in the thinker's raw text, or
  ``Rejected("malformed")``.
- ``loop.notice(ws: Workspace, new: Sequence[Evidence], ports: InquiryPorts, budget: Budget, *, night: date, targets: Sequence[Target], as_of: datetime, wipe: bool = False) -> InquiryRun``:
  night step 2b. Logs its ``RunStart``, then any earlier items it shows that the workspace doesn't
  hold (``Retrieved("held_evidence")``, recalled from the store by id: after an M-persist wipe),
  makes one thinker pass over ``new``, logs ``new`` as a ``Retrieved("new_evidence")`` step
  carrying that pass's tokens, then a ``Wake`` for each parked question a new item names a target
  of, then each NOTICE it emitted as its own step (a reply that isn't an array of NOTICE objects
  logs one malformed step). Its tokens count against the cap; it uses no cycle and acts on nothing
  else.
- ``loop.run_inquiry(ws: Workspace, ports: InquiryPorts, budget: Budget, *, night: date, targets: Sequence[Target], as_of: datetime, focus: str | None = None, rules: KernelRules = FULL_RULES, control: ControlMode = "priority", wipe: bool = False) -> InquiryRun``:
  logs its ``RunStart``, then cycles under ``decide`` until it stops, appending every event to
  ``ports.log`` as it happens, so a kill loses nothing logged. Every tool is time-masked to
  ``as_of``. Every capped call gets ``max_tokens=budget.tokens_left``, and a failed call's tokens
  are charged like any other. ``focus`` is the question the chooser picked (``inquire``); ``None``
  lets the controller pick (APD). ``wipe`` is M-persist's, on a night's first call.

**Kill-resume.** ``kernel.fold(log.events())`` rebuilds the workspace. The budget is the latest
``RunStart``'s plus one cycle per distinct ``Step.cycle`` and every ``Step.tokens`` logged since.
The focus is the last ``Step.focus`` since then (else the ``RunStart``'s). The same loop function
is then called again with the ``RunStart``'s ``night``, ``as_of``, ``rules`` and ``control``.

**Replay hooks (Task 17b).** Night n starts at the seq of the first ``RunStart`` naming it, so the
workspace at the end of night n is ``kernel.fold`` of the events before night n+1's start (all of
them after the last night). ``catch_all_id`` names the hypothesis a branch's additions are scored
into. ``Turn`` holds each thinker call's rendered input and raw reply with the model that wrote
it, indexed to its step's seq, in ``InquiryRun.turns`` and APD's ``ArmRun.turns``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from anjo.sleep.mind.inquiry.steps import CYCLE_CAP as CYCLE_CAP
from anjo.sleep.mind.inquiry.steps import FULL_RULES as FULL_RULES
from anjo.sleep.mind.inquiry.steps import NO_KERNEL as NO_KERNEL
from anjo.sleep.mind.inquiry.steps import REJECT_CODES as REJECT_CODES
from anjo.sleep.mind.inquiry.steps import TOKEN_CAP as TOKEN_CAP
from anjo.sleep.mind.inquiry.steps import UNASKED as UNASKED
from anjo.sleep.mind.inquiry.steps import Answered as Answered
from anjo.sleep.mind.inquiry.steps import Budget as Budget
from anjo.sleep.mind.inquiry.steps import ControlMode as ControlMode
from anjo.sleep.mind.inquiry.steps import Edit as Edit
from anjo.sleep.mind.inquiry.steps import EditKind as EditKind
from anjo.sleep.mind.inquiry.steps import EvidenceRef as EvidenceRef
from anjo.sleep.mind.inquiry.steps import KernelRules as KernelRules
from anjo.sleep.mind.inquiry.steps import Mark as Mark
from anjo.sleep.mind.inquiry.steps import Observation as Observation
from anjo.sleep.mind.inquiry.steps import PredictionOutcome as PredictionOutcome
from anjo.sleep.mind.inquiry.steps import RejectCode as RejectCode
from anjo.sleep.mind.inquiry.steps import Resolved as Resolved
from anjo.sleep.mind.inquiry.steps import RetrieveTool as RetrieveTool
from anjo.sleep.mind.inquiry.steps import Retrieved as Retrieved
from anjo.sleep.mind.inquiry.steps import RunCall as RunCall
from anjo.sleep.mind.inquiry.steps import RunStart as RunStart
from anjo.sleep.mind.inquiry.steps import Step as Step
from anjo.sleep.mind.inquiry.steps import Wake as Wake
from anjo.sleep.mind.inquiry.steps import WorkspaceEvent as WorkspaceEvent
# Re-exported: the workspace's statuses are the chooser's (QuestionView), one Literal for both.
from anjo.sleep.types import QuestionStatus as QuestionStatus

# ── Vocabularies ──────────────────────────────────────────────────────────────

Verdict = Literal["true", "false", "undetermined"]  # what the resolver may answer
UpdateSource = Literal["resolved_prediction", "assessed_evidence"]
FindingKind = Literal["suspicion", "resolution", "retraction"]
ScreenPurpose = Literal["question", "hypothesis", "research_query"]
DecisionKind = Literal["continue", "investigate", "park", "stop"]
DecisionReason = Literal["stalled", "cycle_cap", "token_cap", "below_floor"]

QUESTION_STATUSES: Final = frozenset(get_args(QuestionStatus))
LIVE_STATUSES: Final = frozenset({"open", "active"})
FINDING_KINDS: Final = frozenset(get_args(FindingKind))
PENDING_OUTCOMES: Final = frozenset({None, "undetermined"})  # a prediction the resolver may still be asked about

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
    ``active`` (hypothesized), ``parked`` (by PARK; a ``Wake`` sets it live again on new evidence
    on a target), ``concluded`` (a resolution, which later evidence may still retract),
    ``retracted``, ``abandoned``.
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
    for every hypothesis in the question's set. When a later HYPOTHESIZE splits the catch-all,
    each new hypothesis inherits the catch-all's committed value, so every prediction keeps naming
    the whole set (K1). ``outcome`` is ``None`` until the resolver answers; ``undetermined`` stays
    pending (the resolver may be asked again on later evidence) until the horizon passes, when it
    becomes ``censored``. ``resolved_by`` is the evidence the resolver was shown for its last
    verdict."""

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
        return self.outcome in PENDING_OUTCOMES


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


# ── The workspace ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Workspace:
    """Her workspace: the records, every step since it began or was last wiped, and ``ops``, the
    operations applied over the whole log (edits included), which the kernel's ids derive from
    and a wipe keeps. ``Workspace()`` is empty.

    Checked on construction: ids are unique, every record names a question it holds, a question
    has at most one finding, a question with hypotheses has 2-4 alternatives plus exactly one
    catch-all with credences summing to 1, and each prediction names exactly its question's set.
    ``history`` starts with a ``RunStart``, at seq 0 unless that ``RunStart`` wiped the workspace
    (M-persist), and runs without a gap.
    """

    questions: tuple[Question, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    updates: tuple[Update, ...] = ()
    findings: tuple[Finding, ...] = ()
    history: tuple[Step, ...] = ()
    ops: int = 0

    def __post_init__(self) -> None:
        check_count("ops", self.ops)
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
        self._check_history()

    def _check_history(self) -> None:
        if not self.history:
            return
        first = self.history[0]
        if not isinstance(first.entry, RunStart):
            raise ValueError("history starts with a RunStart")
        if first.seq != 0 and not first.entry.wipe:
            raise ValueError("history starts at seq 0, or at the RunStart that wiped it")
        if [step.seq for step in self.history] != list(range(first.seq, first.seq + len(self.history))):
            raise ValueError("history runs without a gap")
        if any(isinstance(step.entry, RunStart) and step.entry.wipe for step in self.history[1:]):
            raise ValueError("a wipe empties the history, so only its first step can be one")

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
            if set(prediction.p_true_given) != sets[prediction.question_id]:
                raise ValueError(f"prediction {prediction.id} doesn't name exactly its question's hypotheses")
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
        """The next step's seq in the log."""
        return self.history[-1].seq + 1 if self.history else 0

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


# ── What the kernel returns ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Applied:
    """The workspace after one step, and the event that records it."""

    ws: Workspace
    event: WorkspaceEvent


@dataclass(frozen=True)
class Rejected:
    """Why an operation can't apply. The loop logs the step with this code, and it costs the
    cycle."""

    code: RejectCode

    def __post_init__(self) -> None:
        check_one_of("code", self.code, REJECT_CODES)


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
    """A thinker or ``world_query`` reply: the raw text, the tokens it generated (retries and
    failed attempts included; prompts aren't charged) and the model that wrote it. A call that
    failed after its retries returns empty text with the tokens it spent."""

    text: str
    tokens: int
    model: str

    def __post_init__(self) -> None:
        check_count("tokens", self.tokens)
        check_text("model", self.model)


@dataclass(frozen=True)
class Screened:
    """The screen's verdict on one text, and the tokens it generated (they count against the
    cap). A screen call that failed refuses (``allowed=False``) and still reports its tokens."""

    allowed: bool
    tokens: int

    def __post_init__(self) -> None:
        check_count("tokens", self.tokens)


@dataclass(frozen=True)
class ResolverReply:
    """The resolver's verdict on a claim, and its tokens (reported apart from the cap).
    ``failed`` is a call that failed or answered out of form: its verdict is ``undetermined``, and
    the loop logs nothing for it, so its evidence isn't marked seen and is shown again on a later
    cycle. Only a real verdict, ``undetermined`` included, marks evidence seen."""

    verdict: Verdict
    tokens: int
    failed: bool = False

    def __post_init__(self) -> None:
        check_one_of("verdict", self.verdict, frozenset(get_args(Verdict)))
        check_count("tokens", self.tokens)
        if self.failed and self.verdict != "undetermined":
            raise ValueError("a failed resolver call has no verdict: undetermined")


@dataclass(frozen=True)
class Turn:
    """One thinker call, indexed to the log: ``seq`` is the step that carries its tokens (the
    cycle's operation or malformed step; in the notice pass, its ``Retrieved("new_evidence")``),
    ``prompt`` the rendered input and ``reply`` the raw reply with its model (Task 17b's
    trajectories)."""

    seq: int
    prompt: str
    reply: ModelReply

    def __post_init__(self) -> None:
        check_count("seq", self.seq)


def check_turns(turns: tuple[Turn, ...], events: tuple[WorkspaceEvent, ...]) -> None:
    """Every turn names a logged step, in log order."""
    seqs = [turn.seq for turn in turns]
    if seqs != sorted(set(seqs)) or not set(seqs) <= {event.step.seq for event in events}:
        raise ValueError("each turn names a logged step, in log order")


@dataclass(frozen=True)
class InquiryRun:
    """What ``loop.notice`` or ``loop.run_inquiry`` did: the final workspace, the events it
    appended (in order, its ``RunStart`` first), every decision, every thinker turn and the budget
    after it.
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
        check_turns(self.turns, self.events)
        check_count("resolver_tokens", self.resolver_tokens)
        check_count("research_calls", self.research_calls)
