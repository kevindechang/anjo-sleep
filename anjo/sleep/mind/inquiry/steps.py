"""The workspace log's entries: what one logged step holds (Task 17).

Pure. ``anjo.sleep.mind.inquiry.types`` re-exports every name here, so the contract keeps one
import point; the entries live in their own module only to keep each file small. ``KernelRules``
and ``Budget`` live here because a ``RunStart`` records them.

**A step is one of four things:** an operation the thinker proposed (applied, or rejected with a
code), an observation of a tool (``Retrieved``, ``Answered``, ``Resolved``), a mark the harness
made (``RunStart``, ``Edit``, ``Wake``), or a thinker reply that held no operation (no entry, code
``malformed``). Each step also records the cycle of its run, the question in focus and the capped
tokens charged with it. So the log alone rebuilds every arm's workspace (the rules and wipes are
in it), K5's stall counts (cycle and focus) and the budget left (tokens).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal, get_args

from anjo.sleep.mind.inquiry._check import (
    check_count,
    check_one_of,
    check_text,
    freeze_tuple,
)
from anjo.sleep.mind.inquiry.ops import OP_CLASSES, Op

PredictionOutcome = Literal["true", "false", "undetermined", "censored"]
RetrieveTool = Literal["new_evidence", "held_evidence", "evidence_search", "research"]
UNASKED: Final = frozenset({"new_evidence", "held_evidence"})  # what arrives for no question
ControlMode = Literal["priority", "newest"]  # "newest" is M-control: 12 cycles on the newest question
RunCall = Literal["notice", "inquiry"]  # which loop function a RunStart opens
EditKind = Literal["swap_lead", "delete_prediction", "inject_hypothesis"]
RejectCode = Literal[
    "malformed",              # no operation came back: unparsable text, or a call that failed after its retries
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
]
# K5 and K6 never reject an operation. K5 acts through the controller (a Decision with reason
# "stalled"); K6 is every event's before and after digest, which kernel.fold checks. The finding
# cap isn't the kernel's: it is APD's, per life, in report_finding's registry (apd.types).

REJECT_CODES: Final = frozenset(get_args(RejectCode))
CYCLE_CAP: Final = 12  # a night
TOKEN_CAP: Final = 20_000  # a night: generated tokens of the thinker, world_query and screen, retries included


# ── What a run records when it starts ─────────────────────────────────────────


@dataclass(frozen=True)
class KernelRules:
    """Which kernel rules hold. Every arm but M-kernel runs ``FULL_RULES``; M-kernel switches K1,
    K2 and K4 off, so the thinker writes credences directly (``AssessOp.credences``)."""

    k1: bool = True
    k2: bool = True
    k4: bool = True


FULL_RULES: Final = KernelRules()
NO_KERNEL: Final = KernelRules(k1=False, k2=False, k4=False)


@dataclass(frozen=True)
class Budget:
    """The night's inquiry budget and what it has used. ``spend`` returns a new budget;
    ``tokens_left`` is the ``max_tokens`` the next capped model call gets, so the cap is hard."""

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
    def tokens_left(self) -> int:
        return max(self.token_cap - self.tokens_used, 0)

    @property
    def stop_reason(self) -> Literal["cycle_cap", "token_cap"] | None:
        if self.cycles_used >= self.cycle_cap:
            return "cycle_cap"
        if self.tokens_used >= self.token_cap:
            return "token_cap"
        return None


# ── Observations: what entered the workspace from a tool ──────────────────────


@dataclass(frozen=True)
class EvidenceRef:
    """An evidence item as the workspace keeps it: its id, its times and the target it names,
    never its text. The text is fetched again by id (``EvidenceSearch.get``), so a resumed run sees
    the same. ``target_id`` is what wakes a parked question (``Wake``)."""

    id: str
    day: date
    created_at: datetime
    target_id: str | None = None

    def __post_init__(self) -> None:
        check_text("id", self.id)
        if self.target_id is not None:
            check_text("target_id", self.target_id)


@dataclass(frozen=True)
class Retrieved:
    """Items that reached her: the night's new evidence (``new_evidence``, no question), earlier
    items the notice pass recalled from the store that the workspace didn't hold (``held_evidence``,
    no question: after an M-persist wipe), or what ``evidence_search`` or ``research`` returned for
    a question (research: verified sources only, already stored as evidence)."""

    question_id: str | None
    tool: RetrieveTool
    query: str
    items: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        check_one_of("tool", self.tool, frozenset(get_args(RetrieveTool)))
        if (self.question_id is None) != (self.tool in UNASKED):
            raise ValueError("new or held evidence belongs to no question; a search or research to one")
        if self.question_id is not None:
            check_text("question_id", self.question_id)
        freeze_tuple(self, "items", EvidenceRef)


@dataclass(frozen=True)
class Answered:
    """A ``world_query`` answer: kept for the render, and never moves credence. A call that
    failed after its retries answers ``""``."""

    question_id: str
    query: str
    answer: str

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_text("query", self.query)


@dataclass(frozen=True)
class Resolved:
    """The resolver's verdict on a prediction against ``seen`` (evidence after its
    ``committed_at``), or ``censored`` once the horizon passed with no verdict. ``undetermined``
    leaves the prediction pending: the resolver may be asked again on later evidence until the
    horizon, and only ``true`` or ``false`` applies an update."""

    prediction_id: str
    outcome: PredictionOutcome
    seen: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        check_text("prediction_id", self.prediction_id)
        check_one_of("outcome", self.outcome, frozenset(get_args(PredictionOutcome)))
        freeze_tuple(self, "seen", EvidenceRef)


Observation = Retrieved | Answered | Resolved

# ── Marks: what the harness did ───────────────────────────────────────────────


@dataclass(frozen=True)
class RunStart:
    """The first step of every ``loop.notice`` and ``loop.run_inquiry`` call: what a fold and a
    resume need.

    - ``night`` is the night the call belongs to. A night starts at the first ``RunStart`` that
      names it, which is Task 17b's explicit night boundary.
    - ``as_of`` is the time every tool is masked to.
    - ``rules`` holds for every operation until the next ``RunStart``, so ``kernel.fold``
      rebuilds M-kernel's log too.
    - ``budget`` is the budget the call was given.
    - ``focus`` is the question it was given (``None``: the controller picks; always ``None`` for
      the notice pass).
    - ``wipe`` empties the workspace first, history included, leaving only this step (M-persist,
      on each night's first call). The op counter that ids derive from survives it, so ids stay
      unique across a wipe.
    - ``denylist`` names the version of the kernel's domain denylist the call checked under, so a
      fold checks every operation under the terms it was admitted with. ``None`` is the first
      version (a log written before versions were recorded).
    - ``decider`` names who picked each cycle's question when it wasn't ``controller.decide``
      (``None``), so a resume runs on under the same one (``loop.Decider``).
    """

    night: date
    as_of: datetime
    call: RunCall
    rules: KernelRules
    control: ControlMode
    focus: str | None
    budget: Budget
    wipe: bool = False
    denylist: str | None = None
    decider: str | None = None

    def __post_init__(self) -> None:
        check_one_of("call", self.call, frozenset(get_args(RunCall)))
        if self.denylist is not None:
            check_text("denylist", self.denylist)
        if self.decider is not None:
            check_text("decider", self.decider)
        check_one_of("control", self.control, frozenset(get_args(ControlMode)))
        if not isinstance(self.rules, KernelRules) or not isinstance(self.budget, Budget):
            raise TypeError("a RunStart records its KernelRules and its Budget")
        if self.focus is not None:
            check_text("focus", self.focus)
            if self.call == "notice":
                raise ValueError("the notice pass has no focus")


@dataclass(frozen=True)
class Edit:
    """One edit of a mid-inquiry snapshot (the intervention test). It is logged as a step, so the
    edited branch is a fold of its own log. Each edit names exactly what it changes:

    - ``swap_lead``: ``hypothesis_id`` and the question's leading hypothesis swap credences.
    - ``delete_prediction``: pending prediction ``prediction_id`` is removed.
    - ``inject_hypothesis``: ``statement`` joins as a new alternative at 0.5, and every other
      credence is halved. The question needs at most three alternatives, since a set holds four.

    ``kernel.apply`` raises ``ValueError`` on an edit that can't apply; the harness draws
    snapshots where it can.
    """

    kind: EditKind
    question_id: str
    hypothesis_id: str | None = None
    prediction_id: str | None = None
    statement: str | None = None

    def __post_init__(self) -> None:
        check_one_of("kind", self.kind, frozenset(get_args(EditKind)))
        check_text("question_id", self.question_id)
        needs = _EDIT_FIELD[self.kind]
        for name in _EDIT_FIELD.values():
            value = getattr(self, name)
            if (name == needs) != (value is not None):
                raise ValueError(f"a {self.kind} edit carries {needs} and no other target")
            if value is not None:
                check_text(name, value)


_EDIT_FIELD: Final[dict[str, str]] = {
    "swap_lead": "hypothesis_id", "delete_prediction": "prediction_id", "inject_hypothesis": "statement",
}
@dataclass(frozen=True)
class Wake:
    """A parked question set live again, because new evidence names one of its targets. The loop
    logs it right after the notice pass's ``Retrieved("new_evidence")`` step; ``items`` are the
    new items on the question's targets. The question goes back to ``active`` if it has
    hypotheses, else ``open``. ``kernel.apply`` raises ``ValueError`` unless the question is parked
    and every item names one of its targets."""

    question_id: str
    items: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        freeze_tuple(self, "items", EvidenceRef)
        if not self.items:
            raise ValueError("a wake names the new items that woke the question")


Mark = RunStart | Edit | Wake

# ── The log ───────────────────────────────────────────────────────────────────

_ENTRY_CLASSES: Final = (*OP_CLASSES, Retrieved, Answered, Resolved, RunStart, Edit, Wake)


@dataclass(frozen=True)
class Step:
    """One logged entry at ``seq``, on the night's clock ``at``. It holds an operation (applied
    when ``code`` is ``None``, rejected otherwise), an observation, a mark, or nothing: a thinker
    reply that held no operation, or a call that failed after its retries (``code="malformed"``).

    - ``cycle`` is the step's cycle within its run, from 0. It is ``None`` outside a cycle: a
      mark, or the notice pass.
    - ``focus`` is the question in focus, set on every cycle step, so a malformed reply counts
      against the right question in K5's stall count.
    - ``tokens`` is the capped tokens charged with this step. The thinker's go on the step its
      reply produced, the screen's on the step it screened and ``world_query``'s on its
      ``Answered``; the resolver's are reported apart.
    """

    seq: int
    at: datetime
    entry: Op | Observation | Mark | None
    code: RejectCode | None = None
    cycle: int | None = None
    focus: str | None = None
    tokens: int = 0

    def __post_init__(self) -> None:
        check_count("seq", self.seq)
        check_count("tokens", self.tokens)
        if self.entry is not None and not isinstance(self.entry, _ENTRY_CLASSES):
            raise TypeError("a step's entry is an operation, an observation or a mark")
        if self.code is not None:
            check_one_of("code", self.code, REJECT_CODES)
            if self.entry is not None and not isinstance(self.entry, OP_CLASSES):
                raise ValueError("only an operation is rejected")
        if self.entry is None and self.code != "malformed":
            raise ValueError("a step without an entry is malformed thinker output")
        if self.cycle is not None:
            check_count("cycle", self.cycle)
            if self.focus is None:
                raise ValueError("a cycle step names the question in focus")
        if self.focus is not None:
            check_text("focus", self.focus)
        if isinstance(self.entry, (RunStart, Edit, Wake)) and (self.cycle is not None or self.focus or self.tokens):
            raise ValueError("a mark belongs to no cycle and charges nothing")

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
