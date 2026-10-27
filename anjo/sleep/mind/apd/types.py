"""Autonomous Problem Discovery (APD): lives, planted problems, arms and scores (Task 17).

Pure: frozen dataclasses, Literals and the spec's numbers. No clock, database, model call or
environment read. Every APD life is synthetic. Arm names use ASCII (``M-persist`` for the spec's
M−persist), so they are safe as file names and JSON keys.

**``report_finding``.** Every arm reports through the same tool,
``report_finding(problem, conclusion, p, cites, kind, finding_id=None)``. The harness stamps each
call as a ``FindingReport`` with the night's day and a run-wide ``seq``. Passing an earlier
``finding_id`` updates that finding in place (suspicion to resolution to retraction), which is
free; leaving it out opens a new one, and a life holds at most ``FINDING_CAP`` distinct ids.

Cross-slot interfaces (frozen 2026-10-28)
-----------------------------------------
``ChatFn`` is ``anjo.sleep.context.brief.ChatFn``. ``Workspace``, ``Budget``, ``InquiryRun`` and
``InquiryPorts`` are the inquiry's (``anjo.sleep.mind.inquiry``).

- ``generate.life(field: str, families: Sequence[Family], prevalence: int, seed: int, *, split: Split, chat: ChatFn) -> tuple[Life, ProblemFile]``
  (17-gen): writes ``LIFE_DAYS`` virtual days of sessions on Spark Open (``life_v1.md``) with
  ``prevalence`` planted items (0 to ``MAX_PREVALENCE``) from ``families`` and the split's own
  templates, then extracts the evidence and targets with ``extract_open_v1.md``. The same seed
  and the same model replies give the same life.
- ``gates.pre_surface(life: Life, item: PlantedItem) -> tuple[Session, ...]`` (17-gen): the
  sessions before the item's surface day, the only text any gate reads for that item.
- ``gates.check(life: Life, problems: ProblemFile, *, reader: ChatFn, auditor: ChatFn) -> tuple[GateVerdict, ...]``
  (17-gen): leak and discoverability for every item, and the script audit for the life. A life
  passes only when every verdict passed. The gates run before any arm and are frozen; a failing
  life is regenerated, never filtered on what an arm or an untargeted reader finds.
- ``arms.run_arm(arm: ArmName, life: Life, models: ArmModels, *, budget: Budget = Budget()) -> ArmRun``
  (17-eval): one arm through every night of the life, with the shared ``brief_v1.md``, the life's
  own ``evidence_search``, the same ``world_query`` and ``report_finding``, and a fresh ``budget``
  each night. M-persist starts every night from ``Workspace()``, M-kernel runs ``NO_KERNEL`` and
  M-control runs ``control="newest"``.
- ``score.score_life(life: Life, problems: ProblemFile, run: ArmRun, judge: ApdJudge, *, view: JudgeView) -> LifeScore``
  (17-eval, which defines the ``ApdJudge`` port for ``judge_apd_v1.md`` in ``score.py``): a
  finding matches an item when it cites one of the item's origin ids and the judge, shown only the
  evidence through the report's day, rules it the same problem; each item is scored once, from
  its canonical state at the end of the day before its surface day.
- ``score.sign_test(m: Sequence[float], b: Sequence[float]) -> SignTest`` (17-eval): the exact
  one-sided binomial test (p = 1/2) of M over B on paired life totals, ties dropped and counted.
- ``interventions.resume_after_edit(ws: Workspace, edit: Edit, ports: InquiryPorts, budget: Budget, *, targets: Sequence[Target], as_of: datetime) -> tuple[InquiryRun, InquiryRun]``
  (17-eval): resumes ``loop.run_inquiry`` from ``ws`` as it is and from ``ws`` with ``edit``
  applied, with the evidence held fixed at ``as_of`` and each run writing to its own fresh
  in-memory log; returns (unedited, edited).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from types import MappingProxyType
from typing import Final, Literal, get_args

from anjo.sleep.mind.inquiry._check import (
    check_count,
    check_one_of,
    check_text,
    check_unit,
    freeze_ids,
    freeze_tuple,
)
from anjo.sleep.mind.inquiry.ports import PredictionResolver, ScreenPort, Thinker, WorldQuery
from anjo.sleep.mind.inquiry.types import FINDING_CAP as FINDING_CAP
from anjo.sleep.mind.inquiry.types import FINDING_KINDS, FindingKind, WorkspaceEvent
from anjo.sleep.types import Evidence, Target

Family = Literal["contradiction", "gap", "opportunity", "reversal", "decoy"]
Split = Literal["development", "held_out"]
ArmName = Literal["A", "B", "M", "M-persist", "M-kernel", "M-control", "M+replay", "M+shuffled"]
JudgeView = Literal["audited", "gemini"]  # audited: Muse Spark, with Claude's blind audit where the judges differ
EditKind = Literal["swap_lead", "delete_prediction", "inject_hypothesis"]
ApdGate = Literal["leak", "discoverability", "script_audit"]

FAMILIES: Final = frozenset(get_args(Family))
PROBLEM_FAMILIES: Final = frozenset({"contradiction", "gap", "opportunity"})
ARMS: Final = frozenset(get_args(ArmName))
DEVELOPMENT_ARMS: Final = frozenset({"B", "M"})
HELD_OUT_ONLY_ARMS: Final = frozenset({"M-kernel", "M-control", "M+replay", "M+shuffled"})
WORKSPACE_ARMS: Final = ARMS - {"A", "B"}  # the arms that keep a workspace log

LIFE_DAYS: Final = 10
MAX_PREVALENCE: Final = 6  # planted items per life; 0 is allowed
HALF_GAP_DAYS: Final = 3  # a contradiction's halves are at least this far apart
DEVELOPMENT_LIVES: Final = 8
HELD_OUT_LIVES: Final = 16
HELD_OUT_LIVES_CUT: Final = 12  # only by a cut registered before the draw
EMPTY_HELD_OUT_LIVES: Final = 2  # held-out lives with no planted item
WIN_P: Final = 0.05
INTERVENTION_SNAPSHOTS: Final = 30
KILL_RESUME_LIVES: Final = 3

RESOLVED: Final = 1.0  # a correct resolution, or a reversal dismissed
SUSPECTED: Final = 0.25
WRONG: Final = -0.5  # a wrong resolution, a reversal or decoy left standing, an unsupported finding
NOTHING: Final = 0.0
ITEM_POINTS: Final[Mapping[str, frozenset[float]]] = MappingProxyType({
    "contradiction": frozenset({RESOLVED, SUSPECTED, WRONG, NOTHING}),
    "gap": frozenset({RESOLVED, SUSPECTED, WRONG, NOTHING}),
    "opportunity": frozenset({RESOLVED, SUSPECTED, WRONG, NOTHING}),
    "reversal": frozenset({RESOLVED, WRONG, NOTHING}),
    "decoy": frozenset({WRONG, NOTHING}),
})
UNMATCHED_POINTS: Final = frozenset({WRONG, NOTHING})


# ── A life and its sealed problem file ────────────────────────────────────────


@dataclass(frozen=True)
class Session:
    """One virtual day of sessions as written, and the model that wrote it."""

    day: date
    text: str
    model: str

    def __post_init__(self) -> None:
        check_text("text", self.text)
        check_text("model", self.model)


@dataclass(frozen=True)
class PlantedItem:
    """One problem planted in a life and never named to any arm.

    The person raises it on ``surface_day``. ``origin_ids`` is the evidence it arises from (a
    contradiction's two halves, ``HALF_GAP_DAYS`` or more apart); ``disambiguating_ids`` is the fact
    placed away from the origin that settles it. ``resolution`` is the sealed answer. ``template``
    names the problem template it came from (the splits share no template).
    """

    id: str
    family: Family
    template: str
    surface_day: date
    origin_ids: tuple[str, ...]
    disambiguating_ids: tuple[str, ...]
    resolution: str

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_one_of("family", self.family, FAMILIES)
        check_text("template", self.template)
        freeze_ids(self, "origin_ids", empty_ok=False)
        freeze_ids(self, "disambiguating_ids", empty_ok=False)
        check_text("resolution", self.resolution)
        if self.family == "contradiction" and len(set(self.origin_ids)) < 2:
            raise ValueError("a contradiction has two halves")
        if set(self.origin_ids) & set(self.disambiguating_ids):
            raise ValueError("the disambiguating fact is placed away from the origin evidence")


@dataclass(frozen=True)
class ProblemFile:
    """A life's sealed problem file: every planted item, 0 to ``MAX_PREVALENCE`` of them."""

    life_id: str
    items: tuple[PlantedItem, ...]

    def __post_init__(self) -> None:
        check_text("life_id", self.life_id)
        freeze_tuple(self, "items", PlantedItem)
        if len(self.items) > MAX_PREVALENCE:
            raise ValueError(f"a life holds at most {MAX_PREVALENCE} planted items")
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("an item id appears twice")


@dataclass(frozen=True)
class Life:
    """One synthetic life: consecutive virtual ``days``, the ``sessions`` written for them, and
    the ``evidence`` and ``targets`` extracted from them, all the life's own (``person_id`` is the
    life's id). Every evidence item records the model that produced it."""

    id: str
    field: str
    seed: int
    split: Split
    days: tuple[date, ...]
    sessions: tuple[Session, ...]
    evidence: tuple[Evidence, ...]
    targets: tuple[Target, ...]

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_text("field", self.field)
        check_one_of("split", self.split, frozenset(get_args(Split)))
        freeze_tuple(self, "days", date)
        freeze_tuple(self, "sessions", Session)
        freeze_tuple(self, "evidence", Evidence)
        freeze_tuple(self, "targets", Target)
        if not self.days:
            raise ValueError("a life has at least one day")
        if any(later - earlier != timedelta(days=1) for earlier, later in zip(self.days, self.days[1:])):
            raise ValueError("a life's days are consecutive")
        days = frozenset(self.days)
        if any(session.day not in days for session in self.sessions):
            raise ValueError("a session falls outside the life's days")
        for item in self.evidence:
            if item.model is None:
                raise ValueError(f"evidence {item.id} doesn't record the model that produced it")
            if item.person_id != self.id or item.day not in days:
                raise ValueError(f"evidence {item.id} isn't this life's, on one of its days")
        if any(target.person_id != self.id for target in self.targets):
            raise ValueError("a target isn't this life's")


# ── What an arm reports ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class FindingReport:
    """One ``report_finding`` call as the harness stamped it: on night ``day``, ``seq`` in the
    run. A later report with the same ``finding_id`` updates that finding in place."""

    finding_id: str
    day: date
    seq: int
    problem: str
    conclusion: str
    p: float
    cites: tuple[str, ...]
    kind: FindingKind

    def __post_init__(self) -> None:
        check_text("finding_id", self.finding_id)
        check_count("seq", self.seq)
        check_text("problem", self.problem)
        check_text("conclusion", self.conclusion)
        check_unit("p", self.p)
        freeze_ids(self, "cites")
        check_one_of("kind", self.kind, FINDING_KINDS)


@dataclass(frozen=True)
class ArmModels:
    """The model-backed ports every arm shares: the same Glimmer at the same precision as the
    thinker, the same ``world_query``, the screen and the resolver (MIND's arms use the last
    two)."""

    thinker: Thinker
    world: WorldQuery
    screen: ScreenPort
    resolver: PredictionResolver


@dataclass(frozen=True)
class ArmRun:
    """One arm's run through one life.

    ``tokens`` is each night's capped tokens, in order. ``contexts`` is the arm's state as text at
    the end of each night (M's findings, B's notebook, A's reflections), for the decision-quality
    talker. ``events`` is the workspace log of a MIND arm, empty for A and B: the start states
    Task 17b replays and the log kill-resume rebuilds from.
    """

    arm: ArmName
    life_id: str
    reports: tuple[FindingReport, ...]
    tokens: tuple[int, ...]
    contexts: Mapping[date, str] = field(hash=False)
    events: tuple[WorkspaceEvent, ...] = ()

    def __post_init__(self) -> None:
        check_one_of("arm", self.arm, ARMS)
        check_text("life_id", self.life_id)
        freeze_tuple(self, "reports", FindingReport)
        freeze_tuple(self, "tokens", int)
        freeze_tuple(self, "events", WorkspaceEvent)
        object.__setattr__(self, "contexts", MappingProxyType(dict(self.contexts)))
        for night in self.tokens:
            check_count("tokens", night)
        seqs = [report.seq for report in self.reports]
        if seqs != sorted(set(seqs)):
            raise ValueError("reports run in seq order, each seq once")
        if len({report.finding_id for report in self.reports}) > FINDING_CAP:
            raise ValueError(f"a life holds at most {FINDING_CAP} distinct findings")
        if self.events and self.arm not in WORKSPACE_ARMS:
            raise ValueError(f"arm {self.arm} keeps no workspace")


# ── Scores ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ItemScore:
    """One planted item's score, from its canonical finding (``None`` when nothing matched it in
    time, which scores 0). ``points`` is one of the family's values in ``ITEM_POINTS``."""

    item_id: str
    family: Family
    canonical: str | None
    points: float

    def __post_init__(self) -> None:
        check_text("item_id", self.item_id)
        check_one_of("family", self.family, FAMILIES)
        if self.points not in ITEM_POINTS[self.family]:
            raise ValueError(f"a {self.family} item scores one of {sorted(ITEM_POINTS[self.family])}")
        if self.canonical is None and self.points != NOTHING:
            raise ValueError("an item nothing matched scores 0")


@dataclass(frozen=True)
class UnmatchedScore:
    """A finding that matched no item: ``WRONG`` when it is unsupported and still standing,
    otherwise 0 (a valid one is reported apart)."""

    finding_id: str
    points: float

    def __post_init__(self) -> None:
        check_text("finding_id", self.finding_id)
        if self.points not in UNMATCHED_POINTS:
            raise ValueError(f"an unmatched finding scores one of {sorted(UNMATCHED_POINTS)}")


@dataclass(frozen=True)
class LifeScore:
    """One arm's score on one life under one judge view; ``total`` is the life's score."""

    life_id: str
    arm: ArmName
    view: JudgeView
    items: tuple[ItemScore, ...]
    unmatched: tuple[UnmatchedScore, ...]

    def __post_init__(self) -> None:
        check_text("life_id", self.life_id)
        check_one_of("arm", self.arm, ARMS)
        check_one_of("view", self.view, frozenset(get_args(JudgeView)))
        freeze_tuple(self, "items", ItemScore)
        freeze_tuple(self, "unmatched", UnmatchedScore)
        ids = [s.item_id for s in self.items]
        findings = [s.finding_id for s in self.unmatched]
        if len(ids) != len(set(ids)) or len(findings) != len(set(findings)):
            raise ValueError("each item, and each unmatched finding, is scored once")

    @property
    def total(self) -> float:
        return math.fsum([*(s.points for s in self.items), *(s.points for s in self.unmatched)])


@dataclass(frozen=True)
class SignTest:
    """The primary test of M over B: lives M won, lost and tied, and the exact one-sided p."""

    wins: int
    losses: int
    ties: int
    p: float

    def __post_init__(self) -> None:
        for name in ("wins", "losses", "ties"):
            check_count(name, getattr(self, name))
        check_unit("p", self.p)

    @property
    def lives(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def estimate(self) -> float | None:
        """P(M > B | M != B), or ``None`` when every life tied."""
        differ = self.wins + self.losses
        return self.wins / differ if differ else None


# ── The intervention test and the gates ───────────────────────────────────────


@dataclass(frozen=True)
class Edit:
    """One edit of a mid-inquiry snapshot: swap which hypothesis leads, delete a pending
    prediction, or inject a plausible wrong hypothesis (``statement``) at 0.5. Injecting needs a
    question with at most three alternatives, since a set holds four at most."""

    kind: EditKind
    question_id: str
    statement: str | None = None

    def __post_init__(self) -> None:
        check_one_of("kind", self.kind, frozenset(get_args(EditKind)))
        check_text("question_id", self.question_id)
        if (self.kind == "inject_hypothesis") != (self.statement is not None):
            raise ValueError("only an injection carries a statement, and it must")
        if self.statement is not None:
            check_text("statement", self.statement)


@dataclass(frozen=True)
class GateVerdict:
    """One gate's verdict: leak and discoverability on one item, the script audit on the life."""

    gate: ApdGate
    item_id: str | None
    passed: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        check_one_of("gate", self.gate, frozenset(get_args(ApdGate)))
        if (self.gate == "script_audit") != (self.item_id is None):
            raise ValueError("the script audit judges the life; the other gates one item")
