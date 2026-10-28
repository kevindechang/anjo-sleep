"""Autonomous Problem Discovery (APD): lives, planted problems, arms and scores (Task 17).

Pure: frozen dataclasses, Literals and the spec's numbers. No clock, database, model call or
environment read. Every APD life is synthetic. Arm names use ASCII (``M-persist`` for the spec's
M−persist), so they are safe as file names and JSON keys.

**``report_finding``.** Every arm reports through the same tool,
``report_finding(problem, conclusion, p, cites, kind, finding_id=None)``. The harness stamps each
call as a ``FindingReport`` with the night's day and a run-wide ``seq``, and keeps it in one
``FindingRegistry`` per arm and life, across every night. Passing an earlier ``finding_id``
updates that finding in place (suspicion to resolution to retraction), which is free; leaving it
out opens a new one with an id unique in the life. The cap of ``FINDING_CAP`` distinct ids is the
registry's, for every arm (A and B included), and no wipe resets it: a call past it is refused
and recorded nowhere. MIND's arms report each workspace finding under its own id, which the kernel
derives from a log seq, so ids stay unique across M-persist's wipes and an update in place stays
one finding.

Cross-slot interfaces (frozen 2026-10-28)
-----------------------------------------
``ChatFn`` is ``anjo.sleep.context.brief.ChatFn``. ``Workspace``, ``Budget``, ``InquiryRun`` and
``InquiryPorts`` are the inquiry's (``anjo.sleep.mind.inquiry``).

- ``generate.life(field: str, families: Sequence[Family], prevalence: int, seed: int, *, split: Split, chat: ChatFn, model: str = SPARK_OPEN) -> tuple[Life, ProblemFile]``
  (17-gen): writes ``LIFE_DAYS`` virtual days of sessions on Spark Open (``life_v1.md``), under
  ``model``, its provider's exact id (``mind.spark_provider``; added 2026-10-28, optional), with
  ``prevalence`` planted items (0 to ``MAX_PREVALENCE``) from ``families`` and the split's own
  templates, then extracts the evidence and targets day by day with ``extract_open_v1.md``, so
  ``Life.targets`` holds the targets as each day ends. The same seed and the same model replies
  give the same life.
- ``gates.pre_surface(life: Life, item: PlantedItem) -> tuple[Session, ...]`` (17-gen): the
  sessions before the item's surface day, the only text any gate reads for that item.
- ``gates.check(life: Life, problems: ProblemFile, *, reader: ChatFn, auditor: ChatFn, model: str = READER_MODEL) -> tuple[GateVerdict, ...]``
  (17-gen): leak and discoverability for every item, read on ``model`` (as ``generate.life``),
  and the script audit for the life. The
  discoverability reader gets the pre-surface prefix and ``PlantedItem.topic`` only, never the
  resolution or the template. A life passes only when every verdict passed. The gates run before any arm and are frozen; a failing
  life is regenerated, never filtered on what an arm or an untargeted reader finds.
- ``arms.run_arm(arm: ArmName, life: Life, models: ArmModels, *, budget: Budget = Budget()) -> ArmRun``
  (17-eval): one arm through every night of the life, with the shared ``brief_v1.md``, the life's
  own ``evidence_search``, the same ``world_query`` and ``report_finding``, a fresh ``budget``
  each night and ``targets=life.targets[night]``. M-persist passes ``wipe=True`` on each night's
  first call, M-kernel runs ``NO_KERNEL`` and M-control runs ``control="newest"``; all three are
  logged, so ``kernel.fold`` of ``ArmRun.events`` rebuilds each MIND arm's workspace.
- ``score.score_life(life: Life, problems: ProblemFile, run: ArmRun, judge: ApdJudge, *, view: JudgeView) -> LifeScore``
  (17-eval, which defines the ``ApdJudge`` port for ``judge_apd_v1.md`` in ``score.py``): a
  finding matches an item when it cites one of the item's origin ids and the judge, shown only the
  evidence through the report's day, rules it the same problem; each item is scored once, from
  its canonical state at the end of the day before its surface day.
- ``score.sign_test(m: Sequence[float], b: Sequence[float]) -> SignTest`` (17-eval): the exact
  one-sided binomial test (p = 1/2) of M over B on paired life totals, ties dropped and counted.
- ``interventions.resume_after_edit(events: Sequence[WorkspaceEvent], edit: Edit, ports: InquiryPorts, budget: Budget, *, night: date, targets: Sequence[Target], as_of: datetime) -> tuple[InquiryRun, InquiryRun]``
  (17-eval): ``events`` is a snapshot, a prefix of a development life's M log. Each branch gets
  its own fresh in-memory log seeded with ``events``; the edited branch then logs ``edit`` as a
  step stamped ``as_of``. Both resume ``loop.run_inquiry`` from the fold of their log, with the
  evidence held fixed at ``as_of``, so each branch is a fold of its own log. Returns
  (unedited, edited).
"""
from __future__ import annotations

import dataclasses
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
from anjo.sleep.mind.inquiry.types import (
    FINDING_KINDS,
    Answered,
    FindingKind,
    Resolved,
    Turn,
    WorkspaceEvent,
    check_turns,
)
from anjo.sleep.mind.inquiry.types import Edit as Edit
from anjo.sleep.mind.inquiry.types import EditKind as EditKind
from anjo.sleep.types import Evidence, Target

Family = Literal["contradiction", "gap", "opportunity", "reversal", "decoy"]
Split = Literal["development", "held_out"]
CoreSplit = Literal["core-train", "core-val", "core-test"]  # MIND core's lives (``core_splits``), outside APD's protocol
LifeSplit = Literal[Split, CoreSplit]
ArmName = Literal["A", "B", "M", "M-persist", "M-kernel", "M-control", "M+replay", "M+shuffled"]
JudgeView = Literal["audited", "gemini"]  # audited: Muse Spark, with Claude's blind audit where the judges differ
ApdGate = Literal["leak", "discoverability", "script_audit"]

FAMILIES: Final = frozenset(get_args(Family))
PROBLEM_FAMILIES: Final = frozenset({"contradiction", "gap", "opportunity"})
ARMS: Final = frozenset(get_args(ArmName))
DEVELOPMENT_ARMS: Final = frozenset({"B", "M"})
HELD_OUT_ONLY_ARMS: Final = frozenset({"M-kernel", "M-control", "M+replay", "M+shuffled"})
WORKSPACE_ARMS: Final = ARMS - {"A", "B"}  # the arms that keep a workspace log
CORE_SPLITS: Final[tuple[CoreSplit, ...]] = get_args(CoreSplit)
LIFE_SPLITS: Final = frozenset(get_args(LifeSplit))
LABELLED_SPLITS: Final = frozenset({"development", "core-train"})  # the splits whose items the core may train on

FINDING_CAP: Final = 8  # distinct finding ids per arm and life; updating one in place is free
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
    names the problem template it came from (the splits share no template). ``topic`` is the
    neutral area it lies in ("scene 4's shoot"), written with the item and never built from its
    resolution or template: the discoverability gate's reader gets it and nothing else.
    """

    id: str
    family: Family
    template: str
    topic: str
    surface_day: date
    origin_ids: tuple[str, ...]
    disambiguating_ids: tuple[str, ...]
    resolution: str

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_one_of("family", self.family, FAMILIES)
        check_text("template", self.template)
        check_text("topic", self.topic)
        if self.topic in (self.template, self.resolution):
            raise ValueError("an item's topic is neutral: neither its template nor its resolution")
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
    life's id). Every evidence item records the model that produced it.

    ``targets`` holds, for every day, the targets as they stood when that day ended, mentioned on
    or before it. The harness passes ``targets[night]`` to that night's run, so priority's stake
    never sees a plan or deadline the person hasn't mentioned yet.
    """

    id: str
    field: str
    seed: int
    split: LifeSplit
    days: tuple[date, ...]
    sessions: tuple[Session, ...]
    evidence: tuple[Evidence, ...]
    targets: Mapping[date, tuple[Target, ...]] = dataclasses.field(hash=False)  # `field` is a life's field here

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_text("field", self.field)
        check_one_of("split", self.split, LIFE_SPLITS)
        freeze_tuple(self, "days", date)
        freeze_tuple(self, "sessions", Session)
        freeze_tuple(self, "evidence", Evidence)
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
        self._freeze_targets(days)

    def _freeze_targets(self, days: frozenset[date]) -> None:
        targets = dict(self.targets)
        if set(targets) != days:
            raise ValueError("a life has its targets for every day, and only its days")
        for day, held in targets.items():
            if isinstance(held, str) or not all(isinstance(target, Target) for target in held):
                raise TypeError("a day's targets are a sequence of Target")
            if any(target.person_id != self.id or target.last_mentioned > day for target in held):
                raise ValueError("a day's targets are this life's, mentioned by that day")
        object.__setattr__(self, "targets", MappingProxyType({day: tuple(targets[day]) for day in self.days}))


# ── A counterfactual pair (MIND's data) ───────────────────────────────────────

PAIR_DECISIONS: Final = MappingProxyType({"failed": "speak", "kept": "wait"})  # on the last day, about the slot


def _evidence_on(life: Life, day: date) -> tuple[Evidence, ...]:
    return tuple(item for item in life.evidence if item.day == day)


@dataclass(frozen=True)
class CounterfactualPair:
    """Two histories of one person (``pairs``) that differ only in the outcome of a commitment.

    Both share every day but ``fork_days``, where the commitment fails twice (``failed``) or is
    kept twice (``kept``), and end on the same last day, word for word, with the same evidence. On
    that day the failed history should speak about the commitment's slot (its planted gap) and the
    kept one should wait (``PAIR_DECISIONS``); ``unrelated_item``, planted the same in both, should
    be decided the same in both. ``slot_ids`` is the commitment's evidence, ``failed_ids`` and
    ``kept_ids`` each history's outcome evidence, ``final_ids`` the last day's.
    """

    id: str
    template: str
    fork_days: tuple[date, ...]
    failed: Life
    failed_problems: ProblemFile
    kept: Life
    kept_problems: ProblemFile
    slot_ids: tuple[str, ...]
    failed_ids: tuple[str, ...]
    kept_ids: tuple[str, ...]
    unrelated_item: str
    final_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        check_text("id", self.id)
        check_text("template", self.template)
        freeze_tuple(self, "fork_days", date)
        for name in ("slot_ids", "failed_ids", "kept_ids", "final_ids"):
            freeze_ids(self, name, empty_ok=False)
        if {self.failed.id, self.kept.id, self.failed_problems.life_id, self.kept_problems.life_id} != {self.id}:
            raise ValueError("a pair's two histories are one person's, under the pair's id")
        if self.failed.days != self.kept.days or not set(self.fork_days) < set(self.failed.days[:-1]):
            raise ValueError("a pair's histories share their days, and fork before the last")
        self._check_shared()
        self._check_items()

    def _check_shared(self) -> None:
        for day in self.failed.days:
            same = (_evidence_on(self.failed, day) == _evidence_on(self.kept, day)
                    and [s for s in self.failed.sessions if s.day == day] == [s for s in self.kept.sessions if s.day == day])
            if same == (day in self.fork_days):
                raise ValueError(f"a pair's histories differ on exactly its fork days ({day})")
        last = self.failed.days[-1]
        if self.final_ids != tuple(item.id for item in _evidence_on(self.failed, last) if not item.held_out):
            raise ValueError("final_ids is the shared last day's evidence")
        failed_ids = {item.id for item in self.failed.evidence}
        kept_ids = {item.id for item in self.kept.evidence}
        if not set(self.slot_ids) <= failed_ids & kept_ids or not set(self.failed_ids) <= failed_ids - kept_ids \
                or not set(self.kept_ids) <= kept_ids - failed_ids:
            raise ValueError("the slot is in both histories, and each outcome in its own only")

    def _check_items(self) -> None:
        unrelated = [item for item in self.kept_problems.items if item.id == self.unrelated_item]
        if len(self.kept_problems.items) != 1 or not unrelated or unrelated[0] not in self.failed_problems.items:
            raise ValueError("the kept history plants only the unrelated item, the same in both")
        gaps = [item for item in self.failed_problems.items if item.id != self.unrelated_item]
        if len(gaps) != 1 or gaps[0].family != "gap" or not set(self.slot_ids) <= set(gaps[0].origin_ids):
            raise ValueError("the failed history plants one gap, from the slot")


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
class FindingRegistry:
    """``report_finding``'s registry for one arm on one life, kept across every night. A report
    under an id it holds updates that finding in place, free; a new id is accepted while fewer
    than ``FINDING_CAP`` are held. Reports run in ``seq`` order."""

    reports: tuple[FindingReport, ...] = ()

    def __post_init__(self) -> None:
        freeze_tuple(self, "reports", FindingReport)
        seqs = [report.seq for report in self.reports]
        if seqs != sorted(set(seqs)):
            raise ValueError("reports run in seq order, each seq once")
        if len(self.finding_ids) > FINDING_CAP:
            raise ValueError(f"a life holds at most {FINDING_CAP} distinct findings")

    @property
    def finding_ids(self) -> frozenset[str]:
        return frozenset(report.finding_id for report in self.reports)

    def accepts(self, finding_id: str) -> bool:
        """Whether a report under ``finding_id`` fits the cap."""
        return finding_id in self.finding_ids or len(self.finding_ids) < FINDING_CAP

    def add(self, report: FindingReport) -> FindingRegistry:
        """The registry with ``report`` added; ``ValueError`` past the cap or out of order."""
        return FindingRegistry((*self.reports, report))


@dataclass
class NightClock:
    """The night the calls being made belong to; ``None`` before any is set. Every arm sets it as
    each night starts, and a served Glimmer dates its prompts on it, so each prompt is dated on the
    night whose evidence it sees."""

    day: date | None = None


WrittenPort = Literal["world", "resolver"]
WRITTEN_PORTS: Final = frozenset(get_args(WrittenPort))


@dataclass(frozen=True)
class Written:
    """The model that wrote one logged observation, as its API reported it: a ``world_query``
    answer (``Answered``) or a resolver's verdict (``Resolved``), keyed to its step's ``seq``."""

    seq: int
    port: WrittenPort
    model: str

    def __post_init__(self) -> None:
        check_count("seq", self.seq)
        check_one_of("port", self.port, WRITTEN_PORTS)
        check_text("model", self.model)


@dataclass
class Reported:
    """The model the latest ``world_query`` answer and resolver reply named, as reported, held
    until the step that logs it (``arms`` pairs them). A failed call reports nothing."""

    last: dict[str, str] = field(default_factory=dict)

    def saw(self, port: WrittenPort, model: str) -> None:
        self.last[port] = model

    def take(self, port: WrittenPort) -> str | None:
        return self.last.pop(port, None)


@dataclass(frozen=True)
class ArmModels:
    """The model-backed ports every arm shares: the same Glimmer at the same precision as the
    thinker, the same ``world_query``, the screen and the resolver (MIND's arms use the last
    two). ``clock`` is the night the thinker and world query date their prompts on, which every
    arm sets as each night starts; ``reported`` receives the model each world answer and resolver
    reply named (live ports only: ``apd.live``)."""

    thinker: Thinker
    world: WorldQuery
    screen: ScreenPort
    resolver: PredictionResolver
    clock: NightClock | None = field(default=None, compare=False)
    reported: Reported | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ArmRun:
    """One arm's run through one life.

    ``reports`` is its ``FindingRegistry``'s reports. ``tokens`` is each night's capped tokens,
    in order. ``contexts`` is the arm's state as text at the end of each night (M's findings, B's
    notebook, A's reflections), for the decision-quality talker. ``events`` is the workspace log
    of a MIND arm, empty for A and B: the start states Task 17b replays and the log kill-resume
    rebuilds from. ``turns`` is every thinker call of a MIND arm, each indexed to its step in
    ``events``, for Task 17b's replay dataset. ``written`` is the model each logged world answer
    and resolver verdict came from, as reported (``Written``): replay trains on a start state only
    when every one it holds is recorded.
    """

    arm: ArmName
    life_id: str
    reports: tuple[FindingReport, ...]
    tokens: tuple[int, ...]
    contexts: Mapping[date, str] = field(hash=False)
    events: tuple[WorkspaceEvent, ...] = ()
    turns: tuple[Turn, ...] = ()
    written: tuple[Written, ...] = ()

    def __post_init__(self) -> None:
        check_one_of("arm", self.arm, ARMS)
        check_text("life_id", self.life_id)
        freeze_tuple(self, "reports", FindingReport)
        freeze_tuple(self, "tokens", int)
        freeze_tuple(self, "events", WorkspaceEvent)
        freeze_tuple(self, "turns", Turn)
        object.__setattr__(self, "contexts", MappingProxyType(dict(self.contexts)))
        for night in self.tokens:
            check_count("tokens", night)
        FindingRegistry(self.reports)
        freeze_tuple(self, "written", Written)
        if (self.events or self.turns or self.written) and self.arm not in WORKSPACE_ARMS:
            raise ValueError(f"arm {self.arm} keeps no workspace")
        check_turns(self.turns, self.events)
        check_written(self.written, self.events)


def model_written(entry: object) -> WrittenPort | None:
    """The port whose model wrote a logged entry: a world answer with text, or a resolver's
    verdict (a censored prediction is the code's, and a failed world call's empty answer holds no
    model's text); ``None`` for anything else."""
    if isinstance(entry, Answered) and entry.answer:
        return "world"
    if isinstance(entry, Resolved) and entry.outcome != "censored":
        return "resolver"
    return None


def check_written(written: tuple[Written, ...], events: tuple[WorkspaceEvent, ...]) -> None:
    """Each record names a logged world answer or resolver verdict of its own port, once."""
    ports = {event.step.seq: model_written(event.step.entry) for event in events}
    if len({w.seq for w in written}) != len(written) or any(ports.get(w.seq) != w.port for w in written):
        raise ValueError("each written record names a logged world answer or resolver verdict, once")


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


# ── The gates ─────────────────────────────────────────────────────────────────
# (The intervention test's ``Edit`` is the inquiry's, re-exported above: it is a logged step.)


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
