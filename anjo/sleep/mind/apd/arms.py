"""The APD arms (Task 17): one arm through every night of one synthetic life, on equal terms.

Every arm gets the same things, and only these:

- the same thinker (one Glimmer at one precision, ``ArmModels.thinker``), with the same discovery
  brief, ``prompts/brief_v1.md``, in front of every call (``BriefedThinker``);
- the life's own ``evidence_search`` (``LifeSearch``), masked to each night, and the same
  ``world_query``;
- the same budget, fresh each night: ``CYCLE_CAP`` cycles and ``TOKEN_CAP`` generated tokens,
  retries and screening included, with every capped call given the tokens left as ``max_tokens``;
- the same ``report_finding``, stamped into one ``FindingRegistry`` per arm and life, so its
  ``FINDING_CAP`` distinct findings hold across every night and M-persist's wipes. A cite of an
  item the arm couldn't have seen by that night is dropped from the report.

How the arms differ:

- **A, full memory:** one reflection call a night with every item said so far in context. Its
  reply ends with a JSON array of ``report_finding`` calls; its reflections are its state.
- **B, the notebook agent:** a ReAct loop with the same tools and a free-text notebook it keeps
  across nights. Its night has MIND's shape: one opening call (MIND's notice pass) and then up to
  ``cycle_cap`` cycles, one thinker call each; a reply with no action costs its cycle.
- **M and its ablations:** 17-core's ``loop.notice`` over the day's new evidence, then
  ``loop.run_inquiry``, on one in-memory workspace log. M-persist wipes the workspace on each
  night's first call, M-kernel runs ``NO_KERNEL`` and M-control ``control="newest"``. M+replay and
  M+shuffled run as M: they are Task 17b's hooks, and its adapter arrives as ``ArmModels.thinker``.
  After every loop call, each workspace finding that changed is reported under its own id.

A night is masked to the end of its day, UTC (``night_as_of``), so every item must be stamped
within its own day (``check_clock``). 17-core's functions arrive as a
``MindCore``; ``mind_core()`` imports them lazily, so these modules load, and tests run on fakes,
without them.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

from anjo.sleep.context.template import render
from anjo.sleep.mind.apd.types import (
    DEVELOPMENT_ARMS,
    FINDING_CAP,
    ArmModels,
    ArmName,
    ArmRun,
    FindingRegistry,
    FindingReport,
    Life,
)
from anjo.sleep.mind.inquiry.ports import InquiryPorts, Thinker, WorkspaceLog
from anjo.sleep.mind.inquiry.types import (
    FINDING_KINDS,
    FULL_RULES,
    NO_KERNEL,
    RECENT_OPS,
    Applied,
    Budget,
    ControlMode,
    FindingKind,
    InquiryRun,
    KernelRules,
    ModelReply,
    Rejected,
    Step,
    Turn,
    Workspace,
    WorkspaceEvent,
)
from anjo.sleep.types import Evidence, Target

BRIEF_PATH: Final = Path(__file__).parent / "prompts" / "brief_v1.md"
APD_TOOLS: Final = frozenset({"evidence_search", "world_query", "report_finding"})  # no research: APD is offline
SEARCH_K: Final = 5
A_EVIDENCE_HEADING: Final = "## Everything the person has said, up to tonight"
B_STEPS_HEADING: Final = "## Your steps so far tonight"
_NONE: Final = "none"
_WORD: Final = re.compile(r"\w+")


# ── 17-core, injected ─────────────────────────────────────────────────────────


class NoticeFn(Protocol):
    def __call__(
        self, ws: Workspace, new: Sequence[Evidence], ports: InquiryPorts, budget: Budget, *, night: date,
        targets: Sequence[Target], as_of: datetime, wipe: bool = False,
    ) -> InquiryRun: ...


class InquiryFn(Protocol):
    def __call__(
        self, ws: Workspace, ports: InquiryPorts, budget: Budget, *, night: date, targets: Sequence[Target],
        as_of: datetime, focus: str | None = None, rules: KernelRules = FULL_RULES,
        control: ControlMode = "priority", wipe: bool = False,
    ) -> InquiryRun: ...


class ApplyFn(Protocol):
    def __call__(self, ws: Workspace, step: Step, *, rules: KernelRules = FULL_RULES) -> Applied | Rejected: ...


@dataclass(frozen=True)
class MindCore:
    """17-core's frozen functions, as the harness calls them: ``loop.notice``,
    ``loop.run_inquiry``, ``kernel.apply``, ``kernel.fold`` and ``kernel.digest``."""

    notice: NoticeFn
    run_inquiry: InquiryFn
    apply: ApplyFn
    fold: Callable[[Iterable[WorkspaceEvent]], Workspace]
    digest: Callable[[Workspace], str]


def mind_core() -> MindCore:
    """The real ``kernel`` and ``loop``, imported only when a MIND arm runs."""
    kernel = importlib.import_module("anjo.sleep.mind.inquiry.kernel")
    loop = importlib.import_module("anjo.sleep.mind.inquiry.loop")
    return MindCore(
        notice=loop.notice, run_inquiry=loop.run_inquiry, apply=kernel.apply, fold=kernel.fold,
        digest=kernel.digest,
    )


# ── The life's own tools ──────────────────────────────────────────────────────


def night_as_of(night: date) -> datetime:
    """The moment a night's tools are masked to: the end of its day, UTC."""
    return datetime.combine(night, time.max, tzinfo=timezone.utc)


def check_clock(life: Life) -> None:
    """Refuse a life whose items aren't stamped within their own day, UTC, so that each night
    sees its own day's items and none from later."""
    for item in life.evidence:
        if not night_as_of(item.day - timedelta(days=1)) < item.created_at <= night_as_of(item.day):
            raise ValueError(f"evidence {item.id} isn't stamped within its own day (UTC)")


def seen_by(evidence: Iterable[Evidence], as_of: datetime) -> tuple[Evidence, ...]:
    """The items an arm may see as of ``as_of``: said on or before its day and created by then,
    held-out items never."""
    return tuple(
        item for item in evidence if not item.held_out and item.day <= as_of.date() and item.created_at <= as_of
    )


def _words(text: str) -> frozenset[str]:
    return frozenset(word.casefold() for word in _WORD.findall(text))


@dataclass(frozen=True)
class LifeSearch:
    """``evidence_search`` over one life (the ``EvidenceSearch`` port): lexical overlap, as in
    the evidence store, with the newer item first on equal overlap, masked to ``as_of``."""

    evidence: tuple[Evidence, ...]

    def search(self, query: str, as_of: datetime, k: int) -> tuple[Evidence, ...]:
        wanted = _words(query)
        scored = [(len(wanted & _words(item.text)), item) for item in seen_by(self.evidence, as_of)]
        ranked = sorted(
            (pair for pair in scored if pair[0] > 0),
            key=lambda pair: (-pair[0], -pair[1].created_at.timestamp(), pair[1].id),
        )
        return tuple(item for _, item in ranked[: max(k, 0)])

    def get(self, ids: Sequence[str], as_of: datetime) -> tuple[Evidence, ...]:
        held = {item.id: item for item in seen_by(self.evidence, as_of)}
        return tuple(held[item_id] for item_id in ids if item_id in held)


class MemoryLog:
    """An APD run's in-memory ``WorkspaceLog``, seeded with a snapshot's events when given."""

    def __init__(self, events: Sequence[WorkspaceEvent] = ()) -> None:
        self._events: list[WorkspaceEvent] = []
        for event in events:
            self.append(event)

    def append(self, event: WorkspaceEvent) -> None:
        expected = self._events[-1].step.seq + 1 if self._events else 0
        if event.step.seq != expected:
            raise ValueError(f"step {event.step.seq} isn't the log's next ({expected})")
        self._events.append(event)

    def events(self) -> tuple[WorkspaceEvent, ...]:
        return tuple(self._events)


def load_brief() -> str:
    """The shared discovery brief, ``brief_v1.md``."""
    return BRIEF_PATH.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class BriefedThinker:
    """The shared thinker with the brief in front of every call, for every arm. (MIND's
    ``Turn.prompt`` is the loop's rendered text, without the brief.)"""

    inner: Thinker
    brief: str

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        return self.inner.propose(f"{self.brief}\n\n{rendered}", max_tokens)


# ── report_finding ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FindingCall:
    """One ``report_finding(problem, conclusion, p, cites, kind, finding_id=None)`` call."""

    problem: str
    conclusion: str
    p: float
    cites: tuple[str, ...]
    kind: FindingKind
    finding_id: str | None = None


@dataclass(frozen=True)
class Filed:
    """What a ``report_finding`` call did: the registry after it, the stamped report (``None``
    when refused, which records it nowhere) and what the arm is told."""

    registry: FindingRegistry
    report: FindingReport | None
    note: str


def _free_id(held: frozenset[str]) -> str:
    n = len(held) + 1
    while f"f{n}" in held:
        n += 1
    return f"f{n}"


def report_finding(
    registry: FindingRegistry, call: FindingCall, *, day: date, citable: frozenset[str], new_id: str | None = None,
) -> Filed:
    """File ``call`` on night ``day``. A held ``finding_id`` updates that finding in place; an id
    the registry doesn't hold is refused; none opens a new finding under ``new_id`` (MIND's
    arms: the workspace finding's own id) or the next free ``f<n>``. Past ``FINDING_CAP`` distinct
    ids a new finding is refused. Cites outside ``citable`` are dropped."""
    held = registry.finding_ids
    if call.finding_id is not None and call.finding_id not in held:
        return Filed(registry, None, f"refused: there is no finding {call.finding_id} to update")
    finding_id = call.finding_id or new_id or _free_id(held)
    if not registry.accepts(finding_id):
        return Filed(registry, None, f"refused: all {FINDING_CAP} findings are taken; update one in place")
    try:
        report = FindingReport(
            finding_id=finding_id, day=day, seq=registry.reports[-1].seq + 1 if registry.reports else 0,
            problem=call.problem, conclusion=call.conclusion, p=call.p,
            cites=tuple(cite for cite in call.cites if cite in citable), kind=call.kind,
        )
    except (TypeError, ValueError):
        return Filed(registry, None, "refused: a finding needs a problem, a conclusion, p in [0, 1] and a kind")
    return Filed(registry.add(report), report, f"reported {finding_id}")


def finding_call(data: object) -> FindingCall | None:
    """A ``report_finding`` call read from a model's JSON object, or ``None``. Other keys (B's
    ``tool``) are ignored."""
    if not isinstance(data, dict):
        return None
    p, cites, kind, finding_id = data.get("p"), data.get("cites"), data.get("kind"), data.get("finding_id")
    problem, conclusion = data.get("problem"), data.get("conclusion")
    if not (
        isinstance(problem, str) and isinstance(conclusion, str) and kind in FINDING_KINDS
        and isinstance(p, (int, float)) and not isinstance(p, bool)
        and isinstance(cites, list) and all(isinstance(cite, str) for cite in cites)
        and (finding_id is None or isinstance(finding_id, str))
    ):
        return None
    return FindingCall(problem, conclusion, float(p), tuple(cites), cast(FindingKind, kind), finding_id)


def standing(reports: Iterable[FindingReport]) -> tuple[FindingReport, ...]:
    """Each finding's latest report, in the order the findings were opened."""
    latest: dict[str, FindingReport] = {}
    for report in reports:
        latest[report.finding_id] = report
    return tuple(latest.values())


def findings_text(reports: Iterable[FindingReport]) -> str:
    """An arm's findings as it and the decision-quality talker read them."""
    lines = [
        f"- {r.finding_id} ({r.kind}, p {r.p:.2f}, {r.day}): {r.problem} | {r.conclusion} [cites: {', '.join(r.cites)}]"
        for r in standing(reports)
    ]
    return "\n".join(lines) or _NONE


def json_values(text: str) -> list[object]:
    """Every top-level JSON array or object in a model's text, in order."""
    decoder = json.JSONDecoder()
    values: list[object] = []
    index = 0
    while index < len(text):
        if text[index] not in "[{":
            index += 1
            continue
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        values.append(value)
    return values


# ── What every prompt shows ───────────────────────────────────────────────────


@dataclass(frozen=True)
class NightInputs:
    """What one night of a life gives every arm: its clock, its targets, the day's new items and
    the ids an arm may cite."""

    night: date
    as_of: datetime
    targets: tuple[Target, ...]
    seen: tuple[Evidence, ...]
    new: tuple[Evidence, ...]
    citable: frozenset[str]


def night_inputs(life: Life, night: date) -> NightInputs:
    as_of = night_as_of(night)
    seen = seen_by(life.evidence, as_of)
    return NightInputs(
        night=night, as_of=as_of, targets=life.targets[night], seen=seen,
        new=tuple(item for item in seen if item.day == night), citable=frozenset(item.id for item in seen),
    )


def _items_text(items: Iterable[Evidence]) -> str:
    return "\n".join(f"[{item.id}] {item.day}: {item.text}" for item in items) or _NONE


def _targets_text(targets: Iterable[Target]) -> str:
    lines = [
        f"- {t.id}: {t.label} ({t.kind}, {t.status}"
        + (f", due {t.check_after:%Y-%m-%d})" if t.check_after else ")")
        for t in targets
    ]
    return "\n".join(lines) or _NONE


_FINDING_SHAPE: Final = (
    '{"problem": "...", "conclusion": "...", "p": 0.4, "cites": ["<item id>"], "kind": "suspicion", '
    '"finding_id": "<only to update one of yours>"}'
)

# ── A, full memory ────────────────────────────────────────────────────────────

_A_PROMPT: Final = "\n".join((
    "## Tonight: {{NIGHT}}",
    "",
    "## Open plans and threads",
    "{{TARGETS}}",
    "",
    "## Your findings so far",
    "{{FINDINGS}}",
    "",
    A_EVIDENCE_HEADING,
    "{{EVIDENCE}}",
    "",
    "## Your reply",
    "Reflect on everything above, then end with one JSON array of report_finding calls, each:",
    _FINDING_SHAPE,
    "An empty array reports nothing.",
))


def _run_a(life: Life, models: ArmModels, budget: Budget, brief: str) -> ArmRun:
    thinker = BriefedThinker(models.thinker, brief)
    registry = FindingRegistry()
    tokens: tuple[int, ...] = ()
    contexts: dict[date, str] = {}
    for night in life.days:
        inputs = night_inputs(life, night)
        prompt = render(_A_PROMPT, {
            "NIGHT": str(night), "TARGETS": _targets_text(inputs.targets),
            "FINDINGS": findings_text(registry.reports), "EVIDENCE": _items_text(inputs.seen),
        })
        reply = thinker.propose(prompt, budget.tokens_left)
        arrays = [value for value in json_values(reply.text) if isinstance(value, list)]
        for call in (finding_call(item) for item in (arrays[-1] if arrays else ())):
            if call is not None:
                registry = report_finding(registry, call, day=night, citable=inputs.citable).registry
        tokens, contexts = (*tokens, reply.tokens), {**contexts, night: reply.text}
    return ArmRun(arm="A", life_id=life.id, reports=registry.reports, tokens=tokens, contexts=contexts)


# ── B, the notebook agent ─────────────────────────────────────────────────────

_B_PROMPT: Final = "\n".join((
    "## Tonight: {{NIGHT}}, step {{STEP}} of {{STEPS}}",
    "",
    "## Open plans and threads",
    "{{TARGETS}}",
    "",
    "## Your notebook, kept across nights",
    "{{NOTEBOOK}}",
    "",
    "## Your findings so far",
    "{{FINDINGS}}",
    "",
    "## What the person said today",
    "{{NEW}}",
    "",
    B_STEPS_HEADING,
    "{{RECENT}}",
    "",
    "## Your reply",
    'One JSON object: {"notebook": "...", "action": {...}}. "notebook" is optional; when present it '
    "replaces your whole notebook. Track questions, hypotheses and tests in it in any structure you "
    'like. "action" is one of:',
    '{"tool": "evidence_search", "query": "..."}',
    '{"tool": "world_query", "text": "..."}',
    '{"tool": "report_finding", ' + _FINDING_SHAPE[1:],
    '{"tool": "stop"}',
))
_NO_ACTION: Final = "- (no action: the reply held no JSON object with an action)"


@dataclass(frozen=True)
class _Desk:
    """What B keeps across nights, and its tools."""

    registry: FindingRegistry
    notebook: str
    thinker: Thinker
    models: ArmModels
    search: LifeSearch


def _b_reply(text: str) -> tuple[str | None, dict[str, object] | None]:
    """B's notebook (``None``: unchanged) and action from its reply's last JSON object that holds
    an action."""
    for value in reversed(json_values(text)):
        if isinstance(value, dict) and isinstance(value.get("action"), dict):
            notebook = value.get("notebook")
            return (notebook if isinstance(notebook, str) else None), value["action"]
    return None, None


def _b_act(
    action: Mapping[str, object], inputs: NightInputs, desk: _Desk, budget: Budget,
) -> tuple[str, _Desk, Budget]:
    """One tool call: what B is told, its desk and its budget after it."""
    tool, query, text = action.get("tool"), action.get("query"), action.get("text")
    if tool == "evidence_search" and isinstance(query, str):
        found = desk.search.search(query, inputs.as_of, SEARCH_K)
        lines = "\n".join(f"  [{item.id}] {item.day}: {item.text}" for item in found) or "  nothing"
        return f'- evidence_search "{query}" found:\n{lines}', desk, budget
    if tool == "world_query" and isinstance(text, str):
        if not budget.tokens_left:
            return "- world_query: no tokens left tonight", desk, budget
        answer = desk.models.world.ask(text, budget.tokens_left)
        return f'- world_query "{text}": {answer.text}', desk, budget.spend(tokens=answer.tokens)
    call = finding_call(action) if tool == "report_finding" else None
    if call is not None:
        filed = report_finding(desk.registry, call, day=inputs.night, citable=inputs.citable)
        return f"- report_finding: {filed.note}", dataclasses.replace(desk, registry=filed.registry), budget
    return f"- (no action: {tool!r} isn't a tool with its arguments)", desk, budget


def _b_night(inputs: NightInputs, desk: _Desk, budget: Budget) -> tuple[_Desk, Budget]:
    """One opening call (MIND's notice pass), then up to ``cycle_cap`` cycles of one call each."""
    recent: tuple[str, ...] = ()
    step = 0
    while budget.stop_reason is None:
        step += 1
        prompt = render(_B_PROMPT, {
            "NIGHT": str(inputs.night), "STEP": str(step), "STEPS": str(budget.cycle_cap + 1),
            "TARGETS": _targets_text(inputs.targets), "NOTEBOOK": desk.notebook or "(empty)",
            "FINDINGS": findings_text(desk.registry.reports), "NEW": _items_text(inputs.new),
            "RECENT": "\n".join(recent) or _NONE,
        })
        reply = desk.thinker.propose(prompt, budget.tokens_left)
        budget = budget.spend(cycles=0 if step == 1 else 1, tokens=reply.tokens)
        notebook, action = _b_reply(reply.text)
        if notebook is not None:
            desk = dataclasses.replace(desk, notebook=notebook)
        if action is not None and action.get("tool") == "stop":
            break
        line = _NO_ACTION
        if action is not None:
            line, desk, budget = _b_act(action, inputs, desk, budget)
        recent = (*recent, line)[-RECENT_OPS:]
    return desk, budget


def _run_b(life: Life, models: ArmModels, budget: Budget, brief: str) -> ArmRun:
    desk = _Desk(FindingRegistry(), "", BriefedThinker(models.thinker, brief), models, LifeSearch(life.evidence))
    tokens: tuple[int, ...] = ()
    contexts: dict[date, str] = {}
    for night in life.days:
        desk, spent = _b_night(night_inputs(life, night), desk, budget)
        tokens, contexts = (*tokens, spent.tokens_used), {**contexts, night: desk.notebook}
    return ArmRun(arm="B", life_id=life.id, reports=desk.registry.reports, tokens=tokens, contexts=contexts)


# ── MIND's arms ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MindArm:
    """How a MIND arm calls the loop: its kernel rules, its control, and whether each night's
    first call wipes the workspace (M-persist)."""

    rules: KernelRules = FULL_RULES
    control: ControlMode = "priority"
    wipe: bool = False


_MIND_ARMS: dict[ArmName, MindArm] = {
    "M": MindArm(), "M-persist": MindArm(wipe=True), "M-kernel": MindArm(rules=NO_KERNEL),
    "M-control": MindArm(control="newest"), "M+replay": MindArm(), "M+shuffled": MindArm(),
}
MIND_ARMS: Final[Mapping[ArmName, MindArm]] = MappingProxyType(_MIND_ARMS)


def mind_ports(models: ArmModels, life: Life, log: WorkspaceLog, brief: str) -> InquiryPorts:
    """A MIND arm's ports on one life: the shared models behind the brief, the life's search, and
    ``log``. No research: APD runs offline."""
    return InquiryPorts(
        thinker=BriefedThinker(models.thinker, brief), screen=models.screen, search=LifeSearch(life.evidence),
        world=models.world, resolver=models.resolver, log=log,
    )


def mind_night(
    arm: MindArm, inputs: NightInputs, ws: Workspace, ports: InquiryPorts, budget: Budget, core: MindCore,
) -> tuple[InquiryRun, InquiryRun]:
    """One night: the notice pass over the day's new items (the night's first call, so M-persist
    wipes there), then the inquiry under the budget the notice pass left."""
    noticed = core.notice(
        ws, inputs.new, ports, budget, night=inputs.night, targets=inputs.targets, as_of=inputs.as_of, wipe=arm.wipe,
    )
    inquired = core.run_inquiry(
        noticed.ws, ports, noticed.budget, night=inputs.night, targets=inputs.targets, as_of=inputs.as_of,
        rules=arm.rules, control=arm.control,
    )
    return noticed, inquired


def sync_findings(registry: FindingRegistry, ws: Workspace, *, day: date, citable: frozenset[str]) -> FindingRegistry:
    """Report each workspace finding that changed since its last report, under its own id."""
    latest = {report.finding_id: report for report in registry.reports}
    for finding in ws.findings:
        question = ws.question(finding.question_id)
        call = FindingCall(
            problem=question.subject if question else finding.statement, conclusion=finding.statement, p=finding.p,
            cites=tuple(cite for cite in finding.cites if cite in citable), kind=finding.kind,
            finding_id=finding.id if finding.id in latest else None,
        )
        before = latest.get(finding.id)
        if before is not None and (before.problem, before.conclusion, before.p, before.cites, before.kind) == (
            call.problem, call.conclusion, call.p, call.cites, call.kind,
        ):
            continue
        registry = report_finding(registry, call, day=day, citable=citable, new_id=finding.id).registry
    return registry


def _run_mind(name: ArmName, life: Life, models: ArmModels, budget: Budget, core: MindCore, brief: str) -> ArmRun:
    arm, log = MIND_ARMS[name], MemoryLog()
    ports = mind_ports(models, life, log, brief)
    ws, registry = Workspace(), FindingRegistry()
    tokens: tuple[int, ...] = ()
    turns: tuple[Turn, ...] = ()
    contexts: dict[date, str] = {}
    for night in life.days:
        inputs = night_inputs(life, night)
        runs = mind_night(arm, inputs, ws, ports, budget, core)
        for run in runs:
            registry = sync_findings(registry, run.ws, day=night, citable=inputs.citable)
            turns = (*turns, *run.turns)
        ws = runs[-1].ws
        tokens, contexts = (*tokens, runs[-1].budget.tokens_used), {**contexts, night: findings_text(registry.reports)}
    return ArmRun(
        arm=name, life_id=life.id, reports=registry.reports, tokens=tokens, contexts=contexts, events=log.events(),
        turns=turns,
    )


def run_arm(
    arm: ArmName, life: Life, models: ArmModels, *, budget: Budget = Budget(), core: MindCore | None = None,
    brief: str | None = None,
) -> ArmRun:
    """One arm through every night of ``life``, each night starting from ``budget``. A development
    life runs only B and M. ``core`` defaults to the real kernel and loop; ``brief`` to
    ``brief_v1.md``."""
    if life.split == "development" and arm not in DEVELOPMENT_ARMS:
        raise ValueError(f"development lives run {sorted(DEVELOPMENT_ARMS)} only, not {arm}")
    check_clock(life)
    text = load_brief() if brief is None else brief
    if arm == "A":
        return _run_a(life, models, budget, text)
    if arm == "B":
        return _run_b(life, models, budget, text)
    return _run_mind(arm, life, models, budget, core or mind_core(), text)
