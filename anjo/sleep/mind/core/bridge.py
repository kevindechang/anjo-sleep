"""The bridge from MIND's neural core to the rest of her mind. Torch-free: it reads the core's
plain outputs (``Ignition``, ``Decision``, ``Predictions``), so importing it never loads torch.

**Ignition.** The core writes into her workspace only through the kernel. An ignition becomes a
``NOTICE`` that cites the **real** items it names: those the workspace already holds (anything an
imagined state produced is unknown to the kernel and dropped), oldest first. Its targets are the
targets those items name, among the run's. ``risk`` opens an anomaly and ``missing`` a gap. A
``collision`` opens nothing unless ``ANJO_SLEEP_MIND_CONTRADICTIONS=1`` (``contradictions_on``;
off by default: no method has yet beaten the untrained rule "the constraint matches no earlier
event" at telling a conflict from an unrelated plan, so a collision is dropped as
``contradictions_off``); switched on, it opens a contradiction when it cites items from two days
(the kernel's K3), else an anomaly. The subject is neutral, built from the targets' labels. The kernel checks it, the screen
reads it (purpose ``question``) and the step is logged like any other, so a fold of the log
rebuilds the workspace. An ignition with nothing real, no target, or a live question already on
the same targets and kind, is dropped with that reason and logs nothing.

**The switch.** ``ANJO_SLEEP_MIND_CORE=1`` turns the core on for a live night (``core_on``; the
app passes ``os.environ``). Off, ``night_decide`` is the controller (``loop.CONTROLLER``) and
``night_chooser`` the chooser it was given, so the night is today's. On:

- ``CoreController`` keeps the controller's caps, floor, K5 and receipt, and changes one thing:
  which question the next cycle investigates. Questions the core attends to (their origin
  evidence is cited by a core slot) go first, by that slot's risk; the rest follow in the
  controller's priority order. Its runs' ``RunStart`` names it (``CORE_DECIDER``), so a killed run
  resumes under it, never under the controller. In the night, a dream's view picks it
  (``decider_for``) and a resume takes the one its run names (``resume_decider``).
- ``CoreChooser`` lets the core decide whether to speak: a held note (``hold_note``) keeps its
  weight only when the core decides to speak, and only on the targets of the slot it raises.
  Every other want is the inner chooser's, unchanged.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Final, Literal

from anjo.sleep.mind.choose import NOTE_ACTION
from anjo.sleep.mind.core.types import Decision, Ignition, IgnitionKind, Predictions
from anjo.sleep.mind.inquiry import controller, kernel
from anjo.sleep.mind.inquiry.loop import CONTROLLER, Decider
from anjo.sleep.mind.inquiry.ports import ScreenPort, WorkspaceLog
from anjo.sleep.mind.inquiry.types import (
    LIVE_STATUSES,
    PRIORITY_FLOOR,
    Budget,
    ControlMode,
    EvidenceRef,
    NoticeOp,
    QuestionKind,
    Rejected,
    RejectCode,
    Step,
    Workspace,
    WorkspaceEvent,
)
from anjo.sleep.mind.inquiry.types import Decision as ControllerDecision
from anjo.sleep.ports import Chooser
from anjo.sleep.types import Choice, ChoiceContext, Opportunity, Target, Want, WantSnapshot

CORE_ENV: Final = "ANJO_SLEEP_MIND_CORE"
CONTRADICTIONS_ENV: Final = "ANJO_SLEEP_MIND_CONTRADICTIONS"  # "1": a collision opens a question
CORE_DECIDER: Final = "mind-core"  # the name a core-decided run's RunStart records
SCREEN_TOKENS: Final = 256  # the screen's allowance for one ignition's subject
QUESTION_OF: Final[Mapping[IgnitionKind, QuestionKind]] = {
    "risk": "anomaly", "missing": "gap", "collision": "contradiction",
}
DropReason = Literal["nothing_real", "no_targets", "already_open", "contradictions_off"] | RejectCode


def core_on(env: Mapping[str, str]) -> bool:
    """The core's switch for a live night: on only for ``ANJO_SLEEP_MIND_CORE=1``."""
    return env.get(CORE_ENV, "").strip() == "1"


def contradictions_on(env: Mapping[str, str]) -> bool:
    """The switch for the core's collisions: on only for ``ANJO_SLEEP_MIND_CONTRADICTIONS=1``."""
    return env.get(CONTRADICTIONS_ENV, "").strip() == "1"


# ── Ignition ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ignited:
    """The workspace after the ignitions, the events logged, and each ignition dropped (why)."""

    ws: Workspace
    events: tuple[WorkspaceEvent, ...]
    dropped: tuple[tuple[Ignition, DropReason], ...]


def _real(ws: Workspace, ignition: Ignition) -> tuple[EvidenceRef, ...]:
    known = kernel.known_items(ws)
    held = {known[event_id] for event_id in ignition.cited_event_ids if event_id in known}
    return tuple(sorted(held, key=lambda item: (item.day, item.created_at, item.id)))


def _subject(kind: QuestionKind, labels: Sequence[str]) -> str:
    named = " and ".join(labels)
    if kind == "contradiction":
        return f"Whether the plans for {named} agree with each other"
    if kind == "gap":
        return f"What {named} still needs before it can happen"
    return f"Whether {named} goes as planned"


def _notice(ws: Workspace, ignition: Ignition, targets: Sequence[Target]) -> NoticeOp | DropReason:
    real = _real(ws, ignition)
    if not real:
        return "nothing_real"
    held = {target.id: target for target in targets}
    target_ids = tuple(dict.fromkeys(item.target_id for item in real if item.target_id in held))
    if not target_ids:
        return "no_targets"
    kind = QUESTION_OF[ignition.kind]
    if kind == "contradiction" and len({item.day for item in real}) < 2:
        kind = "anomaly"
    if any(q.kind == kind and q.target_ids == target_ids and q.status in LIVE_STATUSES for q in ws.questions):
        return "already_open"
    subject = _subject(kind, [held[target_id].label for target_id in target_ids])
    return NoticeOp(kind, subject, target_ids, tuple(item.id for item in real))


def ignite(
    ws: Workspace, ignitions: Sequence[Ignition], log: WorkspaceLog, *, targets: Sequence[Target], as_of: datetime,
    screen: ScreenPort | None = None, collisions: bool = False,
) -> Ignited:
    """Open a question for each ignition, strongest first, through the kernel and the screen;
    every step is appended to ``log`` the moment it applies. A collision opens one only with
    ``collisions`` (``contradictions_on``)."""
    events: list[WorkspaceEvent] = []
    dropped: list[tuple[Ignition, DropReason]] = []
    target_ids = frozenset(target.id for target in targets)
    for ignition in sorted(ignitions, key=lambda i: (-i.score, i.slot, i.kind)):
        op = _notice(ws, ignition, targets) if collisions or ignition.kind != "collision" else "contradictions_off"
        if not isinstance(op, NoticeOp):
            dropped.append((ignition, op))
            continue
        code, tokens = kernel.check(ws, op, target_ids=target_ids), 0
        if code is None and screen is not None:
            screened = screen.screen(op.subject, "question", SCREEN_TOKENS)
            code, tokens = (None if screened.allowed else "domain_screen"), screened.tokens
        applied = kernel.apply(ws, Step(seq=ws.seq, at=as_of, entry=op, code=code, tokens=tokens))
        if isinstance(applied, Rejected):
            raise RuntimeError(f"the kernel refused an ignition it had admitted ({applied.code})")
        log.append(applied.event)
        ws = applied.ws
        events.append(applied.event)
        if code is not None:
            dropped.append((ignition, code))
    return Ignited(ws, tuple(events), tuple(dropped))


# ── The core's view of a night, and the switch ────────────────────────────────


@dataclass(frozen=True)
class CoreView:
    """What the core said about one person tonight: its decision, its predictions, and the
    target each cited event names (``targets_of``)."""

    decision: Decision
    predictions: Predictions
    targets_of: Mapping[str, str] = field(default_factory=dict)

    def raised_targets(self) -> frozenset[str]:
        """The targets of the slot the core decided to raise; empty when it waits."""
        if not self.decision.speak or self.decision.slot is None:
            return frozenset()
        slot = next((s for s in self.predictions.slots if s.slot == self.decision.slot), None)
        cites = slot.cites if slot is not None else ()
        return frozenset(self.targets_of[c] for c in cites if c in self.targets_of)

    def attention(self, origin: Sequence[str]) -> float:
        """The highest risk of a slot citing any of ``origin``; 0.0 when none does."""
        wanted = set(origin)
        return max((s.risk for s in self.predictions.slots if wanted & set(s.cites)), default=0.0)


@dataclass(frozen=True)
class CoreController:
    """``controller.decide`` with the core choosing which question to investigate next."""

    view: CoreView

    def __call__(
        self, ws: Workspace, budget: Budget, *, focus: str | None, targets: Sequence[Target], as_of: datetime,
        control: ControlMode = "priority",
    ) -> ControllerDecision:
        base = controller.decide(ws, budget, focus=focus, targets=targets, as_of=as_of, control=control)
        if base.kind != "investigate" or control != "priority":
            return base
        order = {q.id: n for n, q in enumerate(ws.questions)}
        open_ones = [
            p for p in base.priorities
            if p.question_id != focus and p.value >= PRIORITY_FLOOR and not controller.stalled(ws, p.question_id)
        ]

        def rank(question_id: str, value: float) -> tuple[float, float, int]:
            question = ws.question(question_id)
            attended = self.view.attention(question.origin_evidence) if question is not None else 0.0
            return (-attended, -value, order.get(question_id, len(order)))

        best = min(open_ones, key=lambda p: rank(p.question_id, p.value), default=None)
        return base if best is None else replace(base, question_id=best.question_id)


def night_decide(env: Mapping[str, str], view: CoreView) -> Decider:
    """The inquiry's decider for tonight: the controller, or the core's when its switch is on."""
    return decider_for(view if core_on(env) else None)


def decider_for(view: CoreView | None) -> Decider:
    """The decider of a night that dreamt (``view``, its core's view): the core's; else the controller."""
    return CONTROLLER if view is None else Decider(CORE_DECIDER, CoreController(view))


def resume_decider(name: str | None, view: CoreView | None) -> Decider:
    """The decider a killed run resumes under: the one its ``RunStart`` names (``name``). A
    core-decided run runs on only under tonight's view, never under the controller in its place."""
    if name is None:
        return CONTROLLER
    if name != CORE_DECIDER:
        raise ValueError(f"a run decided by the unknown {name!r} can't be resumed")
    if view is None:
        raise ValueError("the run was decided by MIND's core; resuming it needs the core's view (ANJO_SLEEP_MIND_CORE)")
    return decider_for(view)


@dataclass(frozen=True)
class CoreChooser:
    """The inner chooser, with the core deciding whether a held note may surface."""

    inner: Chooser
    provider: Callable[[WantSnapshot], CoreView]

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        view = self.provider(snapshot)
        raised = view.raised_targets()
        return tuple(
            replace(want, weight=0.0) if want.action == NOTE_ACTION and want.target_id not in raised else want
            for want in self.inner.derive(snapshot)
        )

    def choose(self, opportunity: Opportunity, wants: Sequence[Want], ctx: ChoiceContext) -> Choice:
        return self.inner.choose(opportunity, wants, ctx)


def night_chooser(env: Mapping[str, str], inner: Chooser, provider: Callable[[WantSnapshot], CoreView]) -> Chooser:
    """The night's chooser: ``inner`` as today, or the core's when its switch is on."""
    return CoreChooser(inner, provider) if core_on(env) else inner
