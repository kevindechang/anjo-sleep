"""Kill-resume (Task 17): kill a MIND run at a registered operation index, rebuild it from its log
alone, require the digest the live run logged, and resume.

The kill is in process: the run's log (``KillingLog``) raises ``Killed`` right after it appends
the event at the registered index, so the loop dies with that event logged and nothing else kept.
The rebuild reads only the log, as the contract says:

- ``kernel.fold`` gives the workspace, and its ``kernel.digest`` must equal the last logged
  ``after`` (the live run's digest at the kill);
- the latest ``RunStart`` names the call to repeat, with its night, ``as_of``, rules and control;
- the budget is that ``RunStart``'s plus one cycle per distinct ``Step.cycle`` and every
  ``Step.tokens`` logged since, and the focus is the last ``Step.focus`` since (``resume_point``).

The same loop function is called again on the same log, then the life's remaining nights run as
in ``arms``. These runs are apart from the scored ones. Reported: digest equality, then whether
the final conclusions agree with an uninterrupted run of the same life. Conclusions are compared
by question subject, kind and statement, since ids derive from log seqs and a resume logs one more
``RunStart``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from anjo.sleep.mind.apd.arms import (
    MIND_ARMS,
    MemoryLog,
    MindCore,
    check_clock,
    load_brief,
    mind_core,
    mind_night,
    mind_ports,
    night_inputs,
)
from anjo.sleep.mind.apd.types import ArmModels, Life
from anjo.sleep.mind.inquiry.ports import InquiryPorts
from anjo.sleep.mind.inquiry.types import Budget, RunStart, Workspace, WorkspaceEvent

Conclusion = tuple[str, str, str]  # (question subject, finding kind, statement)


class Killed(RuntimeError):
    """The registered kill: raised once, right after the event at the kill index is logged."""


class KillingLog(MemoryLog):
    """An in-memory log that kills its run once, after appending the event at ``kill_at``."""

    def __init__(self, kill_at: int) -> None:
        super().__init__()
        self.kill_at = kill_at
        self._fired = False

    def append(self, event: WorkspaceEvent) -> None:
        super().append(event)
        if event.step.seq == self.kill_at and not self._fired:
            self._fired = True
            raise Killed(f"killed after operation {self.kill_at}")


@dataclass(frozen=True)
class ResumePoint:
    """Where a run stood when its log ends: the call to repeat, its budget left and its focus."""

    start: RunStart
    budget: Budget
    focus: str | None


def resume_point(events: Sequence[WorkspaceEvent]) -> ResumePoint:
    """The latest ``RunStart`` in ``events``, the budget it left and the focus it had."""
    index = next((n for n in range(len(events) - 1, -1, -1) if isinstance(events[n].step.entry, RunStart)), None)
    if index is None:
        raise ValueError("a resume needs a RunStart in the log")
    start = events[index].step.entry
    assert isinstance(start, RunStart)
    since = [event.step for event in events[index + 1:]]
    cycles = len({step.cycle for step in since if step.cycle is not None})
    focus = next((step.focus for step in reversed(since) if step.focus is not None), start.focus)
    return ResumePoint(start, start.budget.spend(cycles=cycles, tokens=sum(step.tokens for step in since)), focus)


@dataclass(frozen=True)
class KillResume:
    """One kill-resume run: the digest check, then the conclusions against an uninterrupted run
    (``None`` when the digest differed and nothing resumed)."""

    life_id: str
    kill_at: int
    digest_equal: bool
    conclusions_agree: bool | None
    resumed: tuple[Conclusion, ...]
    uninterrupted: tuple[Conclusion, ...]


def conclusions(ws: Workspace) -> tuple[Conclusion, ...]:
    """The workspace's findings, without ids."""
    held = {q.id: q.subject for q in ws.questions}
    return tuple(sorted((held[f.question_id], f.kind, f.statement) for f in ws.findings))


def _nights(life: Life, nights: Sequence[date], ws: Workspace, ports: InquiryPorts, budget: Budget, core: MindCore) -> Workspace:
    for night in nights:
        ws = mind_night(MIND_ARMS["M"], night_inputs(life, night), ws, ports, budget, core)[-1].ws
    return ws


def _resume(ws: Workspace, point: ResumePoint, life: Life, ports: InquiryPorts, budget: Budget, core: MindCore) -> Workspace:
    """Call the killed loop function again, finish its night, then run the nights after it."""
    start = point.start
    inputs = night_inputs(life, start.night)
    if start.call == "notice":
        noticed = core.notice(ws, inputs.new, ports, point.budget, night=start.night, targets=inputs.targets,
                              as_of=start.as_of)
        ws, left, focus = noticed.ws, noticed.budget, None
        rules, control = MIND_ARMS["M"].rules, MIND_ARMS["M"].control
    else:
        left, focus, rules, control = point.budget, point.focus, start.rules, start.control
    ws = core.run_inquiry(ws, ports, left, night=start.night, targets=inputs.targets, as_of=start.as_of, focus=focus,
                          rules=rules, control=control).ws
    return _nights(life, [night for night in life.days if night > start.night], ws, ports, budget, core)


def kill_resume(
    life: Life, models: ArmModels, kill_at: int, *, budget: Budget = Budget(), core: MindCore | None = None,
    brief: str | None = None,
) -> KillResume:
    """Run M on ``life`` uninterrupted, then again killed after operation ``kill_at``; rebuild the
    killed run from its log, check its digest and resume it."""
    check_clock(life)
    core = core or mind_core()
    text = load_brief() if brief is None else brief
    whole = _nights(life, life.days, Workspace(), mind_ports(models, life, MemoryLog(), text), budget, core)
    log = KillingLog(kill_at)
    ports = mind_ports(models, life, log, text)
    try:
        _nights(life, life.days, Workspace(), ports, budget, core)
    except Killed:
        pass  # the registered kill: everything the run knew is in its log
    else:
        raise ValueError(f"the run ended before operation {kill_at}")
    events = log.events()
    ws = core.fold(events)
    if core.digest(ws) != events[-1].after:
        return KillResume(life.id, kill_at, False, None, (), conclusions(whole))
    final = _resume(ws, resume_point(events), life, ports, budget, core)
    return KillResume(life.id, kill_at, True, conclusions(final) == conclusions(whole), conclusions(final),
                      conclusions(whole))
