"""Kill-resume (Task 17): kill a MIND run's process at a registered operation index, rebuild the
run in a fresh process from its durable log alone, require the digest the live run logged, and
resume.

``kill_resume`` runs three times on one life, apart from the scored runs:

1. here, uninterrupted, on an in-memory log, for the conclusions to compare;
2. in a child process (``python -m anjo.sleep.mind.apd.resume kill``) on a durable log at a fresh
   path, wrapped in ``KillingLog``: right after the event at the kill index is appended, and so
   made durable by the log, the process sends itself SIGKILL. Nothing else survives it;
3. in a fresh child (``resume``) that opens the same file and reads only it: ``kernel.fold``
   gives the workspace, whose ``kernel.digest`` must equal the last logged ``after`` (the killed
   run's digest at the kill); the latest ``RunStart`` names the call to repeat with its night,
   ``as_of``, rules and control; the budget is that ``RunStart``'s plus one cycle per distinct
   ``Step.cycle`` and every ``Step.tokens`` logged since, and the focus is the last ``Step.focus``
   since (``resume_point``). The same loop function is called again, appending to the same file,
   then the life's remaining nights run as in ``arms``.

Both children rebuild everything else from ``setup``, a ``module:function`` path that each process
imports and calls with the life's id (``KillSetup``): the life, the shared models, 17-core's
functions and how to open the durable log at a path (for the pilot, 17-core's JSON Lines log).
Reported: digest equality, then whether the final conclusions agree with the uninterrupted run,
compared by question subject, kind and statement, since ids derive from log seqs and a resume
logs one more ``RunStart``.

``KillSetup.close`` ends what a process's models hold (for the pilot, the open night of Spark
Open): each process calls it when its run ends, which the killed child never does; ``on_killed``
runs in the parent once the kill is confirmed, before the rebuild, to settle what the killed child
left open.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from anjo.sleep.mind.apd.arms import (
    MIND_ARMS,
    MemoryLog,
    MindCore,
    check_clock,
    date_night,
    load_brief,
    mind_night,
    mind_ports,
    night_inputs,
)
from anjo.sleep.mind.apd.types import ArmModels, Life, NightClock
from anjo.sleep.mind.inquiry.ports import InquiryPorts, WorkspaceLog
from anjo.sleep.mind.inquiry.types import Budget, RunStart, Workspace, WorkspaceEvent

Conclusion = tuple[str, str, str]  # (question subject, finding kind, statement)
MODULE: Final = "anjo.sleep.mind.apd.resume"
RAN_OUT: Final = 3  # the kill child's exit code when the run ended before the kill index
CHILD_TIMEOUT: Final = 1800.0  # seconds, per child process
KILL_RUNS: Final = 3  # per life, one after another: uninterrupted, killed, resumed
_SAFE_ID: Final = re.compile(r"^[\w.-]+$")


# ── What each process rebuilds ────────────────────────────────────────────────


@dataclass(frozen=True)
class KillSetup:
    """One life's kill-resume inputs, rebuilt the same in every process: the life, the shared
    models, 17-core's functions, how to open the durable log at a path, the budget and the brief
    (``None``: ``brief_v1.md``)."""

    life: Life
    models: ArmModels
    core: MindCore
    open_log: Callable[[Path], WorkspaceLog]
    budget: Budget = Budget()
    brief: str | None = None
    close: Callable[[], None] = lambda: None

    def ports(self, log: WorkspaceLog) -> InquiryPorts:
        return mind_ports(self.models, self.life, log, load_brief() if self.brief is None else self.brief)


def load_setup(path: str) -> Callable[[str], KillSetup]:
    """The ``module:function`` at ``path``."""
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError("a setup is named module:function")
    found: Callable[[str], KillSetup] = getattr(importlib.import_module(module), name)
    return found


def _sigkill() -> None:
    os.kill(os.getpid(), signal.SIGKILL)


class KillingLog:
    """A durable log that kills its own process once its event at ``kill_at`` is appended."""

    def __init__(self, inner: WorkspaceLog, kill_at: int, kill: Callable[[], None] = _sigkill) -> None:
        self.inner, self.kill_at, self._kill = inner, kill_at, kill

    def append(self, event: WorkspaceEvent) -> None:
        self.inner.append(event)
        if event.step.seq == self.kill_at:
            self._kill()

    def events(self) -> tuple[WorkspaceEvent, ...]:
        return self.inner.events()


# ── Rebuilding from the log ───────────────────────────────────────────────────


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


def conclusions(ws: Workspace) -> tuple[Conclusion, ...]:
    """The workspace's findings, without ids."""
    held = {q.id: q.subject for q in ws.questions}
    return tuple(sorted((held[f.question_id], f.kind, f.statement) for f in ws.findings))


def _nights(life: Life, nights: Sequence[date], ws: Workspace, ports: InquiryPorts, budget: Budget, core: MindCore,
            clock: NightClock | None = None) -> Workspace:
    for night in nights:
        date_night(clock, night)
        ws = mind_night(MIND_ARMS["M"], night_inputs(life, night), ws, ports, budget, core)[-1].ws
    return ws


def _resume(ws: Workspace, point: ResumePoint, life: Life, ports: InquiryPorts, budget: Budget, core: MindCore,
            clock: NightClock | None = None) -> Workspace:
    """Call the killed loop function again, finish its night, then run the nights after it."""
    start = point.start
    date_night(clock, start.night)
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
    return _nights(life, [night for night in life.days if night > start.night], ws, ports, budget, core, clock)


def rebuild(setup: KillSetup, path: Path) -> dict[str, object]:
    """The ``resume`` child's work: fold the log at ``path``, check its digest and, when it holds,
    resume on the same log. The result is what the child prints."""
    log = setup.open_log(path)
    events = log.events()
    try:
        ws = setup.core.fold(events)
    except ValueError:  # fold refuses a log whose digests don't chain: not the logged state
        return {"digest_equal": False, "conclusions": None}
    if not events or setup.core.digest(ws) != events[-1].after:
        return {"digest_equal": False, "conclusions": None}
    final = _resume(ws, resume_point(events), setup.life, setup.ports(log), setup.budget, setup.core,
                    setup.models.clock)
    return {"digest_equal": True, "conclusions": [list(c) for c in conclusions(final)]}


# ── The two children ──────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    """``kill``: run M on the life over a durable log until SIGKILL at ``--kill-at`` (exit
    ``RAN_OUT`` if the run ends first). ``resume``: print ``rebuild``'s result as JSON."""
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}")
    parser.add_argument("mode", choices=("kill", "resume"))
    parser.add_argument("--setup", required=True)
    parser.add_argument("--life", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--kill-at", type=int)
    args = parser.parse_args(argv)
    setup = load_setup(args.setup)(args.life)
    check_clock(setup.life)
    if args.mode == "kill" and args.kill_at is None:
        parser.error("kill needs --kill-at")
    try:
        if args.mode == "resume":
            print(json.dumps(rebuild(setup, args.log)))
            return 0
        log = KillingLog(setup.open_log(args.log), args.kill_at)
        _nights(setup.life, setup.life.days, Workspace(), setup.ports(log), setup.budget, setup.core,
                setup.models.clock)
        return RAN_OUT
    finally:
        setup.close()  # never reached by the SIGKILL


def _child(mode: str, setup: str, life_id: str, path: Path, timeout: float, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", MODULE, mode, "--setup", setup, "--life", life_id, "--log", str(path), *extra],
        stdout=subprocess.PIPE, text=True, timeout=timeout, check=False,
    )


# ── The harness ───────────────────────────────────────────────────────────────


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


def kill_log(workdir: Path, life_id: str, kill_at: int) -> Path:
    """Where ``kill_resume`` keeps the killed run's durable log."""
    if not _SAFE_ID.match(life_id):
        raise ValueError("a life id names a file: letters, digits, '.', '_' and '-' only")
    return workdir / f"{life_id}.kill-{kill_at}.log"


def kill_resume(setup: str, life_id: str, kill_at: int, *, workdir: Path, timeout: float = CHILD_TIMEOUT,
                on_killed: Callable[[], None] = lambda: None) -> KillResume:
    """Kill M's run on ``life_id`` right after operation ``kill_at``, rebuild it in a fresh process
    from its log in ``workdir`` and resume it; compare with an uninterrupted run."""
    path = kill_log(workdir, life_id, kill_at)
    made = load_setup(setup)(life_id)
    check_clock(made.life)
    try:
        whole = _nights(made.life, made.life.days, Workspace(), made.ports(MemoryLog()), made.budget, made.core,
                        made.models.clock)
    finally:
        made.close()
    if path.exists():
        raise ValueError(f"{path.name} already exists: a kill starts on a fresh log")
    killed = _child("kill", setup, life_id, path, timeout, "--kill-at", str(kill_at))
    if killed.returncode == RAN_OUT:
        raise ValueError(f"the run ended before operation {kill_at}")
    if killed.returncode != -signal.SIGKILL:
        raise RuntimeError(f"the run to kill exited with {killed.returncode}, not by SIGKILL")
    on_killed()
    rebuilt = _child("resume", setup, life_id, path, timeout)
    if rebuilt.returncode != 0:
        raise RuntimeError(f"the rebuild exited with {rebuilt.returncode}")
    found = json.loads(rebuilt.stdout)
    uninterrupted = conclusions(whole)
    if not found["digest_equal"]:
        return KillResume(life_id, kill_at, False, None, (), uninterrupted)
    resumed = tuple(sorted((str(s), str(k), str(t)) for s, k, t in found["conclusions"]))
    return KillResume(life_id, kill_at, True, resumed == uninterrupted, resumed, uninterrupted)


if __name__ == "__main__":
    # Run the importable module's main, so the setup's KillSetup and this one are the same class.
    raise SystemExit(importlib.import_module(MODULE).main())
