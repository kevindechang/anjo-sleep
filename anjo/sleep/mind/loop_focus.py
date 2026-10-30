"""MIND → practice: the targets her own questions are working on, at the night's start.

The loop the owner ruled on 2026-10-31 is one cycle: her thinking shapes what Sleep practises,
practice updates her personal model, and her thinking observes the result. This module is the first
arrow. The night reads it only while ``ANJO_SLEEP_MIND_LOOP=1`` (``loop_on``); off, practice draws
its scenes exactly as it does today and nothing here runs.

**Her focus comes from her state.** ``focus_targets`` folds her log to the night's start (the
events before tonight's first ``RunStart``) and ranks the live questions with
``controller.priority``, the controller's own ranking of stake, uncertainty and readiness. The
targets of the questions that clear ``PRIORITY_FLOOR`` are her focus, in that order, at most
``FOCUS_MAX``. So a target she has settled falls below the floor and leaves her focus, and one she
keeps failing on stays: no rule of its own, only the state her log holds.

**The visible evidence** is the person's own items on the target, newest first, as the scene prompt
shows them (``visible``). Practice (``practice/run.py``) draws ``FOCUS_SHARE`` of its scenes from
these targets through ``prompts/scenes_v2.md``; the rest are today's scenes from ``scenes_v1.md``.

Pure: events in, a ``Focus`` out. No clock, file, model or environment read; every time is passed
in on the night's clock, and the store is read by the caller (``night.mind.MindNight.focus``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

from anjo.sleep.mind.inquiry import controller, kernel
from anjo.sleep.mind.inquiry.types import (
    LIVE_STATUSES,
    PRIORITY_FLOOR,
    RunStart,
    WorkspaceEvent,
)
from anjo.sleep.types import Evidence, Target

LOOP_ENV: Final = "ANJO_SLEEP_MIND_LOOP"  # "1": the loop runs; unset or "0": the night is today's
FOCUS_MAX: Final = 2  # targets one night's practice draws its focus scenes from
FOCUS_SHARE: Final = 0.5  # the share of a night's scenes drawn from them
FOCUS_EVIDENCE: Final = 4  # items on a target the scene prompt shows


def loop_on(env: Mapping[str, str]) -> bool:
    """The loop's switch: on only when ``ANJO_SLEEP_MIND_LOOP`` says "1"."""
    return env.get(LOOP_ENV, "").strip() == "1"


@dataclass(frozen=True)
class FocusTarget:
    """One target practice draws a share of its scenes from: the person's target, the question she
    is working it with, its priority, and the visible evidence on it (lines for the scene prompt)."""

    target_id: str
    label: str
    question_id: str
    priority: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Focus:
    """Tonight's focus and, when it is empty, why: the record the night's practice step carries."""

    targets: tuple[FocusTarget, ...] = ()
    why: str | None = None


def to_night_start(events: Sequence[WorkspaceEvent], night: date) -> tuple[WorkspaceEvent, ...]:
    """Her log up to the night's start: every event before tonight's first ``RunStart``. A log with
    none for tonight is taken whole, so the night's focus never depends on tonight's own steps."""
    for index, event in enumerate(events):
        entry = event.step.entry
        if isinstance(entry, RunStart) and entry.night == night:
            return tuple(events[:index])
    return tuple(events)


def focus_targets(
    events: Sequence[WorkspaceEvent], night: date, *, targets: Sequence[Target], as_of: datetime,
) -> Focus:
    """The targets of the live questions that clear ``PRIORITY_FLOOR``, highest priority first, at
    most ``FOCUS_MAX``. Answers from her log folded to the night's start; raises ``ValueError``
    when that log doesn't fold (the caller practices as today and says so)."""
    ws = kernel.fold(to_night_start(events, night))
    labels = {target.id: target.label for target in targets if target.status == "open"}
    ranked = sorted(
        (controller.priority(ws, question.id, targets=targets, as_of=as_of) for question in ws.questions
         if question.status in LIVE_STATUSES),
        key=lambda priority: (-priority.value, priority.question_id),
    )
    chosen: dict[str, FocusTarget] = {}
    for priority in ranked:
        if priority.value < PRIORITY_FLOOR:
            break  # the controller's own floor, first past it nothing is in the running
        question = ws.question(priority.question_id)
        assert question is not None  # priority was just read from it
        for target_id in question.target_ids:
            label = labels.get(target_id)
            if label is None or target_id in chosen:
                continue
            chosen[target_id] = FocusTarget(target_id, label, question.id, priority.value)
            if len(chosen) == FOCUS_MAX:
                return Focus(tuple(chosen.values()))
    if not chosen:
        return Focus((), "no live question clears the priority floor" if ws.questions
                     else "her log holds no question at the night's start")
    return Focus(tuple(chosen.values()))


def visible(items: Sequence[Evidence], target_id: str, *, limit: int = FOCUS_EVIDENCE) -> tuple[str, ...]:
    """The person's own visible items on ``target_id``, newest first, as scene-prompt lines. Held-out
    items and practice scenes never reach her, as they never reach practice's F."""
    own = [
        item for item in items
        if item.target_id == target_id and not item.held_out and item.kind != "practice"
    ]
    own.sort(key=lambda item: (item.created_at, item.id), reverse=True)
    return tuple(f"- ({item.kind}) {' '.join(item.text.split())}" for item in own[:limit])
