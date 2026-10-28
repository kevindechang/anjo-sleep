"""One person's night as ``run.py`` runs it: her window, the working state the steps fill in, and the
error text a night log may keep. ``run.py`` re-exports ``Window``, ``day_window`` and ``GPU_STEPS``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Final

from anjo.sleep.mind import receipt
from anjo.sleep.night.deps import Dreamt, InquiryOutcome, StudyOutcome
from anjo.sleep.night.meter import SpendMeter, WonderMeter
from anjo.sleep.night.nightlog import NightLog
from anjo.sleep.night.nightlog import safe_error as safe_error  # the log's own (the story's failures use it); read here too
from anjo.sleep.night.story import EMPTY, KEPT
from anjo.sleep.ports import EvidenceStore, TargetStore
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.types import CheckpointRef, Choice, GateResult, NightStatus, Probe

Window = tuple[datetime, datetime]
GPU_STEPS: Final = ("practice", "dataset", "train", "gates", "promote")


def day_window(night: date) -> Window:
    start = datetime.combine(night, dtime.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


@dataclass
class PersonNight:
    """One person's night as it runs: the working state the steps fill in."""

    uid: str
    night: date
    window: Window
    log: NightLog
    spend: SpendMeter
    wonder: WonderMeter
    actions_left: int
    store: EvidenceStore
    targets: TargetStore
    as_of: datetime  # the night's clock: the virtual night end, which the window's end is in real time
    failed_at: str | None = None
    status: NightStatus | None = None
    capped: bool = False  # a nap at the month's cap: its paid steps wait (``sleep_night.CAPPED_WHY``)
    caught_up: tuple[date, ...] = ()  # owed days tonight's extract settled (``sleep_night.catch_up``)
    due: bool = False
    gated: bool = False
    notice_from: date | None = None  # with a sleep policy: the first day tonight's notice pass reads
    gpu_batch: tuple[str, int] | None = None  # a shared GPU session: its batch digest and how many it held
    sealed: list[Probe] = field(default_factory=list)
    choices: list[Choice] = field(default_factory=list)
    note_choice: Choice | None = None
    study: StudyOutcome | None = None
    inquiry: InquiryOutcome | None = None
    shared: tuple[str, ...] = ()  # results MIND's findings shared at step 10 (Task 17)
    dreamt: Dreamt = field(default_factory=Dreamt)  # MIND's core: tonight's dream (step 2d)
    data: Dataset | None = None
    checkpoint: CheckpointRef | None = None
    gates: dict[str, GateResult] = field(default_factory=dict)
    paid_actions: int = 0
    wake_seconds: float | None = None
    gates_wake_seconds: float | None = None
    wonder_month_spent: float = 0.0
    receipts: dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> datetime:
        """The window's wall-clock end: budget holds and timeouts read it, never a date."""
        return self.window[1]

    @property
    def stamp(self) -> datetime:
        """What the night's records are created at: never after the wall clock, never after its own."""
        return min(self.end, self.as_of)

    @property
    def napped(self) -> bool:
        return self.status == "napped"


def checkpoint_id(ckpt: CheckpointRef | None) -> str | None:
    return None if ckpt is None else ckpt.id


def study_outcome(stopped: str | None) -> receipt.Outcome:
    """A study's or an inquiry's receipt outcome from how it stopped."""
    return "budget" if stopped == "budget" else "failed" if stopped == "failed" else "ran"


def note_error(stage: str, error: BaseException) -> str:
    """A note's failure: its stage and the error's type, never its words."""
    return f"{stage}: {type(error).__name__}"


def kept_brief(kept: date | None, pending: int, error: BaseException) -> dict[str, Any]:
    """The brief step's receipt when the brief broke a rule twice (``context.brief``): the briefs of
    the ``kept`` night stay (on a first night none, so readers serve no brief, as before her first
    fold), tonight's ``pending`` items wait for the next fold, and the night goes on with them.
    ``run.py`` logs the step ``failed``; carrying ``KEPT``, it never reads as where the night stopped."""
    night = EMPTY if kept is None else kept.isoformat()
    why = "brief kept empty: none written yet" if kept is None else f"brief kept from {night}"
    return {"folded": 0, "pending": pending, KEPT: night, "why": why, "error": safe_error(error)}


def kept_summary(receipts: dict[str, Any]) -> dict[str, Any]:
    """The summary's ``brief`` entry: only a kept brief's, which night's it kept and why."""
    brief = receipts.get("brief") or {}
    return {"brief": {key: brief[key] for key in (KEPT, "why", "error")}} if KEPT in brief else {}
