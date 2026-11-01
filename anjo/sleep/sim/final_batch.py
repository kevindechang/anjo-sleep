"""A final run's schedule (Task 12b, ``sim.accelerated --label final``): each life scored once, on its
last night, after every life's last night.

The final's lives run as one batch, a pair whose members are each other's 3x donor and a single
life whose donor is the pair's first member (``lives.draw.DONOR_RULE``). A life scored before its
donor's last night would read 3x off an adapter its donor no longer serves, or none at all
(``donor_unavailable``: the life counts as not ahead, ``reports/final.py``). So, with ``--label final``:

- **No morning after an earlier night.** Its log records the morning ``skipped`` (``DEFERRED``) and
  its story is told again. Not even the daily-v1 subset as development: ``reports/final.py`` refuses a
  life whose scores folder holds any score not labelled final, and no night reads a morning.
- **One full morning per life, after the last day's nights** (every life's, so every donor has trained
  its last), in the batch's order, labelled final: the last night's own checkpoint when it promoted,
  else the one serving (its last promoted). A nap or a night that kept nothing is scored too: the
  final test scores every life once. A morning that fails is logged and the next life's still runs.
- **A night that fails stops the run** after that day's nights, before any morning (``FinalStopped``
  in ``sim.accelerated``): no life is scored on a batch whose donors aren't all trained.
- **Resume** (``--resume``, the same command again): a night this final run finished (``finished``:
  stamped final, on the frozen sha, its day index, under the same recipe) is kept, neither simulated
  nor run again; a night that failed after its day ran runs again on that day's logged window
  (``logged_window``), as the daily runner's ``--for-date`` rerun does; any other day is simulated.
  A life whose last night already has a scored morning (``scored``) is never scored again.
- **The recipe** (``RECIPE_ENV``): the switches that decide what a night trains and what its morning
  asks, as this process has them (flags, never a token). Each final night's summary and each final
  morning's log step record them, and a resume under another recipe is refused.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, cast

from anjo.sleep.checkpoints import CheckpointError, ref_from_record
from anjo.sleep.night.nightlog import LOG, SUMMARY, NightLog
from anjo.sleep.types import NightResult, NightStatus

DEFERRED: Final = "final: each life is scored once, on its last night, after every life's last night"
RECIPE: Final = "recipe"  # the key a final night's summary and a final morning's log step record it under
# What decides a night's training and its morning's shape (train.objective, probes.morning, night.sleep_pressure,
# the runner and serving choices, app.morning's judge), by name: flags and choices only, never a token or a key.
RECIPE_ENV: Final = (
    "ANJO_SLEEP_OBJECTIVE", "ANJO_SLEEP_CAPACITY", "ANJO_SLEEP_DAY_DREAMS", "ANJO_SLEEP_DREAMS",
    "ANJO_SLEEP_ADAPTER_FOLD", "ANJO_SLEEP_PROTECT", "ANJO_SLEEP_HARNESS", "ANJO_SLEEP_PRACTICE_ROLES",
    "ANJO_SLEEP_CHANNEL", "ANJO_SLEEP_DREAM_CURRICULUM", "ANJO_SLEEP_AGENT_BASELINE", "ANJO_SLEEP_PRESSURE",
    "ANJO_SLEEP_TRAIN_RUNNER", "ANJO_SLEEP_BACKEND", "ANJO_SLEEP_GLIMMER_REVISION",
    "ANJO_SLEEP_SERVE_DETERMINISTIC", "ANJO_SLEEP_CONCURRENCY", "ANJO_SLEEP_ARMS_EXTRA",
    "ANJO_SLEEP_MORNING_SUBSET", "ANJO_SLEEP_JUDGE",
)
Window = tuple[datetime, datetime]


class ResumeRefused(RuntimeError):
    """A resumed final run found a night it can't place: it stops before any day, night or morning."""


def recipe(environ: Mapping[str, str]) -> dict[str, str | None]:
    """Each recipe switch as ``environ`` has it, stripped; None when unset or blank."""
    return {name: (environ.get(name, "").strip() or None) for name in RECIPE_ENV}


def _attempts(logs: Path, night: date) -> list[Path]:
    """The night's log folders, first attempt first: ``<date>``, ``<date>.2``, ... (``nightlog.night_folder``)."""
    day, found = night.isoformat(), []
    for folder in logs.iterdir() if logs.is_dir() else ():
        suffix = folder.name[len(day) + 1:] if folder.name.startswith(f"{day}.") else ""
        if folder.name == day or suffix.isdigit():
            found.append((1 if folder.name == day else int(suffix), folder))
    return [folder for _, folder in sorted(found)]


def _summary(folder: Path) -> dict[str, Any] | None:
    path = folder / SUMMARY
    if not path.is_file():
        return None
    found: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _result(folder: Path, summary: Mapping[str, Any]) -> NightResult:
    """The finished night as the run needs it: status, checkpoint and log. Its gates and choices are
    left out: the morning and the report read the night's own summary and log, never these."""
    record = summary.get("checkpoint")
    try:
        checkpoint = ref_from_record(record) if isinstance(record, dict) else None
    except CheckpointError as error:
        raise ResumeRefused(f"{folder.name}: the finished night's checkpoint can't be read ({error})") from None
    parts = {str(name): float(usd) for name, usd in (summary.get("cost_parts") or {}).items()}
    return NightResult(cast(NightStatus, summary["status"]), summary.get("failed_at"), checkpoint, {}, (),
                       math.fsum(parts.values()), cast(Any, parts), folder / LOG)


def finished(user_id: str, logs: Path, night: date, day_index: int, sha: str,
             now: Mapping[str, str | None]) -> NightResult | None:
    """The night's finished attempt in this final run, or None. A finished attempt this run didn't
    stamp (another label, other code, another day, never stamped) or one under another recipe than
    ``now`` refuses the resume: which night stands is for the operator to decide, by hand."""
    done = None
    for folder in _attempts(logs, night):
        summary = _summary(folder)
        if summary is None or summary.get("status") == "failed":
            continue
        if not (summary.get("accelerated") is True and summary.get("label") == "final"
                and summary.get("git_sha") == sha and summary.get("day_index") == day_index):
            raise ResumeRefused(f"{user_id}: night {folder.name} finished but isn't this final run's (label, sha "
                                "and day index): check it by hand before resuming")
        stamped = summary.get(RECIPE) or {}
        if changed := sorted(name for name in {*stamped, *now} if stamped.get(name) != now.get(name)):
            raise ResumeRefused(f"{user_id}: night {folder.name} ran under another recipe ({', '.join(changed)}): "
                                "a resumed final run runs under the recipe its nights were stamped with")
        done = _result(folder, summary)
    return done


def logged_window(logs: Path, night: date) -> Window | None:
    """The day window the night's latest attempt logged (``close_day``), from a final attempt or one
    cut short: a failed night runs again on the day that already happened, never on a new one."""
    for folder in reversed(_attempts(logs, night)):
        summary = _summary(folder)
        if summary is not None and summary.get("label") != "final":
            continue
        for step in reversed(NightLog(folder).steps()):
            if step.get("step") == "close_day" and step.get("status") == "ok":
                start, end = (datetime.fromisoformat(str(at)) for at in step["window"])
                return start, end
    return None


def scored(result: NightResult) -> bool:
    """Whether the night's log has a morning that scored (its own, or a backfill's)."""
    return any(step.get("step") == "morning" and step.get("status") == "ok"
               for step in NightLog(result.log_path.parent).steps())


def defer(result: NightResult, night: date) -> None:
    """An earlier night's morning, put off to the end of the batch: logged ``skipped``, its story told again."""
    log = NightLog(result.log_path.parent)
    log.step("morning", "skipped", uses="new", system=None, why=DEFERRED)
    log.tell(night, result.status)
