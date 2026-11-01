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
  in ``sim.accelerated``): no life is scored on a batch whose donors aren't all trained. That day's other
  nights, the last day's too, log their morning ``skipped`` (``DEFERRED``), so no backfill scores them.
- **Resume** (``--resume``, the same command again): a night this final run finished (``finished``:
  stamped final, on the frozen sha, its day index, under the same recipe) is kept, neither simulated
  nor run again; a night that failed after its day ran runs again on that day's logged window
  (``logged_window``), as the daily runner's ``--for-date`` rerun does; any other day is simulated.
  A life whose last night already has a scored morning (``scored``) is never scored again. A resume
  repeats the command its nights ran as (``RUN``: ``run_shape``), adding only ``--resume``: one under
  another ``--days``, ``--start``, life list or ``--practice-scenes`` is refused before any day.
- **One final score per life** (``check_scores``): a final run, fresh or resumed, refuses a life whose
  scores folder already holds a final score it can't place (any, on a fresh run; one for a night but this
  run's last, on a resume), before any day: ``reports/final.py`` reads only a life's newest score.
  A fresh run also refuses a life that already has a final night (``check_fresh``): the same command
  with ``--resume`` continues it, where a fresh run would train its lives' nights again.
- **The recipe** (``RECIPE_ENV``): the switches that decide what a night trains and what its morning
  asks, as this process has them (flags, never a token). Each final night's summary and each final
  morning's log step record them, and a resume under another recipe is refused.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, cast

from anjo.sleep.checkpoints import CheckpointError, ref_from_record
from anjo.sleep.night.nightlog import LOG, SUMMARY, NightLog
from anjo.sleep.types import NightResult, NightStatus

DEFERRED: Final = "final: each life is scored once, on its last night, after every life's last night"
RECIPE: Final = "recipe"  # the key a final night's summary and a final morning's log step record it under
RUN: Final = "final_run"  # the key a final night's summary records the command it ran as under (``run_shape``)
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
    """A resumed final run found a night it can't place, or a final run a score it can't: it stops before
    any day, night or morning."""


def recipe(environ: Mapping[str, str]) -> dict[str, str | None]:
    """Each recipe switch as ``environ`` has it, stripped; None when unset or blank."""
    return {name: (environ.get(name, "").strip() or None) for name in RECIPE_ENV}


def run_shape(days: int, start: date, lives: Sequence[tuple[str, str]], seed: int, alone_from: int | None,
              extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The command a final run ran as: ``--days``, ``--start``, each ``--user`` with its persona, the seed,
    ``--alone-from`` and ``extra`` (the command line's own, ``--practice-scenes`` among them), shaped as a
    summary reads it back, so a night's record and a resume's compare equal."""
    return {"days": days, "start": start.isoformat(), "lives": [[uid, person] for uid, person in lives],
            "seed": seed, "alone_from": alone_from, **dict(extra or {})}


def _changed(stamped: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    return sorted(name for name in {*stamped, *now} if stamped.get(name) != now.get(name))


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
             now: Mapping[str, str | None], run: Mapping[str, Any]) -> NightResult | None:
    """The night's finished attempt in this final run, or None. A finished attempt this run didn't
    stamp (another label, other code, another day, never stamped), one under another recipe than
    ``now`` or one another command ran (``run``) refuses the resume: which night stands is for the
    operator to decide, by hand."""
    done = None
    for folder in _attempts(logs, night):
        summary = _summary(folder)
        if summary is None or summary.get("status") == "failed":
            continue
        if not (summary.get("accelerated") is True and summary.get("label") == "final"
                and summary.get("git_sha") == sha and summary.get("day_index") == day_index):
            raise ResumeRefused(f"{user_id}: night {folder.name} finished but isn't this final run's (label, sha "
                                "and day index): check it by hand before resuming")
        if changed := _changed(summary.get(RECIPE) or {}, now):
            raise ResumeRefused(f"{user_id}: night {folder.name} ran under another recipe ({', '.join(changed)}): "
                                "a resumed final run runs under the recipe its nights were stamped with")
        if changed := _changed(summary.get(RUN) or {}, run):
            raise ResumeRefused(f"{user_id}: night {folder.name} ran as another command ({', '.join(changed)}): "
                                "a resume repeats the command its nights ran as, adding only --resume")
        done = _result(folder, summary)
    return done


def last_night_ran(logs: Path, night: date, run: Mapping[str, Any]) -> bool:
    """Whether a life's ``night`` finished (promoted, kept nothing or napped; never failed) in the final run
    ``run`` names: a donor whose last night ran serves its last adapter, the one its 3x reads."""
    for folder in _attempts(logs, night):
        summary = _summary(folder)
        if (summary is not None and summary.get("status") not in (None, "failed") and summary.get("label") == "final"
                and summary.get(RUN) == run):
            return True
    return False


def check_scores(user_id: str, folder: Path, last: date, *, resume: bool) -> None:
    """Refuse a life whose scores folder (``probes.morning``'s ``night-NN.json``) holds a final score this
    run can't place: any, on a fresh run; one for a night but ``last``, on a resume (a changed command).
    A life is scored once, and ``reports/final.py`` reads only its newest score."""
    for path in sorted(folder.glob("night-*.json")) if folder.is_dir() else ():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("label") == "final" and not (resume and record.get("night") == last.isoformat()):
            raise ResumeRefused(f"{user_id} already has a final score for {record.get('night')} ({path.name}): each "
                                "life is scored once, so resume the run that scored it, as the command it ran as")


def check_fresh(user_id: str, logs: Path) -> None:
    """Refuse a fresh final run on a life whose night logs already hold a final night (stamped ``final``,
    finished or failed): run without ``--resume``, it would train every night again and score the result."""
    for folder in sorted(logs.iterdir()) if logs.is_dir() else ():
        summary = _summary(folder) if folder.is_dir() else None
        if summary is not None and summary.get("label") == "final":
            raise ResumeRefused(f"{user_id} already has a final night ({folder.name}): continue that run with the "
                                "same command and --resume; a fresh final run would train its nights again")


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
