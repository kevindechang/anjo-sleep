"""The real-time daily schedule for a synthetic person on the dev host (one day, then the night).

    python -m anjo.sleep.night.daily --phase day                       # 14:00 Taipei: his day
    python -m anjo.sleep.night.daily --phase night                     # 02:00 Taipei: night, then morning
    python -m anjo.sleep.night.daily --phase night --for-date 2026-10-28   # rerun a missed night

He lives in real time from ``LIFE_START``: one simulated day per real day, the night for that day
after midnight, and the morning probe run right after the night, in the same process.

- **day**: plays today's day (or ``--for-date``) through the dev API (``sim_transport.simulate``).
  That wakes Glimmer first and takes the night lock. It then writes a marker under
  ``$ANJO_SLEEP_DATA_DIR/sleep/daily/`` (persona, the date the day was meant for, day index, status
  ``ok``, ``short`` or ``failed``, the failure code, failed turns, and for a day that ended short the
  code of the turn that ended it, ``failure_detail``: a code or an error class name, never message text) and exits 1 unless the day ran
  whole: a server that can't be woken, or one that dies mid-day, is never a quiet short day.
- **night**: runs the night for yesterday's Taipei calendar day (or, with ``--for-date D``, for day D:
  its whole Taipei calendar day is the window and D is the night's label; D may be today but not the
  future, nor before ``LIFE_START``; the night's own clock stays the wall's), as its own virtual date (the date
  the day was lived on), through the night command's ``execute`` (the same lock, sweep, wake,
  training and upload). It reads that day's marker first: a missing, short or failed day fails the
  night at ``simulate`` (``day_missing``, ``short_day`` or the failure code) and nothing is scored.
  Otherwise the morning probe run scores the checkpoint that is serving. After a night that promoted
  new weights, the demo's Dailies line is pre-generated (``app.demo_pairs.after_night``) while
  Glimmer is still awake: the scene answered by the checkpoint before the night and by the new one.
  Its failure is logged in the night's log and never fails the night. Both steps log how long they
  took (``morning`` and ``dailies``), which the Dailies card reads as the morning's GPU time, and the
  morning how many requests it sent again (``retries``), when any. A development night's morning
  runs the committed daily subset (``probes.morning.DAILY``: arms 1 and 3, 14 probes) and logs its
  name, replies and judge calls; ``--label final`` runs the full set.

``--mind`` (or ``ANJO_SLEEP_DAILY_MIND=1``) turns MIND's inquiry on in the night phase; it is off by
default and has no effect on the day phase.

A night that napped (``night/sleep_pressure.py``) trained nothing, so it runs no morning: its log
records the morning ``skipped``, and no Dailies line is made. ``--force-deep`` makes the night a deep
sleep whatever its pressure (the demo); the month's cap (``ANJO_SLEEP_MONTH_USD``) still wins.

The person's account id is never in the code: ``ANJO_SLEEP_DAILY_USERS`` maps persona to account,
for example ``film_founder=<account id>``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from time import monotonic
from typing import Any, Final

from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.night import __main__ as night_command
from anjo.sleep.night.nightlog import NightLog
from anjo.sleep.night.run import Window
from anjo.sleep.night.sleep_night import NAP_WHY
from anjo.sleep.probes.morning import DAILY, MorningResult, run_morning
from anjo.sleep.serving.backend import phase
from anjo.sleep.sim.day import DayRun
from anjo.sleep.types import NightResult

USERS_ENV: Final = "ANJO_SLEEP_DAILY_USERS"
MIND_ENV: Final = "ANJO_SLEEP_DAILY_MIND"  # "1": the scheduled night runs MIND's inquiry, as ``--mind`` does
TAIPEI: Final = timezone(timedelta(hours=8))  # Taipei keeps no daylight saving time
LIFE_START: Final = date(2026, 10, 28)
DEFAULT_PERSONA: Final = "film_founder"  # the demo's hero: the person the real-time nights run for
SEED: Final = 20261028
WIRING: Final = "anjo.sleep.app.night:build"
SIMULATE: Final = "anjo.sleep.app.sim_transport:simulate"
MORNING: Final = "anjo.sleep.app.morning:build"
DAILIES: Final = "anjo.sleep.app.demo_pairs:after_night"
MARKERS: Final = "daily"  # under $ANJO_SLEEP_DATA_DIR/sleep/
DAY_MISSING: Final = "day_missing"
SHORT_DAY: Final = "short_day"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DayPlan:
    day: date
    index: int


@dataclass(frozen=True)
class NightPlan:
    night: date  # the virtual date: the real date the day was lived on
    window: Window  # that Taipei calendar day, in UTC


def daily_users(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``persona=account id`` pairs, comma separated."""
    raw = (os.environ if environ is None else environ).get(USERS_ENV, "")
    pairs = (part.split("=", 1) for part in raw.split(",") if "=" in part)
    return {persona.strip(): user.strip() for persona, user in pairs if persona.strip() and user.strip()}


def day_plan(now: datetime, for_date: date | None = None) -> DayPlan:
    day = for_date or now.astimezone(TAIPEI).date()
    return DayPlan(day, (day - LIFE_START).days)


def marker_dir() -> Path:
    root = os.environ.get(DATA_DIR_ENV, "").strip()
    if not root:
        raise SystemExit(f"{DATA_DIR_ENV} is not set")
    return Path(root) / "sleep" / MARKERS


def _marker_path(folder: Path, persona: str, day: date) -> Path:
    return folder / f"{persona}-{day.isoformat()}.json"


def write_marker(folder: Path, persona: str, day: date, index: int, run: DayRun) -> None:
    """What the day phase tells the night: codes and counts, never a word of the day."""
    status = "failed" if run.failure is not None else "short" if run.short else "ok"
    folder.mkdir(parents=True, exist_ok=True)
    path = _marker_path(folder, persona, day)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(json.dumps({"persona": persona, "date": day.isoformat(), "day_index": index,
                                   "status": status, "failure": run.failure, "failed_turns": run.failed,
                                   **({} if run.failure_detail is None else {"failure_detail": run.failure_detail})},
                                  sort_keys=True), encoding="utf-8")
    os.replace(pending, path)


def missed_day(folder: Path, persona: str, plan: NightPlan) -> DayRun | None:
    """The day the night closes, as a failed ``DayRun`` when it did not run whole; None when it did."""
    path = _marker_path(folder, persona, plan.night)
    if not path.is_file():
        return DayRun(plan.window, 0, 0, (), failure=DAY_MISSING)
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker["status"] == "ok":
        return None
    code = marker["failure"] or SHORT_DAY
    detail = marker.get("failure_detail")  # markers from before the field lack it
    return DayRun(plan.window, 0, int(marker["failed_turns"]), (), failure=str(code),
                  failure_detail=None if detail is None else str(detail))


def night_plan(now: datetime, for_date: date | None = None) -> NightPlan:
    """The night that closes yesterday (Taipei), or ``for_date`` when given: its window is that whole
    Taipei calendar day and the night is labeled with it. A day before her life or after today is refused."""
    today = now.astimezone(TAIPEI).date()
    closing = today - timedelta(days=1) if for_date is None else for_date
    if for_date is not None and for_date > today:
        raise ValueError(f"{for_date.isoformat()} is in the future; today is {today.isoformat()} (Taipei)")
    if for_date is not None and for_date < LIFE_START:
        raise ValueError(f"her life starts on {LIFE_START.isoformat()}")
    start = datetime.combine(closing, time.min, tzinfo=TAIPEI)
    return NightPlan(closing, (start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)))


def run_day_phase(
    user: str, persona: str, now: datetime, *, simulate: Callable[[str, str, int, int], DayRun],
    for_date: date | None = None,
) -> DayRun:
    plan = day_plan(now, for_date)
    if plan.index < 0:
        raise ValueError(f"her life starts on {LIFE_START.isoformat()}")
    return simulate(user, persona, plan.index, SEED)


def run_night_phase(
    user: str, persona: str, now: datetime, *, markers: Path, label: str = "development",
    nights: Callable[..., dict[str, NightResult]] = night_command.execute,
    score: Callable[..., object] = run_morning, morning: Callable[..., object] | None = None,
    dailies: Callable[..., object] | None = None, timer: Callable[[], float] = monotonic,
    for_date: date | None = None, mind: bool = False, force_deep: bool = False,
) -> NightResult:
    plan = night_plan(now, for_date)
    missed = missed_day(markers, persona, plan)
    with phase():  # one phase: Glimmer stays up from the night into the morning, then sleeps once
        result = nights(wiring=WIRING, night=plan.night, users=[user], personas={user: persona},
                        windows={user: plan.window}, days={} if missed is None else {user: missed},
                        as_of={user: plan.window[1]}, mind=mind, force_deep=force_deep)[user]  # real time: the night's clock is the wall's
        if result.status == "napped":  # a nap trained nothing and runs no GPU morning: logged, never silent
            NightLog(result.log_path.parent).step("morning", "skipped", uses="new", system=None, why=NAP_WHY)
        elif result.status != "failed":  # a failed night keeps yesterday's checkpoint: nothing new to score
            log, started = NightLog(result.log_path.parent), timer()
            ports = (morning or _morning())(user, persona, plan.night)
            sent_again = _retries(ports)
            subset = DAILY if label == "development" else None  # the final runs ask every probe and arm
            done = score(user, plan.night, result.checkpoint, morning=ports, label=label,
                         promoted=result.status == "promoted", subset=subset)
            receipt: dict[str, Any] = {"seconds": round(timer() - started, 3)}
            if isinstance(done, MorningResult):  # what the run asked: the cost model's morning lines
                receipt.update(replies=done.replies, judge_calls=done.judge_calls,
                               **({"no_verdict": done.no_verdict} if done.no_verdict else {}))  # unreadable, counted
            if subset is not None:
                receipt["subset"] = subset.name
            if retries := _retries(ports) - sent_again:  # requests the morning sent again: a count, when any
                receipt["retries"] = retries
            log.step("morning", "ok", uses="new", system=None, **receipt)
            if result.status == "promoted":  # new weights: the line he'd now say differently
                _pregenerate(log, dailies, timer, user, persona, plan.night, result)
    return result


def _pregenerate(
    log: NightLog, dailies: Callable[..., object] | None, timer: Callable[[], float],
    user: str, persona: str, night: date, result: NightResult,
) -> None:
    """The Dailies line for ``night``, logged ``ok`` or ``failed`` (by error type) with its time."""
    started, error = timer(), None
    try:  # optional: the demo's line never fails the night or the morning
        (dailies or night_command._named(DAILIES))(user, persona, night, result)
    except Exception as failure:
        error = type(failure).__name__
        logger.warning("dailies line for %s failed: %s", night.isoformat(), error)
    seconds = round(timer() - started, 3)
    if error is None:
        log.step("dailies", "ok", uses="new", system=None, seconds=seconds)
    else:
        log.step("dailies", "failed", uses="new", system=None, seconds=seconds, error=error)


def _retries(ports: object) -> int:
    """The morning's running count of requests sent again (``Morning.retries``), 0 for ports without one."""
    count = getattr(ports, "retries", None)
    return int(count()) if callable(count) else 0


def _morning() -> Callable[..., object]:
    return night_command._named(MORNING)  # type: ignore[no-any-return]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One phase of a synthetic person's real-time day.")
    parser.add_argument("--phase", choices=("day", "night"), required=True)
    parser.add_argument("--persona", default=DEFAULT_PERSONA, help="default: the demo's hero")
    parser.add_argument("--for-date", type=date.fromisoformat,
                        help="day phase: the day to play (default: today, Taipei). night phase: the day the "
                             "night closes, as its window and its label (default: yesterday, Taipei); not "
                             "in the future, not before her life starts")
    parser.add_argument("--mind", action="store_true",
                        help=f"night phase: run MIND's inquiry in the night (also on when {MIND_ENV}=1; default off)")
    parser.add_argument("--force-deep", action="store_true",
                        help="night phase: a deep sleep whatever its pressure (the demo); the month's cap still wins")
    parser.add_argument("--label", choices=("development", "final"), default="development")
    args = parser.parse_args(argv)
    user = daily_users().get(args.persona)
    if user is None:
        parser.error(f"{USERS_ENV} has no account for {args.persona}")
    now = datetime.now(timezone.utc)
    if args.phase == "night":
        try:
            night = night_plan(now, args.for_date)
        except ValueError as refusal:
            parser.error(str(refusal))
    if args.phase == "day":
        plan = day_plan(now, args.for_date)
        day = run_day_phase(user, args.persona, now, simulate=night_command._named(SIMULATE), for_date=args.for_date)
        write_marker(marker_dir(), args.persona, plan.day, plan.index, day)
        print(json.dumps({"phase": "day", "day": plan.index, "answered": day.answered,
                          "failed_turns": day.failed, "short_day": day.short, "failure": day.failure,
                          "wake_seconds": day.wake_seconds}))
        return 1 if day.failure is not None or day.failed > 0 or day.short else 0
    mind = args.mind or os.environ.get(MIND_ENV, "").strip() == "1"
    result = run_night_phase(user, args.persona, now, markers=marker_dir(), label=args.label,
                             for_date=args.for_date, mind=mind, force_deep=args.force_deep)
    print(json.dumps({"phase": "night", "night": night.night.isoformat(), "status": result.status,
                      "mind": mind, "force_deep": args.force_deep, "failed_at": result.failed_at,
                      "log": str(result.log_path)}))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
