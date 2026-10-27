"""The real-time daily schedule for a synthetic person on the dev host (one day, then the night).

    python -m anjo.sleep.night.daily --phase day   --persona founder    # 14:00 Taipei: her day
    python -m anjo.sleep.night.daily --phase night --persona founder    # 02:00 Taipei: night, then morning

She lives in real time from ``LIFE_START``: one simulated day per real day, the night for that day
after midnight, and the morning probe run right after the night, in the same process.

- **day**: plays today's day (or ``--for-date``) through the dev API (``sim_transport.simulate``).
  That wakes Glimmer first and takes the night lock. It then writes a marker under
  ``$ANJO_SLEEP_DATA_DIR/sleep/daily/`` (persona, the date the day was meant for, day index, status
  ``ok``, ``short`` or ``failed``, the failure code, failed turns) and exits 1 unless the day ran
  whole: a server that can't be woken, or one that dies mid-day, is never a quiet short day.
- **night**: runs the night for yesterday's Taipei calendar day, as its own virtual date (the date
  the day was lived on), through the night command's ``execute`` (the same lock, sweep, wake,
  training and upload). It reads that day's marker first: a missing, short or failed day fails the
  night at ``simulate`` (``day_missing``, ``short_day`` or the failure code) and nothing is scored.
  Otherwise the morning probe run scores the checkpoint that is serving.

The person's account id is never in the code: ``ANJO_SLEEP_DAILY_USERS`` maps persona to account,
for example ``founder=<account id>``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Final

from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.night import __main__ as night_command
from anjo.sleep.night.run import Window
from anjo.sleep.probes.morning import run_morning
from anjo.sleep.serving.backend import phase
from anjo.sleep.sim.day import DayRun
from anjo.sleep.types import NightResult

USERS_ENV: Final = "ANJO_SLEEP_DAILY_USERS"
TAIPEI: Final = timezone(timedelta(hours=8))  # Taipei keeps no daylight saving time
LIFE_START: Final = date(2026, 10, 28)
SEED: Final = 20261028
WIRING: Final = "anjo.sleep.app.night:build"
SIMULATE: Final = "anjo.sleep.app.sim_transport:simulate"
MORNING: Final = "anjo.sleep.app.morning:build"
MARKERS: Final = "daily"  # under $ANJO_SLEEP_DATA_DIR/sleep/
DAY_MISSING: Final = "day_missing"
SHORT_DAY: Final = "short_day"


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
                                   "status": status, "failure": run.failure, "failed_turns": run.failed},
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
    return DayRun(plan.window, 0, int(marker["failed_turns"]), (), failure=str(code))


def night_plan(now: datetime) -> NightPlan:
    """The night that closes yesterday (Taipei): its window is that whole calendar day."""
    yesterday = now.astimezone(TAIPEI).date() - timedelta(days=1)
    start = datetime.combine(yesterday, time.min, tzinfo=TAIPEI)
    return NightPlan(yesterday, (start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)))


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
) -> NightResult:
    plan = night_plan(now)
    missed = missed_day(markers, persona, plan)
    with phase():  # one phase: Glimmer stays up from the night into the morning, then sleeps once
        result = nights(wiring=WIRING, night=plan.night, users=[user], personas={user: persona},
                        windows={user: plan.window}, days={} if missed is None else {user: missed})[user]
        if result.status != "failed":  # a failed night keeps yesterday's checkpoint: nothing new to score
            ports = (morning or _morning())(user, persona, plan.night)
            score(user, plan.night, result.checkpoint, morning=ports, label=label, promoted=result.status == "promoted")
    return result


def _morning() -> Callable[..., object]:
    return night_command._named(MORNING)  # type: ignore[no-any-return]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One phase of a synthetic person's real-time day.")
    parser.add_argument("--phase", choices=("day", "night"), required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--for-date", type=date.fromisoformat, help="the day the run is for (default: today, Taipei)")
    parser.add_argument("--label", choices=("development", "final"), default="development")
    args = parser.parse_args(argv)
    user = daily_users().get(args.persona)
    if user is None:
        parser.error(f"{USERS_ENV} has no account for {args.persona}")
    now = datetime.now(timezone.utc)
    if args.phase == "day":
        plan = day_plan(now, args.for_date)
        day = run_day_phase(user, args.persona, now, simulate=night_command._named(SIMULATE), for_date=args.for_date)
        write_marker(marker_dir(), args.persona, plan.day, plan.index, day)
        print(json.dumps({"phase": "day", "day": plan.index, "answered": day.answered,
                          "failed_turns": day.failed, "short_day": day.short, "failure": day.failure,
                          "wake_seconds": day.wake_seconds}))
        return 1 if day.failure is not None or day.failed > 0 or day.short else 0
    result = run_night_phase(user, args.persona, now, markers=marker_dir(), label=args.label)
    print(json.dumps({"phase": "night", "night": night_plan(now).night.isoformat(), "status": result.status,
                      "failed_at": result.failed_at, "log": str(result.log_path)}))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
