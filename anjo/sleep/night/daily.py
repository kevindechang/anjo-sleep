"""The real-time daily schedule for a synthetic person on the dev host (one day, then the night).

    python -m anjo.sleep.night.daily --phase day   --persona founder    # 14:00 Taipei: her day
    python -m anjo.sleep.night.daily --phase night --persona founder    # 02:00 Taipei: night, then morning

She lives in real time from ``LIFE_START``: one simulated day per real day, the night for that day
after midnight, and the morning probe run right after the night, in the same process.

- **day**: plays today's day through the dev API (``sim_transport.simulate``). That wakes Glimmer
  first and takes the night lock; a server that can't be woken fails the run (exit 1) with
  ``glimmer_not_ready``, and the day does not start.
- **night**: runs the night for yesterday's Taipei calendar day, as its own virtual date (the date
  the day was lived on), through the night command's ``execute`` (the same lock, sweep, wake,
  training and upload). Then the morning probe run scores the checkpoint that is serving.

The person's account id is never in the code: ``ANJO_SLEEP_DAILY_USERS`` maps persona to account,
for example ``founder=<account id>``.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Final

from anjo.sleep.night import __main__ as night_command
from anjo.sleep.night.run import Window
from anjo.sleep.probes.morning import run_morning
from anjo.sleep.sim.day import DayRun
from anjo.sleep.types import NightResult

USERS_ENV: Final = "ANJO_SLEEP_DAILY_USERS"
TAIPEI: Final = timezone(timedelta(hours=8))  # Taipei keeps no daylight saving time
LIFE_START: Final = date(2026, 10, 28)
SEED: Final = 20261028
WIRING: Final = "anjo.sleep.app.night:build"
SIMULATE: Final = "anjo.sleep.app.sim_transport:simulate"
MORNING: Final = "anjo.sleep.app.morning:build"


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


def day_plan(now: datetime) -> DayPlan:
    today = now.astimezone(TAIPEI).date()
    return DayPlan(today, (today - LIFE_START).days)


def night_plan(now: datetime) -> NightPlan:
    """The night that closes yesterday (Taipei): its window is that whole calendar day."""
    yesterday = now.astimezone(TAIPEI).date() - timedelta(days=1)
    start = datetime.combine(yesterday, time.min, tzinfo=TAIPEI)
    return NightPlan(yesterday, (start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)))


def run_day_phase(
    user: str, persona: str, now: datetime, *, simulate: Callable[[str, str, int, int], DayRun],
) -> DayRun:
    plan = day_plan(now)
    if plan.index < 0:
        raise ValueError(f"her life starts on {LIFE_START.isoformat()}")
    return simulate(user, persona, plan.index, SEED)


def run_night_phase(
    user: str, persona: str, now: datetime, *, label: str = "development",
    nights: Callable[..., dict[str, NightResult]] = night_command.execute,
    score: Callable[..., object] = run_morning, morning: Callable[..., object] | None = None,
) -> NightResult:
    plan = night_plan(now)
    result = nights(wiring=WIRING, night=plan.night, users=[user], personas={user: persona},
                    windows={user: plan.window})[user]
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
    parser.add_argument("--label", choices=("development", "final"), default="development")
    args = parser.parse_args(argv)
    user = daily_users().get(args.persona)
    if user is None:
        parser.error(f"{USERS_ENV} has no account for {args.persona}")
    now = datetime.now(timezone.utc)
    if args.phase == "day":
        day = run_day_phase(user, args.persona, now, simulate=night_command._named(SIMULATE))
        print(json.dumps({"phase": "day", "day": day_plan(now).index, "answered": day.answered,
                          "failed_turns": day.failed, "short_day": day.short, "failure": day.failure,
                          "wake_seconds": day.wake_seconds}))
        return 1 if day.failure is not None else 0
    result = run_night_phase(user, args.persona, now, label=args.label)
    print(json.dumps({"phase": "night", "night": night_plan(now).night.isoformat(), "status": result.status,
                      "failed_at": result.failed_at, "log": str(result.log_path)}))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
