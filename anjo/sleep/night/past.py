"""What a person's earlier nights say, and what her month has cost so far.

Her nights' ``summary.json`` records (``nightlog.summaries``) give the rolling allowance's paid
actions, Wonder's month and sleep pressure's history: the last deep sleep, the last night that woke,
her first night, the turns since, and the GPU sessions she shared. The budget ledger's settlements
give her month-to-date spend. Numbers, dates and codes only: nothing here reads a clock, a file or the
environment, except the read-only command line (``main``), which reads the ledger and her nights:

    python -m anjo.sleep.night.past --month <account id> [--for-date 2026-10-28] [--nights <folder>]

It prints her spend this month through that night, by kind, and the room left under the cap.
"""
from __future__ import annotations

import argparse
import calendar
import importlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from anjo.sleep.budget import default_ledger
from anjo.sleep.night.gpu import batch_digest
from anjo.sleep.night.nightlog import summaries
from anjo.sleep.night.sleep_pressure import MEASURED_NIGHTS, MEASURED_REUSED_USD, SleepPolicy, read_policy

Summary = Mapping[str, Any]
# <kind>:<date>:<person>:...; a live night's latent encode settles under ``modal_latents:<date>:<person>`` (the
# research runner's ``modal_latents:<job>`` names no person or date, so it is no one's night)
DATED_FIRST: Final = ("morning", "dailies", "modal_train", "colab_train", "colab_life", "modal_latents")
# Colab's list price, 100 compute units for $9.99, rounded up: a night trained on Colab (the train
# runner ``colab``) settles compute units (``colab_train:``, a life's ``colab_life:``), not dollars.
COLAB_USD_PER_CU: Final = 0.10
GATED: Final = ("promoted", "rejected")  # a deep night that reached the gates: one whose cost prices the next
ATTEMPT: Final = re.compile(r"^(?P<base>.+):a(?P<n>\d+)$")  # ``holds.attempt_key``'s suffix
NIGHTS: Final = "anjo.sleep.app.ownership:night_logs"  # where the sleep host keeps a person's nights
_NONE: Final[Mapping[str, Any]] = MappingProxyType({})


def inside(summary: Summary, end: datetime, span: timedelta) -> bool:
    at = summary.get("night_end")
    if not isinstance(at, str):
        return False
    moment = datetime.fromisoformat(at)
    return end - span < moment <= end


def paid_within(history: Sequence[Summary], end: datetime, span: timedelta) -> int:
    return sum(int(s.get("paid_actions", 0)) for s in history if inside(s, end, span))


def spent_within(history: Sequence[Summary], end: datetime, span: timedelta) -> float:
    return math.fsum(float(s.get("wonder_usd", 0.0)) for s in history if inside(s, end, span))


def _night(summary: Summary) -> date | None:
    try:
        return date.fromisoformat(str(summary["night"]))
    except (KeyError, ValueError):
        return None


def nights_before(history: Sequence[Summary], night: date) -> dict[date, Summary]:
    """Each earlier night's last run (``history`` is in folder order, so a rerun comes after)."""
    return {on: s for s in history if (on := _night(s)) is not None and on < night}


def slept_deep(summary: Summary) -> bool:
    """A deep sleep that didn't fail; a night from before naps were logged went deep if it trained."""
    sleep = summary.get("sleep")
    if isinstance(sleep, Mapping):
        return sleep.get("kind") == "deep" and summary.get("status") != "failed"
    return bool(summary.get("trained"))


def last_deep(history: Sequence[Summary], night: date) -> date | None:
    return max((on for on, s in nights_before(history, night).items() if slept_deep(s)), default=None)


def days_since_deep(history: Sequence[Summary], night: date) -> int | None:
    """From the last deep sleep, else from her first logged night; None on her first night."""
    earlier = nights_before(history, night)
    since = last_deep(history, night) or min(earlier, default=None)
    return None if since is None else (night - since).days


def turns_since(history: Sequence[Summary], since: date | None, night: date) -> int:
    """The turns earlier nights counted after ``since`` (every earlier night's when None)."""
    total = 0
    for on, summary in nights_before(history, night).items():
        sleep = summary.get("sleep")
        turns = sleep.get("turns") if isinstance(sleep, Mapping) else None
        if (since is None or on > since) and isinstance(turns, int) and not isinstance(turns, bool):
            total += turns
    return total


def napped(summary: Summary) -> bool:
    sleep = summary.get("sleep")
    return isinstance(sleep, Mapping) and sleep.get("kind") == "nap"


def notice_from(history: Sequence[Summary], night: date) -> date:
    """The first day tonight's notice pass reads: the day after her last night that woke (a nap puts
    the pass off), else her first logged night when every night napped, else tonight."""
    earlier = nights_before(history, night)
    woke = max((on for on, s in earlier.items() if not napped(s)), default=None)
    return woke + timedelta(days=1) if woke is not None else min(earlier, default=night)


def gpu_batches(history: Sequence[Summary]) -> dict[str, int]:
    """The GPU sessions her nights were in: each batch's digest and how many people it held."""
    found = {}
    for summary in history:
        batch = summary.get("gpu_batch")
        digest, people = (batch.get("digest"), batch.get("people")) if isinstance(batch, Mapping) else (None, None)
        if isinstance(digest, str) and isinstance(people, int) and not isinstance(people, bool) and people >= 1:
            found[digest] = people
    return found


def _dated(raw: str) -> date | None:
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def latest_attempts(settlements: Mapping[str, float]) -> dict[str, float]:
    """Each job's last settled attempt, by its key without the attempt: a night run again counts
    once, at its last run's cost."""
    latest: dict[str, tuple[int, float]] = {}
    for job, amount in settlements.items():
        found = ATTEMPT.match(job)
        base, attempt = (found["base"], int(found["n"])) if found else (job, 0)
        if base not in latest or attempt >= latest[base][0]:
            latest[base] = (attempt, amount)
    return {base: amount for base, (_, amount) in latest.items()}


def her_share(job: str, person: str, batches: Mapping[str, int]) -> tuple[date, float] | None:
    """The date a settled job is dated on and the share of it that is ``person``'s; None when none
    is. A batch's GPU session is split evenly among the people it held (``gpu_batches``; alone, she
    holds all of it). The simulated day (``sim:``) is a synthetic person's living, not Sleep's spend,
    so it is never hers here."""
    head, _, rest = job.partition(":")
    parts = rest.split(":")
    if head == "night" and len(parts) >= 3:  # night:<step>:<person>:<date>[:...]
        on = _dated(parts[2]) if parts[1] == person else None
        return None if on is None else (on, 1.0)
    if head in DATED_FIRST and len(parts) >= 2:  # <kind>:<date>:<person>[:...]
        on = _dated(parts[0]) if parts[1] == person else None
        return None if on is None else (on, 1.0)
    if head == "night_gpu" and len(parts) >= 2:  # night_gpu:<date>:<digest>
        people = 1 if parts[1] == batch_digest([person]) else batches.get(parts[1])
        on = _dated(parts[0]) if people else None
        return None if on is None or not people else (on, 1.0 / people)
    return None


def _dated_spend(
    settlements: Mapping[str, float], person: str, batches: Mapping[str, int], units: Mapping[str, float],
) -> dict[str, tuple[date, float]]:
    """Each settled job of ``person``'s: its date and her dollars, Colab units at ``COLAB_USD_PER_CU``."""
    priced = {**latest_attempts(settlements),
              **{job: cu * COLAB_USD_PER_CU for job, cu in latest_attempts(units).items()}}
    found = {}
    for job, amount in priced.items():
        if (owned := her_share(job, person, batches)) is not None:
            found[job] = (owned[0], amount * owned[1])
    return found


def her_spend(
    settlements: Mapping[str, float], person: str, night: date, *,
    batches: Mapping[str, int] = _NONE, units: Mapping[str, float] = _NONE,
) -> dict[str, float]:
    """``person``'s dollars in ``night``'s calendar month through ``night``, by job: the ledger's
    dollar settlements and its Colab compute units (``units``) at ``COLAB_USD_PER_CU``."""
    return {job: usd for job, (on, usd) in _dated_spend(settlements, person, batches, units).items()
            if (on.year, on.month) == (night.year, night.month) and on <= night}


def measured_deep(
    history: Sequence[Summary], settlements: Mapping[str, float], person: str, night: date,
    settings: Mapping[str, object], *, batches: Mapping[str, int] = _NONE, units: Mapping[str, float] = _NONE,
) -> tuple[float, ...]:
    """What her last ``MEASURED_NIGHTS`` deep nights before ``night`` at ``settings`` (their ``sleep``
    records' ``gpu``, ``core``, ``channel`` and ``practice_fresh``) settled at, newest first: every job
    dated on the night, its morning included, Wonder's studies left out (held back apart). Only a night
    that reached the gates (promoted or rejected) counts: a failed or nothing-new one is no deep night's price."""
    by_night: dict[date, float] = {}
    for job, (dated, usd) in _dated_spend(settlements, person, batches, units).items():
        if not job.startswith("night:homework:"):
            by_night[dated] = by_night.get(dated, 0.0) + usd
    found: list[float] = []
    for summary in sorted(history, key=lambda s: str(s.get("night")), reverse=True):
        on, sleep = _night(summary), summary.get("sleep")
        if on is None or on >= night or not isinstance(sleep, Mapping) or sleep.get("kind") != "deep" \
                or summary.get("status") not in GATED:
            continue
        if all(sleep.get(key) == value for key, value in settings.items()) and by_night.get(on, 0.0) > 0:
            found.append(by_night[on])
        if len(found) == MEASURED_NIGHTS:
            break
    return tuple(found)


def month_spent(
    settlements: Mapping[str, float], person: str, night: date, *,
    batches: Mapping[str, int] = _NONE, units: Mapping[str, float] = _NONE,
) -> float:
    """``person``'s settled dollars in ``night``'s calendar month, through ``night`` (``her_spend``)."""
    return math.fsum(her_spend(settlements, person, night, batches=batches, units=units).values())


def month_report(
    dollars: Mapping[str, float], units: Mapping[str, float], history: Sequence[Summary], person: str,
    night: date, policy: SleepPolicy,
) -> dict[str, Any]:
    """The command line's numbers: her month to date by kind, the cap, and the room left under it."""
    spend = her_spend(dollars, person, night, batches=gpu_batches(history), units=units)
    kinds: dict[str, float] = {}
    for job, amount in spend.items():
        kind = job.partition(":")[0]
        kinds[kind] = kinds.get(kind, 0.0) + amount
    spent = math.fsum(spend.values())
    left = max(0.0, policy.month_usd - spent)
    nights_left = calendar.monthrange(night.year, night.month)[1] - night.day
    return {
        "month": f"{night:%Y-%m}", "through": night.isoformat(), "spent_usd": round(spent, 6),
        "by_kind": {kind: round(amount, 6) for kind, amount in sorted(kinds.items())},
        "cap_usd": policy.month_usd, "left_usd": round(left, 6),
        "deep_night_usd": {"modelled fresh night": policy.deep_usd, "measured, practice reused": MEASURED_REUSED_USD},
        "deep_nights_left": int(max(0.0, left - nights_left * policy.nap_usd) // policy.deep_usd),  # the naps held back
    }


def _nights_of(person: str) -> Path:
    module, _, name = NIGHTS.partition(":")
    found: Path = getattr(importlib.import_module(module), name)(person)  # the app, loaded by name
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A person's Sleep spend this month and the room left under "
                                                 "the cap. Reads the ledger and her nights; writes nothing.")
    parser.add_argument("--month", required=True, metavar="PERSON", help="her account id")
    parser.add_argument("--for-date", type=date.fromisoformat, default=None,
                        help="count through this night (default: today, UTC)")
    parser.add_argument("--nights", type=Path, default=None, help="her nights folder (default: the sleep host's)")
    args = parser.parse_args(argv)
    night: date = args.for_date or datetime.now(timezone.utc).date()
    ledger = default_ledger()
    history = summaries(args.nights or _nights_of(args.month))
    report = month_report(ledger.settlements(), ledger.settlements("cu"), history, args.month, night,
                          read_policy(os.environ))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
