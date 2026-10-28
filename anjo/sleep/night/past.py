"""What a person's earlier nights say, and what her month has cost so far.

Her nights' ``summary.json`` records (``nightlog.summaries``) give the rolling allowance's paid
actions, Wonder's month and sleep pressure's history: the last deep sleep, her first night and the
turns since. The budget ledger's dollar settlements give her month-to-date spend. Numbers, dates and
codes only: nothing here reads a clock, a file or the environment.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Final

from anjo.sleep.night.gpu import batch_digest

Summary = Mapping[str, Any]
DATED_FIRST: Final = ("morning", "dailies", "modal_train")  # <kind>:<date>:<person>:...


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


def _dated(raw: str) -> date | None:
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def owner_date(job: str, person: str) -> date | None:
    """The date a settled ledger key is dated on when it is ``person``'s spend; None otherwise.
    A batch's GPU session is hers only when she was alone in it (``gpu.batch_digest``). The simulated
    day (``sim:``) is a synthetic person's living, not Sleep's spend, so it is never hers here."""
    head, _, rest = job.partition(":")
    parts = rest.split(":")
    if head == "night" and len(parts) >= 3:  # night:<step>:<person>:<date>[:...]
        return _dated(parts[2]) if parts[1] == person else None
    if head in DATED_FIRST and len(parts) >= 2:  # <kind>:<date>:<person>[:...]
        return _dated(parts[0]) if parts[1] == person else None
    if head == "night_gpu" and len(parts) >= 2:  # night_gpu:<date>:<digest>[:aN]
        return _dated(parts[0]) if parts[1] == batch_digest([person]) else None
    return None


def month_spent(settlements: Mapping[str, float], person: str, night: date) -> float:
    """``person``'s settled dollars in ``night``'s calendar month, through ``night``."""
    found = [amount for job, amount in settlements.items()
             if (on := owner_date(job, person)) is not None and (on.year, on.month) == (night.year, night.month)
             and on <= night]
    return math.fsum(found)
