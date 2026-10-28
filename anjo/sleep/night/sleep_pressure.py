"""Sleep pressure: whether tonight is a nap or a deep sleep. One pure decision, made once a night.

MIND and Sleep are one loop (dream, rehearse, learn, check, wake), and the loop has two depths:

- a **nap** runs the night's CPU steps: the sweep, ``close_day``, ``extract``, ``notice``, MIND's
  inquiry, the brief and the note. It costs about $0: no probes, practice, training, gates, promotion
  or GPU morning;
- a **deep sleep** is the full night.

``decide`` goes deep when pressure has built up to the threshold *and* there is enough new material
since the last deep sleep (``min_new_evidence``), or when the days since the last deep sleep reach
``max_days``. Anything else naps. ``force_deep`` (the demo's and the tests' ``--force-deep``) skips
those rules. The month's budget comes last and always wins: the threshold rises from
``base_threshold`` to 1 as the person's settled spend this month nears ``month_usd``, and a deep
sleep that would pass the cap becomes a nap whose reason says so (``month_cap``), never a silent one.

``sleep_pressure`` is a stand-in until MIND's core ships its own ``sleep_pressure(state)``, which
drops in through ``NightDeps.sleep_pressure`` with the same signature. It reads what the night
already knows: the person's own turns and the trainable items since the last deep sleep, and the
open targets she is most at risk on (``high_risk``), each over the count that saturates it, then a
weighted mean in [0, 1]. Numbers only: nothing here reads a clock, a file or the environment
(``policy_from`` takes the mapping its caller read).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Final, Literal, cast

from anjo.sleep.types import Target

Kind = Literal["nap", "deep"]
Reason = Literal["forced", "max_days", "pressure", "below_threshold", "too_little_new", "month_cap"]
KINDS: Final[tuple[Kind, ...]] = ("nap", "deep")
REASONS: Final[tuple[Reason, ...]] = ("forced", "max_days", "pressure", "below_threshold", "too_little_new", "month_cap")

MONTH_USD_ENV: Final = "ANJO_SLEEP_MONTH_USD"
MAX_DAYS_ENV: Final = "ANJO_SLEEP_MAX_DAYS"
DEFAULT_MONTH_USD: Final = 15.0
DEFAULT_MAX_DAYS: Final = 7
BASE_THRESHOLD: Final = 0.5
MIN_NEW_EVIDENCE: Final = 3  # trainable items since the last deep sleep: less is not worth a GPU night
# A deep night at list prices: the cost model's current trained night ($1.16) and daily morning
# ($0.47) (``night.cost``), rounded up. A test keeps it at or above the cost model's estimate.
DEEP_NIGHT_USD: Final = 1.65

# The stand-in's saturation counts and weights (estimates). The live night of 2026-10-28 extracted
# 10 items from one simulated day, so about one full day of evidence saturates its term.
TURNS_FULL: Final = 60
EVIDENCE_FULL: Final = 12
RISK_FULL: Final = 3
WEIGHTS: Final = {"turns": 0.25, "evidence": 0.5, "risk": 0.25}
RISK_DUE: Final = timedelta(days=2)  # a commitment due this soon, no note sent yet, is at risk
LOW_CONFIDENCE: Final = 0.4  # a thread she understood less than this is at risk


class SleepPolicyError(ValueError):
    """A sleep setting that can't be used."""


class SleepRecordError(ValueError):
    """A night's ``sleep`` record that isn't one."""


@dataclass(frozen=True)
class SleepState:
    """What tonight knows when it decides. ``days_since_deep`` counts from the last deep sleep, or
    from her first logged night when she has never slept deep; None on her first night."""

    new_turns: int  # the person's own turns since the last deep sleep, tonight's included
    new_evidence: int  # trainable items, practice left out, dated after the last deep sleep
    open_high_risk: int  # open targets she is most at risk on (``high_risk``)
    days_since_deep: int | None
    month_spent_usd: float  # the person's settled dollars this calendar month (``past.month_spent``)


@dataclass(frozen=True)
class SleepPolicy:
    month_usd: float = DEFAULT_MONTH_USD
    max_days: int = DEFAULT_MAX_DAYS
    base_threshold: float = BASE_THRESHOLD
    min_new_evidence: int = MIN_NEW_EVIDENCE
    deep_usd: float = DEEP_NIGHT_USD

    def __post_init__(self) -> None:
        if not (math.isfinite(self.month_usd) and self.month_usd > 0):
            raise SleepPolicyError("the month's cap is a positive number of dollars")
        if self.max_days < 1 or self.min_new_evidence < 0:
            raise SleepPolicyError("max_days is at least 1 and min_new_evidence at least 0")
        if not 0.0 <= self.base_threshold <= 1.0 or not (math.isfinite(self.deep_usd) and self.deep_usd >= 0):
            raise SleepPolicyError("the base threshold is from 0 to 1 and a deep night's cost at least 0")

    def threshold(self, spent: float) -> float:
        """From ``base_threshold`` with nothing spent to 1 at the cap, in a straight line."""
        share = min(1.0, max(0.0, spent) / self.month_usd)
        return self.base_threshold + (1.0 - self.base_threshold) * share


@dataclass(frozen=True)
class SleepDecision:
    kind: Kind
    reason: Reason
    pressure: float
    threshold: float
    forced: bool
    state: SleepState
    month_usd: float

    def record(self) -> dict[str, Any]:
        """The night log's ``sleep`` record: codes and numbers only."""
        return {"kind": self.kind, "reason": self.reason, "pressure": round(self.pressure, 4),
                "threshold": round(self.threshold, 4), "forced": self.forced, "month_usd": self.month_usd,
                **asdict(self.state), "month_spent_usd": round(self.state.month_spent_usd, 6)}


def _share(count: int, full: int) -> float:
    return min(1.0, max(0, count) / full)


def sleep_pressure(state: SleepState) -> float:
    """The stand-in: each signal over the count that saturates it, capped at 1, weighted."""
    return (WEIGHTS["turns"] * _share(state.new_turns, TURNS_FULL)
            + WEIGHTS["evidence"] * _share(state.new_evidence, EVIDENCE_FULL)
            + WEIGHTS["risk"] * _share(state.open_high_risk, RISK_FULL))


def high_risk(targets: Sequence[Target], as_of: datetime) -> int:
    """Open targets she is most at risk on: a commitment due within ``RISK_DUE`` (or past due) that
    no note has met, and a thread she understood less than ``LOW_CONFIDENCE``."""
    count = 0
    for target in targets:
        if target.status != "open":
            continue
        due = target.check_after
        if target.kind == "commitment" and not target.due_noted and due is not None and due <= as_of + RISK_DUE:
            count += 1
        elif target.kind == "thread" and target.confidence is not None and target.confidence < LOW_CONFIDENCE:
            count += 1
    return count


def _wanted(state: SleepState, policy: SleepPolicy, pressure: float, threshold: float,
            force_deep: bool) -> tuple[Kind, Reason]:
    if force_deep:
        return "deep", "forced"
    if state.days_since_deep is not None and state.days_since_deep >= policy.max_days:
        return "deep", "max_days"
    if pressure < threshold:
        return "nap", "below_threshold"
    if state.new_evidence < policy.min_new_evidence:
        return "nap", "too_little_new"
    return "deep", "pressure"


def decide(state: SleepState, policy: SleepPolicy, *, pressure: float, force_deep: bool = False) -> SleepDecision:
    """Tonight's kind and why. ``pressure`` is ``sleep_pressure(state)`` (or the core's)."""
    if not (math.isfinite(pressure) and 0.0 <= pressure <= 1.0):
        raise SleepPolicyError("sleep pressure is a number from 0 to 1")
    threshold = policy.threshold(state.month_spent_usd)
    kind, reason = _wanted(state, policy, pressure, threshold, force_deep)
    if kind == "deep" and state.month_spent_usd + policy.deep_usd > policy.month_usd:
        kind, reason = "nap", "month_cap"  # the budget wins, even over --force-deep
    return SleepDecision(kind, reason, pressure, threshold, force_deep, state, policy.month_usd)


def policy_from(environ: Mapping[str, str]) -> SleepPolicy:
    """The policy ``ANJO_SLEEP_MONTH_USD`` (dollars, default 15) and ``ANJO_SLEEP_MAX_DAYS`` (default 7)
    set; a value that isn't one stops the night before it starts."""
    raw_usd, raw_days = environ.get(MONTH_USD_ENV, "").strip(), environ.get(MAX_DAYS_ENV, "").strip()
    try:
        month_usd = float(raw_usd) if raw_usd else DEFAULT_MONTH_USD
        max_days = int(raw_days) if raw_days else DEFAULT_MAX_DAYS
    except ValueError:
        raise SleepPolicyError(f"{MONTH_USD_ENV} is dollars and {MAX_DAYS_ENV} a whole number of days") from None
    return SleepPolicy(month_usd=month_usd, max_days=max_days)


@dataclass(frozen=True)
class SleepMark:
    """A night's ``sleep`` record as the readers show it (the nights API, the panel, the Dailies card)."""

    kind: Kind
    pressure: float
    threshold: float
    reason: Reason

    def view(self) -> dict[str, Any]:
        return {"kind": self.kind, "pressure": self.pressure, "threshold": self.threshold, "reason": self.reason}


def _unit(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def sleep_mark(raw: object) -> SleepMark | None:
    """A night summary's ``sleep`` record; None for a night without one (from before naps, or run
    without a policy). One that isn't a record this reader knows raises ``SleepRecordError``."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise SleepRecordError("a night's sleep record is not a record")
    kind, reason, pressure, threshold = (raw.get(key) for key in ("kind", "reason", "pressure", "threshold"))
    if kind not in KINDS or reason not in REASONS or not (_unit(pressure) and _unit(threshold)):
        raise SleepRecordError("a night's sleep record holds a kind, a reason, a pressure and a threshold")
    return SleepMark(cast(Kind, kind), float(cast(float, pressure)), float(cast(float, threshold)), cast(Reason, reason))
