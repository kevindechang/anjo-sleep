"""Sleep pressure: whether tonight is a nap or a deep sleep. One pure decision, made once a night.

MIND and Sleep are one loop (dream, rehearse, learn, check, wake), and the loop has two depths:

- a **nap** runs the night's CPU steps: the sweep, ``close_day``, ``extract``, the chooser, a study,
  the brief. It wakes no Glimmer: no probes, practice, training, gates, promotion or GPU morning, and
  MIND's notice and inquiry and the note wait for her next deep sleep (``sleep_night.py``);
- a **deep sleep** is the full night.

``decide`` goes deep when pressure has built up to the threshold *and* there is enough new material
since the last deep sleep (``min_new_evidence``), or when the days since the last deep sleep reach
``max_days``. Anything else naps. ``force_deep`` (the demo's and the tests' ``--force-deep``) skips
those rules. The month's budget comes last and always wins: the threshold rises from
``base_threshold`` to 1 as the person's settled spend this month nears ``month_usd``, and a deep
sleep that would pass the cap becomes a nap whose reason says so (``month_cap``), never a silent one.
The cap bounds the whole month: tonight fits only with every later night of the calendar month held
back as a nap (``nap_usd``) and Wonder's room left this month held back too. A deep night is priced on
what hers really cost (``deep_estimate``): the highest settled cost of her last ``MEASURED_NIGHTS`` deep
nights at tonight's settings (MIND's core, the thought channel, practice fresh or a rerun's), plus
``MEASURED_MARGIN``. Only before she has one is it the model's (``deep_night_usd``: the serving GPU, the
core, the gates' base cache, practice), which assumes the base cache missed; once nights are measured
that assumption matters only after a revision change, whose one recomputed base side the margin covers.
A nap that doesn't fit is ``capped``: it skips its paid steps (extract, the study and the brief; the
latent encode once a live night runs one), which her next night with room catches up.
``ANJO_SLEEP_PRESSURE=0`` turns all of it off (``policy_from`` gives no policy): every night is deep,
as before naps.

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
from types import MappingProxyType
from typing import Any, Final, Literal, cast

from anjo.sleep.types import Target

Kind = Literal["nap", "deep"]
Reason = Literal["forced", "max_days", "pressure", "below_threshold", "too_little_new", "month_cap"]
KINDS: Final[tuple[Kind, ...]] = ("nap", "deep")
REASONS: Final[tuple[Reason, ...]] = ("forced", "max_days", "pressure", "below_threshold", "too_little_new", "month_cap")

PRESSURE_ENV: Final = "ANJO_SLEEP_PRESSURE"  # "0": off, every night deep; "1" or unset: on
MONTH_USD_ENV: Final = "ANJO_SLEEP_MONTH_USD"
MAX_DAYS_ENV: Final = "ANJO_SLEEP_MAX_DAYS"
CORE_ENV: Final = "ANJO_SLEEP_MIND_CORE"  # mind.core.bridge.CORE_ENV: "1" is MIND's core on (a test keeps them equal)
GPU_ENV: Final = "ANJO_SLEEP_SERVE_GPU"  # serving.config.SERVE_GPU_ENV (a test keeps them equal)
DEFAULT_GPU: Final = "L40S"
DEFAULT_MONTH_USD: Final = 15.0
DEFAULT_MAX_DAYS: Final = 7
BASE_THRESHOLD: Final = 0.5
MIN_NEW_EVIDENCE: Final = 3  # trainable items since the last deep sleep: less is not worth a GPU night
# A deep night at list prices, by its settings (``deep_night_usd``): the cost model's trained night,
# the serving GPU's cold start and idle window included, and its daily morning (``night.cost``, anchored
# on 2026-10-29, the safety gate off), rounded up. The modelled fresh night is MIND's core off, the gates'
# base side asked (the cache missed) and practice run fresh, on each serving GPU; the rest adjust it. A
# test keeps every combination at or above the cost model at every setting it projects.
FRESH_NIGHT_USD: Final[Mapping[str, float]] = MappingProxyType({"L40S": 2.45, "A100-80GB": 2.45, "H100": 2.90})
CACHE_HIT_SAVES_USD: Final[Mapping[str, float]] = MappingProxyType({"L40S": 0.10, "A100-80GB": 0.05, "H100": 0.05})
CORE_ADDS_USD: Final = 0.32  # MIND's core: the joint training and the day's latent encode
PRACTICE_REUSED_SAVES_USD: Final = 0.85  # a rerun's practice, already paid for
MEASURED_REUSED_USD: Final = 1.19  # measured: Teo's night 1 (2026-10-29), practice reused; shown beside the estimate
MEASURED_NIGHTS: Final = 3  # her last deep nights at tonight's settings that price tonight's
MEASURED_MARGIN: Final = 0.15  # on the highest of them: a base side recomputed after a revision change is under it
# A deep night is never priced under this or a nap: the cost model's cheapest deep night at any setting it
# projects (practice reused, every saving on: $0.92 on 2026-10-29's numbers), rounded down. A gated night
# measured under it settled only part of its spend (a test keeps it at or under the model).
DEEP_FLOOR_USD: Final = 0.90
# A nap's own paid steps at list prices: extract and the brief ($0.023 for one day's items on 2026-10-29;
# a busy day's brief folds more), rounded up. Wonder's study is held back apart (``SleepState.wonder_left_usd``);
# the nap's latent encode is planned, not live.
NAP_USD: Final = 0.10
MAX_CATCH_UP_DAYS: Final = 7  # days capped naps left unextracted that one night extracts; the rest stay owed

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
    nights_left: int = 0  # the calendar month's nights after tonight, each held back as a nap
    wonder_left_usd: float = 0.0  # what Wonder's studies may still spend (its rolling cap's room), held back
    practice_fresh: bool = True  # False: a rerun of the night, its practice already paid for
    base_cached: bool = False  # the gates' base side known to be cached (a live night can't know before the wake)
    measured_deep_usd: tuple[float, ...] = ()  # her last deep nights' settled dollars at tonight's settings, newest first
    owed_days: int = 0  # days capped naps left unextracted, each held back like a nap until a night extracts it


@dataclass(frozen=True)
class SleepPolicy:
    month_usd: float = DEFAULT_MONTH_USD
    max_days: int = DEFAULT_MAX_DAYS
    base_threshold: float = BASE_THRESHOLD
    min_new_evidence: int = MIN_NEW_EVIDENCE
    gpu: str = DEFAULT_GPU  # the serving GPU (ANJO_SLEEP_SERVE_GPU)
    core: bool = False  # MIND's core on (ANJO_SLEEP_MIND_CORE=1)
    nap_usd: float = NAP_USD

    def __post_init__(self) -> None:
        if not (math.isfinite(self.month_usd) and self.month_usd > 0):
            raise SleepPolicyError("the month's cap is a positive number of dollars")
        if self.max_days < 1 or self.min_new_evidence < 0:
            raise SleepPolicyError("max_days is at least 1 and min_new_evidence at least 0")
        if not 0.0 <= self.base_threshold <= 1.0 or not (math.isfinite(self.nap_usd) and self.nap_usd >= 0):
            raise SleepPolicyError("the base threshold is from 0 to 1 and a nap's cost at least 0")
        if self.gpu not in FRESH_NIGHT_USD:
            raise SleepPolicyError(f"{GPU_ENV} is one of {', '.join(FRESH_NIGHT_USD)}")

    @property
    def deep_usd(self) -> float:
        """The modelled fresh night at the policy's settings."""
        return deep_night_usd(gpu=self.gpu, core=self.core)

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
    deep_usd: float = 0.0  # tonight's deep night as estimated (``deep_night_usd``)
    capped: bool = False  # a nap with no room for its paid steps: they wait for a night with room

    def record(self) -> dict[str, Any]:
        """The night log's ``sleep`` record: codes and numbers only."""
        return {"kind": self.kind, "reason": self.reason, "pressure": round(self.pressure, 4),
                "threshold": round(self.threshold, 4), "forced": self.forced, "month_usd": self.month_usd,
                **asdict(self.state), "month_spent_usd": round(self.state.month_spent_usd, 6),
                "wonder_left_usd": round(self.state.wonder_left_usd, 6), "deep_usd": self.deep_usd,
                "capped": self.capped}


def deep_night_usd(*, gpu: str = DEFAULT_GPU, core: bool = False, cache_hit: bool = False,
                   practice_fresh: bool = True) -> float:
    """A deep night's estimate at its settings (``FRESH_NIGHT_USD`` and what each setting adds or saves)."""
    if gpu not in FRESH_NIGHT_USD:
        raise SleepPolicyError(f"no deep night estimate for the {gpu}")
    return round(FRESH_NIGHT_USD[gpu] + (CORE_ADDS_USD if core else 0.0) - (CACHE_HIT_SAVES_USD[gpu] if cache_hit else 0.0)
                 - (0.0 if practice_fresh else PRACTICE_REUSED_SAVES_USD), 6)


def deep_estimate(state: SleepState, policy: SleepPolicy) -> float:
    """Tonight's deep night: the highest of her last ``MEASURED_NIGHTS`` measured ones at tonight's
    settings plus ``MEASURED_MARGIN``, or the model's (``deep_night_usd``) before she has one; never
    under ``DEEP_FLOOR_USD`` or a nap."""
    if measured := state.measured_deep_usd[:MEASURED_NIGHTS]:
        found = max(measured) * (1.0 + MEASURED_MARGIN)
    else:
        found = deep_night_usd(gpu=policy.gpu, core=policy.core, cache_hit=state.base_cached,
                               practice_fresh=state.practice_fresh)
    return round(max(found, DEEP_FLOOR_USD, policy.nap_usd), 6)


def fits(state: SleepState, policy: SleepPolicy, tonight_usd: float) -> bool:
    """Tonight's cost fits under the month's cap with every later night of the month and every owed day
    held back as a nap and Wonder's room left held back too, so the month as a whole stays under it."""
    held_back = (max(0, state.nights_left) + max(0, state.owed_days)) * policy.nap_usd + max(0.0, state.wonder_left_usd)
    return state.month_spent_usd + tonight_usd + held_back <= policy.month_usd + 1e-9


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


def decide(state: SleepState, policy: SleepPolicy, *, pressure: float, force_deep: bool = False,
           at_cap: bool = False) -> SleepDecision:
    """Tonight's kind and why. ``pressure`` is ``sleep_pressure(state)`` (or the core's). ``at_cap``: the
    night was found with no room even for a nap before extract, so it is a capped nap whatever it wants."""
    if not (math.isfinite(pressure) and 0.0 <= pressure <= 1.0):
        raise SleepPolicyError("sleep pressure is a number from 0 to 1")
    threshold = policy.threshold(state.month_spent_usd)
    kind, reason = _wanted(state, policy, pressure, threshold, force_deep)
    deep_usd = deep_estimate(state, policy)
    if kind == "deep" and (at_cap or not fits(state, policy, deep_usd)):
        kind, reason = "nap", "month_cap"  # the budget wins, even over --force-deep
    capped = kind == "nap" and (at_cap or not fits(state, policy, policy.nap_usd))
    return SleepDecision(kind, reason, pressure, threshold, force_deep, state, policy.month_usd, deep_usd, capped)


def read_policy(environ: Mapping[str, str]) -> SleepPolicy:
    """The policy ``ANJO_SLEEP_MONTH_USD`` (dollars, default 15) and ``ANJO_SLEEP_MAX_DAYS`` (default 7)
    set, its deep night sized by ``ANJO_SLEEP_SERVE_GPU`` and ``ANJO_SLEEP_MIND_CORE``; a value that
    isn't one stops the night before it starts."""
    raw_usd, raw_days = environ.get(MONTH_USD_ENV, "").strip(), environ.get(MAX_DAYS_ENV, "").strip()
    try:
        month_usd = float(raw_usd) if raw_usd else DEFAULT_MONTH_USD
        max_days = int(raw_days) if raw_days else DEFAULT_MAX_DAYS
    except ValueError:
        raise SleepPolicyError(f"{MONTH_USD_ENV} is dollars and {MAX_DAYS_ENV} a whole number of days") from None
    gpu = environ.get(GPU_ENV, "").strip() or DEFAULT_GPU
    return SleepPolicy(month_usd=month_usd, max_days=max_days, gpu=gpu, core=environ.get(CORE_ENV, "").strip() == "1")


def policy_from(environ: Mapping[str, str]) -> SleepPolicy | None:
    """The night's policy (``read_policy``), or None when ``ANJO_SLEEP_PRESSURE=0`` turns sleep
    pressure off: no nap, every night deep."""
    switch = environ.get(PRESSURE_ENV, "").strip()
    if switch not in ("", "0", "1"):
        raise SleepPolicyError(f"{PRESSURE_ENV} is 0 (off) or 1 (on)")
    return None if switch == "0" else read_policy(environ)


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
