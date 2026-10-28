"""One night's cost by step, and a person-month projected at the night's settings (spec §5.5).

    python -m anjo.sleep.night.cost --from <night folder> [--prices in,out,train,serve]
        [--trained-share S] [--concurrency N]

**The night as logged** reads the folder's ``summary.json`` and ``log.jsonl`` (counts, steps, costs
and seconds; no conversation text) and prints a line a step, the biggest first:

- the summary's cost parts, practice split into the serving GPU's wall time (the step's seconds at
  the serving price) and Spark Open (the rest). The ceiling counts every part but MIND
  (``demo.dailies.CEILING_PARTS``);
- what the summary never holds, outside the ceiling: Wonder's homework (``wonder_usd``) and the
  morning run (its GPU seconds and its judge calls at ``JUDGE_USD_PER_CALL``).

**The month projected** is 30 × (trained share × a trained night + the rest × a quiet night, which
runs extraction and the brief only), at two settings:

- ``BEFORE``, the first live nights (2026-10-28): practice's scenes one at a time, the generation
  gates at the server's 16 with both sides asked every night, Spark Open uncapped, the full morning;
- ``CURRENT``: practice and the generation gates at ``ANJO_SLEEP_CONCURRENCY`` (8 by default), the
  gates' base side kept across nights (``gates.base_cache``), Spark Open capped per role
  (``practice.full.SPARK_CAPS``), the daily morning subset (``probes.morning.DAILY``), and the safety
  gate skipped (``ANJO_SLEEP_SAFETY_GATE`` off for the hackathon: no case is sent; ``--safety-gate on``
  costs it). A night whose log shows the gate skipped says so; it is never costed as passed.

**The month from a pressure trace** (``--pressure-trace FILE``) runs a month of nights through the
nights' own nap-or-deep decision (``night/sleep_pressure.py``): the file is a JSON list, one entry a
day, each the day's new turns, new trainable items and open high-risk targets (``{"turns": 40,
"evidence": 9, "high_risk": 1}``). Turns and items build up from one deep sleep to the next, the
pressure each night reads is the stand-in's, and the month's spend raises the threshold and holds the
cap (``--month-usd``, ``--max-days``). A deep night costs the current setting's trained night and
morning, a nap its quiet night. It prints each night's pressure, threshold, kind and reason.

Extraction, the brief and training are the night's own numbers. What a setting changes is
estimated from ``UNITS``, each with its source: the 2026-10-28 measurements anchor practice (its
serial seconds and Spark dollars) and the gates (their seconds, which scale the gates' estimate).
The caps bound practice's Spark Open at its worst; they don't lower a night that stays under them.
Numbers only: nothing here reads a clock, the environment or any file but the night's two.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.dailies import CEILING_PARTS, CEILING_USD, MONTH_NIGHTS
from anjo.sleep.gates.safety import SKIPPED_REASON
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.sleep_pressure import Kind, Reason, SleepPolicy, SleepPolicyError, SleepState, decide, sleep_pressure
from anjo.sleep.practice.full import SPARK_CAPS
from anjo.sleep.probes.generate import COUNTS
from anjo.sleep.probes.morning import ARM_ORDER, DAILY, JUDGE_USD_PER_CALL, PAIRS, ProbeSubset
from anjo.sleep.serving.config import CONCURRENT_REQUESTS

# The dev host's list prices: Spark Open per million tokens in and out (dev_host.md), the training
# GPU hour (A100, the train receipt's rate), the serving GPU hour (L40S: $0.64 over 1,174 s on 2026-10-28).
DEFAULT_PRICES: Final = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.50,
                               serve_gpu_per_hour=1.96)
SUMMARY_LINES: Final = ("extract", "brief", "practice_spark", "practice_gpu", "training", "gates", "mind")
TRAINED_LINES: Final = ("extract", "brief", "practice_spark", "practice_gpu", "training", "gates")
QUIET_LINES: Final = ("extract", "brief")
JUDGE_NOTE: Final = (f"the morning judge's ${JUDGE_USD_PER_CALL:g} a call (JUDGE_USD_PER_CALL) is a placeholder"
                     " estimate, not yet measured")


class CostError(ValueError):
    """The night's files can't be costed."""


@dataclass(frozen=True)
class Line:
    step: str
    usd: float
    gpu_minutes: float | None
    in_ceiling: bool


@dataclass(frozen=True)
class Settings:
    name: str
    practice_concurrency: int
    gate_concurrency: int  # collapse and safety; the general gate's log-probabilities run at the server's 16
    base_cached: bool
    spark_capped: bool
    morning: ProbeSubset | None  # None: the full set
    safety_gate: bool = True  # False: switched off (ANJO_SLEEP_SAFETY_GATE), no case sent


@dataclass(frozen=True)
class Units:
    """The estimate's inputs, each with its source."""

    stream_tokens_per_s: float = 1024 / 40  # serving.client: a full 1,024-token reply in 40 s at 16 at once
    prefill_call_s: float = 0.45  # estimate: one short prompt's prefill and a token
    general_calls: int = 799  # sets/general: 200 items, one log-probability call a choice
    collapse_prompts: int = 60  # sets/collapse, before the person's fields take some out
    collapse_reply_tokens: int = 400  # estimate
    safety_cases: int = 2015  # the app's crisis corpus, every case each night
    safety_reply_tokens: int = 150  # estimate
    gates_before_s: float = 1200.0  # measured 2026-10-28: the three gates, about 20 L40S minutes
    practice_scenes: int = 16
    practice_serial_s: float = 1174.0  # measured 2026-10-28: 16 scenes one at a time
    practice_spark_usd: float = 0.83  # measured 2026-10-28, uncapped
    practice_spark_in_tokens: int = 150_000  # estimate: F and the scene in each of 33 calls
    training_usd: float = 0.12  # measured 2026-10-28 (2.83 A100 minutes); a night's receipt replaces it
    morning_reply_tokens: int = 400  # estimate


BEFORE: Final = Settings("before", practice_concurrency=1, gate_concurrency=CONCURRENT_REQUESTS,
                         base_cached=False, spark_capped=False, morning=None)
CURRENT: Final = Settings("current", practice_concurrency=8, gate_concurrency=8, base_cached=True,
                          spark_capped=True, morning=DAILY, safety_gate=False)
UNITS: Final = Units()


@dataclass(frozen=True)
class Measured:
    """What the night measured that no setting here changes."""

    extract: float
    brief: float
    training: float | None  # the train receipt's GPU-minutes at the training price


@dataclass(frozen=True)
class Projection:
    settings: Settings
    trained: Mapping[str, float]
    quiet: Mapping[str, float]
    morning: float  # outside the ceiling
    spark_bound: float | None  # practice's Spark Open at its worst under the caps


# ── the night as logged ──────────────────────────────────────────────────────


def read_night(folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The night's summary and its step records."""
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        lines = (folder / "log.jsonl").read_text(encoding="utf-8").splitlines()
        steps = [record for line in lines if line.strip() if isinstance(record := json.loads(line), dict)]
    except (OSError, json.JSONDecodeError) as error:
        raise CostError(f"the night's files can't be read ({type(error).__name__})") from None
    if not isinstance(summary, dict) or not isinstance(summary.get("cost_parts"), dict):
        raise CostError("the summary carries no cost parts (a night from before they were logged)")
    return summary, steps


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _step(steps: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    return next((s for s in reversed(steps) if s.get("step") == name), {})


def _seconds(steps: Sequence[Mapping[str, Any]], *names: str) -> float | None:
    found = [s for name in names if (s := _number(_step(steps, name).get("seconds"))) is not None]
    return math.fsum(found) if found else None


def _train_minutes(steps: Sequence[Mapping[str, Any]]) -> float | None:
    receipt = _step(steps, "train").get("receipt")
    return _number(receipt.get("gpu_minutes")) if isinstance(receipt, Mapping) else None


def night_lines(summary: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], prices: Prices) -> list[Line]:
    """Every line of the night, the biggest first."""
    parts = {name: _number(value) or 0.0 for name, value in summary["cost_parts"].items()}
    serve = prices.serve_gpu_per_hour / 3600.0
    found: list[Line] = []
    for name, usd in parts.items():
        if name == "practice":
            seconds = _seconds(steps, "practice")
            gpu = 0.0 if seconds is None else min(usd, seconds * serve)
            found.append(Line("practice_gpu", gpu, None if seconds is None else seconds / 60.0, True))
            found.append(Line("practice_spark", usd - gpu, None, True))
            continue
        minutes = _train_minutes(steps) if name == "training" else None
        if name == "gates" and (seconds := _seconds(steps, "gates")) is not None:
            minutes = seconds / 60.0
        found.append(Line(name, usd, minutes, name in CEILING_PARTS))
    found.append(Line("homework", _number(summary.get("wonder_usd")) or 0.0, None, False))
    if (seconds := _seconds(steps, "morning", "dailies")) is not None:
        found.append(Line("morning_gpu", seconds * serve, seconds / 60.0, False))
    if (calls := _number(_step(steps, "morning").get("judge_calls"))) is not None:
        found.append(Line("morning_judge", calls * JUDGE_USD_PER_CALL, None, False))
    return sorted(found, key=lambda line: line.usd, reverse=True)


def ceiling_usd(lines: Sequence[Line]) -> float:
    return math.fsum(line.usd for line in lines if line.in_ceiling)


def month(trained: float, quiet: float, *, trained_share: float) -> float:
    """A person-month (§5.5): 30 × (trained share × a trained night + the rest × a quiet one)."""
    return MONTH_NIGHTS * (trained_share * trained + (1.0 - trained_share) * quiet)


# ── the month projected ──────────────────────────────────────────────────────


def measured(summary: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], prices: Prices) -> Measured:
    parts = summary["cost_parts"]
    minutes = _train_minutes(steps)
    return Measured(extract=_number(parts.get("extract")) or 0.0, brief=_number(parts.get("brief")) or 0.0,
                    training=None if minutes is None else minutes / 60.0 * prices.train_gpu_per_hour)


def _rate(concurrency: int, units: Units) -> float:
    """Tokens a second across the server: decoding is bandwidth-bound, so streams add up to its 16."""
    return min(concurrency, CONCURRENT_REQUESTS) * units.stream_tokens_per_s


def _gate_s(settings: Settings, units: Units) -> float:
    """The gates' serving seconds by the units, before the 2026-10-28 measurement scales them."""
    rate, at_once = _rate(settings.gate_concurrency, units), min(settings.gate_concurrency, CONCURRENT_REQUESTS)
    general = units.general_calls * units.prefill_call_s / CONCURRENT_REQUESTS
    collapse = units.collapse_prompts * (units.collapse_reply_tokens / rate + units.prefill_call_s / at_once)
    safety = units.safety_cases * units.safety_reply_tokens / rate if settings.safety_gate else 0.0
    sides = 1 if settings.base_cached else 2  # the checkpoint's side is asked every night, the base's once
    return sides * (general + collapse + safety)


def _morning_calls(subset: ProbeSubset | None) -> tuple[int, int]:
    """Replies and judge calls of one morning: every probe by every arm, each arm against arm 1 and
    the headline pairs both ways (``taste_ab`` by log-probability)."""
    counts = dict(COUNTS) if subset is None else dict(subset.per_kind)
    arms = ARM_ORDER if subset is None else subset.arms
    judged = sum(n for kind, n in counts.items() if kind != "taste_ab")
    comparisons = len(arms) - 1 + sum(1 for a, b in PAIRS.values() if a in arms and b in arms)
    return sum(counts.values()) * len(arms), comparisons * judged * 2


def _spark_bound(units: Units, prices: Prices) -> float:
    calls = {"person": 1, "teacher": units.practice_scenes, "critic": units.practice_scenes}
    out = sum(calls[role] * cap for role, cap in SPARK_CAPS.items())
    return (units.practice_spark_in_tokens * prices.spark_in_per_m + out * prices.spark_out_per_m) / 1e6


def project(settings: Settings, units: Units, prices: Prices, night: Measured) -> Projection:
    serve = prices.serve_gpu_per_hour / 3600.0
    waves = math.ceil(units.practice_scenes / settings.practice_concurrency)
    gates_s = _gate_s(settings, units) * units.gates_before_s / _gate_s(BEFORE, units)
    trained = {
        "extract": night.extract, "brief": night.brief, "practice_spark": units.practice_spark_usd,
        "practice_gpu": units.practice_serial_s * waves / units.practice_scenes * serve,
        "training": units.training_usd if night.training is None else night.training,
        "gates": gates_s * serve,
    }
    replies, judge = _morning_calls(settings.morning)
    morning_s = replies * units.morning_reply_tokens / _rate(settings.practice_concurrency, units)
    return Projection(settings, trained, {k: trained[k] for k in QUIET_LINES},
                      judge * JUDGE_USD_PER_CALL + morning_s * serve,
                      _spark_bound(units, prices) if settings.spark_capped else None)


# ── the month from a pressure trace ──────────────────────────────────────────


@dataclass(frozen=True)
class DayLoad:
    """One day of a trace: the person's new turns, new trainable items and open high-risk targets."""

    turns: int
    evidence: int
    high_risk: int = 0


@dataclass(frozen=True)
class TracedNight:
    kind: Kind
    reason: Reason
    pressure: float
    threshold: float
    usd: float


@dataclass(frozen=True)
class MonthTrace:
    nights: tuple[TracedNight, ...]

    @property
    def deep(self) -> int:
        return sum(night.kind == "deep" for night in self.nights)

    @property
    def naps(self) -> int:
        return len(self.nights) - self.deep

    @property
    def usd(self) -> float:
        return math.fsum(night.usd for night in self.nights)


def read_trace(text: str) -> list[DayLoad]:
    """A trace file's days; anything but a list of whole, non-negative counts is refused."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise CostError("the pressure trace is not JSON") from None
    days: list[DayLoad] = []
    for entry in raw if isinstance(raw, list) else [None]:
        counts = [entry.get(key, 0) for key in ("turns", "evidence", "high_risk")] if isinstance(entry, dict) else [None]
        whole = [n for n in counts if isinstance(n, int) and not isinstance(n, bool) and n >= 0]
        if len(whole) != 3:
            raise CostError("a pressure trace is a list of days, each with whole, non-negative turns, evidence and high_risk")
        days.append(DayLoad(*whole))
    return days


def month_from_trace(trace: Sequence[DayLoad], policy: SleepPolicy, projection: Projection) -> MonthTrace:
    """Each day's night, decided as the nights decide it, at the projection's deep and quiet costs."""
    deep_usd = math.fsum(projection.trained.values()) + projection.morning
    nap_usd = math.fsum(projection.quiet.values())
    spent, turns, evidence, last_deep = 0.0, 0, 0, None
    nights: list[TracedNight] = []
    for i, day in enumerate(trace):
        turns, evidence = turns + day.turns, evidence + day.evidence
        since = None if i == 0 else i - (0 if last_deep is None else last_deep)
        state = SleepState(new_turns=turns, new_evidence=evidence, open_high_risk=day.high_risk,
                           days_since_deep=since, month_spent_usd=spent)
        made = decide(state, policy, pressure=sleep_pressure(state))
        usd = deep_usd if made.kind == "deep" else nap_usd
        nights.append(TracedNight(made.kind, made.reason, made.pressure, made.threshold, usd))
        spent += usd
        if made.kind == "deep":
            turns, evidence, last_deep = 0, 0, i
    return MonthTrace(tuple(nights))


def _trace_report(found: MonthTrace, policy: SleepPolicy, name: str) -> list[str]:
    out = [f"a month from the pressure trace, at {name}", f"  {'night':<7}{'pressure':>9}{'threshold':>10}  {'kind':<6}reason"]
    for n, night in enumerate(found.nights, start=1):
        out.append(f"  {n:<7}{night.pressure:>9.2f}{night.threshold:>10.2f}  {night.kind:<6}{night.reason}")
    out.append(f"  {found.deep} deep, {found.naps} naps: {_usd(found.usd)} of the month's ${policy.month_usd:g} cap"
               f" (deep sleep after at most {policy.max_days} days)")
    return out


# ── the report ───────────────────────────────────────────────────────────────


def _usd(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}"


def _night_report(folder: Path, summary: Mapping[str, Any], lines: Sequence[Line]) -> list[str]:
    out = [f"night {summary.get('night')} ({summary.get('status')}, trained: {summary.get('trained')}) "
           f"from {folder.name}", f"  {'line':<16}{'usd':>10}{'gpu min':>9}  ceiling"]
    for line in lines:
        minutes = "-" if line.gpu_minutes is None else f"{line.gpu_minutes:.1f}"
        out.append(f"  {line.step:<16}{_usd(line.usd):>10}{minutes:>9}  {'yes' if line.in_ceiling else 'no'}")
    ceiling = ceiling_usd(lines)
    out.append(f"  ceiling parts {_usd(ceiling)}; a month of nights like this: {_usd(MONTH_NIGHTS * ceiling)}"
               f" (ceiling ${CEILING_USD:g})")
    if any(line.step == "morning_judge" for line in lines):
        out.append(f"  {JUDGE_NOTE}")
    safety = (summary.get("gates") or {}).get("safety")
    if isinstance(safety, Mapping) and safety.get("skipped"):
        out.append(f"  safety gate: skipped ({SKIPPED_REASON}); the gates line is general and collapse only")
    return out


def _projection_report(found: Sequence[Projection], share: float) -> list[str]:
    out = ["projected month, per step of a trained night", "  " + f"{'':<16}" + "".join(
        f"{p.settings.name:>14}" for p in found)]
    for step in TRAINED_LINES:
        out.append(f"  {step:<16}" + "".join(f"{_usd(p.trained[step]):>14}" for p in found))

    def row(label: str, value: Any) -> str:
        return f"  {label:<16}" + "".join(f"{_usd(value(p)):>14}" for p in found)

    out.append(row("trained night", lambda p: math.fsum(p.trained.values())))
    out.append(row("quiet night", lambda p: math.fsum(p.quiet.values())))
    out.append(row(f"month at {share:.2f}", lambda p: month(math.fsum(p.trained.values()),
                                                         math.fsum(p.quiet.values()), trained_share=share)))
    out.append(row("worst case", lambda p: month(math.fsum(p.trained.values()), 0.0, trained_share=1.0)))
    out.append(row("morning (out)", lambda p: p.morning))
    out.append(row("spark cap bound", lambda p: p.spark_bound))
    out.append(f"  month = {MONTH_NIGHTS} x (trained share x trained + rest x quiet); worst case: every night "
               f"trained; ceiling ${CEILING_USD:g}. The morning run and MIND are outside the ceiling.")
    out.append("  measured: extract, brief, training (this night); practice and gates anchored on 2026-10-28;"
               " the rest estimated (night.cost.UNITS)")
    out.append(f"  morning (out): {JUDGE_NOTE}")
    if off := [p.settings.name for p in found if not p.settings.safety_gate]:
        out.append(f"  safety gate: skipped in {', '.join(off)} ({SKIPPED_REASON}); no case sent, never counted as passed")
    return out


def _prices(raw: str) -> Prices:
    parts = [float(part) for part in raw.split(",")]
    if len(parts) != 4 or not all(math.isfinite(p) and p >= 0 for p in parts):
        raise argparse.ArgumentTypeError("four prices: spark in, spark out per M; train, serve GPU per hour")
    return Prices(*parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One night's cost by step, and a person-month projected.")
    parser.add_argument("--from", dest="folder", type=Path, required=True, help="the night's log folder")
    parser.add_argument("--prices", type=_prices, default=DEFAULT_PRICES, help="in,out,train,serve")
    parser.add_argument("--trained-share", type=float, default=1.0, help="the share of nights that train")
    parser.add_argument("--concurrency", type=int, default=CURRENT.gate_concurrency,
                        help="ANJO_SLEEP_CONCURRENCY for the current projection (8 by default)")
    parser.add_argument("--safety-gate", choices=("on", "off"), default="off",
                        help="ANJO_SLEEP_SAFETY_GATE for the current projection (off for the hackathon)")
    parser.add_argument("--pressure-trace", type=Path, help="a JSON list of days: a month by the nights' nap-or-deep decision")
    parser.add_argument("--month-usd", type=float, default=SleepPolicy().month_usd, help="ANJO_SLEEP_MONTH_USD")
    parser.add_argument("--max-days", type=int, default=SleepPolicy().max_days, help="ANJO_SLEEP_MAX_DAYS")
    args = parser.parse_args(argv)
    if not 0.0 <= args.trained_share <= 1.0 or args.concurrency < 1:
        parser.error("--trained-share is from 0 to 1 and --concurrency at least 1")
    try:
        summary, steps = read_night(args.folder)
        trace = None if args.pressure_trace is None else read_trace(args.pressure_trace.read_text(encoding="utf-8"))
        policy = SleepPolicy(month_usd=args.month_usd, max_days=args.max_days)
    except (CostError, OSError, SleepPolicyError) as error:
        raise SystemExit(f"cost: {error}") from None
    night = measured(summary, steps, args.prices)
    current = replace(CURRENT, name=f"current@{args.concurrency}", practice_concurrency=args.concurrency,
                      gate_concurrency=args.concurrency, safety_gate=args.safety_gate == "on")
    settings = [BEFORE, current]
    if args.concurrency != CONCURRENT_REQUESTS:
        settings.append(replace(current, name=f"current@{CONCURRENT_REQUESTS}",
                                practice_concurrency=CONCURRENT_REQUESTS, gate_concurrency=CONCURRENT_REQUESTS))
    found = [project(s, UNITS, args.prices, night) for s in settings]
    traced = [] if trace is None else ["", *_trace_report(month_from_trace(trace, policy, found[1]), policy, current.name)]
    print("\n".join([*_night_report(args.folder, summary, night_lines(summary, steps, args.prices)), "",
                     *_projection_report(found, args.trained_share), *traced]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
