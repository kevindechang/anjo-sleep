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
runs extraction and the brief, and with MIND its inquiry and the day's latent encode), at two settings:

- ``BEFORE``, the first live nights (2026-10-28): practice's scenes one at a time, the generation
  gates at the server's 16 with both sides asked every night, Spark Open uncapped, the full morning;
- ``CURRENT``: practice and the generation gates at ``ANJO_SLEEP_CONCURRENCY`` (8 by default), the
  gates' base side kept across nights (``gates.base_cache``), the safety gate off
  (``ANJO_SLEEP_SAFETY_GATE``, off for the hackathon; the floor still runs), Spark Open capped per role
  (``practice.full.SPARK_CAPS``), and the daily morning subset (``probes.morning.DAILY``).

``--scenarios`` adds the whole night loop under sleep pressure (``SCENARIOS``). Every night naps (on CPU:
MIND's consolidation, dreams, inquiry and note, with extraction and the brief) and reads the day's
events through Glimmer (``latent_encode``, the nap's one GPU cost). A deep sleep (practice, training,
gates and the morning, on GPU) comes only when MIND's pressure crosses its threshold, and at least
every 7th night. A month is a pressure trace (``Trace``): an active person deep-sleeps 2 to 3 times a
week (``ACTIVE``), a quiet one about once (``QUIET``); ``--trace DNNDN...`` costs a real one. The rows:
the 2026-10-28.4 night as measured, the base cache, the safety gate off, the one system (MIND's latent
encode and the joint training beside the LoRA), then each cost proposal on it alone and all of them
together, with what turning the safety gate on adds. Every deep night also pays the serving GPU's
``serve_overhead``, which the summary never logs: the cold start and Modal's idle window after the last
request (shared by every person a wake serves, ``Settings.batch``), and its wait while the dataset is
built and training runs.

Extraction, the brief, MIND's inquiry and training are the night's own numbers. What a setting
changes is estimated from ``UNITS``, each with its source: the 2026-10-28 measurements anchor
practice (its serial seconds and Spark dollars) and the gates (the 77 L40S minutes of the night
before the base cache, which scale the gates' estimate). The caps bound practice's Spark Open at its
worst; they don't lower a night that stays under them. Numbers only: nothing here reads a clock, the
environment or any file but the night's two.
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
from anjo.sleep.night.deps import Prices
from anjo.sleep.practice.critics import GEMINI_IN_PER_M, GEMINI_OUT_PER_M
from anjo.sleep.practice.full import SPARK_CAPS
from anjo.sleep.practice.roles import Roles
from anjo.sleep.probes.generate import COUNTS
from anjo.sleep.probes.morning import ARM_ORDER, DAILY, JUDGE_USD_PER_CALL, PAIRS, ProbeSubset
from anjo.sleep.serving import gpus
from anjo.sleep.serving.config import CONCURRENT_REQUESTS, DEFAULT_MAX_INPUTS, SCALEDOWN_S

# The dev host's list prices: Spark Open per million tokens in and out (dev_host.md), the training
# GPU hour (A100, the train receipt's rate), the serving GPU hour (L40S: $0.64 over 1,174 s on 2026-10-28).
DEFAULT_PRICES: Final = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.50,
                               serve_gpu_per_hour=1.96)
SUMMARY_LINES: Final = ("extract", "brief", "practice_spark", "practice_gpu", "training", "gates", "mind")
TRAINED_LINES: Final = ("extract", "brief", "mind", "practice_spark", "practice_critic", "practice_gpu", "training",
                        "latent_encode", "gates", "serve_overhead")
QUIET_LINES: Final = ("extract", "brief", "mind", "latent_encode")
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
    safety: bool = False  # the safety gate's 2,015 cases (ANJO_SLEEP_SAFETY_GATE), off for the hackathon
    gpu: str = "L40S"  # ANJO_SLEEP_SERVE_GPU
    max_inputs: int = DEFAULT_MAX_INPUTS  # ANJO_SLEEP_SERVE_MAX_INPUTS: requests a container takes at once
    roles: Roles = Roles()  # ANJO_SLEEP_PRACTICE_ROLES
    batch: int = 1  # people one wake serves (run_night_batch): the cold start and the idle window shared
    idle_tail: bool = True  # Modal bills its idle window after the last request (False: retired at the end)
    one_system: bool = False  # MIND's nightly lines: the day's latent encode and the joint training
    encode_in_training: bool = False  # a deep night's encode in the training container, which loads Glimmer anyway
    encode_people: int = 1  # people whose day goes up in one nap encode call (latents encode takes many lives)
    encode_snapshot: bool = False  # Modal's GPU memory snapshot restores the encoder instead of loading it
    encode_deferred: bool = False  # naps read the last encode; the next deep wake encodes the days since


@dataclass(frozen=True)
class Units:
    """The estimate's inputs, each with its source."""

    prefill_call_s: float = 0.45  # estimate: one short prompt's prefill and a token, on the L40S
    general_calls: int = 799  # sets/general: 200 items, one log-probability call a choice
    collapse_prompts: int = 60  # sets/collapse, before the person's fields take some out
    collapse_reply_tokens: int = 400  # estimate
    safety_cases: int = 2015  # the app's crisis corpus, every case each night
    safety_reply_tokens: int = 150  # estimate
    gates_anchor_s: float = 4620.0  # measured 2026-10-28.4: 77 L40S minutes ($2.50 at $1.95/h), at ANCHOR
    practice_scenes: int = 16
    practice_serial_s: float = 1174.0  # measured 2026-10-28: 16 scenes one at a time
    practice_spark_usd: float = 0.83  # measured 2026-10-28, uncapped
    practice_spark_in_tokens: int = 150_000  # estimate: F and the scene in each of 33 calls
    training_usd: float = 0.12  # measured 2026-10-28 (2.83 A100 minutes); a night's receipt replaces it
    morning_reply_tokens: int = 400  # estimate
    wake_s: float = 180.0  # estimate: the serving app's cold start (weights off the volume, vLLM up)
    train_idle_s: float = 230.0  # estimate: the serving GPU awake while the dataset is built and training runs
    # estimate: the measured $0.83 by role (one scenes call, 16 teacher and 16 critic calls of about the
    # same size); a night's log carries the real split (``spark`` by role)
    spark_share: tuple[tuple[str, float], ...] = (("person", 0.04), ("teacher", 0.48), ("critic", 0.48))
    person_reply_tokens: int = 1_500  # estimate: twenty scenes as JSON, when Glimmer writes them
    critic_in_tokens: int = 6_000  # estimate: F, the message and four candidates
    critic_out_tokens: int = 700  # estimate: Gemini's low thinking and the JSON
    all_miss_share: float = 0.25  # estimate: scenes whose three Glimmer candidates all miss (the A/B measures it)
    encode_load_s: float = 370.0  # estimate (mind.core.latents.LOAD_S): an A100 start, 56 GB of bf16, 10 s idle
    encode_s: float = 33.0  # estimate: a day's ~120k tokens to layer 35, at latents' 2,500 tokens/s full depth
    snapshot_load_s: float = 45.0  # estimate, unmeasured: a GPU memory snapshot's restore instead of the load
    joint_training_share: float = 0.3  # estimate: the projector, the reader and the core beside the LoRA


BEFORE: Final = Settings("before", practice_concurrency=1, gate_concurrency=CONCURRENT_REQUESTS,
                         base_cached=False, spark_capped=False, morning=None)
CURRENT: Final = Settings("current", practice_concurrency=8, gate_concurrency=8, base_cached=True,
                          spark_capped=True, morning=DAILY)
UNITS: Final = Units()
# The night the gates are anchored on, and the loop's settings from it to every proposal at once.
ANCHOR: Final = Settings("2026-10-28.4", practice_concurrency=8, gate_concurrency=8, base_cached=False,
                         spark_capped=True, morning=DAILY, safety=True)
ONE_SYSTEM: Final = replace(CURRENT, name="one system", one_system=True)
ALL: Final = replace(ONE_SYSTEM, name="all", roles=Roles("glimmer", "gemini", "lazy"), batch=5, idle_tail=False,
                     encode_in_training=True, encode_people=5)
SCENARIOS: Final[tuple[Settings, ...]] = (
    ANCHOR, replace(ANCHOR, name="+ base cache", base_cached=True), replace(CURRENT, name="+ safety off"),
    ONE_SYSTEM,
    replace(ONE_SYSTEM, name="critic=gemini", roles=Roles(critic="gemini")),
    replace(ONE_SYSTEM, name="person=glimmer", roles=Roles(person="glimmer")),
    replace(ONE_SYSTEM, name="teacher=lazy", roles=Roles(teacher="lazy")),
    replace(ONE_SYSTEM, name="H100 at 32", gpu="H100", max_inputs=32, gate_concurrency=32, practice_concurrency=16),
    replace(ONE_SYSTEM, name="batch of 5", batch=5),
    replace(ONE_SYSTEM, name="retire at end", idle_tail=False),
    replace(ONE_SYSTEM, name="encode in train", encode_in_training=True),
    replace(ONE_SYSTEM, name="encode 5 at once", encode_people=5),
    replace(ONE_SYSTEM, name="encode snapshot", encode_snapshot=True),
    replace(ONE_SYSTEM, name="encode deferred", encode_deferred=True, encode_in_training=True),
    ALL,
)


@dataclass(frozen=True)
class Trace:
    """A month of nights as sleep pressure schedules them: ``D`` a deep sleep, ``N`` a nap. A trace
    never naps 7 nights running (the scheduler's 7-day cap); it repeats to fill a month."""

    name: str
    nights: str

    def __post_init__(self) -> None:
        if not self.nights or set(self.nights) - {"D", "N"} or "N" * 7 in self.nights + self.nights:
            raise ValueError("a trace is D (deep) and N (nap) nights, with a deep sleep at least every 7th night")

    @property
    def deep_share(self) -> float:
        return self.nights.count("D") / len(self.nights)


ACTIVE: Final = Trace("active", "DNNDNDN" + "NDNNDNN")  # 5 deep in 14 nights: 2.5 a week
QUIET: Final = Trace("quiet", "DNNNNNN")  # one a week, the cap


@dataclass(frozen=True)
class Measured:
    """What the night measured that no setting here changes."""

    extract: float
    brief: float
    training: float | None  # the train receipt's GPU-minutes at the training price
    mind: float = 0.0  # MIND's inquiry, as the night logged it


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


def trace_month(found: Projection, trace: Trace) -> float:
    """A person-month on a pressure trace: its deep nights and its naps."""
    return month(math.fsum(found.trained.values()) + found.morning, math.fsum(found.quiet.values()),
                 trained_share=trace.deep_share)


# ── the month projected ──────────────────────────────────────────────────────


def measured(summary: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], prices: Prices) -> Measured:
    parts = summary["cost_parts"]
    minutes = _train_minutes(steps)
    return Measured(extract=_number(parts.get("extract")) or 0.0, brief=_number(parts.get("brief")) or 0.0,
                    training=None if minutes is None else minutes / 60.0 * prices.train_gpu_per_hour,
                    mind=_number(parts.get("mind")) or 0.0)


def _at_once(concurrency: int, settings: Settings) -> int:
    gpu = gpus.GPUS[settings.gpu]
    return max(1, min(concurrency, settings.max_inputs, gpu.fits(gpus.GATE_REQUEST_TOKENS)))


def _rate(concurrency: int, settings: Settings) -> float:
    """Tokens a second across the server: decoding is bandwidth-bound, so streams add up to what it takes."""
    return _at_once(concurrency, settings) * gpus.GPUS[settings.gpu].stream_tokens_s


def _serve_usd_s(settings: Settings, prices: Prices) -> float:
    """The serving GPU's dollars a second: the prices' L40S hour, scaled by the GPU's list price."""
    gpu = gpus.GPUS[settings.gpu]
    return prices.serve_gpu_per_hour * gpu.usd_per_hour / gpus.L40S.usd_per_hour / 3600.0


def _gate_s(settings: Settings, units: Units) -> float:
    """The gates' serving seconds by the units, before the 2026-10-28.4 measurement scales them."""
    gpu = gpus.GPUS[settings.gpu]
    rate, at_once = _rate(settings.gate_concurrency, settings), _at_once(settings.gate_concurrency, settings)
    prefill = units.prefill_call_s * gpus.L40S.bandwidth_gb_s / gpu.bandwidth_gb_s  # an estimate, as decoding
    general = units.general_calls * prefill / _at_once(settings.max_inputs, settings)
    collapse = units.collapse_prompts * (units.collapse_reply_tokens / rate + prefill / at_once)
    safety = units.safety_cases * units.safety_reply_tokens / rate if settings.safety else 0.0
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


def _practice(settings: Settings, units: Units, serve: float) -> dict[str, float]:
    """Practice's Spark Open, its third-party critic and its serving GPU, by who plays each role."""
    roles, share = settings.roles, dict(units.spark_share)
    lazy = roles.teacher == "lazy"
    critic_calls = 1.0 + (units.all_miss_share if lazy else 0.0)  # a lazy teacher's scene is judged twice
    spark = units.practice_spark_usd * (
        (0.0 if roles.person == "glimmer" else share["person"])
        + share["teacher"] * (units.all_miss_share if lazy else 1.0)
        + (share["critic"] * critic_calls if roles.critic == "spark" else 0.0))
    per_call = (units.critic_in_tokens * GEMINI_IN_PER_M + units.critic_out_tokens * GEMINI_OUT_PER_M) / 1e6
    critic = units.practice_scenes * critic_calls * per_call if roles.critic == "gemini" else 0.0
    gpu = gpus.GPUS[settings.gpu]
    slower = gpus.L40S.stream_tokens_s / gpu.stream_tokens_s
    waves = math.ceil(units.practice_scenes / min(settings.practice_concurrency, settings.max_inputs))
    seconds = units.practice_serial_s * waves / units.practice_scenes * slower
    if roles.person == "glimmer":
        seconds += units.person_reply_tokens / gpu.stream_tokens_s  # the scenes, before any candidate
    return {"practice_spark": spark, "practice_critic": critic, "practice_gpu": seconds * serve}


def project(settings: Settings, units: Units, prices: Prices, night: Measured) -> Projection:
    serve = _serve_usd_s(settings, prices)
    gates_s = _gate_s(settings, units) * units.gates_anchor_s / _gate_s(ANCHOR, units)
    training = units.training_usd if night.training is None else night.training
    train_s = prices.train_gpu_per_hour / 3600.0
    encode = _nap_encode_s(settings, units) * train_s if settings.one_system else 0.0
    overhead = (units.wake_s + (SCALEDOWN_S if settings.idle_tail else 0.0)) / settings.batch + units.train_idle_s
    trained = {
        "extract": night.extract, "brief": night.brief, "mind": night.mind,
        **_practice(settings, units, serve),
        "training": training * (1.0 + (units.joint_training_share if settings.one_system else 0.0)),
        "latent_encode": units.encode_s * train_s if settings.encode_in_training and settings.one_system else encode,
        "gates": gates_s * serve, "serve_overhead": overhead * serve,
    }
    replies, judge = _morning_calls(settings.morning)
    morning_s = replies * units.morning_reply_tokens / _rate(settings.practice_concurrency, settings)
    return Projection(settings, trained, {"extract": night.extract, "brief": night.brief, "mind": night.mind,
                                          "latent_encode": encode},
                      judge * JUDGE_USD_PER_CALL + morning_s * serve,
                      _spark_bound(units, prices) if settings.spark_capped else None)


def _nap_encode_s(settings: Settings, units: Units) -> float:
    """A nap's A100 seconds for one person's day: the start shared by everyone in the call, then the
    day's tokens. Deferred, a nap pays the tokens only, at the next deep wake, in its training container."""
    if settings.encode_deferred:
        return units.encode_s
    load = units.snapshot_load_s if settings.encode_snapshot else units.encode_load_s
    return load / settings.encode_people + units.encode_s


def safety_adds(settings: Settings, units: Units, prices: Prices, night: Measured) -> float:
    """What turning the safety gate on adds to a trained night at ``settings``."""
    on = project(replace(settings, safety=True), units, prices, night).trained["gates"]
    return on - project(replace(settings, safety=False), units, prices, night).trained["gates"]


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
    return out


def _scenario_report(found: Sequence[Projection], traces: Sequence[Trace], safety_on: float) -> list[str]:
    """One row a setting of the whole loop: a deep night (the morning in it), a nap, a month per trace."""
    head = "".join(f"{t.name + ' ' + format(t.deep_share * 7, '.1f') + '/wk':>16}" for t in traces)
    out = ["the night loop under sleep pressure, a setting a row (each proposal alone on the one system, then all)",
           f"  {'setting':<18}{'deep':>9}{'nap':>9}{head}  biggest deep line"]
    for p in found:
        deep, nap = math.fsum(p.trained.values()) + p.morning, math.fsum(p.quiet.values())
        top = max(p.trained, key=lambda step: p.trained[step])
        out.append(f"  {p.settings.name:<18}{_usd(deep):>9}{_usd(nap):>9}"
                   + "".join(f"{_usd(trace_month(p, t)):>16}" for t in traces) + f"  {top} {_usd(p.trained[top])}")
    out.append(f"  safety on adds {_usd(safety_on)} a deep night to the one system")
    out.append("  measured: the anchor night's gates, practice and training; the rest estimated (night.cost.UNITS);"
               f" the morning's judge is {JUDGE_NOTE.split(' is ', 1)[1]}")
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
    parser.add_argument("--scenarios", action="store_true", help="also the whole loop under each cost proposal")
    parser.add_argument("--trace", help="a pressure trace to cost beside the active and quiet ones, such as DNNDNNN")
    args = parser.parse_args(argv)
    if not 0.0 <= args.trained_share <= 1.0 or args.concurrency < 1:
        parser.error("--trained-share is from 0 to 1 and --concurrency at least 1")
    try:
        summary, steps = read_night(args.folder)
    except CostError as error:
        raise SystemExit(f"cost: {error}") from None
    night = measured(summary, steps, args.prices)
    current = replace(CURRENT, name=f"current@{args.concurrency}", practice_concurrency=args.concurrency,
                      gate_concurrency=args.concurrency)
    settings = [BEFORE, current]
    if args.concurrency != CONCURRENT_REQUESTS:
        settings.append(replace(current, name=f"current@{CONCURRENT_REQUESTS}",
                                practice_concurrency=CONCURRENT_REQUESTS, gate_concurrency=CONCURRENT_REQUESTS))
    found = [project(s, UNITS, args.prices, night) for s in settings]
    print("\n".join([*_night_report(args.folder, summary, night_lines(summary, steps, args.prices)), "",
                     *_projection_report(found, args.trained_share)]))
    if args.scenarios:
        try:
            traces = [ACTIVE, QUIET, *([Trace("trace", args.trace)] if args.trace else [])]
        except ValueError as error:
            parser.error(str(error))
        loop = [project(s, UNITS, args.prices, night) for s in SCENARIOS]
        print("\n".join(["", *_scenario_report(loop, traces, safety_adds(ONE_SYSTEM, UNITS, args.prices, night))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
