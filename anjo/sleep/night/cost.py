"""One night's cost by step, and a person-month projected at the night's settings (spec §5.5).

    python -m anjo.sleep.night.cost --from <night folder> [--prices in,out,train,serve]
        [--concurrency N] [--pressure-trace FILE] [--month-usd D] [--max-days N] [--scenarios]

**The night as logged** reads the folder's ``summary.json`` and ``log.jsonl`` (counts, steps, costs
and seconds; no conversation text) and prints a line a step, the biggest first:

- the summary's cost parts, practice split into the serving GPU's wall time (the step's seconds at
  the serving price) and Spark Open (the rest). The ceiling counts every part but MIND
  (``demo.dailies.CEILING_PARTS``);
- what the summary never holds, outside the ceiling: Wonder's homework (``wonder_usd``) and the
  morning run (its GPU seconds, and its judge as the ledger settled it: the replies' tokens, and each call
  that reported none at ``JUDGE_USD_PER_CALL``, the hold's bound, shown ``estimated``; a log from before
  the judge was metered gives only its calls, all at the bound and ``estimated``).

**The month** is one projection, from pressure traces: a month of days, each the day's new turns, new
trainable items and open high-risk targets (``DayLoad``), run through the nights' own nap-or-deep
decision (``night/sleep_pressure.py``): its deep night sized by the setting (``deep_night_usd``), the
month's later nights held back as naps and Wonder's room held back, so the month stays under the cap
(``--month-usd``, ``--max-days``). A deep night costs the setting's trained night and its morning. A nap
wakes no Glimmer, so it costs what the live ledger counts on one: extract and the brief (MIND waits for
her next deep sleep); a capped nap, extract alone. Wonder's homework (``night:homework:``) is a study a
night at ``UNITS.homework_usd`` until its $3 a month (``NightSettings.wonder_usd_month``) is spent, the
worst case. With MIND's core on (``ANJO_SLEEP_MIND_CORE``) a nap's share of one batched latent encode is
shown apart, planned, not live, until a live night runs it. The headline is a person-month at ``ACTIVE``
and ``QUIET`` usage; ``--pressure-trace FILE`` (a JSON list of days, ``{"turns": 40, "evidence": 8,
"high_risk": 1}``) adds one, printed night by night with its pressure, threshold, kind and reason.
The night and the month are projected at two settings:

- ``BEFORE``, the first live nights (2026-10-28): practice's scenes one at a time, the generation
  gates at the server's 16 with both sides asked every night, Spark Open uncapped, the full morning;
- ``CURRENT``: practice and the generation gates at ``ANJO_SLEEP_CONCURRENCY`` (8 by default), the
  gates' base side kept across nights (``gates.base_cache``), Spark Open capped per role
  (``practice.full.SPARK_CAPS``), the daily morning subset (``probes.morning.DAILY``), and the safety
  gate skipped (``ANJO_SLEEP_SAFETY_GATE`` off for the hackathon: no case is sent, and its floor still
  runs, with no model and no GPU; ``--safety-gate on`` costs it). A night whose log shows the gate
  skipped says so; it is never costed as passed.

``--scenarios`` costs the whole loop at each cost proposal, a row a setting (``SCENARIOS``): today's
night as measured (``ANCHOR``: 2026-10-29, the first night on the trunk, safety gate off and the base
cache missed), the base cache, the one system (MIND's core on: the day's latent encode and the joint
training beside the LoRA), then each proposal on it alone and all of them together, with what turning
the safety gate on adds. Every deep night also pays the serving GPU's ``serve_overhead``, which the
summary never logs: the cold start and Modal's idle window after the last request (shared by every
person a wake serves, ``Settings.batch``), and its wait while the dataset is built and training runs.

Extraction, the brief, MIND's inquiry and training are the night's own numbers. What a setting
changes is estimated from ``UNITS``, each with its source: the 2026-10-29 night anchors the gates (their
measured dollars, which scale the gates' estimate), the morning's arms and the cold start; the
2026-10-28 measurements anchor practice (its serial seconds and Spark dollars; night 1 reused its
practice). The caps bound practice's Spark Open at its worst; they don't lower a night that stays under
them. Numbers only: nothing here reads a clock, the environment or any file but the night's two (and a
pressure trace's).
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
from anjo.sleep.night.sleep_pressure import (
    MEASURED_REUSED_USD,
    Kind,
    Reason,
    SleepPolicy,
    SleepPolicyError,
    SleepState,
    decide,
    sleep_pressure,
)
from anjo.sleep.practice.critics import DEEPSEEK_IN_PER_M, DEEPSEEK_OUT_PER_M
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
QUIET_LINES: Final = ("extract", "brief")  # a nap's own lines; MIND waits for her next deep sleep
PLANNED: Final = "planned, not live"  # the nap's latent encode, until a live night runs it
WONDER_MONTH_USD: Final = 3.0  # NightSettings.wonder_usd_month, Wonder's rolling cap (a test keeps them equal)
JUDGE_NOTE: Final = (f"the morning judge is estimated at ${JUDGE_USD_PER_CALL:g} a call (JUDGE_USD_PER_CALL, the hold's"
                     " per-call bound) until a night's log carries its tokens: 2026-10-29's $0.44 is 22 calls at it")
ESTIMATED_NOTE: Final = (f"estimated: judge calls that reported no tokens, counted at ${JUDGE_USD_PER_CALL:g} a call (the"
                         " hold's per-call bound); the rest at their tokens")


class CostError(ValueError):
    """The night's files can't be costed."""


@dataclass(frozen=True)
class Line:
    step: str
    usd: float
    gpu_minutes: float | None
    in_ceiling: bool
    estimated: bool = False  # some of it at a per-call bound, not tokens


@dataclass(frozen=True)
class Settings:
    name: str
    practice_concurrency: int
    gate_concurrency: int  # collapse and safety; the general gate's log-probabilities run at the server's 16
    base_cached: bool
    spark_capped: bool
    morning: ProbeSubset | None  # None: the full set
    safety_gate: bool = True  # False: switched off (ANJO_SLEEP_SAFETY_GATE): no case sent, the floor alone (no GPU)
    gpu: str = "L40S"  # ANJO_SLEEP_SERVE_GPU
    max_inputs: int = DEFAULT_MAX_INPUTS  # ANJO_SLEEP_SERVE_MAX_INPUTS: requests a container takes at once
    roles: Roles = Roles()  # ANJO_SLEEP_PRACTICE_ROLES
    batch: int = 1  # people one wake serves (run_night_batch): the cold start and the idle window shared
    idle_tail: bool = True  # Modal bills its idle window after the last request (False: retired at the end)
    mind_core: bool = False  # ANJO_SLEEP_MIND_CORE=1: the day's latent encode and the joint training
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
    gates_anchor_s: float = 393.0  # measured 2026-10-29 (night 1 on the trunk): $0.214 at $1.96/h, at ANCHOR
    practice_scenes: int = 16
    practice_serial_s: float = 1174.0  # measured 2026-10-28: 16 scenes one at a time
    practice_spark_usd: float = 0.83  # measured 2026-10-28, uncapped
    practice_spark_in_tokens: int = 150_000  # estimate: F and the scene in each of 33 calls
    training_usd: float = 0.107  # measured 2026-10-29; a night's receipt replaces it
    morning_reply_tokens: int = 400  # estimate: scales the measured arms to another subset or GPU
    morning_arms_s: float = 165.0  # measured 2026-10-29: the daily subset's arms, $0.09 at $1.96/h
    dailies_usd: float = 0.005  # measured 2026-10-29
    homework_usd: float = 0.227  # measured 2026-10-29: one Wonder study (Muse Wonder), outside the nights
    wake_s: float = 225.0  # measured 2026-10-29: the night's GPU wake (weights off the volume, vLLM up)
    train_idle_s: float = 230.0  # estimate: the serving GPU awake while the dataset is built and training runs
    # estimate: the measured $0.83 by role (one scenes call, 16 teacher and 16 critic calls of about the
    # same size); a night's log carries the real split (``spark`` by role)
    spark_share: tuple[tuple[str, float], ...] = (("person", 0.04), ("teacher", 0.48), ("critic", 0.48))
    person_reply_tokens: int = 1_500  # estimate: twenty scenes as JSON, when Glimmer writes them
    critic_in_tokens: int = 6_000  # estimate: F, the message and four candidates
    critic_out_tokens: int = 300  # estimate: DeepSeek's JSON, non-thinking
    all_miss_share: float = 0.25  # estimate: scenes whose three Glimmer candidates all miss (the A/B measures it)
    encode_load_s: float = 370.0  # estimate (mind.core.latents.LOAD_S): an A100 start, 56 GB of bf16, 10 s idle
    encode_s: float = 33.0  # estimate: a day's ~120k tokens to layer 35, at latents' 2,500 tokens/s full depth
    snapshot_load_s: float = 45.0  # estimate, unmeasured: a GPU memory snapshot's restore instead of the load
    joint_training_share: float = 0.3  # estimate: the projector, the reader and the core beside the LoRA


BEFORE: Final = Settings("before", practice_concurrency=1, gate_concurrency=CONCURRENT_REQUESTS,
                         base_cached=False, spark_capped=False, morning=None)
CURRENT: Final = Settings("current", practice_concurrency=8, gate_concurrency=8, base_cached=True,
                          spark_capped=True, morning=DAILY, safety_gate=False)
UNITS: Final = Units()
# Today's night, the one the gates and the morning are anchored on (2026-10-29: the safety gate off and the
# base cache missed, so both sides asked), and the loop's settings from it to every proposal at once.
ANCHOR: Final = Settings("today (10-29)", practice_concurrency=8, gate_concurrency=8, base_cached=False,
                         spark_capped=True, morning=DAILY, safety_gate=False)
ONE_SYSTEM: Final = replace(CURRENT, name="one system", mind_core=True)
ALL: Final = replace(ONE_SYSTEM, name="all", roles=Roles("glimmer", "deepseek", "lazy"), batch=5, idle_tail=False,
                     encode_in_training=True, encode_people=5)
SCENARIOS: Final[tuple[Settings, ...]] = (
    ANCHOR, replace(CURRENT, name="+ base cache"),
    ONE_SYSTEM,
    replace(ONE_SYSTEM, name="critic=deepseek", roles=Roles(critic="deepseek")),
    replace(ONE_SYSTEM, name="person=glimmer", roles=Roles(person="glimmer")),
    replace(ONE_SYSTEM, name="teacher=lazy", roles=Roles(teacher="lazy")),
    replace(ONE_SYSTEM, name="H100 at 32 (list)", gpu="H100", max_inputs=32, gate_concurrency=32, practice_concurrency=16),
    replace(ONE_SYSTEM, name="batch of 5", batch=5),
    replace(ONE_SYSTEM, name="retire at end", idle_tail=False),
    replace(ONE_SYSTEM, name="encode in train", encode_in_training=True),
    replace(ONE_SYSTEM, name="encode 5 at once", encode_people=5),
    replace(ONE_SYSTEM, name="encode snapshot", encode_snapshot=True),
    replace(ONE_SYSTEM, name="encode deferred", encode_deferred=True, encode_in_training=True),
    ALL,
)


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
    quiet: Mapping[str, float]  # a nap's own lines (``QUIET_LINES``)
    morning: float  # outside the ceiling
    spark_bound: float | None  # practice's Spark Open at its worst under the caps
    nap_encode: float = 0.0  # a nap's share of the latent encode with the core on: planned, not live

    @property
    def deep(self) -> float:
        """A deep night: the trained night and its morning."""
        return math.fsum(self.trained.values()) + self.morning

    def policy(self, policy: SleepPolicy) -> SleepPolicy:
        """``policy`` with its deep night sized by this setting's GPU and MIND's core."""
        return replace(policy, gpu=self.settings.gpu, core=self.settings.mind_core)


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
    morning = _step(steps, "morning")
    if (spent := _number(morning.get("judge_usd"))) is not None:  # settled: tokens, and the bound for calls without
        found.append(Line("morning_judge", spent, None, False, bool(_number(morning.get("judge_estimated")))))
    elif (calls := _number(morning.get("judge_calls"))) is not None:  # logged before the judge was metered
        found.append(Line("morning_judge", calls * JUDGE_USD_PER_CALL, None, False, True))
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


def _practice(settings: Settings, units: Units, serve: float) -> dict[str, float]:
    """Practice's Spark Open, its third-party critic and its serving GPU, by who plays each role."""
    roles, share = settings.roles, dict(units.spark_share)
    lazy = roles.teacher == "lazy"
    critic_calls = 1.0 + (units.all_miss_share if lazy else 0.0)  # a lazy teacher's scene is judged twice
    spark = units.practice_spark_usd * (
        (0.0 if roles.person == "glimmer" else share["person"])
        + share["teacher"] * (units.all_miss_share if lazy else 1.0)
        + (share["critic"] * critic_calls if roles.critic == "spark" else 0.0))
    per_call = (units.critic_in_tokens * DEEPSEEK_IN_PER_M + units.critic_out_tokens * DEEPSEEK_OUT_PER_M) / 1e6
    critic = units.practice_scenes * critic_calls * per_call if roles.critic == "deepseek" else 0.0
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
    encode = _nap_encode_s(settings, units) * train_s if settings.mind_core else 0.0
    overhead = (units.wake_s + (SCALEDOWN_S if settings.idle_tail else 0.0)) / settings.batch + units.train_idle_s
    trained = {
        "extract": night.extract, "brief": night.brief, "mind": night.mind,
        **_practice(settings, units, serve),
        "training": training * (1.0 + (units.joint_training_share if settings.mind_core else 0.0)),
        "latent_encode": units.encode_s * train_s if settings.encode_in_training and settings.mind_core else encode,
        "gates": gates_s * serve, "serve_overhead": overhead * serve,
    }
    replies, judge = _morning_calls(settings.morning)
    model_s = replies * units.morning_reply_tokens / _rate(settings.practice_concurrency, settings)
    anchor_s = _morning_calls(ANCHOR.morning)[0] * units.morning_reply_tokens / _rate(ANCHOR.practice_concurrency, ANCHOR)
    morning_s = units.morning_arms_s * model_s / anchor_s  # today's measured arms, scaled by the estimate
    return Projection(settings, trained, {"extract": night.extract, "brief": night.brief},
                      judge * JUDGE_USD_PER_CALL + morning_s * serve + units.dailies_usd,
                      _spark_bound(units, prices) if settings.spark_capped else None, encode)


def _nap_encode_s(settings: Settings, units: Units) -> float:
    """A nap's A100 seconds for one person's day: the start shared by everyone in the call, then the
    day's tokens. Deferred, a nap pays the tokens only, at the next deep wake, in its training container."""
    if settings.encode_deferred:
        return units.encode_s
    load = units.snapshot_load_s if settings.encode_snapshot else units.encode_load_s
    return load / settings.encode_people + units.encode_s


def safety_adds(settings: Settings, units: Units, prices: Prices, night: Measured) -> float:
    """What turning the safety gate on adds to a trained night at ``settings``."""
    on = project(replace(settings, safety_gate=True), units, prices, night).trained["gates"]
    return on - project(replace(settings, safety_gate=False), units, prices, night).trained["gates"]


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
    capped: bool = False  # a nap with no room under the cap: extract alone


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


@dataclass(frozen=True)
class Usage:
    """A named month of days: the headline's people, or a ``--pressure-trace`` file."""

    name: str
    days: tuple[DayLoad, ...]


# The headline's two people, steady months (estimates until real traces replace them): an active one
# talks every day and leaves about half a day's worth of items, so pressure builds to a deep sleep in a
# few days; a quiet one leaves too little for pressure, and sleeps deep only at ``max_days``.
ACTIVE: Final = Usage("active", (DayLoad(turns=30, evidence=5),) * MONTH_NIGHTS)
QUIET: Final = Usage("quiet", (DayLoad(turns=4, evidence=1),) * MONTH_NIGHTS)


def month_from_trace(trace: Sequence[DayLoad], policy: SleepPolicy, projection: Projection, *,
                     study_usd: float = UNITS.homework_usd) -> MonthTrace:
    """Each day's night, decided as the live nights decide it (the deep night sized by the projection's
    settings, the base cache unknown before the wake, the month's later nights and Wonder's room held
    back), at the projection's costs: a deep night, a nap's own lines (a capped nap's extract alone), and
    a Wonder study a night (``study_usd``) until its month's dollars are spent, never on a capped nap."""
    policy = projection.policy(policy)
    nap_usd = math.fsum(projection.quiet.values())
    spent, wonder, turns, evidence, last_deep = 0.0, 0.0, 0, 0, None
    nights: list[TracedNight] = []
    for i, day in enumerate(trace):
        turns, evidence = turns + day.turns, evidence + day.evidence
        since = None if i == 0 else i - (0 if last_deep is None else last_deep)
        state = SleepState(new_turns=turns, new_evidence=evidence, open_high_risk=day.high_risk,
                           days_since_deep=since, month_spent_usd=spent, nights_left=len(trace) - i - 1,
                           wonder_left_usd=WONDER_MONTH_USD - wonder)
        made = decide(state, policy, pressure=sleep_pressure(state))
        study = 0.0 if made.capped else min(study_usd, WONDER_MONTH_USD - wonder)
        own = projection.deep if made.kind == "deep" else projection.quiet["extract"] if made.capped else nap_usd
        nights.append(TracedNight(made.kind, made.reason, made.pressure, made.threshold, own + study, made.capped))
        spent, wonder = spent + own + study, wonder + study
        if made.kind == "deep":
            turns, evidence, last_deep = 0, 0, i
    return MonthTrace(tuple(nights))


def _trace_report(found: MonthTrace, policy: SleepPolicy, name: str) -> list[str]:
    out = [f"a month from the pressure trace, at {name}", f"  {'night':<7}{'pressure':>9}{'threshold':>10}  {'kind':<6}reason"]
    for n, night in enumerate(found.nights, start=1):
        out.append(f"  {n:<7}{night.pressure:>9.2f}{night.threshold:>10.2f}  {night.kind:<6}{night.reason}"
                   + (" (capped)" if night.capped else ""))
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
        out.append(f"  {line.step:<16}{_usd(line.usd):>10}{minutes:>9}  {'yes' if line.in_ceiling else 'no'}"
                   + ("  estimated" if line.estimated else ""))
    ceiling = ceiling_usd(lines)
    out.append(f"  ceiling parts {_usd(ceiling)}; a month of nights like this: {_usd(MONTH_NIGHTS * ceiling)}"
               f" (ceiling ${CEILING_USD:g})")
    if any(line.estimated for line in lines):
        out.append(f"  {ESTIMATED_NOTE}")
    safety = (summary.get("gates") or {}).get("safety")
    if isinstance(safety, Mapping) and safety.get("skipped"):
        out.append(f"  safety gate: skipped ({SKIPPED_REASON}); the gates line is general and collapse only")
    return out


def _months(p: Projection, usages: Sequence[Usage], policy: SleepPolicy) -> list[MonthTrace]:
    return [month_from_trace(u.days, policy, p) for u in usages]


def _projection_report(found: Sequence[Projection], usages: Sequence[Usage], policy: SleepPolicy) -> list[str]:
    out = ["projected month, per step of a trained night", "  " + f"{'':<16}" + "".join(
        f"{p.settings.name:>14}" for p in found)]
    for step in TRAINED_LINES:
        out.append(f"  {step:<16}" + "".join(f"{_usd(p.trained[step]):>14}" for p in found))

    def row(label: str, value: Any) -> str:
        return f"  {label:<16}" + "".join(f"{_usd(value(p)):>14}" for p in found)

    out.append(row("trained night", lambda p: math.fsum(p.trained.values())))
    out.append(row("deep estimate", lambda p: p.policy(policy).deep_usd))
    out.append(row("quiet night", lambda p: math.fsum(p.quiet.values())))
    out.append(row("nap encode", lambda p: p.nap_encode))
    months = {p.settings.name: _months(p, usages, policy) for p in found}
    for i, usage in enumerate(usages):
        out.append(row(f"month, {usage.name}", lambda p, i=i: months[p.settings.name][i].usd))
        out.append(f"  {'  deep nights':<16}" + "".join(f"{months[p.settings.name][i].deep:>14}" for p in found))
    out.append(row("worst case", lambda p: month(math.fsum(p.trained.values()), 0.0, trained_share=1.0)))
    out.append(row("morning (out)", lambda p: p.morning))
    out.append(row("spark cap bound", lambda p: p.spark_bound))
    out.append(f"  a month: its days through the nap-or-deep decision (a deep night with its morning, a nap its "
               f"quiet night, Wonder a ${UNITS.homework_usd:g} study a night until its ${WONDER_MONTH_USD:g} is spent), "
               f"under the ${policy.month_usd:g} cap with the month's later nights held back; worst case: every "
               f"night trained, no cap (ceiling ${CEILING_USD:g}). The morning run and MIND are outside the ceiling.")
    out.extend(_estimate_notes(found, policy))
    out.append("  measured: extract, brief, training (this night); gates, the morning's arms and the cold start "
               "anchored on 2026-10-29, practice on 2026-10-28; the rest estimated (night.cost.UNITS)")
    out.append(f"  morning (out): {JUDGE_NOTE}")
    if off := [p.settings.name for p in found if not p.settings.safety_gate]:
        out.append(f"  safety gate: skipped in {', '.join(off)} ({SKIPPED_REASON}); no case sent, never counted as passed")
    return out


def _estimate_notes(found: Sequence[Projection], policy: SleepPolicy) -> list[str]:
    """What the decision sizes a deep night at, beside a measured one; and any setting it can't hold."""
    out = [f"  deep estimate: the modelled fresh night the decision holds back, at the setting's GPU and MIND's core (base"
           f" cache missed, practice fresh); measured, practice reused: ${MEASURED_REUSED_USD:.2f} (2026-10-29); nap"
           f" encode: {PLANNED}"]
    if over := [p.settings.name for p in found if p.deep > p.policy(policy).deep_usd + 1e-9]:
        out.append(f"  over the estimate, so the cap can't hold its month: {', '.join(over)}")
    return out


def _scenario_report(found: Sequence[Projection], usages: Sequence[Usage], policy: SleepPolicy,
                     safety_on: float) -> list[str]:
    """One row a setting of the whole loop: a deep night (the morning in it), a nap, a month per usage."""
    out = ["the night loop under sleep pressure, a setting a row (each proposal alone on the one system, then all)",
           f"  {'setting':<18}{'deep':>9}{'nap':>9}" + "".join(f"{u.name + ' month':>22}" for u in usages)
           + "  biggest deep line"]
    for p in found:
        deep, nap = p.deep, math.fsum(p.quiet.values())
        lines = {**p.trained, "morning": p.morning}
        top = max(lines, key=lambda step: lines[step])
        cells = "".join(f"{_usd(m.usd) + f' ({m.deep} deep)':>22}" for m in _months(p, usages, policy))
        out.append(f"  {p.settings.name:<18}{_usd(deep):>9}{_usd(nap):>9}{cells}  {top} {_usd(lines[top])}")
    out.append(f"  safety on adds {_usd(safety_on)} a deep night to the one system")
    out.extend(_estimate_notes(found, policy))
    if off := [p.settings.name for p in found if not p.settings.safety_gate]:
        out.append(f"  safety gate: skipped in {len(off)} of {len(found)} rows ({SKIPPED_REASON}): no case sent, never"
                   " counted as passed; its floor still runs, with no model and no GPU")
    for gpu in sorted({gpus.GPUS[p.settings.gpu] for p in found} - {gpus.L40S}, key=lambda g: g.name):
        out.append(f"  {gpu.name} rows: ${gpu.usd_per_hour:.2f}/h is {gpu.price_note}, scaled from the L40S's"
                   " billed rate")
    out.append("  measured: today's (2026-10-29) gates, training, morning arms and cold start; practice 2026-10-28's"
               f" (night 1 reused its practice); the rest estimated (night.cost.UNITS); {JUDGE_NOTE}")
    out.append(f"  a month at the ${policy.month_usd:g} cap: a deep sleep that wouldn't leave room for the month's later"
               " nights and Wonder naps (month_cap); a nap with no room skips its paid steps (capped)")
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
    parser.add_argument("--concurrency", type=int, default=CURRENT.gate_concurrency,
                        help="ANJO_SLEEP_CONCURRENCY for the current projection (8 by default)")
    parser.add_argument("--scenarios", action="store_true", help="also the whole loop under each cost proposal")
    parser.add_argument("--safety-gate", choices=("on", "off"), default="off",
                        help="ANJO_SLEEP_SAFETY_GATE for the current projection (off for the hackathon)")
    parser.add_argument("--pressure-trace", type=Path,
                        help="a JSON list of days, costed beside the active and quiet months and printed night by night")
    parser.add_argument("--month-usd", type=float, default=SleepPolicy().month_usd, help="ANJO_SLEEP_MONTH_USD")
    parser.add_argument("--max-days", type=int, default=SleepPolicy().max_days, help="ANJO_SLEEP_MAX_DAYS")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency is at least 1")
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
    usages = [ACTIVE, QUIET, *([Usage("trace", tuple(trace))] if trace is not None else [])]
    traced = [] if trace is None else ["", *_trace_report(month_from_trace(trace, policy, found[1]), policy, current.name)]
    print("\n".join([*_night_report(args.folder, summary, night_lines(summary, steps, args.prices)), "",
                     *_projection_report(found, usages, policy), *traced]))
    if args.scenarios:
        loop = [project(s, UNITS, args.prices, night) for s in SCENARIOS]
        safety_on = safety_adds(ONE_SYSTEM, UNITS, args.prices, night)
        print("\n".join(["", *_scenario_report(loop, usages, policy, safety_on)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
