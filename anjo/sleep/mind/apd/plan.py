"""What a live APD command would start and hold (Task 17), printed by ``--live`` before anything
runs: its Colab session and hard timeout, and each hold it takes on each pool, priced at the list
prices given, Spark Open's on its provider's pool (``mind.spark_provider``). Beside each worst
case it shows the expected spend when a timing has been measured (``timing.json``), and beside the
holds the command's ceiling (``--max-usd``, ``ceiling``) when it takes one. Printing a plan spends
nothing and reaches nothing."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.mind.apd import colab_vllm, generate
from anjo.sleep.mind.apd.backend import gen_holds
from anjo.sleep.mind.apd.ceiling import dollars
from anjo.sleep.mind.apd.intervene import EDIT_SPARK_CALLS
from anjo.sleep.mind.apd.interventions import EDIT_KINDS
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, POOLS, Rate
from anjo.sleep.mind.apd.resume import KILL_RUNS
from anjo.sleep.mind.apd.timing import Timing, cu, kill_cu
from anjo.sleep.mind.apd.types import INTERVENTION_SNAPSHOTS, KILL_RESUME_LIVES, LIFE_DAYS, WORKSPACE_ARMS
from anjo.sleep.mind.spark_provider import SETTING_ENV, SparkProvider
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_worst_usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, MIND_SESSION_PREFIX

NIGHT_WHAT: Final = "screen and resolver, held at a night's first call and settled at what it priced"


@dataclass(frozen=True)
class Hold:
    """One kind of hold: ``each`` on ``pool``, taken ``count`` times; ``expected`` each, as timing
    measured it, when it has."""

    pool: str
    each: float
    count: int
    what: str
    expected: float | None = None


@dataclass(frozen=True)
class Plan:
    command: str
    session_s: float | None  # the Colab session's hard timeout; None starts no session
    holds: tuple[Hold, ...]
    notes: tuple[str, ...] = ()
    expected_cu: float | None = None  # the session's compute units, as timing projects them
    ceiling_usd: float | None = None  # --max-usd, for the commands that take it
    ceiling_unit: str = ""  # what the ceiling is checked before
    parallel: int = 1  # the units that may be in flight at once


def _night(provider: SparkProvider, prices: Prices, nights: int, what: str, timing: Timing | None) -> Hold:
    return Hold(provider.pool, spark_worst_usd(prices, MIND_SPARK_CALLS), nights, what,
                None if timing is None else timing.spark_per_mind_night)


def _spark(provider: SparkProvider) -> str:
    """The note naming where the plan's Spark Open runs and is held."""
    return f"Spark Open on {provider.label} ({provider.model}), held on {provider.pool}, as {SETTING_ENV} sets it"


def _session_cu(timing: Timing | None, nights: int) -> float | None:
    return None if timing is None else cu(timing.setup_s) + nights * timing.cu_per_night


def generate_plan(lives: int, prices: Prices, rates: Mapping[str, Rate], split: str, *,
                  provider: SparkProvider) -> Plan:
    worst = gen_holds(prices, rates, provider.pool)
    attempts = lives * generate.MAX_ATTEMPTS
    return Plan(f"generate {lives} {split} li{'fe' if lives == 1 else 'ves'}", None, (
        Hold(provider.pool, worst[provider.pool], attempts, "Spark Open per attempt: writer, extractor, gate readers"),
        Hold("meta_usd", worst["meta_usd"], attempts, "Muse Spark per attempt: the gates' auditor"),
    ), (_spark(provider), f"at most {generate.MAX_ATTEMPTS} attempts a life; each attempt held before it starts"))


def run_plan(lives: int, arms: Sequence[str], prices: Prices, session_s: float, parallel: int,
             timing: Timing | None, *, provider: SparkProvider) -> Plan:
    mind = sum(1 for arm in arms if arm in WORKSPACE_ARMS)
    nights = lives * len(arms) * LIFE_DAYS
    return Plan(f"run {', '.join(arms)} on {lives} li{'fe' if lives == 1 else 'ves'}, {parallel} at once",
                session_s, (_night(provider, prices, lives * mind * LIFE_DAYS,
                                   f"Spark Open per MIND life-arm night: {NIGHT_WHAT}", timing),),
                (_spark(provider),
                 "a run already on disk is skipped; an unfinished one starts again, its spend recorded as discarded",
                 "A and B call no Spark Open"), _session_cu(timing, nights))


def timing_plan(prices: Prices, session_s: float, parallel: int, *, provider: SparkProvider) -> Plan:
    lives, pool = max(1, math.ceil(parallel / 2)), provider.pool
    return Plan(f"timing at {parallel} at once: B and M on {lives} development li{'fe' if lives == 1 else 'ves'}, "
                "17b's labels and branches on one, the intervention test on one snapshot", session_s, (
                    _night(provider, prices, lives * LIFE_DAYS, f"Spark Open per M night: {NIGHT_WHAT}", None),
                    Hold(pool, spark_worst_usd(prices, 1), 0, "Spark Open per start state's label, held first"),
                    Hold(pool, spark_worst_usd(prices, 4 * MIND_SPARK_CALLS), 0,
                         "Spark Open per start state's branches (screen and resolver), one start state at a time"),
                    Hold(pool, spark_worst_usd(prices, EDIT_SPARK_CALLS), len(EDIT_KINDS),
                         "Spark Open per snapshot-edit of the intervention test"),
                ), (_spark(provider), "the training is planned only: no training job runs"))


def intervene_plan(prices: Prices, session_s: float, timing: Timing | None, *, provider: SparkProvider) -> Plan:
    items = INTERVENTION_SNAPSHOTS * len(EDIT_KINDS)
    expected = None if timing is None or not timing.intervene_items else timing.intervene_usd / timing.intervene_items
    return Plan(f"the intervention test: {INTERVENTION_SNAPSHOTS} snapshots x {len(EDIT_KINDS)} edits, one at a time",
                session_s, (
                    Hold(provider.pool, spark_worst_usd(prices, EDIT_SPARK_CALLS), items, expected=expected,
                         what="Spark Open per snapshot-edit: both twins' screen and resolver, and the wrong "
                              "hypothesis's writer"),
                ), (_spark(provider), "a snapshot-edit already on disk is skipped",
                    "one at a time: vLLM at temperature 0 isn't batch-invariant, and the twins are compared token "
                    "for token"),
                None if timing is None else cu(timing.setup_s) + timing.intervene_cu(items))


def judge_plan(rates: Mapping[str, Rate]) -> Plan:
    return Plan("judge every run on disk, blind", None, tuple(
        Hold(POOLS[name], rates[name].worst, 0, f"{name} per new prompt (a prompt judged before costs nothing)")
        for name in (MUSE, GEMINI)
    ), ("disagreements go to the blind audit file for Claude; nothing else is called",))


def kill_plan(prices: Prices, session_s: float, timing: Timing | None, lives: int = KILL_RESUME_LIVES, *,
              provider: SparkProvider) -> Plan:
    return Plan(f"kill-resume on {lives} held-out lives, one at a time", session_s, (
        _night(provider, prices, lives * KILL_RUNS * LIFE_DAYS,
               "Spark Open per night of each uninterrupted, killed and resumed run, in that run's process; a "
               "killed night settles at its journal plus one call's worst case", timing),
    ), (_spark(provider), "a life done before is skipped",
        "one at a time: vLLM at temperature 0 isn't batch-invariant, and the runs are compared token for token"),
        None if timing is None else cu(timing.setup_s) + kill_cu(timing, lives))


def _ceiling(plan: Plan) -> tuple[str, ...]:
    if plan.ceiling_usd is None:
        return ()
    flight = (f"the {plan.ceiling_unit} already running finishes under its own hold above, so the run can pass "
              "the ceiling by at most that one" if plan.parallel == 1 else
              f"each {plan.ceiling_unit} already running (up to {plan.parallel} at once) finishes under its own hold "
              "above, so the run can pass the ceiling by at most those")
    return (
        f"Ceiling: {dollars(plan.ceiling_usd)} settled by this run (--max-usd): before each new {plan.ceiling_unit} "
        "it sums what its own holds have settled, not the pools' totals, and starts nothing more once that reaches "
        "the ceiling; Colab's compute units are capped apart (colab_cu).",
        f"Note: the ceiling stops new work, not a unit in flight: {flight}.",
    )


def describe(plan: Plan) -> str:
    lines = [f"Plan: {plan.command}."]
    if plan.session_s is None:
        lines.append("Colab: no session.")
    else:
        hold = colab_vllm.hold_amount(plan.session_s)
        expected = "" if plan.expected_cu is None else f"; expected about {plan.expected_cu:.1f} CU (timing)"
        lines.append(f"Colab: one A100 session ({MIND_SESSION_PREFIX}{colab_vllm.SESSION_LABEL}<random>), hard timeout "
                     f"{plan.session_s / 3600:.1f} h, held first on colab_cu: {hold:.1f} CU "
                     f"({A100_CU_PER_HOUR} CU/h){expected}; every CLI call and the tunnel under their own timeouts.")
    for hold_ in plan.holds:
        amount = f"{hold_.each:.1f} CU" if hold_.pool == "colab_cu" else f"${hold_.each:.2f}"
        times = f" x {hold_.count} (at most ${hold_.each * hold_.count:.2f} in all)" if hold_.count else ""
        expected = "" if hold_.expected is None else (
            f"; expected about ${hold_.expected:.4f} each (timing), ${hold_.expected * hold_.count:.2f} in all")
        lines.append(f"Hold on {hold_.pool}: {amount} each{times}{expected}: {hold_.what}.")
    lines.extend(_ceiling(plan))
    lines.extend(f"Note: {note}." for note in plan.notes)
    lines.append("Nothing has run. Add --yes to start it.")
    return "\n".join(lines)
