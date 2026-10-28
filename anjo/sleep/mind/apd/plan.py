"""What a live APD command would start and hold (Task 17), printed by ``--live`` before anything
runs: its Colab session and hard timeout, and each hold it takes on each pool, priced at the list
prices given. Beside each worst case it shows the expected spend when a timing has been measured
(``timing.json``). Printing a plan spends nothing and reaches nothing."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.mind.apd import colab_vllm, generate
from anjo.sleep.mind.apd.backend import gen_holds
from anjo.sleep.mind.apd.intervene import EDIT_SPARK_CALLS
from anjo.sleep.mind.apd.interventions import EDIT_KINDS
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, POOLS, Rate
from anjo.sleep.mind.apd.timing import KILL_RUNS, Timing, cu
from anjo.sleep.mind.apd.types import INTERVENTION_SNAPSHOTS, KILL_RESUME_LIVES, LIFE_DAYS, WORKSPACE_ARMS
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_worst_usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, MIND_SESSION_PREFIX

SPARK_POOL: Final = "baseten_mind_usd"
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


def _night(prices: Prices, nights: int, what: str, timing: Timing | None) -> Hold:
    return Hold(SPARK_POOL, spark_worst_usd(prices, MIND_SPARK_CALLS), nights, what,
                None if timing is None else timing.spark_per_mind_night)


def _session_cu(timing: Timing | None, nights: int) -> float | None:
    return None if timing is None else cu(timing.setup_s) + nights * timing.cu_per_night


def generate_plan(lives: int, prices: Prices, rates: Mapping[str, Rate], split: str) -> Plan:
    worst = gen_holds(prices, rates)
    attempts = lives * generate.MAX_ATTEMPTS
    return Plan(f"generate {lives} {split} li{'fe' if lives == 1 else 'ves'}", None, (
        Hold(SPARK_POOL, worst[SPARK_POOL], attempts, "Spark Open per attempt: writer, extractor, gate readers"),
        Hold("meta_usd", worst["meta_usd"], attempts, "Muse Spark per attempt: the gates' auditor"),
    ), (f"at most {generate.MAX_ATTEMPTS} attempts a life; each attempt held before it starts",))


def run_plan(lives: int, arms: Sequence[str], prices: Prices, session_s: float, parallel: int,
             timing: Timing | None) -> Plan:
    mind = sum(1 for arm in arms if arm in WORKSPACE_ARMS)
    nights = lives * len(arms) * LIFE_DAYS
    return Plan(f"run {', '.join(arms)} on {lives} li{'fe' if lives == 1 else 'ves'}, {parallel} at once",
                session_s, (_night(prices, lives * mind * LIFE_DAYS, f"Spark Open per MIND life-arm night: {NIGHT_WHAT}",
                                   timing),),
                ("a run already on disk is skipped; an unfinished one starts again, its spend recorded as discarded",
                 "A and B call no Spark Open"), _session_cu(timing, nights))


def timing_plan(prices: Prices, session_s: float, parallel: int) -> Plan:
    lives = max(1, math.ceil(parallel / 2))
    return Plan(f"timing at {parallel} at once: B and M on {lives} development li{'fe' if lives == 1 else 'ves'}, "
                "17b's labels and branches on one, the intervention test on one snapshot", session_s, (
                    _night(prices, lives * LIFE_DAYS, f"Spark Open per M night: {NIGHT_WHAT}", None),
                    Hold(SPARK_POOL, spark_worst_usd(prices, 1), 0, "Spark Open per start state's label, held first"),
                    Hold(SPARK_POOL, spark_worst_usd(prices, 4 * MIND_SPARK_CALLS), 0,
                         "Spark Open per start state's branches (screen and resolver), one start state at a time"),
                    Hold(SPARK_POOL, spark_worst_usd(prices, EDIT_SPARK_CALLS), len(EDIT_KINDS),
                         "Spark Open per snapshot-edit of the intervention test"),
                ), ("the training is planned only: no training job runs",))


def intervene_plan(prices: Prices, session_s: float, parallel: int, timing: Timing | None) -> Plan:
    items = INTERVENTION_SNAPSHOTS * len(EDIT_KINDS)
    expected = None if timing is None or not timing.intervene_items else timing.intervene_usd / timing.intervene_items
    return Plan(f"the intervention test: {INTERVENTION_SNAPSHOTS} snapshots x {len(EDIT_KINDS)} edits, {parallel} at once",
                session_s, (
                    Hold(SPARK_POOL, spark_worst_usd(prices, EDIT_SPARK_CALLS), items, expected=expected,
                         what="Spark Open per snapshot-edit: both twins' screen and resolver, and the wrong "
                              "hypothesis's writer"),
                ), ("a snapshot-edit already on disk is skipped",),
                None if timing is None else cu(timing.setup_s) + timing.intervene_cu(items))


def judge_plan(rates: Mapping[str, Rate]) -> Plan:
    return Plan("judge every run on disk, blind", None, tuple(
        Hold(POOLS[name], rates[name].worst, 0, f"{name} per new prompt (a prompt judged before costs nothing)")
        for name in (MUSE, GEMINI)
    ), ("disagreements go to the blind audit file for Claude; nothing else is called",))


def kill_plan(prices: Prices, session_s: float, timing: Timing | None, lives: int = KILL_RESUME_LIVES) -> Plan:
    return Plan(f"kill-resume on {lives} held-out lives at once", session_s, (
        _night(prices, lives * KILL_RUNS * LIFE_DAYS,
               "Spark Open per night of each uninterrupted, killed and resumed run, in that run's process; a "
               "killed night settles at its journal plus one call's worst case", timing),
    ), ("a life done before is skipped",),
        None if timing is None else cu(timing.setup_s + KILL_RUNS * LIFE_DAYS * timing.night_wall_s))


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
    lines.extend(f"Note: {note}." for note in plan.notes)
    lines.append("Nothing has run. Add --yes to start it.")
    return "\n".join(lines)
