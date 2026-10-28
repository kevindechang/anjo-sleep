"""What a live APD command would start and hold (Task 17), printed by ``--live`` before anything
runs: its Colab session and hard timeout, and each hold it takes on each pool, priced at the list
prices given. Printing a plan spends nothing and reaches nothing."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.mind.apd import colab_vllm, generate
from anjo.sleep.mind.apd.backend import gen_holds
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, POOLS, Rate
from anjo.sleep.mind.apd.types import KILL_RESUME_LIVES, LIFE_DAYS
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_worst_usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, MIND_SESSION_PREFIX

SPARK_POOL: Final = "baseten_mind_usd"


@dataclass(frozen=True)
class Hold:
    """One kind of hold: ``each`` on ``pool``, taken ``count`` times, one at a time."""

    pool: str
    each: float
    count: int
    what: str


@dataclass(frozen=True)
class Plan:
    command: str
    session_s: float | None  # the Colab session's hard timeout; None starts no session
    holds: tuple[Hold, ...]
    notes: tuple[str, ...] = ()


def _run_spark(prices: Prices, runs: int, what: str) -> Hold:
    """A run's Spark Open calls, held a night at a time as the night's MIND step holds them."""
    return Hold(SPARK_POOL, spark_worst_usd(prices, MIND_SPARK_CALLS), runs * LIFE_DAYS, what)


def generate_plan(lives: int, prices: Prices, rates: Mapping[str, Rate], split: str) -> Plan:
    worst = gen_holds(prices, rates)
    attempts = lives * generate.MAX_ATTEMPTS
    return Plan(f"generate {lives} {split} li{'fe' if lives == 1 else 'ves'}", None, (
        Hold(SPARK_POOL, worst[SPARK_POOL], attempts, "Spark Open per attempt: writer, extractor, gate readers"),
        Hold("meta_usd", worst["meta_usd"], attempts, "Muse Spark per attempt: the gates' auditor"),
    ), (f"at most {generate.MAX_ATTEMPTS} attempts a life; each attempt held before it starts",))


def run_plan(lives: int, arms: Sequence[str], prices: Prices, session_s: float) -> Plan:
    runs = lives * len(arms)
    return Plan(f"run {', '.join(arms)} on {lives} li{'fe' if lives == 1 else 'ves'}", session_s, (
        _run_spark(prices, runs, "Spark Open per life-arm night: screen and resolver"),
    ), ("a run already on disk is skipped",))


def timing_plan(prices: Prices, session_s: float) -> Plan:
    return Plan("timing: B and M on one development life, 17b's labels and branches, a training dry run",
                session_s, (
                    _run_spark(prices, 2, "Spark Open per night of B's and M's runs"),
                    Hold(SPARK_POOL, spark_worst_usd(prices, 1), 0, "Spark Open per start state's label, held first"),
                    Hold(SPARK_POOL, spark_worst_usd(prices, MIND_SPARK_CALLS), 0,
                         "Spark Open per branch's resolver and screen, held per start state first"),
                ), ("the training is planned only: no training job runs",))


def judge_plan(rates: Mapping[str, Rate]) -> Plan:
    return Plan("judge every run on disk, blind", None, tuple(
        Hold(POOLS[name], rates[name].worst, 0, f"{name} per new prompt (a prompt judged before costs nothing)")
        for name in (MUSE, GEMINI)
    ), ("disagreements go to the blind audit file for Claude; nothing else is called",))


def kill_plan(prices: Prices, session_s: float, lives: int = KILL_RESUME_LIVES) -> Plan:
    return Plan(f"kill-resume on {lives} held-out lives", session_s, (
        Hold(SPARK_POOL, 3 * LIFE_DAYS * spark_worst_usd(prices, MIND_SPARK_CALLS), lives,
             "Spark Open per life, its uninterrupted, killed and resumed runs' every night, settled in full"),
    ), ("the child processes hold nothing themselves: each life's hold covers them",))


def describe(plan: Plan) -> str:
    lines = [f"Plan: {plan.command}."]
    if plan.session_s is None:
        lines.append("Colab: no session.")
    else:
        hold = colab_vllm.hold_amount(plan.session_s)
        lines.append(f"Colab: one A100 session ({MIND_SESSION_PREFIX}{colab_vllm.SESSION_LABEL}<random>), hard timeout "
                     f"{plan.session_s / 3600:.1f} h, held first on colab_cu: {hold:.1f} CU "
                     f"({A100_CU_PER_HOUR} CU/h); every CLI call and the tunnel under their own timeouts.")
    for hold_ in plan.holds:
        unit = "CU" if hold_.pool == "colab_cu" else "$"
        amount = f"{hold_.each:.1f} CU" if unit == "CU" else f"${hold_.each:.2f}"
        times = f" x {hold_.count} (one at a time; at most ${hold_.each * hold_.count:.2f} in all)" if hold_.count else ""
        lines.append(f"Hold on {hold_.pool}: {amount} each{times}: {hold_.what}.")
    lines.extend(f"Note: {note}." for note in plan.notes)
    lines.append("Nothing has run. Add --yes to start it.")
    return "\n".join(lines)
