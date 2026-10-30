"""What a live APD command would start and hold (Task 17), printed by ``--live`` before anything
runs: its Colab session and hard timeout, and each hold it takes on each pool, priced at the list
prices given, Spark Open's on its provider's pool (``mind.spark_provider``). Beside each worst
case it shows the expected spend when a timing has been measured (``timing.json``), and beside the
holds the command's ceiling (``--max-usd``, ``ceiling``) when it takes one. ``generate`` and
``generate-pairs`` also print their worst case in all and their expected cost, in all and on each
pool (the writer's, the reader's and Muse Spark's): each life's (or pair's) calls in one attempt
that passes, counted from its own items as the gates read them and priced by each call's token
shape (``costs``) at its model's prices, and say when a pool's part passes its whole cap. Printing a
plan spends nothing and reaches nothing."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from anjo.sleep.budget import CAPS, Pool
from anjo.sleep.mind.apd import colab_vllm, costs, generate
from anjo.sleep.mind.apd.backend import (
    AUDITOR_ROOM_TOKENS,
    AUDITOR_ROOMY_ASKS,
    GEMINI_THINKING,
    GEN_SPARK_RETRIES,
    GenCalls,
    gen_holds,
    life_calls,
)
from anjo.sleep.mind.apd.ceiling import dollars
from anjo.sleep.mind.apd.intervene import EDIT_SPARK_CALLS
from anjo.sleep.mind.apd.interventions import EDIT_KINDS
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, POOLS, Rate
from anjo.sleep.mind.apd.generate import LifeSpec
from anjo.sleep.mind.apd.generators import SPARK, Generator
from anjo.sleep.mind.apd.pairs import LAST_DAY, OUTCOME_DAYS, PAIR_EXPECTED, PAIR_ITEMS, pair_calls
from anjo.sleep.mind.apd.readers import SPARK_READER, Reader
from anjo.sleep.mind.apd.resume import KILL_RUNS
from anjo.sleep.mind.apd.timing import Timing, cu, kill_cu
from anjo.sleep.mind.apd.types import INTERVENTION_SNAPSHOTS, KILL_RESUME_LIVES, LIFE_DAYS, WORKSPACE_ARMS
from anjo.sleep.mind.spark_provider import SETTING_ENV, SparkProvider
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_worst_usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, MIND_SESSION_PREFIX

NIGHT_WHAT: Final = "resolver, held at a night's first call and settled at what it priced"


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
    warnings: tuple[str, ...] = ()  # printed last, before nothing runs
    totals: tuple[str, ...] = ()  # the worst case in all and the expected cost, when the plan prints them


def _night(provider: SparkProvider, prices: Prices, nights: int, what: str, timing: Timing | None) -> Hold:
    return Hold(provider.pool, spark_worst_usd(prices, MIND_SPARK_CALLS), nights, what,
                None if timing is None else timing.spark_per_mind_night)


def _spark(provider: SparkProvider) -> str:
    """The note naming where the plan's Spark Open runs and is held."""
    return f"Spark Open on {provider.label} ({provider.model}), held on {provider.pool}, as {SETTING_ENV} sets it"


def _session_cu(timing: Timing | None, nights: int) -> float | None:
    return None if timing is None else cu(timing.setup_s) + nights * timing.cu_per_night


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _lives(count: int) -> str:
    return f"{count} li{'fe' if count == 1 else 'ves'}"


def surface_days(spec: LifeSpec) -> tuple[int, ...]:
    """The day index each planted item of ``spec``'s first attempt surfaces on."""
    return tuple(item.surface_day for item in generate.plan(spec.split, spec.field, spec.families, spec.prevalence,
                                                            spec.seed))


@dataclass(frozen=True)
class Pricing:
    """Who writes and reads a generation command's lives, at what prices, on which pools."""

    writer: str
    write: Rate
    write_pool: Pool
    reader: str
    read: Rate
    read_pool: Pool
    muse: Rate
    thinks: bool  # the reader is billed for its thoughts (Gemini)
    day_retries: int = 0  # the more times the writer is asked for a day it answered unreadably


def pricing(prices: Prices, rates: Mapping[str, Rate], provider: SparkProvider, generator: Generator = SPARK,
            generator_rate: Rate | None = None, reader: Reader = SPARK_READER,
            reader_rate: Rate | None = None) -> Pricing:
    spark = Rate(prices.spark_in_per_m, prices.spark_out_per_m)
    write = spark if generator.model is None or generator_rate is None else generator_rate
    read = spark if reader.model is None or reader_rate is None else reader_rate
    return Pricing("Spark Open" if generator.model is None else generator.name, write, provider.pool,
                   "Spark Open" if reader.model is None else f"Gemini ({reader.model})", read,
                   reader.pool_on(provider), rates[MUSE], reader.model is not None,
                   generate.day_retries(generator.writes_with(provider)))


def expected(tokens: costs.AttemptTokens, on: Pricing) -> dict[Pool, float]:
    """One attempt that passes, priced by its calls' shapes on each pool it uses."""
    found: dict[Pool, float] = {on.write_pool: tokens.writes.usd(on.write)}
    found[on.read_pool] = found.get(on.read_pool, 0.0) + tokens.reads.usd(on.read)
    found["meta_usd"] = tokens.muse.usd(on.muse)
    return found


def life_expected(surfaces: Sequence[int], on: Pricing) -> dict[Pool, float]:
    """One life whose items surface on ``surfaces``, one attempt that passes, on each pool."""
    return expected(costs.life_tokens(tuple(surfaces), thinks=on.thinks), on)


def _worst(on: Pricing, prices: Prices, rates: Mapping[str, Rate], calls: GenCalls) -> dict[Pool, float]:
    return gen_holds(prices, rates, on.write_pool, calls, on.write, read_rate=on.read, read_pool=on.read_pool)


def _pools(on: Pricing) -> tuple[Pool, ...]:
    return tuple(dict.fromkeys((on.write_pool, on.read_pool, "meta_usd")))


def _who(on: Pricing, pool: Pool) -> str:
    asked = f", each day asked up to {on.day_retries} more times" if on.day_retries else ""
    parts = [*(["writer and extractor (" + on.writer + ")" + asked] if pool == on.write_pool else []),
             *(["gate readers (" + on.reader + ")"] if pool == on.read_pool else []),
             *(["the gates' auditor (Muse Spark)"] if pool == "meta_usd" else [])]
    extra = f", and {GEN_SPARK_RETRIES} extra tries" if pool != "meta_usd" else (
        f", and {AUDITOR_ROOMY_ASKS} ask with room for {AUDITOR_ROOM_TOKENS:,} tokens for a reply cut at its cap")
    return " and ".join(parts) + extra


def _shapes(on: Pricing) -> str:
    thoughts = f", {costs.GEMINI_THOUGHTS} thought tokens a Gemini read" if on.thinks else ""
    return (f"expected by each call's token shape (costs): a day of {costs.DAY_TOKENS:,} tokens, the size that makes "
            f"the life measured on 2026-10-28 cost its ${costs.MEASURED_LIFE_USD:.2f}{thoughts}; replies' sizes "
            "assumed; `calls` prints the real shapes once a life is logged")


def _per_pool(on: Pricing, worst: Mapping[Pool, float], each: Sequence[Mapping[Pool, float]], what: str) -> tuple[str, ...]:
    lines = []
    for pool in _pools(on):
        found = [one.get(pool, 0.0) for one in each]
        lines.append(f"On {pool}: worst case ${worst.get(pool, 0.0):,.2f}; expected about ${sum(found):,.2f} for "
                     f"{what}, ${min(found, default=0.0):,.2f} to ${max(found, default=0.0):,.2f} each.")
        if sum(found) > CAPS[pool]:
            lines.append(f"Over the cap: about ${sum(found):,.2f} of it is on {pool}, more than its whole "
                         f"${CAPS[pool]:,.2f} cap; the run stops once the pool is full (BudgetStop), and a rerun goes "
                         "on from what was saved.")
    return tuple(lines)


def _writer_note(generator: Generator, on: Pricing) -> tuple[str, ...]:
    if generator.model is None:
        return ()
    effort = "" if generator.reasoning_effort is None else (
        f", reasoning effort {generator.reasoning_effort}, any reasoning held and settled as output but not in the "
        "expected")
    trains = "" if generator.trains is None else f"; its lives may train the core: {generator.trains}"
    return (f"the writer and extractor are {generator.name} on Fireworks ({generator.model}{effort}) at "
            f"${on.write.per_m_in:g}/${on.write.per_m_out:g} per 1M tokens{trains}",)


def _reader_note(reader: Reader, on: Pricing) -> tuple[str, ...]:
    if reader.model is None:
        return ("the gates read on Spark Open",)
    return (f"the gates read on Gemini ({reader.model}, Vertex, thinking {GEMINI_THINKING}) at "
            f"${on.read.per_m_in:g}/${on.read.per_m_out:g} per 1M tokens, held on {on.read_pool}: verdicts only, "
            "never a life's text",)


def pairs_plan(count: int, prices: Prices, rates: Mapping[str, Rate], *, provider: SparkProvider,
               generator: Generator = SPARK, generator_rate: Rate | None = None, reader: Reader = SPARK_READER,
               reader_rate: Rate | None = None) -> Plan:
    """Generating ``count`` counterfactual pairs, each attempt held for both histories."""
    on = pricing(prices, rates, provider, generator, generator_rate, reader, reader_rate)
    worst = _worst(on, prices, rates, pair_calls(on.day_retries))
    attempts = count * generate.MAX_ATTEMPTS
    holds = tuple(Hold(pool, worst[pool], attempts, f"{pool} per pair attempt: " + (
        f"{LIFE_DAYS} days written and extracted once and each history's {len(OUTCOME_DAYS)} outcomes extracted, "
        if pool == on.write_pool else "") + _who(on, pool)) for pool in _pools(on))
    total = sum(hold.each * hold.count for hold in holds)
    pairs = f"{count} pair{'' if count == 1 else 's'}"
    one = expected(costs.pair_tokens((LAST_DAY,) * PAIR_ITEMS, len(OUTCOME_DAYS), thinks=on.thinks), on)
    worst_all = {pool: worst[pool] * attempts for pool in _pools(on)}
    return Plan(f"generate {pairs} of histories in core-test's fields", None, holds, (
        _spark(provider), *_writer_note(generator, on), *_reader_note(reader, on),
        f"at most {generate.MAX_ATTEMPTS} attempts a pair; both histories pass the gates or the pair is written again",
        "a pair that still fails is skipped and recorded, and the run goes on; a rerun generates only what is missing",
    ), totals=(
        f"Worst case: ${total:,.2f} for {pairs}: every pair taking all {generate.MAX_ATTEMPTS} attempts, every call "
        "at its cap and every extra try used.",
        f"Expected: about ${count * sum(one.values()):,.2f} for {pairs}, one attempt each: "
        f"${sum(one.values()):,.2f} a pair, {PAIR_EXPECTED.spark + PAIR_EXPECTED.muse} calls, {_shapes(on)}.",
        *_per_pool(on, worst_all, [one] * count, pairs),
    ))


def generate_plan(lives: Sequence[Sequence[int]], prices: Prices, rates: Mapping[str, Rate], split: str, *,
                  provider: SparkProvider, generator: Generator = SPARK, generator_rate: Rate | None = None,
                  reader: Reader = SPARK_READER, reader_rate: Rate | None = None) -> Plan:
    """Generating one life per entry of ``lives``, each its items' surface days: each attempt held
    for its own items, and expected by its own calls' shapes (``costs``) on each pool."""
    count, on = len(lives), pricing(prices, rates, provider, generator, generator_rate, reader, reader_rate)
    worst = [_worst(on, prices, rates, life_calls(len(items), day_retries=on.day_retries)) for items in lives]
    each = [life_expected(items, on) for items in lives]
    attempts = count * generate.MAX_ATTEMPTS
    sized = "sized to its life's planted items; the mean shown"
    holds = tuple(Hold(pool, _mean([w[pool] for w in worst]), attempts, f"{pool} per attempt: {_who(on, pool)}, {sized}")
                  for pool in _pools(on))
    total = sum(hold.each * hold.count for hold in holds)
    items = sorted(len(found) for found in lives) or [0]
    lives_usd = [sum(one.values()) for one in each]
    worst_all = {pool: sum(w[pool] for w in worst) * generate.MAX_ATTEMPTS for pool in _pools(on)}
    return Plan(f"generate {_lives(count)} of {split}", None, holds, (
        _spark(provider), *_writer_note(generator, on), *_reader_note(reader, on),
        f"at most {generate.MAX_ATTEMPTS} attempts a life; each attempt held before it starts, its calls sharing "
        f"{GEN_SPARK_RETRIES} extra tries for a transient failure",
        "a life that still fails is skipped and recorded, and the run goes on; a rerun generates only what is missing",
    ), totals=(
        f"Worst case: ${total:,.2f} for {_lives(count)}: every life taking all {generate.MAX_ATTEMPTS} attempts, every "
        "call at its cap and every extra try used.",
        f"Expected: about ${sum(lives_usd):,.2f} for {_lives(count)}, one attempt each: "
        f"${min(lives_usd, default=0.0):,.2f} to ${max(lives_usd, default=0.0):,.2f} a life by its planted items "
        f"({items[0]} to {items[-1]}), {_shapes(on)}.",
        *_per_pool(on, worst_all, each, _lives(count)),
    ))


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
                         "Spark Open per start state's branches (resolvers), one start state at a time"),
                    Hold(pool, spark_worst_usd(prices, EDIT_SPARK_CALLS), len(EDIT_KINDS),
                         "Spark Open per snapshot-edit of the intervention test"),
                ), (_spark(provider), "the training is planned only: no training job runs"))


def intervene_plan(prices: Prices, session_s: float, timing: Timing | None, *, provider: SparkProvider) -> Plan:
    items = INTERVENTION_SNAPSHOTS * len(EDIT_KINDS)
    expected = None if timing is None or not timing.intervene_items else timing.intervene_usd / timing.intervene_items
    return Plan(f"the intervention test: {INTERVENTION_SNAPSHOTS} snapshots x {len(EDIT_KINDS)} edits, one at a time",
                session_s, (
                    Hold(provider.pool, spark_worst_usd(prices, EDIT_SPARK_CALLS), items, expected=expected,
                         what="Spark Open per snapshot-edit: both twins' resolvers, and the wrong "
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
    lines.extend(_ceiling(plan))  # beside the holds
    lines.extend(plan.totals)
    lines.extend(f"Note: {note}." for note in plan.notes)
    lines.extend(f"Warning: {warning}." for warning in plan.warnings)
    lines.append("Nothing has run. Add --yes to start it.")
    return "\n".join(lines)
