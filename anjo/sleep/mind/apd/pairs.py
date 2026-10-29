"""Counterfactual autobiography pairs (MIND's data): two histories of one person that differ
only in the outcome of an earlier commitment, ending on the same last day word for word.

**One pair** (``write_pair``), in a ``core-test`` field, from a ``PairTemplate`` and one unrelated
``core-test`` template of the same field (``PairSpec``):

- every day is written and extracted **once**, in order, on the days before it, and added to both
  histories: the commitment on ``COMMIT_DAY``, the unrelated item's facts before ``FORK_DAY``
  (``prefix_plan``), and on the last day the person's question about the commitment and the
  unrelated item's surface. The ``OUTCOME_DAYS`` are written with no beat;
- each history then says its own outcome as one more message of each outcome day, the template's
  sentence word for word (``Draft.with_message``): the commitment failing twice, or kept twice. That
  message alone is extracted for its history, on what that history holds.

So the histories differ only in those messages, their evidence and the targets they mention, whose
state follows from the outcome; every other message, evidence item and target is the same, ids
included. No shared day is written knowing either outcome. The failed history plants the unrelated
item and the commitment's gap (``PairTemplate.gap``); the kept one plants the unrelated item alone.
Both histories pass the same gates as any life (``gates.check``) before the pair is kept, or the
pair is written again under the next attempt's seed. ``CounterfactualPair`` records what should
differ (the slot's decision: speak in the failed history, wait in the kept one) and what should
not (the unrelated item, the last day).

**Many pairs** (``generate_pairs``): as ``gen_batch.generate_lives``, side by side, written by the
backend's generator (``generators``) and read by its reader (``readers``), never mixed with
another's, each saved with its record and its calls, each attempt held before it
starts (``PAIR_CALLS``), a pair that still fails skipped and recorded, the ceiling checked before
each attempt, the run stopped once the network looks down (``gen_batch.Breaker``). Pairs are test
material: the core never trains on them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.apd import failures, gates, readers
from anjo.sleep.mind.apd.backend import GEN_READS_PER_ITEM, Backend, GenCalls, cool_down, held_attempt, item_reads
from anjo.sleep.mind.apd.batch import Out, check_failed
from anjo.sleep.mind.apd.calls import CallLog, failed_log_id
from anjo.sleep.mind.apd.core_families import PAIR_TEMPLATES, PairTemplate, template, templates_for, texts_digest
from anjo.sleep.mind.apd.families import Template
from anjo.sleep.mind.apd.gen_batch import (
    GEN_PARALLEL,
    SKIPPED,
    Breaker,
    CallFailed,
    failed_at,
    failing,
    kept_off,
    last_call,
    made_record,
    network_stop,
    recorder,
    refuse_changed,
    refuse_mixed,
    report_failures,
    side_by_side,
)
from anjo.sleep.mind.apd.generate import (
    AWAY_DAYS,
    MAX_ATTEMPTS,
    ORIGIN_GAP_DAYS,
    Dice,
    Draft,
    GenerationError,
    GenerationFailed,
    ItemPlan,
    attempt_seed,
    beats_of,
    day_retries,
    expressed,
    life_days,
    problem_file,
    ranked,
)
from anjo.sleep.mind.apd.generators import check
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import HALF_GAP_DAYS, LIFE_DAYS, CounterfactualPair, GateVerdict, Life

PAIR_LABEL: Final = "pairs:core-v1"
PAIR_COUNT: Final = 20
COMMIT_DAY: Final = 1
FORK_DAY: Final = 6  # the first day the histories may differ
OUTCOME_DAYS: Final = (6, 8)  # the commitment's two outcomes, one history's each
LAST_DAY: Final = LIFE_DAYS - 1
PAIR_SPARK_WRITES: Final = 2 * LIFE_DAYS + 2 * len(OUTCOME_DAYS)  # each day written and extracted, each outcome extracted
PAIR_ITEMS: Final = 3  # the unrelated item in both histories, the commitment's gap in the failed one
PAIR_CALLS: Final = GenCalls(PAIR_SPARK_WRITES, PAIR_ITEMS * GEN_READS_PER_ITEM, PAIR_ITEMS + 2)  # + both audits


def pair_calls(day_retries: int = 0) -> GenCalls:
    """``PAIR_CALLS`` and ``day_retries`` more writer calls a day (``generate.day_retries``)."""
    return dataclasses.replace(PAIR_CALLS, writes=PAIR_CALLS.writes + day_retries * LIFE_DAYS)


PAIR_EXPECTED: Final = GenCalls(PAIR_CALLS.writes, PAIR_ITEMS * item_reads(LAST_DAY), PAIR_CALLS.muse, retries=0)
FAILED_BEATS: Final = ("i2.o2", "i2.d")  # the gap's second origin and its disambiguating fact
KEPT_BEATS: Final = ("k.o", "k.d")


@dataclass(frozen=True)
class PairSpec:
    """One pair to generate: its pair template, its unrelated template and its seed."""

    template: str
    unrelated: str
    seed: int


def pair_specs(count: int = PAIR_COUNT) -> tuple[PairSpec, ...]:
    """``count`` pairs, a pure function of ``PAIR_LABEL``: the pair templates in turn, each with an
    unrelated template of its field drawn by hash."""
    specs = []
    for n in range(count):
        pair = PAIR_TEMPLATES[n % len(PAIR_TEMPLATES)]
        unrelated = ranked(f"{PAIR_LABEL}:{n}", templates_for("core-test", pair.field), lambda t: t.id)[0]
        seed = int.from_bytes(hashlib.sha256(f"{PAIR_LABEL}:seed:{n}".encode()).digest()[:6], "big")
        specs.append(PairSpec(pair.id, unrelated.id, seed))
    return tuple(specs)


def pair_template(template_id: str) -> PairTemplate:
    found = next((pair for pair in PAIR_TEMPLATES if pair.id == template_id), None)
    if found is None:
        raise ValueError(f"no pair template {template_id!r}")
    return found


def pair_id(field: str, seed: int) -> str:
    return f"pair-{re.sub(r'[^a-z0-9]+', '-', field.lower()).strip('-')}-{seed}"


def spec_pair_ids(spec: PairSpec) -> frozenset[str]:
    """The ids ``spec``'s pair may take, one per attempt."""
    field = pair_template(spec.template).field
    return frozenset(pair_id(field, attempt_seed(spec.seed, n)) for n in range(MAX_ATTEMPTS))


def prefix_plan(item: Template, dice: Dice) -> ItemPlan:
    """The unrelated item's days, all before the fork: origins (``HALF_GAP_DAYS`` apart for a
    contradiction), the disambiguating fact ``AWAY_DAYS`` after, the slack spread by the dice;
    surfaced on the last day."""
    gap = HALF_GAP_DAYS if item.family == "contradiction" else ORIGIN_GAP_DAYS
    steps = [*([gap] if len(item.origins) == 2 else []), AWAY_DAYS]
    slack = FORK_DAY - 1 - sum(steps)
    at = dice.below(slack + 1)
    slack -= at
    days = [at]
    for step in steps:
        extra = dice.below(slack + 1)
        slack -= extra
        at += step + extra
        days.append(at)
    return ItemPlan(item, tuple(days[:-1]), days[-1], LAST_DAY)


# ── Writing one pair ──────────────────────────────────────────────────────────


def _ids(life: Life, spots: dict[str, str], beat: str, fact: str) -> tuple[str, ...]:
    found = [item.id for item in life.evidence if not item.held_out and item.source == spots[beat]]
    return expressed(fact, found, {item.id: item.text for item in life.evidence})


def write_pair(spec: PairSpec, seed: int, *, chat: ChatFn, model: str = SPARK_OPEN,
               keep_off: bool = False) -> CounterfactualPair:
    """One pair under ``seed`` (the spec's seed, or an attempt's), not yet gated; with ``keep_off``,
    each write is given its two items' topics (``generate.KEEP_OFF``)."""
    pair, unrelated = pair_template(spec.template), template(spec.unrelated)
    if unrelated is None or unrelated not in templates_for("core-test", pair.field):
        raise ValueError(f"{spec.unrelated!r} isn't a core-test template of {pair.field!r}")
    plans = (prefix_plan(unrelated, Dice(f"{PAIR_LABEL}:{seed}")),
             ItemPlan(pair.gap, (COMMIT_DAY, OUTCOME_DAYS[0]), OUTCOME_DAYS[1], LAST_DAY))
    beats = beats_of(plans)  # i1: the unrelated item; i2: the commitment's gap, its outcomes the failed ones
    topics = tuple(item.template.topic for item in plans) if keep_off else ()
    shared = failed = kept = Draft(pair_id(pair.field, seed), pair.field, seed, model, keep_off=topics)
    for index, day in enumerate(life_days()):
        done = shared.write(chat, index, day, [] if index in OUTCOME_DAYS else beats.get(index, []))
        shared, failed, kept = shared.add(done), failed.add(done), kept.add(done)
        if index in OUTCOME_DAYS:
            n = OUTCOME_DAYS.index(index)
            failed = failed.with_message(chat, pair.failed[n], FAILED_BEATS[n], branch="f")
            kept = kept.with_message(chat, pair.kept[n], KEPT_BEATS[n], branch="k")
    return _pair(spec, plans, failed, kept)


def _pair(spec: PairSpec, plans: Sequence[ItemPlan], failed: Draft, kept: Draft) -> CounterfactualPair:
    pair = pair_template(spec.template)
    failed_life, kept_life = failed.life("core-test"), kept.life("core-test")
    failed_spots, kept_spots = failed.spots, kept.spots
    kept_ids = tuple(i for beat, fact in zip(KEPT_BEATS, pair.kept) for i in _ids(kept_life, kept_spots, beat, fact))
    failed_ids = tuple(i for beat, fact in zip(FAILED_BEATS, pair.failed)
                       for i in _ids(failed_life, failed_spots, beat, fact))
    failed_problems = problem_file(failed_life, plans, failed_spots)
    try:
        return CounterfactualPair(
            id=failed_life.id, template=pair.id, fork_days=tuple(failed_life.days[d] for d in OUTCOME_DAYS),
            failed=failed_life, failed_problems=failed_problems, kept=kept_life,
            kept_problems=problem_file(kept_life, plans[:1], kept_spots),
            slot_ids=_ids(failed_life, failed_spots, "i2.o1", pair.commitment),
            failed_ids=failed_ids,
            kept_ids=kept_ids, unrelated_item=failed_problems.items[0].id,
            final_ids=tuple(e.id for e in failed_life.evidence if e.day == failed_life.days[-1] and not e.held_out),
        )
    except ValueError as error:  # an outcome written the same in both histories, or none left on the last day
        raise GenerationFailed(f"the two histories aren't a pair: {error}", step="pair") from None


# ── Gated, held and many ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeneratedPair:
    pair: CounterfactualPair
    attempts: int
    verdicts: tuple[GateVerdict, ...]


def generate_pair(spec: PairSpec, backend: Backend, log: CallLog | None = None,
                  breaker: Breaker | None = None) -> GeneratedPair:
    """``write_pair``'s attempts, each held before it starts, written by the backend's generator,
    both histories gated by its reader, every try logged to ``log`` when given. With a ``breaker``,
    as ``gen_batch.generate_life``: how each attempt ended is counted, and no attempt starts, nor is
    the pair skipped, once it has tripped."""
    model, writes = backend.read_by.reads_with(backend.provider), backend.generator.writes_with(backend.provider)
    breaker = Breaker(limit=None) if breaker is None else breaker
    on_call: CallFailed | None = None  # the last attempt ended by a call
    for attempt in range(MAX_ATTEMPTS):
        breaker.check()
        if on_call is not None:
            cool_down(backend)
            breaker.check()
            on_call = None
        if backend.ceiling is not None:
            backend.ceiling.check()
        if log is not None:
            log.attempt = attempt
        try:
            with failing(log, attempt), held_attempt(backend, f"pairs:{spec.seed}:{attempt}",
                                                     pair_calls(day_retries(writes)), log) as (writer, reader, auditor):
                try:
                    made = write_pair(spec, attempt_seed(spec.seed, attempt), chat=writer, model=writes,
                                      keep_off=backend.keep_off)
                except GenerationFailed as failed:
                    failed_at(log, failures.of_rule(attempt, failed))
                    breaker.ended(None)
                    continue
                verdicts = (*gates.check(made.failed, made.failed_problems, reader=reader, auditor=auditor,
                                         model=model),
                            *gates.check(made.kept, made.kept_problems, reader=reader, auditor=auditor, model=model))
        except CallFailed as failed:
            breaker.ended(failed)
            on_call = failed
            continue
        except SKIPPED as error:
            breaker.ended(error)
            raise
        breaker.ended(None)
        if gates.passed(verdicts):
            return GeneratedPair(made, attempt + 1, verdicts)
        failed_at(log, *failures.of_verdicts(attempt, made.failed, verdicts))
    breaker.check()
    raise GenerationError(f"no pair passed the gates in {MAX_ATTEMPTS} attempts{last_call(on_call)}")


@dataclass(frozen=True)
class _Skip:
    spec: PairSpec
    error: str


@dataclass(frozen=True)
class _Outcome:
    spec: PairSpec
    found: GeneratedPair | _Skip
    failed: tuple[failures.AttemptFailure, ...]


def _name(spec: PairSpec) -> str:
    return f"pair {spec.seed} ({spec.template})"


def _one(data: DataDir, spec: PairSpec, backend: Backend, *, breaker: Breaker | None = None) -> _Outcome:
    log, made = CallLog(on_failure=recorder(data, "generate-pairs-attempt", _name(spec))), None
    try:
        made = generate_pair(spec, backend, log, breaker)
    except SKIPPED as error:
        return _Outcome(spec, _Skip(spec, str(error) or type(error).__name__), tuple(log.failures))
    finally:
        if made is None and log.calls:  # skipped, killed or stopped (the ceiling, a hold): what it made still cost
            first = pair_id(pair_template(spec.template).field, spec.seed)
            data.save_calls(failed_log_id(first, data.call_log_ids()), [call.to_json() for call in log.calls])
    data.save_made(made.pair.id, {**made_record(backend), "templates": texts_digest("core-test", made.pair.failed.field)})
    data.save_calls(made.pair.id, [call.to_json() for call in log.calls])
    data.save_pair(made.pair)
    return _Outcome(spec, made, tuple(log.failures))


@dataclass(frozen=True)
class PairsKeepOff:
    """Whether new pairs' writer is given their topics (``on``), following what (``source``: the
    pairs on disk, the core-test majority or the default) and its counts (``detail``); ``mixed``
    when the pairs on disk disagree, and what the plan warns of."""

    on: bool
    source: str
    detail: str
    mixed: bool = False
    warnings: tuple[str, ...] = ()


def pairs_keep_off(data: DataDir) -> PairsKeepOff:
    """New pairs follow the pairs on disk (a record with no ``keep_off`` reads as off), so pairs
    never mix; with none on disk, most ``core-test`` lives on disk (the pairs are core-test's; the
    other splits' lives don't count); with neither, or a tie, they are written without it."""
    flags = [bool((data.made(pair.id) or {}).get("keep_off")) for pair in data.pairs()]
    lives = kept_off(data, "core-test")
    if not flags:
        if lives.majority is not None:
            return PairsKeepOff(lives.majority, "the core-test majority", str(lives))
        return PairsKeepOff(False, "the default", "no pairs or core-test lives on disk" if lives.without == 0
                            else f"no pairs on disk, and a tie: {lives}")
    with_ = sum(flags)
    detail = f"pairs on disk: {len(flags) - with_} without keep_off, {with_} with"
    if 0 < with_ < len(flags):
        return PairsKeepOff(False, "the pairs on disk", detail, mixed=True, warnings=(
            f"the pairs on disk are mixed on keep_off ({detail}): a run with pairs to write refuses",))
    on = with_ > 0
    lean = () if lives.majority in (None, on) else (
        f"new pairs follow the pairs on disk, {'with' if on else 'without'} keep_off, though most core-test lives "
        f"were written {'without' if on else 'with'} it ({lives})",)
    return PairsKeepOff(on, "the pairs on disk", detail, warnings=lean)


def generate_pairs(data: DataDir, specs: Sequence[PairSpec], backend: Backend, out: Out, *,
                   parallel: int = GEN_PARALLEL) -> tuple[str, ...]:
    """Every pair of ``specs`` not yet on disk, ``parallel`` at a time, generated, gated and saved;
    a pair that fails is skipped and recorded, and ``RunsFailed`` names them at the end. A pair's
    writer is given its topics (``keep_off``) as ``pairs_keep_off`` says, its made record saying so;
    pairs on disk that disagree on it refuse a run with pairs to write. The network looking down
    stops the run (``gen_batch.Breaker``, ``network_stop``)."""
    check("core-test", backend.generator, backend.provider)
    keep_off = pairs_keep_off(data)
    backend = dataclasses.replace(backend, keep_off=keep_off.on)
    readers.check("core-test", backend.read_by)
    on_disk = data.pairs()
    refuse_mixed((life for pair in on_disk for life in (pair.failed, pair.kept)), backend, "the pairs", data,
                 [pair.id for pair in on_disk])
    refuse_changed(((pair.id, pair.failed) for pair in on_disk), data, "the pairs")
    todo = [spec for spec in specs if not any(data.has_pair(found) for found in spec_pair_ids(spec))]
    if todo and keep_off.mixed:
        raise StoreError(f"the pairs on disk are mixed on keep_off ({keep_off.detail}), and a new pair would mix "
                         "them further; move the pairs written otherwise out of the data dir, or start a new one")
    if backend.ceiling is not None:
        backend.ceiling.begin("pair(s) generated", len(todo))
    made: list[str] = []
    skipped: list[str] = []
    breaker = Breaker()  # one for the run, as generate_lives has
    work = [partial(_one, data, spec, breaker=breaker) for spec in todo]
    with network_stop(out, skipped), side_by_side(backend, work, parallel) as results:
        for outcome in results:
            found = outcome.found
            report_failures(out, _name(outcome.spec), outcome.failed)
            if isinstance(found, _Skip):
                name = _name(found.spec)
                data.record_failed({"what": "generate-pairs", "pair": name, "error": found.error})
                out(f"SKIPPED {name}: {found.error}. Not saved; a rerun generates it again")
                skipped.append(name)
                continue
            made.append(found.pair.id)
            out(f"generated {found.pair.id} ({found.attempts} attempt(s), {found.pair.template})")
            if backend.ceiling is not None:
                out(backend.ceiling.life_generated("pair"))
    check_failed(skipped)
    return tuple(made)
