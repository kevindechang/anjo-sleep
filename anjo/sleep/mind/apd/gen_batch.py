"""APD's lives generated side by side, each failure alone (Task 17; MIND's data).

- **One life** (``generate_life``): ``generate.generate``'s attempts, written by the backend's
  generator (``generators``) and read by its reader (``readers``; a split never mixes two of
  either), saved with its record of both (``made``) and its calls (``calls``), each attempt's worst case,
  sized to the life's planted items, held before it starts (``backend.held_attempt``), after the
  run's ceiling is checked. A
  transient Spark Open failure is tried again inside the attempt's own hold (``backend``).
- **Side by side** (``generate_lives``): ``parallel`` lives at once, each on a thread of its own
  and under its own seed, so a life is the same whatever runs beside it. Each is saved as it
  ends; its line is printed in the specs' order, once it and every life before it are done.
- **Why each attempt failed** (``failures``): the step, day and item, the error's class or the
  gate's verdict, and for an unreadable reply its field, length and finish reason, never its
  text: recorded in ``failed.jsonl`` (``generate-attempt``) the moment the attempt fails, so a
  killed run keeps it (``recorder``), kept in the call log (a skipped life's too,
  ``calls.failed_log_id``) and printed, one line each, once the life ends, passed or not.
- **A call that still fails on a transient failure** (a 429, a 5xx, a timeout or a reset, after
  its tries: ``retry``) ends its attempt, not its life (``CallFailed``); the next attempt starts
  after a cool-down (``backend.cool_down``), and the attempt's 429s are kept with its failure.
- **A life that still fails is skipped:** every attempt failed a rule, a gate or a call
  (``GenerationError``), a model call failed for good (``ModelApiError``: a 400, say), or a reader's
  reply was unreadable (``gates.GateError``). It is recorded (``DataDir.record_failed``), the run
  goes on, and at its end ``live.RunsFailed`` names every skipped life; a rerun generates only
  those (a life on disk is never generated again).
- **What stops new work:** the run's ceiling (``CeilingReached``), a hold with no room while no
  other attempt is in flight (``BudgetStop``; with one in flight it waits for it to end), a reply
  from the wrong model (``ModelMismatch``) or anything unexpected. No further life starts, the ones
  running finish and are saved and printed, and the error is raised.
- **A kill** (Ctrl-C, a signal's ``SystemExit``, any error out of the loop, ``Parallel``): no
  further life starts, the running ones make no further call or attempt, each settling its hold,
  and the command waits for every worker to end before MIND's lock is let go.

The threads share only what is safe to share: the ledger locks its file, the store writes each
file whole under its own name, and each attempt has its own counted chats.
"""
from __future__ import annotations

import dataclasses
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Final, TypeVar

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.apd import failures, gates, generate, readers, repair, retry
from anjo.sleep.mind.apd.backend import Backend, Flight, GenCalls, Halted, cool_down, held_attempt, life_calls
from anjo.sleep.mind.apd.calls import CallLog, failed_log_id
from anjo.sleep.mind.apd.core_families import texts_digest
from anjo.sleep.mind.apd.batch import Out, check_failed
from anjo.sleep.mind.apd.generate import Draft, GenerationError, GenerationFailed, Generated, ItemPlan, LifeSpec
from anjo.sleep.mind.apd.generators import check, mixed
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import CORE_SPLITS, CoreSplit, Life
from anjo.sleep.model_api import ChatResult, ModelApiError

GEN_PARALLEL: Final = 4  # lives generated at once
SKIPPED: Final = (GenerationError, ModelApiError, gates.GateError)  # a life failing with these is skipped
WORKER: Final = "apd-gen"  # the workers' thread names
T = TypeVar("T")


class _NotStarted(Exception):
    """An item that never began, since another failed first or the command is ending."""


class Parallel:
    """``parallel`` items at once, each on a thread of its own, for as long as the ``with`` lasts.

    - After an item fails (``stop``), no further item starts; the ones running finish.
    - When the ``with`` ends on an error (Ctrl-C, a signal's ``SystemExit``, anything raised in its
      body), ``killed`` is set: no further item starts, the running ones make no further call
      (``Backend.halt``), and the ``with`` lets go only once every worker has ended, a second Ctrl-C
      included, so MIND's lock is never released under a running worker.
    """

    def __init__(self, parallel: int) -> None:
        if parallel < 1:
            raise ValueError("--parallel is at least 1")
        self.stop, self.killed = threading.Event(), threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix=WORKER)

    def __enter__(self) -> Parallel:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        if kind is not None:
            self._kill()
        self._pool.shutdown(wait=False, cancel_futures=True)  # what has not started never will
        while True:
            try:
                self._pool.shutdown(wait=True)
            except (KeyboardInterrupt, SystemExit):  # again: the workers are still ending
                self._kill()
                continue
            return

    def _kill(self) -> None:
        self.killed.set()
        self.stop.set()

    def in_order(self, work: Sequence[Callable[[], T]]) -> Iterator[T]:
        """Every item started now, its result yielded in the items' order once it and every item
        before it are done; after a failure, the ones running finish and are yielded, and the first
        failure in order is raised at the end."""
        return _results([self._pool.submit(self._guarded, item) for item in work])

    def _guarded(self, item: Callable[[], T]) -> T:
        if self.stop.is_set():
            raise _NotStarted
        try:
            return item()
        except BaseException:
            self.stop.set()
            raise


def _results(futures: Sequence[Future[T]]) -> Iterator[T]:
    failed: BaseException | None = None
    for future in futures:
        try:
            result = future.result()
        except _NotStarted:
            continue
        except Exception as error:  # noqa: BLE001 - kept and raised once the running items finish
            failed = failed or error
            continue
        yield result
    if failed is not None:
        raise failed


@contextmanager
def side_by_side(backend: Backend, work: Sequence[Callable[[Backend], T]], parallel: int) -> Iterator[Iterator[T]]:
    """``work``'s results in order (``Parallel.in_order``), each item given ``backend`` with the
    pool's ``killed`` as its halt and one ``Flight`` for all their holds; the caller's loop goes
    inside the ``with``, so an error in it kills the pool."""
    with Parallel(parallel) as pool:
        running = dataclasses.replace(backend, halt=pool.killed, flight=Flight())
        yield pool.in_order([partial(item, running) for item in work])


# ── One life ──────────────────────────────────────────────────────────────────


Keep = Callable[[repair.Tried, int, int], None]  # a failed check's life kept, by attempt and round (``--keep-failed``)


def generate_life(spec: LifeSpec, backend: Backend, log: CallLog | None = None, keep: Keep | None = None) -> Generated:
    """``generate.generate``'s attempts, each attempt's worst case held before it starts: written by
    the backend's generator (``generators``), its gates read by its reader (``readers``), every
    try logged to ``log`` when given. With ``backend.repair``, an attempt failing only a leak, a
    missed origin day or the script audit first takes up to ``repair.ROUNDS`` repair rounds, each
    held as an attempt is; ``keep`` is handed each check that failed."""
    on_call: CallFailed | None = None  # the last attempt ended by a call
    for attempt in range(generate.MAX_ATTEMPTS):
        if on_call is not None:
            cool_down(backend)
            on_call = None
        if log is not None:
            log.attempt = attempt
        try:
            made = _attempt(spec, backend, log, attempt, keep)
        except CallFailed as failed:
            on_call = failed
            continue
        if made is not None:
            return made
    raise GenerationError(f"no {spec.split} life passed the gates in {generate.MAX_ATTEMPTS} attempts{last_call(on_call)}")


def _attempt(spec: LifeSpec, backend: Backend, log: CallLog | None, attempt: int, keep: Keep | None) -> Generated | None:
    """One attempt and its repair rounds: the life that passed, or ``None`` with why it failed
    recorded (its last check's verdicts, or the rule a draft broke)."""
    memo: dict[str, ChatResult] = {}  # the attempt's gate replies, by request, shared by its rounds
    seed, writes = generate.attempt_seed(spec.seed, attempt), backend.generator.writes_with(backend.provider)
    retries = generate.day_retries(writes)  # each day's asks again, held with the attempt and its rounds
    tried = _checked(spec, backend, log, attempt, 0, life_calls(spec.prevalence, day_retries=retries), memo,
                     lambda chat: generate.drafted(spec.field, spec.families, spec.prevalence, seed, split=spec.split,
                                                   chat=chat, model=writes, keep_off=backend.keep_off))
    round_ = 0
    while tried is not None and not tried.passed:
        if keep is not None:
            keep(tried, attempt, round_)
        found = tried.fixes() if backend.repair and round_ < repair.ROUNDS else None
        if found is None:
            failed_at(log, *failures.of_verdicts(attempt, tried.life, tried.verdicts))
            return None
        round_ += 1
        tried = _checked(spec, backend, log, attempt, round_,
                         repair.round_calls(spec.prevalence, found, day_retries=retries), memo,
                         partial(repair.rewritten, tried, found, round_))
    return None if tried is None else Generated(tried.life, tried.problems, attempt + 1, tried.verdicts, round_)


def _checked(spec: LifeSpec, backend: Backend, log: CallLog | None, attempt: int, round_: int, calls: GenCalls,
             memo: dict[str, ChatResult], make: Callable[[ChatFn], tuple[Draft, tuple[ItemPlan, ...]]]
             ) -> repair.Tried | None:
    """One hold's work, after the run's ceiling is checked: the draft ``make`` writes with the
    writer, its life gated, a gate request sent before in the attempt answered from ``memo``;
    ``None`` when the draft broke a rule (recorded)."""
    if backend.ceiling is not None:
        backend.ceiling.check()  # before the hold
    if log is not None:
        log.repair = round_
    key = f"{spec.split}:{spec.seed}:{attempt}{f':r{round_}' if round_ else ''}"
    with failing(log, attempt), held_attempt(backend, key, calls, log) as (writer, reader, auditor):
        try:
            draft, plans = make(writer)
            built, problems = generate.finished(draft, plans, spec.split)
        except GenerationFailed as failed:
            failed_at(log, failures.of_rule(attempt, failed))
            return None
        read, audit = repair.Memo(reader, memo), repair.Memo(auditor, memo)
        model = backend.read_by.reads_with(backend.provider)
        verdicts = gates.check(built, problems, reader=read, auditor=audit, model=model)
        audited = all(verdict.passed for verdict in verdicts if verdict.gate == "script_audit")
        issues = () if audited else gates.audit_issues(built, problems, audit)  # the audit's reply, from the memo
        if not backend.repair:
            return repair.Tried(draft, tuple(plans), built, problems, verdicts, issues)
        items = {item.id: item for item in problems.items}
        windows = {  # each flagged window's days, its single day's reply from the memo
            verdict.item_id: found for verdict in verdicts if verdict.item_id in items
            and (found := gates.flagged_window(built, items[verdict.item_id], verdict, read, model)) is not None}
        missed = {  # each origin day a discoverability reader passed over, its reply from the memo
            verdict.item_id: gates.missed_origin_days(built, items[verdict.item_id], read, model)
            for verdict in verdicts if verdict.item_id in items and verdict.reason == gates.MISSED_ORIGIN}
        return repair.Tried(draft, tuple(plans), built, problems, verdicts, issues, windows, missed)


class CallFailed(RuntimeError):
    """An attempt's call still failed on a transient failure after its tries: the attempt ends, and
    the next one starts after a cool-down."""


def last_call(failed: CallFailed | None) -> str:
    return "" if failed is None else f"; the last ended on a call: {failed}"


def failed_at(log: CallLog | None, *found: failures.AttemptFailure) -> None:
    """Keep why an attempt failed in ``log``, when there is one, with the attempt's 429s, and hand
    each to its ``on_failure`` at once."""
    if log is None:
        return
    count, waits = log.rate_limits()
    kept = [dataclasses.replace(failure, rate_limits=count, waits_s=waits) for failure in found]
    log.failures.extend(kept)
    if log.on_failure is not None:
        for failure in kept:
            log.on_failure(failure)


@contextmanager
def failing(log: CallLog | None, attempt: int) -> Iterator[None]:
    """An attempt ended by a call that still failed or a reply that couldn't be read: kept in
    ``log`` at its last call. A transient failure (``retry.transient``) ends the attempt
    (``CallFailed``); any other is raised on (the life is skipped). An attempt a kill cut short
    after its first call (``Halted``, ``Abandoned`` with a call in flight) is kept marked killed,
    and the kill raised on."""
    try:
        yield
    except Halted as error:
        last = None if log is None else log.last()
        if last is not None:
            failed_at(log, dataclasses.replace(failures.of_error(attempt, error, last), killed=True))
        raise
    except (ModelApiError, gates.GateError) as error:
        failed_at(log, failures.of_error(attempt, error, None if log is None else log.last()))
        if isinstance(error, ModelApiError) and retry.transient(error):
            raise CallFailed(str(error)) from error
        raise


def spec_ids(spec: LifeSpec) -> frozenset[str]:
    """The ids ``spec``'s life may take, one per attempt (``generate.life`` names a life by its
    split, field and attempt seed)."""
    field_ = re.sub(r"[^a-z0-9]+", "-", spec.field.lower()).strip("-")
    return frozenset(f"{spec.split}-{field_}-{generate.attempt_seed(spec.seed, n)}" for n in range(generate.MAX_ATTEMPTS))


def spec_name(spec: LifeSpec) -> str:
    """How a life is named before it exists: its split, field and seed."""
    return f"{spec.split} life {spec.seed} ({spec.field})"


# ── Many lives ────────────────────────────────────────────────────────────────


def _named(ids: Sequence[str]) -> str:
    return f"{', '.join(ids[:3])}{', ...' if len(ids) > 3 else ''}"


def refuse_mixed(lives: Iterable[Life], backend: Backend, where: str, data: DataDir | None = None,
                 made: Iterable[str] = ()) -> None:
    """``StoreError`` when another generator wrote any of ``lives``, or (given ``data``) another
    reader read any of them or of ``made`` (the ids their records are kept under): a mismatch,
    never a reuse."""
    found = list(lives)
    other = mixed(found, backend.generator.writes_with(backend.provider))
    if other:
        raise StoreError(f"{where} holds {len(other)} written by another generator than {backend.generator.name} "
                         f"({_named(other)}); one never mixes generators: go on with theirs, or start a new data dir")
    if data is None:
        return
    ids = list(made) or [life.id for life in found]
    records = {made_id: str(record.get("reader")) for made_id in ids if (record := data.made(made_id)) is not None}
    read = readers.mixed(ids, records, backend.read_by.name)
    if read:
        raise StoreError(f"{where} holds {len(read)} read by another reader than {backend.read_by.name} "
                         f"({_named(read)}); one never mixes readers: go on with theirs, or start a new data dir")


def refuse_changed(made: Iterable[tuple[str, Life]], data: DataDir, where: str) -> None:
    """``StoreError`` when any record on disk (``made``: the id it is kept under and a life it holds)
    was written from other template texts than its field's now (``core_families.texts_digest``): a
    mismatch, never a reuse. A record from before the digest was kept is not checked."""
    changed = [made_id for made_id, life in made
               if (kept := (data.made(made_id) or {}).get("templates")) is not None
               and kept != texts_digest(life.split, life.field)]
    if changed:
        raise StoreError(f"{where} holds {len(changed)} written from other template texts than their fields' now "
                         f"({', '.join(changed[:3])}{', ...' if len(changed) > 3 else ''}); one never mixes "
                         "template texts: go on with theirs, or start a new data dir")


def made_record(backend: Backend) -> dict[str, object]:
    """Who wrote and read a life or pair, and why its lives may train the MIND core (``None``: no
    clearance recorded for its writer); ``keep_off`` when its writer was given its topics
    (``--keep-off``), absent otherwise, as on every life made before it."""
    return {"writer": backend.generator.name, "writer_model": backend.generator.writes_with(backend.provider),
            "trains": backend.generator.trains, "reader": backend.read_by.name,
            "reader_model": backend.read_by.reads_with(backend.provider),
            **({"keep_off": True} if backend.keep_off else {})}


@dataclass(frozen=True)
class KeptOff:
    """How many core lives on disk (``of`` them) were written without ``--keep-off`` and with it
    (``made_record``)."""

    without: int
    with_: int
    of: str = "core lives"

    @property
    def majority(self) -> bool | None:
        """Whether most were written with it; ``None`` on a tie, none on disk included."""
        return None if self.without == self.with_ else self.with_ > self.without

    def __str__(self) -> str:
        return f"{self.of} on disk: {self.without} without keep_off, {self.with_} with"


def kept_off(data: DataDir, split: CoreSplit | None = None) -> KeptOff:
    """``data``'s core lives (``split``'s alone, when given), counted by whether their writer was
    given their topics."""
    splits = CORE_SPLITS if split is None else (split,)
    found = [bool((data.made(life.id) or {}).get("keep_off")) for core in splits for life in data.lives(core)]
    return KeptOff(found.count(False), found.count(True), "core lives" if split is None else f"{split} lives")


@dataclass(frozen=True)
class Skip:
    """A life that failed and was not saved, and why."""

    spec: LifeSpec
    error: str


@dataclass(frozen=True)
class Outcome:
    """A life generated or skipped, and why each of its failed attempts failed."""

    spec: LifeSpec
    found: Generated | Skip
    failed: tuple[failures.AttemptFailure, ...]


def _one(data: DataDir, spec: LifeSpec, backend: Backend) -> Outcome:
    log, made = CallLog(on_failure=recorder(data, "generate-attempt", spec_name(spec))), None
    keep = partial(quarantine, data) if backend.keep_failed else None
    try:
        made = generate_life(spec, backend, log, keep)
    except SKIPPED as error:
        return Outcome(spec, Skip(spec, str(error) or type(error).__name__), tuple(log.failures))
    finally:
        if made is None and log.calls:  # skipped, killed or stopped (the ceiling, a hold): what it made still cost
            first = generate.life_id_for(spec.split, spec.field, spec.seed)
            data.save_calls(failed_log_id(first, data.call_log_ids()), [call.to_json() for call in log.calls])
    data.save_made(made.life.id, {**made_record(backend), "templates": texts_digest(spec.split, spec.field),
                                  **({"repair_rounds": made.repaired} if made.repaired else {})})
    data.save_calls(made.life.id, [call.to_json() for call in log.calls])
    data.save_life(made.life, made.problems)
    return Outcome(spec, made, tuple(log.failures))


def quarantine(data: DataDir, tried: repair.Tried, attempt: int, round_: int) -> None:
    """A check that failed, its life kept apart under ``quarantine`` (``--keep-failed``), named by
    its round (``r0``: as first written)."""
    data.save_quarantined(f"{tried.life.id}.r{round_}", tried.life, tried.problems, tried.why(attempt, round_))


def recorder(data: DataDir, what: str, name: str) -> Callable[[failures.AttemptFailure], None]:
    """Each failed attempt recorded in ``failed.jsonl`` as it fails, so a run that is killed or
    stopped keeps why its attempts failed."""
    return lambda failure: data.record_failed({"what": what, "name": name, **failure.to_json()})


def report_failures(out: Out, name: str, found: Sequence[failures.AttemptFailure]) -> None:
    """Each failed attempt (recorded as it failed: ``recorder``) printed, one line each."""
    for failure in found:
        out(failures.line(name, failure))


def generate_lives(data: DataDir, specs: Sequence[LifeSpec], backend: Backend, out: Out, *,
                   parallel: int = GEN_PARALLEL) -> tuple[str, ...]:
    """Every life of ``specs`` not yet on disk, ``parallel`` at a time, generated, gated and saved,
    its problems sealed; lines in the specs' order, and with a ceiling what the run has settled per
    life after each. A life that fails is skipped and recorded; ``RunsFailed`` names them at the end."""
    for split in dict.fromkeys(spec.split for spec in specs):
        check(split, backend.generator, backend.provider)
        readers.check(split, backend.read_by)
        refuse_mixed(data.lives(split), backend, split, data)
        refuse_changed(((life.id, life) for life in data.lives(split)), data, split)
    todo = [spec for spec in specs if not any(data.has_life(life_id) for life_id in spec_ids(spec))]
    if backend.ceiling is not None:
        backend.ceiling.begin("life(s) generated", len(todo))
    made: list[str] = []
    skipped: list[str] = []
    with side_by_side(backend, [partial(_one, data, spec) for spec in todo], parallel) as results:
        for outcome in results:
            found = outcome.found
            report_failures(out, spec_name(outcome.spec), outcome.failed)
            if isinstance(found, Skip):
                name = spec_name(found.spec)
                data.record_failed({"what": "generate", "life": name, "error": found.error})
                out(f"SKIPPED {name}: {found.error}. Not saved; a rerun generates it again")
                skipped.append(name)
                continue
            made.append(found.life.id)
            rounds = f", {found.repaired} repair round(s)" if found.repaired else ""
            out(f"generated {found.life.id} ({found.attempts} attempt(s){rounds}, {len(found.problems.items)} planted)")
            if backend.ceiling is not None:
                out(backend.ceiling.life_generated())
    check_failed(skipped)
    return tuple(made)
