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
  text: recorded in ``failed.jsonl`` (``generate-attempt``), kept in the call log (a skipped life's
  too, ``calls.failed_log_id``) and printed, one line each, whether the life then passed or not.
- **A life that still fails is skipped:** every attempt failed a rule or a gate
  (``GenerationError``), a model call failed after its tries (``ModelApiError``), or a reader's
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

from anjo.sleep.mind.apd import failures, gates, generate, readers
from anjo.sleep.mind.apd.backend import Backend, Flight, held_attempt, life_calls
from anjo.sleep.mind.apd.calls import CallLog, failed_log_id
from anjo.sleep.mind.apd.batch import Out, check_failed
from anjo.sleep.mind.apd.generate import GenerationError, GenerationFailed, Generated, LifeSpec
from anjo.sleep.mind.apd.generators import check, mixed
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.model_api import ModelApiError

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


def generate_life(spec: LifeSpec, backend: Backend, log: CallLog | None = None) -> Generated:
    """``generate.generate``'s attempts, each attempt's worst case held before it starts: written by
    the backend's generator (``generators``), its gates read by its reader (``readers``), every
    try logged to ``log`` when given."""
    model, writes = backend.read_by.reads_with(backend.provider), backend.generator.writes_with(backend.provider)
    for attempt in range(generate.MAX_ATTEMPTS):
        if backend.ceiling is not None:
            backend.ceiling.check()  # before the attempt's hold
        if log is not None:
            log.attempt = attempt
        with failing(log, attempt), held_attempt(backend, f"{spec.split}:{spec.seed}:{attempt}",
                                                 life_calls(spec.prevalence), log) as (writer, reader, auditor):
            try:
                built, problems = generate.life(spec.field, spec.families, spec.prevalence,
                                                generate.attempt_seed(spec.seed, attempt), split=spec.split, chat=writer,
                                                model=writes)
            except GenerationFailed as failed:
                failed_at(log, failures.of_rule(attempt, failed))
                continue
            verdicts = gates.check(built, problems, reader=reader, auditor=auditor, model=model)
        if gates.passed(verdicts):
            return Generated(built, problems, attempt + 1, verdicts)
        failed_at(log, *failures.of_verdicts(attempt, built, verdicts))
    raise GenerationError(f"no {spec.split} life passed the gates in {generate.MAX_ATTEMPTS} attempts")


def failed_at(log: CallLog | None, *found: failures.AttemptFailure) -> None:
    """Keep why an attempt failed in ``log``, when there is one."""
    if log is not None:
        log.failures.extend(found)


@contextmanager
def failing(log: CallLog | None, attempt: int) -> Iterator[None]:
    """An attempt ended by a call that still failed or a reply that couldn't be read: kept in
    ``log`` at its last call, and raised on (the life is skipped)."""
    try:
        yield
    except (ModelApiError, gates.GateError) as error:
        failed_at(log, failures.of_error(attempt, error, None if log is None else log.last()))
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


def made_record(backend: Backend) -> dict[str, object]:
    """Who wrote and read a life or pair, and why its lives may train the MIND core (``None``: no
    clearance recorded for its writer)."""
    return {"writer": backend.generator.name, "writer_model": backend.generator.writes_with(backend.provider),
            "trains": backend.generator.trains, "reader": backend.read_by.name,
            "reader_model": backend.read_by.reads_with(backend.provider)}


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
    log, made = CallLog(), None
    try:
        made = generate_life(spec, backend, log)
    except SKIPPED as error:
        return Outcome(spec, Skip(spec, str(error) or type(error).__name__), tuple(log.failures))
    finally:
        if made is None and log.calls:  # skipped, killed or stopped (the ceiling, a hold): what it made still cost
            first = generate.life_id_for(spec.split, spec.field, spec.seed)
            data.save_calls(failed_log_id(first), [call.to_json() for call in log.calls])
    data.save_made(made.life.id, made_record(backend))
    data.save_calls(made.life.id, [call.to_json() for call in log.calls])
    data.save_life(made.life, made.problems)
    return Outcome(spec, made, tuple(log.failures))


def report_failures(data: DataDir, out: Out, what: str, name: str, found: Sequence[failures.AttemptFailure]) -> None:
    """Each failed attempt: recorded in ``failed.jsonl`` and printed, one line each."""
    for failure in found:
        data.record_failed({"what": what, "name": name, **failure.to_json()})
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
    todo = [spec for spec in specs if not any(data.has_life(life_id) for life_id in spec_ids(spec))]
    if backend.ceiling is not None:
        backend.ceiling.begin("life(s) generated", len(todo))
    made: list[str] = []
    skipped: list[str] = []
    with side_by_side(backend, [partial(_one, data, spec) for spec in todo], parallel) as results:
        for outcome in results:
            found = outcome.found
            report_failures(data, out, "generate-attempt", spec_name(outcome.spec), outcome.failed)
            if isinstance(found, Skip):
                name = spec_name(found.spec)
                data.record_failed({"what": "generate", "life": name, "error": found.error})
                out(f"SKIPPED {name}: {found.error}. Not saved; a rerun generates it again")
                skipped.append(name)
                continue
            made.append(found.life.id)
            out(f"generated {found.life.id} ({found.attempts} attempt(s), {len(found.problems.items)} planted)")
            if backend.ceiling is not None:
                out(backend.ceiling.life_generated())
    check_failed(skipped)
    return tuple(made)
