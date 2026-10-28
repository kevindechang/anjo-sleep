"""APD's lives generated side by side, each failure alone (Task 17; MIND's data).

- **One life** (``generate_life``): ``generate.generate``'s attempts, each attempt's worst case,
  sized to the life's planted items, held before it starts (``backend.held_attempt``), after the
  run's ceiling is checked. A
  transient Spark Open failure is tried again inside the attempt's own hold (``backend``).
- **Side by side** (``generate_lives``): ``parallel`` lives at once, each on a thread of its own
  and under its own seed, so a life is the same whatever runs beside it. Each is saved as it
  ends; its line is printed in the specs' order, once it and every life before it are done.
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
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Final, TypeVar

from anjo.sleep.mind.apd import gates, generate
from anjo.sleep.mind.apd.backend import Backend, Flight, held_attempt, life_calls
from anjo.sleep.mind.apd.batch import Out, check_failed
from anjo.sleep.mind.apd.generate import GenerationError, GenerationFailed, Generated, LifeSpec
from anjo.sleep.mind.apd.store import DataDir
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


def generate_life(spec: LifeSpec, backend: Backend) -> Generated:
    """``generate.generate``'s attempts, each attempt's worst case held before it starts, Spark Open
    on the backend's provider."""
    model = backend.provider.model
    for attempt in range(generate.MAX_ATTEMPTS):
        if backend.ceiling is not None:
            backend.ceiling.check()  # before the attempt's hold
        with held_attempt(backend, f"{spec.split}:{spec.seed}:{attempt}",
                          life_calls(spec.prevalence)) as (writer, reader, auditor):
            try:
                built, problems = generate.life(spec.field, spec.families, spec.prevalence,
                                                generate.attempt_seed(spec.seed, attempt), split=spec.split, chat=writer,
                                                model=model)
            except GenerationFailed:
                continue
            verdicts = gates.check(built, problems, reader=reader, auditor=auditor, model=model)
        if gates.passed(verdicts):
            return Generated(built, problems, attempt + 1, verdicts)
    raise GenerationError(f"no {spec.split} life passed the gates in {generate.MAX_ATTEMPTS} attempts")


def spec_ids(spec: LifeSpec) -> frozenset[str]:
    """The ids ``spec``'s life may take, one per attempt (``generate.life`` names a life by its
    split, field and attempt seed)."""
    field_ = re.sub(r"[^a-z0-9]+", "-", spec.field.lower()).strip("-")
    return frozenset(f"{spec.split}-{field_}-{generate.attempt_seed(spec.seed, n)}" for n in range(generate.MAX_ATTEMPTS))


def spec_name(spec: LifeSpec) -> str:
    """How a life is named before it exists: its split, field and seed."""
    return f"{spec.split} life {spec.seed} ({spec.field})"


# ── Many lives ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Skip:
    """A life that failed and was not saved, and why."""

    spec: LifeSpec
    error: str


def _one(data: DataDir, spec: LifeSpec, backend: Backend) -> Generated | Skip:
    try:
        made = generate_life(spec, backend)
    except SKIPPED as error:
        return Skip(spec, str(error) or type(error).__name__)
    data.save_life(made.life, made.problems)
    return made


def generate_lives(data: DataDir, specs: Sequence[LifeSpec], backend: Backend, out: Out, *,
                   parallel: int = GEN_PARALLEL) -> tuple[str, ...]:
    """Every life of ``specs`` not yet on disk, ``parallel`` at a time, generated, gated and saved,
    its problems sealed; lines in the specs' order, and with a ceiling what the run has settled per
    life after each. A life that fails is skipped and recorded; ``RunsFailed`` names them at the end."""
    todo = [spec for spec in specs if not any(data.has_life(life_id) for life_id in spec_ids(spec))]
    if backend.ceiling is not None:
        backend.ceiling.begin("life(s) generated", len(todo))
    made: list[str] = []
    skipped: list[str] = []
    with side_by_side(backend, [partial(_one, data, spec) for spec in todo], parallel) as results:
        for found in results:
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
