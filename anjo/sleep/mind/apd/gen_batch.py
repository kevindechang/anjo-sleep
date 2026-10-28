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
- **What stops new work:** the run's ceiling (``CeilingReached``), a hold with no room
  (``BudgetStop``), a reply from the wrong model (``ModelMismatch``) or anything unexpected. No
  further life starts, the ones running finish and are saved and printed, and the error is raised.

The threads share only what is safe to share: the ledger locks its file, the store writes each
file whole under its own name, and each attempt has its own counted chats.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Final, TypeVar

from anjo.sleep.mind.apd import gates, generate
from anjo.sleep.mind.apd.backend import Backend, held_attempt, life_calls
from anjo.sleep.mind.apd.batch import Out, check_failed
from anjo.sleep.mind.apd.generate import GenerationError, GenerationFailed, Generated, LifeSpec
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.model_api import ModelApiError

GEN_PARALLEL: Final = 4  # lives generated at once
SKIPPED: Final = (GenerationError, ModelApiError, gates.GateError)  # a life failing with these is skipped
T = TypeVar("T")


class _NotStarted(Exception):
    """An item that never began, since another failed first."""


def in_order(work: Sequence[Callable[[], T]], parallel: int) -> Iterator[T]:
    """Each item on a thread of its own, ``parallel`` at a time, its result yielded in the items'
    order once it and every item before it are done. After a failure no further item starts, the
    ones running finish and are yielded, and the first failure in order is raised at the end.
    When the caller stops (a signal's ``SystemExit``, Ctrl-C), nothing further starts."""
    if parallel < 1:
        raise ValueError("--parallel is at least 1")
    stop = threading.Event()

    def guarded(item: Callable[[], T]) -> T:
        if stop.is_set():
            raise _NotStarted
        try:
            return item()
        except BaseException:
            stop.set()
            raise

    pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="apd-gen")
    futures = [pool.submit(guarded, item) for item in work]
    failed: BaseException | None = None
    try:
        for future in futures:
            try:
                result = future.result()
            except _NotStarted:
                continue
            except Exception as error:  # noqa: BLE001 - kept and raised once the running items finish
                failed = failed or error
                continue
            yield result
    except BaseException:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    if failed is not None:
        raise failed


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
    for found in in_order([partial(_one, data, spec, backend) for spec in todo], parallel):
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
