"""APD's runs in parallel against the one Glimmer server, none done twice (Task 17).

- **Parallel:** ``in_parallel`` runs each life-arm on a thread of its own, ``parallel`` at a time;
  each life-arm stays sequential within itself. The threads share the session's vLLM server, which
  batches their requests, and the A100 is billed by the hour however many run, so throughput sets
  the compute units. What they share is safe across threads: the ledger takes a file lock on
  every reserve and settle, a journal appends under its file's lock, the store writes each file to
  a temporary file of its own and renames it, and each run has its own models, clock and log.
- **Never twice:** a finished life-arm is stored whole (``DataDir.save_run``) and skipped on a
  rerun. One that didn't finish starts again from its first night: what its attempt spent is
  settled from its journal (``journal.recover``, for a night its process never closed), recorded
  in ``spend/discarded.jsonl`` and the journal moved aside (``fresh_journal``), so the ledger stays
  true.
- **A failure:** a run with any failed model call stops and is never saved (``live.RunFailed``).
  Before each run starts, the session is checked (``ColabSession.check_room``): its tunnel up,
  vLLM answering, and ``run_s``, a whole life-arm's time, left before its deadline; if not, the
  run doesn't start (``colab_vllm.SessionEnding``). Either way the life-arms not started are
  cancelled, the ones running finish (their spend is real and their runs are kept), and the first
  failure is raised.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Final, TypeVar

from anjo.sleep.mind.apd.backend import Backend
from anjo.sleep.mind.apd.colab_vllm import MIN_SERVE_S, ColabSession
from anjo.sleep.mind.apd.journal import SpendJournal, recover
from anjo.sleep.mind.apd.live import Meter, run_live
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import ArmName, ArmRun, Life
from anjo.sleep.types import CheckpointRef

PARALLEL: Final = 8  # life-arms at once against the one server
Out = Callable[[str], None]
T = TypeVar("T")


def serial(out: Out) -> Out:
    """``out`` for many threads: one line at a time."""
    lock = threading.Lock()

    def say(line: str) -> None:
        with lock:
            out(line)

    return say


class _NotStarted(Exception):
    """An item that never began, since another failed first."""


def in_parallel(work: Sequence[Callable[[], T]], parallel: int) -> list[T]:
    """Each item on a thread of its own, ``parallel`` at a time; the results in order. On a failure
    no further item starts, the ones running finish, and the first failure is raised. When this
    thread is interrupted (a signal's ``SystemExit``, Ctrl-C), no further item starts and it goes
    on at once to its own cleanup, which stops the session the running ones need."""
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

    pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="apd")
    futures = [pool.submit(guarded, item) for item in work]
    failed: BaseException | None = None
    try:
        for future in as_completed(futures):
            error = None if future.cancelled() else future.exception()
            if error is not None and not isinstance(error, _NotStarted) and failed is None:
                failed = error
                stop.set()
                for other in futures:
                    other.cancel()
    except BaseException:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    if failed is not None:
        raise failed
    return [future.result() for future in futures]


def settle_orphans(backend: Backend, spend: SpendJournal) -> float:
    """Settle the nights ``spend`` held and never closed (a killed process's), when held."""
    if backend.ledger is None or backend.prices is None:
        return 0.0
    return recover(backend.ledger, spend, backend.prices)


def fresh_journal(data: DataDir, name: str, backend: Backend, out: Out, *, life_id: str, arm: str) -> SpendJournal:
    """The journal for a new attempt at ``name``: an earlier attempt's, if any, settled, recorded
    as discarded and moved aside first."""
    spend = SpendJournal(data.journal(name))
    if not spend.path.exists():
        return spend
    settle_orphans(backend, spend)
    nights = spend.jobs()
    usd = spend.spent()
    data.record_discarded({"name": name, "life": life_id, "arm": arm, "usd": usd, "nights": len(nights),
                           "calls": sum(len(night.calls) for night in nights)})
    spend.retire()
    out(f"discarded an unfinished attempt at {arm} on {life_id}: ${usd:.4f} of Spark Open, settled in the ledger")
    return spend


def _run_one(
    data: DataDir, life: Life, arm: ArmName, session: ColabSession, backend: Backend, out: Out,
    adapter: CheckpointRef | None, meter: Meter | None, prefix: str, run_s: float,
) -> ArmRun:
    session.check_room(run_s)
    spend = fresh_journal(data, f"{prefix}{life.id}.{arm}", backend, out, life_id=life.id, arm=arm)
    found = run_live(arm, life, session, backend.spark, ledger=backend.ledger, prices=backend.prices,
                     adapter=adapter, key=f"{prefix}run:{life.id}:{arm}", journal=spend, meter=meter)
    if data.run(life.id, arm) is None:
        data.save_run(found)
    out(f"ran {arm} on {life.id}: {len(found.reports)} report(s)")
    return found


def run_lives(
    data: DataDir, lives: Sequence[Life], arms: Sequence[ArmName], session: ColabSession, backend: Backend, out: Out,
    *, adapters: Mapping[ArmName, CheckpointRef] | None = None, parallel: int = PARALLEL, meter: Meter | None = None,
    measure: bool = False, run_s: float = MIN_SERVE_S,
) -> tuple[ArmRun, ...]:
    """Every arm on every life, ``parallel`` life-arms at a time, each run stored as it ends. A run
    already on disk is kept and not run again, except to ``measure`` (timing), which runs every one
    under journals of its own and stores only the new. Each run starts only when the session can
    carry ``run_s`` more."""
    say, prefix = serial(out), "timing-" if measure else ""
    todo = [(life, arm) for life in lives for arm in arms if measure or data.run(life.id, arm) is None]
    done = in_parallel([
        partial(_run_one, data, life, arm, session, backend, say, None if adapters is None else adapters.get(arm),
                meter, prefix, run_s)
        for life, arm in todo
    ], parallel)
    ran = {(run.life_id, run.arm): run for run in done}
    found = [ran.get((life.id, arm)) or data.run(life.id, arm) for life in lives for arm in arms]
    return tuple(run for run in found if run is not None)
