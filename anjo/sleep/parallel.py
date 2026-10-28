"""How many of a night step's model calls run at once (``ANJO_SLEEP_CONCURRENCY``), and the one
way they run side by side: ``Calls``.

vLLM batches requests that arrive together almost for free, and the serving GPU bills while it
waits, so a step that asks one call at a time pays mostly for an idle GPU. Practice (its scenes,
each scene's candidates and its critic) and the morning run (its arms, then its judge) send their
calls through ``Calls``; the gates map their items through it too (``gates.in_order``).

- ``map(fn, items)`` runs ``fn`` over ``items`` on up to ``workers`` threads and returns the results
  in item order, never in completion order, so a step's output is the serial run's given the same
  replies.
- ``call(fn)`` runs one model call in one of ``workers`` shared slots, so a step whose tasks run
  calls of their own side by side (a practice scene's candidates) still has at most ``workers``
  calls in flight.
- **The first failure stops the step.** No task or call starts after it; the ones already running
  end; then the failure of the lowest item is raised. No thread outlives ``map``, so the hold around
  a step settles every call it metered, and none is left open.

``workers`` 1 is the serial run: ``map`` loops on the calling thread and stops at the first failure.

The setting is read by the composition root (``concurrency(os.environ)``); nothing here reads the
environment.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from functools import partial
from typing import Final, TypeVar

from anjo.sleep.serving.config import CONCURRENT_REQUESTS

CONCURRENCY_ENV: Final = "ANJO_SLEEP_CONCURRENCY"
DEFAULT_CONCURRENCY: Final = 8
MAX_CONCURRENCY: Final = CONCURRENT_REQUESTS  # never more than one serving container takes at once
THREAD_PREFIX: Final = "sleep-calls"

In = TypeVar("In")
Out = TypeVar("Out")


class ConcurrencyError(ValueError):
    """``ANJO_SLEEP_CONCURRENCY`` is not a whole number the night can run at."""


class Stopped(RuntimeError):
    """A task or call that never started: an earlier one of its step failed."""


def concurrency(environ: Mapping[str, str]) -> int:
    """The calls a night step runs at once: ``ANJO_SLEEP_CONCURRENCY``, 8 when unset or blank."""
    raw = environ.get(CONCURRENCY_ENV, "").strip()
    if not raw:
        return DEFAULT_CONCURRENCY
    workers = int(raw) if raw.isdigit() else 0
    if not 1 <= workers <= MAX_CONCURRENCY:
        raise ConcurrencyError(f"{CONCURRENCY_ENV} is a whole number from 1 to {MAX_CONCURRENCY}")
    return workers


class Calls:
    """One step's model calls, at most ``workers`` at a time. Make one per step run: once a call
    or task has failed, nothing else starts through it."""

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("a step runs at least one call at a time")
        self.workers = workers
        self._slots = threading.BoundedSemaphore(workers)
        self._failed = threading.Event()

    def call(self, fn: Callable[[], Out]) -> Out:
        """``fn()``, one model call, once a slot is free."""
        with self._slots:
            return self._watched(fn)

    def map(self, fn: Callable[[In], Out], items: Sequence[In]) -> list[Out]:
        """``fn`` over ``items``, side by side, the results in item order."""
        if self.workers == 1 or len(items) <= 1:
            return [self._watched(partial(fn, item)) for item in items]
        pool = ThreadPoolExecutor(max_workers=min(self.workers, len(items)), thread_name_prefix=THREAD_PREFIX)
        try:
            futures = [pool.submit(self._watched, partial(fn, item)) for item in items]
            wait(futures, return_when=FIRST_EXCEPTION)
        except BaseException:
            self._failed.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)  # what hasn't started never does; the rest end
        return _in_order(futures)

    def _watched(self, fn: Callable[[], Out]) -> Out:
        if self._failed.is_set():
            raise Stopped("an earlier call of this step failed")
        try:
            return fn()
        except BaseException:
            self._failed.set()
            raise


def _in_order(futures: Sequence[Future[Out]]) -> list[Out]:
    """The results by index, or the lowest item's failure (one that never started only if none failed)."""
    failures = [error for f in futures if not f.cancelled() and (error := f.exception()) is not None]
    first = next((error for error in failures if not isinstance(error, Stopped)), None)
    if first is not None or failures:
        raise first or failures[0]
    return [f.result() for f in futures]
