"""One limit on a run's Gemini reads (MIND's data; ``gen_args``).

Lives generated side by side each read their gates on Gemini, so a run of many lives can send
Vertex more reads at once than its quota takes, and every one of them answers 429. A ``Limiter``
is made once per run and shared by every attempt's reader (``Backend.read_limit``): at most
``concurrency`` reads in flight at once (``--vertex-concurrency``), and, with ``per_minute``
(``--vertex-rpm``), their starts spaced evenly, ``60 / per_minute`` seconds apart.

Live, the starts are paced to ``DEFAULT_PER_MINUTE`` unless the command says otherwise: Gemini on
Vertex runs on a shared quota with no fixed rate to read from the code or its docs
(``docs/VERTEX_EVAL_LOAD.md``), so the default is a conservative one.

**A kill reaches the limiter.** A read waiting for its place or its paced start waits on the
command's halt (``Backend.halt``) and ends with ``Halted`` once it is set, so a halted run sends
nothing more.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.apd.backend import halted, pause
from anjo.sleep.model_api import ChatResult

DEFAULT_CONCURRENCY: Final = 4
DEFAULT_PER_MINUTE: Final = 30.0  # conservative: Vertex's shared quota gives no rate to read
SECONDS_PER_MINUTE: Final = 60.0
HALT_POLL_S: Final = 0.1  # how often a read queued for its place looks at the halt


class Limiter:
    """At most ``concurrency`` calls at once, their starts paced to ``per_minute`` when given."""

    def __init__(self, concurrency: int, per_minute: float | None = None, *,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        if concurrency < 1:
            raise ValueError(f"a limiter lets at least one call through at once, not {concurrency}")
        if per_minute is not None and per_minute <= 0:
            raise ValueError(f"a limiter paces more than 0 calls a minute, not {per_minute:g}")
        self.concurrency, self.per_minute = concurrency, per_minute
        self._slots = threading.BoundedSemaphore(concurrency)
        self._lock = threading.Lock()
        self._next: float | None = None  # the earliest the next call may start
        self._clock, self._sleep = clock, sleep

    @contextmanager
    def slot(self, halt: threading.Event | None = None) -> Iterator[None]:
        """Hold one of the limiter's places for a call, once its paced start has come; ``Halted``
        instead once ``halt`` is set, whether it waited for its place or its start."""
        while not self._slots.acquire(timeout=None if halt is None else HALT_POLL_S):
            halted(halt)
        try:
            halted(halt)
            self._wait_for_start(halt)
            yield
        finally:
            self._slots.release()

    def _wait_for_start(self, halt: threading.Event | None) -> None:
        if self.per_minute is None:
            return
        with self._lock:
            now = self._clock()
            start = now if self._next is None else max(now, self._next)
            self._next = start + SECONDS_PER_MINUTE / self.per_minute
        if start > now:
            pause(start - now, self._sleep, halt)

    def limit(self, chat: ChatFn, halt: threading.Event | None = None) -> ChatFn:
        """``chat``, each call through one of this limiter's places, ending on ``halt``."""

        def limited(model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                    timeout_s: float, *, temperature: float | None = None) -> ChatResult:
            with self.slot(halt):
                return chat(model, messages, tools, timeout_s, temperature=temperature)

        return limited
