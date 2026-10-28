"""One limit on a run's Gemini reads (MIND's data; ``gen_args``).

Lives generated side by side each read their gates on Gemini, so a run of many lives can send
Vertex more reads at once than its quota takes, and every one of them answers 429. A ``Limiter``
is made once per run and shared by every attempt's reader: at most ``concurrency`` reads in flight
at once (``--vertex-concurrency``), and, with ``per_minute`` (``--vertex-rpm``), their starts
spaced evenly, ``60 / per_minute`` seconds apart.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.model_api import ChatResult

DEFAULT_CONCURRENCY: Final = 4
SECONDS_PER_MINUTE: Final = 60.0


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
    def slot(self) -> Iterator[None]:
        """Hold one of the limiter's places for a call, once its paced start has come."""
        with self._slots:
            self._wait_for_start()
            yield

    def _wait_for_start(self) -> None:
        if self.per_minute is None:
            return
        with self._lock:
            now = self._clock()
            start = now if self._next is None else max(now, self._next)
            self._next = start + SECONDS_PER_MINUTE / self.per_minute
        if start > now:
            self._sleep(start - now)

    def limit(self, chat: ChatFn) -> ChatFn:
        """``chat``, each call through one of this limiter's places."""

        def limited(model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                    timeout_s: float, *, temperature: float | None = None) -> ChatResult:
            with self.slot():
                return chat(model, messages, tools, timeout_s, temperature=temperature)

        return limited
