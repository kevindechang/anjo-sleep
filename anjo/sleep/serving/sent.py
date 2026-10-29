"""What a night client and its copies share about their requests (``client._Requests``)."""
from __future__ import annotations

import threading

from anjo.sleep.serving import config


class Sent:
    """What a client and its copies share about their requests: the slots one client may fill at once
    (``config.CONCURRENT_REQUESTS``, all one container takes), when the server last answered, and how
    many requests went again."""

    def __init__(self) -> None:
        self.slots = threading.BoundedSemaphore(config.CONCURRENT_REQUESTS)
        self._lock = threading.Lock()
        self._answered: float | None = None
        self._retries = 0
        self._down_until = float("-inf")  # the server answered nothing, not even WAKE_PATH: refuse until then

    def down(self, now: float) -> bool:
        with self._lock:
            return now < self._down_until

    def down_until(self, until: float) -> None:
        with self._lock:
            self._down_until = max(self._down_until, until)

    def warm(self, now: float) -> bool:
        """Whether the server answered within its scale-down window (``config.SCALEDOWN_S``)."""
        with self._lock:
            return self._answered is not None and now - self._answered < config.SCALEDOWN_S

    def answered(self, now: float) -> None:
        with self._lock:
            self._answered = now if self._answered is None else max(self._answered, now)

    def retried(self) -> None:
        with self._lock:
            self._retries += 1

    @property
    def retries(self) -> int:
        with self._lock:
            return self._retries
