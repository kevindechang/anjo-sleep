"""Bounded acquisition for Sleep's local bookkeeping and cross-process log locks.

A lock timeout uses the existing transient model-error vocabulary, allowing an APD
attempt to fail and retry without changing reservations or freeing somebody else's lock.
"""
from __future__ import annotations

import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

from anjo.sleep.model_api import RETRY_AFTER_MAX_S, ModelApiError

LOCK_TIMEOUT_S: Final = RETRY_AFTER_MAX_S
LOCK_POLL_S: Final = 0.01


@contextmanager
def locked(lock: Any) -> Iterator[None]:
    if not lock.acquire(timeout=LOCK_TIMEOUT_S):
        raise ModelApiError(f"a bookkeeping lock timed out after {LOCK_TIMEOUT_S:g}s")
    try:
        yield
    finally:
        lock.release()


def file_lock(file: Any) -> None:
    """Take an exclusive flock, leaving ownership/release with the caller's descriptor."""
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ModelApiError(f"a file lock timed out after {LOCK_TIMEOUT_S:g}s") from None
            time.sleep(min(LOCK_POLL_S, left))
