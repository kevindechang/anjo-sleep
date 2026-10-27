"""A bounded wait for the night lock (Task 9): the same ``flock`` file the night command and the
simulated day take (``night/sweep.py``), so while it is held no night or day is writing."""
from __future__ import annotations

import fcntl
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

POLL_S = 0.1


@contextmanager
def drain(
    path: Path, timeout_s: float, *, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[None]:
    """Take the lock at ``path``, waiting up to ``timeout_s`` for a live night to reach a step
    boundary and leave. Raises ``TimeoutError`` if it never does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = clock() + timeout_s
    with open(path, "a+b") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if clock() >= deadline:
                    raise TimeoutError("a night is still running for this host") from None
                sleep(POLL_S)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
