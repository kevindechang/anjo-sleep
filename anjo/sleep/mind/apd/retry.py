"""APD's transient failures, retried at its own transport level (Task 17 review).

A 429, a 5xx, a timeout or a dropped connection is tried again, at most ``ATTEMPTS`` tries in
all, ``backoff`` apart (doubling, jittered). Anything else fails at once.

**A 429** (``rate_limited``: the provider was busy, and billed nothing) waits longer: the
provider's hint (``Retry-After``, Vertex's ``RetryInfo``) or ``RATE_BACKOFF_S`` doubling, jittered,
never more than ``MAX_WAIT_S`` a wait, at most ``RATE_LIMIT_ATTEMPTS`` tries of the call. In
generation, a call that still fails on any transient failure ends its attempt, not its life, and
the next attempt starts after ``ATTEMPT_COOL_DOWN_S``.

- **Glimmer** (vLLM on Colab): ``RetryTransport`` under the session's client
  (``ColabSession.client``), so the thinker, the world query and 17b's adapters all retry. Its
  tries are held by the session's own ``colab_cu`` hold, billed by the hour, not the call.
- **Spark Open:** ``live.SparkCalls`` retries a call whose ``ModelApiError`` is ``transient``
  (read from ``model_api``'s own messages, which name the HTTP status or the transport error's
  type and nothing else). Each try is a call of its own: counted against the night's hold, priced
  at one call's worst case when it failed, and journaled.

- **Generation** (``backend.CountedChat``): the lives' writer, extractor and gate readers try a
  transient failure again inside the attempt's own hold, from ``GEN_SPARK_RETRIES`` extra tries the
  attempt holds; a failed try is priced at one call's worst case, a 429 at nothing (it was billed
  nothing). The Muse auditor tries again only on a 429.

**How long a reply may take.** A generation call goes out capped at ``REPLY_MAX_TOKENS``
(``backend.gen_chat``), which is the most one call is held for, and waits ``reply_timeout_s`` of
that cap: the first token's allowance, then every token at a slow but healthy decode speed. A long
reply that is still coming is never cut off; a call that passes even that wait is a stalled one,
and its retry gets the same long wait (2026-10-28: an uncapped extraction reply passed a flat
180 s twice in a row on Fireworks).

Only the APD runner retries. The night's calls, ``spark_inquiry``'s single attempt for Sleep and
``model_api`` are unchanged.
"""
from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import Final

import httpx

from anjo.sleep.model_api import RateLimited
from anjo.sleep.night.holds import SPARK_MAX_OUT_TOKENS

ATTEMPTS: Final = 3  # tries in all, the first included
BACKOFF_S: Final = 2.0  # before the second try; doubled before each next one
JITTER: Final = (0.5, 1.5)  # each wait is its backoff times a uniform draw in this range
RETRIED: Final = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)  # timeouts and resets
RATE_LIMIT_ATTEMPTS: Final = 5  # tries in all of a call answered 429, while the attempt's extra tries last
RATE_BACKOFF_S: Final = 5.0  # before a 429's second try, without a hint; doubled before each next one
MAX_WAIT_S: Final = 60.0  # the longest any one wait for a 429 lasts, a hint included
ATTEMPT_COOL_DOWN_S: Final = 60.0  # before the attempt after one a call ended (rate limits are per minute)
REPLY_MAX_TOKENS: Final = SPARK_MAX_OUT_TOKENS  # every generation reply's cap: the most one call is held for
FIRST_TOKEN_S: Final = 60.0  # queueing and reading a long prompt, before the first token
SLOW_TOKENS_PER_S: Final = 20.0  # a slow but healthy decode speed for Spark Open
_TRANSIENT: Final = re.compile(
    r" (?:returned HTTP (?:429|5\d\d)$|timed out after "
    r"|failed in transport \((?:ConnectError|ReadError|WriteError|CloseError|RemoteProtocolError)\))"
)


def retryable(status: int) -> bool:
    """A status worth another try: rate-limited or a server error."""
    return status == 429 or 500 <= status < 600


def transient(error: Exception) -> bool:
    """``model_api``'s message for a 429, a 5xx, a timeout or a dropped connection."""
    return _TRANSIENT.search(str(error)) is not None


def rate_limited(error: Exception) -> bool:
    """A 429: ``model_api``'s ``RateLimited``, or its message for one."""
    return isinstance(error, RateLimited) or str(error).endswith(" returned HTTP 429")


def rate_limit_wait(attempt: int, hint: float | None, rand: Callable[[], float] = random.random) -> float:
    """The wait after a 429 on try ``attempt``: the provider's ``hint``, or ``RATE_BACKOFF_S``
    doubling per try, jittered; never more than ``MAX_WAIT_S``."""
    if hint is not None:
        return min(hint, MAX_WAIT_S)
    low, high = JITTER
    return min(RATE_BACKOFF_S * 2.0 ** (attempt - 1) * (low + (high - low) * rand()), MAX_WAIT_S)


def hint(error: Exception) -> float | None:
    """A 429's own wait, when the provider gave one."""
    return error.retry_after_s if isinstance(error, RateLimited) else None


def reply_timeout_s(max_tokens: int) -> float:
    """How long a reply capped at ``max_tokens`` may take when healthy: the first token's
    allowance, then every token at ``SLOW_TOKENS_PER_S``."""
    if max_tokens < 1:
        raise ValueError("a reply is capped at one token or more")
    return FIRST_TOKEN_S + max_tokens / SLOW_TOKENS_PER_S


def backoff(attempt: int, rand: Callable[[], float] = random.random) -> float:
    """The wait after try ``attempt`` failed: ``BACKOFF_S`` doubling per try, jittered."""
    low, high = JITTER
    return BACKOFF_S * 2.0 ** (attempt - 1) * (low + (high - low) * rand())


class RetryTransport(httpx.BaseTransport):  # type: ignore[misc]  # httpx is unfollowed here (pyproject)
    """``inner``, each request tried again on a 429, a 5xx, a timeout or a reset, at most
    ``attempts`` tries; the last try's answer or error is the request's. ``retries`` counts the
    extra tries."""

    def __init__(self, inner: httpx.BaseTransport, *, attempts: int = ATTEMPTS,
                 sleep: Callable[[float], None] = time.sleep, rand: Callable[[], float] = random.random) -> None:
        if attempts < 1:
            raise ValueError("a request is tried at least once")
        self._inner, self._attempts, self._sleep, self._rand = inner, attempts, sleep, rand
        self.retries = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        for attempt in range(1, self._attempts):
            try:
                response = self._inner.handle_request(request)
            except RETRIED:
                pass
            else:
                if not retryable(response.status_code):
                    return response
                response.close()
            self.retries += 1
            self._sleep(backoff(attempt, self._rand))
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()
