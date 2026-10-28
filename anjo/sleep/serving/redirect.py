"""Modal's 303 to a long call's result, followed for ``client.HttpPoster``.

Modal holds a web request at most 150 s (all its web function types, ``web_server`` included), then
answers 303 to the original URL with a query naming the call. That result URL blocks until the call
completes, up to 150 s a hop, and serves its result to a GET without starting the call again. Seen
live on 2026-10-29 (trunk e24a0106): a 4,000-token completion took 188 s; Modal's proxy logged
``GET /v1/completions -> 200 OK (duration: 37.4 s, execution: 187.5 s)``, vLLM saw one POST, and at
most one request ran. The GET reads the original POST's result.

Only a 303 to the server's own scheme, host and port is followed, so the bearer token is never sent
anywhere else, and at most ``RESULT_HOPS`` of them. Once a 303 has come the call is running on vLLM:
a result GET that drops, or that Modal's proxy answers itself (``PROXY_STATUSES``), asks the same URL
again after a pause (``HOP_PAUSE_S``, doubling to ``HOP_PAUSE_MAX_S``), and the POST is never sent
again, which would start a second generation beside the first. Every GET ends by the caller's
deadline; a result not read by then is ``Unread``, naming the last status or error, never a body.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

import httpx

# Statuses Modal's proxy sends by itself: a POST that met one goes again once (``client._Requests``),
# inside the same attempts and bound; the same idempotent post, so at worst vLLM computes it twice. A
# result GET that meets one asks again (``follow``). The proxy answered: no liveness question.
PROXY_STATUSES: Final = frozenset({502, 503, 504})
RESULT_HOPS: Final = 4  # Modal's 303s one post follows: 300 s holds two 150 s waits; never forever
HOP_PAUSE_S: Final = 2.0  # before a result GET that failed asks again; doubled each time
HOP_PAUSE_MAX_S: Final = 16.0


class Unread(Exception):
    """A call answered 303, and its result wasn't read by the deadline. The message is the last status
    or error type seen, never a body."""


def result_url(response: httpx.Response, home: httpx.URL) -> httpx.URL | None:
    """Where Modal's 303 sends a post for its result, when that is ``home`` (scheme, host, port); else None."""
    location = response.headers.get("location")
    if response.status_code != httpx.codes.SEE_OTHER or not location:
        return None
    try:
        target = response.url.join(location)
    except httpx.InvalidURL:
        return None
    return target if (target.scheme, target.host, target.port) == (home.scheme, home.host, home.port) else None


def follow(
    http: httpx.Client, response: httpx.Response, *, home: httpx.URL, deadline: float,
    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """``response``, or when it is Modal's 303 on ``home``, the call's result read by GET before
    ``deadline`` (on ``clock``). Another status after the hops is returned as it is."""
    target, hops, pause, last = result_url(response, home), 0, HOP_PAUSE_S, "HTTP 303"
    while target is not None and hops < RESULT_HOPS:
        left = deadline - clock()
        if left <= 0:
            raise Unread(last)
        try:
            got = http.get(target, timeout=left)
        except httpx.HTTPError as error:  # dropped, or held past the deadline: the call may still run
            last = type(error).__name__
        else:
            last = f"HTTP {got.status_code}"
            if got.status_code not in PROXY_STATUSES:
                response, hops, target = got, hops + 1, result_url(got, home)
                continue
        if deadline - clock() <= pause:
            raise Unread(last)
        sleep(pause)
        pause = min(pause * 2, HOP_PAUSE_MAX_S)
    return response
