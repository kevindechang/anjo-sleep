"""The night's client for Glimmer's vLLM server (Task 5): practice, every gate and every arm go
through it, so all of them sample one server at one precision (``config.SERVING``).

It is the ``ModelClient`` and ``TokenLogprobs`` port over vLLM's OpenAI-compatible API. Every call
renders the chat the way the adapter is trained (``reply_prompt``): its text cleaned
(``framing.clean_messages``), ``/tokenize`` turns it into token ids with the generation prompt
(``<|start|>assistant``) at the rendering's reasoning strength and the client's date, and the
reply's channel header (``config.USER_CHANNEL``) follows as its own ids. So Glimmer answers
directly, never reasoning first, and the prompt is token for token a training row's.

- ``complete`` sends those ids to ``/v1/completions`` for one sampled reply, naming the person's
  adapter as the model (the base model when there is none) with the registered sampling
  (temperature as given, top_p 0.95, top_k 64) and a seed, so evals repeat; the reply ends at
  ``config.REPLY_STOPS``;
- ``continuation_logprobs`` scores a continuation token by token: the same ids, then the
  continuation's, and ``/v1/completions`` returns the prompt's log-probabilities with
  ``prompt_logprobs``, one generated token at most. Only the continuation's tokens are scored.
  A ``reusing_tokens`` copy asks ``/tokenize`` once per body, whatever model it names
  (``serving.token_memo``): the knowledge scorer's, which scores the same lines on many adapters.

An adapter is served only under the rendering it was trained under (``CheckpointRef.rendering``);
any other is refused with ``ServingError``, before a request is sent.

**Adapters load by an explicit call.** vLLM 0.28.0 answers 404 for an adapter it hasn't been told
to load, whatever sits in its folder (seen live, 2026-10-28). So a client whose server reads adapters
from a folder (``adapters_dir``: Modal's adapters volume, the APD Colab session's LoRA folder) loads
each one through ``/v1/load_lora_adapter`` before the first request that names it, once per client
(its dated copies share the set). One already loaded counts as loaded. Modal's container never reloads
its adapters volume (night 1c of 2026-10-28 stalled in the Modal process that did), so an adapter committed
since the container started isn't on its mount: the client then stages it onto the container's own
disk (``stage``: ``Serve.stage``, through ``backend.stage_on_modal``) and loads it from there. One
found nowhere is asked for again for up to ``LOAD_WAIT_S``. Only work nobody waits on takes that wait
(the night, the morning, the demo's pre-generated pairs); a chat turn and the live demo page load once
(``load_adapter`` with no wait, ``with_load_wait(0)``), and the live page's ``once()`` copy never
stages: it only tries where a stage puts adapters. A later 404 naming a loaded adapter (its container
restarted and forgot it) loads it again and tries once more. SageMaker's shim fetches and loads each
adapter itself, so its client has no ``adapters_dir`` and never loads. One client serves many threads at once (the night's
calls run side by side, ``parallel.Calls``): ``httpx.Client`` may be shared between threads, the
loaded set changes only under one lock its copies share, and each adapter loads under a lock of its
own, so threads that first name an adapter together load it once, and a call to one already loaded
never waits behind another's load.

**A request lost on the way goes again** (the night of 2026-10-28 failed its safety gate on one
request Modal's proxy never delivered). The night's client sends a request that got no status back
(``TransportFailed``) again, ``ATTEMPTS`` in all, after a short pause (``BACKOFF_S``): every request
is an idempotent post (``/tokenize``, ``/v1/completions`` with its seed, an adapter load), so a repeat
gives the same answer. A status is an answer and never goes again, but for a 408, 502, 503 or 504
(``PROXY_STATUSES``), which Modal's proxy sends by itself (2026-10-29: a morning's 502, a /tokenize's 408)
while vLLM was healthy: that post goes again once, inside the same attempts. Until the server has answered
within its scale-down window an attempt may wait out a cold start (``REQUEST_TIMEOUT_S``); after that
a first attempt gives up at ``WARM_TIMEOUT_S``, and any later one, which may meet a restarted
container, gets the rest of the budget. All the attempts of one request end within
``REQUEST_TIMEOUT_S`` of the first, so no bound built on one request's time changes. ``once()``
sends each request once with the full timeout, as before: the live page, where a visitor waits. A
client and its copies hold at most ``config.CONCURRENT_REQUESTS`` requests in flight, all one Modal
container takes, so none waits in Modal's queue behind the client's own; ``retries()`` counts the
requests sent again, for the night log. **A wedged server fails the client fast.** When a request lost
in transport gives up (its attempts or its ``REQUEST_TIMEOUT_S`` spent), the client asks ``WAKE_PATH``,
which never reaches vLLM's engine, within ``LIVENESS_S``. With no answer (night 1c: the container took
every request and answered none, and each gate thread waited out 300 s per request) it fails as
``ServerUnresponsive``, and so does every request the client and its copies send for the next
``DOWN_HOLD_S``, at once and unsent. So a wedge costs one round, and a container that restarts
mid-night (a cold start of about 3 minutes) gets the rest of the round to come back. Only a client that
sends again asks, and only once the server has answered within its scale-down window, never in a
cold start; a ``once()`` copy is refused while the server is down. A server that is only
busy past the bound looks the same, and an engine that stalls while the API server answers isn't
caught. A turn's poster (``app.runtime``) is its own ``HttpPoster``, under its 20 s deadline, and
never retries.

**A long call is read from Modal's result URL** (``serving.redirect``), on the server's own host, inside
the request's bound, never posted again; any other redirect is a ``ServingError`` naming its status.

Each request is one POST of a JSON body to a vLLM path, through a ``Poster``: ``HttpPoster`` over
Modal's https URL with the bearer token (``ANJO_SLEEP_MODAL_URL``, ``ANJO_SLEEP_MODAL_TOKEN``, read
by the app, never here), or the SageMaker endpoint's signed envelope
(``serving.sagemaker.invoke.SageMakerPoster``). The serving backend (``serving.backend``) picks one.
The URL is https, or plain http on a loopback host only (``check_url``): the local end of an SSH
tunnel to Glimmer on Colab (``serving.colab_vllm``), which never leaves this machine.
Errors name the endpoint and the status, never a message's words.
"""
from __future__ import annotations

import ipaddress
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Any, Final, Protocol, TypeVar

import httpx

from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving import config, framing, redirect, soft_tokens, token_memo
from anjo.sleep.serving.redirect import PROXY_STATUSES
from anjo.sleep.serving.sent import Sent
from anjo.sleep.types import CheckpointRef

TOP_P: Final = 0.95
TOP_K: Final = 64
MAX_TOKENS: Final = 1024
REQUEST_TIMEOUT_S: Final = 300.0  # a cold start waits inside the server's own limit; one request's whole bound
# The night's client sends a request lost in transport (no status came back) again: every night post
# is idempotent, and its seed fixed, so a repeat gives the same answer. A first attempt at a warm server
# gives up at WARM_TIMEOUT_S, past the 150 s at which Modal answers any request it holds with a 303 the
# client follows (at 90 s, warm completions of 128 s and 147 s were cut and sent again): only a request
# with no status by then is lost. Hops after a 303, and any later attempt, get the rest of
# REQUEST_TIMEOUT_S (300 - 165 - 2 = 133 s after a warm timeout): all of them end inside it.
WARM_TIMEOUT_S: Final = 165.0
BACKOFF_S: Final = (2.0, 4.0)  # before the second attempt, and before the third
ATTEMPTS: Final = len(BACKOFF_S) + 1

# Waking a scaled-to-zero server (``wake``): poll until it answers, under one hard total timeout.
WAKE_TOTAL_TIMEOUT_S: Final = 600.0  # a cold start of the 30B model, with room to spare
WAKE_POLL_TIMEOUT_S: Final = 10.0  # one poll; Modal's proxy may hold it while the container starts
WAKE_BACKOFF_START_S: Final = 2.0
WAKE_BACKOFF_MAX_S: Final = 20.0
WAKE_PATH: Final = "/v1/models"  # needs the bearer token and reaches vLLM, so it resets Modal's idle timer
LOCALHOST: Final = "localhost"
LOAD_PATH: Final = "/v1/load_lora_adapter"
LOAD_RETRY_S: Final = 5.0  # between loads of an adapter the server can't find yet
LOAD_WAIT_S: Final = 45.0  # how long a load waits for an adapter found nowhere: its upload may still be committing
# When a request lost in transport gives up, the server must answer WAKE_PATH within LIVENESS_S or the
# client stops sending for DOWN_HOLD_S. /v1/models never reaches vLLM's engine; 15 s covers the slowest
# /tokenize seen under load (8.6 s, night 1c), which does more.
LIVENESS_S: Final = 15.0
DOWN_HOLD_S: Final = 60.0
_ALREADY_LOADED: Final = b"already been loaded"
T = TypeVar("T")


class ServingError(RuntimeError):
    """The server refused or failed a request, or sent a reply that can't be read."""


class ModelNotFound(ServingError):
    """The server answered 404: it doesn't serve the model named. A restarted container has
    forgotten the adapters loaded into it."""


class TransportFailed(ServingError):
    """The request or its reply was lost on the way: no status came back."""


class ProxyFailed(ServingError):
    """Modal's proxy answered 502, 503 or 504 itself: the request, or vLLM's answer, was lost behind it."""


class ResultUnread(TransportFailed):
    """A call answered Modal's 303 and its result wasn't read in time. It may still be running on vLLM,
    so it is never posted again; the liveness question is asked as after any lost request."""


class ServerUnresponsive(ServingError):
    """The server answered nothing, not even ``WAKE_PATH``, within ``LIVENESS_S``: a wedged container.
    Every request fails at once for the next ``DOWN_HOLD_S``."""

    def __init__(self) -> None:
        super().__init__(f"the server didn't answer {WAKE_PATH} within {LIVENESS_S:.0f} s; "
                         f"requests fail at once for {DOWN_HOLD_S:.0f} s")


class GlimmerNotReady(ServingError):
    """The server didn't answer in time. The message is a code, never a request."""

    def __init__(self) -> None:
        super().__init__("glimmer_not_ready")


@dataclass(frozen=True)
class WakeResult:
    """How long the wake took, and whether the server answered."""

    seconds: float
    ok: bool


def ping(
    base_url: str, token: str, *, timeout_s: float = WAKE_POLL_TIMEOUT_S,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """One cheap authenticated request that only a ready vLLM answers with 200. Modal holds the
    request while it starts a container and may redirect it (303) to keep it open, so redirects are
    followed. A failure is False, never an exception, and no message carries the token."""
    try:
        with httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {token}"},
            transport=transport, timeout=timeout_s, follow_redirects=True,
        ) as http:
            return bool(http.get(WAKE_PATH).status_code == 200)
    except httpx.HTTPError:
        return False  # still starting, or unreachable: the caller polls again or gives up


def wake(
    base_url: str, token: str, timeout_s: float = WAKE_TOTAL_TIMEOUT_S, *,
    transport: httpx.BaseTransport | None = None, clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WakeResult:
    """Poll the served endpoint until it answers, for at most ``timeout_s`` in all."""
    if not base_url.startswith("https://"):
        raise ServingError("the serving URL is https")
    if not token:
        raise ServingError("the serving token is set")
    started = clock()
    backoff = WAKE_BACKOFF_START_S
    while True:
        remaining = timeout_s - (clock() - started)
        if remaining <= 0:
            return WakeResult(clock() - started, False)
        if ping(base_url, token, timeout_s=min(WAKE_POLL_TIMEOUT_S, remaining), transport=transport):
            return WakeResult(clock() - started, True)
        remaining = timeout_s - (clock() - started)
        if remaining <= 0:
            return WakeResult(clock() - started, False)
        sleep(min(backoff, remaining))
        backoff = min(backoff * 2, WAKE_BACKOFF_MAX_S)


def model_name(adapter: CheckpointRef | None) -> str:
    """What vLLM serves: the adapter's name, or the base model."""
    return config.MODEL if adapter is None else config.check_adapter_name(adapter.id)


class Poster(Protocol):
    """One JSON POST to a vLLM path, read back as a JSON object. Raises ``ServingError`` naming
    the path and the status, never the body."""

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


def json_object(path: str, raw: bytes | str, status: int | None = None) -> dict[str, Any]:
    """A reply's JSON object, or ``ServingError`` naming the path, and the reply's ``status`` when given."""
    named = "" if status is None else f" (HTTP {status})"
    try:
        data = json.loads(raw)
    except ValueError:
        raise ServingError(f"{path} sent a reply that is not JSON{named}") from None
    if not isinstance(data, dict):
        raise ServingError(f"{path} sent a reply that is not an object{named}")
    return data


def is_loopback(host: str) -> bool:
    """Whether ``host`` names this machine: ``localhost`` or a loopback address, never a name
    that merely starts like one."""
    if host == LOCALHOST:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_url(base_url: str) -> str:
    """``base_url`` without its trailing slash: https, or plain http on a loopback host only."""
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL:
        raise ServingError("the serving URL is https, or http on a loopback host only") from None
    if not (url.scheme == "https" or (url.scheme == "http" and is_loopback(url.host))):
        raise ServingError("the serving URL is https, or http on a loopback host only")
    return base_url.rstrip("/")


class HttpPoster:
    """vLLM's server with the bearer token: Modal's https URL, or a tunnel's loopback end. A
    loopback request never takes a proxy from the environment (``trust_env``), so the token
    never leaves this machine on its way to the tunnel. ``clock`` and ``sleep`` time a post's hops to
    Modal's result URL (``redirect.follow``)."""

    def __init__(
        self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = check_url(base_url)
        if not token:
            raise ServingError("the serving token is set")
        self._timeout_s, self._clock, self._sleep, self._home = timeout_s, clock, sleep, httpx.URL(self.base_url)
        self._http = httpx.Client(
            base_url=self.base_url, headers={"Authorization": f"Bearer {token}"},
            transport=transport, timeout=timeout_s, trust_env=not is_loopback(httpx.URL(self.base_url).host),
        )

    def close(self) -> None:
        self._http.close()

    def _send(self, path: str, body: Mapping[str, Any], timeout_s: float | None, hops_s: float | None) -> httpx.Response:
        """One POST, waiting ``timeout_s`` (None: the poster's) for an answer, and Modal's 303 to its result
        followed with GETs, all within ``hops_s`` (None: ``timeout_s``). Any other redirect is ``ServingError``."""
        budget = self._timeout_s if timeout_s is None else timeout_s
        bound = budget if hops_s is None else max(budget, hops_s)
        deadline = self._clock() + bound
        try:
            posted = self._http.post(path, json=dict(body), timeout=budget)
        except httpx.HTTPError as error:
            raise TransportFailed(f"{path} failed in transport ({type(error).__name__})") from None
        try:  # a call that answered 303 is running on vLLM: only its result is asked again, never the POST
            response = redirect.follow(self._http, posted, home=self._home, deadline=deadline, clock=self._clock,
                                       sleep=self._sleep)
        except redirect.Unread as unread:
            raise ResultUnread(f"{path}'s result wasn't read within {bound:.0f} s of its HTTP 303 (last: {unread})") from None
        if 300 <= response.status_code < 400:
            raise ServingError(f"{path} returned HTTP {response.status_code}, a redirect it doesn't follow")
        return response

    def post(self, path: str, body: Mapping[str, Any], *, timeout_s: float | None = None,
             hops_s: float | None = None) -> dict[str, Any]:
        """``timeout_s`` is this request's alone (None: the poster's); ``hops_s`` bounds its 303 hops too."""
        response = self._send(path, body, timeout_s, hops_s)
        if response.status_code == 404:
            raise ModelNotFound(f"{path} returned HTTP 404")
        if response.status_code in PROXY_STATUSES:
            raise ProxyFailed(f"{path} returned HTTP {response.status_code}")
        if response.is_error:
            raise ServingError(f"{path} returned HTTP {response.status_code}")
        return json_object(path, response.content, response.status_code)

    def answers(self, timeout_s: float) -> bool:
        """Whether vLLM answers ``WAKE_PATH`` within ``timeout_s``. Only its API server handles it, so a
        busy engine doesn't delay it. A failure is False, never an exception."""
        try:
            return bool(self._http.get(WAKE_PATH, timeout=timeout_s, follow_redirects=True).status_code == 200)
        except httpx.HTTPError:
            return False  # no answer within the bound: the caller decides what that means

    def get(self, path: str) -> dict[str, Any]:
        """One GET of ``path`` as a JSON object; ``ServingError`` names the path and the status."""
        try:
            response = self._http.get(path)
        except httpx.HTTPError as error:
            raise TransportFailed(f"{path} failed in transport ({type(error).__name__})") from None
        if response.is_error:
            raise ServingError(f"{path} returned HTTP {response.status_code}")
        return json_object(path, response.content)

    def load_adapter(self, name: str, folder: str, *, timeout_s: float | None = None, hops_s: float | None = None) -> bool:
        """vLLM's runtime load of adapter ``name`` from ``folder`` on the server: True once it is
        loaded (one loaded already counts), False when the server finds no adapter there (404). Any
        other failure is ``ServingError`` naming the endpoint, never what the server said."""
        response = self._send(LOAD_PATH, {"lora_name": name, "lora_path": folder}, timeout_s, hops_s)
        if response.status_code == 400 and _ALREADY_LOADED in response.content:
            return True
        if response.status_code == 404:
            return False
        if response.is_error:
            raise ServingError(f"{LOAD_PATH} returned HTTP {response.status_code}")
        return True


class Loader(Protocol):
    """What loads an adapter into vLLM (``HttpPoster.load_adapter``)."""

    def load_adapter(self, name: str, folder: str) -> bool: ...


Stage = Callable[[str], "str | None"]
"""Copies adapter ``name`` onto the server's own disk and names the folder it is in; None when there is
no such adapter yet (Modal's ``Serve.stage``, through ``backend.stage_on_modal``)."""


def load_adapter(
    poster: Loader, name: str, folder: str, *, stage: Stage | None = None, wait_s: float = 0.0,
    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Load adapter ``name`` from ``folder/name`` into the server; one already loaded counts. One the
    server can't find there is staged (``stage``, when given) and loaded from the folder it names. One
    found nowhere is asked for again every ``LOAD_RETRY_S`` until ``wait_s`` has passed; the default, no
    wait, is one attempt. Anything else is ``ServingError`` naming the endpoint only."""
    deadline = clock() + wait_s
    while not _load_once(poster, name, folder, stage):
        if clock() >= deadline:
            raise ServingError(f"{LOAD_PATH} found no adapter" + (f" in {wait_s:.0f} s" if wait_s else ""))
        sleep(LOAD_RETRY_S)


def _staged_already(name: str) -> str:
    """A ``Stage`` that copies nothing: the folder a stage puts adapters in, for a client nobody may wait
    on a stage for (``VllmClient.once``)."""
    return config.STAGED_ADAPTERS_DIR


def _load_once(poster: Loader, name: str, folder: str, stage: Stage | None) -> bool:
    if poster.load_adapter(name, f"{folder}/{name}"):
        return True
    staged = None if stage is None else stage(name)
    return staged is not None and poster.load_adapter(name, f"{staged}/{name}")


def tokens(poster: Poster, body: Mapping[str, Any]) -> list[int]:
    """``/tokenize``'s token ids for ``body``, or ``ServingError``."""
    data = poster.post("/tokenize", body)
    ids = data.get("tokens")
    if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
        raise ServingError("/tokenize sent no token ids")
    return ids


def reply_header(poster: Poster, model: str) -> list[int]:
    """The reply's channel header (``config.USER_CHANNEL``) as ``model``'s ids."""
    return tokens(poster, {"model": model, "prompt": config.USER_CHANNEL, "add_special_tokens": False})


def reply_prompt(
    poster: Poster, model: str, messages: Sequence[Mapping[str, Any]], strength: str, day: date,
    header: Callable[[str], Sequence[int]] | None = None,
) -> list[int]:
    """The ids Glimmer is prompted with, as a training row's prompt: the cleaned chat with the generation
    prompt, rendered at ``strength`` on ``day``, then the reply's header (``header``'s when given)."""
    chat = tokens(poster, {
        "model": model, "messages": framing.clean_messages(messages), "add_generation_prompt": True,
        "chat_template_kwargs": framing.template_kwargs(strength, day),
    })
    return chat + list(reply_header(poster, model) if header is None else header(model))


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


@dataclass(frozen=True)
class _Requests:
    """``Poster`` and ``Loader`` for one client. Each request takes one of the client's slots; one lost
    in transport (``TransportFailed``) goes again, ``attempts`` in all, after the pauses in
    ``BACKOFF_S``. A first attempt gives up at a warm server's timeout, or at ``cold_s`` until the server
    has answered within its scale-down window; a later one, which may meet a restarted container, gets
    whatever is left of ``cold_s`` after the first began, and none runs past it.
    A status is an answer, and never goes again, but for a post the proxy answered itself
    (``ProxyFailed``), which goes again once, within the same attempts. A poster other than ``HttpPoster`` keeps its own
    timeout (SageMaker's), and reports no transport failure apart from the rest, so it is never retried."""

    poster: Poster
    sent: Sent
    attempts: int
    cold_s: float
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    memo: token_memo.TokenMemo | None = None  # ``VllmClient.reusing_tokens``: /tokenize once per body

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        poster = self.poster
        if self.memo is not None and path == token_memo.TOKENIZE:
            return self.memo.post(body, lambda: replace(self, memo=None).post(path, body))
        if isinstance(poster, HttpPoster):
            return self._send(lambda timeout, left: poster.post(path, body, timeout_s=timeout, hops_s=left))
        return self._send(lambda timeout, left: poster.post(path, body))

    def load_adapter(self, name: str, folder: str) -> bool:
        poster = self.poster
        if not isinstance(poster, HttpPoster):
            raise ServingError("only a server that reads adapters from a folder loads one")
        return self._send(lambda timeout, left: poster.load_adapter(name, folder, timeout_s=timeout, hops_s=left))

    def stage(self, stage: Stage, name: str) -> str | None:
        """``stage`` under one of the client's slots: it is one more input for the container."""
        with self.sent.slots:
            self._refuse_if_down()
            return stage(name)

    def close(self) -> None:
        self.poster.close()

    def _send(self, attempt: Callable[[float, float], T]) -> T:  # its timeout, and the rest of cold_s for 303 hops
        deadline = self.clock() + self.cold_s
        proxied = False  # a status from the proxy has gone again: it never goes a third time
        for tried, pause in enumerate((*BACKOFF_S[: self.attempts - 1], None)):
            try:
                with self.sent.slots:
                    self._refuse_if_down()
                    got = attempt(self._timeout(deadline, first=tried == 0), deadline - self.clock())
            except ProxyFailed:
                if proxied or pause is None or deadline - self.clock() <= pause:
                    raise
                proxied = True
                self.sent.retried()
                self.sleep(pause)
                continue
            except TransportFailed as lost:  # a call that answered 303 is running: never posted again
                if pause is None or deadline - self.clock() <= pause or isinstance(lost, ResultUnread):
                    self._check_alive()  # only a request about to give up asks: a restart gets the whole round
                    raise
                self.sent.retried()
                self.sleep(pause)
                continue
            self.sent.answered(self.clock())
            return got
        raise AssertionError("the last attempt returns or raises")

    def _refuse_if_down(self) -> None:
        if self.sent.down(self.clock()):
            raise ServerUnresponsive()

    def _check_alive(self) -> None:
        """When a request lost in transport gives up: an ``HttpPoster``'s server must answer
        ``WAKE_PATH`` within ``LIVENESS_S``, asked under a slot so the client's own requests never queue
        it at Modal. With no answer, every request fails for ``DOWN_HOLD_S``, this one first. Only a
        client that sends again asks (the night's): ``once()`` fails as before, with no wait added for
        someone waiting on it. Nor is a server asked before it has answered within its scale-down window:
        a cold start (right after a wake, or after a silence) is slow by nature."""
        poster = self.poster
        if not isinstance(poster, HttpPoster) or self.attempts == 1 or not self.sent.warm(self.clock()):
            return
        self._refuse_if_down()  # another thread found it so already: don't ask again
        with self.sent.slots:
            if poster.answers(LIVENESS_S):
                return
        self.sent.down_until(self.clock() + DOWN_HOLD_S)
        raise ServerUnresponsive()

    def _timeout(self, deadline: float, *, first: bool) -> float:
        warm = first and self.attempts > 1 and self.sent.warm(self.clock())
        return min(WARM_TIMEOUT_S if warm else self.cold_s, deadline - self.clock())


class VllmClient:
    """``ModelClient`` and ``TokenLogprobs`` over one vLLM server: Modal's https URL with its
    bearer token, or any other ``Poster`` (``VllmClient.over``). ``on(day)`` dates its prompts
    (the night's date, virtual in an accelerated run); unset, they carry today's UTC date.
    ``adapters_dir`` is the folder the server reads adapters from, which the client loads them from
    before their first use, staging one the server can't find there (``stage``, Modal's) and waiting up
    to ``load_wait_s`` for one found nowhere; ``None`` when the backend loads them itself."""

    precision: Final = config.SERVING.name

    def __init__(
        self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S, max_tokens: int = MAX_TOKENS, adapters_dir: str | None = None,
        stage: Stage | None = None, load_wait_s: float = LOAD_WAIT_S, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        poster = HttpPoster(base_url, token, transport=transport, timeout_s=timeout_s, clock=clock, sleep=sleep)
        self._load_wait_s, self._clock, self._sleep = load_wait_s, clock, sleep
        self._setup(poster, max_tokens, adapters_dir, timeout_s, stage)

    @classmethod
    def over(cls, poster: Poster, *, max_tokens: int = MAX_TOKENS) -> VllmClient:
        """A client over ``poster``, whose server loads adapters itself (SageMaker's shim)."""
        client = cls.__new__(cls)
        client._load_wait_s, client._clock, client._sleep = LOAD_WAIT_S, time.monotonic, time.sleep
        client._setup(poster, max_tokens, None, REQUEST_TIMEOUT_S)
        return client

    def _setup(
        self, poster: Poster, max_tokens: int, adapters_dir: str | None, cold_s: float, stage: Stage | None = None,
    ) -> None:
        self._strength = config.reasoning_strength()
        self._poster, self._max_tokens = poster, max_tokens
        self._requests = _Requests(poster, Sent(), ATTEMPTS, cold_s, self._clock, self._sleep)
        self._day: date | None = None
        self._adapters_dir = adapters_dir
        self._stage = stage
        self._loaded: set[str] = set()
        self._headers: dict[str, tuple[int, ...]] = {}  # the reply's header per model (``_header``), and its lock
        self._header_lock = threading.Lock()  # both shared with every copy: /tokenize slowed to 11 s in a burst
        self._loading: dict[str, threading.Lock] = {}  # one lock per adapter, held while it loads
        self._load_lock = threading.Lock()  # guards both; shared with every copy ``on``, ``with_load_wait`` and ``once`` make
        self._soft: tuple[soft_tokens.EmbeddingRows, Callable[[soft_tokens.Matrix], str]] | None = None

    def on(self, day: date) -> VllmClient:
        """This client, its prompts dated ``day``; the poster is shared."""
        dated = type(self).__new__(type(self))
        dated.__dict__.update(self.__dict__)
        dated._day = day
        return dated

    def with_load_wait(self, seconds: float) -> VllmClient:
        """This client, waiting up to ``seconds`` for an adapter folder the server can't see yet (0:
        one attempt, for an answer someone is waiting on); the poster and the loaded set are shared."""
        waiting = type(self).__new__(type(self))
        waiting.__dict__.update(self.__dict__)
        waiting._load_wait_s = seconds
        return waiting

    def once(self) -> VllmClient:
        """This client, sending each request once with the full ``REQUEST_TIMEOUT_S``, as a turn does: for
        an answer someone is waiting on (the live page). It never stages an adapter (a stage may take
        ``backend.STAGE_TIMEOUT_S``): it loads one from the server's folder or from where a stage put it
        already, and otherwise fails at once. Its slots and its count are shared."""
        single = type(self).__new__(type(self))
        single.__dict__.update(self.__dict__)
        single._requests = replace(self._requests, attempts=1)
        single._stage = None if self._stage is None else _staged_already
        return single

    def with_soft_tokens(
        self, rows: soft_tokens.EmbeddingRows, *, encode: Callable[[soft_tokens.Matrix], str] = soft_tokens.encode_embeds,
    ) -> VllmClient:
        """This client, able to send a turn with soft tokens (``complete(..., soft=...)``) as its whole
        prompt's embeddings (``serving.soft_tokens``); a turn without them is sent as today."""
        switched = type(self).__new__(type(self))
        switched.__dict__.update(self.__dict__)
        switched._soft = (rows, encode)
        return switched

    def reusing_tokens(self) -> VllmClient:
        """This client, asking ``/tokenize`` once per body whatever model it names, for a run that scores the
        same lines on many adapters (``serving.token_memo``, which says why that is sound); copies share it."""
        reusing = type(self).__new__(type(self))
        reusing.__dict__.update(self.__dict__)
        reusing._requests = replace(self._requests, memo=token_memo.TokenMemo())
        return reusing

    @property
    def attempts(self) -> int:
        """How many times one request lost in transport is sent at most."""
        return self._requests.attempts

    def retries(self) -> int:
        """How many requests this client and its copies have sent again, so far."""
        return self._requests.sent.retries

    @property
    def load_wait_s(self) -> float:
        """How long a load waits for an adapter folder the server can't see yet."""
        return self._load_wait_s

    @property
    def rendering(self) -> str:
        return framing.rendering(self._strength)

    @property
    def max_tokens(self) -> int:
        """The cap on a reply whose call names none."""
        return self._max_tokens

    def models(self) -> dict[str, Any]:
        """``GET /v1/models``, what the server serves, in one of this client's slots and sent once:
        a failure is ``ServingError``, never sent again."""
        poster = self._requests.poster
        with self._requests.sent.slots:
            if isinstance(poster, HttpPoster):
                return poster.get(WAKE_PATH)
            return poster.post(WAKE_PATH, {})  # SageMaker's shim sends it on as a GET

    @property
    def adapters_dir(self) -> str | None:
        """The folder the server reads adapters from, or ``None`` when the backend loads them itself."""
        return self._adapters_dir

    @property
    def stages(self) -> bool:
        """Whether an adapter the server can't find on its folder is staged onto its own disk (Modal's
        night client; never a ``once()`` copy)."""
        return self._stage is not None and self._stage is not _staged_already

    @property
    def base_url(self) -> str | None:
        """The server's URL, or ``None`` for a client over another ``Poster``."""
        url = getattr(self._poster, "base_url", None)
        return url if isinstance(url, str) else None

    def close(self) -> None:
        self._poster.close()

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return self._requests.post(path, body)

    def _tokens(self, body: Mapping[str, Any]) -> list[int]:
        return tokens(self._requests, body)

    def _prompt(self, model: str, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        return reply_prompt(self._requests, model, messages, self._strength, self._day or utc_today(), self._header)

    def _header(self, model: str) -> tuple[int, ...]:
        """The reply's header as ``model``'s ids, asked once per model for the client's life (a burst waits)."""
        with self._header_lock:  # never empty: a header is a few ids
            return self._headers.get(model) or self._headers.setdefault(model, tuple(reply_header(self._requests, model)))

    def _served(self, adapter: CheckpointRef | None, call: Callable[[str], T]) -> T:
        """``call`` with the model to name. An adapter trained under another rendering is refused
        before anything is sent; one the server reads from a folder is loaded before its first use, and
        loaded again once if the server has forgotten it (a 404)."""
        if adapter is not None and adapter.rendering != self.rendering:
            raise ServingError(f"adapter {adapter.id} was trained under rendering {adapter.rendering}, "
                               f"not {self.rendering}; it is never served under another")
        model = model_name(adapter)
        if adapter is None or self._adapters_dir is None:
            return call(model)
        self._ensure_loaded(model)
        try:
            return call(model)
        except ModelNotFound:
            with self._load_lock:
                self._loaded.discard(model)
            self._ensure_loaded(model)
            return call(model)

    def _is_loaded(self, name: str) -> bool:
        with self._load_lock:
            return name in self._loaded

    def _ensure_loaded(self, name: str) -> None:
        """Load ``name`` unless this client has (``load_adapter``, waiting ``load_wait_s``). A thread
        that asks while another loads it waits for that load rather than sending its own; a call to an
        adapter already loaded never waits on any load."""
        if self._adapters_dir is None or self._is_loaded(name):
            return
        with self._load_lock:
            loading = self._loading.setdefault(name, threading.Lock())
        with loading:
            if self._is_loaded(name):
                return
            load_adapter(self._requests, name, self._adapters_dir, stage=self._staging(),
                         wait_s=self._load_wait_s, clock=self._clock, sleep=self._sleep)
            with self._load_lock:
                self._loaded.add(name)

    def _staging(self) -> Stage | None:
        stage, requests = self._stage, self._requests
        return None if stage is None else lambda name: requests.stage(stage, name)

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None, soft: soft_tokens.SoftTokens | None = None,
    ) -> ChatResult:
        """One reply; ``max_tokens`` caps this call alone (MIND's thinker passes its allowance). ``soft``
        puts soft tokens in the latest user turn, on a client ``with_soft_tokens`` only."""
        if soft is not None and self._soft is None:
            raise ServingError("soft tokens are sent only by a client switched on with with_soft_tokens")
        if soft is not None and (not messages or messages[-1].get("role") != "user"):
            raise ServingError("soft tokens go in the latest turn, which is the user's")
        return self._served(adapter, lambda model: self._complete(model, messages, temperature, seed, max_tokens, soft))

    def _complete(
        self, model: str, messages: list[dict[str, str]], temperature: float, seed: int | None, max_tokens: int | None,
        soft: soft_tokens.SoftTokens | None = None,
    ) -> ChatResult:
        prompt = self._prompt(model, messages)
        sent: dict[str, Any] = {"prompt": prompt}
        if soft is not None and self._soft is not None:  # the whole prompt as embeddings, the soft tokens spliced in
            rows, encode = self._soft
            sent = {"prompt_embeds": encode(soft_tokens.splice(prompt, rows, soft))}
        body: dict[str, Any] = {
            "model": model, **sent, "temperature": temperature, "top_p": TOP_P, "top_k": TOP_K,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stop_token_ids": list(config.REPLY_STOPS),
        }
        if seed is not None:
            body["seed"] = seed
        data = self._post("/v1/completions", body)
        try:
            choice, usage = data["choices"][0], data["usage"]
            text = choice.get("text")
            return ChatResult(
                text="" if text is None else str(text), tool_calls=(), model=str(data["model"]),
                input_tokens=int(usage["prompt_tokens"]), output_tokens=int(usage["completion_tokens"]),
                finish_reason=choice.get("finish_reason"),
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ServingError("/v1/completions sent a malformed reply") from None

    def continuation_logprobs(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], continuation: str,
    ) -> Sequence[float]:
        return self._served(adapter, lambda model: self._logprobs(model, messages, continuation))

    def _logprobs(self, model: str, messages: list[dict[str, str]], continuation: str) -> Sequence[float]:
        prompt = self._prompt(model, messages)
        tail = self._tokens({"model": model, "prompt": framing.clean_text(continuation), "add_special_tokens": False})
        if not tail:
            raise ServingError("the continuation has no tokens")
        data = self._post("/v1/completions", {
            "model": model, "prompt": prompt + tail, "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 0,
        })
        try:
            scored = data["choices"][0]["prompt_logprobs"][-len(tail):]
            return tuple(float(entry[str(token)]["logprob"]) for entry, token in zip(scored, tail, strict=True))
        except (KeyError, IndexError, TypeError, ValueError):
            raise ServingError("/v1/completions sent no prompt log-probabilities") from None
