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

An adapter is served only under the rendering it was trained under (``CheckpointRef.rendering``);
any other is refused with ``ServingError``, before a request is sent.

**Adapters load by an explicit call.** vLLM 0.28.0 answers 404 for an adapter it hasn't been told
to load, whatever sits in its folder (seen live, 2026-10-28). So a client whose server reads adapters
from a folder (``adapters_dir``: Modal's adapters volume, the APD Colab session's LoRA folder) loads
each one through ``/v1/load_lora_adapter`` before the first request that names it, once per client
(its dated copies share the set). One already loaded counts as loaded. A folder the server can't see
yet is asked for again for up to ``LOAD_WAIT_S``: a warm Modal container sees a new adapter within
``config.ADAPTER_REFRESH_S``. Only work nobody waits on takes that wait (the night, the morning, the
demo's pre-generated pairs); a chat turn and the live demo page load once (``load_adapter`` with no
wait, ``with_load_wait(0)``). A later 404 naming a loaded adapter (its container restarted and forgot
it) loads it again and tries once more. SageMaker's shim fetches and loads each adapter itself, so its
client has no ``adapters_dir`` and never loads.

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
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Final, Protocol, TypeVar

import httpx

from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving import config, framing
from anjo.sleep.types import CheckpointRef

TOP_P: Final = 0.95
TOP_K: Final = 64
MAX_TOKENS: Final = 1024
REQUEST_TIMEOUT_S: Final = 300.0  # a cold start waits inside the server's own limit

# Waking a scaled-to-zero server (``wake``): poll until it answers, under one hard total timeout.
WAKE_TOTAL_TIMEOUT_S: Final = 600.0  # a cold start of the 30B model, with room to spare
WAKE_POLL_TIMEOUT_S: Final = 10.0  # one poll; Modal's proxy may hold it while the container starts
WAKE_BACKOFF_START_S: Final = 2.0
WAKE_BACKOFF_MAX_S: Final = 20.0
WAKE_PATH: Final = "/v1/models"  # needs the bearer token and reaches vLLM, so it resets Modal's idle timer
LOCALHOST: Final = "localhost"
LOAD_PATH: Final = "/v1/load_lora_adapter"
LOAD_RETRY_S: Final = 5.0  # between loads of a folder the server can't see yet
LOAD_WAIT_S: Final = 2 * config.ADAPTER_REFRESH_S + LOAD_RETRY_S  # two of a warm server's volume refreshes
_ALREADY_LOADED: Final = b"already been loaded"
T = TypeVar("T")


class ServingError(RuntimeError):
    """The server refused or failed a request, or sent a reply that can't be read."""


class ModelNotFound(ServingError):
    """The server answered 404: it doesn't serve the model named. A restarted container has
    forgotten the adapters loaded into it."""


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


def json_object(path: str, raw: bytes | str) -> dict[str, Any]:
    """A reply's JSON object, or ``ServingError`` naming the path."""
    try:
        data = json.loads(raw)
    except ValueError:
        raise ServingError(f"{path} sent a reply that is not JSON") from None
    if not isinstance(data, dict):
        raise ServingError(f"{path} sent a reply that is not an object")
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
    never leaves this machine on its way to the tunnel."""

    def __init__(
        self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self.base_url = check_url(base_url)
        if not token:
            raise ServingError("the serving token is set")
        self._http = httpx.Client(
            base_url=self.base_url, headers={"Authorization": f"Bearer {token}"},
            transport=transport, timeout=timeout_s, trust_env=not is_loopback(httpx.URL(self.base_url).host),
        )

    def close(self) -> None:
        self._http.close()

    def _send(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        try:
            return self._http.post(path, json=dict(body))
        except httpx.HTTPError as error:
            raise ServingError(f"{path} failed in transport ({type(error).__name__})") from None

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        response = self._send(path, body)
        if response.status_code == 404:
            raise ModelNotFound(f"{path} returned HTTP 404")
        if response.is_error:
            raise ServingError(f"{path} returned HTTP {response.status_code}")
        return json_object(path, response.content)

    def load_adapter(self, name: str, folder: str) -> bool:
        """vLLM's runtime load of adapter ``name`` from ``folder`` on the server: True once it is
        loaded (one loaded already counts), False when the server finds no adapter there (404). Any
        other failure is ``ServingError`` naming the endpoint, never what the server said."""
        response = self._send(LOAD_PATH, {"lora_name": name, "lora_path": folder})
        if response.status_code == 400 and _ALREADY_LOADED in response.content:
            return True
        if response.status_code == 404:
            return False
        if response.is_error:
            raise ServingError(f"{LOAD_PATH} returned HTTP {response.status_code}")
        return True


def load_adapter(
    poster: HttpPoster, name: str, folder: str, *, wait_s: float = 0.0,
    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Load adapter ``name`` from ``folder/name`` into the server; one already loaded counts. A folder
    the server can't see yet is asked for again every ``LOAD_RETRY_S`` until ``wait_s`` has passed; the
    default, no wait, is one attempt. Anything else is ``ServingError`` naming the endpoint only."""
    deadline = clock() + wait_s
    while not poster.load_adapter(name, f"{folder}/{name}"):
        if clock() >= deadline:
            raise ServingError(f"{LOAD_PATH} found no adapter" + (f" in {wait_s:.0f} s" if wait_s else ""))
        sleep(LOAD_RETRY_S)


def tokens(poster: Poster, body: Mapping[str, Any]) -> list[int]:
    """``/tokenize``'s token ids for ``body``, or ``ServingError``."""
    data = poster.post("/tokenize", body)
    ids = data.get("tokens")
    if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
        raise ServingError("/tokenize sent no token ids")
    return ids


def reply_prompt(
    poster: Poster, model: str, messages: Sequence[Mapping[str, Any]], strength: str, day: date,
) -> list[int]:
    """The ids Glimmer is prompted with, as a training row's prompt: the cleaned chat with the
    generation prompt, rendered at ``strength`` on ``day``, then the reply's header."""
    chat = tokens(poster, {
        "model": model, "messages": framing.clean_messages(messages), "add_generation_prompt": True,
        "chat_template_kwargs": framing.template_kwargs(strength, day),
    })
    header = tokens(poster, {"model": model, "prompt": config.USER_CHANNEL, "add_special_tokens": False})
    return chat + header


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


class VllmClient:
    """``ModelClient`` and ``TokenLogprobs`` over one vLLM server: Modal's https URL with its
    bearer token, or any other ``Poster`` (``VllmClient.over``). ``on(day)`` dates its prompts
    (the night's date, virtual in an accelerated run); unset, they carry today's UTC date.
    ``adapters_dir`` is the folder the server reads adapters from, which the client loads them from
    before their first use, waiting up to ``load_wait_s`` for one the server can't see yet; ``None``
    when the backend loads them itself."""

    precision: Final = config.SERVING.name

    def __init__(
        self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S, max_tokens: int = MAX_TOKENS, adapters_dir: str | None = None,
        load_wait_s: float = LOAD_WAIT_S, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        poster = HttpPoster(base_url, token, transport=transport, timeout_s=timeout_s)
        self._setup(poster, max_tokens, None if adapters_dir is None else poster, adapters_dir)
        self._load_wait_s, self._clock, self._sleep = load_wait_s, clock, sleep

    @classmethod
    def over(cls, poster: Poster, *, max_tokens: int = MAX_TOKENS) -> VllmClient:
        """A client over ``poster``, whose server loads adapters itself (SageMaker's shim)."""
        client = cls.__new__(cls)
        client._setup(poster, max_tokens, None, None)
        client._load_wait_s, client._clock, client._sleep = LOAD_WAIT_S, time.monotonic, time.sleep
        return client

    def _setup(self, poster: Poster, max_tokens: int, loader: HttpPoster | None, adapters_dir: str | None) -> None:
        self._strength = config.reasoning_strength()
        self._poster, self._max_tokens = poster, max_tokens
        self._day: date | None = None
        self._loader, self._adapters_dir = loader, adapters_dir
        self._loaded: set[str] = set()

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

    @property
    def load_wait_s(self) -> float:
        """How long a load waits for an adapter folder the server can't see yet."""
        return self._load_wait_s

    @property
    def rendering(self) -> str:
        return framing.rendering(self._strength)

    @property
    def adapters_dir(self) -> str | None:
        """The folder the server reads adapters from, or ``None`` when the backend loads them itself."""
        return self._adapters_dir

    @property
    def base_url(self) -> str | None:
        """The server's URL, or ``None`` for a client over another ``Poster``."""
        url = getattr(self._poster, "base_url", None)
        return url if isinstance(url, str) else None

    def close(self) -> None:
        self._poster.close()

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return self._poster.post(path, body)

    def _tokens(self, body: Mapping[str, Any]) -> list[int]:
        return tokens(self._poster, body)

    def _prompt(self, model: str, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        return reply_prompt(self._poster, model, messages, self._strength, self._day or utc_today())

    def _served(self, adapter: CheckpointRef | None, call: Callable[[str], T]) -> T:
        """``call`` with the model to name. An adapter trained under another rendering is refused
        before anything is sent; one the server reads from a folder is loaded before its first use, and
        loaded again once if the server has forgotten it (a 404)."""
        if adapter is not None and adapter.rendering != self.rendering:
            raise ServingError(f"adapter {adapter.id} was trained under rendering {adapter.rendering}, "
                               f"not {self.rendering}; it is never served under another")
        model = model_name(adapter)
        if adapter is None or self._loader is None:
            return call(model)
        self._ensure_loaded(model)
        try:
            return call(model)
        except ModelNotFound:
            self._loaded.discard(model)
            self._ensure_loaded(model)
            return call(model)

    def _ensure_loaded(self, name: str) -> None:
        """Load ``name`` unless this client has (``load_adapter``, waiting ``load_wait_s``)."""
        if name in self._loaded or self._loader is None or self._adapters_dir is None:
            return
        load_adapter(self._loader, name, self._adapters_dir, wait_s=self._load_wait_s,
                     clock=self._clock, sleep=self._sleep)
        self._loaded.add(name)

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None,
    ) -> ChatResult:
        """One reply; ``max_tokens`` caps this call alone (MIND's thinker passes its allowance)."""
        return self._served(adapter, lambda model: self._complete(model, messages, temperature, seed, max_tokens))

    def _complete(
        self, model: str, messages: list[dict[str, str]], temperature: float, seed: int | None, max_tokens: int | None,
    ) -> ChatResult:
        prompt = self._prompt(model, messages)
        body: dict[str, Any] = {
            "model": model, "prompt": prompt, "temperature": temperature, "top_p": TOP_P, "top_k": TOP_K,
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
