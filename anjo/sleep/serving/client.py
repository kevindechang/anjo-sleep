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
from typing import Any, Final, Protocol

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


class ServingError(RuntimeError):
    """The server refused or failed a request, or sent a reply that can't be read."""


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

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(path, json=dict(body))
        except httpx.HTTPError as error:
            raise ServingError(f"{path} failed in transport ({type(error).__name__})") from None
        if response.is_error:
            raise ServingError(f"{path} returned HTTP {response.status_code}")
        return json_object(path, response.content)


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
    (the night's date, virtual in an accelerated run); unset, they carry today's UTC date."""

    precision: Final = config.SERVING.name

    def __init__(
        self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S, max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._strength = config.reasoning_strength()
        self._poster: Poster = HttpPoster(base_url, token, transport=transport, timeout_s=timeout_s)
        self._max_tokens = max_tokens
        self._day: date | None = None

    @classmethod
    def over(cls, poster: Poster, *, max_tokens: int = MAX_TOKENS) -> VllmClient:
        client = cls.__new__(cls)
        client._strength = config.reasoning_strength()
        client._poster, client._max_tokens, client._day = poster, max_tokens, None
        return client

    def on(self, day: date) -> VllmClient:
        """This client, its prompts dated ``day``; the poster is shared."""
        dated = type(self).__new__(type(self))
        dated.__dict__.update(self.__dict__)
        dated._day = day
        return dated

    @property
    def rendering(self) -> str:
        return framing.rendering(self._strength)

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

    def _prompt(self, adapter: CheckpointRef | None, messages: Sequence[Mapping[str, Any]]) -> tuple[str, list[int]]:
        """The model to name and the prompt's ids; an adapter trained under another rendering is refused."""
        if adapter is not None and adapter.rendering != self.rendering:
            raise ServingError(f"adapter {adapter.id} was trained under rendering {adapter.rendering}, "
                               f"not {self.rendering}; it is never served under another")
        model = model_name(adapter)
        return model, reply_prompt(self._poster, model, messages, self._strength, self._day or utc_today())

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None,
    ) -> ChatResult:
        """One reply; ``max_tokens`` caps this call alone (MIND's thinker passes its allowance)."""
        model, prompt = self._prompt(adapter, messages)
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
        model, prompt = self._prompt(adapter, messages)
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
