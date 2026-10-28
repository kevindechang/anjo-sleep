"""Glimmer (on SageMaker, or Modal as the fallback) as a host of Anjo's model, and the route that
sends a flagged person there (Task 4).

**The pin.** ``GlimmerPin`` is a ``HostPin`` for host ``sleep_modal``. Anjo's processor vocabulary
(N-PROC) has no family for Glimmer, and adding one is a change to a closed vocabulary and its
mobile twin, outside this task. So the pin names no family and has no processor record: the
reply row stores no model id, and the true identity (Glimmer, the precision, the adapter) is in
the pin's own identity and the sleep ancestry record. Prices are nominal so the turn's spend gate
still works; Modal bills by the GPU hour, which the budget ledger tracks.

**The runtime.** ``SleepRuntime`` compiles the turn as for any host, then prompts Glimmer the way
its adapter was trained (``serving.client.reply_prompt``): the chat's text cleaned of Glimmer's
special tokens (``serving.framing``), ``/tokenize`` with the generation prompt at the rendering's
reasoning strength and today's date, and the reply's header after it. It streams the reply from
``/v1/completions``, naming the person's adapter as the model (the base model when there is none
yet), so Glimmer answers directly and takes no tools. An adapter trained under another rendering
is never sent a turn. On Modal, vLLM serves an adapter only once told to load it, so the runtime
loads the person's adapter before their first turn on that server (``client.load_adapter``, as the
night's client does): one attempt, under the reply deadline, never the night's wait for a folder to
appear. A missing adapter or a failed load ends the turn as transient. It refuses a request whose person's words trip Anjo's crisis floor, the last
of four checks (the chat route's floor, the harness's no-model tiers and the prompt hook come
first). A refusal, a cold or stopped server and any error before the first token end the call
with an error, so the chat route's existing fallback answers with the frontier talker (RF3).
Modal holds a request through a cold start of up to 20 minutes, so the runtime calls it under a
short deadline of its own (``REPLY_DEADLINE_S``), apart from the container's three-hour limit.

**SageMaker** (``ANJO_SLEEP_BACKEND``, the default): the same request goes through the endpoint's
envelope (``sagemaker_transport``), signed by the dev host's role, with no bearer token, under the
same deadline. A turn never wakes SageMaker: while Glimmer sleeps (no copy) the endpoint refuses at
once, and the frontier talker answers. Days, nights, mornings and demos wake it first.

**The route.** ``sleep_route`` is the route override: a flagged person who chose the open-model
talker gets a route pinned to ``sleep_modal``; everyone else's route is unchanged.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final

import httpx

from anjo.core.anjo_model.assignment import OVERRIDE_CAUSE, ROUTE_ANJO, ModelRoute, QualifiedPin
from anjo.core.anjo_model.hosted_runtime import ChatCompletionsRuntime, bearer, status_error_code
from anjo.core.anjo_model.pin import (
    DecodeClamps,
    HostPin,
    ModelTraits,
    PinError,
    Prices,
    Routing,
    ThinkingControl,
)
from anjo.core.anjo_model.runtime import RequestNotSent, chat_body
from anjo.core.anjo_model.transport import HttpTransport, Transport, TransportError, Upstream
from anjo.core.best_effort import best_effort
from anjo.core.crisis import NO_CRISIS
from anjo.core.floor import FloorRules, check_message, normalize_message
from anjo.core.model_io import Finish, ModelEvent, ModelRequest, Served, TextDelta, TextPart
from anjo.core.processors import Family
from anjo.sleep.app.ancestry import record_served
from anjo.sleep.app.storage import glimmer_turns
from anjo.sleep.app.flags import sleep_enabled_for
from anjo.sleep.serving import activity, backend, config, framing
from anjo.sleep.serving.client import (
    HttpPoster,
    ModelNotFound,
    Poster,
    ServingError,
    load_adapter,
    reply_prompt,
    utc_today,
)
from anjo.sleep.types import CheckpointRef

logger = logging.getLogger(__name__)

HOST: Final = "sleep_modal"
URL_ENV: Final = backend.MODAL_URL_ENV
TOKEN_ENV: Final = backend.MODAL_TOKEN_ENV
INTERNAL_MODEL_ID: Final = "anjo-sleep-30b"
PRICES_AS_OF: Final = "2026-10-27"
# Nominal, per million tokens: the turn's spend gate needs a price, and Modal's real cost is GPU
# time, which the budget ledger holds.
NOMINAL_PRICES: Final = Prices(input_per_million=0.15, output_per_million=0.6, cache_read_per_million=None)
# Glimmer's named special tokens, its chat framing (<|start|>, <|message|>, <|eot|>) among them, for
# core's own pass. Every one of the 2,048 is broken before core sees the text (``clean_request``,
# serving.framing), so that pass finds none and stays one pass, however the text is nested.
CONTROL_TOKENS: Final = tuple(config.NAMED_SPECIAL_TOKENS.values())
# Modal must answer, and each later read arrive, within REPLY_DEADLINE_S. Past it the call ends
# through the transient-error path (httpx's timeout, then TransportError), so a cold, stopped or
# stalled server is answered by the frontier talker (RF3), well inside a turn's patience.
REPLY_DEADLINE_S: Final = 20.0
CONNECT_DEADLINE_S: Final = 5.0
_URL = re.compile(r"https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?(?:/[A-Za-z0-9._~/-]*)?")
_REVISION_ID = re.compile(r"rv-[0-9a-f]{16}")
# The adapters this process has loaded into Modal's server, as (server URL, adapter). Each turn gets a
# runtime of its own, so this outlives them; a server that answers 404 for one forgot it (a new container).
_LOADED: set[tuple[str, str]] = set()


@dataclass(frozen=True, kw_only=True)
class GlimmerPin(HostPin):  # type: ignore[misc, unused-ignore]  # Any when mypy skips core
    """A pin for Glimmer on Modal, for one person: ``adapter`` is the person's adapter, or None
    for the base model, and ``adapter_rendering`` the rendering it was trained under.
    ``reasoning_strength`` is what every call is rendered at ("" when the environment names no
    level, and then nothing is sent). ``family`` is always None (no N-PROC family names Glimmer)."""

    family: Family | None = None  # type: ignore[assignment, unused-ignore]
    adapter: str | None = None
    reasoning_strength: str = config.DEFAULT_REASONING_STRENGTH
    adapter_rendering: str | None = None
    user_id: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.host != HOST or self.family is not None:
            raise PinError(f"a Glimmer pin is host {HOST} and names no family")
        if not _REVISION_ID.fullmatch(self.revision_id) or self.model_id != config.MODEL:
            raise PinError("a Glimmer pin is built by glimmer_pin()")
        if self.adapter is not None:
            config.check_adapter_name(self.adapter)
        if self.reasoning_strength and self.reasoning_strength not in config.REASONING_LEVELS:
            raise PinError("a Glimmer pin's reasoning strength is a level the template knows")

    @property
    def rendering(self) -> str | None:
        """The rendering every call is made under, or None while the strength is unknown."""
        return framing.rendering(self.reasoning_strength) if self.reasoning_strength else None

    def renders_as_trained(self) -> bool:
        """Whether a call can go out: a known rendering, and the adapter's own if there is one."""
        return self.rendering is not None and (self.adapter is None or self.adapter_rendering == self.rendering)

    def identity(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id, "host": self.host, "family": None,
            "version": self.version, "internal_model_id": self.internal_model_id,
            "model_id": self.model_id, "license": self.license,
            "upstream_repo": self.upstream_repo, "upstream_commit": self.upstream_commit,
            "quantization": self.quantization, "adapter": self.adapter,
            "rendering": self.rendering, "adapter_rendering": self.adapter_rendering,
            "routing": self.routing.to_obj(), "traits": self.traits.to_obj(),
        }


def glimmer_traits() -> ModelTraits:
    """What the runtime sends. The reply's header is written for Glimmer (to the person), so it
    answers directly, as trained, and turns run without tools."""
    return ModelTraits(
        system_role="native",
        tool_calls="none",
        tool_choice_modes=(),
        tool_call_ids="native",
        parallel_tool_calls=False,
        max_tools_per_request=0,
        max_parallel_tool_calls=1,
        thinking=ThinkingControl(field="none", values=()),
        thinking_replay="n_a",
        reasoning_stream_field="none",
        max_context_tokens=config.MAX_MODEL_LEN,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        decode_clamps=DecodeClamps(temperature=(0.0, 2.0), top_p=(0.0, 1.0), top_k_accepted=True),
        usage_in_stream=True,
        cached_tokens_reported=False,
        control_tokens=CONTROL_TOKENS,
    )


def glimmer_pin(
    user_id: str, adapter: str | None, settings: config.ServeConfig | None = None, *,
    adapter_rendering: str | None = None,
) -> GlimmerPin:
    """The pin a flagged person's turn is served under, from the same settings Modal serves with.
    An unset revision leaves the commit empty, and an unknown reasoning strength the strength, and
    the runtime then sends nothing."""
    serve = config.ServeConfig.from_env() if settings is None else settings
    commit = serve.revision or ""
    try:
        strength = config.reasoning_strength()
    except config.ServeConfigError:
        strength = ""
    rendering = framing.rendering(strength) if strength else ""
    seed = f"{config.MODEL}@{commit}:{serve.precision.name}:{serve.tool_parser}:{rendering}"
    return GlimmerPin(
        revision_id="rv-" + hashlib.sha256(seed.encode()).hexdigest()[:16],
        host=HOST, version="30b", internal_model_id=INTERNAL_MODEL_ID, model_id=config.MODEL,
        license=config.LICENSE, upstream_repo=config.MODEL, upstream_commit=commit,
        quantization=serve.precision.name, status="active", prices_as_of=PRICES_AS_OF,
        prices=NOMINAL_PRICES,
        routing=Routing(zdr=True, allow_fallbacks=False, provider_order=()),
        traits=glimmer_traits(), notes="", adapter=adapter, reasoning_strength=strength,
        adapter_rendering=adapter_rendering, user_id=user_id,
    )


def carries_crisis(request: ModelRequest) -> bool:
    """Whether the person's latest words trip the crisis floor, harm routing included."""
    words = next((message for message in reversed(request.messages) if message.role == "user"), None)
    if words is None:
        return False
    text = " ".join(part.text for part in words.parts if isinstance(part, TextPart))
    decision = check_message(normalize_message(text), "en", FloorRules(harm_routing=True))
    return bool(decision.tier != NO_CRISIS)


def deadline_transport(inner: httpx.BaseTransport | None = None) -> HttpTransport:
    """The transport the sleep runtime calls Modal over, under the reply deadline. ``inner`` is
    the httpx transport underneath (tests pass httpx.MockTransport)."""
    timeout = httpx.Timeout(REPLY_DEADLINE_S, connect=CONNECT_DEADLINE_S)
    return HttpTransport(client=httpx.Client(transport=inner, timeout=timeout, follow_redirects=False))


def on_sagemaker() -> bool | None:
    """Whether turns go to SageMaker; None when ``ANJO_SLEEP_BACKEND`` names neither backend."""
    try:
        return backend.backend_name() == "sagemaker"
    except config.ServeConfigError:
        return None


def default_transport() -> Transport:
    if on_sagemaker():
        from anjo.sleep.app.sagemaker_transport import default_transport as through_sagemaker

        return through_sagemaker(REPLY_DEADLINE_S, CONNECT_DEADLINE_S)
    return deadline_transport()


def clean_request(request: ModelRequest) -> ModelRequest:
    """``request`` with every special token's text broken in the system turn and each message's
    words (``framing.clean_text``), before core compiles it."""
    system = tuple(replace(segment, text=framing.clean_text(segment.text)) for segment in request.system)
    messages = tuple(
        replace(message, parts=tuple(
            replace(part, text=framing.clean_text(part.text)) if isinstance(part, TextPart) else part
            for part in message.parts))
        for message in request.messages
    )
    return replace(request, system=system, messages=messages)


def completion_body(body: Mapping[str, Any], prompt: list[int]) -> dict[str, Any]:
    """The streamed ``/v1/completions`` request for a compiled chat body: its model, output limit and
    sampling, the prompt's ids in place of its messages, and the reply's stops."""
    sent = {
        "model": body["model"], "prompt": prompt, "max_tokens": body["max_tokens"], "stream": True,
        "stream_options": {"include_usage": True}, "stop_token_ids": list(config.REPLY_STOPS),
    }
    return {**sent, **{key: body[key] for key in ("temperature", "top_p", "top_k") if key in body}}


def as_chat_chunk(chunk: object) -> object:
    """One of vLLM's completion chunks in the chat shape core's stream parser reads: each choice's
    text as its delta's content. Anything else passes as it is, for the parser to judge."""
    if not isinstance(chunk, Mapping) or not isinstance(chunk.get("choices"), list):
        return chunk
    choices = [
        {"index": choice.get("index", 0), "finish_reason": choice.get("finish_reason"),
         "delta": {"content": choice["text"]} if isinstance(choice.get("text"), str) else {}}
        if isinstance(choice, Mapping) else choice
        for choice in chunk["choices"]
    ]
    return {**chunk, "choices": choices}


class _AsChat:
    """A completion stream read as a chat stream."""

    def __init__(self, upstream: Upstream) -> None:
        self._upstream = upstream

    def __iter__(self) -> Iterator[object]:
        return (as_chat_chunk(chunk) for chunk in self._upstream)

    def close(self) -> None:
        self._upstream.close()


class SleepRuntime(ChatCompletionsRuntime):  # type: ignore[misc, unused-ignore]
    """Glimmer through vLLM's OpenAI-compatible server. A reply is recorded as served once
    Glimmer's first word reaches the turn; a call that breaks off before that is answered by the
    fallback, so it records nothing. ``poster`` makes the ``/tokenize`` calls (tests pass one);
    unset, each call makes one for the configured backend under the reply deadline."""

    def __init__(self, pin: HostPin, *, transport: Transport | None = None, poster: Poster | None = None) -> None:
        if not isinstance(pin, GlimmerPin):
            raise PinError(f"the {HOST} runtime serves a Glimmer pin")
        super().__init__(pin, transport=transport or default_transport())
        self._glimmer = pin
        self._poster = poster

    def stream(
        self, request: ModelRequest, cancel: threading.Event
    ) -> Generator[ModelEvent, None, None]:
        if carries_crisis(request):
            yield Finish("error", "rejected")
            return
        if not self._glimmer.renders_as_trained():
            logger.error("%s sends nothing: adapter %s was trained under rendering %s, calls render as %s",
                         HOST, self._glimmer.adapter, self._glimmer.adapter_rendering, self._glimmer.rendering)
            yield Finish("error", "rejected")
            return
        recorded = False
        with activity.turn(glimmer_turns()):  # in flight until it ends: the night's GPU stop never cuts it
            for event in self._generate(request, cancel):
                if not recorded and isinstance(event, TextDelta) and event.text.strip():
                    recorded = True
                    with best_effort("record a sleep reply as served", user_id=self._glimmer.user_id):
                        record_served(self._glimmer.user_id, request.turn_id, model=self._wire_model())
                yield event

    def _generate(
        self, request: ModelRequest, cancel: threading.Event
    ) -> Generator[ModelEvent, None, None]:
        if cancel.is_set():
            yield Finish("error", "cancelled")
            return
        try:
            body = chat_body(clean_request(request), self._pin, model=self._wire_model())
            url, headers = self._endpoint()
            upstream = self._transport.open(url, headers, completion_body(body, self._prompt(body)))
        except RequestNotSent as refused:
            yield Finish("error", refused.code)
            return
        except ServingError as error:
            logger.warning("%s could not render the request: %s", HOST, error)
            yield Finish("error", "transient")
            return
        except TransportError as error:
            code = status_error_code(error.status)
            logger.warning("%s refused the request: %s (%s)", HOST, error.status, code)
            yield Finish("error", code)
            return
        yield from self._read(_AsChat(upstream), request, cancel)

    def _prompt(self, body: Mapping[str, Any]) -> list[int]:
        """The prompt's ids, as a training row's (``reply_prompt``), dated today (UTC), with the person's
        adapter loaded first (``_loaded_prompt``)."""
        poster = self._poster or self._made_poster()
        try:
            return self._loaded_prompt(poster, body)
        except ValueError:
            raise RequestNotSent("rejected") from None
        finally:
            if poster is not self._poster:
                poster.close()

    def _render(self, poster: Poster, body: Mapping[str, Any]) -> list[int]:
        return reply_prompt(poster, body["model"], body["messages"], self._glimmer.reasoning_strength, utc_today())

    def _loaded_prompt(self, poster: Poster, body: Mapping[str, Any]) -> list[int]:
        """``_render``, the person's adapter loaded into the server first unless this process has
        loaded it there. One load attempt per turn at most, under the poster's reply deadline: a
        loaded adapter the server has forgotten (a 404) gets that one load and one more try, and a
        fresh load it still doesn't serve ends the turn. Only an ``HttpPoster`` reaches vLLM itself
        (Modal's); SageMaker's shim loads by itself."""
        adapter = self._glimmer.adapter
        if adapter is None or not isinstance(poster, HttpPoster):
            return self._render(poster, body)
        key = (poster.base_url, adapter)
        loaded_before = key in _LOADED
        if not loaded_before:
            _load(poster, key)
        try:
            return self._render(poster, body)
        except ModelNotFound:
            _LOADED.discard(key)
            if not loaded_before:
                raise
            _load(poster, key)
            return self._render(poster, body)

    def _made_poster(self) -> Poster:
        if on_sagemaker():
            from anjo.sleep.app.sagemaker_transport import default_poster

            return default_poster(REPLY_DEADLINE_S, CONNECT_DEADLINE_S)
        token = os.environ.get(TOKEN_ENV, "").strip()
        return HttpPoster(os.environ.get(URL_ENV, "").strip().rstrip("/"), token, timeout_s=REPLY_DEADLINE_S)

    def _wire_model(self) -> str:
        return self._glimmer.adapter or config.MODEL

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        sagemaker = on_sagemaker()
        if sagemaker is None or not self._glimmer.upstream_commit:
            raise RequestNotSent("rejected")
        if sagemaker:
            from anjo.sleep.app.sagemaker_transport import endpoint_url

            return endpoint_url("/v1/completions"), {}  # SigV4 signs it; no token travels
        base = os.environ.get(URL_ENV, "").strip().rstrip("/")
        if not _URL.fullmatch(base):
            raise RequestNotSent("rejected")
        return f"{base}/v1/completions", bearer(os.environ.get(TOKEN_ENV, "").strip())

    def _reported(self, served: Served) -> Served:
        if served.model in (self._wire_model(), config.MODEL):
            return Served(model=config.MODEL, provider="")
        return served


def _load(poster: HttpPoster, key: tuple[str, str]) -> None:
    """One load of ``key``'s adapter from the adapters volume, with no wait for its folder."""
    load_adapter(poster, key[1], config.ADAPTERS_DIR)
    _LOADED.add(key)


# The adapter a person's turns are served with: Task 5's promoted checkpoint, once it sets one.
AdapterSource = Callable[[str], CheckpointRef | None]
_adapter_source: list[AdapterSource] = []


def set_adapter_source(source: AdapterSource | None) -> None:
    _adapter_source[:] = [] if source is None else [source]


def current_adapter(user_id: str) -> CheckpointRef | None:
    return _adapter_source[0](user_id) if _adapter_source else None


def sleep_route(user_id: str) -> ModelRoute | None:
    """The route override: a flagged person's open-model turns go to Glimmer, everyone else's
    route is unchanged (None)."""
    if not sleep_enabled_for(user_id):
        return None
    adapter = current_adapter(user_id)
    pin = glimmer_pin(user_id, None if adapter is None else adapter.id,
                      adapter_rendering=None if adapter is None else adapter.rendering)
    route: ModelRoute = ModelRoute(ROUTE_ANJO, None, OVERRIDE_CAUSE, None, (QualifiedPin(pin, None, None),))
    return route
