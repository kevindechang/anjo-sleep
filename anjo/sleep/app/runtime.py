"""Glimmer on Modal as a host of Anjo's model, and the route that sends a flagged person there
(Task 4).

**The pin.** ``GlimmerPin`` is a ``HostPin`` for host ``sleep_modal``. Anjo's processor vocabulary
(N-PROC) has no family for Glimmer, and adding one is a change to a closed vocabulary and its
mobile twin, outside this task. So the pin names no family and has no processor record: the
reply row stores no model id, and the true identity (Glimmer, the precision, the adapter) is in
the pin's own identity and the sleep ancestry record. Prices are nominal so the turn's spend gate
still works; Modal bills by the GPU hour, which the budget ledger tracks.

**The runtime.** ``SleepRuntime`` sends vLLM's OpenAI-compatible request to the Modal URL with
the bearer token, naming the person's adapter as the model, or the base model when there is none
yet. It refuses a request whose person's words trip Anjo's crisis floor, the last of four checks
(the chat route's floor, the harness's no-model tiers and the prompt hook come first). A refusal,
a cold or stopped server and any error before the first token end the call with an error, so the
chat route's existing fallback answers with the frontier talker (RF3). Modal holds a request
through a cold start of up to 20 minutes, so the runtime calls it under a short deadline of its
own (``REPLY_DEADLINE_S``), apart from the container's three-hour limit.

**The route.** ``sleep_route`` is the route override: a flagged person who chose the open-model
talker gets a route pinned to ``sleep_modal``; everyone else's route is unchanged.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from anjo.core.anjo_model.assignment import OVERRIDE_CAUSE, ROUTE_ANJO, ModelRoute, QualifiedPin
from anjo.core.anjo_model.hosted_runtime import ChatCompletionsRuntime, bearer
from anjo.core.anjo_model.pin import (
    DecodeClamps,
    HostPin,
    ModelTraits,
    PinError,
    Prices,
    Routing,
    ThinkingControl,
)
from anjo.core.anjo_model.runtime import RequestNotSent
from anjo.core.anjo_model.transport import HttpTransport, Transport
from anjo.core.best_effort import best_effort
from anjo.core.crisis import NO_CRISIS
from anjo.core.floor import FloorRules, check_message, normalize_message
from anjo.core.model_io import (
    Finish, ModelEvent, ModelRequest, Served, TextDelta, TextPart, ToolCallProposal,
)
from anjo.core.processors import Family
from anjo.sleep.app.ancestry import record_served
from anjo.sleep.app.flags import sleep_enabled_for
from anjo.sleep.serving import config

HOST: Final = "sleep_modal"
URL_ENV: Final = "ANJO_SLEEP_MODAL_URL"
TOKEN_ENV: Final = "ANJO_SLEEP_MODAL_TOKEN"
INTERNAL_MODEL_ID: Final = "anjo-sleep-30b"
PRICES_AS_OF: Final = "2026-10-27"
# Nominal, per million tokens: the turn's spend gate needs a price, and Modal's real cost is GPU
# time, which the budget ledger holds.
NOMINAL_PRICES: Final = Prices(input_per_million=0.15, output_per_million=0.6, cache_read_per_million=None)
# Glimmer's special tokens, kept out of any text the model is sent. Not yet read from the pinned
# tokenizer: the Oct 28 check confirms them from its tokenizer_config.json.
CONTROL_TOKENS: Final = (
    "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
    "<|eot_id|>", "<|eom_id|>", "<|python_tag|>", "<|finetune_right_pad_id|>",
)
MAX_TOOLS: Final = 16
# Modal must answer, and each later read arrive, within REPLY_DEADLINE_S. Past it the call ends
# through the transient-error path (httpx's timeout, then TransportError), so a cold, stopped or
# stalled server is answered by the frontier talker (RF3), well inside a turn's patience.
REPLY_DEADLINE_S: Final = 20.0
CONNECT_DEADLINE_S: Final = 5.0
_URL = re.compile(r"https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?(?:/[A-Za-z0-9._~/-]*)?")
_REVISION_ID = re.compile(r"rv-[0-9a-f]{16}")


@dataclass(frozen=True, kw_only=True)
class GlimmerPin(HostPin):  # type: ignore[misc, unused-ignore]  # Any when mypy skips core
    """A pin for Glimmer on Modal, for one person: ``adapter`` is the person's adapter, or None
    for the base model. ``family`` is always None (no N-PROC family names Glimmer)."""

    family: Family | None = None  # type: ignore[assignment, unused-ignore]
    adapter: str | None = None
    user_id: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.host != HOST or self.family is not None:
            raise PinError(f"a Glimmer pin is host {HOST} and names no family")
        if not _REVISION_ID.fullmatch(self.revision_id) or self.model_id != config.MODEL:
            raise PinError("a Glimmer pin is built by glimmer_pin()")
        if self.adapter is not None:
            config.check_adapter_name(self.adapter)

    def identity(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id, "host": self.host, "family": None,
            "version": self.version, "internal_model_id": self.internal_model_id,
            "model_id": self.model_id, "license": self.license,
            "upstream_repo": self.upstream_repo, "upstream_commit": self.upstream_commit,
            "quantization": self.quantization, "adapter": self.adapter,
            "routing": self.routing.to_obj(), "traits": self.traits.to_obj(),
        }


def glimmer_traits(tool_parser: str | None) -> ModelTraits:
    """What the server accepts. Without a tool-call parser for Glimmer, turns run without tools."""
    tools = tool_parser is not None
    return ModelTraits(
        system_role="native",
        tool_calls="native" if tools else "none",
        tool_choice_modes=("auto", "none") if tools else (),
        tool_call_ids="native",
        parallel_tool_calls=False,
        max_tools_per_request=MAX_TOOLS if tools else 0,
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


def glimmer_pin(user_id: str, adapter: str | None, settings: config.ServeConfig | None = None) -> GlimmerPin:
    """The pin a flagged person's turn is served under, from the same settings Modal serves with.
    An unset revision leaves the commit empty, and the runtime then sends nothing."""
    serve = config.ServeConfig.from_env() if settings is None else settings
    commit = serve.revision or ""
    seed = f"{config.MODEL}@{commit}:{serve.precision.name}:{serve.tool_parser}"
    return GlimmerPin(
        revision_id="rv-" + hashlib.sha256(seed.encode()).hexdigest()[:16],
        host=HOST, version="30b", internal_model_id=INTERNAL_MODEL_ID, model_id=config.MODEL,
        license=config.LICENSE, upstream_repo=config.MODEL, upstream_commit=commit,
        quantization=serve.precision.name, status="active", prices_as_of=PRICES_AS_OF,
        prices=NOMINAL_PRICES,
        routing=Routing(zdr=True, allow_fallbacks=False, provider_order=()),
        traits=glimmer_traits(serve.tool_parser), notes="", adapter=adapter, user_id=user_id,
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


def holds_words(request: ModelRequest) -> bool:
    """Whether the call is a tool round, whose words the loop holds back until the round ends
    cleanly (anjo/agent/loop.py); any other call's words are shown as they stream."""
    return bool(request.tools) and request.tool_choice.mode != "none"


class SleepRuntime(ChatCompletionsRuntime):  # type: ignore[misc, unused-ignore]
    """Glimmer through vLLM's OpenAI-compatible server on Modal. A reply is recorded as served
    once Glimmer's words or calls reach the turn: the first word shown, or a tool round's clean
    end. A call that breaks off before that is answered by the fallback, so it records nothing."""

    def __init__(self, pin: HostPin, *, transport: Transport | None = None) -> None:
        if not isinstance(pin, GlimmerPin):
            raise PinError(f"the {HOST} runtime serves a Glimmer pin")
        super().__init__(pin, transport=transport or deadline_transport())
        self._glimmer = pin

    def stream(
        self, request: ModelRequest, cancel: threading.Event
    ) -> Generator[ModelEvent, None, None]:
        if carries_crisis(request):
            yield Finish("error", "rejected")
            return
        held = holds_words(request)
        wrote = recorded = False
        for event in super().stream(request, cancel):
            wrote = wrote or isinstance(event, ToolCallProposal) or (
                isinstance(event, TextDelta) and bool(event.text.strip())
            )
            clean_end = isinstance(event, Finish) and event.error_code is None
            if wrote and not recorded and (clean_end or not held):
                recorded = True
                with best_effort("record a sleep reply as served", user_id=self._glimmer.user_id):
                    record_served(self._glimmer.user_id, request.turn_id, model=self._wire_model())
            yield event

    def _wire_model(self) -> str:
        return self._glimmer.adapter or config.MODEL

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        base = os.environ.get(URL_ENV, "").strip().rstrip("/")
        if not _URL.fullmatch(base) or not self._glimmer.upstream_commit:
            raise RequestNotSent("rejected")
        return f"{base}/v1/chat/completions", bearer(os.environ.get(TOKEN_ENV, "").strip())

    def _reported(self, served: Served) -> Served:
        if served.model in (self._wire_model(), config.MODEL):
            return Served(model=config.MODEL, provider="")
        return served


# The adapter a person's turns are served with: Task 5's promoted checkpoint, once it sets one.
AdapterSource = Callable[[str], str | None]
_adapter_source: list[AdapterSource] = []


def set_adapter_source(source: AdapterSource | None) -> None:
    _adapter_source[:] = [] if source is None else [source]


def current_adapter(user_id: str) -> str | None:
    return _adapter_source[0](user_id) if _adapter_source else None


def sleep_route(user_id: str) -> ModelRoute | None:
    """The route override: a flagged person's open-model turns go to Glimmer, everyone else's
    route is unchanged (None)."""
    if not sleep_enabled_for(user_id):
        return None
    pin = glimmer_pin(user_id, current_adapter(user_id))
    route: ModelRoute = ModelRoute(ROUTE_ANJO, None, OVERRIDE_CAUSE, None, (QualifiedPin(pin, None, None),))
    return route
