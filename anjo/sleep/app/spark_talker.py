"""Muse Spark through the Meta Model API as the judges' talker while Glimmer is cold, waking or
past the judges' hard spend cap (``judge_talker.judge_route``).

``SparkPin`` is a pin for host ``sleep_spark``, off the N-PROC vocabulary like Glimmer's
(``client.OFF_VOCABULARY_HOSTS``): it names no family and has no processor record. ``SparkRuntime``
is the hosted chat-completions runtime every hosted pin shares, pointed at the Meta Model API's
OpenAI-compatible ``/chat/completions`` (``meta_api``: ``ANJO_META_API_BASE``, key
``ANJO_META_API_KEY``, read when a request is sent), naming the pinned ``muse-spark-1.3``. It takes no
tools, as Glimmer's turns take none, and every reply passes the chat route's own event filter and
reply guard (``AnjoModelClient``), as any open-model turn's does.

**The crisis path is unchanged.** Like ``SleepRuntime``, it refuses a request whose person's words
trip Anjo's crisis floor, so such a turn ends with every open-model pin refused and the chat route's
existing fallback answers it exactly as before (``chat_routes``: Gemini, or Haiku).
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Final

from anjo.core.anjo_model.hosted_runtime import ChatCompletionsRuntime, bearer
from anjo.core.anjo_model.pin import DecodeClamps, HostPin, ModelTraits, PinError, Prices, Routing, ThinkingControl
from anjo.core.anjo_model.runtime import pin_served
from anjo.core.anjo_model.transport import Transport
from anjo.core.model_io import Finish, ModelEvent, ModelRequest, Served
from anjo.core.processors import Family
from anjo.sleep import meta_api
from anjo.sleep.app.runtime import NOMINAL_PRICES, carries_crisis

HOST: Final = "sleep_spark"
MODEL: Final = meta_api.PINNED_MODELS["muse-spark"]
INTERNAL_MODEL_ID: Final = "anjo-sleep-fallback"
PRICES_AS_OF: Final = "2026-11-02"
CONTEXT_TOKENS: Final = 131_072
OUTPUT_TOKENS: Final = 4_096
LABEL: Final = "spark-fallback"  # what ``GET /api/sleep/me/talker`` calls a reply it serves
# Its tokenizer isn't published: the Llama family's special tokens, which retrieved context and tool
# results are stripped of before they reach it (``turn_input``), as a guard.
CONTROL_TOKENS: Final = ("<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                         "<|eot_id|>", "<|eom_id|>", "<|header_start|>", "<|header_end|>", "<|eot|>")


def spark_configured() -> bool:
    """Whether the Meta Model API's key is set: without it the judges' fallback is the chat route's own."""
    return bool(os.environ.get(meta_api.KEY_ENV, "").strip())


@dataclass(frozen=True, kw_only=True)
class SparkPin(HostPin):  # type: ignore[misc, unused-ignore]  # Any when mypy skips core
    """Muse Spark for one person's turn; ``family`` is always None (no N-PROC family names it)."""

    family: Family | None = None  # type: ignore[assignment, unused-ignore]
    user_id: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.host != HOST or self.family is not None or self.model_id != MODEL:
            raise PinError(f"a Spark pin is host {HOST}, names no family and serves {MODEL}")

    def identity(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id, "host": self.host, "family": None, "version": self.version,
            "internal_model_id": self.internal_model_id, "model_id": self.model_id,
            "routing": self.routing.to_obj(), "traits": self.traits.to_obj(),
        }


def spark_traits() -> ModelTraits:
    return ModelTraits(
        system_role="native", tool_calls="none", tool_choice_modes=(), tool_call_ids="native",
        parallel_tool_calls=False, max_tools_per_request=0, max_parallel_tool_calls=1,
        thinking=ThinkingControl(field="none", values=()), thinking_replay="n_a", reasoning_stream_field="none",
        max_context_tokens=CONTEXT_TOKENS, max_output_tokens=OUTPUT_TOKENS,
        decode_clamps=DecodeClamps(temperature=(0.0, 2.0), top_p=(0.0, 1.0), top_k_accepted=False),
        usage_in_stream=False,  # no stream_options: a field the API might refuse
        cached_tokens_reported=False, control_tokens=CONTROL_TOKENS,
    )


def spark_pin(user_id: str) -> SparkPin:
    seed = f"{HOST}:{MODEL}"
    return SparkPin(
        revision_id="rv-" + hashlib.sha256(seed.encode()).hexdigest()[:16], host=HOST, version="spark",
        internal_model_id=INTERNAL_MODEL_ID, model_id=MODEL, license="proprietary", upstream_repo=MODEL,
        upstream_commit="", quantization="undisclosed", status="active", prices_as_of=PRICES_AS_OF,
        prices=Prices(input_per_million=NOMINAL_PRICES.input_per_million,
                      output_per_million=NOMINAL_PRICES.output_per_million, cache_read_per_million=None),
        routing=Routing(zdr=True, allow_fallbacks=False, provider_order=()), traits=spark_traits(), notes="",
        user_id=user_id,
    )


class SparkRuntime(ChatCompletionsRuntime):  # type: ignore[misc, unused-ignore]
    """Muse Spark through the Meta Model API's streamed chat completions."""

    def __init__(self, pin: HostPin, *, transport: Transport | None = None) -> None:
        if not isinstance(pin, SparkPin):
            raise PinError(f"the {HOST} runtime serves a Spark pin")
        super().__init__(pin, transport=transport)

    def stream(self, request: ModelRequest, cancel: threading.Event) -> Generator[ModelEvent, None, None]:
        if carries_crisis(request):  # the chat route's own fallback answers it, as before
            yield Finish("error", "rejected")
            return
        yield from super().stream(request, cancel)

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        base = os.environ.get(meta_api.BASE_ENV, "").strip().rstrip("/") or meta_api.DEFAULT_BASE
        return f"{base}/chat/completions", bearer(os.environ.get(meta_api.KEY_ENV, "").strip())

    def _reported(self, served: Served) -> Served:
        """The API serves only the pinned model it was sent, whatever name its chunks carry."""
        return pin_served(self.pin)
