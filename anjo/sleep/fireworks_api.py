"""Muse Spark Open on Fireworks (MIND's experiment, Tasks 17 and 17b).

The same open-weight model as Baseten's (``baseten_api``), under the same Muse Model License, so
its exact id is cleared for training beside Baseten's (``provenance``). MIND's experiment calls it
when ``ANJO_SLEEP_MIND_SPARK`` says so (``mind.spark_provider``); Sleep's night stays on Baseten.
It's served through Fireworks' OpenAI-compatible API. The key comes from
``ANJO_FIREWORKS_API_KEY``, read at call time.
"""
from __future__ import annotations

from typing import Any, Final, Literal

import httpx

from anjo.sleep.model_api import ChatResult, post_chat

KEY_ENV = "ANJO_FIREWORKS_API_KEY"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SPARK_OPEN: Final = "accounts/fireworks/models/muse-spark-open"

SparkOpenModel = Literal["accounts/fireworks/models/muse-spark-open"]


def chat(
    model: SparkOpenModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ChatResult:
    """One chat completion on Spark Open, its reply capped at ``max_tokens`` when given (MIND's
    calls are). ``transport`` is for tests (``httpx.MockTransport``)."""
    return post_chat(
        provider="Fireworks",
        url=URL,
        key_env=KEY_ENV,
        model=model,
        messages=messages,
        tools=tools,
        timeout_s=timeout_s,
        temperature=temperature,
        transport=transport,
        max_tokens=max_tokens,
    )
