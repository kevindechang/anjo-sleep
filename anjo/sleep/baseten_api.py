"""Muse Spark Open on Baseten (spec §3.2, Task 1).

Spark Open plays the person and the critic in practice, compiles the brief, extracts evidence
and runs the simulator, so every training ancestor is licence-cleared (§3). It's served through
Baseten's OpenAI-compatible Model API. The key comes from ``ANJO_BASETEN_API_KEY``, read at call
time.
"""
from __future__ import annotations

from typing import Any, Final, Literal

import httpx

from anjo.sleep.model_api import ChatResult, post_chat

KEY_ENV = "ANJO_BASETEN_API_KEY"
URL = "https://inference.baseten.co/v1/chat/completions"
SPARK_OPEN: Final = "meta-models/muse-spark-open-1026"

SparkOpenModel = Literal["meta-models/muse-spark-open-1026"]


def chat(
    model: SparkOpenModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
    *,
    temperature: float | None = None,
    transport: httpx.BaseTransport | None = None,
    max_tokens: int | None = None,
) -> ChatResult:
    """One chat completion on Spark Open, its reply capped at ``max_tokens`` when given (full
    practice caps each role). ``transport`` is for tests (``httpx.MockTransport``)."""
    return post_chat(
        provider="Baseten",
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
