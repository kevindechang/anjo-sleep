"""Muse Spark and Muse Wonder through the Meta Model API (spec §3.2, Task 1).

Muse Spark writes and judges the sealed tests and never writes anything that trains. Muse
Wonder does the night's homework. Both go through the API's OpenAI-compatible chat completions.
The key comes from ``ANJO_META_API_KEY`` and the base URL from ``ANJO_META_API_BASE``, both read
at call time. The base URL defaults to ``DEFAULT_BASE``. Each public model name is sent as a
pinned, versioned id from ``PINNED_MODELS``, and the returned ``ChatResult`` records that id, so a
pre-registered judge run names the exact model version it used.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Final, Literal

import httpx

from anjo.sleep.model_api import ChatResult, Retries, post_chat

KEY_ENV = "ANJO_META_API_KEY"
BASE_ENV = "ANJO_META_API_BASE"
DEFAULT_BASE: Final = "https://api.meta.ai/v1"

MetaModel = Literal["muse-spark", "muse-wonder"]
PINNED_MODELS: Final[dict[MetaModel, str]] = {
    "muse-spark": "muse-spark-1.3",
    "muse-wonder": "muse-wonder-1.0",
}


def chat(
    model: MetaModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
    *,
    temperature: float | None = None,
    transport: httpx.BaseTransport | None = None,
    max_tokens: int | None = None,
    retries: Retries | None = None,
) -> ChatResult:
    """One chat completion on ``model``, sent as its pinned version, its reply capped at
    ``max_tokens`` when given; ``retries`` sends a 429 or 5xx again (the morning's judge).
    ``transport`` is for tests (``httpx.MockTransport``)."""
    base = os.environ.get(BASE_ENV, "").strip().rstrip("/") or DEFAULT_BASE
    pinned = PINNED_MODELS[model]
    result = post_chat(
        provider="Meta Model API",
        url=f"{base}/chat/completions",
        key_env=KEY_ENV,
        model=pinned,
        messages=messages,
        tools=tools,
        timeout_s=timeout_s,
        temperature=temperature,
        transport=transport,
        max_tokens=max_tokens,
        retries=retries,
    )
    return replace(result, model=pinned)
