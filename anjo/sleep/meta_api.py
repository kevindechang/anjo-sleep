"""Muse Spark and Muse Wonder through the Meta Model API (spec §3.2, Task 1).

Muse Spark writes and judges the sealed tests and never writes anything that trains. Muse
Wonder does the night's homework. Both go through the API's OpenAI-compatible chat completions.
The key comes from ``ANJO_META_API_KEY`` and the base URL from ``ANJO_META_API_BASE`` (for
example ``https://<host>/v1``), both read at call time.
"""
from __future__ import annotations

import os
from typing import Any, Literal

import httpx

from anjo.sleep.model_api import ChatResult, ModelApiError, post_chat

KEY_ENV = "ANJO_META_API_KEY"
BASE_ENV = "ANJO_META_API_BASE"

MetaModel = Literal["muse-spark", "muse-wonder"]


def chat(
    model: MetaModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
    *,
    temperature: float | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ChatResult:
    """One chat completion on ``model``. ``transport`` is for tests (``httpx.MockTransport``)."""
    base = os.environ.get(BASE_ENV, "").strip().rstrip("/")
    if not base:
        raise ModelApiError(f"{BASE_ENV} is not set; export the Meta Model API base URL there")
    return post_chat(
        provider="Meta Model API",
        url=f"{base}/chat/completions",
        key_env=KEY_ENV,
        model=model,
        messages=messages,
        tools=tools,
        timeout_s=timeout_s,
        temperature=temperature,
        transport=transport,
    )
