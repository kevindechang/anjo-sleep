"""Muse Spark Open on Baseten (spec §3.2, Task 1), and Sleep's way to another provider.

Spark Open plays the person and the critic in practice, compiles the brief, extracts evidence
and runs the simulator, so every training ancestor is licence-cleared (§3). It's served through
Baseten's OpenAI-compatible Model API. The key comes from ``ANJO_BASETEN_API_KEY``, read at call
time.

**One switch.** ``ANJO_SLEEP_SPARK`` selects the provider (``spark_provider``); when it selects the
other one, a call asking for ``SPARK_OPEN`` is sent to ``fireworks_api`` instead, with the same
messages, tools, temperature, cap and seed, and its reply names that provider's id. The other
provider's own retry policy goes with it (Fireworks sends a dropped connection again; Baseten
passes none, so its calls are exactly as before). Unset or ``baseten`` calls Baseten exactly as
before, so ``chat`` is the one place Sleep's Spark Open traffic turns.
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
    seed: int | None = None,
) -> ChatResult:
    """One chat completion on Spark Open, its reply capped at ``max_tokens`` when given (full
    practice caps each role, the brief its folds), sampled under ``seed`` when given (the brief's
    retry). ``transport`` is for tests (``httpx.MockTransport``). When ``ANJO_SLEEP_SPARK`` selects
    the other provider, a call for ``SPARK_OPEN`` goes to ``fireworks_api`` instead."""
    if model == SPARK_OPEN:
        from anjo.sleep import spark_provider  # imported here: spark_provider imports this module

        selected = spark_provider.current()
        if selected.model != SPARK_OPEN:
            # A capped reply waits what its cap needs at a slow but healthy decode speed on the
            # serverless provider (a practice scene capped at thousands of tokens is not cut at 180 s).
            if max_tokens is not None:
                from anjo.sleep.mind.apd.retry import reply_timeout_s

                timeout_s = max(timeout_s, reply_timeout_s(max_tokens))
            return selected.chat(selected.model, messages, tools, timeout_s, temperature=temperature,
                                 max_tokens=max_tokens, transport=transport, seed=seed, retries=selected.retries)
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
        extra=None if seed is None else {"seed": seed},
    )
