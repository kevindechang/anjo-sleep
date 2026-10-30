"""Muse Spark Open on Fireworks (MIND's experiment, Tasks 17 and 17b).

The same open-weight model as Baseten's (``baseten_api``), under the same Muse Model License, so
its exact id is cleared for training beside Baseten's (``provenance``). MIND's experiment calls it
when ``ANJO_SLEEP_MIND_SPARK`` says so (``mind.spark_provider``), and Sleep's night when
``ANJO_SLEEP_SPARK=fireworks`` (``spark_provider``), through ``baseten_api.chat``'s switch.
It's served through Fireworks' OpenAI-compatible API. The key comes from
``ANJO_FIREWORKS_API_KEY``, read at call time. ``generate`` calls the cheaper models that may write
MIND's core splits and pairs instead (``mind.apd.generators``); they are not Spark Open, and
``provenance`` doesn't clear them for Glimmer's training (``generators`` records which may train
the MIND core, and why).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final, Literal

import httpx

from anjo.sleep.model_api import ChatResult, post_chat

KEY_ENV = "ANJO_FIREWORKS_API_KEY"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
SPARK_OPEN: Final = "accounts/fireworks/models/muse-spark-open"
# Cheaper writers of MIND's core splits and pairs (``mind.apd.generators``); never Spark Open's stand-in.
GPT_OSS_120B: Final = "accounts/fireworks/models/gpt-oss-120b"
GLM_5P3_FLASH: Final = "accounts/fireworks/models/glm-5p3-flash"
DEEPSEEK_V4P1_FLASH: Final = "accounts/fireworks/models/deepseek-v4p1-flash"

SparkOpenModel = Literal["accounts/fireworks/models/muse-spark-open"]
GeneratorModel = Literal["accounts/fireworks/models/gpt-oss-120b", "accounts/fireworks/models/glm-5p3-flash",
                         "accounts/fireworks/models/deepseek-v4p1-flash"]


def chat(
    model: SparkOpenModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ChatResult:
    """One chat completion on Spark Open, its reply capped at ``max_tokens`` when given (MIND's
    calls are), sampled under ``seed`` when given (Sleep's brief retry). ``transport`` is for tests
    (``httpx.MockTransport``)."""
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
        extra=None if seed is None else {"seed": seed},
    )


def generate(
    model: GeneratorModel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    transport: httpx.BaseTransport | None = None,
    on_reply: Callable[[Mapping[str, Any]], None] | None = None,
) -> ChatResult:
    """One chat completion on a core split's generator, its reply capped at ``max_tokens``; a
    reasoning model is asked for ``reasoning_effort``, and its reasoning tokens count as output.
    ``on_reply`` sees the reply's whole JSON body (``mind.apd.probe``)."""
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
        extra=None if reasoning_effort is None else {"reasoning_effort": reasoning_effort},
        count_reasoning=True,
        on_reply=on_reply,
    )
