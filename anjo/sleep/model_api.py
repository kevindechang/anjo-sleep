"""One OpenAI-compatible chat call, shared by ``meta_api`` and ``baseten_api``.

A provider key is read from the environment at call time and goes only into the request's
``Authorization`` header. It never reaches a log line, an exception message, a traceback or a
repr: errors name the provider, the model and the HTTP status, never a response body or a
transport error's text, and they don't chain the transport's exception. No object here holds a
key.

There are no retries unless the caller passes ``Retries``: then a reply of 429 or 5xx (the
provider was busy, and charged nothing) is asked for again, ``Retries.attempts`` in all, after the
provider's ``Retry-After`` (seconds, at most ``RETRY_AFTER_MAX_S``) or the backoff in ``BACKOFF_S``.
Any other status, and a transport failure, is raised at once. Each call still counts once in what
the caller meters: a retried status carries no tokens. ``budget.py`` debits every attempt the caller
makes, failed ones included (Task 5).
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
BACKOFF_S: Final = (2.0, 4.0)  # before the second attempt, and the third, without a Retry-After
RETRY_AFTER_MAX_S: Final = 30.0  # the longest a provider's Retry-After is waited for


class ModelApiError(RuntimeError):
    """A provider call failed: missing configuration, transport error, timeout, an HTTP error
    status or a malformed reply."""


class MissingApiKey(ModelApiError):
    """The environment variable that should hold a provider key is unset or blank. A
    ``ModelApiError``, so one ``except ModelApiError`` catches every failed call."""


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model asked for. ``arguments`` is the JSON text the provider sent."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResult:
    """A chat reply. ``text`` is empty when the model only called tools. The token counts are
    the provider's, for the budget."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None


@dataclass(frozen=True)
class Retries:
    """How a call answered 429 or 5xx goes again: ``len(BACKOFF_S) + 1`` attempts in all. ``counted``
    hears each one sent again (the night log's count)."""

    sleep: Callable[[float], None] = time.sleep
    counted: Callable[[], None] = field(default=lambda: None)

    def pause(self, attempt: int, response: httpx.Response) -> float | None:
        """The wait before sending again after ``response`` on ``attempt`` (from 0); None: never again."""
        if response.status_code not in RETRY_STATUSES or attempt >= len(BACKOFF_S):
            return None
        return _retry_after(response.headers.get("retry-after")) or BACKOFF_S[attempt]


def _retry_after(value: str | None) -> float | None:
    """A Retry-After in seconds, capped; None for an HTTP date or anything else."""
    try:
        seconds = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return min(seconds, RETRY_AFTER_MAX_S) if math.isfinite(seconds) and seconds > 0 else None


def read_key(env_var: str) -> str:
    """The key in ``env_var``, or ``MissingApiKey`` naming the variable (never a value)."""
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise MissingApiKey(f"{env_var} is not set; export the provider key there before calling it")
    return key


def post_chat(
    *,
    provider: str,
    url: str,
    key_env: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_s: float,
    temperature: float | None,
    transport: httpx.BaseTransport | None,
    max_tokens: int | None = None,
    retries: Retries | None = None,
) -> ChatResult:
    """POST one chat completion to ``url`` and parse the reply. ``max_tokens`` caps the reply
    (sent as ``max_completion_tokens``); ``retries`` sends a busy provider's 429 or 5xx again."""
    body: dict[str, Any] = {"model": model, "messages": messages}
    if tools is not None:
        body["tools"] = tools
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_completion_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {read_key(key_env)}"}
    where = f"{provider} chat with {model}"
    try:
        with httpx.Client(timeout=timeout_s, transport=transport) as client:
            response = client.post(url, json=body, headers=headers)
            attempt = 0
            while retries is not None and (pause := retries.pause(attempt, response)) is not None:
                retries.counted()
                retries.sleep(pause)
                response, attempt = client.post(url, json=body, headers=headers), attempt + 1
    except httpx.TimeoutException:
        raise ModelApiError(f"{where} timed out after {timeout_s:g}s") from None
    except httpx.HTTPError as exc:
        raise ModelApiError(f"{where} failed in transport ({type(exc).__name__})") from None
    except (httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError) as exc:
        # httpx errors outside HTTPError; their text can carry the URL, so only the type is kept.
        raise ModelApiError(f"{where} failed before a reply ({type(exc).__name__})") from None
    if response.is_error:
        raise ModelApiError(f"{where} returned HTTP {response.status_code}")
    result = _parse(response, where)
    logger.info(
        "%s: HTTP %d, %d tokens in, %d out",
        where, response.status_code, result.input_tokens, result.output_tokens,
    )
    return result


def _parse(response: httpx.Response, where: str) -> ChatResult:
    try:
        data = response.json()
    except ValueError:
        raise ModelApiError(f"{where} sent a malformed reply (not JSON)") from None
    try:
        choice = data["choices"][0]
        message = choice["message"]
        usage = data["usage"]
        return ChatResult(
            text=_text(message.get("content")),
            tool_calls=tuple(_tool_call(call) for call in message.get("tool_calls") or ()),
            model=_str(data["model"]),
            input_tokens=_count(usage["prompt_tokens"]),
            output_tokens=_count(usage["completion_tokens"]),
            finish_reason=choice.get("finish_reason"),
        )
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        raise ModelApiError(f"{where} sent a malformed reply") from None


def _text(content: object) -> str:
    return "" if content is None else _str(content)


def _tool_call(call: dict[str, Any]) -> ToolCall:
    function = call["function"]
    return ToolCall(id=_str(call["id"]), name=_str(function["name"]), arguments=_str(function["arguments"]))


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("expected a token count")
    return value
