"""One OpenAI-compatible chat call, shared by ``meta_api`` and ``baseten_api``.

A provider key is read from the environment at call time and goes only into the request's
``Authorization`` header. It never reaches a log line, an exception message, a traceback or a
repr: errors name the provider, the model and the HTTP status, never a response body or a
transport error's text, and they don't chain the transport's exception. No object here holds a
key.

There are no retries. The caller retries if it wants to, and ``budget.py`` debits every attempt,
failed ones included (Task 5).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
) -> ChatResult:
    """POST one chat completion to ``url`` and parse the reply. ``max_tokens`` caps the reply
    (sent as ``max_completion_tokens``)."""
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
