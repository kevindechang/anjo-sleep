"""One OpenAI-compatible chat call, shared by ``meta_api`` and ``baseten_api``.

A provider key is read from the environment at call time and goes only into the request's
``Authorization`` header. It never reaches a log line, an exception message, a traceback or a
repr: errors name the provider, the model and the HTTP status, never a response body or a
transport error's text, and they don't chain the transport's exception. No object here holds a
key.

There are no retries unless the caller passes ``Retries``: then a reply of 429 or 5xx (the
provider was busy, and charged nothing) is asked for again, one attempt per entry in the policy's
``backoff_s`` (``BACKOFF_S`` unless the caller names its own schedule), after the provider's
``Retry-After`` (seconds, at most ``RETRY_AFTER_MAX_S``) or that entry. A call whose one failure
costs a whole night may wait longer than the shared schedule (the night's Spark Open call does).
A connection that never opened (``httpx.ConnectError`` or ``ConnectTimeout``) is sent again too, up
to ``Retries.connect_attempts`` in all: the request never reached the provider, so nothing was
charged. A request sent whose reply never came whole (a read timeout, a dropped or broken read) is
sent again only when ``Retries.reply_attempts`` says so, and that resend may be charged twice: only
a caller whose single failure costs a whole night or morning opts in, so 1, the default, never sends
one again. Any other status, and any other transport failure, is raised at once. Each call still
counts once in what the caller meters: a retried status carries no tokens. ``budget.py`` debits
every attempt the caller makes, failed ones included (Task 5).
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Mapping
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


class RateLimited(ModelApiError):
    """A provider answered 429 (it was busy, and charged nothing); ``retry_after_s`` is its
    ``Retry-After`` in seconds, uncapped, when it gave one."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ConnectFailed(ModelApiError):
    """A connection that never opened: refused, its host's name not found, its network unreachable
    or its connect timing out. The request never reached the provider, so nothing was charged, and
    a caller may send it again when its policy allows it."""


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
    """How a call answered 429 or 5xx goes again: ``len(backoff_s) + 1`` attempts in all. A
    connection that never opened goes again up to ``connect_attempts`` in all (1: only the first
    try), waiting the backoff in ``backoff_s``; a reply that never came whole goes again up to
    ``reply_attempts`` in all (1: never), the same way. Such a resend may be charged twice, so only
    a caller whose single failure costs a whole night or morning sets ``reply_attempts`` above 1.
    ``counted`` hears each one sent again (the night log's count). ``backoff_s`` is the shared
    schedule unless a caller whose one refusal costs a whole day names a longer one; a 429 or 5xx
    was charged nothing, so waiting longer spends wall time, never the ledger."""

    sleep: Callable[[float], None] = time.sleep
    counted: Callable[[], None] = field(default=lambda: None)
    connect_attempts: int = 1
    reply_attempts: int = 1
    backoff_s: tuple[float, ...] = BACKOFF_S  # before each attempt after a busy reply, in order

    def pause(self, attempt: int, response: httpx.Response) -> float | None:
        """The wait before sending again after ``response`` on ``attempt`` (from 0); None: never again."""
        if response.status_code not in RETRY_STATUSES or attempt >= len(self.backoff_s):
            return None
        return _retry_after(response.headers.get("retry-after")) or self.backoff_s[attempt]


def _retry_after(value: str | None) -> float | None:
    """A Retry-After in seconds, capped; None for an HTTP date or anything else."""
    seconds = _seconds(value)
    return None if seconds is None else min(seconds, RETRY_AFTER_MAX_S)


def _seconds(value: str | None) -> float | None:
    """A Retry-After in seconds; None for an HTTP date or anything else."""
    try:
        seconds = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


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
    extra: Mapping[str, Any] | None = None,
    count_reasoning: bool = False,
    on_reply: Callable[[Mapping[str, Any]], None] | None = None,
) -> ChatResult:
    """POST one chat completion to ``url`` and parse the reply. ``max_tokens`` caps the reply
    (sent as ``max_completion_tokens``); ``retries`` sends a busy provider's 429 or 5xx again.
    ``extra`` adds fields to the request (a reasoning model's effort); ``count_reasoning`` counts
    every generated token as output, reasoning included however the provider reports it;
    ``on_reply`` is handed the parsed reply's whole JSON body (a probe of its shape)."""
    body: dict[str, Any] = {**(extra or {}), "model": model, "messages": messages}
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
            response = _send(client, url, body, headers, where=where, retries=retries)
            attempt = 0
            while retries is not None and (pause := retries.pause(attempt, response)) is not None:
                retries.counted()
                retries.sleep(pause)
                response, attempt = _send(client, url, body, headers, where=where, retries=retries), attempt + 1
    except httpx.ConnectTimeout:  # no connection made in time: a failed connection, not a slow reply
        raise ConnectFailed(f"{where} failed in transport (ConnectTimeout)") from None
    except httpx.ConnectError:
        raise ConnectFailed(f"{where} failed in transport (ConnectError)") from None
    except httpx.TimeoutException:
        raise ModelApiError(f"{where} timed out after {timeout_s:g}s") from None
    except httpx.HTTPError as exc:
        raise ModelApiError(f"{where} failed in transport ({type(exc).__name__})") from None
    except (httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError) as exc:
        # httpx errors outside HTTPError; their text can carry the URL, so only the type is kept.
        raise ModelApiError(f"{where} failed before a reply ({type(exc).__name__})") from None
    if response.status_code == 429:
        raise RateLimited(f"{where} returned HTTP 429", retry_after_s=_seconds(response.headers.get("retry-after")))
    if response.is_error:
        raise ModelApiError(f"{where} returned HTTP {response.status_code}")
    result = _parse(response, where, count_reasoning=count_reasoning)
    if on_reply is not None:
        on_reply(response.json())
    logger.info(
        "%s: HTTP %d, %d tokens in, %d out",
        where, response.status_code, result.input_tokens, result.output_tokens,
    )
    return result


def _send(
    client: httpx.Client, url: str, body: Mapping[str, Any], headers: Mapping[str, str], *,
    where: str, retries: Retries | None,
) -> httpx.Response:
    """POST once, sending again when the connection never opened (``Retries.connect_attempts`` in
    all) or, only when the caller asks for it, when the reply never came whole
    (``Retries.reply_attempts`` in all). A connection that never opened was charged nothing; a reply
    too slow or dropped mid-read may have been, so it goes again only because that caller's one
    failure costs a whole night or morning. The last failure is raised as it came, so ``post_chat``
    names it exactly as it always has."""
    connects = 1 if retries is None else max(1, retries.connect_attempts)
    replies = 1 if retries is None else max(1, retries.reply_attempts)
    connect_sends = reply_sends = 0  # resends after each kind of failure, capped above
    while True:
        try:
            return client.post(url, json=body, headers=headers)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if connect_sends + 1 >= connects:
                raise ConnectFailed(f"{where} failed in transport ({type(exc).__name__})") from None
            assert retries is not None  # connects > 1 only with a policy
            retries.counted()
            retries.sleep(retries.backoff_s[min(connect_sends, len(retries.backoff_s) - 1)])
            connect_sends += 1
        except httpx.HTTPError as exc:
            if reply_sends + 1 >= replies or not _reply_lost(exc):
                raise
            assert retries is not None  # replies > 1 only with a policy
            retries.counted()
            retries.sleep(retries.backoff_s[min(reply_sends, len(retries.backoff_s) - 1)])
            reply_sends += 1


def _reply_lost(exc: httpx.HTTPError) -> bool:
    """A request sent whose reply never came whole: a timeout other than a connection's, a read
    dropped, or a protocol broken mid-reply. The provider may have charged it; ``post_chat`` still
    names it as it always has when it is the last try."""
    return isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError)) or (
        isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout))


def _parse(response: httpx.Response, where: str, *, count_reasoning: bool = False) -> ChatResult:
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
            output_tokens=_generated(usage) if count_reasoning else _count(usage["completion_tokens"]),
            finish_reason=choice.get("finish_reason"),
        )
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        raise ModelApiError(f"{where} sent a malformed reply") from None


def _generated(usage: Mapping[str, Any]) -> int:
    """Every token generated, reasoning included: inside ``completion_tokens`` (OpenAI's way), or
    reported apart from it, in ``total_tokens`` or as more reasoning than the completion holds."""
    completion = _count(usage["completion_tokens"])
    reasoning = _count((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    total = usage.get("total_tokens")
    beyond = 0 if total is None else _count(total) - _count(usage["prompt_tokens"])
    return max(completion, beyond, completion + reasoning if reasoning > completion else 0)


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
