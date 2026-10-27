# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The Meta Model API and Baseten clients: one chat call each, keys from the environment only.

Every test runs on httpx.MockTransport, so nothing opens a socket.
"""
from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from anjo.sleep import baseten_api, meta_api
from anjo.sleep.model_api import ChatResult, MissingApiKey, ModelApiError

FAKE_KEY = "sk-fake-0123456789abcdef-never-leaks"
META_BASE = "https://meta.example.test/v1"
MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
TOOLS: list[dict[str, Any]] = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

Handler = Callable[[httpx.Request], httpx.Response]
Call = Callable[[httpx.BaseTransport | None], ChatResult]


def _ok(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "hi there",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "lenses"}'},
                        },
                    ],
                },
            },
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


def _meta(transport: httpx.BaseTransport | None) -> ChatResult:
    return meta_api.chat("muse-spark", MESSAGES, TOOLS, 30.0, temperature=0.0, transport=transport)


def _baseten(transport: httpx.BaseTransport | None) -> ChatResult:
    return baseten_api.chat(baseten_api.SPARK_OPEN, MESSAGES, None, 30.0, transport=transport)


CLIENTS: dict[str, tuple[str, Call]] = {
    "meta": (meta_api.KEY_ENV, _meta),
    "baseten": (baseten_api.KEY_ENV, _baseten),
}


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(meta_api.KEY_ENV, FAKE_KEY)
    monkeypatch.setenv(baseten_api.KEY_ENV, FAKE_KEY)
    monkeypatch.setenv(meta_api.BASE_ENV, META_BASE)


def _recording(seen: list[httpx.Request], respond: Handler) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request)

    return httpx.MockTransport(handler)


def test_meta_chat_sends_an_openai_style_request_and_parses_the_reply() -> None:
    seen: list[httpx.Request] = []
    result = _meta(_recording(seen, lambda r: httpx.Response(200, json=_ok("muse-spark"))))

    (request,) = seen
    assert str(request.url) == f"{META_BASE}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body == {"model": "muse-spark", "messages": MESSAGES, "tools": TOOLS, "temperature": 0.0}
    assert result.text == "hi there"
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        ("call_1", "search", '{"q": "lenses"}'),
    ]
    assert (result.model, result.input_tokens, result.output_tokens) == ("muse-spark", 12, 3)
    assert result.finish_reason == "tool_calls"


def test_baseten_chat_calls_spark_open_on_its_openai_compatible_endpoint() -> None:
    seen: list[httpx.Request] = []
    result = _baseten(_recording(seen, lambda r: httpx.Response(200, json=_ok(baseten_api.SPARK_OPEN))))

    (request,) = seen
    assert request.url.host == "inference.baseten.co"
    assert request.url.path == "/v1/chat/completions"
    body = json.loads(request.content)
    assert body == {"model": "meta-models/muse-spark-open-1026", "messages": MESSAGES}
    assert result.model == "meta-models/muse-spark-open-1026"


def test_meta_base_url_must_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(meta_api.BASE_ENV)
    with pytest.raises(ModelApiError, match=meta_api.BASE_ENV):
        _meta(httpx.MockTransport(lambda r: httpx.Response(200, json=_ok("muse-spark"))))


@pytest.mark.parametrize("client", sorted(CLIENTS))
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_fails_cleanly_before_any_request(
    client: str, value: str | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_env, call = CLIENTS[client]
    if value is None:
        monkeypatch.delenv(key_env)
    else:
        monkeypatch.setenv(key_env, value)
    seen: list[httpx.Request] = []
    with pytest.raises(MissingApiKey) as caught:
        call(_recording(seen, lambda r: httpx.Response(200, json={})))
    assert seen == []
    assert key_env in str(caught.value)
    assert FAKE_KEY not in str(caught.value)


def _fail_with_status(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"error": {"message": f"bad key {FAKE_KEY}"}})


def _fail_with_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout(f"timed out; headers {dict(request.headers)}", request=request)


def _fail_with_garbage(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=f"<html>{FAKE_KEY}</html>")


def _fail_with_shape(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [], "echo": FAKE_KEY})


def _assert_no_key(text: str) -> None:
    assert FAKE_KEY not in text
    assert FAKE_KEY[-12:] not in text


@pytest.mark.parametrize("client", sorted(CLIENTS))
def test_api_keys_never_logged(client: str, caplog: pytest.LogCaptureFixture) -> None:
    """A fake key never shows in logs, exception text, tracebacks or reprs, for both clients."""
    caplog.set_level(logging.DEBUG)
    _, call = CLIENTS[client]

    result = call(httpx.MockTransport(lambda r: httpx.Response(200, json=_ok("m"))))
    _assert_no_key(repr(result))
    _assert_no_key(str(result))

    for failure in (_fail_with_status, _fail_with_timeout, _fail_with_garbage, _fail_with_shape):
        with pytest.raises(ModelApiError) as caught:
            call(httpx.MockTransport(failure))
        _assert_no_key(str(caught.value))
        _assert_no_key(repr(caught.value))
        _assert_no_key("".join(traceback.format_exception(caught.value)))

    assert caplog.records, "the client logs each call, so the check above is not vacuous"
    _assert_no_key(caplog.text)
    _assert_no_key(" ".join(repr(r.__dict__) for r in caplog.records))


def test_error_status_is_reported_without_the_body() -> None:
    with pytest.raises(ModelApiError, match="HTTP 401") as caught:
        _baseten(httpx.MockTransport(_fail_with_status))
    assert "bad key" not in str(caught.value)


def test_timeout_names_the_limit() -> None:
    with pytest.raises(ModelApiError, match="timed out after 30s"):
        _baseten(httpx.MockTransport(_fail_with_timeout))


def test_reply_without_usage_is_refused_so_spend_is_never_undercounted() -> None:
    reply = _ok("m")
    del reply["usage"]
    with pytest.raises(ModelApiError, match="malformed"):
        _baseten(httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))


def test_reply_with_no_text_and_no_tools_parses_to_empty() -> None:
    reply = _ok("m")
    reply["choices"][0]["message"] = {"role": "assistant", "content": None}
    result = _baseten(httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))
    assert (result.text, result.tool_calls) == ("", ())
