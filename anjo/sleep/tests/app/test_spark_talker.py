# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""Muse Spark as the judges' fallback talker: the Meta Model API's chat completions, its pin off the
N-PROC vocabulary, and a crisis turn refused so the chat route's own fallback answers it as before.
The API is a fake transport: no Meta call."""
from __future__ import annotations

import threading
from typing import Any

import pytest

from anjo.sleep import meta_api
from anjo.sleep.app import spark_talker
from anjo.sleep.tests.app.glimmer_server import Upstream


class FakeApi:
    def __init__(self, *chunks: dict[str, Any]) -> None:
        self.chunks = list(chunks)
        self.opened: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def open(self, url: str, headers: Any, body: Any) -> Upstream:
        self.opened.append((url, dict(headers), dict(body)))
        return Upstream(self.chunks)

    def close(self) -> None:
        return None


def _request(text: str) -> Any:
    from anjo.core.model_io import Message, ModelRequest, TextPart, ToolChoice

    return ModelRequest(turn_id="t1", system=(), messages=(Message("user", (TextPart(text),)),),
                        max_output_tokens=64, round_index=0, tools=(), tool_choice=ToolChoice("auto"))


def _reply(text: str) -> list[dict[str, Any]]:
    return [
        {"id": "c", "model": "whatever-the-api-says", "choices": [{"index": 0, "delta": {"content": text}}]},
        {"id": "c", "model": "whatever-the-api-says", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]


def test_the_pin_names_no_family_and_serves_the_pinned_spark() -> None:
    pin = spark_talker.spark_pin("judge-1")
    assert (pin.host, pin.family, pin.model_id) == ("sleep_spark", None, meta_api.PINNED_MODELS["muse-spark"])
    assert pin.traits.tool_calls == "none"
    assert pin.identity()["family"] is None


def test_it_streams_from_the_meta_api_with_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core.model_io import Finish, TextDelta

    monkeypatch.setenv(meta_api.KEY_ENV, "test-meta-key")
    monkeypatch.setenv(meta_api.BASE_ENV, "https://meta.test/v1/")
    api = FakeApi(*_reply("Hello there"))
    runtime = spark_talker.SparkRuntime(spark_talker.spark_pin("judge-1"), transport=api)
    events = list(runtime.stream(_request("hey"), threading.Event()))
    url, headers, body = api.opened[0]
    assert url == "https://meta.test/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-meta-key"
    assert body["model"] == meta_api.PINNED_MODELS["muse-spark"] and "tools" not in body
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello there"
    assert isinstance(events[-1], Finish) and events[-1].reason == "stop"


def test_a_crisis_turn_never_reaches_spark(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core.model_io import Finish
    from anjo.sleep.tests.app._serving_fakes import _crisis_case

    monkeypatch.setenv(meta_api.KEY_ENV, "test-meta-key")
    api = FakeApi(*_reply("no"))
    runtime = spark_talker.SparkRuntime(spark_talker.spark_pin("judge-1"), transport=api)
    assert list(runtime.stream(_request(_crisis_case("hard")), threading.Event())) == [Finish("error", "rejected")]
    assert api.opened == []


def test_configured_only_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(meta_api.KEY_ENV, raising=False)
    assert not spark_talker.spark_configured()
    monkeypatch.setenv(meta_api.KEY_ENV, " k ")
    assert spark_talker.spark_configured()


def test_a_turn_asks_for_minimal_reasoning_with_room_above_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    # Spark always reasons before it answers (~110 tokens at minimal, ~400-700 at low), and those tokens
    # count against max_tokens: a bare reply cap ended judge turns with no words at all.
    monkeypatch.setenv(meta_api.KEY_ENV, "test-meta-key")
    api = FakeApi(*_reply("Hi"))
    runtime = spark_talker.SparkRuntime(spark_talker.spark_pin("judge-1"), transport=api)
    list(runtime.stream(_request("hey"), threading.Event()))
    body = api.opened[0][2]
    assert body["reasoning_effort"] == "minimal"
    assert body["max_tokens"] == 64 + spark_talker.REASONING_HEADROOM_TOKENS
