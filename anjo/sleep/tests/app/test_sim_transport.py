# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The dev API transport (``anjo.sleep.app.sim_transport``), which needs private Anjo, so it lives
with the adapter tests the public repo doesn't collect."""
from __future__ import annotations

import json

import httpx
import pytest

from anjo.sleep.app.sim_transport import MAX_BUSY_WAIT_S, MAX_RETRY_AFTER_S, DevApiTransport
from anjo.sleep.sim.day import TurnFailed


def test_the_dev_api_transport_reads_the_stream_and_fails_a_turn_it_cannot_answer() -> None:
    calls: list[str] = []
    down = {"now": False}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer t"
        if request.url.path == "/api/sleep/dev/sessions":
            return httpx.Response(200, json={"session_id": "s-1", "cleared": True})
        if down["now"]:
            return httpx.Response(200, text='event: error\ndata: {"error": "unavailable"}\n\n')
        return httpx.Response(200, text='event: token\ndata: {"text": "Start "}\n\nevent: token\ndata: {"text": "small."}\n\n'
                                        'event: done\ndata: {}\n\n')

    transport = DevApiTransport("http://127.0.0.1:8000", "t", transport=httpx.MockTransport(handler))
    assert transport.send("sim-founder-d1", "Where do I start?", cleared=True) == "Start small."
    down["now"] = True
    with pytest.raises(TurnFailed):
        transport.send("sim-founder-d1", "And then?", cleared=True)
    assert calls == ["/api/sleep/dev/sessions", "/api/chat/s-1/message", "/api/chat/s-1/message"]


def _server(chat: list[httpx.Response]) -> tuple[httpx.MockTransport, list[tuple[str, str]]]:
    """A dev API whose chat route answers from ``chat`` in order; every call is recorded."""
    seen: list[tuple[str, str]] = []
    answers = iter(chat)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/sleep/dev/sessions":
            return httpx.Response(200, json={"session_id": "s-1", "cleared": True})
        seen.append((request.url.path, json.loads(request.content)["text"]))
        return next(answers)

    return httpx.MockTransport(handler), seen


_OK = httpx.Response(200, text='event: token\ndata: {"text": "Fine."}\n\nevent: done\ndata: {}\n\n')
_BUSY = {"error": "Another response is already in progress.", "retry": True, "session_id": "s-1"}


def test_busy_409_is_waited_out_and_resent() -> None:
    slept: list[float] = []
    mock, seen = _server([httpx.Response(409, json=_BUSY, headers={"Retry-After": "1"}),
                          httpx.Response(409, json=_BUSY), _OK])
    transport = DevApiTransport("http://x", "t", transport=mock, sleep=slept.append)
    assert transport.send("sim-d1", "Hello there", cleared=True) == "Fine."
    assert seen == [("/api/chat/s-1/message", "Hello there")] * 3
    assert slept[0] == 1.0 and len(slept) == 2 and all(0 < s <= MAX_RETRY_AFTER_S for s in slept)


def test_busy_409_past_the_bound_fails_the_turn() -> None:
    slept: list[float] = []
    mock, seen = _server([httpx.Response(409, json=_BUSY, headers={"Retry-After": "600"})] * 100)
    transport = DevApiTransport("http://x", "t", transport=mock, sleep=slept.append)
    with pytest.raises(TurnFailed, match="busy"):
        transport.send("sim-d1", "Hello there", cleared=True)
    assert sum(slept) <= MAX_BUSY_WAIT_S and max(slept) <= MAX_RETRY_AFTER_S
    assert len(seen) < 100


def test_stale_session_409_is_not_adopted() -> None:
    stale = httpx.Response(409, json={"error": "Session expired. Please restart this chat session.",
                                      "session_id": "other"})
    mock, seen = _server([stale, _OK])
    transport = DevApiTransport("http://x", "t", transport=mock, sleep=lambda s: None)
    with pytest.raises(TurnFailed, match="stale session") as raised:
        transport.send("sim-d1", "Hello there", cleared=True)
    assert "Session expired" not in str(raised.value) and "other" not in str(raised.value)
    assert raised.value.reason == "stale session"
    assert seen == [("/api/chat/s-1/message", "Hello there")]


def test_keyed_refusal_409_fails_with_its_code() -> None:
    spent = httpx.Response(409, json={"error": "This message was already answered.", "code": "request_key_spent"})
    mock, seen = _server([spent, _OK])
    transport = DevApiTransport("http://x", "t", transport=mock, sleep=lambda s: None)
    with pytest.raises(TurnFailed, match="request_key_spent") as raised:
        transport.send("sim-d1", "Hello there", cleared=True)
    assert "already answered" not in str(raised.value) and raised.value.reason == "request_key_spent"
    assert len(seen) == 1
