# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The dev API transport (``anjo.sleep.app.sim_transport``), which needs private Anjo, so it lives
with the adapter tests the public repo doesn't collect."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from anjo.sleep.app import sim_transport
from anjo.sleep.app.sim_transport import MAX_BUSY_WAIT_S, MAX_RETRY_AFTER_S, DevApiTransport
from anjo.sleep.sim.day import TurnFailed
from anjo.sleep.tools import sleep_objective_live


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


def _whoami(answer: httpx.Response, seen: list[tuple[str, str]]) -> httpx.MockTransport:
    """A dev API that answers its who-am-I read with ``answer``; every call is recorded."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return answer if request.url.path == sim_transport.WHOAMI else httpx.Response(500)

    return httpx.MockTransport(handler)


def test_simulate_refuses_a_token_for_another_account(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []
    other = httpx.Response(200, json={"payments_enabled": False, "user_id": "syn-other", "products": {}})
    real = sim_transport.DevApiTransport
    monkeypatch.setattr(sim_transport, "DevApiTransport", lambda url, token: real(url, token, transport=_whoami(other, seen)))
    monkeypatch.setattr(sim_transport, "require_sleep_host", lambda: None)
    monkeypatch.setenv(sim_transport.DEV_URL_ENV, "http://127.0.0.1:8000")
    monkeypatch.setenv(sim_transport.SIM_TOKEN_ENV, "tok-other")

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("the day started")

    for name in ("Ledger", "run_day", "night_lock"):  # no hold, no wake, no line
        monkeypatch.setattr(sim_transport, name, never)
    with pytest.raises(sim_transport.TokenRefused, match="another account") as raised:
        sim_transport.simulate("syn-a", "founder", 0, 7)
    assert "tok-other" not in str(raised.value) and "syn-other" not in str(raised.value)
    assert seen == [("GET", sim_transport.WHOAMI)]


def test_the_account_check_passes_only_its_own_token() -> None:
    seen: list[tuple[str, str]] = []
    DevApiTransport("http://x", "t", transport=_whoami(httpx.Response(200, json={"user_id": "syn-a"}), seen)
                    ).check_account("syn-a")
    for answer in (httpx.Response(401, json={"detail": "Not authenticated"}), httpx.Response(200, text="not json"),
                   httpx.Response(200, json={"user_id": "syn-b"})):
        with pytest.raises(sim_transport.TokenRefused):
            DevApiTransport("http://x", "t", transport=_whoami(answer, seen)).check_account("syn-a")
    assert all(method == "GET" for method, _ in seen)


def test_the_live_comparison_sets_the_token_simulate_reads() -> None:
    assert sleep_objective_live.SIM_TOKEN_ENV == sim_transport.SIM_TOKEN_ENV


# ── several lives in one process: each account's own token (the final test's batch) ─


def _env(monkeypatch: pytest.MonkeyPatch, **names: str) -> None:
    for name in (sim_transport.SIM_TOKEN_FILES_ENV, sim_transport.SIM_TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in names.items():
        monkeypatch.setenv(getattr(sim_transport, name), value)


def test_each_account_named_in_the_token_map_reads_its_own_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a").write_text("tok-a", encoding="utf-8")
    (tmp_path / "b").write_text("tok-b\n", encoding="utf-8")
    _env(monkeypatch, SIM_TOKEN_FILES_ENV=f"syn-a={tmp_path / 'a'}, syn-b={tmp_path / 'b'},",
         SIM_TOKEN_ENV="tok-single")
    assert [sim_transport.sim_token(uid) for uid in ("syn-a", "syn-b")] == ["tok-a", "tok-b"]
    assert sim_transport.sim_token("syn-c") == "tok-single"  # an account the map doesn't name: as before


def test_one_life_reads_the_single_token_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, SIM_TOKEN_ENV="tok")
    assert sim_transport.sim_token("syn-a") == "tok"
    _env(monkeypatch)
    with pytest.raises(KeyError):  # unset, as it always was: no day starts
        sim_transport.sim_token("syn-a")


@pytest.mark.parametrize("mapping", ["syn-a", "syn-a=", "=/x", "syn-a=/x,syn-a=/y"])
def test_a_token_map_it_cant_read_refuses_the_day_quoting_nothing(mapping: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, SIM_TOKEN_FILES_ENV=mapping, SIM_TOKEN_ENV="t")
    with pytest.raises(sim_transport.TokenRefused) as raised:
        sim_transport.sim_token("syn-a")
    assert "syn-a" not in str(raised.value) and "/x" not in str(raised.value)


def test_a_token_file_that_is_missing_or_empty_refuses_the_day_naming_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "empty").write_text("\n", encoding="utf-8")
    for path in (tmp_path / "missing", tmp_path / "empty"):
        _env(monkeypatch, SIM_TOKEN_FILES_ENV=f"syn-a={path}", SIM_TOKEN_ENV="tok-single")
        with pytest.raises(sim_transport.TokenRefused) as raised:
            sim_transport.sim_token("syn-a")
        assert str(tmp_path) not in str(raised.value) and "syn-a" not in str(raised.value)
