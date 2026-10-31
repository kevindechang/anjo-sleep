# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Sleep's Spark Open on Fireworks survives a dropped connection and a long practice reply.

A connection that never opened was charged nothing, so the delegated Spark Open call sends it again
a bounded number of times. A reply that never came whole is sent again only when the caller's policy
asks for it (extraction and the morning judge do: one lost reply there costs a whole run), because a
resend may have been charged twice. A capped reply's wait is scaled to its cap, so a practice scene
isn't cut at 180 s. Fakes only: ``httpx.MockTransport`` and a fake ``post_chat``, no network."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

import httpx
import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.mind.apd.retry import reply_timeout_s
from anjo.sleep.model_api import BACKOFF_S, RETRY_AFTER_MAX_S, ChatResult, ConnectFailed, ModelApiError, Retries

MESSAGES: Final = [{"role": "user", "content": "hello"}]


def _ok(model: str) -> dict[str, Any]:
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-fake")


def _chat(transport: httpx.BaseTransport, *, retries: Retries | None = None, timeout_s: float = 180.0,
          max_tokens: int | None = None) -> ChatResult:
    return fireworks_api.chat(fireworks_api.SPARK_OPEN, MESSAGES, None, timeout_s, transport=transport,
                              retries=retries, max_tokens=max_tokens)


def test_a_connection_that_never_opened_is_sent_again_then_succeeds() -> None:
    seen: list[httpx.Request] = []
    slept: list[float] = []
    counted: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) < 3:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=_ok(fireworks_api.SPARK_OPEN))

    retries = Retries(connect_attempts=3, sleep=slept.append, counted=lambda: counted.append(1))
    result = _chat(httpx.MockTransport(handler), retries=retries)
    assert result.text == "ok" and len(seen) == 3
    assert slept == list(BACKOFF_S) and counted == [1, 1]


def test_a_connection_that_never_opens_is_sent_a_bounded_number_of_times() -> None:
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    retries = Retries(connect_attempts=3, sleep=slept.append)
    with pytest.raises(ConnectFailed, match="ConnectError"):
        _chat(httpx.MockTransport(handler), retries=retries)
    assert len(seen) == 3 and slept == list(BACKOFF_S)  # no fourth try, no wait after the last


def test_a_lost_reply_policy_never_lengthens_the_connections_tries() -> None:
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    retries = Retries(connect_attempts=2, reply_attempts=2, sleep=slept.append)
    with pytest.raises(ConnectFailed, match="ConnectError"):
        _chat(httpx.MockTransport(handler), retries=retries)
    assert len(seen) == 2 and slept == [BACKOFF_S[0]]


def test_without_a_policy_a_dropped_connection_fails_at_once() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ConnectFailed):
        _chat(httpx.MockTransport(handler))
    assert len(seen) == 1


def test_a_read_timeout_is_never_sent_again_even_with_a_policy() -> None:
    """The request reached the provider and may have been charged: only a connect failure retries."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ReadTimeout("slow reply", request=request)

    with pytest.raises(ModelApiError, match="timed out after 180s"):
        _chat(httpx.MockTransport(handler), retries=Retries(connect_attempts=3, sleep=lambda _s: None))
    assert len(seen) == 1


def test_a_policy_that_asks_for_it_sends_a_timed_out_reply_once_more() -> None:
    """Extraction and the morning judge opt in: one lost reply there costs a whole run (2026-10-30)."""
    seen: list[httpx.Request] = []
    slept: list[float] = []
    counted: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadTimeout("slow reply", request=request)
        return httpx.Response(200, json=_ok(fireworks_api.SPARK_OPEN))

    retries = Retries(reply_attempts=2, sleep=slept.append, counted=lambda: counted.append(1))
    result = _chat(httpx.MockTransport(handler), retries=retries)

    assert result.text == "ok" and len(seen) == 2
    assert slept == [BACKOFF_S[0]] and counted == [1]  # one resend, counted, after the first backoff


def test_two_lost_replies_raise_todays_transport_error_after_the_last_one() -> None:
    seen: list[httpx.Request] = []
    slept: list[float] = []
    counted: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ReadError("connection reset", request=request)

    retries = Retries(reply_attempts=2, sleep=slept.append, counted=lambda: counted.append(1))
    with pytest.raises(ModelApiError, match=r"failed in transport \(ReadError\)"):
        _chat(httpx.MockTransport(handler), retries=retries)

    assert len(seen) == 2 and slept == [BACKOFF_S[0]] and counted == [1]


def test_sleeps_delegated_policy_sends_a_dropped_connection_again() -> None:
    assert fireworks_api.SLEEP_RETRIES.connect_attempts == fireworks_api.CONNECT_RETRIES + 1 > 1


def test_sleeps_delegated_policy_sends_a_lost_reply_once_more() -> None:
    """Practice, the brief and MIND go through this policy: one lost reply no longer fails a night."""
    assert fireworks_api.SLEEP_RETRIES.reply_attempts == fireworks_api.REPLY_ATTEMPTS == 2


def test_sleeps_busy_reply_waits_a_longer_named_schedule() -> None:
    """Fireworks' 503 on 2026-10-31 outlasted the shared 2 s then 4 s and killed a life on day 1."""
    assert fireworks_api.SLEEP_RETRIES.backoff_s == fireworks_api.SPARK_BUSY_BACKOFF_S
    assert len(fireworks_api.SPARK_BUSY_BACKOFF_S) > len(BACKOFF_S)
    assert sum(fireworks_api.SPARK_BUSY_BACKOFF_S) > 200 > sum(BACKOFF_S)


def test_a_503_then_200_succeeds_after_sleeps_longer_waits() -> None:
    replies = iter([httpx.Response(503), httpx.Response(503),
                    httpx.Response(200, json=_ok(fireworks_api.SPARK_OPEN))])
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return next(replies)

    retries = replace(fireworks_api.SLEEP_RETRIES, sleep=slept.append)
    result = _chat(httpx.MockTransport(handler), retries=retries)
    assert result.text == "ok" and len(seen) == 3
    assert slept == list(fireworks_api.SPARK_BUSY_BACKOFF_S[:2])  # the longer waits, not BACKOFF_S


def test_sleeps_longer_schedule_still_honours_a_capped_retry_after() -> None:
    busy = httpx.Response(503, headers={"Retry-After": "999"})
    assert fireworks_api.SLEEP_RETRIES.pause(0, busy) == RETRY_AFTER_MAX_S


def test_a_callers_policy_replaces_the_selected_providers_and_reaches_fireworks(monkeypatch: pytest.MonkeyPatch) -> None:
    """``baseten_api.chat``'s own ``retries`` wins over the provider's default on the Fireworks path."""
    from anjo.sleep import baseten_api

    served: list[dict[str, Any]] = []

    def fake_chat(model: str, *a: Any, **kw: Any) -> ChatResult:
        served.append({"model": model, **kw})
        return ChatResult("ok", (), model, 5, 1, "stop")

    monkeypatch.setenv("ANJO_SLEEP_SPARK", "fireworks")
    monkeypatch.setattr(fireworks_api, "chat", fake_chat)

    baseten_api.chat(baseten_api.SPARK_OPEN, MESSAGES, None, 300.0)
    mine = Retries(connect_attempts=3, reply_attempts=2)
    baseten_api.chat(baseten_api.SPARK_OPEN, MESSAGES, None, 300.0, retries=mine)

    assert served[0]["retries"] is fireworks_api.SLEEP_RETRIES  # unchanged without one
    assert served[1]["retries"] is mine


def test_a_callers_policy_reaches_the_baseten_path_too(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep import baseten_api

    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadTimeout("slow reply", request=request)
        return httpx.Response(200, json=_ok(baseten_api.SPARK_OPEN))

    monkeypatch.setenv("ANJO_SLEEP_SPARK", "baseten")
    monkeypatch.setenv(baseten_api.KEY_ENV, "baseten-fake")
    retries = Retries(reply_attempts=2, sleep=slept.append)
    result = baseten_api.chat(baseten_api.SPARK_OPEN, MESSAGES, None, 300.0, transport=httpx.MockTransport(handler),
                              retries=retries)
    assert result.text == "ok" and len(seen) == 2 and slept == [BACKOFF_S[0]]


def _recording_post(served: list[dict[str, Any]]):
    def post(**kw: Any) -> ChatResult:
        served.append(kw)
        return ChatResult("ok", (), kw["model"], 5, 1, "stop")

    return post


def _routed(monkeypatch: pytest.MonkeyPatch, timeout_s: float, max_tokens: int | None) -> float:
    """The wait Sleep's Spark Open call reaches Fireworks with, through ``baseten_api.chat``."""
    from anjo.sleep import baseten_api

    served: list[dict[str, Any]] = []
    monkeypatch.setenv("ANJO_SLEEP_SPARK", "fireworks")
    monkeypatch.setattr(fireworks_api, "post_chat", _recording_post(served))
    baseten_api.chat(baseten_api.SPARK_OPEN, [{"role": "user", "content": "hi"}], None, timeout_s,
                     max_tokens=max_tokens)
    return served[-1]["timeout_s"]


def test_a_capped_reply_waits_for_its_cap_not_a_flat_180_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _routed(monkeypatch, 180.0, 4_096) == reply_timeout_s(4_096) > 180.0


def test_an_uncapped_call_keeps_the_callers_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _routed(monkeypatch, 180.0, None) == 180.0


def test_a_callers_longer_wait_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _routed(monkeypatch, 900.0, 1_024) == 900.0


def test_a_direct_fireworks_call_keeps_its_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[dict[str, Any]] = []
    monkeypatch.setattr(fireworks_api, "post_chat", _recording_post(served))
    _chat(httpx.MockTransport(lambda r: httpx.Response(200)), timeout_s=30.0, max_tokens=4_096)
    assert served[-1]["timeout_s"] == 30.0  # MIND's experiment and the judge set their own waits
