# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""APD's transient failures retried at its own transport level (Task 17 review): a 429, a 5xx, a
timeout or a dropped connection, at most three tries with jittered backoff, each Spark Open try
held and journaled as a call of its own. Anything else fails at once. Mock transports and fakes
only: no model, no network."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.mind.apd import live, retry
from anjo.sleep.mind.apd.journal import SpendJournal
from anjo.sleep.model_api import ChatResult, ModelApiError, post_chat
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_usd, spark_worst_usd

PRICES = Prices(spark_in_per_m=0.6, spark_out_per_m=2.4, train_gpu_per_hour=1.5, serve_gpu_per_hour=2.61)
MESSAGES = [{"role": "user", "content": "y"}]
REPLY = {"model": "spark-open", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def _post(handler: Any, monkeypatch: pytest.MonkeyPatch) -> ModelApiError | None:
    monkeypatch.setenv("APD_RETRY_TEST_KEY", "k")
    try:
        post_chat(provider="Baseten", url="https://example.invalid/v1/chat/completions", key_env="APD_RETRY_TEST_KEY",
                  model="spark-open", messages=MESSAGES, tools=None, timeout_s=5.0, temperature=0.0,
                  transport=httpx.MockTransport(handler))
    except ModelApiError as error:
        return error
    return None


def _raise(error: type[httpx.HTTPError]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("down", request=request)
    return handler


@pytest.mark.parametrize(("handler", "expected"), [
    (lambda _r: httpx.Response(429), True),
    (lambda _r: httpx.Response(500), True),
    (lambda _r: httpx.Response(503), True),
    (_raise(httpx.ReadTimeout), True),
    (_raise(httpx.ConnectTimeout), True),
    (_raise(httpx.ConnectError), True),
    (_raise(httpx.ReadError), True),
    (_raise(httpx.RemoteProtocolError), True),
    (lambda _r: httpx.Response(400), False),
    (lambda _r: httpx.Response(401), False),
    (lambda _r: httpx.Response(404), False),
    (lambda _r: httpx.Response(200, content=b"not json"), False),
    (_raise(httpx.UnsupportedProtocol), False),
])
def test_transient_reads_model_apis_own_messages(handler: Any, expected: bool,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    error = _post(handler, monkeypatch)
    assert error is not None and retry.transient(error) is expected
    assert not retry.transient(ModelApiError("past the Spark Open calls this night holds"))


def test_backoff_doubles_and_jitters() -> None:
    assert retry.backoff(1, lambda: 0.5) == pytest.approx(retry.BACKOFF_S)
    assert retry.backoff(2, lambda: 0.5) == pytest.approx(2 * retry.BACKOFF_S)
    assert retry.backoff(1, lambda: 0.0) == pytest.approx(retry.BACKOFF_S / 2)
    assert retry.backoff(1, lambda: 0.999) < 1.5 * retry.BACKOFF_S


def test_the_transport_retries_429_5xx_timeouts_and_resets_at_most_three_tries() -> None:
    def flaky(answers: list[Any]) -> tuple[retry.RetryTransport, list[float], list[int]]:
        tries: list[int] = []
        slept: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            tries.append(1)
            answer = answers[min(len(tries), len(answers)) - 1]
            if isinstance(answer, type):
                raise answer("down", request=request)
            return httpx.Response(answer, json={"ok": answer == 200})

        return retry.RetryTransport(httpx.MockTransport(handler), sleep=slept.append, rand=lambda: 0.5), slept, tries

    for first in (429, 502, 503, httpx.ReadTimeout, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
        transport, slept, tries = flaky([first, 200])
        with httpx.Client(transport=transport, base_url="http://127.0.0.1:9") as client:
            assert client.post("/v1/completions", json={"prompt": [1]}).status_code == 200
        assert len(tries) == 2 and slept == [retry.BACKOFF_S] and transport.retries == 1

    transport, slept, tries = flaky([503])
    with httpx.Client(transport=transport, base_url="http://127.0.0.1:9") as client:
        assert client.post("/v1/completions", json={}).status_code == 503  # the last try's answer
    assert len(tries) == retry.ATTEMPTS and slept == [retry.BACKOFF_S, 2 * retry.BACKOFF_S]

    transport, slept, tries = flaky([httpx.ConnectError])
    with httpx.Client(transport=transport, base_url="http://127.0.0.1:9") as client, \
            pytest.raises(httpx.ConnectError):
        client.post("/v1/completions", json={})
    assert len(tries) == retry.ATTEMPTS

    for hard in (400, 401, 404, httpx.UnsupportedProtocol):
        transport, slept, tries = flaky([hard, 200])
        with httpx.Client(transport=transport, base_url="http://127.0.0.1:9") as client:
            try:
                assert client.post("/v1/completions", json={}).status_code == hard
            except httpx.UnsupportedProtocol:
                pass
        assert len(tries) == 1 and slept == []  # never retried


class _Flaky:
    """Spark Open answering ``errors`` in turn, then a reply."""

    def __init__(self, *errors: str) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.calls += 1
        if self.errors:
            raise ModelApiError(self.errors.pop(0))
        return ChatResult(text="{}", tool_calls=(), model="spark-open", input_tokens=10, output_tokens=2,
                          finish_reason="stop")


def test_a_transient_spark_failure_is_retried_each_try_held_and_journaled(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    spend = SpendJournal(tmp_path / "spend.jsonl")
    waits: list[float] = []
    chat = _Flaky("Baseten chat with spark-open returned HTTP 429")
    calls = live.SparkCalls(chat, PRICES, ledger, "night", journal=spend, wait=waits.append, rand=lambda: 0.5)
    assert calls(MESSAGES, 10).text == "{}"
    calls.close()
    (night,) = spend.jobs()
    assert [call.failed for call in night.calls] == [True, False] and chat.calls == 2 and waits == [retry.BACKOFF_S]
    assert night.usd == pytest.approx(spark_worst_usd(PRICES, 1) + spark_usd(PRICES, 10, 2))  # the failed try's worst
    assert calls.total_calls == 2 and ledger.open_holds() == ()


@pytest.mark.parametrize(("errors", "tries"), [
    (["Baseten chat with spark-open returned HTTP 400"], 1),
    (["Baseten chat with spark-open sent a malformed reply"], 1),
    (["Baseten chat with spark-open returned HTTP 503"] * 5, retry.ATTEMPTS),
    (["Baseten chat with spark-open timed out after 120s"] * 5, retry.ATTEMPTS),
])
def test_a_hard_or_lasting_spark_failure_fails_the_run(tmp_path: Path, errors: list[str], tries: int) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    spend = SpendJournal(tmp_path / "spend.jsonl")
    chat = _Flaky(*errors)
    calls = live.SparkCalls(chat, PRICES, ledger, "night", journal=spend, wait=lambda _s: None)
    with pytest.raises(live.RunFailed, match="Spark Open call failed"):
        calls(MESSAGES, 10)
    calls.close()
    assert chat.calls == tries and [call.failed for call in spend.jobs()[0].calls] == [True] * tries
    assert ledger.open_holds() == ()


def test_a_retry_never_goes_past_the_calls_a_night_holds(tmp_path: Path) -> None:
    chat = _Flaky("Baseten chat with spark-open returned HTTP 429")
    calls = live.SparkCalls(chat, PRICES, None, "night", per_night=1, wait=lambda _s: None)
    with pytest.raises(live.RunFailed, match="holds no call"):
        calls(MESSAGES, 10)
    assert chat.calls == 1
