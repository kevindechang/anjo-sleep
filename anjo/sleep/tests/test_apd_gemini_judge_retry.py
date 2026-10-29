"""The Gemini judge asks a 429, a 5xx or a timeout again, and names a refusal's status."""

from __future__ import annotations

import time
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.mind.apd import backend, judging, retry
from anjo.sleep.model_api import ModelApiError

RULING = '{"same_problem": true, "holds": false}'


class _Failed(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"code {code}")
        self.code = code


class _Models:
    def __init__(self, failures: list[int]) -> None:
        self.failures, self.calls = failures, 0

    def generate_content(self, **_: Any) -> Any:
        self.calls += 1
        if self.failures:
            raise _Failed(self.failures.pop(0))
        usage = types.SimpleNamespace(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=2)
        return types.SimpleNamespace(text="A", usage_metadata=usage)


def _client(monkeypatch: pytest.MonkeyPatch, failures: list[int]) -> _Models:
    models = _Models(failures)
    monkeypatch.setattr(backend, "_vertex", lambda: types.SimpleNamespace(models=models))
    monkeypatch.setattr(backend.time, "sleep", lambda _s: None)
    return models


def test_a_rate_limit_and_a_server_error_are_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _client(monkeypatch, [429, 503])
    assert backend._gemini("judge this") == ("A", 10, 5)
    assert models.calls == 3


def test_a_client_error_names_its_status_and_is_not_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _client(monkeypatch, [400])
    with pytest.raises(ModelApiError, match=r"^Gemini on Vertex returned HTTP 400$"):
        backend._gemini("judge this")
    assert models.calls == 1


def test_rate_limits_past_the_attempts_fail_as_a_429(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _client(monkeypatch, [429] * retry.RATE_LIMIT_ATTEMPTS)
    with pytest.raises(ModelApiError, match="returned HTTP 429"):
        backend._gemini("judge this")
    assert models.calls == retry.RATE_LIMIT_ATTEMPTS


def test_server_errors_are_asked_again_at_most_the_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _client(monkeypatch, [503] * retry.RATE_LIMIT_ATTEMPTS)
    with pytest.raises(ModelApiError, match="returned HTTP 503"):
        backend._gemini("judge this")
    assert models.calls == retry.ATTEMPTS


class _OneTimeout:
    """The first ``failures`` tries time out; every try after them answers with a ruling."""

    def __init__(self, failures: int) -> None:
        self.failures, self.calls = failures, 0

    def generate_content(self, **_: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise httpx.ReadTimeout("too slow")
        usage = types.SimpleNamespace(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=2)
        return types.SimpleNamespace(text=RULING, usage_metadata=usage)


def _timing_client(monkeypatch: pytest.MonkeyPatch, failures: int) -> _OneTimeout:
    models = _OneTimeout(failures)
    monkeypatch.setattr(backend, "_vertex", lambda: types.SimpleNamespace(models=models))
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    return models


def test_a_billed_timeout_then_a_reply_is_settled_within_the_hold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    models = _timing_client(monkeypatch, 1)
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    rate = judging.Rate(0.75, 3.75)
    judge = judging.CachedJudge(judging.GEMINI, backend._gemini, tmp_path / "gemini.jsonl", ledger, rate)

    assert judge.complete("judge this") == RULING
    assert models.calls == 2  # a timeout, then the reply
    used = ledger.used(judging.POOLS[judging.GEMINI])
    assert used == pytest.approx(rate.worst + rate.usd(10, 5))  # the failed try at its worst, then the reply
    assert used <= rate.worst * retry.ATTEMPTS
    assert ledger.overages() == () and ledger.open_holds() == ()


def test_timeouts_past_the_attempts_settle_the_hold_not_beyond(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    models = _timing_client(monkeypatch, retry.ATTEMPTS)
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    rate = judging.Rate(0.75, 3.75)
    judge = judging.CachedJudge(judging.GEMINI, backend._gemini, tmp_path / "gemini.jsonl", ledger, rate)

    with pytest.raises(ModelApiError, match="timed out"):
        judge.complete("judge this")
    assert models.calls == retry.ATTEMPTS
    assert ledger.used(judging.POOLS[judging.GEMINI]) == pytest.approx(rate.worst * retry.ATTEMPTS)
    assert ledger.overages() == () and ledger.open_holds() == ()


def test_a_client_error_is_not_asked_again_after_a_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _client(monkeypatch, [429, 400])
    with pytest.raises(ModelApiError, match=r"returned HTTP 400$"):
        backend._gemini("judge this")
    assert models.calls == 2
