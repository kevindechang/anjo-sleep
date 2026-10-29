"""The Gemini judge asks a 429, a 5xx or a timeout again, and names a refusal's status."""

from __future__ import annotations

import types
from typing import Any

import pytest

from anjo.sleep.mind.apd import backend
from anjo.sleep.model_api import ModelApiError


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
    models = _client(monkeypatch, [429] * backend.retry.RATE_LIMIT_ATTEMPTS)
    with pytest.raises(ModelApiError, match="returned HTTP 429"):
        backend._gemini("judge this")
    assert models.calls == backend.retry.RATE_LIMIT_ATTEMPTS
