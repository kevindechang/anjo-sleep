# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's vLLM client (Task 5): one server, the adapter as the model, registered sampling,
and continuation log-probabilities from prompt_logprobs. A fake vLLM answers; no network."""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pytest

from anjo.sleep.serving import config
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.types import CheckpointRef

ADAPTER = CheckpointRef("p-aaaaaaaaaaaa-n1", "p1", date(2026, 10, 28), "modal-volume://x", None)


class FakeVllm:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any], str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body, request.headers["authorization"]))
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "model": body["model"], "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})
        if request.url.path == "/tokenize":
            tokens = [1, 2, 3] if "messages" in body else [5, 6] if body["prompt"] == config.USER_CHANNEL else [9, 8]
            return httpx.Response(200, json={"tokens": tokens})
        if request.url.path == "/v1/completions":
            prompt = body["prompt"]
            logprobs = [None] + [{str(t): {"logprob": -0.5 * i, "rank": 1}} for i, t in enumerate(prompt[1:], 1)]
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": logprobs, "text": "x"}]})
        return httpx.Response(404)


def _client(fake: FakeVllm) -> VllmClient:
    return VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake))


def test_complete_names_the_adapter_and_samples_as_registered() -> None:
    fake = FakeVllm()
    result = _client(fake).complete(ADAPTER, [{"role": "user", "content": "hi"}], temperature=1.0, seed=7)
    path, body, auth = fake.requests[0]
    assert path == "/v1/chat/completions" and auth == "Bearer tok"
    assert (body["model"], body["temperature"], body["top_p"], body["top_k"], body["seed"]) == (
        ADAPTER.id, 1.0, 0.95, 64, 7)
    assert body["chat_template_kwargs"] == {"reasoning_strength": "low"}  # as the adapter was trained
    assert result.text == "hello" and result.input_tokens == 7
    _client(fake).complete(None, [{"role": "user", "content": "hi"}], temperature=1.0)
    assert fake.requests[-1][1]["model"] == config.MODEL


def test_continuation_logprobs_scores_only_the_continuation_after_the_replys_header() -> None:
    fake = FakeVllm()
    scores = _client(fake).continuation_logprobs(None, [{"role": "user", "content": "q"}], " A")
    assert list(scores) == [-2.5, -3.0]  # tokens 9 and 8, at positions 5 and 6
    (_, chat, _), (_, header, _), (_, tail, _), (_, completion, _) = fake.requests
    assert chat["add_generation_prompt"] is True and chat["chat_template_kwargs"] == {"reasoning_strength": "low"}
    assert header == {"model": config.MODEL, "prompt": " to=user<|message|>", "add_special_tokens": False}
    assert tail["add_special_tokens"] is False
    assert completion["prompt"] == [1, 2, 3, 5, 6, 9, 8] and completion["prompt_logprobs"] == 0


def test_the_reasoning_strength_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.REASONING_STRENGTH_ENV, "xhigh")
    fake = FakeVllm()
    _client(fake).complete(None, [{"role": "user", "content": "hi"}], temperature=1.0)
    assert fake.requests[-1][1]["chat_template_kwargs"] == {"reasoning_strength": "xhigh"}
    monkeypatch.setenv(config.REASONING_STRENGTH_ENV, "max")
    with pytest.raises(config.ServeConfigError):
        _client(fake)


def test_errors_name_the_endpoint_never_the_words() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "cold"})

    client = VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(refuse))
    with pytest.raises(ServingError) as raised:
        client.complete(None, [{"role": "user", "content": "private words"}], temperature=1.0)
    assert "503" in str(raised.value) and "private" not in str(raised.value)
    with pytest.raises(ServingError):
        VllmClient("http://serve.example", "tok")


def test_the_client_serves_at_the_registered_precision() -> None:
    assert VllmClient.precision == config.SERVING.name
