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
            return httpx.Response(200, json={"tokens": [1, 2, 3] if "messages" in body else [9, 8]})
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
    assert result.text == "hello" and result.input_tokens == 7
    _client(fake).complete(None, [{"role": "user", "content": "hi"}], temperature=1.0)
    assert fake.requests[-1][1]["model"] == config.MODEL


def test_continuation_logprobs_reads_the_tail_of_prompt_logprobs() -> None:
    fake = FakeVllm()
    scores = _client(fake).continuation_logprobs(None, [{"role": "user", "content": "q"}], " A")
    assert list(scores) == [-1.5, -2.0]  # tokens 9 and 8, at positions 3 and 4
    completion = fake.requests[-1][1]
    assert completion["prompt"] == [1, 2, 3, 9, 8] and completion["prompt_logprobs"] == 0


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
