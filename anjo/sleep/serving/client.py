"""The night's client for Glimmer on the Modal vLLM server (Task 5): practice, every gate and every
arm go through it, so all of them sample one server at one precision (``config.SERVING``).

It is the ``ModelClient`` and ``TokenLogprobs`` port over vLLM's OpenAI-compatible API:

- ``complete`` asks ``/v1/chat/completions`` for one sampled reply, naming the person's adapter
  as the model (the base model when there is none) with the registered sampling (temperature as
  given, top_p 0.95, top_k 64) and a seed, so evals repeat;
- ``continuation_logprobs`` scores a continuation token by token: ``/tokenize`` turns the chat
  (with the generation prompt) and the continuation into token ids, and ``/v1/completions`` returns
  the prompt's log-probabilities with ``prompt_logprobs``, one generated token at most.

The URL and token come from the app (``ANJO_SLEEP_MODAL_URL``, ``ANJO_SLEEP_MODAL_TOKEN``), never
from here. Errors name the endpoint and the status, never a message's words.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx

from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving import config
from anjo.sleep.types import CheckpointRef

TOP_P: Final = 0.95
TOP_K: Final = 64
MAX_TOKENS: Final = 1024
REQUEST_TIMEOUT_S: Final = 300.0  # a cold start waits inside the server's own limit


class ServingError(RuntimeError):
    """The server refused or failed a request, or sent a reply that can't be read."""


def model_name(adapter: CheckpointRef | None) -> str:
    """What vLLM serves: the adapter's name, or the base model."""
    return config.MODEL if adapter is None else config.check_adapter_name(adapter.id)


class VllmClient:
    """``ModelClient`` and ``TokenLogprobs`` over one vLLM server."""

    precision: Final = config.SERVING.name

    def __init__(
        self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S, max_tokens: int = MAX_TOKENS,
    ) -> None:
        if not base_url.startswith("https://"):
            raise ServingError("the serving URL is https")
        if not token:
            raise ServingError("the serving token is set")
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {token}"},
            transport=transport, timeout=timeout_s,
        )
        self._max_tokens = max_tokens

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(path, json=dict(body))
        except httpx.HTTPError as error:
            raise ServingError(f"{path} failed in transport ({type(error).__name__})") from None
        if response.is_error:
            raise ServingError(f"{path} returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise ServingError(f"{path} sent a reply that is not JSON") from None
        if not isinstance(data, dict):
            raise ServingError(f"{path} sent a reply that is not an object")
        return data

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": model_name(adapter), "messages": messages, "temperature": temperature,
            "top_p": TOP_P, "top_k": TOP_K, "max_tokens": self._max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        data = self._post("/v1/chat/completions", body)
        try:
            choice, usage = data["choices"][0], data["usage"]
            content = choice["message"].get("content")
            return ChatResult(
                text="" if content is None else str(content), tool_calls=(), model=str(data["model"]),
                input_tokens=int(usage["prompt_tokens"]), output_tokens=int(usage["completion_tokens"]),
                finish_reason=choice.get("finish_reason"),
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ServingError("/v1/chat/completions sent a malformed reply") from None

    def _tokens(self, body: Mapping[str, Any]) -> list[int]:
        data = self._post("/tokenize", body)
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in tokens):
            raise ServingError("/tokenize sent no token ids")
        return tokens

    def continuation_logprobs(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], continuation: str,
    ) -> Sequence[float]:
        model = model_name(adapter)
        prompt = self._tokens({"model": model, "messages": messages, "add_generation_prompt": True})
        tail = self._tokens({"model": model, "prompt": continuation, "add_special_tokens": False})
        if not tail:
            raise ServingError("the continuation has no tokens")
        data = self._post("/v1/completions", {
            "model": model, "prompt": prompt + tail, "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 0,
        })
        try:
            scored = data["choices"][0]["prompt_logprobs"][-len(tail):]
            return tuple(float(entry[str(token)]["logprob"]) for entry, token in zip(scored, tail, strict=True))
        except (KeyError, IndexError, TypeError, ValueError):
            raise ServingError("/v1/completions sent no prompt log-probabilities") from None
