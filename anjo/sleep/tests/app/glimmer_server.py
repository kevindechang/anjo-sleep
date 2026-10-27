"""A scripted stand-in for Glimmer's vLLM server under the real ``SleepRuntime``: the transport its
``/v1/completions`` streams come over, and the ``Poster`` its ``/tokenize`` calls go to. No network."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from anjo.core.anjo_model.transport import TransportError
from anjo.sleep.serving import config

CHAT_IDS = [200000, 11, 12, 13]  # what /tokenize gives any chat
HEADER_IDS = [21, 22, 200023]  # and the reply's header


class Upstream:
    def __init__(self, chunks: Sequence[object]) -> None:
        self._chunks = list(chunks)

    def __iter__(self) -> Iterator[object]:
        for chunk in self._chunks:
            if isinstance(chunk, TransportError):
                raise chunk  # the server breaks off mid-reply
            yield chunk

    def close(self) -> None:
        return None


class Modal:
    """The server: each completion gets the next scripted reply, a chunk list or an error. Every
    ``/tokenize`` call is answered (a chat with ``CHAT_IDS``, the header with ``HEADER_IDS``) and a
    chat's request kept in ``chats``; ``opened`` holds each completion's url, headers and body."""

    def __init__(self, *replies: Sequence[object] | TransportError) -> None:
        self.replies = list(replies)
        self.opened: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self.chats: list[dict[str, Any]] = []

    def open(self, url: str, headers: Any, body: Any) -> Upstream:
        self.opened.append((url, dict(headers), json.loads(json.dumps(body))))
        reply = self.replies.pop(0)
        if isinstance(reply, TransportError):
            raise reply
        return Upstream(reply)

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        assert path == "/tokenize"
        if "messages" in body:
            self.chats.append(json.loads(json.dumps(body)))
            return {"tokens": CHAT_IDS}
        assert body == {"model": body["model"], "prompt": config.USER_CHANNEL, "add_special_tokens": False}
        return {"tokens": HEADER_IDS}

    def close(self) -> None:
        return None


def started(model: str = config.MODEL) -> dict[str, Any]:
    """A completion's first chunk, before any word."""
    return {"id": "c", "model": model, "choices": [{"index": 0, "text": ""}]}


def words(text: str, model: str = config.MODEL) -> dict[str, Any]:
    return {"id": "c", "model": model, "choices": [{"index": 0, "text": text}]}


def chunks(text: str, model: str = config.MODEL) -> list[dict[str, Any]]:
    """A whole streamed completion: the words, the stop, then the usage."""
    return [
        words(text, model),
        {"id": "c", "model": model, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
        {"id": "c", "model": model, "choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 6}},
    ]
