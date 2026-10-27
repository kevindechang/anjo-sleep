"""Anjo's streaming transport over Glimmer's SageMaker endpoint, for ``SleepRuntime``.

The runtime builds the chat body as for any host and names the vLLM path in a ``sagemaker://`` URL;
this transport sends them as the shim's envelope through ``invoke_endpoint_with_response_stream``,
signed by the dev host's role, and reads the reply back as server-sent events, exactly as
``HttpTransport`` does. A refusal before the stream (no capacity while Glimmer sleeps, read as a
503; a container error) raises ``TransportError`` with its status; a stream that drops raises it
with none. So both
end the turn the way a Modal failure does, and the frontier talker answers (RF3).

The boto3 client is made once per process, with the reply deadline as its read timeout and one
attempt: a retry would only stretch the turn past the fallback's patience.
"""
from __future__ import annotations

import codecs
import threading
from collections.abc import Generator, Iterator, Mapping
from typing import Any, Final
from urllib.parse import urlsplit

from anjo.core.anjo_model.transport import TransportError, Upstream, sse_lines, sse_payloads
from anjo.sleep.serving.sagemaker import settings
from anjo.sleep.serving.sagemaker.invoke import InvokeError, Invoker, Stream, make_client

SCHEME: Final = "sagemaker"

_client: list[Any] = []
_client_lock = threading.Lock()


def endpoint_url(path: str) -> str:
    """The pseudo URL the runtime hands this transport: the endpoint, then the vLLM path."""
    return f"{SCHEME}://{settings.ENDPOINT}{path}"


class _EnvelopeUpstream:
    def __init__(self, stream: Stream) -> None:
        self._stream = stream
        self._payloads = self._read()

    def _text(self) -> Iterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for chunk in self._stream:
            yield decoder.decode(chunk)
        yield decoder.decode(b"", final=True)

    def _read(self) -> Generator[object, None, None]:
        try:
            yield from sse_payloads(sse_lines(self._text()))
        except InvokeError as error:
            raise TransportError(None) from error

    def __iter__(self) -> Iterator[object]:
        return self._payloads

    def close(self) -> None:
        self._payloads.close()
        self._stream.close()


class SageMakerTransport:
    """``Transport`` through one ``Invoker``."""

    def __init__(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def open(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> Upstream:
        parts = urlsplit(url)
        if parts.scheme != SCHEME or parts.netloc != settings.ENDPOINT:
            raise TransportError(None)
        try:
            stream = self._invoker.stream(parts.path, body)
        except InvokeError as error:
            raise TransportError(error.status) from None
        return _EnvelopeUpstream(stream)

    def close(self) -> None:
        self._invoker.close()


def shared_client(read_s: float, connect_s: float) -> Any:
    """The process's ``sagemaker-runtime`` client for turns, made on first use."""
    with _client_lock:
        if not _client:
            _client.append(make_client("sagemaker-runtime", read_s=read_s, attempts=1, connect_s=connect_s))
        return _client[0]


def default_transport(read_s: float, connect_s: float) -> SageMakerTransport:
    return SageMakerTransport(Invoker(shared_client(read_s, connect_s), settings.ENDPOINT, settings.COMPONENT))
