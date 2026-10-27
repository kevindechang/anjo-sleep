"""The signed calls to Glimmer's SageMaker endpoint, and the boto3 clients under them.

Every call carries the envelope the container's shim reads, ``{"path", "body", "stream"}``: the vLLM
path, its JSON body, and whether the reply streams. SigV4 signs each call with the dev host's role
(``token_refresh``), so no bearer token travels. ``InvokeError`` carries the status a failure maps
to (the container's own status for a ``ModelError``), or None when the connection or a read
failed, and never a word of the request or the reply.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-untyped]
from urllib3.exceptions import HTTPError as Urllib3Error

from anjo.sleep.serving.client import ServingError, json_object
from anjo.sleep.serving.sagemaker import settings

CONNECT_TIMEOUT_S = 10.0
CONTROL_READ_TIMEOUT_S = 60.0
CONTROL_ATTEMPTS = 3  # describe and set the copy count: a throttle is worth a retry
REFUSED = frozenset({401, 403})  # the role can't make this call: an error, never "not ready yet"


class InvokeError(Exception):
    """The endpoint gave no usable reply: ``status`` is the HTTP status, or None."""

    def __init__(self, status: int | None, code: str = "") -> None:
        super().__init__(f"the endpoint gave no usable reply ({'no reply' if status is None else status} {code})")
        self.status, self.code = status, code


def make_client(service: str, *, read_s: float, attempts: int, connect_s: float = CONNECT_TIMEOUT_S) -> Any:
    """A boto3 client in our region with both timeouts set and a bounded number of attempts."""
    return boto3.client(service, region_name=settings.REGION, config=Config(
        connect_timeout=connect_s, read_timeout=read_s,
        retries={"total_max_attempts": attempts, "mode": "standard"},
    ))


def envelope(path: str, body: Mapping[str, Any], *, stream: bool) -> bytes:
    return json.dumps({"path": path, "body": dict(body), "stream": stream}, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def invoke_error(error: ClientError) -> InvokeError:
    """What a refused call maps to: a ModelError carries the container's own status."""
    details: Mapping[str, Any] = error.response.get("Error", {})
    code = str(details.get("Code", ""))
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    original = details.get("OriginalStatusCode")
    if code == "ModelError" and original is not None:
        status = original
    return InvokeError(int(status) if status is not None else None, code)


class Invoker:
    """One endpoint's inference component, through a ``sagemaker-runtime`` client."""

    def __init__(self, client: Any, endpoint: str = settings.ENDPOINT, component: str = settings.COMPONENT) -> None:
        self._client, self._endpoint, self._component = client, endpoint, component

    def _target(self) -> dict[str, str]:
        return {"EndpointName": self._endpoint, "InferenceComponentName": self._component}

    def call(self, path: str, body: Mapping[str, Any]) -> bytes:
        """One reply, read whole."""
        try:
            reply = self._client.invoke_endpoint(**self._target(), Body=envelope(path, body, stream=False),
                                                 ContentType="application/json", Accept="application/json")
            data: bytes = reply["Body"].read()
        except ClientError as error:
            raise invoke_error(error) from None
        except (BotoCoreError, Urllib3Error, OSError):
            raise InvokeError(None) from None
        return data

    def stream(self, path: str, body: Mapping[str, Any]) -> Stream:
        """A streamed reply's bytes. A refusal raises here; a drop raises while reading."""
        try:
            reply = self._client.invoke_endpoint_with_response_stream(
                **self._target(), Body=envelope(path, body, stream=True),
                ContentType="application/json", Accept="text/event-stream")
        except ClientError as error:
            raise invoke_error(error) from None
        except (BotoCoreError, Urllib3Error, OSError):
            raise InvokeError(None) from None
        return Stream(reply["Body"])

    def close(self) -> None:
        """Nothing to free: the client is shared."""


class Stream:
    """The payload bytes of one event stream. Any failure mid-stream (the container's own
    ModelStreamError, a cut connection, a read past the timeout) is ``InvokeError(None)``."""

    def __init__(self, events: Any) -> None:
        self._events = events

    def __iter__(self) -> Iterator[bytes]:
        try:
            for event in self._events:
                part = event.get("PayloadPart") if isinstance(event, Mapping) else None
                if part is not None and part.get("Bytes"):
                    yield bytes(part["Bytes"])
        except (ClientError, BotoCoreError, Urllib3Error, OSError):
            raise InvokeError(None) from None

    def close(self) -> None:
        self._events.close()


class SageMakerPoster:
    """``client.Poster`` through the endpoint, for the night's ``VllmClient``."""

    def __init__(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            raw = self._invoker.call(path, body)
        except InvokeError as error:
            if error.status is None:
                raise ServingError(f"{path} failed in transport") from None
            raise ServingError(f"{path} returned HTTP {error.status}") from None
        return json_object(path, raw)

    def close(self) -> None:
        self._invoker.close()
