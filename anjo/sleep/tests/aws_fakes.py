"""Stand-ins for boto3 clients, offline: every call's parameters are checked against botocore's own
service model (as ``botocore.stub.Stubber`` checks them), then a scripted handler answers. Nothing
is signed or sent, and no credentials are read."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import botocore.session  # type: ignore[import-untyped]
from botocore import xform_name
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from botocore.validate import validate_parameters  # type: ignore[import-untyped]

Handler = Callable[..., Any]


class FakeAws:
    """A boto3 client for ``service``: ``calls`` records (method, params) in order, and each
    method is answered by ``handlers[method]``. A method without a handler fails the test."""

    def __init__(self, service: str, **handlers: Handler) -> None:
        model = botocore.session.get_session().get_service_model(service)
        self._operations = {xform_name(name): model.operation_model(name) for name in model.operation_names}
        self._handlers = handlers
        self.calls: list[tuple[str, dict[str, Any]]] = []
        unknown = set(handlers) - set(self._operations)
        if unknown:
            raise AssertionError(f"{service} has no operation {sorted(unknown)}")

    def __getattr__(self, method: str) -> Handler:
        if method.startswith("_") or method not in self._operations:
            raise AttributeError(method)
        operation = self._operations[method]

        def call(**params: Any) -> Any:
            validate_parameters(params, operation.input_shape)  # a wrong name or type raises here
            self.calls.append((method, params))
            handler = self._handlers.get(method)
            if handler is None:
                raise AssertionError(f"unexpected call {method}")
            return handler(**params)

        return call

    def named(self, method: str) -> list[dict[str, Any]]:
        return [params for name, params in self.calls if name == method]


def client_error(code: str, status: int, operation: str = "InvokeEndpoint", **extra: Any) -> ClientError:
    """The ClientError botocore raises for an error reply."""
    return ClientError({"Error": {"Code": code, "Message": code, **extra},
                        "ResponseMetadata": {"HTTPStatusCode": status}}, operation)


class Body:
    """A StreamingBody stand-in."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._at = 0
        self.closed = False

    def read(self, amt: int | None = None) -> bytes:
        end = len(self._data) if amt is None else self._at + amt
        chunk, self._at = self._data[self._at:end], min(end, len(self._data))
        return chunk

    def close(self) -> None:
        self.closed = True


class EventBody:
    """The event stream ``invoke_endpoint_with_response_stream`` returns: PayloadPart events, then
    whatever ``end`` raises (a dropped stream) or nothing."""

    def __init__(self, parts: Iterable[bytes], end: BaseException | None = None) -> None:
        self._parts = list(parts)
        self._end = end
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for part in self._parts:
            yield {"PayloadPart": {"Bytes": part}}
        if self._end is not None:
            raise self._end

    def close(self) -> None:
        self.closed = True
