# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The dev host's route to Glimmer on SageMaker: the turn streams through the endpoint's envelope
under the 20 s reply deadline, and a stream that drops ends exactly as a Modal failure does, so the
frontier talker answers. The endpoint is a stand-in checked against botocore's service model."""
from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from botocore.exceptions import EventStreamError, ReadTimeoutError  # type: ignore[import-untyped]
from urllib3.exceptions import ProtocolError

from anjo.core.anjo_model.transport import TransportError
from anjo.core.model_io import Finish, Message, ModelRequest, TextDelta, TextPart
from anjo.sleep.app import ancestry, runtime
from anjo.sleep.app.sagemaker_transport import SageMakerTransport
from anjo.sleep.serving import config
from anjo.sleep.serving.backend import BACKEND_ENV
from anjo.sleep.serving.sagemaker import settings
from anjo.sleep.serving.sagemaker.invoke import Invoker
from anjo.sleep.tests.app.test_sleep_serving import (  # noqa: F401  (fixtures: flagged and what it needs)
    REVISION,
    Modal,
    _chunks,
    flagged,
    paid_env,
    sleep_hooks,
)
from anjo.sleep.tests.aws_fakes import EventBody, FakeAws, client_error
from tests.test_gemma_cloud_chat_stream import _post, _shown
from tests.test_sol_chat_stream import _mock_gemini_stream

STARTED = {"id": "c", "model": config.MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
WORDS = {"id": "c", "model": config.MODEL, "choices": [{"index": 0, "delta": {"content": "Hi there"}}]}


def _sse(*chunks: dict[str, Any]) -> bytes:
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)


def _drops() -> list[BaseException]:
    return [
        EventStreamError({"Error": {"Code": "ModelStreamError", "Message": "the container stopped"}},
                         "InvokeEndpointWithResponseStream"),
        ProtocolError("Connection broken"),
        ReadTimeoutError(endpoint_url="https://runtime.sagemaker.us-east-1.amazonaws.com"),
    ]


class Endpoint:
    """The SageMaker endpoint's streaming call: each request gets the next scripted body or error."""

    def __init__(self, *replies: EventBody | BaseException) -> None:
        self.replies = list(replies)
        self.fake = FakeAws("sagemaker-runtime", invoke_endpoint_with_response_stream=self._answer)

    def _answer(self, **params: Any) -> dict[str, Any]:
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return {"Body": reply, "ContentType": "text/event-stream"}

    def transport(self) -> SageMakerTransport:
        return SageMakerTransport(Invoker(self.fake, settings.ENDPOINT, settings.COMPONENT))


def _request() -> ModelRequest:
    return ModelRequest(turn_id="t1", system=(), messages=(Message("user", (TextPart("hey"),)),),
                        max_output_tokens=64, round_index=0)


def _stream(sleep: runtime.SleepRuntime) -> list[Any]:
    return list(sleep.stream(_request(), threading.Event()))


@pytest.fixture
def on_sagemaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV, "sagemaker")
    monkeypatch.setenv(config.REVISION_ENV, REVISION)


@pytest.mark.parametrize("drop", _drops(), ids=["model_stream_error", "connection_cut", "read_timeout"])
def test_stream_drop_falls_back_like_modal(on_sagemaker: None, monkeypatch: pytest.MonkeyPatch,
                                           drop: BaseException) -> None:
    served: list[str] = []
    monkeypatch.setattr(runtime, "record_served", lambda _user_id, turn_id, **_kw: served.append(turn_id))
    pin = runtime.glimmer_pin("u-1", None)

    before_words = Endpoint(EventBody([_sse(STARTED)], end=drop))
    on_modal = Modal([STARTED, TransportError(None)])
    assert _stream(runtime.SleepRuntime(pin, transport=before_words.transport())) == \
        _stream(runtime.SleepRuntime(pin, transport=on_modal))
    assert _stream(runtime.SleepRuntime(pin, transport=Endpoint(EventBody([_sse(STARTED)], end=drop)).transport()))[-1] \
        == Finish("error", "transient") and served == []  # nothing of Glimmer's was shown

    after_words = Endpoint(EventBody([_sse(STARTED, WORDS)[:40], _sse(STARTED, WORDS)[40:]], end=drop))
    events = _stream(runtime.SleepRuntime(pin, transport=after_words.transport()))
    modal_events = _stream(runtime.SleepRuntime(pin, transport=Modal([STARTED, WORDS, TransportError(None)])))
    assert events == modal_events and events[-1] == Finish("error", "transient")
    assert served == ["t1", "t1"]  # a partial reply after a word is Glimmer's, on both backends


def test_a_turn_streams_through_the_envelope_under_the_reply_deadline(on_sagemaker: None) -> None:
    endpoint = Endpoint(EventBody([_sse(*_chunks("Hello from Glimmer."))]),
                        client_error("ValidationError", 400, "InvokeEndpointWithResponseStream"))
    pin = runtime.glimmer_pin("u-1", "p-1a2b3c4d-n2")
    events = _stream(runtime.SleepRuntime(pin, transport=endpoint.transport()))
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("Hello from Glimmer.")]
    assert events[-1] == Finish("stop", None)
    (params,) = endpoint.fake.named("invoke_endpoint_with_response_stream")
    assert (params["EndpointName"], params["InferenceComponentName"]) == (settings.ENDPOINT, settings.COMPONENT)
    sent = json.loads(params["Body"])
    assert (sent["path"], sent["stream"], sent["body"]["model"]) == ("/v1/chat/completions", True, "p-1a2b3c4d-n2")
    assert "Authorization" not in json.dumps({k: v for k, v in params.items() if k != "Body"}) + params["Body"].decode()
    # (SigV4 signs the call; no bearer token travels)

    # No capacity (asleep) refuses at once, before any word: the fallback answers, as for a cold Modal.
    assert _stream(runtime.SleepRuntime(pin, transport=endpoint.transport())) == [Finish("error", "rejected")]

    default = runtime.SleepRuntime(pin)
    made: Any = default._transport
    cfg = made._invoker._client.meta.config
    assert isinstance(made, SageMakerTransport)
    assert (cfg.read_timeout, cfg.connect_timeout) == (runtime.REPLY_DEADLINE_S, runtime.CONNECT_DEADLINE_S)
    assert cfg.retries["total_max_attempts"] == 1 and made._invoker._client.meta.region_name == settings.REGION


def test_a_dropped_sagemaker_stream_is_answered_by_the_frontier_talker(
    auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
) -> None:
    from anjo.core.anjo_model import hosts

    monkeypatch.setenv(BACKEND_ENV, "sagemaker")
    endpoint = Endpoint(EventBody([_sse(STARTED)], end=_drops()[0]))
    monkeypatch.setattr(hosts, "RUNTIMES", {
        runtime.HOST: lambda pin: runtime.SleepRuntime(pin, transport=endpoint.transport()),  # type: ignore[arg-type, unused-ignore]
    })

    events, _spend, _gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    assert _shown(events) == "Gemini here." and len(endpoint.fake.calls) == 1
    (context,) = ancestry.export_ancestry(flagged)
    folded = ancestry.reply_ancestry(flagged, context["turn_id"])
    assert folded is not None and folded.served_by == "fallback" and folded.reply is None
