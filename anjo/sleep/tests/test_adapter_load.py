# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Adapters load by an explicit call (seen live, 2026-10-28): vLLM 0.28.0 served base Glimmer but
answered 404 for an adapter on the volume until ``/v1/load_lora_adapter`` loaded it. So the client
loads each adapter before the first request that names it, from the folder its backend's server reads
(Modal's adapters volume, the APD Colab session's LoRA folder), and SageMaker's shim, which fetches and
loads by itself, is never asked. A fake vLLM answers as the live one did; no network."""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pytest

from anjo.sleep.mind.apd import colab_vllm
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.backend import ModalServing
from anjo.sleep.serving.client import LOAD_PATH, LOAD_RETRY_S, LOAD_WAIT_S, ServingError, VllmClient
from anjo.sleep.serving.sagemaker.serving import SageMakerServing
from anjo.sleep.tests.aws_fakes import EventBody, FakeAws
from anjo.sleep.tests.test_sagemaker import BUCKET
from anjo.sleep.types import CheckpointRef

NIGHT = date(2026, 10, 28)
ADAPTER = CheckpointRef("p-aaaaaaaaaaaa-n1", "p1", NIGHT, "modal-volume://x", None, framing.rendering("low"))
MESSAGES = [{"role": "user", "content": "hi"}]


class LiveVllm:
    """vLLM as seen live: the base is served, and an adapter only once ``/v1/load_lora_adapter`` loaded
    it from a folder the server sees (``on_disk``). Loading one already loaded is a 400, and a restart
    forgets every load. ``status`` answers every load instead, when set; ``sticky`` False forgets
    each load at once."""

    def __init__(self, on_disk: set[str] | None = None) -> None:
        self.on_disk = {ADAPTER.id} if on_disk is None else on_disk
        self.loaded: set[str] = set()
        self.paths: list[str] = []
        self.loads: list[dict[str, Any]] = []
        self.status: int | None = None
        self.sticky = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.paths.append(request.url.path)
        if request.url.path == LOAD_PATH:
            return self._load(body)
        if body.get("model") not in (config.MODEL, *self.loaded):
            return httpx.Response(404, json={"error": {"message": f"The model `{body.get('model')}` does not exist."}})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3] if "messages" in body else [5, 6]})
        if "prompt_logprobs" in body:
            logprobs = [None] + [{str(t): {"logprob": -0.5}} for t in body["prompt"][1:]]
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": logprobs, "text": "x"}]})
        return httpx.Response(200, json={"model": body["model"], "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                                         "choices": [{"text": "hello", "finish_reason": "stop"}]})

    def _load(self, body: dict[str, Any]) -> httpx.Response:
        self.loads.append(body)
        name = body["lora_name"]
        if self.status is not None:
            return httpx.Response(self.status, json={"error": {"message": f"private {body['lora_path']}"}})
        if name in self.loaded:
            return httpx.Response(400, json={"error": {"message": f"The lora adapter '{name}' has already been loaded."}})
        if body["lora_path"] != f"{config.ADAPTERS_DIR}/{name}" or name not in self.on_disk:
            return httpx.Response(404, json={"error": {"message": f"No adapter found for {name}"}})
        if self.sticky:
            self.loaded.add(name)
        return httpx.Response(200, text=f"Success: LoRA adapter '{name}' added successfully.")


class Waits:
    """The client's clock and sleep: time moves only as it sleeps; ``then`` runs after each sleep."""

    def __init__(self, then: Any = None) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        self.then = then

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        if self.then is not None:
            self.then()


def _client(fake: LiveVllm, waits: Waits | None = None) -> VllmClient:
    waits = waits or Waits()
    return VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake),
                      adapters_dir=config.ADAPTERS_DIR, clock=waits.clock, sleep=waits.sleep)


def test_an_adapter_is_loaded_before_its_first_use_and_only_once() -> None:
    fake = LiveVllm()
    client = _client(fake)
    assert client.complete(ADAPTER, MESSAGES, temperature=1.0, seed=7).text == "hello"
    assert fake.paths == [LOAD_PATH, "/tokenize", "/tokenize", "/v1/completions"]  # loaded before it is named
    assert fake.loads == [{"lora_name": ADAPTER.id, "lora_path": f"{config.ADAPTERS_DIR}/{ADAPTER.id}"}]
    client.on(NIGHT).complete(ADAPTER, MESSAGES, temperature=1.0)  # a dated copy shares the loaded set
    client.continuation_logprobs(ADAPTER, MESSAGES, " A")
    client.complete(None, MESSAGES, temperature=1.0)  # the base model is never loaded
    assert len(fake.loads) == 1 and fake.paths.count(LOAD_PATH) == 1


def test_an_adapter_already_loaded_counts_as_loaded() -> None:
    fake = LiveVllm()
    fake.loaded.add(ADAPTER.id)  # another client loaded it into this container
    assert _client(fake).complete(ADAPTER, MESSAGES, temperature=1.0).text == "hello"
    assert len(fake.loads) == 1  # the 400 "already been loaded" was success


def test_a_404_after_loading_loads_again_once() -> None:
    fake = LiveVllm()
    client = _client(fake)
    client.complete(ADAPTER, MESSAGES, temperature=1.0)
    fake.loaded.clear()  # the container restarted, and forgot every load
    assert client.complete(ADAPTER, MESSAGES, temperature=1.0).text == "hello"
    assert len(fake.loads) == 2

    fake.loaded.clear()
    fake.sticky = False  # a load that never holds: one reload, then the 404 stands
    with pytest.raises(ServingError, match="HTTP 404"):
        client.complete(ADAPTER, MESSAGES, temperature=1.0)
    assert len(fake.loads) == 3  # one load after the 404, then the 404 stands: never a loop


def test_a_load_failure_is_a_serving_error_naming_the_endpoint_only() -> None:
    fake = LiveVllm()
    fake.status = 500
    with pytest.raises(ServingError) as raised:
        _client(fake).complete(ADAPTER, [{"role": "user", "content": "private words"}], temperature=1.0)
    assert str(raised.value) == f"{LOAD_PATH} returned HTTP 500"
    assert ADAPTER.id not in str(raised.value) and "private" not in str(raised.value)
    assert fake.paths == [LOAD_PATH]  # nothing names an adapter that didn't load


def test_a_new_adapter_is_waited_for_until_a_warm_server_sees_it() -> None:
    fake = LiveVllm(on_disk=set())
    waits = Waits(then=lambda: fake.on_disk.add(ADAPTER.id))  # the server's next volume refresh shows it
    assert _client(fake, waits).complete(ADAPTER, MESSAGES, temperature=1.0).text == "hello"
    assert waits.slept == [LOAD_RETRY_S] and len(fake.loads) == 2

    never = LiveVllm(on_disk=set())
    waits = Waits()
    with pytest.raises(ServingError, match=LOAD_PATH) as raised:
        _client(never, waits).complete(ADAPTER, MESSAGES, temperature=1.0)
    assert ADAPTER.id not in str(raised.value)
    assert LOAD_WAIT_S - LOAD_RETRY_S <= sum(waits.slept) <= LOAD_WAIT_S + LOAD_RETRY_S
    assert LOAD_WAIT_S > config.ADAPTER_REFRESH_S  # the wait outlasts the server's refresh period


def test_the_rendering_is_checked_before_any_load() -> None:
    fake = LiveVllm()
    other = CheckpointRef(ADAPTER.id, "p1", NIGHT, ADAPTER.uri, None, framing.rendering("high"))
    with pytest.raises(ServingError, match="rendering"):
        _client(fake).complete(other, MESSAGES, temperature=1.0)
    assert fake.paths == []


def test_each_backend_loads_from_its_own_adapters_folder_and_sagemaker_never_loads() -> None:
    assert ModalServing("https://serve.example", "tok").client().adapters_dir == config.ADAPTERS_DIR
    session = colab_vllm.attach("anjo-mind-apd-1a2b3c4d", 8123, "k" * 40)
    assert session.client().adapters_dir == colab_vllm.ADAPTERS_DIR  # the folder upload_adapter fills

    paths: list[str] = []

    def stream(**params: Any) -> dict[str, Any]:
        paths.append(json.loads(params["Body"])["path"])
        if paths[-1] == "/tokenize":
            return {"Body": EventBody([b'{"tokens": [1, 2]}']), "ContentType": "application/json"}
        done = {"model": ADAPTER.id, "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1}}
        return {"Body": EventBody([b"data: " + json.dumps(done).encode() + b"\n\ndata: [DONE]\n\n"]),
                "ContentType": "text/event-stream"}

    night = SageMakerServing(bucket=BUCKET, runtime=FakeAws(
        "sagemaker-runtime", invoke_endpoint_with_response_stream=stream)).client()
    assert night.adapters_dir is None
    night.complete(ADAPTER, MESSAGES, temperature=1.0)
    assert paths == ["/tokenize", "/tokenize", "/v1/completions"]  # the shim fetches and loads by itself
