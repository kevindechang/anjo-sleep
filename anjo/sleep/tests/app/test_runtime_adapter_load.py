# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The chat runtime loads a person's adapter into Modal's server before its first turn there (vLLM
0.28.0 answers 404 for one it hasn't been told to load, seen live 2026-10-28), through the client's
own load and its "already loaded" handling. A turn makes one load attempt at most and never waits for
a folder to appear: a missing adapter or a failed load ends the turn as transient, under the reply
deadline, so the frontier talker answers. Only the night, the morning and the demo's pre-generated
pairs wait for a folder (``client.LOAD_WAIT_S``). A fake vLLM answers as the live one did; no network."""
from __future__ import annotations

import json
import threading
import time
from datetime import date
from typing import Any

import httpx
import pytest

from anjo.core.model_io import Finish, Message, ModelRequest, TextDelta, TextPart
from anjo.sleep.app import demo_live, runtime
from anjo.sleep.serving import config
from anjo.sleep.serving.client import LOAD_PATH, LOAD_RETRY_S, LOAD_WAIT_S, HttpPoster, VllmClient
from anjo.sleep.tests.app.glimmer_server import CHAT_IDS, HEADER_IDS, Modal, chunks

REVISION = "0123456789abcdef0123456789abcdef01234567"
URL = "https://anjo--anjo-sleep-serve-serve-web.modal.run"
TOKEN = "test-modal-token"
ADAPTER = "p-aaaaaaaaaaaa-n1"


class LiveVllm:
    """``/tokenize`` and ``/v1/load_lora_adapter`` as vLLM 0.28.0 answered live: an adapter is known
    only once loaded from a folder it sees (``on_disk``), loading one twice is a 400, and a restart
    forgets every load. ``status`` answers every load instead, when set; ``timeout`` makes it hang
    past the deadline. ``deadlines`` holds each load's read timeout."""

    def __init__(self) -> None:
        self.on_disk = {ADAPTER}
        self.loaded: set[str] = set()
        self.paths: list[str] = []
        self.deadlines: list[float] = []
        self.status: int | None = None
        self.timeout = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.paths.append(request.url.path)
        if request.url.path == LOAD_PATH:
            self.deadlines.append(request.extensions["timeout"]["read"])
            return self._load(request, body)
        if body["model"] not in (config.MODEL, *self.loaded):
            return httpx.Response(404, json={"error": {"message": f"The model `{body['model']}` does not exist."}})
        return httpx.Response(200, json={"tokens": CHAT_IDS if "messages" in body else HEADER_IDS})

    def _load(self, request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        name = body["lora_name"]
        if self.timeout:
            raise httpx.ReadTimeout("no reply within the deadline", request=request)
        if self.status is not None:
            return httpx.Response(self.status, json={"error": {"message": "failed"}})
        if name in self.loaded:
            return httpx.Response(400, json={"error": {"message": f"The lora adapter '{name}' has already been loaded."}})
        if body["lora_path"] != f"{config.ADAPTERS_DIR}/{name}" or name not in self.on_disk:
            return httpx.Response(404, json={"error": {"message": f"No adapter found for {name}"}})
        self.loaded.add(name)
        return httpx.Response(200, text=f"Success: LoRA adapter '{name}' added successfully.")


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> LiveVllm:
    """Modal's server, reached by the poster the runtime makes for each call (``_made_poster``, as on a
    simulated day), whose deadline and bearer token are the runtime's own; only the network is fake."""
    server = LiveVllm()

    class Reaching(HttpPoster):
        def __init__(self, base_url: str, token: str, *, timeout_s: float) -> None:
            super().__init__(base_url, token, transport=httpx.MockTransport(server), timeout_s=timeout_s)

    for name, value in {"ANJO_SLEEP_BACKEND": "modal", runtime.URL_ENV: URL, runtime.TOKEN_ENV: TOKEN,
                        config.REVISION_ENV: REVISION}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(runtime, "HttpPoster", Reaching)
    monkeypatch.setattr(runtime, "record_served", lambda *_args, **_kw: None)
    monkeypatch.setattr(runtime, "_LOADED", set())  # what this process has loaded, fresh per test
    return server


def _turn(server: Modal, adapter: str | None = ADAPTER) -> list[object]:
    """One turn through a runtime of its own, as the chat route makes one per turn."""
    rendering = runtime.glimmer_pin("u-1", None).rendering
    pin = runtime.glimmer_pin("u-1", adapter, adapter_rendering=None if adapter is None else rendering)
    request = ModelRequest(turn_id="t1", system=(), messages=(Message("user", (TextPart("hey"),)),),
                           max_output_tokens=64, round_index=0)
    return list(runtime.SleepRuntime(pin, transport=server).stream(request, threading.Event()))


def _said(events: list[object]) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


def test_chat_runtime_loads_adapter_before_first_turn(live: LiveVllm) -> None:
    server = Modal(chunks("Hi."), chunks("Again."), chunks("Back."), chunks("Base."))

    assert _said(_turn(server)) == "Hi."
    assert live.paths == [LOAD_PATH, "/tokenize", "/tokenize"]  # loaded before the adapter is named
    assert server.opened[0][2]["model"] == ADAPTER

    assert _said(_turn(server)) == "Again."  # its next turn, on the same server: no load
    assert live.paths.count(LOAD_PATH) == 1

    live.loaded.clear()  # the container restarted, and forgot every load
    assert _said(_turn(server)) == "Back."  # one load, then the turn goes on
    assert live.paths.count(LOAD_PATH) == 2

    assert _said(_turn(server, adapter=None)) == "Base."  # the base model is never loaded
    assert live.paths.count(LOAD_PATH) == 2


def test_an_adapter_the_night_already_loaded_counts_as_loaded(live: LiveVllm) -> None:
    live.loaded.add(ADAPTER)  # the night's client loaded it into this container
    assert _said(_turn(Modal(chunks("Hi.")))) == "Hi."
    assert live.paths == [LOAD_PATH, "/tokenize", "/tokenize"]  # its 400 "already been loaded" was success


@pytest.mark.parametrize("failure", ["missing", "refused", "hung"])
def test_chat_runtime_load_failure_falls_back_within_deadline(live: LiveVllm, failure: str) -> None:
    server = Modal(chunks("never sent"))
    if failure == "missing":
        live.on_disk.clear()  # the folder isn't there, or the warm server can't see it yet
    live.status = 500 if failure == "refused" else None
    live.timeout = failure == "hung"
    started = time.monotonic()

    assert _turn(server) == [Finish("error", "transient")]  # so the frontier talker answers

    assert time.monotonic() - started < LOAD_RETRY_S  # one attempt: never the night's wait for a folder
    assert live.paths == [LOAD_PATH] and live.deadlines == [runtime.REPLY_DEADLINE_S]
    assert server.opened == []
    assert LOAD_WAIT_S > runtime.REPLY_DEADLINE_S  # which is why a turn never waits as the night does
    live.status, live.timeout, live.on_disk = None, False, {ADAPTER}
    _turn(Modal(chunks("Hi.")))
    assert live.paths.count(LOAD_PATH) == 2  # a failed load isn't remembered: the next turn tries again


def test_only_pre_generated_demo_pairs_wait_for_an_adapter_folder() -> None:
    class Serving:
        def client(self) -> VllmClient:
            return VllmClient(URL, TOKEN, adapters_dir=config.ADAPTERS_DIR)

    assert VllmClient(URL, TOKEN, adapters_dir=config.ADAPTERS_DIR).load_wait_s == LOAD_WAIT_S  # night, morning
    live_page = demo_live.glimmer_deps("u-1", Serving(), load_wait_s=0.0)  # type: ignore[arg-type]
    assert live_page(date(2026, 10, 28)).model.load_wait_s == 0.0  # type: ignore[attr-defined]
    pairs = demo_live.glimmer_deps("u-1", Serving(), load_wait_s=LOAD_WAIT_S)  # type: ignore[arg-type]
    assert pairs(date(2026, 10, 28)).model.load_wait_s == LOAD_WAIT_S  # type: ignore[attr-defined]


def test_chat_deadline_has_no_retry(live: LiveVllm) -> None:
    from anjo.sleep.serving import client as vllm

    live.timeout = True  # a request lost on the way: the turn ends at its deadline, sent once
    started = time.monotonic()
    assert _turn(Modal(chunks("never sent"))) == [Finish("error", "transient")]
    assert live.paths == [LOAD_PATH] and live.deadlines == [runtime.REPLY_DEADLINE_S]
    assert time.monotonic() - started < vllm.BACKOFF_S[0]  # never a pause before a second attempt

    lost: list[str] = []

    def lose(request: httpx.Request) -> httpx.Response:
        lost.append(request.url.path)
        raise httpx.ReadTimeout("lost on the way", request=request)

    class Serving:
        def client(self) -> VllmClient:
            return VllmClient(URL, TOKEN, transport=httpx.MockTransport(lose), adapters_dir=config.ADAPTERS_DIR)

    page = demo_live.glimmer_deps("u-1", Serving(), load_wait_s=0.0)(date(2026, 10, 28)).model  # type: ignore[arg-type]
    with pytest.raises(vllm.ServingError, match="failed in transport"):
        page.complete(None, [{"role": "user", "content": "hi"}], temperature=1.0)
    assert lost == ["/tokenize"] and page.retries() == 0 and page.attempts == 1  # type: ignore[attr-defined]
    pairs = demo_live.glimmer_deps("u-1", Serving(), load_wait_s=LOAD_WAIT_S)(date(2026, 10, 28)).model  # type: ignore[arg-type]
    assert pairs.attempts == vllm.ATTEMPTS  # type: ignore[attr-defined]  # nobody waits on a pre-generated pair
