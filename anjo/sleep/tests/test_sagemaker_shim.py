# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The SageMaker container's shim: ``/ping`` only once vLLM answers, ``/invocations`` forwarding the
envelope to vLLM (streamed when asked), and each personal adapter fetched from S3, checked as the
Modal upload checks it, and loaded once. A fake vLLM and a fake bucket answer; the HTTP test runs on
loopback only."""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from anjo.sleep.serving import config
from anjo.sleep.serving.sagemaker import shim
from anjo.sleep.tests.aws_fakes import Body, FakeAws, client_error

ADAPTER = "p-1a2b3c4d-n3"
BUCKET = "anjo-sleep-sagemaker-test"


def _envelope(path: str, body: dict[str, Any] | None = None, stream: bool = False) -> bytes:
    return json.dumps({"path": path, "body": body or {}, "stream": stream}).encode()


class FakeVllm:
    """vLLM's local server: health, the forwarded requests, and runtime adapter loads."""

    def __init__(self, healthy: bool = True) -> None:
        self.is_healthy = healthy
        self.forwarded: list[tuple[str, str, Any]] = []
        self.loads: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def healthy(self) -> bool:
        return self.is_healthy

    def forward(self, method: str, path: str, body: bytes | None) -> shim.Reply:
        with self._lock:
            self.forwarded.append((method, path, None if body is None else json.loads(body)))
        if path == "/v1/models":
            return shim.Reply(200, "application/json", [b'{"object": "list", "data": []}'])
        parsed = json.loads(body or b"{}")
        if parsed.get("stream"):
            return shim.Reply(200, "text/event-stream", [b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
                                                          b"data: [DONE]\n\n"])
        return shim.Reply(200, "application/json", [json.dumps({"model": parsed.get("model")}).encode()])

    def load_adapter(self, name: str, folder: Path) -> int:
        with self._lock:
            self.loads.append((name, str(folder)))
        assert sorted(p.name for p in folder.iterdir()) == sorted(config.ADAPTER_FILES)
        return 200


class Store:
    """The adapters prefix of the bucket, through the real S3 reader over a fake client."""

    def __init__(self, **meta: Any) -> None:
        self.meta = {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16, **meta}
        self.gets: list[str] = []
        self._lock = threading.Lock()

    def get(self, **params: Any) -> dict[str, Any]:
        assert params["Bucket"] == BUCKET
        with self._lock:
            self.gets.append(params["Key"])
        if not params["Key"].startswith(f"adapters/{ADAPTER}/"):
            raise client_error("NoSuchKey", 404, "GetObject")
        if params["Key"].endswith("adapter_config.json"):
            return {"Body": Body(json.dumps(self.meta).encode())}
        return {"Body": Body(b"\0" * 64)}

    def source(self) -> shim.S3Adapters:
        return shim.S3Adapters(FakeAws("s3", get_object=self.get), BUCKET)


def _make(tmp_path: Path, **meta: Any) -> tuple[shim.Shim, FakeVllm, Store]:
    vllm, store = FakeVllm(), Store(**meta)
    return shim.Shim(vllm, shim.Adapters(vllm, store.source(), tmp_path)), vllm, store


def _read(reply: shim.Reply) -> Any:
    return json.loads(b"".join(reply.chunks))


def test_shim_ping_waits_for_vllm(tmp_path: Path) -> None:
    vllm = FakeVllm(healthy=False)
    server = shim.Shim(vllm, shim.Adapters(vllm, Store().source(), tmp_path))
    assert server.ping() == 503  # vLLM still loading: SageMaker keeps waiting
    vllm.is_healthy = True
    assert server.ping() == 200

    with _serving(server) as base:
        vllm.is_healthy = False
        assert httpx.get(f"{base}/ping").status_code == 503
        vllm.is_healthy = True
        assert httpx.get(f"{base}/ping").status_code == 200
        models = httpx.post(f"{base}/invocations", content=_envelope("/v1/models"),
                            headers={"Content-Type": "application/json"})
        assert models.status_code == 200 and models.json() == {"object": "list", "data": []}
        assert vllm.forwarded[-1] == ("GET", "/v1/models", None)  # the cheap wake poll is a GET to vLLM
        body = {"model": config.MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True}
        with httpx.stream("POST", f"{base}/invocations", content=_envelope("/v1/chat/completions", body, True),
                          headers={"Content-Type": "application/json"}) as streamed:
            assert streamed.status_code == 200 and streamed.headers["content-type"] == "text/event-stream"
            assert b"".join(streamed.iter_bytes()).endswith(b"data: [DONE]\n\n")
        # The dev host sends every call on the stream API; a JSON reply goes back chunked too.
        tokens = httpx.post(f"{base}/invocations", content=_envelope("/tokenize", {"model": config.MODEL}, True),
                            headers={"Content-Type": "application/json"})
        assert tokens.status_code == 200 and tokens.headers.get("transfer-encoding") == "chunked"
        assert tokens.json() == {"model": config.MODEL}


def test_shim_loads_adapter_once_and_checks_rank_and_base(tmp_path: Path) -> None:
    server, vllm, store = _make(tmp_path / "ok")
    body = {"model": ADAPTER, "messages": [{"role": "user", "content": "hi"}]}
    replies: list[shim.Reply] = []

    def ask() -> None:
        replies.append(server.invoke(_envelope("/v1/chat/completions", body)))

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [r.status for r in replies] == [200] * 8
    assert [name for name, _ in vllm.loads] == [ADAPTER]  # loaded once, however many asked at once
    assert sorted(store.gets) == sorted(f"adapters/{ADAPTER}/{f}" for f in config.ADAPTER_FILES)  # exactly the two
    assert len([f for f in vllm.forwarded if f[1] == "/v1/chat/completions"]) == 8
    server.invoke(_envelope("/v1/chat/completions", {**body, "model": config.MODEL}))
    assert len(vllm.loads) == 1  # the base model needs no adapter

    for meta, code in (({"r": 64}, "rank"), ({"base_model_name_or_path": "other/model"}, "base model"),
                       ({"peft_type": "IA3"}, "LoRA")):
        refused, refused_vllm, _ = _make(tmp_path / code.replace(" ", "_"), **meta)
        reply = refused.invoke(_envelope("/v1/chat/completions", body))
        assert reply.status == 422 and code in _read(reply)["error"]
        assert refused_vllm.loads == [] and refused_vllm.forwarded == []  # never loaded, never answered

    missing, missing_vllm, _ = _make(tmp_path / "missing")
    other = "p-9f9f9f9f-n1"
    reply = missing.invoke(_envelope("/v1/chat/completions", {**body, "model": other}))
    assert reply.status == 404 and missing_vllm.loads == []


def test_shim_forwards_only_the_vllm_paths_the_clients_use(tmp_path: Path) -> None:
    server, vllm, _ = _make(tmp_path)
    for raw, status in ((_envelope("/v1/load_lora_adapter", {"lora_name": "x", "lora_path": "/tmp"}), 400),
                        (b"not json", 400), (json.dumps({"path": "/v1/models", "body": [], "stream": False}).encode(), 400),
                        (json.dumps({"path": "/v1/models", "body": {}, "stream": "yes"}).encode(), 400)):
        assert server.invoke(raw).status == status
    assert vllm.forwarded == []  # nothing outside the envelope reaches vLLM

    reply = server.invoke(_envelope("/v1/chat/completions", {"model": config.MODEL, "messages": []}))
    assert reply.status == 200 and vllm.forwarded[-1][2]["max_tokens"] == config.MAX_OUTPUT_TOKENS
    too_long = {"model": config.MODEL, "messages": [], "max_tokens": config.MAX_OUTPUT_TOKENS + 1}
    assert server.invoke(_envelope("/v1/chat/completions", too_long)).status == 400
    tokens = server.invoke(_envelope("/tokenize", {"model": config.MODEL, "prompt": "x"}))
    assert tokens.status == 200 and "max_tokens" not in vllm.forwarded[-1][2]


def test_shim_caps_max_completion_tokens_like_max_tokens(tmp_path: Path) -> None:
    # OpenAI's newer name for the same limit: held to the same range, and the smaller one wins.
    server, vllm, _ = _make(tmp_path)
    chat = {"model": config.MODEL, "messages": []}
    top = config.MAX_OUTPUT_TOKENS
    for limits in ({"max_completion_tokens": top + 1}, {"max_completion_tokens": 0},
                   {"max_completion_tokens": True}, {"max_completion_tokens": "64"},
                   {"max_tokens": 64, "max_completion_tokens": top + 1}):
        assert server.invoke(_envelope("/v1/chat/completions", {**chat, **limits})).status == 400, limits
    for limits, sent in (({"max_completion_tokens": 64}, 64), ({"max_tokens": 64, "max_completion_tokens": 32}, 32),
                         ({"max_tokens": 32, "max_completion_tokens": 64}, 32)):
        assert server.invoke(_envelope("/v1/chat/completions", {**chat, **limits})).status == 200
        forwarded = vllm.forwarded[-1][2]
        assert forwarded["max_tokens"] == sent and "max_completion_tokens" not in forwarded, limits


def test_the_container_serves_from_s3_weights_with_the_registered_flags(tmp_path: Path) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    command = shim.vllm_command(config.ServeConfig.from_env({config.REVISION_ENV: revision}))
    assert command[:3] == ["vllm", "serve", shim.MODEL_DIR]  # SageMaker copies the staged S3 weights here
    assert command[command.index("--host") + 1] == "127.0.0.1"  # only the shim is reachable
    assert command[command.index("--port") + 1] == str(shim.VLLM_PORT) != str(shim.SHIM_PORT)
    sizes = {flag: command[command.index(flag) + 1] for flag in
             ("--max-lora-rank", "--max-loras", "--max-cpu-loras", "--max-model-len", "--quantization")}
    assert sizes == {"--max-lora-rank": "32", "--max-loras": "8", "--max-cpu-loras": "16",
                     "--max-model-len": str(config.MAX_MODEL_LEN), "--quantization": "fp8"}
    assert "--enable-lora" in command and "--language-model-only" in command
    env = shim.vllm_env({})
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True" and env["HF_HUB_OFFLINE"] == "1"
    assert "VLLM_PLUGINS" not in env  # adapters come from the shim's checked S3 fetch, not a resolver
    assert shim.SHIM_PORT == 8080


def _serving(server: shim.Shim) -> Any:
    from contextlib import contextmanager

    @contextmanager
    def running() -> Iterator[str]:
        httpd = shim.http_server(server, "127.0.0.1", 0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    return running()
