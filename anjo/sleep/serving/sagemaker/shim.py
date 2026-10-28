"""The SageMaker container's front door on port 8080, in front of vLLM on loopback.

    python3 -m anjo.sleep.serving.sagemaker.shim serve     # the image's entrypoint

- ``GET /ping`` answers 200 only once vLLM's own ``/health`` does, so SageMaker routes nothing to a
  copy still loading Glimmer.
- ``POST /invocations`` takes the envelope ``{"path", "body", "stream"}`` and forwards it to vLLM:
  only the paths the dev host uses (``ROUTES``), streamed back as vLLM streams it, and chunked
  whenever the envelope asks for a stream. A chat or a completion gets the server's output limit
  when it names none, and is refused above it, under either name (``max_tokens``,
  ``max_completion_tokens``); vLLM gets the smaller as ``max_tokens``.
- A request naming a personal adapter ``p-<hash>-n<night>`` loads it first, once: exactly
  ``adapter_config.json`` and ``adapter_model.safetensors`` from ``s3://<bucket>/adapters/<name>/``,
  held to the same checks as Modal's upload (``config.check_adapter``), then vLLM's runtime
  ``/v1/load_lora_adapter``. Requests racing for one adapter wait for the one load.

The weights are the staged S3 copy SageMaker puts in ``MODEL_DIR``; nothing is read from Hugging
Face. Standard library, boto3, ``config`` and ``settings`` only: the image carries just these files.
The shim logs request lines, never a body.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, NoReturn, Protocol

from anjo.sleep.serving import config
from anjo.sleep.serving.sagemaker import settings

SHIM_PORT: Final = 8080  # SageMaker's
VLLM_PORT: Final = 8000  # loopback only
MODEL_DIR: Final = "/opt/ml/model"  # where SageMaker copies the staged weights
ADAPTERS_DIR: Final = "/tmp/anjo-sleep-adapters"
MAX_REQUEST_BYTES: Final = 6 * 1024 * 1024  # SageMaker's own payload limit
FORWARD_TIMEOUT_S: Final = 300.0
HEALTH_TIMEOUT_S: Final = 5.0
LOAD_TIMEOUT_S: Final = 120.0
CHUNK: Final = 1 << 20
ROUTES: Final[Mapping[str, str]] = {
    "/v1/chat/completions": "POST", "/v1/completions": "POST", "/tokenize": "POST", "/v1/models": "GET",
}
GENERATES: Final = frozenset({"/v1/chat/completions", "/v1/completions"})
_LIMITS: Final = ("max_tokens", "max_completion_tokens")


@dataclass(frozen=True)
class Reply:
    status: int
    content_type: str
    chunks: Iterable[bytes]
    chunked: bool = False  # sent chunked even if not events: the caller is on the stream API


def _error(status: int, message: str) -> Reply:
    return Reply(status, "application/json", [json.dumps({"error": message}).encode()])


class Vllm(Protocol):
    def healthy(self) -> bool: ...

    def forward(self, method: str, path: str, body: bytes | None) -> Reply: ...

    def load_adapter(self, name: str, folder: Path) -> int: ...


class AdapterSource(Protocol):
    def fetch(self, name: str, into: Path) -> None: ...


class AdapterMissing(LookupError):
    """No adapter of that name in the bucket."""


class S3Adapters:
    """The bucket's ``adapters/`` prefix: exactly the two PEFT files of one adapter."""

    def __init__(self, s3: Any, bucket: str) -> None:
        self._s3, self._bucket = s3, bucket

    def fetch(self, name: str, into: Path) -> None:
        from botocore.exceptions import ClientError  # type: ignore[import-untyped]

        into.mkdir(parents=True, exist_ok=True)
        for file in config.ADAPTER_FILES:
            try:
                reply = self._s3.get_object(Bucket=self._bucket, Key=f"{settings.ADAPTERS_PREFIX}{name}/{file}")
            except ClientError as error:
                status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status in (403, 404):  # without ListBucket on the key, S3 says 403 for a missing one
                    raise AdapterMissing(name) from None
                raise
            _write(reply["Body"], into / file)


def _write(body: Any, target: Path) -> None:
    written = 0
    try:
        with target.open("wb") as out:
            while chunk := body.read(CHUNK):
                written += len(chunk)
                if written > config.MAX_ADAPTER_BYTES:
                    raise config.ServeConfigError("the adapter is larger than any rank the server takes")
                out.write(chunk)
    finally:
        body.close()


class Adapters:
    """Each adapter fetched, checked and loaded into vLLM once, however many requests ask at once."""

    def __init__(self, vllm: Vllm, source: AdapterSource, root: Path) -> None:
        self._vllm, self._source, self._root = vllm, source, root
        self._loaded: set[str] = set()
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def ensure(self, name: str) -> Reply | None:
        """None once the adapter is loaded; otherwise the refusal to send back."""
        if name in self._loaded:
            return None
        with self._guard:
            lock = self._locks.setdefault(name, threading.Lock())
        with lock:
            if name in self._loaded:
                return None
            refusal = self._load(name)
            if refusal is None:
                self._loaded.add(name)
            return refusal

    def _load(self, name: str) -> Reply | None:
        folder = self._root / name
        shutil.rmtree(folder, ignore_errors=True)  # what a failed try left
        try:
            self._source.fetch(name, folder)
            config.check_adapter(folder, name)
        except AdapterMissing:
            return _error(404, f"no adapter named {name}")
        except config.ServeConfigError as error:
            shutil.rmtree(folder, ignore_errors=True)
            return _error(422, str(error))
        status = self._vllm.load_adapter(name, folder)
        return None if status == 200 else _error(502, f"vLLM refused the adapter (HTTP {status})")


class Shim:
    def __init__(self, vllm: Vllm, adapters: Adapters) -> None:
        self._vllm, self._adapters = vllm, adapters

    def ping(self) -> int:
        return 200 if self._vllm.healthy() else 503

    def invoke(self, raw: bytes) -> Reply:
        try:
            request = json.loads(raw)
        except ValueError:
            return _error(400, "the envelope is JSON")
        if not isinstance(request, dict):
            return _error(400, "the envelope is an object")
        path, body, stream = request.get("path"), request.get("body"), request.get("stream")
        if not isinstance(path, str) or path not in ROUTES or not isinstance(body, dict) or not isinstance(stream, bool):
            return _error(400, "the envelope is {path, body, stream}, for a path the dev host uses")
        reply = self._route(path, body)
        return replace(reply, chunked=True) if stream else reply

    def _route(self, path: str, body: dict[str, Any]) -> Reply:
        if ROUTES[path] == "GET":
            return self._forward("GET", path, None)
        model = body.get("model")
        if model != config.MODEL:
            if not isinstance(model, str) or not config.ADAPTER_NAME.fullmatch(model):
                return _error(400, "the model is Glimmer or one person's adapter")
            refusal = self._adapters.ensure(model)
            if refusal is not None:
                return refusal
        if path in GENERATES:  # either name for the output limit, held to the same range; the smaller wins
            limits = [body.get(name, config.MAX_OUTPUT_TOKENS) for name in _LIMITS]
            if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= config.MAX_OUTPUT_TOKENS
                   for n in limits):
                return _error(400, f"max_tokens and max_completion_tokens are 1 to {config.MAX_OUTPUT_TOKENS}")
            body = {**{k: v for k, v in body.items() if k not in _LIMITS}, "max_tokens": min(limits)}
        return self._forward("POST", path, json.dumps(body).encode())

    def _forward(self, method: str, path: str, body: bytes | None) -> Reply:
        try:
            return self._vllm.forward(method, path, body)
        except OSError:
            return _error(503, "vLLM is not answering")


class LocalVllm:
    """vLLM's server on loopback, started by ``main``."""

    def __init__(self, port: int = VLLM_PORT, process: subprocess.Popen[bytes] | None = None) -> None:
        self._port, self._process = port, process

    def _connect(self, timeout_s: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout_s)

    def healthy(self) -> bool:
        if self._process is not None and self._process.poll() is not None:
            return False  # vLLM exited: SageMaker replaces this copy
        connection = self._connect(HEALTH_TIMEOUT_S)
        try:
            connection.request("GET", "/health")
            return connection.getresponse().status == 200
        except OSError:
            return False  # still loading
        finally:
            connection.close()

    def forward(self, method: str, path: str, body: bytes | None) -> Reply:
        connection = self._connect(FORWARD_TIMEOUT_S)
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        response = connection.getresponse()

        def chunks() -> Iterator[bytes]:
            try:
                while chunk := response.read1(CHUNK):
                    yield chunk
            finally:
                connection.close()

        return Reply(response.status, response.getheader("Content-Type") or "application/json", chunks())

    def load_adapter(self, name: str, folder: Path) -> int:
        connection = self._connect(LOAD_TIMEOUT_S)
        try:
            connection.request("POST", "/v1/load_lora_adapter", headers={"Content-Type": "application/json"},
                               body=json.dumps({"lora_name": name, "lora_path": str(folder)}).encode())
            response = connection.getresponse()
            said = response.read()
        finally:
            connection.close()
        if response.status == 400 and b"already been loaded" in said:
            return 200
        return response.status


def vllm_command(serve: config.ServeConfig) -> list[str]:
    """Modal's command line, on the staged weights and on loopback."""
    return config.vllm_command(serve, model_path=MODEL_DIR, host="127.0.0.1", port=VLLM_PORT)


def vllm_env(base: Mapping[str, str]) -> dict[str, str]:
    """Runtime adapter loads on (the shim's checked fetch, not a resolver plugin), and nothing
    fetched from Hugging Face."""
    kept = {k: v for k, v in base.items() if k not in ("VLLM_PLUGINS", "VLLM_LORA_RESOLVER_CACHE_DIR")}
    return {**kept, **config.NATIVE_SAMPLER_ENV, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def http_server(shim: Shim, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802  (http.server's name)
            self._send(Reply(shim.ping(), "application/json", [b""]) if self.path == "/ping" else _error(404, "no route"))

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            if self.path != "/invocations" or not 0 < length <= MAX_REQUEST_BYTES:
                self.close_connection = True  # the body stays unread: this connection can't carry another
                self._send(_error(404, "no route") if self.path != "/invocations"
                           else _error(413, "the request is empty or too large"))
                return
            self._send(shim.invoke(self.rfile.read(length)))

        def _send(self, reply: Reply) -> None:
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.content_type)
            try:
                if reply.chunked or reply.content_type.startswith("text/event-stream"):
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for chunk in reply.chunks:
                        if chunk:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                            self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    data = b"".join(reply.chunks)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            finally:
                close = getattr(reply.chunks, "close", None)
                if close is not None:
                    close()  # a client gone mid-stream ends vLLM's request too

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            if self.path != "/ping":  # the request line only, never a body
                super().log_request(code, size)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _stop(*_args: object) -> NoReturn:
    raise SystemExit(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Glimmer's SageMaker container: vLLM behind the shim.")
    parser.add_argument("command", nargs="?", default="serve", choices=("serve",))  # SageMaker passes "serve"
    parser.parse_args(argv)
    import boto3  # type: ignore[import-untyped]
    from botocore.config import Config  # type: ignore[import-untyped]

    serve = config.ServeConfig.from_env()
    bucket = settings.check_bucket(os.environ.get(settings.BUCKET_ENV, "").strip())
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or settings.REGION, config=Config(
        connect_timeout=10, read_timeout=60, retries={"total_max_attempts": 3, "mode": "standard"}))
    process = subprocess.Popen(vllm_command(serve), env=vllm_env(os.environ))
    vllm = LocalVllm(VLLM_PORT, process)
    server = http_server(Shim(vllm, Adapters(vllm, S3Adapters(s3, bucket), Path(ADAPTERS_DIR))), "0.0.0.0", SHIM_PORT)
    signal.signal(signal.SIGTERM, _stop)  # SageMaker stops a copy with SIGTERM: stop vLLM too
    try:
        server.serve_forever()
    finally:
        server.server_close()
        process.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
