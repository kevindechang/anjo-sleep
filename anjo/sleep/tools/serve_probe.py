"""Time a live Modal server step by step, to tell a wedged server from a slow one
(``python -m anjo.sleep.tools.serve_probe``).

    python -m anjo.sleep.tools.serve_probe                                    # print the plan; sends nothing
    python -m anjo.sleep.tools.serve_probe --go                               # the base model's steps
    python -m anjo.sleep.tools.serve_probe --adapter p-<hash>-n<night> --go   # then one adapter's

With ``--go`` it wakes the server (``client.wake``), then times ``GET /v1/models``, ``POST /tokenize`` and
a 4-token ``/v1/completions`` on the base model, one after another, each under ``STEP_TIMEOUT_S``. With
``--adapter`` it then loads that adapter from the adapters volume's mount, stages it onto the
container's own disk when the mount doesn't show it (``backend.stage_on_modal``, which needs the modal
package and its login), loads it from there, and repeats the three steps naming it. Each step prints one
JSON line: its name, whether it worked, its milliseconds, and the client's error (endpoint and status, or
the transport error's type) when it failed. Never the URL, the token, a prompt or a reply.

It costs money, so only the coordinator runs it: it wakes the GPU container if it is down (minutes of
L40S time, which scales down ``config.SCALEDOWN_S`` after the last request) and spends a few tokens.
The URL and token come from ``ANJO_SLEEP_MODAL_URL`` and ``ANJO_SLEEP_MODAL_TOKEN``, read only with
``--go``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import httpx

from anjo.sleep.serving import backend, client, config
from anjo.sleep.serving.client import HttpPoster, ServingError, Stage

STEP_TIMEOUT_S: Final = 30.0  # past the slowest warm /tokenize seen under load (8.6 s, night 1c)
PROMPT: Final = "Hello"
COMPLETION_TOKENS: Final = 4
_PLAN: Final = f"""plan (nothing is sent; add --go to run it):
  wake      GET /v1/models until it answers, at most {client.WAKE_TOTAL_TIMEOUT_S:.0f} s (a cold start takes minutes)
  models    GET /v1/models
  tokenize  POST /tokenize, base model
  complete  POST /v1/completions, base model, {COMPLETION_TOKENS} tokens
with --adapter NAME, then:
  load         POST /v1/load_lora_adapter from {config.ADAPTERS_DIR}/NAME (the mount as the container started)
  stage        on a 404 there: Serve.stage copies NAME onto {config.STAGED_ADAPTERS_DIR} (needs modal and its login)
  load_staged  POST /v1/load_lora_adapter from {config.STAGED_ADAPTERS_DIR}/NAME
  models, tokenize, complete  naming NAME
each step: {STEP_TIMEOUT_S:.0f} s at most. Cost: wakes the GPU container if it is down, and a few tokens.
needs {backend.MODAL_URL_ENV} and {backend.MODAL_TOKEN_ENV} (never printed)"""


def _timed(name: str, call: Callable[[], object], clock: Callable[[], float]) -> tuple[dict[str, Any], object]:
    """One step's line and its result: not ok when it returns False or None, or fails with the client's
    error, whose words name the endpoint and the status only."""
    started = clock()
    try:
        result = call()
    except ServingError as error:
        return {"step": name, "ok": False, "ms": round((clock() - started) * 1000), "error": str(error)}, None
    ok = result is not None and result is not False
    return {"step": name, "ok": ok, "ms": round((clock() - started) * 1000)}, result


def _model_steps(
    poster: HttpPoster, model: str, clock: Callable[[], float], emit: Callable[[dict[str, Any]], None],
) -> bool:
    """``/v1/models``, ``/tokenize`` and a short completion naming ``model``; True when all worked."""
    completion = {"model": model, "prompt": PROMPT, "max_tokens": COMPLETION_TOKENS, "temperature": 0.0, "seed": 0}
    steps: list[tuple[str, Callable[[], object]]] = [
        ("models", lambda: poster.answers(STEP_TIMEOUT_S)),
        ("tokenize", lambda: poster.post("/tokenize", {"model": model, "prompt": PROMPT}, timeout_s=STEP_TIMEOUT_S)),
        ("complete", lambda: poster.post("/v1/completions", completion, timeout_s=STEP_TIMEOUT_S)),
    ]
    ok = True
    for name, call in steps:
        line, _ = _timed(name, call, clock)
        emit(line)
        ok = ok and line["ok"]
    return ok


def _load(
    poster: HttpPoster, name: str, stage: Stage, clock: Callable[[], float], emit: Callable[[dict[str, Any]], None],
) -> bool:
    """Load the adapter from the mount; on a 404 there, stage it and load it from where it landed."""
    line, _ = _timed("load", lambda: poster.load_adapter(
        name, f"{config.ADAPTERS_DIR}/{name}", timeout_s=STEP_TIMEOUT_S), clock)
    emit(line)
    if line["ok"] or "error" in line:
        return bool(line["ok"])
    line, folder = _timed("stage", lambda: stage(name), clock)
    emit(line)
    if not line["ok"]:
        return False
    line, _ = _timed("load_staged", lambda: poster.load_adapter(
        name, f"{folder}/{name}", timeout_s=STEP_TIMEOUT_S), clock)
    emit(line)
    return bool(line["ok"])


def probe(
    poster: HttpPoster, token: str, adapter: str | None, stage: Stage, *, transport: httpx.BaseTransport | None,
    clock: Callable[[], float], sleep: Callable[[float], None], emit: Callable[[dict[str, Any]], None],
) -> bool:
    """Run the steps, emitting each line; True when every one needed worked. A wake that fails ends it."""
    woke = client.wake(poster.base_url, token, transport=transport, clock=clock, sleep=sleep)
    emit({"step": "wake", "ok": woke.ok, "ms": round(woke.seconds * 1000)})
    if not woke.ok:
        return False
    ok = _model_steps(poster, config.MODEL, clock, emit)
    if adapter is None:
        return ok
    return _load(poster, adapter, stage, clock, emit) and _model_steps(poster, adapter, clock, emit) and ok


def _setting(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def main(
    argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None, stage: Stage | None = None,
    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(description="Time a live Modal server step by step.")
    parser.add_argument("--adapter", help="an adapter p-<hash>-n<night> to load (staging it if needed) and probe")
    parser.add_argument("--go", action="store_true", help="send the requests; without it, print the plan only")
    args = parser.parse_args(argv)
    if args.adapter is not None and not config.ADAPTER_NAME.fullmatch(args.adapter):
        parser.error("--adapter is named p-<hash>-n<night>")
    if not args.go:
        print(_PLAN)
        return 0
    env = os.environ if environ is None else environ
    url, token = _setting(env, backend.MODAL_URL_ENV), _setting(env, backend.MODAL_TOKEN_ENV)
    poster = HttpPoster(url, token, transport=transport)
    try:
        ok = probe(poster, token, args.adapter, stage or backend.stage_on_modal, transport=transport,
                   clock=clock, sleep=sleep, emit=lambda line: print(json.dumps(line)))
    finally:
        poster.close()
    print(json.dumps({"step": "done", "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
