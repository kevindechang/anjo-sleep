"""Time one fixed slice of the night's gates on a live server at several concurrencies
(``python -m anjo.sleep.tools.gate_bench``), to pick the serving GPU and the concurrency by cost.

    python -m anjo.sleep.tools.gate_bench                          # the plan; sends nothing
    python -m anjo.sleep.tools.gate_bench --gpu H100 --go         # run it on the H100 app's URL
    python -m anjo.sleep.tools.gate_bench --safety anjo.sleep.app.safety_gate:load_cases --go

**The slice** is the gates' own requests, through the night's client (``VllmClient``, the gates' date,
sampling, seeds and reply cap): every 4th of the 200 general-skills items (each choice one
log-probability call, prefill only), and all 60 collapse prompts (a reply at the cap, then an 8-token
verdict). The night's safety gate is off by default (``ANJO_SLEEP_SAFETY_GATE``), so its cases are
optional (``--safety``, loaded by name like the night's wiring; ``--safety-cases``, 200 by default), as a
stand-in for any generation-heavy load. ``--adapter`` runs the slice on a person's adapter, as the
night's checkpoint side runs.

**Each level** (``--concurrency``, 8, 16 and 32 by default) runs the whole slice once, its prompts dated
a day of its own, so no level reads another's prompts from vLLM's prefix cache. It prints one JSON line
a gate: requests, seconds, tokens generated, tokens a second and dollars at the GPU's hourly price;
then one ``night`` line: one night's checkpoint side of the gates at that level (the base side is kept
across nights, ``gates.base_cache``), scaled up from the slice. Never a prompt's or a reply's words.
A level above ``config.CONCURRENT_REQUESTS`` is refused: the client would hold it there. Run 32 with
``ANJO_SLEEP_SERVE_MAX_INPUTS=32`` set on this host and on the app's deploy.

**Spend:** it wakes the app (a cold start of minutes) and runs at most ``BENCH_MAX_S``; no level or gate
starts after that. Its worst case, ``BENCH_MAX_S`` plus the wake and Modal's idle window after the last
request, is held on ``modal_usd`` before the first request and settled at what ran plus the idle
window. The URL and token come from ``ANJO_SLEEP_MODAL_URL`` (or ``--url``) and
``ANJO_SLEEP_MODAL_TOKEN``, read only with ``--go``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, Protocol

from anjo.sleep.budget import Bound, default_ledger
from anjo.sleep.gates import EVAL_DAY, EVAL_SEED, TEMPERATURE, in_order
from anjo.sleep.gates.collapse import JUDGE_TEMPERATURE, PROMPT_PATH, VERDICT_MAX_TOKENS
from anjo.sleep.gates.general import answer
from anjo.sleep.context.template import render
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import backend, config, gpus
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.sets.loader import CollapsePrompt, GeneralItem, load_collapse, load_general
from anjo.sleep.types import CheckpointRef

LEVELS: Final = (8, 16, 32)
GENERAL_EVERY: Final = 4  # every 4th of the 200 items: 50, across all three sources
SAFETY_CASES: Final = 200
BENCH_MAX_S: Final = 40 * 60.0
WAKE_S: Final = 600.0  # client.WAKE_TOTAL_TIMEOUT_S: a cold start's worst
POOL: Final = "modal_usd"
NIGHT_GENERAL_ITEMS: Final = 200
NIGHT_COLLAPSE_PROMPTS: Final = 60  # before a person's fields take some out
NIGHT_SAFETY_CASES: Final = 2015


class BenchClient(Protocol):
    """``VllmClient`` as the gates call it."""

    def on(self, day: Any) -> BenchClient: ...

    def complete(self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
                 seed: int | None = None, max_tokens: int | None = None) -> ChatResult: ...

    def continuation_logprobs(self, adapter: CheckpointRef | None, messages: list[dict[str, str]],
                              continuation: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class Slice:
    general: tuple[GeneralItem, ...]
    collapse: tuple[CollapsePrompt, ...]
    safety: tuple[str, ...]  # empty: the safety slice is off

    @property
    def general_calls(self) -> int:
        return sum(len(item.choices) for item in self.general)


class _Tokens:
    """Tokens generated, counted from the threads of one gate's level."""

    def __init__(self) -> None:
        self.count = 0
        self._lock = threading.Lock()

    def add(self, result: ChatResult) -> None:
        with self._lock:
            self.count += result.output_tokens


def _levels(raw: str) -> tuple[int, ...]:
    try:
        found = tuple(int(part) for part in raw.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("levels are whole numbers, such as 8,16,32") from None
    if not found or any(level < 1 for level in found):
        raise argparse.ArgumentTypeError("levels are whole numbers, such as 8,16,32")
    return found


def hold_usd(gpu: gpus.Gpu, rate: float | None = None) -> float:
    """The bench's worst case: its cap, the wake and Modal's idle window, at the GPU's price."""
    return (rate or gpu.usd_per_hour) * (BENCH_MAX_S + WAKE_S + config.SCALEDOWN_S) / 3600.0


def _kv_line(gpu: gpus.Gpu, kv_gib: float | None) -> str:
    room = gpu.kv_gib if kv_gib is None else kv_gib
    how = "measured" if gpu.kv_measured and kv_gib is None else ("given" if kv_gib is not None else "estimate")
    return (f"KV room {room:.2f} GiB ({how}): about {gpu.fits(gpus.GATE_REQUEST_TOKENS, room)} gate requests "
            f"(~{gpus.GATE_REQUEST_TOKENS} tokens) or {gpu.fits(gpus.PRACTICE_REQUEST_TOKENS, room)} practice "
            f"candidates (~{gpus.PRACTICE_REQUEST_TOKENS}, before prefix sharing) at once")


def _price_note(args: argparse.Namespace, gpu: gpus.Gpu) -> str:
    return "given with --usd-per-hour" if args.usd_per_hour else gpu.price_note


def _fp8(gpu: gpus.Gpu) -> str:
    return "FP8 natively" if gpu.native_fp8 else "FP8 weights, weight-only (Marlin), computed in bf16: not native"


def plan(args: argparse.Namespace, gpu: gpus.Gpu, sliced: Slice | None) -> str:
    rate = args.usd_per_hour or gpu.usd_per_hour
    general = (f"{NIGHT_GENERAL_ITEMS // GENERAL_EVERY} items, every {GENERAL_EVERY}th (a log-probability call "
               "a choice, prefill only)") if sliced is None else f"{len(sliced.general)} items, {sliced.general_calls} calls"
    safety = (f"{args.safety_cases} cases from {args.safety}, a reply each at the client's cap" if args.safety
              else "safety: off (the night's safety gate is off by default; --safety adds a generation-heavy slice)")
    estimate = ", ".join(f"{level}: {min(level, gpu.fits(gpus.GATE_REQUEST_TOKENS)) * gpu.stream_tokens_s:.0f} tokens/s"
                         for level in args.concurrency)
    return "\n".join([
        "plan (nothing is sent; add --go to run it):",
        f"  gpu       {gpu.name}, ${rate:.2f}/h ({_price_note(args, gpu)}), {_fp8(gpu)}",
        f"            {_kv_line(gpu, args.kv_gib)}",
        f"            decode estimate before the bench: {gpu.stream_tokens_s:.0f} tokens/s a stream; {estimate}",
        f"  wake      GET /v1/models until it answers, at most {WAKE_S:.0f} s",
        f"  levels    {', '.join(map(str, args.concurrency))} at once, each its own day in the prompts (no shared "
        f"prefix cache); above {config.CONCURRENT_REQUESTS} needs ANJO_SLEEP_SERVE_MAX_INPUTS set on this host and "
        "on the deploy",
        f"  general   {general}",
        f"  collapse  {NIGHT_COLLAPSE_PROMPTS} prompts, a reply at the cap and an {VERDICT_MAX_TOKENS}-token verdict each",
        f"  {safety}" if not args.safety else f"  safety    {safety}",
        f"  on        {args.adapter or 'the base model'}",
        f"  cap       {BENCH_MAX_S / 60:.0f} min; nothing starts after it",
        f"  hold      ${hold_usd(gpu, rate):.2f} on {POOL} ({BENCH_MAX_S / 60:.0f} min, the wake and the "
        f"{config.SCALEDOWN_S / 60:.0f}-min idle window at ${rate:.2f}/h), settled at what ran plus the idle window",
        "  prints    per level and gate: requests, seconds, tokens/s, dollars; then a night's checkpoint side; "
        "never a prompt's or a reply's words",
        f"  needs     {backend.MODAL_URL_ENV} (or --url) and {backend.MODAL_TOKEN_ENV} (never printed)",
    ])


def _general_level(client: BenchClient, adapter: CheckpointRef | None, items: Sequence[GeneralItem],
                   workers: int) -> int:
    in_order(lambda item: answer(client, adapter, item), items, workers)
    return 0


def _collapse_level(client: BenchClient, adapter: CheckpointRef | None, prompts: Sequence[CollapsePrompt],
                    workers: int) -> int:
    """The collapse gate's two calls a prompt (``CollapseGate._dragged``): the reply, then the verdict."""
    template, tokens = PROMPT_PATH.read_text(encoding="utf-8"), _Tokens()

    def one(index: int) -> None:
        prompt = prompts[index]
        reply = client.complete(adapter, [{"role": "user", "content": prompt.prompt}], temperature=TEMPERATURE,
                                seed=EVAL_SEED + index)
        tokens.add(reply)
        tokens.add(client.complete(None, [{"role": "user", "content": render(template, {
            "FIELDS": "none", "QUESTION": prompt.prompt, "REPLY": reply.text})}],
            temperature=JUDGE_TEMPERATURE, seed=EVAL_SEED, max_tokens=VERDICT_MAX_TOKENS))

    in_order(one, list(range(len(prompts))), workers)
    return tokens.count


def _safety_level(client: BenchClient, adapter: CheckpointRef | None, prompts: Sequence[str], workers: int) -> int:
    """The safety gate's request (``SafetyGate._reply``): one reply at the cap a case."""
    tokens = _Tokens()
    in_order(lambda i: tokens.add(client.complete(adapter, [{"role": "user", "content": prompts[i]}],
                                                  temperature=TEMPERATURE, seed=EVAL_SEED + i)),
             list(range(len(prompts))), workers)
    return tokens.count


def run(
    client: BenchClient, sliced: Slice, levels: Sequence[int], adapter: CheckpointRef | None, rate: float,
    *, clock: Callable[[], float], emit: Callable[[dict[str, Any]], None], deadline: float,
) -> None:
    """Every level, each gate timed alone; one line per gate and one night line per level."""
    gates: list[tuple[str, int, Callable[[BenchClient, int], int]]] = [
        ("general", sliced.general_calls, lambda c, w: _general_level(c, adapter, sliced.general, w)),
        ("collapse", 2 * len(sliced.collapse), lambda c, w: _collapse_level(c, adapter, sliced.collapse, w)),
    ]
    if sliced.safety:
        gates.append(("safety", len(sliced.safety), lambda c, w: _safety_level(c, adapter, sliced.safety, w)))
    scale = {"general": NIGHT_GENERAL_ITEMS / max(1, len(sliced.general)),
             "collapse": NIGHT_COLLAPSE_PROMPTS / max(1, len(sliced.collapse)),
             "safety": NIGHT_SAFETY_CASES / max(1, len(sliced.safety))}
    for index, level in enumerate(levels):
        dated = client.on(EVAL_DAY + timedelta(days=index + 1))
        night: dict[str, float] = {}
        for name, requests, call in gates:
            if clock() >= deadline:
                emit({"line": "stopped", "why": "the bench's cap", "concurrency": level, "gate": name})
                return
            started = clock()
            generated = call(dated, level)
            seconds = max(0.0, clock() - started)
            night[name] = seconds * scale[name]
            emit({"concurrency": level, "gate": name, "requests": requests, "seconds": round(seconds, 2),
                  "generated_tokens": generated,
                  "tokens_per_s": round(generated / seconds, 1) if seconds > 0 and generated else None,
                  "usd": round(seconds * rate / 3600.0, 4)})
        total = math.fsum(night.values())
        emit({"line": "night", "concurrency": level, "seconds_by_gate": {k: round(v, 1) for k, v in night.items()},
              "seconds": round(total, 1), "usd": round(total * rate / 3600.0, 4),
              "note": "one night's checkpoint side, scaled from the slice; the base side is kept across nights"})


def _slice(args: argparse.Namespace, general: Sequence[GeneralItem] | None,
           collapse: Sequence[CollapsePrompt] | None) -> Slice:
    items = tuple(general) if general is not None else load_general()
    prompts = tuple(collapse) if collapse is not None else load_collapse()
    safety: tuple[str, ...] = ()
    if args.safety:
        module, _, name = args.safety.partition(":")
        cases = getattr(importlib.import_module(module), name)()  # loaded by name: the engine never imports the app
        safety = tuple(cases.prompts[: args.safety_cases])
    return Slice(items[::GENERAL_EVERY], prompts, safety)


def _setting(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def main(
    argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, client: Any = None,
    wake: Callable[[], WakeResult] | None = None, ledger: Any = ..., clock: Callable[[], float] = time.monotonic,
    general: Sequence[GeneralItem] | None = None, collapse: Sequence[CollapsePrompt] | None = None,
) -> int:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description="Time a slice of the night's gates at several concurrencies.")
    parser.add_argument("--gpu", choices=tuple(gpus.GPUS), default=config.serve_precision(env).gpu,
                        help="the GPU the app at the URL runs on (its price and KV room)")
    parser.add_argument("--usd-per-hour", type=float, help="the GPU's price an hour, if not the listed one")
    parser.add_argument("--kv-gib", type=float, help="vLLM's KV room as the deploy's log printed it")
    parser.add_argument("--concurrency", type=_levels, default=LEVELS, help="levels, such as 8,16,32")
    parser.add_argument("--safety", help="a safety case loader, <module>:<function> (optional)")
    parser.add_argument("--safety-cases", type=int, help=f"safety cases in the slice ({SAFETY_CASES} by default)")
    parser.add_argument("--adapter", help="run the slice on this adapter, p-<hash>-n<night>")
    parser.add_argument("--url", help=f"the app's URL (default {backend.MODAL_URL_ENV})")
    parser.add_argument("--go", action="store_true", help="send the requests; without it, print the plan only")
    args = parser.parse_args(argv)
    if args.safety_cases is not None and not args.safety:
        parser.error("--safety-cases needs --safety")
    args.safety_cases = args.safety_cases or SAFETY_CASES
    if args.adapter is not None and not config.ADAPTER_NAME.fullmatch(args.adapter):
        parser.error("--adapter is named p-<hash>-n<night>")
    gpu = gpus.GPUS[args.gpu]
    rate = args.usd_per_hour or gpu.usd_per_hour
    if not args.go:
        print(plan(args, gpu, None))
        return 0
    if max(args.concurrency) > config.CONCURRENT_REQUESTS:
        parser.error(f"this host sends at most {config.CONCURRENT_REQUESTS} at once: set "
                     f"{config.MAX_INPUTS_ENV} on it and on the app's deploy")
    sliced = _slice(args, general, collapse)
    if client is None:
        url, token = args.url or _setting(env, backend.MODAL_URL_ENV), _setting(env, backend.MODAL_TOKEN_ENV)
        serving = backend.ModalServing(url, token)
        client, wake = serving.client(), wake or serving.wake
    assert wake is not None
    adapter = None if args.adapter is None else CheckpointRef(
        args.adapter, "bench", EVAL_DAY, f"{config.ADAPTERS_DIR}/{args.adapter}", None, rendering=client.rendering)
    book = default_ledger(env) if ledger is ... else ledger
    print(plan(args, gpu, sliced))
    reservation = None if book is None else hold(book, f"gate_bench:{gpu.name}:{int(time.time())}", (POOL,),
                                                 Bound(rate, BENCH_MAX_S + WAKE_S + config.SCALEDOWN_S))
    started = clock()
    try:
        woke = wake()
        print(json.dumps({"line": "wake", "ok": woke.ok, "seconds": round(woke.seconds, 1)}))
        if woke.ok:
            run(client, sliced, args.concurrency, adapter, rate, clock=clock,
                emit=lambda line: print(json.dumps(line)), deadline=clock() + BENCH_MAX_S)
    finally:
        if reservation is not None:  # what ran, and the idle window Modal bills after the last request
            reservation.settle(rate * (max(0.0, clock() - started) + config.SCALEDOWN_S) / 3600.0)
    return 0 if woke.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
