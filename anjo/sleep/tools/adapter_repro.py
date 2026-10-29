"""Plan-first LoRA reproducibility diagnostic (no traffic without --go).

Run with --x p-<hash>-n<N> --y p-<hash>-n<N>, optionally --go. Uses the frozen
knowledge set's first 30 choices, the scorer's fixed date and token memo, serially.
A repeats identical payloads; B uses a fresh cache_salt per pass; C scores base,
X, Y, X; D explicitly unloads/reloads X and scores again. Never prints prompts or
continuations. Flips count changed winning choices, only on fully sampled items.

Gauges bracket each POST, including initial tokenizations, loads and D's unload.
Metrics GETs are not recursively sampled. They are boundary samples, possibly stale: zero does not prove isolation
during prefill. H2 is a correlation; two adapters cannot exhaust 8/16 LoRA slots,
so an inconclusive H3 check does not rule out eviction under many other loads.
The 15-minute active cap is checked between calls; a request already sent may
finish later. Wake and the server's billed idle window are additional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final, Protocol

from anjo.sleep.budget import Bound, default_ledger
from anjo.sleep.gates import EVAL_DAY
from anjo.sleep.night.holds import hold
from anjo.sleep.night.sweep import NightBusy, night_lock
from anjo.sleep.serving import backend, config, gpus
from anjo.sleep.serving.client import ServingError, WakeResult
from anjo.sleep.sets import knowledge
from anjo.sleep.sets.loader import SetError
from anjo.sleep.tools.knowledge_score import LOAD_S, LOCKED, POOL, PREFILL_S, ROUND_TRIP_S, WAKE_S
from anjo.sleep.types import CheckpointRef

MAX_CHOICES: Final = 30
PASSES: Final = 11
MAX_S: Final = 15 * 60.0
NOISE_NATS: Final = 0.01
_GAUGE: Final = re.compile(r'^(vllm:(?:num_requests_running|num_requests_waiting|lora_requests_info))'
                           r'(\{.*\})?\s+(\S+)(?:\s+\S+)?$')


class Client(Protocol):
    @property
    def rendering(self) -> str: ...
    def on(self, day: date) -> Client: ...
    def reusing_tokens(self) -> Client: ...
    def once(self) -> Client: ...
    def observing_requests(self, observer: Callable[[str], AbstractContextManager[None]]) -> Client: ...
    def metrics(self) -> str: ...
    def reload_adapter(self, name: str) -> None: ...
    def continuation_logprobs(self, adapter: CheckpointRef | None, messages: list[dict[str, str]],
                              continuation: str, *, cache_salt: str | None = None) -> Sequence[float]: ...


@dataclass(frozen=True)
class Choice:
    item: str
    index: int
    total: int
    prompt: str
    continuation: str


@dataclass(frozen=True)
class Pass:
    label: str
    adapter: str
    scores: tuple[float, ...]
    in_flight: list[dict[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)


class TimeLimit(RuntimeError):
    """Stop before starting another request after the active cap."""


def choices(the_set: knowledge.KnowledgeSet) -> tuple[Choice, ...]:
    return tuple(Choice(item.id, n, len(item.choices), item.prompt, text)
                 for item in the_set.items for n, text in enumerate(item.choices))[:MAX_CHOICES]


def estimate_s(selected: Sequence[Choice]) -> float:
    texts = 1 + len({c.prompt for c in selected}) + len({c.continuation for c in selected})
    # Every POST, including tokenizations and 3 loads + 1 unload, gets 2 metrics GETs.
    return float((texts + PASSES * len(selected) + 4) * 3 * ROUND_TRIP_S
                 + PASSES * len(selected) * PREFILL_S + 3 * LOAD_S)


def plan(selected: Sequence[Choice], rate: float, gpu: str) -> str:
    seconds = estimate_s(selected)
    expected = rate * (seconds + config.SCALEDOWN_S) / 3600
    reserved = rate * (MAX_S + WAKE_S + config.SCALEDOWN_S) / 3600
    texts = 1 + len({c.prompt for c in selected}) + len({c.continuation for c in selected})
    return "\n".join([
        "plan (nothing is sent; add --go to run it):",
        "  A: X x3; B: X x3, fresh cache_salt each pass; C: base, X, Y, X; D: unload/load X, X",
        f"  {PASSES} passes x {len(selected)} choices = {PASSES * len(selected)} prefills, "
        f"{2 * PASSES * len(selected)} scoring metrics reads, "
        f"plus {2 * (texts + 4)} around tokenizations/load/unload; one request at a time, under the local night lock",
        f"  expected ~{seconds / 60:.1f} active GPU min, plus {config.SCALEDOWN_S / 60:.0f} min idle; "
        f"~${expected:.2f} warm at ${rate:.2f}/h on {gpu}; cold wake adds time/cost",
        f"  measured round trip {ROUND_TRIP_S}s; prefill {PREFILL_S}s; adapter load estimated {LOAD_S}s",
        f"  active cap {MAX_S / 60:.0f} min; hold ${reserved:.2f} on {POOL}, including wake and idle; "
        "settle elapsed plus idle even on failure",
        "  output: hashes, numeric differences, answer flips and per-choice boundary gauges; no prompt text",
        "  gauges can miss short traffic between samples; H2/H3 are diagnostic evidence, not proof",
        f"  needs {backend.MODAL_URL_ENV} (or --url) and {backend.MODAL_TOKEN_ENV}",
    ])


def parse_metrics(raw: str) -> dict[str, Any]:
    """Sum engine/model gauge series; refuse absent/invalid occupancy rather than call it idle."""
    found: dict[str, list[dict[str, Any]]] = {}
    for line in raw.splitlines():
        match = _GAUGE.fullmatch(line.strip())
        if match is None:
            continue
        name, labels, number = match.groups()
        try:
            value = float(number)
        except ValueError:
            raise ServingError("/metrics contains an invalid gauge") from None
        if not math.isfinite(value) or value < 0:
            raise ServingError("/metrics contains an invalid gauge")
        found.setdefault(name, []).append({"labels": labels or "", "value": value})
    needed = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    if any(name not in found for name in needed):
        raise ServingError("/metrics is missing running/waiting gauges")
    return {"running": math.fsum(s["value"] for s in found[needed[0]]),
            "waiting": math.fsum(s["value"] for s in found[needed[1]]),
            "lora": found.get("vllm:lora_requests_info", [])}


def _busy(reading: Mapping[str, Any]) -> bool:
    return any(reading[edge][key] > 0 for edge in ("before", "after") for key in ("running", "waiting"))


def _diff(a: Pass, b: Pass) -> float:
    return max((abs(x - y) for x, y in zip(a.scores, b.scores, strict=True)), default=0.0)


def verdict(passes: Sequence[Pass]) -> str:
    """Hypotheses supported by completed X passes; base/Y differences are not drift."""
    xs = [p for p in passes if p.adapter == "X"]
    if len(xs) < 2:
        return "inconclusive"
    first = xs[0]
    if all(_diff(first, p) < NOISE_NATS for p in xs[1:]):
        return "noise"
    evidence = []
    a, b = ([p for p in xs if p.label.startswith(group)] for group in ("A", "B"))
    if len(a) == len(b) == 3 and any(_diff(a[0], p) >= NOISE_NATS for p in a[1:]) and all(
            _diff(b[0], p) < NOISE_NATS for p in b[1:]):
        evidence.append("H1")
    if any(abs(x - y) >= NOISE_NATS and (_busy(before) or _busy(after))
           for p in xs[1:] for x, y, before, after in zip(first.scores, p.scores,
                                                        first.in_flight, p.in_flight, strict=True)):
        evidence.append("H2")
    by_label = {p.label: p for p in xs}
    if all(label in by_label for label in ("C.X1", "C.X2", "D.X")):
        early = [p for p in xs if p.label.startswith(("A", "B")) or p.label == "C.X1"]
        if (all(_diff(first, p) < NOISE_NATS for p in early)
                and _diff(first, by_label["C.X2"]) >= NOISE_NATS
                and _diff(first, by_label["D.X"]) < NOISE_NATS):
            evidence.append("H3")
    return "+".join(evidence) if evidence else "inconclusive"


def pass_line(got: Pass, first: Pass, selected: Sequence[Choice]) -> dict[str, Any]:
    per_item: dict[str, list[float]] = {}
    original: dict[str, list[float]] = {}
    totals = {}
    for choice, score, baseline in zip(selected, got.scores, first.scores, strict=True):
        per_item.setdefault(choice.item, []).append(score)
        original.setdefault(choice.item, []).append(baseline)
        totals[choice.item] = choice.total
    flips = sum(max(range(len(values)), key=values.__getitem__) !=
                max(range(len(values)), key=original[key].__getitem__)
                for key, values in per_item.items() if len(values) == totals[key])
    digest = hashlib.sha256(json.dumps(per_item, sort_keys=True, allow_nan=False,
                                      separators=(",", ":")).encode()).hexdigest()
    return {"label": got.label, "adapter": got.adapter, "logprobs_sha256": digest,
            "max_logprob_diff": _diff(first, got), "flips": flips,
            "in_flight": got.requests, "other_in_flight": any(map(_busy, got.requests))}


def _check_time(clock: Callable[[], float], deadline: float) -> None:
    if clock() >= deadline:
        raise TimeLimit


def run(client: Client, x: str, y: str, selected: Sequence[Choice], *, deadline: float,
        clock: Callable[[], float], out: Callable[[str], None]) -> list[Pass]:
    passes: list[Pass] = []
    unsupported = False
    requests: list[dict[str, Any]] = []

    @contextmanager
    def observe(path: str) -> Iterator[None]:
        _check_time(clock, deadline)
        before = parse_metrics(client.metrics())
        _check_time(clock, deadline)
        try:
            yield
        finally:
            after = parse_metrics(client.metrics())
            requests.append({"path": path, "before": before, "after": after})

    client = client.observing_requests(observe)
    order = [(f"A{n}", "X") for n in range(1, 4)] + [(f"B{n}", "X") for n in range(1, 4)] + [
        ("C.base", "base"), ("C.X1", "X"), ("C.Y", "Y"), ("C.X2", "X"), ("D.X", "X")]
    for label, side in order:
        if label.startswith("B") and unsupported:
            continue
        _check_time(clock, deadline)
        requests = []
        if label == "D.X":
            client.reload_adapter(x)
        salt = uuid.uuid4().hex if label.startswith("B") else None
        name = x if side == "X" else y
        ref = None if side == "base" else CheckpointRef(
            name, "adapter_repro", EVAL_DAY, f"{config.ADAPTERS_DIR}/{name}", None, rendering=client.rendering)
        scores, readings = [], []
        try:
            for choice in selected:
                _check_time(clock, deadline)
                values = client.continuation_logprobs(ref, [{"role": "user", "content": choice.prompt}],
                                                     choice.continuation, cache_salt=salt)
                if not values or any(not math.isfinite(v) for v in values):
                    raise ServingError("scoring returned empty/non-finite log-probabilities")
                scores.append(math.fsum(values))
                readings.append(requests[-1])  # this choice's completion, excluding load/tokenization traffic
        except ServingError as error:
            if salt is None or str(error) != "/v1/completions returned HTTP 400":
                raise
            out(json.dumps({"cache_salt": "unsupported"}))
            out(json.dumps({"line": "rejected_requests", "in_flight": requests}))
            unsupported = True
            continue
        got = Pass(label, side, tuple(scores), readings, requests)
        passes.append(got)
        out(json.dumps(pass_line(got, passes[0], selected)))
    return passes


def _setting(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         client: Client | None = None, wake: Callable[[], WakeResult] | None = None, ledger: Any = ...,
         clock: Callable[[], float] = time.monotonic, out: Callable[[str], None] = print) -> int:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", required=True, help="adapter X: p-<hash>-n<night>")
    parser.add_argument("--y", required=True, help="different adapter Y: p-<hash>-n<night>")
    parser.add_argument("--persona", default="film_founder")
    parser.add_argument("--gpu", choices=tuple(gpus.GPUS), default=config.serve_precision(env).gpu)
    parser.add_argument("--usd-per-hour", type=float)
    parser.add_argument("--url")
    parser.add_argument("--go", action="store_true", help="send requests; otherwise only print the plan")
    args = parser.parse_args(argv)
    if not all(config.ADAPTER_NAME.fullmatch(n) for n in (args.x, args.y)) or args.x == args.y:
        parser.error("--x and --y must name two different p-<hash>-n<night> adapters")
    try:
        selected = choices(knowledge.load(args.persona))
    except SetError as error:
        parser.error(str(error))
    gpu = gpus.GPUS[args.gpu]
    rate = gpu.usd_per_hour if args.usd_per_hour is None else args.usd_per_hour
    if not math.isfinite(rate) or rate <= 0:
        parser.error("--usd-per-hour must be finite and positive")
    out(plan(selected, rate, gpu.name))
    if estimate_s(selected) > 0.9 * MAX_S:
        out(json.dumps({"line": "refused", "why": "estimate exceeds 90% of active cap"}))
        return 2
    if not args.go:
        return 0
    with ExitStack() as stack:
        if client is None:
            serving = backend.ModalServing(args.url or _setting(env, backend.MODAL_URL_ENV),
                                           _setting(env, backend.MODAL_TOKEN_ENV))
            owned = serving.client()
            stack.callback(owned.close)
            client = owned
            wake = wake or serving.wake
        if wake is None:
            raise ValueError("an injected client needs a wake function")
        book = default_ledger(env) if ledger is ... else ledger
        try:
            stack.enter_context(nullcontext() if book is None else night_lock(book))
        except NightBusy:
            out(json.dumps({"line": "refused", "why": LOCKED}))
            return 2
        reservation = None if book is None else hold(
            book, f"adapter_repro:{uuid.uuid4().hex}", (POOL,), Bound(rate, MAX_S + WAKE_S + config.SCALEDOWN_S))
        started = clock()
        try:
            woke = wake()
            out(json.dumps({"line": "wake", "ok": woke.ok, "seconds": woke.seconds}))
            if not woke.ok:
                return 1
            passes = run(client.on(EVAL_DAY).reusing_tokens().once(), args.x, args.y, selected,
                         deadline=clock() + MAX_S, clock=clock, out=out)
            out(json.dumps({"verdict": verdict(passes), "boundary_samples_only": True}))
            return 0
        except TimeLimit:
            out(json.dumps({"verdict": "incomplete", "why": "active cap reached"}))
            return 2
        finally:
            if reservation is not None:
                reservation.settle(rate * (max(0.0, clock() - started) + config.SCALEDOWN_S) / 3600)


if __name__ == "__main__":
    raise SystemExit(main())
