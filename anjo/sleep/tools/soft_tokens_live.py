"""The live test of soft tokens on the served Glimmer (``serving.soft_tokens``): plan first, then, with
``--yes``, a hold on ``modal_usd`` before the first request, the steps below, and the spend settled.

    python -m anjo.sleep.tools.soft_tokens_live [--k 8] [--requests 16] [--concurrency 4] [--adapter NAME] [--yes]

Steps, each printed with its latency:

1. **parity**: one turn sent as token ids and as its own embeddings (K = 0), temperature 0. The two
   replies should match: the rows and their norm are what vLLM computes from the ids.
2. **splice**: the same turn without its last word, with that word's own rows as soft tokens. Its
   embeddings are the full turn's, so its reply should match the full turn's.
3. **soft**: K random post-norm soft tokens in the turn: the server takes them.
4. **adapter** (``--adapter``): step 3 on a person's adapter, LoRA with prompt embeddings.
5. **load**: ``--requests`` turns, half with K soft tokens, at ``--concurrency``.

**The ceiling.** The hold is ``TEST_S`` of the serving GPU (its hour is the fourth of
``ANJO_SLEEP_PRICES``) and the embedding call, then the idle window. Each request is sent once
(``VllmClient.once``, an adapter's load waited on once), and a step or a load turn starts only while
one more turn's worst case (its requests' timeouts, and an embedding call's cold start and timeout)
still fits in ``TEST_S``,
so the wall time, and the spend settled from it, stay within the hold. ``--requests`` is at most
``MAX_REQUESTS``.

It needs the serving app redeployed with ``ANJO_SLEEP_PROMPT_EMBEDS=1``, the training app redeployed
(``embed_rows``), ``ANJO_SLEEP_MODAL_URL``, ``ANJO_SLEEP_MODAL_TOKEN`` and ``ANJO_SLEEP_PRICES`` set,
and torch on this host.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Final

import numpy as np

from anjo.sleep.budget import Bound, Ledger, default_ledger
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import backend, config, soft_tokens
from anjo.sleep.serving.client import REQUEST_TIMEOUT_S, ServingError, VllmClient
from anjo.sleep.serving.soft_tokens import EmbeddingRows, Matrix, SoftTokens
from anjo.sleep.stops import BudgetStop
from anjo.sleep.types import CheckpointRef

POOL: Final = "modal_usd"
PRICES_ENV: Final = "ANJO_SLEEP_PRICES"  # app.night.PRICES_ENV: "spark_in,spark_out,train_gpu_hour,serve_gpu_hour"
EMBED_USD_PER_HOUR: Final = soft_tokens.EMBED_USD_PER_HOUR
TEST_S: Final = 3600.0  # the test's own hard ceiling
POSTS_PER_TURN: Final = 4  # two /tokenize calls, an adapter's load at most once, the completion
EMBED_CALL_MAX_S: Final = soft_tokens.EMBED_COLD_START_S + soft_tokens.EMBED_TIMEOUT_S  # a cold start, then its run
STEP_MAX_S: Final = POSTS_PER_TURN * REQUEST_TIMEOUT_S + EMBED_CALL_MAX_S  # one turn, after one embedding call
MAX_REQUESTS: Final = 64
IDLE_S: Final = float(config.SCALEDOWN_S)  # the server stays up this long after the last request
MAX_TOKENS: Final = 24
SYSTEM: Final = "Answer in one short sentence."
FULL: Final = "We wrapped the night shoot at the harbour. Say back the last word of this message: lighthouse"
CUT: Final = "We wrapped the night shoot at the harbour. Say back the last word of this message:"


def serve_rate(environ: Mapping[str, str]) -> float:
    """The serving GPU's list price an hour: the fourth of ``ANJO_SLEEP_PRICES``."""
    parts = environ.get(PRICES_ENV, "").split(",")
    try:
        rate = float(parts[3]) if len(parts) == 4 else -1.0
    except ValueError:
        rate = -1.0
    if not rate > 0:
        raise ValueError(f"{PRICES_ENV} holds four prices, the serving GPU's hour last")
    return rate


def hold_bound(serve_usd_per_hour: float) -> Bound:
    """The worst case: the serving GPU and the embedding call for the whole test, then the idle window."""
    return Bound(serve_usd_per_hour + EMBED_USD_PER_HOUR, TEST_S + IDLE_S)


@dataclass(frozen=True)
class _Deadline:
    clock: Callable[[], float]
    ends: float

    def room(self) -> bool:
        """Whether one more step's worst case still ends within the test's ceiling."""
        return self.clock() + STEP_MAX_S <= self.ends


class _OutOfTime(Exception):
    """The next step might pass the ceiling: the steps stop here."""


def _turn(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


class _Recording:
    """The rows, remembering the last prompt asked for (to find a word's ids without a tokenizer here)."""

    def __init__(self, rows: EmbeddingRows) -> None:
        self.rows_of, self.last = rows, list[int]()

    def rows(self, ids: Sequence[int]) -> Matrix:
        self.last = list(ids)
        return self.rows_of.rows(ids)


def _inserted(full: Sequence[int], cut: Sequence[int]) -> list[int] | None:
    """The ids ``full`` has in place of the end of ``cut``'s latest user turn, if that is all that differs."""
    if soft_tokens.EOT_ID not in cut:
        return None
    at = len(cut) - 1 - list(cut)[::-1].index(soft_tokens.EOT_ID)
    extra = len(full) - len(cut)
    ok = 0 < extra <= soft_tokens.MAX_SOFT_TOKENS and list(full[:at]) == list(cut[:at])
    return list(full[at:at + extra]) if ok and list(full[at + extra:]) == list(cut[at:]) else None


def _plan(args: argparse.Namespace, bound: Bound, out: Callable[[str], None]) -> None:
    out("soft tokens live test: plan only, nothing sent, nothing held")
    out(f"needs: the serving app redeployed with {config.PROMPT_EMBEDS_ENV}=1 (vLLM --enable-prompt-embeds); "
        f"the training app redeployed (embed_rows); {backend.MODAL_URL_ENV} and {backend.MODAL_TOKEN_ENV} set; "
        "torch on this host")
    out(f"steps: wake; parity; splice; {args.k} random soft tokens"
        f"{'; the same on adapter ' + args.adapter if args.adapter else ''}; "
        f"{args.requests} turns at concurrency {args.concurrency}, half with soft tokens")
    out(f"hold ${bound.amount:.2f} on {POOL} (${bound.rate_per_hour:.2f}/h for {TEST_S / 60:.0f} min of test and "
        f"{IDLE_S / 60:.0f} min idle); no step starts that could pass {TEST_S / 60:.0f} min; settled at the wall "
        "time plus the idle window")
    out("run again with --yes to send")


def _steps(args: argparse.Namespace, live: VllmClient, rows: _Recording, out: Callable[[str], None],
           deadline: _Deadline) -> None:
    clock = deadline.clock

    def ask(text: str, soft: SoftTokens | None = None, adapter: CheckpointRef | None = None) -> str:
        return live.complete(adapter, _turn(text), temperature=0.0, seed=0, max_tokens=MAX_TOKENS, soft=soft).text

    def step(name: str, call: Callable[[], str]) -> str:
        if not deadline.room():
            raise _OutOfTime(name)
        started = clock()
        text = call()
        out(f"{name}: {clock() - started:.2f} s, {text!r}")
        return text

    by_ids = step("parity, token ids", lambda: ask(FULL))
    by_rows = step("parity, embeddings (K=0)", lambda: ask(FULL, SoftTokens.none()))
    full_ids = rows.last
    out(f"parity: {'match' if by_ids == by_rows else 'DIFFER'}")
    step("splice, cut turn (K=0)", lambda: ask(CUT, SoftTokens.none()))
    word = _inserted(full_ids, rows.last)
    if word is None:
        out("splice: the last word isn't a clean insertion at the turn's end; skipped")
    else:
        spliced = step(f"splice, its {len(word)} rows as soft tokens",
                         lambda: ask(CUT, SoftTokens(rows.rows(word))))
        out(f"splice: {'match' if spliced == by_rows else 'DIFFER'}")
    random = SoftTokens(soft_tokens.normalize(np.random.default_rng(0).standard_normal((args.k, soft_tokens.HIDDEN))))
    step(f"soft, K={args.k}", lambda: ask(FULL, random))
    if args.adapter:
        ref = CheckpointRef(args.adapter, "live-test", date.today(), "", None, live.rendering)
        step(f"adapter, K={args.k}", lambda: ask(FULL, random, ref))
    _load(args, ask, random, out, deadline)


def _load(args: argparse.Namespace, ask: Callable[[str, SoftTokens | None], str], random: SoftTokens,
          out: Callable[[str], None], deadline: _Deadline) -> None:
    clock = deadline.clock

    def one(i: int) -> tuple[bool, float, str]:
        soft = random if i % 2 else None
        if not deadline.room():
            return soft is not None, 0.0, "skipped"
        started = clock()
        try:
            ask(FULL, soft)
            return soft is not None, clock() - started, "answered"
        except ServingError:  # counted and reported below
            return soft is not None, clock() - started, "failed"

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.requests)))
    for soft in (False, True):
        times = sorted(seconds for kind, seconds, how in results if kind == soft and how == "answered")
        counts = {how: sum(1 for kind, _, said in results if kind == soft and said == how) for how in ("failed", "skipped")}
        p50 = statistics.median(times) if times else float("nan")
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))] if times else float("nan")
        out(f"load, {'soft' if soft else 'plain'}: {len(times)} answered, {counts['failed']} failed, "
            f"{counts['skipped']} skipped at the ceiling, p50 {p50:.2f} s, p95 {p95:.2f} s")


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="soft_tokens_live", description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--k", type=int, default=8, choices=range(1, soft_tokens.MAX_SOFT_TOKENS + 1), metavar="K")
    parser.add_argument("--requests", type=int, default=16, choices=range(2, MAX_REQUESTS + 1), metavar="N")
    parser.add_argument("--concurrency", type=int, default=4, choices=range(1, config.CONCURRENT_REQUESTS + 1),
                        metavar="C")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--yes", action="store_true", help="hold, then send")
    args = parser.parse_args(list(argv))
    if args.adapter:
        config.check_adapter_name(args.adapter)
    return args


def main(
    argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print,
    client: Callable[[], VllmClient] | None = None, rows: EmbeddingRows | None = None, ledger: Ledger | None = None,
    encode: Callable[[Matrix], str] = soft_tokens.encode_embeds, clock: Callable[[], float] = time.monotonic,
    wake: Callable[[], bool] | None = None, environ: Mapping[str, str] | None = None,
) -> int:
    args = _parse([] if argv is None else argv)
    try:
        bound = hold_bound(serve_rate(os.environ if environ is None else environ))
    except ValueError as error:
        out(f"not run: {error}")
        return 2
    if not args.yes:
        _plan(args, bound, out)
        return 0
    if client is None or wake is None:
        url, token = os.environ.get(backend.MODAL_URL_ENV, ""), os.environ.get(backend.MODAL_TOKEN_ENV, "")
        if not url or not token:
            out(f"not run: {backend.MODAL_URL_ENV} and {backend.MODAL_TOKEN_ENV} are set to the serving app")
            return 2
        serving = backend.ModalServing(url, token)
        client = client or serving.client
        wake = wake or (lambda: serving.wake().ok)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    try:
        reservation = hold(ledger or default_ledger(), f"soft_tokens_live:{stamp}", (POOL,), bound)
    except BudgetStop as stop:
        out(f"not run: {stop}")
        return 2
    started = clock()
    try:
        if not wake():
            out("not run: the server didn't wake")
            return 1
        recording = _Recording(rows or soft_tokens.ModalEmbeddingRows())
        _steps(args, client().once().with_load_wait(0).with_soft_tokens(recording, encode=encode), recording, out,
               _Deadline(clock, started + TEST_S))
        return 0
    except _OutOfTime as stopped:
        out(f"stopped before {stopped}: one more step might pass the {TEST_S / 60:.0f}-minute ceiling")
        return 1
    finally:
        elapsed = clock() - started
        usd = (elapsed + IDLE_S) * bound.rate_per_hour / 3600.0  # within the hold: no step starts that could pass it
        reservation.settle(usd)
        out(f"spend: ${usd:.2f} settled on {POOL} for {elapsed:.0f} s and the idle window (hold ${bound.amount:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
