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

It needs the serving app redeployed with ``ANJO_SLEEP_PROMPT_EMBEDS=1``, the training app redeployed
(``embed_rows``), ``ANJO_SLEEP_MODAL_URL`` and ``ANJO_SLEEP_MODAL_TOKEN`` set, and torch on this host.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Final

import numpy as np

from anjo.sleep.budget import Bound, Ledger, default_ledger
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import backend, config, soft_tokens
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.serving.soft_tokens import EmbeddingRows, Matrix, SoftTokens
from anjo.sleep.stops import BudgetStop
from anjo.sleep.types import CheckpointRef

POOL: Final = "modal_usd"
SERVE_USD_PER_HOUR: Final = 2.61  # one L40S with its CPU and memory, rounded up to the serving list price
EMBED_USD_PER_HOUR: Final = 0.10  # the CPU container that sends the embedding rows
TEST_S: Final = 1800.0  # the test's own hard ceiling
IDLE_S: Final = float(config.SCALEDOWN_S)  # the server stays up this long after the last request
MAX_TOKENS: Final = 24
SYSTEM: Final = "Answer in one short sentence."
FULL: Final = "We wrapped the night shoot at the harbour. Say back the last word of this message: lighthouse"
CUT: Final = "We wrapped the night shoot at the harbour. Say back the last word of this message:"


def hold_bound() -> Bound:
    """The worst case: the serving GPU and the embedding call for the whole test, then the idle window."""
    return Bound(SERVE_USD_PER_HOUR + EMBED_USD_PER_HOUR, TEST_S + IDLE_S)


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


def _plan(args: argparse.Namespace, out: Callable[[str], None]) -> None:
    bound = hold_bound()
    out("soft tokens live test: plan only, nothing sent, nothing held")
    out(f"needs: the serving app redeployed with {config.PROMPT_EMBEDS_ENV}=1 (vLLM --enable-prompt-embeds); "
        f"the training app redeployed (embed_rows); {backend.MODAL_URL_ENV} and {backend.MODAL_TOKEN_ENV} set; "
        "torch on this host")
    out(f"steps: wake; parity; splice; {args.k} random soft tokens"
        f"{'; the same on adapter ' + args.adapter if args.adapter else ''}; "
        f"{args.requests} turns at concurrency {args.concurrency}, half with soft tokens")
    out(f"hold ${bound.amount:.2f} on {POOL} (${bound.rate_per_hour:.2f}/h for {TEST_S / 60:.0f} min of test and "
        f"{IDLE_S / 60:.0f} min idle); settled at the wall time plus the idle window")
    out("run again with --yes to send")


def _timed(out: Callable[[str], None], clock: Callable[[], float], name: str, call: Callable[[], str]) -> str:
    started = clock()
    text = call()
    out(f"{name}: {clock() - started:.2f} s, {text!r}")
    return text


def _steps(args: argparse.Namespace, live: VllmClient, rows: _Recording, out: Callable[[str], None],
           clock: Callable[[], float]) -> None:
    def ask(text: str, soft: SoftTokens | None = None, adapter: CheckpointRef | None = None) -> str:
        return live.complete(adapter, _turn(text), temperature=0.0, seed=0, max_tokens=MAX_TOKENS, soft=soft).text

    by_ids = _timed(out, clock, "parity, token ids", lambda: ask(FULL))
    by_rows = _timed(out, clock, "parity, embeddings (K=0)", lambda: ask(FULL, SoftTokens.none()))
    full_ids = rows.last
    out(f"parity: {'match' if by_ids == by_rows else 'DIFFER'}")
    _timed(out, clock, "splice, cut turn (K=0)", lambda: ask(CUT, SoftTokens.none()))
    word = _inserted(full_ids, rows.last)
    if word is None:
        out("splice: the last word isn't a clean insertion at the turn's end; skipped")
    else:
        spliced = _timed(out, clock, f"splice, its {len(word)} rows as soft tokens",
                         lambda: ask(CUT, SoftTokens(rows.rows(word))))
        out(f"splice: {'match' if spliced == by_rows else 'DIFFER'}")
    random = SoftTokens(soft_tokens.normalize(np.random.default_rng(0).standard_normal((args.k, soft_tokens.HIDDEN))))
    _timed(out, clock, f"soft, K={args.k}", lambda: ask(FULL, random))
    if args.adapter:
        ref = CheckpointRef(args.adapter, "live-test", date.today(), "", None, live.rendering)
        _timed(out, clock, f"adapter, K={args.k}", lambda: ask(FULL, random, ref))
    _load(args, ask, random, out, clock)


def _load(args: argparse.Namespace, ask: Callable[[str, SoftTokens | None], str], random: SoftTokens,
          out: Callable[[str], None], clock: Callable[[], float]) -> None:
    def one(i: int) -> tuple[bool, float, bool]:
        started, soft = clock(), (random if i % 2 else None)
        try:
            ask(FULL, soft)
            return soft is not None, clock() - started, True
        except ServingError:  # counted and reported below
            return soft is not None, clock() - started, False

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.requests)))
    for soft in (False, True):
        times = sorted(seconds for kind, seconds, ok in results if kind == soft and ok)
        failed = sum(1 for kind, _, ok in results if kind == soft and not ok)
        p50 = statistics.median(times) if times else float("nan")
        p95 = times[min(len(times) - 1, int(0.95 * len(times)))] if times else float("nan")
        out(f"load, {'soft' if soft else 'plain'}: {len(times)} answered, {failed} failed, "
            f"p50 {p50:.2f} s, p95 {p95:.2f} s")


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="soft_tokens_live", description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--k", type=int, default=8, choices=range(1, soft_tokens.MAX_SOFT_TOKENS + 1), metavar="K")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4, choices=range(1, config.CONCURRENT_REQUESTS + 1),
                        metavar="C")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--yes", action="store_true", help="hold, then send")
    args = parser.parse_args(list(argv))
    if args.adapter:
        config.check_adapter_name(args.adapter)
    if args.requests < 2:
        parser.error("--requests is at least 2")
    return args


def main(
    argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print,
    client: Callable[[], VllmClient] | None = None, rows: EmbeddingRows | None = None, ledger: Ledger | None = None,
    encode: Callable[[Matrix], str] = soft_tokens.encode_embeds, clock: Callable[[], float] = time.monotonic,
    wake: Callable[[], bool] | None = None,
) -> int:
    args = _parse([] if argv is None else argv)
    if not args.yes:
        _plan(args, out)
        return 0
    if client is None or wake is None:
        url, token = os.environ.get(backend.MODAL_URL_ENV, ""), os.environ.get(backend.MODAL_TOKEN_ENV, "")
        if not url or not token:
            out(f"not run: {backend.MODAL_URL_ENV} and {backend.MODAL_TOKEN_ENV} are set to the serving app")
            return 2
        serving = backend.ModalServing(url, token)
        client = client or serving.client
        wake = wake or (lambda: serving.wake().ok)
    bound = hold_bound()
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
        _steps(args, client().with_soft_tokens(recording, encode=encode), recording, out, clock)
        return 0
    finally:
        elapsed = clock() - started
        usd = min(bound.amount, (elapsed + IDLE_S) * bound.rate_per_hour / 3600.0)
        reservation.settle(usd)
        out(f"spend: ${usd:.2f} settled on {POOL} for {elapsed:.0f} s and the idle window (hold ${bound.amount:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())
