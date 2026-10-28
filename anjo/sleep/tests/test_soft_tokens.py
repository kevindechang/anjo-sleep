# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Soft tokens (a stretch, off by default): a turn whose whole prompt goes to vLLM as Glimmer's own
post-norm input embeddings, with K vectors spliced into the latest user turn. Off, every request is
today's. A fake vLLM answers and the embedding rows are made up; no network, no torch."""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

import httpx
import numpy as np
import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.serving import config, soft_tokens
from anjo.sleep.serving.client import REQUEST_TIMEOUT_S, ServingError, VllmClient
from anjo.sleep.serving.soft_tokens import SoftTokenError, SoftTokens
from anjo.sleep.tools import soft_tokens_live
from anjo.sleep.train import latent_job

NIGHT = date(2026, 10, 28)
BOS, EOT, START = 200_000, 200_008, 200_022
CHAT = [BOS, 21, 22, EOT, 31, 32, EOT, START, 41]  # a system turn, a user turn, then <|start|>assistant
HEADER = [5, 6]  # " to=user<|message|>"
MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "how was the shoot?"}]


class FakeVllm:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": CHAT if "messages" in body else HEADER})
        if request.url.path == "/v1/completions":
            return httpx.Response(200, json={
                "model": body["model"], "usage": {"prompt_tokens": 11, "completion_tokens": 2},
                "choices": [{"text": "fine", "finish_reason": "stop"}]})
        return httpx.Response(404)


class Rows:
    """Made-up post-norm rows: row ``i`` is ``i`` in every place, scaled to RMS 1 (so the sign tells)."""

    def __init__(self) -> None:
        self.asked: list[list[int]] = []

    def rows(self, ids: Sequence[int]) -> soft_tokens.Matrix:
        self.asked.append(list(ids))
        return np.stack([np.full(soft_tokens.HIDDEN, 1.0 if i % 2 else -1.0, dtype=np.float32) * 1.0
                         for i in ids]).astype(np.float32)


def _client(fake: FakeVllm) -> VllmClient:
    return VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake)).on(NIGHT)


def _vectors(k: int, seed: int = 0, scale: float = 1.0) -> soft_tokens.Matrix:
    raw = np.random.default_rng(seed).standard_normal((k, soft_tokens.HIDDEN)).astype(np.float32)
    return soft_tokens.normalize(raw) * np.float32(scale)


def _soft(k: int, seed: int = 0) -> SoftTokens:
    return SoftTokens(_vectors(k, seed))


def test_soft_tokens_off_sends_todays_request() -> None:
    plain, switched, rows = FakeVllm(), FakeVllm(), Rows()
    _client(plain).complete(None, MESSAGES, temperature=0.0, seed=3)
    _client(switched).with_soft_tokens(rows, encode=lambda matrix: "unused").complete(
        None, MESSAGES, temperature=0.0, seed=3)
    assert switched.requests == plain.requests  # the switch alone changes nothing
    body = plain.requests[-1][1]
    assert body["prompt"] == CHAT + HEADER and "prompt_embeds" not in body
    assert rows.asked == []


def test_a_turn_with_soft_tokens_sends_post_norm_embeddings_with_the_vectors_in_the_latest_user_turn() -> None:
    fake, rows, sent = FakeVllm(), Rows(), []

    def encode(matrix: soft_tokens.Matrix) -> str:
        sent.append(matrix)
        return "BASE64"

    soft = _soft(3)
    result = _client(fake).with_soft_tokens(rows, encode=encode).complete(
        None, MESSAGES, temperature=0.0, seed=3, soft=soft)
    body = fake.requests[-1][1]
    assert body["prompt_embeds"] == "BASE64" and "prompt" not in body
    assert (body["temperature"], body["seed"], body["stop_token_ids"]) == (0.0, 3, list(config.REPLY_STOPS))
    (matrix,) = sent
    prompt = CHAT + HEADER
    cut = 6  # the latest user turn's <|eot|>: the soft vectors go just before it
    assert rows.asked == [prompt] and matrix.shape == (len(prompt) + 3, soft_tokens.HIDDEN)
    np.testing.assert_array_equal(matrix[:cut], rows.rows(prompt[:cut]))
    np.testing.assert_array_equal(matrix[cut:cut + 3], soft.vectors)
    np.testing.assert_array_equal(matrix[cut + 3:], rows.rows(prompt[cut:]))
    assert matrix.dtype == np.float32 and result.text == "fine"

    # K = 0 sends the prompt as embeddings and nothing else: the parity check against token ids.
    _client(fake).with_soft_tokens(rows, encode=encode).complete(None, MESSAGES, temperature=0.0, soft=SoftTokens.none())
    np.testing.assert_array_equal(sent[-1], rows.rows(prompt))


def test_soft_tokens_are_post_norm_vectors_in_a_users_turn_and_need_the_switch() -> None:
    for scale in (10.0, 0.1):
        with pytest.raises(SoftTokenError, match="RMS"):
            SoftTokens(_vectors(2, scale=scale))
    with pytest.raises(SoftTokenError, match="width"):
        SoftTokens(np.ones((2, 8), dtype=np.float32))
    with pytest.raises(SoftTokenError, match="at most"):
        _soft(soft_tokens.MAX_SOFT_TOKENS + 1)
    with pytest.raises(SoftTokenError, match="finite"):
        SoftTokens(np.full((1, soft_tokens.HIDDEN), np.nan, dtype=np.float32))
    fake = FakeVllm()
    with pytest.raises(ServingError, match="with_soft_tokens"):
        _client(fake).complete(None, MESSAGES, temperature=0.0, soft=_soft(1))
    after_reply = [*MESSAGES, {"role": "assistant", "content": "ok"}]
    with pytest.raises(ServingError, match="user"):
        _client(fake).with_soft_tokens(Rows(), encode=lambda m: "x").complete(
            None, after_reply, temperature=0.0, soft=_soft(1))
    assert fake.requests == []  # refused before anything was sent
    with pytest.raises(SoftTokenError, match="user turn"):
        soft_tokens.splice([BOS, 21, START], Rows(), _soft(1))


def test_embedding_rows_are_glimmers_bf16_rows_normalized_and_fetched_once() -> None:
    raw = np.array([[3.0, -4.0] * (soft_tokens.HIDDEN // 2), [0.5, 0.25] * (soft_tokens.HIDDEN // 2)],
                   dtype=np.float32)
    bits = (raw.view(np.uint32) >> 16).astype("<u2")  # exactly representable in bf16
    fetched: list[list[int]] = []

    def fetch(ids: list[int]) -> bytes:
        fetched.append(ids)
        return b"".join(bits[0 if i == 7 else 1].tobytes() for i in ids)

    cached = soft_tokens.CachedRows(fetch)
    first = cached.rows([7, 9, 7])
    again = cached.rows([9, 7])
    assert fetched == [[7, 9]]  # each id once, in first-seen order
    rms = np.sqrt((first ** 2).mean(axis=1))
    np.testing.assert_allclose(rms, 1.0, rtol=1e-4)  # Glimmer's weightless embedding norm, as vLLM applies it
    np.testing.assert_allclose(first[0], raw[0] / np.sqrt(12.5 + soft_tokens.RMS_EPS), rtol=1e-6)
    np.testing.assert_array_equal(again, first[[1, 0]])


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


PRICES = {"ANJO_SLEEP_PRICES": "0.6,2.4,2.5,1.95"}


def test_the_live_test_plans_first_and_holds_before_its_first_request(tmp_path: Any) -> None:
    ledger, lines, fake = Ledger(tmp_path / "budget.jsonl"), list[str](), FakeVllm()
    seen_holds: list[int] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen_holds.append(len(ledger.open_holds()))
        return fake(request)

    def client() -> VllmClient:
        return VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(transport))

    rows = Rows()
    assert soft_tokens_live.main([], out=lines.append, client=client, rows=rows, ledger=ledger,
                                 encode=lambda m: "B64", clock=Clock(), wake=lambda: True, environ=PRICES) == 0
    plan = "\n".join(lines)
    hold = soft_tokens_live.hold_bound(1.95).amount  # the serving GPU's hour from ANJO_SLEEP_PRICES
    assert f"hold ${hold:.2f}" in plan and "modal_usd" in plan and config.PROMPT_EMBEDS_ENV in plan
    assert fake.requests == [] and rows.asked == [] and not ledger.path.exists()  # touched nothing

    assert soft_tokens_live.main(["--yes", "--concurrency", "4"], out=lines.append, client=client, rows=rows,
                                 ledger=ledger, encode=lambda m: "B64", clock=Clock(), wake=lambda: True,
                                 environ=PRICES) == 0
    assert seen_holds and min(seen_holds) == 1  # every request ran under the hold
    assert ledger.open_holds() == () and 0 < sum(ledger.settlements().values()) <= hold
    embeds = [body for path, body in fake.requests if path == "/v1/completions" and "prompt_embeds" in body]
    assert len(embeds) >= 3  # parity, the spliced phrase, random soft tokens, then the load
    assert any("spend" in line for line in lines)


def test_the_live_splice_finds_the_words_the_full_turn_adds_before_its_end() -> None:
    cut = [BOS, 21, 22, EOT, 31, 32, EOT, START, 41]
    full = [BOS, 21, 22, EOT, 31, 32, 33, 34, EOT, START, 41]
    assert soft_tokens_live._inserted(full, cut) == [33, 34]
    assert soft_tokens_live._inserted(cut, cut) is None  # nothing added
    assert soft_tokens_live._inserted([BOS, 20, 22, EOT, 31, 32, 33, EOT, START, 41], cut) is None  # not only added


class Work:
    """Time that passes only while a request is out: each one takes nearly its whole timeout."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_live_test_never_passes_its_ceiling_or_its_hold(tmp_path: Any) -> None:
    ledger, lines, fake, work = Ledger(tmp_path / "budget.jsonl"), list[str](), FakeVllm(), Work()

    def slow(request: httpx.Request) -> httpx.Response:
        work.now += REQUEST_TIMEOUT_S - 10.0
        return fake(request)

    def client() -> VllmClient:
        return VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(slow))

    run = {"out": lines.append, "client": client, "rows": Rows(), "ledger": ledger, "encode": lambda m: "B64",
           "wake": lambda: True}
    assert soft_tokens_live.main(["--yes"], clock=work, environ=PRICES, **run) == 1  # type: ignore[arg-type]
    assert any(line.startswith("stopped before") for line in lines)
    assert work.now <= soft_tokens_live.TEST_S
    assert 0 < sum(ledger.settlements().values()) <= soft_tokens_live.hold_bound(1.95).amount
    assert soft_tokens_live.main(["--yes"], clock=work, environ={}, **run) == 2  # type: ignore[arg-type]
    assert "ANJO_SLEEP_PRICES" in lines[-1] and len(ledger.settlements()) == 1  # no price, nothing held
    with pytest.raises(SystemExit):
        soft_tokens_live.main(["--requests", str(soft_tokens_live.MAX_REQUESTS + 1)], environ=PRICES, **run)  # type: ignore[arg-type]
    assert soft_tokens.EMBED_TIMEOUT_S == latent_job.EMBED_TIMEOUT_S
