# mypy: disable-error-code="untyped-decorator"
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Rate limits in generation (MIND's data): a 429 waits (its hint, or a longer backoff, at most
``retry.MAX_WAIT_S``) and tries again from the attempt's extra tries, unbilled; a call that still
fails ends the attempt, not the life, and the next attempt starts after a cool-down; and one
limiter per run caps the Gemini reads in flight. All offline."""
from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import calls, core_splits, fakes, gen_batch, retry
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.limiter import Limiter
from anjo.sleep.mind.apd.live import RunsFailed
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult, ModelApiError, RateLimited

SPEC = next(spec for spec in core_splits.core_specs("core-train") if spec.prevalence > 0)
VERTEX_429 = "Gemini on Vertex returned HTTP 429"
LIVE = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]


class LimitedFirst:
    """A chat whose first ``times`` calls of ``kind`` answer 429 (with ``hint``), then ``inner``'s."""

    def __init__(self, inner: Any, times: int, *, kind: str = "leak", hint: float | None = None,
                 error: ModelApiError | None = None) -> None:
        self.inner, self.times, self.kind, self.hint, self.error = inner, times, kind, hint, error
        self.refused = 0

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if calls.kind_of(messages) == self.kind and self.refused < self.times:
            self.refused += 1
            raise self.error or RateLimited(VERTEX_429, retry_after_s=self.hint)
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def _backend(tmp_path: Path, slept: list[float], **chats: Any) -> backends.Backend:
    return dataclasses.replace(backends.fake_backend(FIREWORKS), sleep=slept.append,
                               ledger=RunLedger(tmp_path / "budget.jsonl"), prices=cli.FAKE_PRICES,
                               rates=cli.FAKE_RATES, **chats)


def _settled(path: Path) -> float:
    return sum(float(e["amount"]) for e in read_records(path, PLAIN) if e["op"] == "settle")


def _jittered(wait: float, base: float) -> bool:
    low, high = retry.JITTER
    return base * low <= wait <= min(base * high, retry.MAX_WAIT_S)


# ── A 429 waits and tries again ───────────────────────────────────────────────


def test_a_429_then_a_reply_succeeds_within_one_attempt_unbilled(tmp_path: Path) -> None:
    # Arrange: the same life clean, then with its first leak read answered 429 once
    clean = gen_batch.generate_life(SPEC, _backend(tmp_path / "clean", []))
    slept: list[float] = []
    log = calls.CallLog()

    # Act
    made = gen_batch.generate_life(
        SPEC, _backend(tmp_path / "limited", slept, reader=LimitedFirst(fakes.FakeGateReader(), 1)), log)

    # Assert: one attempt, one wait of the 429's backoff, and the 429 logged at no cost
    assert made.attempts == 1 and made.life == clean.life
    assert len(slept) == 1 and _jittered(slept[0], retry.RATE_BACKOFF_S)
    (limited,) = [c for c in log.calls if c.failed]
    assert (limited.kind, limited.rate_limited, limited.usd, limited.wait_s) == ("leak", True, 0.0, slept[0])
    assert _settled(tmp_path / "limited" / "budget.jsonl") == pytest.approx(_settled(tmp_path / "clean" / "budget.jsonl"))
    assert not log.failures


@pytest.mark.parametrize(("hint", "waited"), [(17.0, 17.0), (240.0, retry.MAX_WAIT_S)])
def test_a_429s_retry_hint_is_honoured_up_to_the_cap(tmp_path: Path, hint: float, waited: float) -> None:
    slept: list[float] = []
    reader = LimitedFirst(fakes.FakeGateReader(), 1, hint=hint)
    assert gen_batch.generate_life(SPEC, _backend(tmp_path, slept, reader=reader)).attempts == 1
    assert slept == [waited]


def test_repeated_429s_exhaust_the_tries_then_the_next_attempt_cools_down(tmp_path: Path) -> None:
    # Arrange: the first leak read is answered 429 on every try the call may take
    slept: list[float] = []
    reader = LimitedFirst(fakes.FakeGateReader(), retry.RATE_LIMIT_ATTEMPTS)
    data, lines = DataDir(tmp_path / "pilot"), list[str]()

    # Act
    gen_batch.generate_lives(data, [SPEC], _backend(tmp_path, slept, reader=reader), lines.append)

    # Assert: the call waited out every try, doubling, then its attempt failed and the next one
    # started only after the cool-down, and passed
    waits = slept[:-1]
    assert len(waits) == retry.RATE_LIMIT_ATTEMPTS - 1 and slept[-1] == retry.ATTEMPT_COOL_DOWN_S
    assert all(_jittered(wait, retry.RATE_BACKOFF_S * 2.0 ** n) for n, wait in enumerate(waits))
    assert len(data.lives("core-train")) == 1
    (failed,) = data.failed()
    assert (failed["what"], failed["attempt"], failed["step"], failed["error"]) == (
        "generate-attempt", 0, "leak reader", "RateLimited")
    assert failed["rate_limits"] == retry.RATE_LIMIT_ATTEMPTS and failed["waits_s"] == [round(w, 1) for w in waits]
    (line,) = [line for line in lines if line.startswith("FAILED ")]
    assert f"{retry.RATE_LIMIT_ATTEMPTS} x HTTP 429, waited " in line
    assert any(line.startswith("generated ") and "(2 attempt(s)" in line for line in lines)


def test_a_life_whose_every_attempt_is_rate_limited_is_skipped_after_two_cool_downs(tmp_path: Path) -> None:
    slept: list[float] = []
    reader = LimitedFirst(fakes.FakeGateReader(), 10 ** 6)
    data, lines = DataDir(tmp_path / "pilot"), list[str]()

    with pytest.raises(RunsFailed):
        gen_batch.generate_lives(data, [SPEC], _backend(tmp_path, slept, reader=reader), lines.append)

    assert slept.count(retry.ATTEMPT_COOL_DOWN_S) == 2  # before the second attempt and the third
    assert [raw["attempt"] for raw in data.failed() if raw["what"] == "generate-attempt"] == [0, 1, 2]
    (skipped,) = [line for line in lines if line.startswith("SKIPPED ")]
    assert "HTTP 429" in skipped
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_a_rejected_call_still_skips_the_life_at_once(tmp_path: Path) -> None:
    slept: list[float] = []
    reader = LimitedFirst(fakes.FakeGateReader(), 1, error=ModelApiError("Gemini on Vertex returned HTTP 400"))
    with pytest.raises(ModelApiError, match="HTTP 400"):
        gen_batch.generate_life(SPEC, _backend(tmp_path, slept, reader=reader))
    assert slept == []


def test_the_muse_auditor_waits_out_a_429_too(tmp_path: Path) -> None:
    slept: list[float] = []
    auditor = LimitedFirst(fakes.FakeGateAuditor(), 1, kind="identify")
    made = gen_batch.generate_life(SPEC, _backend(tmp_path, slept, auditor=auditor))
    assert made.attempts == 1 and len(slept) == 1 and auditor.refused == 1


def test_the_writer_on_fireworks_waits_out_a_429(tmp_path: Path) -> None:
    slept: list[float] = []
    writer = LimitedFirst(fakes.FakeLifeChat(), 2, kind="write", hint=3.0)
    made = gen_batch.generate_life(SPEC, _backend(tmp_path, slept, life_chat=writer))
    assert made.attempts == 1 and slept == [3.0, 3.0]


# ── Where a 429 and its hint come from ────────────────────────────────────────


def test_a_fireworks_429_carries_its_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "12"}, json={"error": {"message": "slow down"}})

    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")
    with pytest.raises(RateLimited) as limited:
        fireworks_api.generate(fireworks_api.DEEPSEEK_V4P1_FLASH, [{"role": "user", "content": "x"}], None, 30.0,
                               transport=httpx.MockTransport(busy))
    assert limited.value.retry_after_s == 12.0 and str(limited.value).endswith("returned HTTP 429")
    assert "slow down" not in str(limited.value) and retry.rate_limited(limited.value)


class VertexError(Exception):
    """The shape of google-genai's ``APIError``: ``code``, ``status`` and ``details``."""

    def __init__(self, code: int | None, status: str, details: dict[str, Any]) -> None:
        super().__init__("the SDK's text, never kept")
        self.code, self.status, self.details = code, status, details


@pytest.mark.parametrize(("code", "delay", "hint"), [(429, "17s", 17.0), (None, "2.5s", 2.5), (429, None, None)])
def test_a_vertex_429_or_resource_exhausted_is_rate_limited_with_its_retry_delay(
        code: int | None, delay: str | None, hint: float | None) -> None:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}] if delay else []
    error = backends._vertex_error(VertexError(code, "RESOURCE_EXHAUSTED", {"error": {"details": details}}), 30.0)
    assert isinstance(error, RateLimited) and str(error) == VERTEX_429 and error.retry_after_s == hint


# ── One limiter per run ───────────────────────────────────────────────────────


class Counting:
    """A gate reader that counts the reads in flight at once."""

    def __init__(self) -> None:
        self.lock, self.now, self.most, self.inner = threading.Lock(), 0, 0, fakes.FakeGateReader()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        with self.lock:
            self.now += 1
            self.most = max(self.most, self.now)
        time.sleep(0.002)
        with self.lock:
            self.now -= 1
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def test_the_limiter_never_exceeds_its_count_under_parallel_attempts(tmp_path: Path) -> None:
    # Arrange: four lives side by side, their reads through one limiter of two
    specs = [spec for spec in core_splits.core_specs("core-train") if spec.prevalence > 0][:4]
    counting = Counting()
    backend = _backend(tmp_path, [], reader=Limiter(2).limit(counting))

    # Act
    gen_batch.generate_lives(DataDir(tmp_path / "pilot"), specs, backend, lambda line: None, parallel=4)

    # Assert
    assert counting.most == 2


def test_the_limiter_paces_its_starts_per_minute() -> None:
    now, slept = [100.0], list[float]()

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = Limiter(4, per_minute=30.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        with limiter.slot():
            pass
    assert slept == [2.0, 2.0]  # 30 a minute: one start every 2 s
    with pytest.raises(ValueError):
        Limiter(0)


def test_the_plan_prints_the_vertex_limits(tmp_path: Path) -> None:
    lines: list[str] = []
    assert cli.main([*LIVE, "--data", str(tmp_path), "generate", "--split", "core-train"], out=lines.append) == 0
    assert any("Gemini reads: at most 4 at once across the run" in line for line in lines)
    lines.clear()
    assert cli.main([*LIVE, "--data", str(tmp_path), "generate", "--split", "core-train", "--vertex-concurrency", "2",
                     "--vertex-rpm", "30"], out=lines.append) == 0
    assert any("at most 2 at once across the run, their starts paced to 30 a minute" in line for line in lines)
    assert any(f"cools down {retry.ATTEMPT_COOL_DOWN_S:g} s" in line for line in lines)
