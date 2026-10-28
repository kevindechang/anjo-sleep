# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""APD's life generation, made to survive Fireworks (MIND's data): a transient failure is tried
again inside the attempt's own hold, every reply goes out capped and waits as long as its cap
needs, a life that still fails is skipped and the run goes on, and lives are generated side by
side with their results and lines in life order. Every model is a fake: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import fakes, gates, gen_batch, generate, plan, retry
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.judging import MUSE, Rate
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import LIFE_DAYS
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import SPARK_MAX_OUT_TOKENS

EXTRACT = "# extract_open_v1"
WRITE = "# life_v1"
SPEC = generate.development_specs()[1]  # two planted items: the life that died live


@dataclasses.dataclass
class TimesOut:
    """A generation chat whose ``nth`` extraction call times out ``times`` times first, as
    Fireworks did live (``model_api``'s own message); ``seeds`` makes every writer call of those
    lives (by its ``variation``) time out instead. It records each call's system line and timeout."""

    inner: fakes.FakeLifeChat = dataclasses.field(default_factory=fakes.FakeLifeChat)
    nth: int = 4
    times: int = 1
    seeds: frozenset[int] = frozenset()
    extracts: int = 0
    failed: int = 0
    seen: list[tuple[str, float]] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        system = str(messages[0]["content"]).splitlines()[0]
        self.seen.append((system, timeout_s))
        if system == WRITE and json.loads(messages[-1]["content"])["variation"] in self.seeds:
            raise ModelApiError(f"Fireworks chat with {model} timed out after {timeout_s:g}s")
        if system == EXTRACT:
            self.extracts += 1
            if self.extracts == self.nth and self.failed < self.times:
                self.failed += 1
                self.extracts -= 1
                raise ModelApiError(f"Fireworks chat with {model} timed out after {timeout_s:g}s")
        return self.inner(model, messages, tools, timeout_s, temperature=temperature)


def _backend(tmp_path: Path, chat: Any, slept: list[float]) -> backends.Backend:
    return dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=chat, sleep=slept.append,
                               ledger=RunLedger(tmp_path / "budget.jsonl"), prices=cli.FAKE_PRICES,
                               rates=cli.FAKE_RATES)


def _events(path: Path) -> list[dict[str, Any]]:
    return list(read_records(path, PLAIN))


def _spark_hold(path: Path) -> tuple[float, float]:
    """The attempt's Spark Open hold: what it reserved and what it settled."""
    events = _events(path)
    job = next(str(e["job"]) for e in events if e["op"] == "reserve" and ":spark" in str(e["job"]))
    reserved = next(float(e["amount"]) for e in events if e["op"] == "reserve" and e["job"] == job)
    settled = next(float(e["amount"]) for e in events if e["op"] == "settle" and e["job"] == job)
    return reserved, settled


# ── Retries ───────────────────────────────────────────────────────────────────


def test_fireworks_timeout_retries_inside_the_hold(tmp_path: Path) -> None:
    # Arrange: the same life once without a failure, then with its fourth extraction timing out once
    clean = gen_batch.generate_life(SPEC, _backend(tmp_path / "clean", TimesOut(times=0), []))
    _, clean_settled = _spark_hold(tmp_path / "clean" / "budget.jsonl")
    slept: list[float] = []
    chat = TimesOut(times=1)

    # Act
    made = gen_batch.generate_life(SPEC, _backend(tmp_path / "flaky", chat, slept))

    # Assert: the call was tried again within the attempt, after one backoff, and the life is the same
    assert chat.failed == 1 and made.attempts == 1 and len(slept) == 1
    assert made.life == clean.life and made.problems == clean.problems
    reserved, settled = _spark_hold(tmp_path / "flaky" / "budget.jsonl")
    worst = backends.spark_rate(cli.FAKE_PRICES)
    assert worst is not None
    calls = backends.GEN_WRITE_CALLS + SPEC.prevalence * backends.GEN_READS_PER_ITEM + backends.GEN_SPARK_RETRIES
    assert reserved == pytest.approx(worst.worst * calls)  # sized to the life's two planted items
    assert settled == pytest.approx(clean_settled + worst.worst)  # the failed try, at its worst case
    assert settled <= reserved
    assert {timeout for system, timeout in chat.seen if system == EXTRACT} == {generate.TIMEOUT_S}  # the retry too
    assert RunLedger(tmp_path / "flaky" / "budget.jsonl").open_holds() == ()


def test_a_call_still_failing_after_its_tries_fails_inside_the_hold(tmp_path: Path) -> None:
    chat = TimesOut(times=retry.ATTEMPTS)
    with pytest.raises(ModelApiError, match="timed out"):
        gen_batch.generate_life(SPEC, _backend(tmp_path, chat, []))
    assert chat.failed == retry.ATTEMPTS
    reserved, settled = _spark_hold(tmp_path / "budget.jsonl")
    assert 0 < settled <= reserved
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_extraction_timeout_scales_with_max_tokens() -> None:
    # The wait is the first token's allowance plus every token at a slow, healthy decode speed
    assert retry.reply_timeout_s(8_192) == pytest.approx(retry.FIRST_TOKEN_S + 8_192 / retry.SLOW_TOKENS_PER_S)
    assert retry.reply_timeout_s(1_024) < retry.reply_timeout_s(4_096) < retry.reply_timeout_s(8_192)
    assert retry.REPLY_MAX_TOKENS <= SPARK_MAX_OUT_TOKENS  # one call's worst case stays a true bound
    assert generate.TIMEOUT_S == gates.TIMEOUT_S == retry.reply_timeout_s(retry.REPLY_MAX_TOKENS) > 180.0

    # Live wiring: every generation call goes out capped, with the timeout its cap needs
    sent: list[tuple[str, float, int | None]] = []

    def write(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
              temperature: float | None = None, max_tokens: int | None = None) -> ChatResult:
        sent.append((str(messages[0]["content"]).splitlines()[0], timeout_s, max_tokens))
        return fakes.FakeLifeChat()(model, messages, tools, timeout_s, temperature=temperature)

    provider = dataclasses.replace(FIREWORKS, write=write)
    backend = dataclasses.replace(backends.fake_backend(provider), life_chat=backends.gen_chat(provider))
    gen_batch.generate_life(SPEC, backend)
    extracts = [(timeout, cap) for system, timeout, cap in sent if system == EXTRACT]
    assert len(extracts) == LIFE_DAYS
    assert set(extracts) == {(generate.TIMEOUT_S, retry.REPLY_MAX_TOKENS)}
    assert {cap for _, _, cap in sent} == {retry.REPLY_MAX_TOKENS}


# ── Skipping and parallel ─────────────────────────────────────────────────────


def test_failed_life_is_skipped_and_the_run_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: every writer call of the second development life times out, whatever its attempt
    specs = generate.development_specs()[:3]
    dead = frozenset(generate.attempt_seed(specs[1].seed, n) for n in range(generate.MAX_ATTEMPTS))
    real = cli._fake_backend

    def flaky(data: DataDir, provider: Any) -> backends.Backend:
        return dataclasses.replace(real(data, provider), life_chat=TimesOut(times=0, seeds=dead))

    monkeypatch.setattr(cli, "_fake_backend", flaky)
    data = tmp_path / "pilot"
    lines: list[str] = []

    # Act
    code = cli.main(["--fake", "--data", str(data), "--only", "3", "generate", "--parallel", "2"], out=lines.append)

    # Assert: the first and third lives are saved, the second is skipped, recorded and named
    text = "\n".join(lines)
    assert code == cli.STOPPED, text
    saved = {life.seed for life in DataDir(data).lives("development")}
    assert saved == {specs[0].seed, specs[2].seed}
    done = [line for line in lines if line.startswith(("generated ", "SKIPPED "))]
    assert [line.split()[0] for line in done] == ["generated", "SKIPPED", "generated"]  # in life order
    assert str(specs[1].seed) in done[1] and "timed out" in done[1]
    failed = DataDir(data).failed()
    assert len(failed) == 1 and failed[0]["what"] == "generate" and str(specs[1].seed) in str(failed[0]["life"])
    assert "Stopped: 1 failed and weren't saved" in text
    assert RunLedger(data / "fake-budget.jsonl").open_holds() == ()

    # A rerun generates only the skipped life
    monkeypatch.setattr(cli, "_fake_backend", real)
    lines.clear()
    assert cli.main(["--fake", "--data", str(data), "--only", "3", "generate"], out=lines.append) == 0
    assert len([line for line in lines if line.startswith("generated ")]) == 1
    assert len(DataDir(data).lives("development")) == 3


class Together:
    """A life chat whose first writer call of each life waits until ``count`` lives are writing at
    once: it fails (BrokenBarrierError) unless they really run side by side."""

    def __init__(self, count: int) -> None:
        self.barrier = threading.Barrier(count, timeout=30)
        self.first: set[int] = set()
        self.lock = threading.Lock()
        self.inner = fakes.FakeLifeChat()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if str(messages[0]["content"]).startswith(WRITE):
            seed = json.loads(messages[-1]["content"])["variation"]
            with self.lock:
                new = seed not in self.first and len(self.first) < self.barrier.parties
                self.first.add(seed)
            if new:
                self.barrier.wait()
        return self.inner(model, messages, tools, timeout_s, temperature=temperature)


def _files(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): path.read_text(encoding="utf-8")
            for folder in ("lives", "sealed") for path in sorted((root / folder).glob("*.json"))}


def test_parallel_generation_matches_serial_output(tmp_path: Path) -> None:
    # Arrange: the 8 development lives, once one at a time and once four at once
    specs = generate.development_specs()
    serial_lines: list[str] = []
    parallel_lines: list[str] = []

    # Act
    gen_batch.generate_lives(DataDir(tmp_path / "serial"), specs, backends.fake_backend(FIREWORKS),
                             serial_lines.append, parallel=1)
    gen_batch.generate_lives(DataDir(tmp_path / "parallel"),
                             specs, dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=Together(4)),
                             parallel_lines.append, parallel=4)

    # Assert: the same lives, sealed files and lines, in life order
    assert parallel_lines == serial_lines and len(serial_lines) == len(specs)
    assert all(line.split()[1] in gen_batch.spec_ids(spec) for spec, line in zip(specs, serial_lines, strict=True))
    assert _files(tmp_path / "parallel") == _files(tmp_path / "serial") and len(_files(tmp_path / "serial")) == 16


def test_parallel_generation_stops_new_lives_at_the_ceiling(tmp_path: Path) -> None:
    data = tmp_path / "pilot"
    lines: list[str] = []
    code = cli.main(["--fake", "--data", str(data), "--only", "4", "generate", "--parallel", "2", "--max-usd", "0.0001"],
                    out=lines.append)
    assert code == cli.CEILING, "\n".join(lines)
    made = DataDir(data).lives("development")
    assert 1 <= len(made) <= 2  # the lives in flight finish; nothing new starts
    specs = generate.development_specs()
    reserved = {str(e["job"]) for e in _events(data / "fake-budget.jsonl") if e["op"] == "reserve"}
    assert not any(job.startswith(f"apd:gen:development:{spec.seed}:") for spec in specs[2:4] for job in reserved)
    assert RunLedger(data / "fake-budget.jsonl").open_holds() == ()


LIVE = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]


def test_generate_plan_prints_worst_case_and_expected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_backend(*args: object, **kwargs: object) -> None:
        raise AssertionError("a plan builds no live backend")

    monkeypatch.setattr(backends, "live_backend", no_backend)
    lines: list[str] = []
    assert cli.main([*LIVE, "--data", str(tmp_path), "generate"], out=lines.append) == 0
    text = "\n".join(lines)
    specs = generate.development_specs()
    lives = len(specs)
    on = plan.pricing(Prices(1.25, 4.25, 1.5, 2.61), {MUSE: Rate(1.25, 4.25)}, FIREWORKS)
    expected = sum(sum(plan.life_expected(plan.surface_days(spec), on).values()) for spec in specs)
    worst = sum(float(line.split("at most $")[1].split(" ")[0].replace(",", ""))
                for line in text.splitlines() if line.startswith("Hold on "))
    assert f"Worst case: ${worst:,.2f} for {lives} lives" in text
    assert f"Expected: about ${expected:,.2f} for {lives} lives" in text
    assert "up to 4 at once" in text and f"{backends.GEN_SPARK_RETRIES} extra tries" in text
    assert list(tmp_path.iterdir()) == []
