# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Who reads the gates of MIND's core splits and pairs, and who writes them (MIND's data): by
default DeepSeek V4.1 Flash writes and Gemini on Vertex reads, each life records both and why it
may train the core, a split never mixes readers, APD's registered lives stay all Spark Open, a
Gemini reader is held on its own pool by the same hold code, and every call is logged so the plan's
shapes can be checked. Every model is a fake or a mock: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import calls, core_splits, fakes, gen_batch, generate, generators, labels, readers, retry
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, Rate
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import CoreSplit
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import usd
from anjo.sleep.provenance import CLEARED
from anjo.sleep.stops import BudgetStop

GEMINI_ID = readers.GEMINI_READER.model
PRICES = Prices(1.25, 4.25, 1.5, 2.61)
RATES = {MUSE: Rate(1.25, 4.25), GEMINI: Rate(0.75, 3.75)}
READ_RATE, WRITE_RATE = Rate(0.75, 3.75), Rate(0.30, 1.20)


class Seen:
    """A chat that records the model and the prompt's first line of each call."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.models, self.kinds = inner, list[Any](), list[str]()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        self.models.append(model)
        self.kinds.append(calls.kind_of(messages))
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def _items_spec(split: CoreSplit = "core-train") -> generate.LifeSpec:
    return next(spec for spec in core_splits.core_specs(split) if spec.prevalence >= 2)


def _gemini_backend(tmp_path: Path, **changes: Any) -> backends.Backend:
    return dataclasses.replace(backends.fake_backend(FIREWORKS), ledger=RunLedger(tmp_path / "budget.jsonl"),
                               prices=PRICES, rates=RATES, generator=generators.DEEPSEEK, generator_rate=WRITE_RATE,
                               read_by=readers.GEMINI_READER, reader_rate=READ_RATE, **changes)


def test_core_split_records_its_reader(tmp_path: Path) -> None:
    # Arrange: a core-train life with items, written by DeepSeek and read by Gemini
    writes, reads = Seen(fakes.FakeLifeChat()), Seen(fakes.FakeGateReader())
    backend = _gemini_backend(tmp_path, life_chat=writes, reader=reads)
    data = DataDir(tmp_path / "data")

    # Act
    gen_batch.generate_lives(data, [_items_spec()], backend, lambda line: None)

    # Assert: the gates asked Gemini, DeepSeek wrote, and the life's record says both and why it may train
    (life,) = data.lives("core-train")
    assert set(reads.models) == {GEMINI_ID} and set(reads.kinds) == {"leak", "discover"}
    assert set(writes.models) == {generators.DEEPSEEK.model} and generators.generator_of(life) == "deepseek-v4p1-flash"
    made = data.made(life.id)
    assert made is not None and (made["reader"], made["reader_model"]) == ("gemini", GEMINI_ID)
    assert made["writer_model"] == fireworks_api.DEEPSEEK_V4P1_FLASH and "MIT" in made["trains"]
    assert "MIND core only" in made["trains"] and generators.DEEPSEEK.model not in CLEARED  # the core only, not Glimmer
    # the labels export carries it, as the reason the life may train
    labels.export(data, "core-train", tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["lives"][0]["provenance"]["trains"] == generators.DEEPSEEK.trains

    # and the split refuses another reader's lives beside them, before any call
    with pytest.raises(StoreError, match="read by another reader than spark"):
        gen_batch.generate_lives(data, core_splits.core_specs("core-train")[:3],
                                 dataclasses.replace(backend, read_by=readers.SPARK_READER), lambda line: None)


def test_the_command_line_defaults_a_core_split_to_deepseek_and_gemini_and_refuses_mixing(tmp_path: Path) -> None:
    data, lines = tmp_path / "pilot", list[str]()
    base = ["--fake", "--data", str(data)]
    assert cli.main([*base, "--only", "1", "generate", "--split", "core-val"], out=lines.append) == 0
    assert cli.main([*base, "generate-pairs", "--n", "1"], out=lines.append) == 0
    (life,), (pair,) = DataDir(data).lives("core-val"), DataDir(data).pairs()
    for made_id in (life.id, pair.id):
        assert DataDir(data).made(made_id) == {
            "writer": "deepseek-v4p1-flash", "writer_model": fireworks_api.DEEPSEEK_V4P1_FLASH,
            "trains": generators.DEEPSEEK.trains, "reader": "gemini", "reader_model": GEMINI_ID}
    lines.clear()
    assert cli.main([*base, "--only", "2", "generate", "--split", "core-val", "--reader", "spark"],
                    out=lines.append) == cli.REFUSED_CODE
    assert cli.main([*base, "generate-pairs", "--n", "2", "--reader", "spark"], out=lines.append) == cli.REFUSED_CODE
    assert sum(line.startswith("Refused:") and "readers" in line for line in lines) == 2
    assert len(DataDir(data).lives("core-val")) == 1 and len(DataDir(data).pairs()) == 1


def test_apd_splits_refuse_a_non_spark_reader(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "pilot"
    for argv in (["generate", "--reader", "gemini"],
                 ["generate", "--split", "development", "--reader", "gemini"],
                 ["generate", "--held-out", "--draw-seed", "x", "--reader", "gemini"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(["--fake", "--data", str(data), *argv], out=lambda line: None)
        assert stopped.value.code == 2
        assert "read by Spark Open, as registered" in capsys.readouterr().err
    # below the command line too, before anything is written
    with pytest.raises(ValueError, match="Spark Open"):
        gen_batch.generate_lives(DataDir(data), generate.development_specs()[:1],
                                 dataclasses.replace(backends.fake_backend(FIREWORKS), read_by=readers.GEMINI_READER),
                                 lambda line: None)
    assert not data.exists()
    # APD's lives default to Spark Open, writer and reader both
    assert cli.main(["--fake", "--data", str(data), "--only", "1", "generate"], out=lambda line: None) == 0
    (life,) = DataDir(data).lives("development")
    assert DataDir(data).made(life.id) == {
        "writer": "spark", "writer_model": FIREWORKS.model, "trains": generators.SPARK.trains, "reader": "spark",
        "reader_model": FIREWORKS.model}


def test_reader_holds_on_its_own_pool(tmp_path: Path) -> None:
    # Arrange
    spec = _items_spec()
    backend = _gemini_backend(tmp_path)
    expected = backends.attempt_holds(backend, backends.life_calls(spec.prevalence))
    retries = backends.GEN_SPARK_RETRIES

    # Act
    gen_batch.generate_lives(DataDir(tmp_path / "data"), [spec], backend, lambda line: None)

    # Assert: the reads held on vertex_mind_usd at Gemini's prices, the writes alone on the provider's
    events = read_records(tmp_path / "budget.jsonl", PLAIN)
    held = {str(e["job"]).rsplit(":", 2)[-2]: e for e in events if e["op"] == "reserve"}
    assert held["reader"]["pools"] == ["vertex_mind_usd"] and held["spark"]["pools"] == [FIREWORKS.pool]
    reads = backends.life_calls(spec.prevalence).reads
    writes = backends.life_calls(spec.prevalence, day_retries=generate.DAY_RETRIES).writes  # DeepSeek's days asked again
    assert float(held["reader"]["amount"]) == pytest.approx((reads + retries) * READ_RATE.worst)
    assert float(held["spark"]["amount"]) == pytest.approx((writes + retries) * WRITE_RATE.worst)
    assert expected["vertex_mind_usd"] == pytest.approx(float(held["reader"]["amount"]))
    ledger = Ledger(tmp_path / "budget.jsonl")
    assert ledger.used("vertex_mind_usd") > 0 and ledger.open_holds() == ()
    assert CAPS["vertex_mind_usd"] > 0 and "vertex_mind_usd" != FIREWORKS.pool


def test_a_reader_without_room_stops_the_attempt_before_any_call(tmp_path: Path) -> None:
    # Arrange: vertex_mind_usd nearly full; another job's hold, never settled
    path = tmp_path / "budget.jsonl"
    Ledger(path).reserve("someone-else", ("vertex_mind_usd",), usd(CAPS["vertex_mind_usd"] - 0.01))
    writes = Seen(fakes.FakeLifeChat())
    backend = dataclasses.replace(_gemini_backend(tmp_path, life_chat=writes), ledger=RunLedger(path))

    # Act / Assert: the same hold code: no room and nothing in flight is a BudgetStop, before any write
    with pytest.raises(BudgetStop):
        gen_batch.generate_life(_items_spec(), backend)
    assert writes.models == [] and Ledger(path).used(FIREWORKS.pool) == 0.0


def test_core_train_only_1_runs_the_gates_and_regenerates_a_life_that_fails(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the leak reader flags the first window it reads, once
    flagged: list[bool] = []

    class FlagsOnce(fakes.FakeGateReader):
        def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                     temperature: float | None = None) -> ChatResult:
            if calls.kind_of(messages) == "leak" and not flagged:
                flagged.append(True)
                return ChatResult(json.dumps({"flag": True}), (), model, 10, 2, "stop")
            return super().__call__(model, messages, tools, timeout_s, temperature=temperature)

    monkeypatch.setattr(fakes, "FakeGateReader", FlagsOnce)
    data, lines = tmp_path / "pilot", list[str]()

    # Act
    code = cli.main(["--fake", "--data", str(data), "--only", "1", "generate", "--split", "core-train"],
                    out=lines.append)

    # Assert: the first attempt failed the leak gate and the life was written again, then gated again
    (life,) = DataDir(data).lives("core-train")
    assert code == 0 and flagged and any(line.startswith(f"generated {life.id} (2 attempt(s)") for line in lines)
    logged = [calls.Call.from_json(raw) for raw in DataDir(data).call_logs()[life.id]]
    assert {c.kind for c in logged if c.attempt == 1} >= {"write", "extract", "leak", "discover", "identify", "audit"}
    assert {c.kind for c in logged if c.attempt == 0} >= {"write", "extract", "leak"}


def test_every_call_is_logged_and_broken_down_without_the_sealed_topics(tmp_path: Path) -> None:
    # Arrange
    data, lines = tmp_path / "pilot", list[str]()
    assert cli.main(["--fake", "--data", str(data), "--only", "1", "generate", "--split", "core-test"],
                    out=lambda line: None) == 0
    (life,) = DataDir(data).lives("core-test")
    problems = DataDir(data).sealed(life.id)

    # Act
    assert cli.main(["--data", str(data), "calls"], out=lines.append) == 0

    # Assert: each item's reads are the gates' own (leak windows of both views, one discoverability read)
    logged = [calls.Call.from_json(raw) for raw in DataDir(data).call_logs()[life.id]]
    for number, item in enumerate(problems.items, 1):
        surface = life.days.index(item.surface_day)
        mine = [c for c in logged if c.item == number]
        assert sum(c.kind == "leak" for c in mine) == 2 * (2 * surface - 1)
        assert [c.days for c in mine if c.kind == "discover"] == [surface]
        assert {c.days for c in mine if c.kind == "leak"} <= {1, 2}
    assert sum(c.kind == "write" for c in logged) == sum(c.kind == "extract" for c in logged) == 10
    assert all(c.tokens_in > 0 and c.usd > 0 for c in logged)
    text = "\n".join(lines)
    assert "Per gate (kind):" in text and "leak, 2 day(s)" in text and f"{life.id} attempt 1 item 1" in text
    log_text = (data / "calls" / f"{life.id}.jsonl").read_text()
    assert all(item.topic not in log_text and item.topic not in text for item in problems.items)


# ── The calls ─────────────────────────────────────────────────────────────────


class _Models:
    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        self.reply, self.error, self.sent = reply, error, list[dict[str, Any]]()

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.sent.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.reply


def test_gemini_reads_at_low_thinking_capped_and_counts_its_thoughts(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = SimpleNamespace(prompt_token_count=1200, candidates_token_count=9, thoughts_token_count=140)
    models = _Models(SimpleNamespace(text='{"flag": false}', usage_metadata=usage))
    monkeypatch.setattr(backends, "_vertex", lambda: SimpleNamespace(models=models))
    messages = [{"role": "system", "content": "# leak_v1\nRead."}, {"role": "user", "content": '{"topic": "t"}'}]

    result = backends.gemini_chat(GEMINI_ID, messages, None, 90.0, temperature=0.0)

    (sent,) = models.sent
    config = sent["config"]
    assert sent["model"] == GEMINI_ID and sent["contents"] == '{"topic": "t"}'
    assert config.system_instruction == "# leak_v1\nRead." and config.temperature == 0.0
    assert config.thinking_config.thinking_level.value.lower() == "low"
    assert config.max_output_tokens == retry.REPLY_MAX_TOKENS and config.http_options.timeout == 90_000
    assert (result.text, result.input_tokens, result.output_tokens, result.model) == ('{"flag": false}', 1200, 149,
                                                                                      GEMINI_ID)


@pytest.mark.parametrize(("error", "transient"), [
    (SimpleNamespace(code=429), True), (SimpleNamespace(code=503), True), (SimpleNamespace(code=400), False),
    (httpx.ReadTimeout("slow"), True), (httpx.ConnectError("down"), True), (ValueError("odd"), False),
])
def test_a_failed_gemini_read_is_retried_only_when_transient(monkeypatch: pytest.MonkeyPatch, error: Any,
                                                             transient: bool) -> None:
    failure = error if isinstance(error, Exception) else type("APIError", (Exception,), {"code": error.code})()
    monkeypatch.setattr(backends, "_vertex", lambda: SimpleNamespace(models=_Models(error=failure)))
    with pytest.raises(ModelApiError) as raised:
        backends.gemini_chat(GEMINI_ID, [{"role": "user", "content": "{}"}], None, 30.0)
    assert retry.transient(raised.value) is transient and "Gemini on Vertex" in str(raised.value)


def test_deepseek_is_asked_for_its_non_thinking_mode_and_its_reasoning_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"model": sent[-1]["model"], "choices": [
            {"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {
            "prompt_tokens": 100, "completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 12}}})

    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")
    called: list[Any] = []
    real = fireworks_api.generate

    def generated(model: Any, *args: Any, **kw: Any) -> ChatResult:
        called.append(kw.get("reasoning_effort"))
        return real(model, *args, **kw, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(fireworks_api, "generate", generated)
    result = backends.gen_chat(FIREWORKS)(generators.DEEPSEEK.model, [{"role": "user", "content": "x"}], None, 30.0)
    assert called == ["none"] and sent[0]["reasoning_effort"] == "none"
    assert sent[0]["model"] == fireworks_api.DEEPSEEK_V4P1_FLASH and result.output_tokens >= 30
