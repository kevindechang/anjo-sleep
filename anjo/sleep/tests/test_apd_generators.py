# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A cheaper writer for MIND's core splits and pairs (MIND's data): ``--generator`` picks the model
that writes and extracts them, each life records it, a split never mixes two, APD's registered
development and held-out lives stay on Spark Open, the gates still read on Spark Open, and the plan
prices the generator chosen. Every model is a fake or a mock transport: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import core_splits, fakes, gen_batch, generate, generators, pairs, plan, retry
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices

GPT, GLM = generators.GENERATORS["gpt-oss-120b"], generators.GENERATORS["glm-5p3-flash"]
GPT_ID, GLM_ID = fireworks_api.GPT_OSS_120B, fireworks_api.GLM_5P3_FLASH
GPT_PRICES, GLM_PRICES = Rate(0.15, 0.60), Rate(0.15, 0.50)
LIVE = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]
PRICES = Prices(1.25, 4.25, 1.5, 2.61)
RATES = cli._rates("1.25,4.25,0.75,3.75")


class Seen:
    """A chat that records the model each call asks for."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.models: list[Any] = []

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        self.models.append(model)
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def _writers(life: Any) -> set[str]:
    return {session.model for session in life.sessions}


def test_core_split_records_its_generator(tmp_path: Path) -> None:
    # Arrange: one core-train life with items, written by gpt-oss-120b, held at real-like prices
    spec = next(spec for spec in core_splits.core_specs("core-train") if spec.prevalence > 0)
    writes, reads = Seen(fakes.FakeLifeChat()), Seen(fakes.FakeGateReader())
    backend = dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=writes, reader=reads,
                                  ledger=RunLedger(tmp_path / "budget.jsonl"), prices=PRICES, rates=RATES,
                                  generator=GPT, generator_rate=GPT_PRICES)

    # Act
    gen_batch.generate_lives(DataDir(tmp_path / "data"), [spec], backend, lambda line: None)

    # Assert: written and extracted by the generator, and recorded so; read by the gates on Spark Open
    (life,) = DataDir(tmp_path / "data").lives("core-train")
    assert generators.generator_of(life) == "gpt-oss-120b"
    assert _writers(life) == {item.model for item in life.evidence} == {GPT.model}
    assert set(writes.models) == {GPT.model} and set(reads.models) == {FIREWORKS.model}
    # its attempt held the generator's worst case for its writes, Spark Open's for its reads
    held = next(float(e["amount"]) for e in read_records(tmp_path / "budget.jsonl", PLAIN)
                if e["op"] == "reserve" and str(e["job"]).endswith(":spark:a1"))
    calls = backends.life_calls(spec.prevalence)
    assert held == pytest.approx(backends.gen_holds(PRICES, RATES, FIREWORKS.pool, calls, GPT_PRICES)[FIREWORKS.pool])
    assert held < backends.gen_holds(PRICES, RATES, FIREWORKS.pool, calls)[FIREWORKS.pool]


def test_the_command_line_generates_a_core_split_and_pairs_with_the_generator(tmp_path: Path) -> None:
    data, lines = tmp_path / "pilot", list[str]()
    base = ["--fake", "--data", str(data)]
    assert cli.main([*base, "--only", "1", "generate", "--split", "core-test", "--generator", "glm-5p3-flash"],
                    out=lines.append) == 0
    assert cli.main([*base, "generate-pairs", "--n", "1", "--generator", "gpt-oss-120b"], out=lines.append) == 0
    (life,) = DataDir(data).lives("core-test")
    (pair,) = DataDir(data).pairs()
    assert generators.generator_of(life) == "glm-5p3-flash" and _writers(life) == {GLM.model}
    assert _writers(pair.failed) | _writers(pair.kept) == {GPT.model}


def test_split_refuses_mixed_generators(tmp_path: Path) -> None:
    # Arrange: a core-val life and a pair Spark Open wrote
    data, lines = tmp_path / "pilot", list[str]()
    base = ["--fake", "--data", str(data)]
    assert cli.main([*base, "--only", "1", "generate", "--split", "core-val"], out=lines.append) == 0
    assert cli.main([*base, "generate-pairs", "--n", "1"], out=lines.append) == 0
    (first,) = DataDir(data).lives("core-val")
    lines.clear()

    # Act: more of each with another generator
    lives = cli.main([*base, "--only", "2", "generate", "--split", "core-val", "--generator", "glm-5p3-flash"],
                     out=lines.append)
    paired = cli.main([*base, "generate-pairs", "--n", "2", "--generator", "gpt-oss-120b"], out=lines.append)

    # Assert: both refused before anything was written; the life on disk is a mismatch, not a reuse
    assert lives == paired == cli.REFUSED_CODE
    refused = [line for line in lines if line.startswith("Refused:")]
    assert len(refused) == 2 and first.id in refused[0]
    assert [life.id for life in DataDir(data).lives("core-val")] == [first.id] and len(DataDir(data).pairs()) == 1
    # the same generator still goes on from what is on disk
    assert cli.main([*base, "--only", "2", "generate", "--split", "core-val"], out=lines.append) == 0
    assert len(DataDir(data).lives("core-val")) == 2


def test_apd_splits_refuse_a_non_spark_generator(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "pilot"
    for argv in (["generate", "--generator", "gpt-oss-120b"],
                 ["generate", "--split", "development", "--generator", "glm-5p3-flash"],
                 ["generate", "--held-out", "--draw-seed", "x", "--generator", "gpt-oss-120b"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(["--fake", "--data", str(data), *argv], out=lambda line: None)
        assert stopped.value.code == 2
        assert "Spark Open" in capsys.readouterr().err
    # below the command line too, before anything is written
    with pytest.raises(ValueError, match="Spark Open"):
        gen_batch.generate_lives(DataDir(data), generate.development_specs()[:1],
                                 dataclasses.replace(backends.fake_backend(FIREWORKS), generator=GPT),
                                 lambda line: None)
    assert not data.exists()


def test_plan_prices_the_chosen_generator(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    specs = core_splits.core_specs("core-test")
    scale = plan.write_scale(GPT_PRICES, PRICES)
    expected = sum(plan.expected_usd(plan.expected_calls(plan.surface_days(spec)), scale) for spec in specs)
    spark = sum(plan.expected_usd(plan.expected_calls(plan.surface_days(spec))) for spec in specs)
    held = [backends.gen_holds(PRICES, RATES, FIREWORKS.pool, backends.life_calls(spec.prevalence), GPT_PRICES)
            for spec in specs]
    lines: list[str] = []

    # Act
    assert cli.main([*LIVE, "--data", str(tmp_path), "generate", "--split", "core-test", "--generator", "gpt-oss-120b",
                     "--generator-prices", "0.15,0.60"], out=lines.append) == 0
    assert cli.main([*LIVE, "--data", str(tmp_path), "generate-pairs", "--generator", "glm-5p3-flash",
                     "--generator-prices", "0.15,0.50"], out=lines.append) == 0

    # Assert: the writes at the generator's prices, the reads at Spark Open's, in the holds and the expected
    text = "\n".join(lines)
    mean = sum(h[FIREWORKS.pool] for h in held) / len(held)
    assert f"Hold on {FIREWORKS.pool}: ${mean:.2f} each" in text
    assert f"Expected: about ${expected:,.2f} for 40 lives" in text and 0 < scale < 1 and expected < spark
    assert GPT_ID in text and "reasoning effort low" in text
    pair = plan.expected_usd(pairs.PAIR_EXPECTED, plan.write_scale(GLM_PRICES, PRICES))
    assert f"Expected: about ${20 * pair:,.2f} for 20 pairs" in text and GLM_ID in text
    # a generator without its prices is refused
    with pytest.raises(SystemExit) as stopped:
        cli.main([*LIVE, "--data", str(tmp_path), "generate", "--split", "core-test", "--generator", "gpt-oss-120b"],
                 out=lines.append)
    assert stopped.value.code == 2 and "--generator-prices" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# ── The calls ─────────────────────────────────────────────────────────────────


def _reply(model: str, usage: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"model": model, "usage": usage,
                                     "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 100, "completion_tokens": 70, "total_tokens": 170,
     "completion_tokens_details": {"reasoning_tokens": 50}},  # reasoning inside the completion
    {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 170,
     "completion_tokens_details": {"reasoning_tokens": 50}},  # apart, in the total
    {"prompt_tokens": 100, "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 50}},  # apart
])
def test_gpt_oss_asks_for_low_reasoning_and_counts_it_as_output(monkeypatch: pytest.MonkeyPatch,
                                                                  usage: dict[str, Any]) -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(sent[-1]["model"], usage)

    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-test")
    transport = httpx.MockTransport(handler)
    messages = [{"role": "user", "content": "a day"}]
    result = fireworks_api.generate(fireworks_api.GPT_OSS_120B, messages, None, 30.0, temperature=0.8, max_tokens=8192,
                                    reasoning_effort="low", transport=transport)
    fireworks_api.generate(fireworks_api.GLM_5P3_FLASH, messages, None, 30.0, max_tokens=8192, transport=transport)
    assert sent[0] == {"model": GPT.model, "messages": messages, "temperature": 0.8, "max_completion_tokens": 8192,
                       "reasoning_effort": "low"}
    assert "reasoning_effort" not in sent[1]
    assert (result.input_tokens, result.output_tokens) == (100, 70)


def test_generation_calls_go_to_the_generator_and_reads_to_spark_open(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[tuple[str, str, Any]] = []

    def generated(model: str, *args: Any, **kw: Any) -> ChatResult:
        called.append(("generator", model, kw.get("reasoning_effort")))
        return ChatResult("{}", (), model, 1, 1, "stop")

    def spark(model: str, *args: Any, **kw: Any) -> ChatResult:
        called.append(("spark", model, kw.get("max_tokens")))
        return ChatResult("{}", (), model, 1, 1, "stop")

    monkeypatch.setattr(fireworks_api, "generate", generated)
    chat = backends.gen_chat(dataclasses.replace(FIREWORKS, write=spark))
    for model in (GPT.model, GLM.model, FIREWORKS.model):
        chat(model, [{"role": "user", "content": "x"}], None, 30.0, temperature=0.0)
    assert called == [("generator", GPT.model, "low"), ("generator", GLM.model, None),
                      ("spark", FIREWORKS.model, retry.REPLY_MAX_TOKENS)]
