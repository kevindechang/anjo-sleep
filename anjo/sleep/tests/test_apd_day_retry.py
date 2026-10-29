# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A core writer's day that can't be read is asked for again (MIND's data): at most
``generate.DAY_RETRIES`` more times inside the attempt, each ask logged as its own ``write`` call
with its ``retry`` count, the attempt held for them; a day still unreadable fails the attempt,
recorded with how many beats it left unplaced and how many keys it added, never its text. APD's
own writer (Spark Open) is asked once, as registered. Every model is a fake: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import calls, core_splits, fakes, gen_batch, generate, generators, pairs, plan, readers, repair
from anjo.sleep.mind.apd.calls import CallLog
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, Rate
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import LIFE_DAYS
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices

PRICES = Prices(1.25, 4.25, 2.50, 1.95)
RATES = {MUSE: Rate(1.25, 4.25), GEMINI: Rate(0.75, 3.75)}
WRITER = Rate(0.30, 1.20)
SPEC = core_splits.core_specs("core-train")[0]  # 6 items
LifeChat = fakes.FakeLifeChat


def _beat_day(spec: Any) -> int:
    """The first day (from 1) of ``spec``'s life that holds a beat."""
    beats = generate.beats_of(generate.plan(spec.split, spec.field, spec.families, spec.prevalence, spec.seed))
    return min(beats) + 1


@dataclasses.dataclass
class Misplaces:
    """The fakes' writer, its day ``day`` answered ``bad`` times with one beat left unplaced and one
    unknown key added before it is answered whole; each write's day kept."""

    day: int
    bad: int
    inner: Any = dataclasses.field(default_factory=LifeChat)
    days: list[int] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        if calls.kind_of(messages) != "write":
            return result
        day = json.loads(messages[-1]["content"])["day"]
        self.days.append(day)
        if day != self.day or self.days.count(day) > self.bad:
            return result
        body = json.loads(result.text)
        placed = dict(list(body["placed"].items())[1:]) | {"i9.x": 0}
        return dataclasses.replace(result, text=json.dumps({**body, "placed": placed}))


def _backend(tmp_path: Path, writer: Any, generator: generators.Generator = generators.DEEPSEEK) -> backends.Backend:
    """Written by ``generator`` and read by Gemini, held at real-like prices."""
    return dataclasses.replace(backends.fake_backend(FIREWORKS), ledger=RunLedger(tmp_path / "budget.jsonl"),
                               prices=PRICES, rates=RATES, generator=generator,
                               generator_rate=None if generator.model is None else WRITER,
                               read_by=readers.GEMINI_READER, reader_rate=RATES[GEMINI], life_chat=writer)


def test_a_day_the_writer_misplaces_once_is_asked_again_and_the_life_passes(tmp_path: Path) -> None:
    # Arrange
    day = _beat_day(SPEC)
    writer, log = Misplaces(day, bad=1), CallLog()

    # Act
    made = gen_batch.generate_life(SPEC, _backend(tmp_path, writer), log)

    # Assert: one attempt; the day asked twice, the second ask logged as retry 1, with no text
    assert made.attempts == 1 and not log.failures and writer.days.count(day) == 2
    writes = [(c.days, c.retry) for c in log.calls if c.kind == "write"]
    assert writes == [(1 + n, 0) for n in range(day)] + [(day, 1)] + [(n, 0) for n in range(day + 1, LIFE_DAYS + 1)]
    assert all(set(c.to_json()) == set(calls.Call.__dataclass_fields__) for c in log.calls)
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_a_day_misplaced_on_every_ask_fails_the_attempt_with_its_counts(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: one attempt allowed; the day's beats misplaced on every ask
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 1)
    day = _beat_day(SPEC)
    writer, data = Misplaces(day, bad=generate.DAY_RETRIES + 1), DataDir(tmp_path / "pilot")

    # Act
    with pytest.raises(Exception):
        gen_batch.generate_lives(data, [SPEC], _backend(tmp_path, writer), lambda line: None)

    # Assert: asked 1 + DAY_RETRIES times, then the attempt failed, recorded with the counts and no text
    assert writer.days.count(day) == 1 + generate.DAY_RETRIES and writer.days[-1] == day
    (raw,) = [raw for raw in data.failed() if raw["what"] == "generate-attempt"]
    assert (raw["step"], raw["day"], raw["field"], raw["reason"]) == ("writer", day, "placed",
                                                                      "a beat is unplaced or unknown")
    assert (raw["missing"], raw["extra"]) == (1, 1) and "i9.x" not in json.dumps(raw)  # counts, never the keys
    retries = [call["retry"] for found in data.call_logs().values() for call in found
               if call["kind"] == "write" and call["days"] == day]
    assert retries == list(range(1 + generate.DAY_RETRIES))


def test_apds_own_writer_is_asked_once_as_registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 1)
    spec = next(found for found in generate.development_specs() if found.prevalence)
    day = _beat_day(spec)
    writer, log = Misplaces(day, bad=1), CallLog()
    backend = dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=writer)
    with pytest.raises(generate.GenerationError):
        gen_batch.generate_life(spec, backend, log)
    assert writer.days.count(day) == 1 and writer.days[-1] == day
    (failed,) = log.failures
    assert (failed.missing, failed.extra) == (1, 1) and failed.field == "placed"


def test_a_core_attempt_holds_its_day_retries_and_apds_does_not() -> None:
    # every hold of a core writer's work grows by its days' retries: an attempt, a repair round, a pair
    retries = generate.DAY_RETRIES
    assert generate.day_retries(str(generators.DEEPSEEK.model)) == retries
    assert generate.day_retries(FIREWORKS.model) == 0
    assert backends.life_calls(6, day_retries=retries).writes == backends.GEN_WRITE_CALLS + retries * LIFE_DAYS
    assert repair.round_calls(4, {1, 3}, day_retries=retries).writes == repair.round_calls(4, {1, 3}).writes + 2 * retries
    assert pairs.pair_calls(retries).writes == pairs.PAIR_CALLS.writes + retries * LIFE_DAYS
    # the plan's worst case: a core attempt's writer hold grows by DAY_RETRIES writer calls a day; APD's is unchanged
    core = plan.generate_plan([plan.surface_days(SPEC)], PRICES, RATES, "core-train", provider=FIREWORKS,
                              generator=generators.DEEPSEEK, generator_rate=WRITER, reader=readers.GEMINI_READER,
                              reader_rate=RATES[GEMINI])
    (writes,) = [hold for hold in core.holds if hold.pool == FIREWORKS.pool]
    held = backends.GEN_WRITE_CALLS + retries * LIFE_DAYS + backends.GEN_SPARK_RETRIES
    assert writes.each == pytest.approx(held * WRITER.worst)
    spec = next(found for found in generate.development_specs() if found.prevalence)
    apd = plan.generate_plan([plan.surface_days(spec)], PRICES, RATES, "development", provider=FIREWORKS)
    (spark,) = [hold for hold in apd.holds if hold.pool == FIREWORKS.pool]
    assert spark.each == pytest.approx(backends.gen_holds(PRICES, RATES, FIREWORKS.pool,
                                                          backends.life_calls(spec.prevalence))[FIREWORKS.pool])
