# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The APD arms on their live models (Tasks 17 and 17b): Glimmer on Colab vLLM dated per night for
every arm, every Spark Open call held, the model of each world answer and resolver verdict
recorded as reported, and replay training only on runs that recorded them. Glimmer, Spark Open and
the VM are fakes: no model, no network."""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.mind.apd import fakes, generate, live
from anjo.sleep.mind.apd.arms import load_brief
from anjo.sleep.mind.apd.types import ArmName, ArmRun, Life, Written, model_written
from anjo.sleep.mind.inquiry.types import Answered, RunStart
from anjo.sleep.mind.replay import dataset, resolve, sample, served, settings
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices
from anjo.sleep.provenance import GLIMMER, may_train
from anjo.sleep.spark_inquiry import RESOLVE_PROMPT

PRICES = Prices(spark_in_per_m=0.6, spark_out_per_m=2.4, train_gpu_per_hour=1.5, serve_gpu_per_hour=2.61)
_TONIGHT = re.compile(r"## Tonight: (\d{4}-\d{2}-\d{2}), step")


@cache
def _development() -> Life:
    return fakes.fake_life(generate.development_specs()[1]).life


@cache
def _held_out() -> Life:
    spec = next(s for s in generate.draw_held_out("test-draw") if s.prevalence)
    return fakes.fake_life(spec).life


@pytest.fixture
def vm() -> Iterator[fakes.FakeColab]:
    yield fakes.FakeColab()


def _run(arm: ArmName, life: Life, vm: fakes.FakeColab, spark: Any = None) -> tuple[ArmRun, fakes.FakeVllm]:
    with fakes.serving(vm=vm) as session:
        run = live.run_live(arm, life, session, spark or fakes.FakeSparkChat(), ledger=None, prices=None)
        return run, vm.live[session.session]


def _night_of(run: ArmRun) -> dict[int, date]:
    """Each logged step's night: the night of the ``RunStart`` it follows."""
    nights: dict[int, date] = {}
    night: date | None = None
    for event in run.events:
        if isinstance(event.step.entry, RunStart):
            night = event.step.entry.night
        assert night is not None
        nights[event.step.seq] = night
    return nights


@pytest.mark.parametrize(("arm", "split"), [("A", "held_out"), ("B", "development"), ("M", "development"),
                                            ("M-persist", "held_out")])
def test_thinker_dated_per_night_for_every_arm(arm: ArmName, split: str, vm: fakes.FakeColab) -> None:
    life = _held_out() if split == "held_out" else _development()
    run, vllm = _run(arm, life, vm)
    days = [day for _, day, _ in vllm.served]
    assert vllm.served and sorted(days) == days and set(days) <= {d.isoformat() for d in life.days}
    if arm in ("A", "B"):  # each prompt names its night; a world query is dated as the step that asked it
        night = ""
        for kind, day, prompt in vllm.served:
            if kind == "think":
                found = _TONIGHT.search(prompt)
                assert found is not None
                night = found.group(1)
            assert day == night
        return
    nights = _night_of(run)
    think = [day for kind, day, _ in vllm.served if kind == "think"]
    world = [day for kind, day, _ in vllm.served if kind == "world"]
    assert think == [nights[turn.seq].isoformat() for turn in run.turns]
    asked = [e.step.seq for e in run.events if isinstance(e.step.entry, Answered)]
    assert asked and world == [nights[seq].isoformat() for seq in asked]


def test_reported_model_recorded_per_world_answer_and_verdict(vm: fakes.FakeColab) -> None:
    run, _ = _run("M", _development(), vm)
    observed = {e.step.seq: model_written(e.step.entry) for e in run.events if model_written(e.step.entry)}
    assert {"world", "resolver"} <= set(observed.values())
    assert {w.seq: w.port for w in run.written} == observed
    assert {w.port: w.model for w in run.written} == {"world": GLIMMER, "resolver": SPARK_OPEN}
    other, _ = _run("M", _development(), fakes.FakeColab(), fakes.FakeSparkChat(model="some-other-model"))
    assert {w.model for w in other.written if w.port == "resolver"} == {"some-other-model"}  # as reported
    with pytest.raises(ValueError, match="written record"):
        dataclasses.replace(run, written=(*run.written, Written(run.written[0].seq, "resolver", GLIMMER)))


@dataclasses.dataclass
class _HeldSpark:
    """Spark Open, checking that a hold on ``baseten_mind_usd`` is open at every call."""

    ledger: Ledger
    inner: fakes.FakeSparkChat = dataclasses.field(default_factory=fakes.FakeSparkChat)
    systems: list[str] = dataclasses.field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        assert any("baseten_mind_usd" in r.pools for r in self.ledger.open_holds())
        self.systems.append(str(messages[0]["content"]))
        return self.inner(messages, max_tokens)


def test_every_spark_open_call_held_on_baseten_mind_usd(tmp_path: Path, vm: fakes.FakeColab) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    spark = _HeldSpark(ledger)
    with fakes.serving(vm=vm) as session:
        live.run_live("M", _development(), session, spark, ledger=ledger, prices=PRICES)
    assert RESOLVE_PROMPT in spark.systems  # the resolver's calls
    assert ledger.open_holds() == () and ledger.used("baseten_mind_usd") > 0
    capped = live.SparkCalls(fakes.FakeSparkChat(), PRICES, ledger, "capped", per_night=2)
    with pytest.raises(Exception, match="holds"):
        for _ in range(3):
            capped([{"role": "system", "content": RESOLVE_PROMPT}, {"role": "user", "content": "x"}], 10)
    assert capped.calls == 2 and capped.chat.calls == 2  # type: ignore[attr-defined]
    assert len(ledger.open_holds()) == 1  # held before the first call, open while the night runs
    capped.close()
    assert ledger.open_holds() == ()
    with pytest.raises(ValueError, match="priced"):
        live.SparkCalls(fakes.FakeSparkChat(), None, ledger)([{"role": "user", "content": "x"}], 10)


def test_replay_accepts_a_recorded_m_run_and_refuses_an_unrecorded_one(vm: fakes.FakeColab) -> None:
    life = _development()
    with fakes.serving(vm=vm) as session:
        run = live.run_live("M", life, session, fakes.FakeSparkChat(), ledger=None, prices=None)
        resolving = {e.step.seq for e in run.events if model_written(e.step.entry) == "resolver"}
        nights = _night_of(run)
        assert len({nights[seq] for seq in resolving}) >= len(life.days) // 2  # predictions resolve most nights
        starts = sample.start_states(life, run)
        assert starts and all(s.ancestry is not None and may_train(s.ancestry, hosted_in_training=False)
                              for s in starts)
        unrecorded = sample.start_states(life, dataclasses.replace(run, written=()))
        assert any(s.ancestry is None for s in unrecorded)
        held = {w.seq for w in run.written}
        assert all(s.ancestry is None for s in unrecorded if any(e.step.seq in held for e in s.events))
        stranger = tuple(dataclasses.replace(w, model="unknown-model") for w in run.written)
        assert not any(may_train(s.ancestry, hosted_in_training=False) for s in
                       sample.start_states(life, dataclasses.replace(run, written=stranger))
                       if s.ancestry is not None and "unknown-model" in s.ancestry)
        labels = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=fakes.FakeSparkChat()))
        spark = live.SparkCalls(fakes.FakeSparkChat(), None)
        arm = live.arm_models(session, spark)
        models = sample.ReplayModels(thinker_for=served.branch_thinkers(session), world=arm.world,
                                     resolver_chat=fakes.FakeSparkChat())
        branches = sample.sample_all([(life, starts)], models, labels, k=2)
    data = dataset.build([life], starts, branches, labels, brief=load_brief(), penalty=settings.token_penalty())
    assert data.thinker1.examples  # a real-shaped M run gives replay data
