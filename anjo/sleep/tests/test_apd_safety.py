# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""What keeps a live APD command honest when things go wrong (Task 17 review): a transient failure
is tried again, a run with a call that still fails is never saved and fails alone while the others
go on, a batch stops cleanly before a run its session can't finish, nothing calls Spark Open once
the session stopped, and an interrupted batch starts nothing more. Fakes only: no model, no VM, no
network."""
from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import batch, colab_vllm, fakes, generate, live, pilot
from anjo.sleep.mind.apd.journal import SpendJournal
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import ArmRun, Life
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices

PRICES = Prices(spark_in_per_m=0.6, spark_out_per_m=2.4, train_gpu_per_hour=1.5, serve_gpu_per_hour=2.61)


def _lives(data: DataDir, count: int) -> tuple[Life, ...]:
    for spec in generate.development_specs()[:count]:
        made = fakes.fake_life(spec)
        data.save_life(made.life, made.problems)
    return data.lives("development")[:count]


class _FailingSpark(fakes.FakeSparkChat):
    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        if self.calls == 2:
            self.calls += 1
            raise ModelApiError("Spark Open timed out")
        return super().__call__(messages, max_tokens)


class _FlakySpark(fakes.FakeSparkChat):
    """Spark Open failing with ``errors`` in turn, then answering as the fake does."""

    def __init__(self, *errors: str) -> None:
        super().__init__()
        self.errors = list(errors)
        self.tries = 0

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.tries += 1
        if self.errors:
            raise ModelApiError(self.errors.pop(0))
        return super().__call__(messages, max_tokens)


@pytest.mark.parametrize("broken", ["think", "world", "spark"])
def test_a_run_with_a_call_that_still_fails_is_never_saved_and_fails_alone(tmp_path: Path, broken: str) -> None:
    data = DataDir(tmp_path)
    lives = _lives(data, 2)
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    backend = dataclasses.replace(backends.fake_backend(), ledger=ledger, prices=PRICES)
    if broken == "spark":
        backend = dataclasses.replace(backend, spark=_FailingSpark())  # its third call, in the first run
    started: list[str] = []
    real = live.run_live

    def run_live(arm: Any, life: Life, *args: Any, **kwargs: Any) -> ArmRun:
        started.append(life.id)
        return real(arm, life, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "run_live", run_live)
        with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
            assert backend.vm is not None
            if broken != "spark":
                backend.vm.live[session.session].failing = {broken}  # every try: past retry's three
            with pytest.raises(live.RunsFailed, match="weren't saved"):
                batch.run_lives(data, lives, ("M",), session, backend, lambda _line: None, parallel=1)
    failed = {record["life"] for record in data.failed()}
    assert lives[0].id in failed and all(data.run(life.id, "M") is None for life in lives if life.id in failed)
    assert all(data.run(life.id, "M") is not None for life in lives if life.id not in failed)
    assert started == [life.id for life in lives]  # the batch went on past the failed run
    assert ledger.open_holds() == ()  # what the failed run held is settled


def test_a_transient_failure_recovers_and_its_run_is_saved(tmp_path: Path) -> None:
    data = DataDir(tmp_path)
    lives = _lives(data, 1)
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    spark = _FlakySpark("Baseten chat with spark-open returned HTTP 429")
    backend = dataclasses.replace(backends.fake_backend(), ledger=ledger, prices=PRICES, spark=spark)
    with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
        assert backend.vm is not None
        backend.vm.live[session.session].flaky = {"think": 1, "world": 1}  # Glimmer answers 503 once each
        (run,) = batch.run_lives(data, lives, ("M",), session, backend, lambda _line: None, parallel=1)
        assert backend.vm.live[session.session].flaky == {"think": 0, "world": 0}
    assert data.run(lives[0].id, "M") == run and data.failed() == []
    nights = SpendJournal(data.journal(f"{lives[0].id}.M")).jobs()
    calls = [call for night in nights for call in night.calls]
    assert [call.failed for call in calls].count(True) == 1 and spark.tries == len(calls)  # each try journaled
    assert ledger.open_holds() == ()


def test_a_hard_failure_at_parallel_2_fails_only_its_own_run(tmp_path: Path) -> None:
    data = DataDir(tmp_path)
    lives = _lives(data, 3)
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    backend = dataclasses.replace(backends.fake_backend(), ledger=ledger, prices=PRICES)
    real = live.run_live
    broken = lives[1].id

    def run_live(arm: Any, life: Life, session: Any, spark_chat: Any, **kwargs: Any) -> ArmRun:
        chat = _FlakySpark(*["Baseten chat with spark-open returned HTTP 400"] * 99) if life.id == broken else spark_chat
        return real(arm, life, session, chat, **kwargs)

    lines: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "run_live", run_live)
        with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
            with pytest.raises(live.RunsFailed, match=f"1 failed and weren't saved \\(M on {broken}\\)"):
                batch.run_lives(data, lives, ("M",), session, backend, lines.append, parallel=2)
    assert data.run(broken, "M") is None and [record["life"] for record in data.failed()] == [broken]
    assert all(data.run(life.id, "M") is not None for life in lives if life.id != broken)  # the others saved
    assert any(line.startswith(f"FAILED M on {broken}") for line in lines) and ledger.open_holds() == ()

    with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:  # the rerun
        rerun = batch.run_lives(data, lives, ("M",), session, backend, lines.append, parallel=2)
    assert len(rerun) == 3 and data.run(broken, "M") is not None
    assert sum(line.startswith("ran M") for line in lines) == 3  # two the first time, the failed one now


def test_the_batch_stops_before_a_run_the_session_cannot_finish(tmp_path: Path) -> None:
    data = DataDir(tmp_path)
    lives = _lives(data, 1)
    backend = backends.fake_backend()
    started: list[str] = []

    def run_live(*args: Any, **kwargs: Any) -> ArmRun:
        started.append("run")
        raise AssertionError("no run starts")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "run_live", run_live)
        with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
            with pytest.raises(colab_vllm.SessionEnding, match="time"):  # not enough left for a whole life-arm
                batch.run_lives(data, lives, ("M",), session, backend, lambda _line: None, parallel=1,
                                run_s=fakes.SERVE_S * 2)
            assert backend.vm is not None
            backend.vm.live[session.session].healthy = False
            with pytest.raises(colab_vllm.SessionEnding, match="answer"):  # vLLM stopped answering
                batch.run_lives(data, lives, ("M",), session, backend, lambda _line: None, parallel=1)
            backend.vm.live[session.session].healthy = True
            backend.vm.dropped.add(session.session)
            with pytest.raises(colab_vllm.SessionEnding, match="tunnel"):  # the tunnel dropped
                batch.run_lives(data, lives, ("M",), session, backend, lambda _line: None, parallel=1)
    assert started == []


def test_no_spark_open_call_once_the_session_stopped() -> None:
    chat = fakes.FakeSparkChat()
    spark = live.SparkCalls(chat, None)
    with fakes.serving() as session:
        live.arm_models(session, spark)
    with pytest.raises(live.RunFailed, match="stopped"):
        spark([{"role": "user", "content": "a question"}], 16)
    assert chat.calls == 0  # refused before any spend


def test_an_interrupted_batch_starts_nothing_more_and_returns_at_once() -> None:
    started: list[int] = []
    gate = threading.Event()

    def item(n: int) -> Callable[[], int]:
        def run() -> int:
            started.append(n)
            gate.wait(10)
            return n
        return run

    def interrupted(_futures: object) -> Iterator[object]:
        raise KeyboardInterrupt  # a signal's SystemExit, or Ctrl-C, while the batch waits

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "as_completed", interrupted)
        with pytest.raises(KeyboardInterrupt):
            batch.in_parallel([item(n) for n in range(6)], 2)  # returns without waiting on the gate
    gate.set()
    for thread in threading.enumerate():
        if thread.name.startswith("apd"):
            thread.join(10)
    assert set(started) <= {0, 1}  # only what was running when it was interrupted
