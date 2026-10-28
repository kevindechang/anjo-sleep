# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""What keeps a live APD command honest when things go wrong (Task 17 review): a run with any
failed model call is never saved, a batch stops cleanly before a run its session can't finish,
nothing calls Spark Open once the session stopped, and an interrupted batch starts nothing more.
Fakes only: no model, no VM, no network."""
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


@pytest.mark.parametrize("broken", ["think", "world", "spark"])
def test_a_run_with_any_failed_call_is_never_saved_and_the_batch_stops(tmp_path: Path, broken: str) -> None:
    data = DataDir(tmp_path)
    lives = _lives(data, 2)
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    backend = dataclasses.replace(backends.fake_backend(), ledger=ledger, prices=PRICES)
    if broken == "spark":
        backend = dataclasses.replace(backend, spark=_FailingSpark())
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
                backend.vm.live[session.session].failing = {broken}
            with pytest.raises(live.RunFailed, match="failed"):
                batch.run_lives(data, lives, ("M",), session, backend, lambda _line: None, parallel=1)
    assert all(data.run(life.id, "M") is None for life in lives)  # never saved, so never judged
    assert started == [lives[0].id]  # and the batch stopped: the next run never began
    assert ledger.open_holds() == ()  # what the failed run held is settled


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
