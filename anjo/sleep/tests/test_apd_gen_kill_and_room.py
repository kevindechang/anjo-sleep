# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""APD's side-by-side generation under a kill and a full pool (MIND's data): a kill stops every
new Spark Open call and attempt at once, and the command keeps MIND's lock until every worker has
ended; a hold that finds no room waits for the attempts in flight to settle, and stops the run only
when nothing is in flight. Every model is a fake: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import fakes, gen_batch, generate
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices
from anjo.sleep.stops import BudgetStop

WRITE = "# life_v1"
WORKER = "apd-gen"


def _seed(messages: list[dict[str, Any]]) -> int | None:
    if not str(messages[0]["content"]).startswith(WRITE):
        return None
    return int(json.loads(messages[-1]["content"])["variation"])


class Blocks:
    """A life chat whose first writer call for ``seed`` waits for ``go``; every call that starts
    once ``killed`` is set is recorded in ``after``."""

    def __init__(self, seed: int) -> None:
        self.seed, self.go, self.killed = seed, threading.Event(), threading.Event()
        self.after: list[str] = []
        self.writes: Counter[int] = Counter()
        self.blocked = False
        self.inner = fakes.FakeLifeChat()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if self.killed.is_set():
            self.after.append(str(messages[0]["content"]).splitlines()[0])
        seed = _seed(messages)
        if seed is not None:
            self.writes[seed] += 1
        if seed == self.seed and not self.blocked:
            self.blocked = True
            self.go.wait(timeout=10)
        return self.inner(model, messages, tools, timeout_s, temperature=temperature)


def _workers() -> list[str]:
    return [thread.name for thread in threading.enumerate() if thread.name.startswith(WORKER) and thread.is_alive()]


def test_kill_stops_new_calls_and_holds_the_lock_until_workers_end(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: two lives at once; the second's first writer call is in flight when Ctrl-C comes,
    # as the first life's line prints, and returns a moment later
    specs = generate.development_specs()[:2]
    chat = Blocks(specs[1].seed)
    real_backend, real_swept = cli._fake_backend, cli._swept

    def backend(data: DataDir, provider: Any) -> backends.Backend:
        return dataclasses.replace(real_backend(data, provider), life_chat=chat)

    at_unlock: list[list[str]] = []

    @contextmanager
    def swept(data: DataDir, found: backends.Backend) -> Iterator[Any]:
        with real_swept(data, found) as done:
            try:
                yield done
            finally:  # MIND's lock is released right after this
                at_unlock.append(_workers())

    def out(line: str) -> None:
        if line.startswith("generated "):
            chat.killed.set()
            threading.Timer(0.3, chat.go.set).start()
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_fake_backend", backend)
    monkeypatch.setattr(cli, "_swept", swept)
    data = tmp_path / "pilot"

    # Act
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--fake", "--data", str(data), "--only", "2", "generate", "--parallel", "2"], out=out)

    # Assert: no worker outlived the lock, none made a call after the kill, the second life tried
    # once and was not saved, and every hold settled
    assert at_unlock == [[]]
    assert chat.after == []
    assert chat.writes[specs[1].seed] == 1
    assert [life.seed for life in DataDir(data).lives("development")] == [specs[0].seed]
    assert RunLedger(data / "fake-budget.jsonl").open_holds() == ()
    jobs = [str(event["job"]) for event in _events(data / "fake-budget.jsonl") if event["op"] == "reserve"]
    assert [job for job in jobs if f":{specs[1].seed}:" in job] == [
        f"apd:gen:development:{specs[1].seed}:0:spark:a1", f"apd:gen:development:{specs[1].seed}:0:muse:a1"]


def test_a_kill_cancels_what_has_not_started_and_waits_for_what_has() -> None:
    # Arrange: one at a time; the first item is running when the command stops
    started, go, ran = threading.Event(), threading.Event(), []

    def first() -> str:
        started.set()
        go.wait(timeout=10)
        ran.append("first")
        return "first"

    def second() -> str:
        ran.append("second")
        return "second"

    # Act
    with pytest.raises(KeyboardInterrupt), gen_batch.Parallel(1) as pool:
        pool.in_order([first, second])
        started.wait(timeout=10)
        threading.Timer(0.2, go.set).start()
        raise KeyboardInterrupt

    # Assert: the running item ended before the pool let go; the queued one never started
    assert pool.killed.is_set()
    assert ran == ["first"]
    assert _workers() == []


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Order:
    """A life chat that records which life each writer call is for."""

    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.inner = fakes.FakeLifeChat()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        seed = _seed(messages)
        if seed is not None:
            self.seeds.append(seed)
        return self.inner(model, messages, tools, timeout_s, temperature=temperature)


def _held(tmp_path: Path, chat: Any, spark_in: float, spark_out: float) -> backends.Backend:
    return dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=chat,
                               ledger=RunLedger(tmp_path / "budget.jsonl"),
                               prices=Prices(spark_in, spark_out, 1.5, 2.61), rates=cli.FAKE_RATES)


def test_reserve_waits_for_room_instead_of_stopping(tmp_path: Path) -> None:
    # Arrange: two lives whose attempts each hold more than half the Fireworks pool's cap
    specs = generate.development_specs()[1:3]
    chat = Order()
    backend = _held(tmp_path, chat, 2.25, 6.75)  # holds of about $13.5 and $18.5 against the $20 cap
    held = [backends.gen_holds(backend.prices, backend.rates, FIREWORKS.pool, backends.life_calls(s.prevalence))
            for s in specs]
    cap = CAPS[FIREWORKS.pool]
    assert all(cap / 2 < h[FIREWORKS.pool] <= cap for h in held)
    lines: list[str] = []

    # Act
    made = gen_batch.generate_lives(DataDir(tmp_path / "data"), specs, backend, lines.append, parallel=2)

    # Assert: the second waited for the first to settle, then ran; nothing stopped
    assert len(made) == 2
    first = chat.seeds[0]
    assert chat.seeds == [first] * chat.seeds.count(first) + [s for s in chat.seeds if s != first]
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_a_reserve_with_nothing_in_flight_still_stops_the_run(tmp_path: Path) -> None:
    backend = _held(tmp_path, Order(), 40.0, 120.0)  # one attempt alone passes the cap
    with pytest.raises(BudgetStop):
        gen_batch.generate_lives(DataDir(tmp_path / "data"), generate.development_specs()[1:3], backend,
                                 lambda line: None, parallel=2)
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()
