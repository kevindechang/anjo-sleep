# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A cheap night: practice's scenes and candidates and the morning run's arms and judge send their
model calls side by side, up to ``ANJO_SLEEP_CONCURRENCY`` at once, since vLLM batches calls that
arrive together almost for free and an idle GPU still bills. The results stay the serial run's, in
its order, the meters stay exact, and a failure stops the rest and settles its hold. Fakes with a
little injected latency, so the overlap is real and the tests stay fast and offline."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep import parallel
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import Ledger
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.meter import SpendMeter, WonderMeter
from anjo.sleep.practice.full import FullPractice
from anjo.sleep.probes.arms import ArmDeps, reply_seed
from anjo.sleep.probes.morning import ARM_ORDER, Morning, run_morning
from anjo.sleep.probes.score import Verdict
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.serving import config
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeRegistry, StoreSpy, ckpt, probe, taste
from anjo.sleep.tests.test_adapter_load import ADAPTER as LOADED, MESSAGES, NIGHT as LOAD_NIGHT, LiveVllm
from anjo.sleep.tests.test_full_practice import ADAPTER, AT, NIGHT, _evidence, _scenes, candidates_of
from anjo.sleep.types import Probe, ProbeKind

DELAY_S = 0.004
MORNING_NIGHT = date(2026, 11, 3)
PRICES = Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)


def _digit(text: str, mod: int) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % mod


class Recorder:
    """Every call the fakes answer, in the order they started, with how many ran at once and on
    which threads. Each call sleeps ``delay`` so calls really overlap."""

    def __init__(self, delay: float = DELAY_S) -> None:
        self.delay = delay
        self.log: list[tuple[Any, ...]] = []
        self.threads: set[int] = set()
        self.most = 0
        self._now = 0
        self._lock = threading.Lock()

    @contextmanager
    def calling(self, *what: Any) -> Iterator[None]:
        with self._lock:
            self._now += 1
            self.most = max(self.most, self._now)
            self.threads.add(threading.get_ident())
            self.log.append(what)
        try:
            time.sleep(self.delay)
            yield
        finally:
            with self._lock:
                self._now -= 1


def _result(text: str, finish: str = "stop") -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model="m", input_tokens=10, output_tokens=5, finish_reason=finish)


# ── practice ─────────────────────────────────────────────────────────────────


class SlowSpark:
    """Spark Open answering by content only: the scenes, the teacher, and a critic whose marks come
    from each candidate's words, never from the order calls arrive in."""

    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        assert model == SPARK_OPEN
        content = messages[-1]["content"]
        if content.startswith("# scenes_v1"):
            with self.rec.calling("scenes"):
                return _result(json.dumps({"scenes": _scenes()}))
        if content.startswith("# critic_v1"):
            found = candidates_of(content)
            with self.rec.calling("critic", content.split("<message>", 1)[-1][:60]):
                return _result(json.dumps({
                    "scores": {label: _digit(text, 4) for label, text in found.items()},
                    "misses_constraint": [label for label, text in found.items() if _digit(text, 3) == 0]}))
        with self.rec.calling("teacher", content):
            return _result(f"TEACHER on {content[:40]}")


class SlowGlimmer:
    """Her Glimmer, its reply set by the seed and the message; ``cut`` seeds end at max tokens and
    ``fail`` messages raise."""

    def __init__(self, rec: Recorder, *, cut: Callable[[int], bool] = lambda seed: False,
                 fail: Callable[[str], bool] = lambda message: False) -> None:
        self.rec, self.cut, self.fail = rec, cut, fail

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        assert seed is not None
        message = messages[-1]["content"]
        if self.fail(message):
            with self.rec.calling("glimmer", message, seed):
                pass
            raise ServingError("/v1/completions returned HTTP 500")
        with self.rec.calling("glimmer", message, seed):
            return _result(f"GLIMMER {seed % 9973} on {message[:24]}", "length" if self.cut(seed) else "stop")


def _practice(rec: Recorder, concurrency: int, **glimmer: Any) -> FullPractice:
    return FullPractice(chat=SlowSpark(rec), client=SlowGlimmer(rec, **glimmer), adapter=lambda uid: ADAPTER, seed=7,
                        snapshot=lambda uid, message: (f"C for {message[:12]}", frozenset({SPARK_OPEN})),
                        concurrency=concurrency)


def test_practice_concurrent_matches_serial_output_order() -> None:
    def cut(seed: int) -> bool:
        return seed % 5 == 0  # some candidates are cut, so some scenes keep fewer than three

    serial_rec, parallel_rec = Recorder(), Recorder()
    serial = _practice(serial_rec, 1, cut=cut).run("p1", NIGHT, _evidence(), None, at=AT)
    together = _practice(parallel_rec, 8, cut=cut).run("p1", NIGHT, _evidence(), None, at=AT)
    assert together == serial  # every transcript, target, pair and count, in scene order
    assert [t.scene for t in together.transcripts] == sorted(t.scene for t in together.transcripts)
    assert sorted(map(repr, parallel_rec.log)) == sorted(map(repr, serial_rec.log))  # the very same calls
    assert serial_rec.most == 1 and 1 < parallel_rec.most <= 8  # side by side, never past the setting


# ── the morning run ──────────────────────────────────────────────────────────


class SlowModel:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> ChatResult:
        name = adapter.id if adapter else "base"
        with self.rec.calling("reply", name, seed, json.dumps(messages)):
            return ChatResult(text=f"reply from {name}" + "!" * ((seed or 0) % 5), tool_calls=(), model=name,
                              input_tokens=7, output_tokens=3, finish_reason="stop")


class SlowLogprobs:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    def continuation_logprobs(self, adapter: Any, messages: list[dict[str, str]], continuation: str) -> tuple[float, ...]:
        with self.rec.calling("logprobs", adapter.id if adapter else "base", json.dumps(messages), continuation):
            return (-0.1,) if _digit(json.dumps(messages) + str(adapter), 2) == ("A", "B").index(continuation) else (-2.0,)


class SlowJudge:
    """Judges by the two replies' words alone, in the order shown, so some verdicts flip with the order."""

    def __init__(self, rec: Recorder) -> None:
        self.rec = rec
        self._lock = threading.Lock()
        self.calls = 0

    def verdict(self, p: Probe, first: str, second: str) -> Verdict:
        with self._lock:
            self.calls += 1
        with self.rec.calling("judge", p.id, first, second):
            verdicts: tuple[Verdict, ...] = ("A", "B", "tie")
            return verdicts[_digit(f"{p.id}|{first}|{second}", 3)]


def _probes() -> list[Probe]:
    kinds: tuple[ProbeKind, ...] = ("new_work", "vocab", "tone_hold")
    return [probe(f"q{i}", kinds[i % 3], prompt=f"Question {i} about the ferry cut?") for i in range(9)] + [
        taste("t1"), taste("t2", source="item-2")]


def _morning(tmp_path: Path, rec: Recorder, concurrency: int, ledger: Ledger | None = None) -> tuple[Morning, SlowJudge]:
    judge = SlowJudge(rec)
    deps = ArmDeps(model=SlowModel(rec), adapters=FakeRegistry({"p1": ckpt("p1", 1), "donor": ckpt("donor", 1)}),
                   contexts=FakeContexts(), evidence=lambda uid: StoreSpy([]), logprobs=SlowLogprobs(rec),
                   donors={"p1": "donor"}, until=MORNING_NIGHT)
    probes = _probes()
    return Morning(arms=deps, sealed=lambda uid: probes, judge=judge, prices=PRICES, ledger=ledger,
                   root=tmp_path / "results", concurrency=concurrency), judge


def _score(tmp_path: Path, rec: Recorder, concurrency: int, ledger: Ledger | None = None) -> tuple[Any, SlowJudge]:
    morning, judge = _morning(tmp_path, rec, concurrency, ledger)
    return run_morning("p1", MORNING_NIGHT, ckpt("p1", 1), morning=morning), judge


def test_morning_concurrent_matches_serial_scores(tmp_path: Path) -> None:
    serial_rec, parallel_rec = Recorder(), Recorder()
    serial, serial_judge = _score(tmp_path / "serial", serial_rec, 1)
    together, judge = _score(tmp_path / "parallel", parallel_rec, 8, Ledger(tmp_path / "budget.jsonl"))
    assert json.loads(together.path.read_text()) == json.loads(serial.path.read_text())  # every score, row and CI
    assert (together.replies, together.judge_calls) == (serial.replies, serial.judge_calls) == (
        len(ARM_ORDER) * 11, judge.calls)
    assert sorted(map(repr, parallel_rec.log)) == sorted(map(repr, serial_rec.log))  # 2C and 4 share their contexts
    assert serial_rec.most == 1 and 1 < parallel_rec.most <= 8


def test_concurrency_one_is_serial(tmp_path: Path) -> None:
    rec = Recorder(delay=0.0)
    run = _practice(rec, 1).run("p1", NIGHT, _evidence(), None, at=AT)
    assert rec.threads == {threading.get_ident()} and rec.most == 1  # every call on the caller's thread
    scenes = _scenes()[:16]
    expected: list[tuple[str, ...]] = [("scenes",)]
    for message in scenes:  # as before: three candidates in turn, the teacher, then the critic
        expected += [("glimmer", message), ("glimmer", message), ("glimmer", message), ("teacher", message), ("critic",)]
    assert [entry[:2] if entry[0] != "critic" else ("critic",) for entry in rec.log] == expected
    assert len(run.transcripts) == 16

    morning_rec = Recorder(delay=0.0)
    _score(tmp_path, morning_rec, 1)
    assert morning_rec.threads == {threading.get_ident()} and morning_rec.most == 1
    seeds = [entry[2] for entry in morning_rec.log if entry[0] == "reply"]
    assert seeds == [reply_seed(0, p.id, arm) for arm in ARM_ORDER for p in _probes() if p.kind != "taste_ab"]


def test_the_setting_defaults_to_eight_and_refuses_anything_else() -> None:
    assert parallel.concurrency({}) == parallel.DEFAULT_CONCURRENCY == 8
    assert parallel.concurrency({parallel.CONCURRENCY_ENV: " 1 "}) == 1
    assert parallel.concurrency({parallel.CONCURRENCY_ENV: str(parallel.MAX_CONCURRENCY)}) == parallel.MAX_CONCURRENCY
    for wrong in ("0", "-2", "eight", "2.5", str(parallel.MAX_CONCURRENCY + 1)):
        with pytest.raises(parallel.ConcurrencyError, match=parallel.CONCURRENCY_ENV):
            parallel.concurrency({parallel.CONCURRENCY_ENV: wrong})


# ── meters, holds and failures ───────────────────────────────────────────────


class SlowInt(int):
    """A token count whose arithmetic yields the GIL, so an unguarded read-modify-write loses updates."""

    def __mul__(self, other: Any) -> Any:
        time.sleep(0)
        return int(self) * other


class SlowUsd(float):
    def __radd__(self, other: Any) -> Any:
        time.sleep(0)
        return other + float(self)

    def __rsub__(self, other: Any) -> Any:
        time.sleep(0)
        return other - float(self)


def test_meters_exact_under_concurrency() -> None:
    spend = SpendMeter(spark_in_per_m=1_000_000.0, spark_out_per_m=2_000_000.0)  # $1 and $2 a token: exact sums
    spend.part = "practice"
    wonder = WonderMeter(10**6, 10.0**6)
    charged = ChatResult(text="", tool_calls=(), model="m", input_tokens=SlowInt(3), output_tokens=SlowInt(1),
                         finish_reason="stop")

    def work(_: int) -> None:
        for _ in range(150):
            spend.charge(charged)
            spend.add("gates", 0.5)
            wonder.debit(SlowUsd(0.25))

    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(work, range(24)))
    calls = 24 * 150
    assert spend.parts()["practice"] == calls * (3 * 1.0 + 1 * 2.0)
    assert spend.parts()["gates"] == calls * 0.5
    assert (wonder.calls, wonder.usd, wonder.calls_left, wonder.usd_left) == (calls, calls * 0.25, 10**6 - calls,
                                                                           10.0**6 - calls * 0.25)


def _ours() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(parallel.THREAD_PREFIX)]


def test_failure_in_one_scene_drains_and_settles(tmp_path: Path) -> None:
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.night_fakes import NIGHT as NIGHT_DATE, World, step, summary

    rec = Recorder(delay=0.01)
    broken = _scenes()[2]
    world = World(tmp_path)
    world.plan("p1", NIGHT_DATE)
    spark, glimmer = SlowSpark(rec), SlowGlimmer(rec, fail=lambda message: message == broken)

    def practice(chat: Any, night: date) -> FullPractice:
        return FullPractice(chat=chat, client=glimmer, adapter=world.registry.last_promoted, seed=7, concurrency=8,
                            snapshot=lambda uid, message: (None, None))

    def chat(model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        if messages[-1]["content"].startswith("# brief_v1"):
            return world.spark(model, messages, tools, timeout_s, temperature=temperature)
        return spark(model, messages, tools, timeout_s, temperature=temperature)

    ledger = Ledger(tmp_path / "budget.jsonl")
    result = run_night_batch_with(world.deps(spark_chat=chat, practice=practice, ledger=ledger), {"p1": NIGHT_DATE})["p1"]
    assert result.status == "failed" and result.failed_at == "practice"
    assert step(result.log_path, "practice")["status"] == "failed" and summary(result.log_path)["status"] == "failed"
    assert _ours() == []  # every thread ended before the step did
    assert ledger.open_holds() == ()
    spark_calls = [entry for entry in rec.log if entry[0] in ("scenes", "teacher", "critic")]
    [settled] = [amount for job, amount in ledger.settlements().items() if job.startswith("night:practice:")]
    assert settled == pytest.approx(len(spark_calls) * (10 * 0.2 + 5 * 0.6) / 1e6, rel=1e-9)  # every call it made
    started = {entry[1] for entry in rec.log if entry[0] == "glimmer"}
    assert broken in started and len(started) < 16  # scenes not yet begun never start


def test_vllm_client_safe_across_threads() -> None:
    class SlowLoad(LiveVllm):
        def _load(self, body: dict[str, Any]) -> httpx.Response:
            time.sleep(0.02)  # a load takes a while: other threads ask for the same adapter meanwhile
            return super()._load(body)

    fake = SlowLoad()
    client = VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake),
                        adapters_dir=config.ADAPTERS_DIR)

    def ask(index: int) -> str:
        dated = client.on(LOAD_NIGHT) if index % 2 else client
        if index % 3 == 0:
            return str(dated.continuation_logprobs(LOADED, MESSAGES, " A"))
        return dated.complete(LOADED, MESSAGES, temperature=1.0, seed=index).text

    with ThreadPoolExecutor(max_workers=16) as pool:
        answers = list(pool.map(ask, range(48)))
    assert all(a == "hello" for i, a in enumerate(answers) if i % 3)
    assert len(fake.loads) == 1  # one load, however many threads first named the adapter at once
    assert LOADED.id in fake.loaded
