# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""APD's Spark Open spend under kills, failures and parallel runs (Task 17): every metered call is
journaled before it returns, a killed process's open night settles at its journal plus one call's
worst case, a rerun redoes only what didn't finish and records what it threw away, and the ledger,
the journals and the store hold under threads. Fakes only: no model, no VM, no network."""
from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger, Reservation
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import batch, fakes, generate, journal, live, pilot
from anjo.sleep.mind.apd.journal import SpendJournal
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import ArmRun, Life
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import spark_usd, spark_worst_usd, usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.stops import BudgetStop

PRICES = Prices(spark_in_per_m=0.6, spark_out_per_m=2.4, train_gpu_per_hour=1.5, serve_gpu_per_hour=2.61)
ONE_CALL = spark_worst_usd(PRICES, 1)
NIGHT = spark_worst_usd(PRICES, MIND_SPARK_CALLS)
MESSAGES = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]


def _events(ledger: Ledger) -> list[dict[str, Any]]:
    return read_records(ledger.path, PLAIN)


def _settled(ledger: Ledger) -> dict[str, float]:
    return {e["job"]: float(e["amount"]) for e in _events(ledger) if e["op"] == "settle"}


def _held(ledger: Ledger) -> dict[str, float]:
    return {e["job"]: float(e["amount"]) for e in _events(ledger) if e["op"] == "reserve"}


def _backend(ledger: Ledger) -> backends.Backend:
    return dataclasses.replace(backends.fake_backend(), ledger=ledger, prices=PRICES)


def _lives(data: DataDir, split: str, count: int) -> tuple[Life, ...]:
    specs = generate.development_specs() if split == "development" else generate.draw_held_out("spend-test")
    for spec in specs[:count]:
        made = fakes.fake_life(spec)
        data.save_life(made.life, made.problems)
    return data.lives("development" if split == "development" else "held_out")[:count]


# ── The journal ───────────────────────────────────────────────────────────────


def test_every_call_journaled_before_it_returns_and_a_night_settles_at_its_journal(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    spend = SpendJournal(tmp_path / "spend.jsonl")
    seen: list[int] = []

    def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        seen.append(len(spend.jobs()[0].calls) if spend.jobs() else -1)  # what the journal held at the call
        return ChatResult(text="{}", tool_calls=(), model="spark-open", input_tokens=1000, output_tokens=100,
                          finish_reason="stop")

    calls = live.SparkCalls(chat, PRICES, ledger, "night", journal=spend)
    for _ in range(3):
        calls(MESSAGES, 10)
    assert seen == [0, 1, 2]  # each call was journaled, with its tokens, before the next one started
    calls.close()
    (job,) = spend.jobs()
    assert job.closed and job.usd == pytest.approx(3 * spark_usd(PRICES, 1000, 100))
    assert _settled(ledger) == {job.job: pytest.approx(job.usd)} and ledger.open_holds() == ()


def test_recover_settles_an_open_night_at_its_journal_plus_one_calls_worst_case(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    spend = SpendJournal(tmp_path / "spend.jsonl")
    calls = live.SparkCalls(fakes.FakeSparkChat(), PRICES, ledger, "dies", journal=spend)
    calls(MESSAGES, 10)
    calls(MESSAGES, 10)  # then the process dies: no close
    (open_,) = ledger.open_holds()
    priced = spend.jobs()[0].usd
    assert journal.recover(ledger, spend, PRICES) == pytest.approx(priced + ONE_CALL)
    assert _settled(ledger) == {open_.job: pytest.approx(priced + ONE_CALL)} and ledger.open_holds() == ()
    assert spend.spent() == pytest.approx(priced + ONE_CALL)
    assert journal.recover(ledger, spend, PRICES) == 0.0  # once


def test_a_night_waits_for_room_another_night_holds_and_stops_when_none_will_come(tmp_path: Path) -> None:
    caps = {**CAPS, "baseten_mind_usd": NIGHT * 1.5}
    ledger = Ledger(tmp_path / "budget.jsonl", caps=caps)
    first = live.SparkCalls(fakes.FakeSparkChat(), PRICES, ledger, "first")
    first(MESSAGES, 10)
    waits: list[float] = []

    def wait(seconds: float) -> None:
        waits.append(seconds)
        first.close()  # the other night ends while this one waits

    second = live.SparkCalls(fakes.FakeSparkChat(), PRICES, ledger, "second", wait=wait)
    second(MESSAGES, 10)
    assert len(waits) == 1 and second.calls == 1
    second.close()
    room = ledger.remaining("baseten_mind_usd")
    starved = Ledger(tmp_path / "starved.jsonl", caps={**CAPS, "baseten_mind_usd": min(room, NIGHT / 2)})
    with pytest.raises(BudgetStop):  # nothing else holds: no room is coming, so it stops at once
        live.SparkCalls(fakes.FakeSparkChat(), PRICES, starved, "never", wait=wait)(MESSAGES, 10)
    assert len(waits) == 1


def _sleep_holds(ledger: Ledger, amount: float) -> Reservation:
    """A Sleep night's open hold on its own Baseten pool."""
    found = ledger.reserve("night:sleep:a1", ("baseten_sleep_usd",), usd(amount))
    assert found is not None
    return found


def test_a_night_waits_up_to_ten_minutes_on_sleeps_open_holds_then_stops_cleanly(tmp_path: Path) -> None:
    """Sleep has priority on the shared Baseten credit, so its open holds can leave MIND's night
    no room though MIND's own cap has plenty: the night waits for them, at most ``SHARED_WAIT_S``."""
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    total = CAPS["baseten_sleep_usd"] + CAPS["baseten_mind_usd"]
    sleep_night = _sleep_holds(ledger, total - NIGHT / 2)
    waits: list[float] = []

    def wait(seconds: float) -> None:
        waits.append(seconds)
        sleep_night.settle(1.0)  # Sleep's night ends while MIND's waits

    calls = live.SparkCalls(fakes.FakeSparkChat(), PRICES, ledger, "waits", wait=wait)
    calls(MESSAGES, 10)
    assert waits == [live.ROOM_POLL_S] and calls.calls == 1
    calls.close()

    stuck = Ledger(tmp_path / "stuck.jsonl", caps=CAPS)
    _sleep_holds(stuck, total - NIGHT / 2)  # never ends
    waits.clear()
    never = live.SparkCalls(fakes.FakeSparkChat(), PRICES, stuck, "stuck", wait=waits.append)
    with pytest.raises(BudgetStop, match="Sleep's open Baseten holds left MIND's night no room for 10 minutes"):
        never(MESSAGES, 10)
    assert sum(waits) == live.SHARED_WAIT_S and [held.job for held in stuck.open_holds()] == ["night:sleep:a1"]


@pytest.mark.parametrize("why", ["mind_cap", "settled"])
def test_a_night_stops_at_once_when_sleeps_holds_are_not_what_blocks_it(tmp_path: Path, why: str) -> None:
    total = CAPS["baseten_sleep_usd"] + CAPS["baseten_mind_usd"]
    if why == "mind_cap":  # MIND's own cap binds: nothing Sleep releases makes room
        ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "baseten_mind_usd": NIGHT / 2})
        _sleep_holds(ledger, 1.0)
    else:  # Sleep's spend is settled: its small open hold can't free enough
        ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
        _sleep_holds(ledger, total - NIGHT / 2).settle(total - NIGHT / 2)
        assert ledger.reserve("night:sleep2:a1", ("baseten_sleep_usd",), usd(NIGHT / 8)) is not None
    waits: list[float] = []
    with pytest.raises(BudgetStop):
        live.SparkCalls(fakes.FakeSparkChat(), PRICES, ledger, "never", wait=waits.append)(MESSAGES, 10)
    assert waits == []


def test_a_budget_stop_ends_the_command_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.mind.apd import __main__ as cli

    monkeypatch.setenv("ANJO_SLEEP_MIND_SPARK", "baseten")  # MIND on Baseten shares its credit with Sleep
    lines: list[str] = []
    assert cli.main(["--fake", "--data", str(tmp_path), "--only", "1", "generate"], out=lines.append) == 0
    ledger = Ledger(tmp_path / "fake-budget.jsonl", caps=CAPS)
    _sleep_holds(ledger, ledger.remaining("baseten_sleep_usd") - NIGHT / 2)  # Sleep's night holds the rest
    lines.clear()
    assert cli.main(["--fake", "--data", str(tmp_path), "--only", "1", "run"], out=lines.append) == cli.STOPPED
    assert any(line.startswith("Stopped: Sleep's open Baseten holds") for line in lines)
    assert [held.job for held in ledger.open_holds()] == ["night:sleep:a1"]  # MIND's own all settled


def _nap(_seconds: float) -> None:
    threading.Event().wait(0.001)


def test_ledger_journals_and_store_hold_under_threads(tmp_path: Path) -> None:
    cap = NIGHT * 5.5
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "baseten_mind_usd": cap})
    spend = SpendJournal(tmp_path / "shared.jsonl")
    data = DataDir(tmp_path / "data")
    peak = [0.0]
    lock = threading.Lock()

    def worker(n: int) -> None:
        calls = live.SparkCalls(fakes.FakeSparkChat(), PRICES, ledger, f"t{n}", journal=spend,
                                wait=_nap)
        for _ in range(4):
            calls(MESSAGES, 10)
            with lock:
                peak[0] = max(peak[0], ledger.used("baseten_mind_usd"))
        calls.close()
        data.save_run(ArmRun(arm="B", life_id=f"life-{n}", reports=(), tokens=(1,), contexts={}))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] <= cap + 1e-9 and ledger.open_holds() == ()
    assert len(spend.jobs()) == 16 and all(len(job.calls) == 4 and job.closed for job in spend.jobs())
    assert ledger.used("baseten_mind_usd") == pytest.approx(spend.spent())
    assert all(data.run(f"life-{n}", "B") is not None for n in range(16))
    assert not list((data.root / "runs").glob(".*"))  # no scratch file left behind


# ── Kill-resume ───────────────────────────────────────────────────────────────


def test_a_sigkilled_child_settles_its_journal_plus_one_calls_worst_case(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    data = DataDir(tmp_path / "data")
    (life,) = _lives(data, "held_out", 1)
    backend = _backend(ledger)
    # The second night's CONCLUDE, after its resolver call: the killed child has a night's Spark Open
    # hold open (nothing else now calls Spark Open before it).
    kill_at = 24
    with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
        (result,) = pilot.kill_lives(data, [life], session, kill_at, backend, lambda _line: None, parallel=1)
    assert result.digest_equal and result.conclusions_agree
    held, settled = _held(ledger), _settled(ledger)
    kill_holds = {job: amount for job, amount in held.items() if job.startswith(f"apd:kill:{life.id}:")}
    assert kill_holds and all(amount == pytest.approx(NIGHT) for amount in kill_holds.values())  # night by night
    assert ledger.open_holds() == ()
    spend = SpendJournal(data.journal(f"kill-{life.id}"))
    (killed,) = [job for job in spend.jobs() if job.recovered]
    assert settled[killed.job] == pytest.approx(min(sum(c.usd for c in killed.calls) + ONE_CALL, NIGHT))
    others = [job for job in spend.jobs() if not job.recovered]
    assert others and all(settled.get(job.job, 0.0) == pytest.approx(job.usd) for job in others)
    assert data.kill_result(life.id) is not None  # done: a rerun skips it
    lines: list[str] = []
    with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
        pilot.kill_lives(data, [life], session, kill_at, backend, lines.append, parallel=1)
    spark_jobs = {job for job in _settled(ledger) if job.startswith("apd:")}
    assert any("already" in line for line in lines) and spark_jobs == {job for job in settled if job.startswith("apd:")}


# ── Never redo completed work ─────────────────────────────────────────────────


class _Crash(RuntimeError):
    pass


def test_a_rerun_after_a_mid_batch_failure_redoes_only_the_unfinished_life_arms(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    data = DataDir(tmp_path / "data")
    lives = _lives(data, "development", 3)
    backend = _backend(ledger)
    target = threading.local()
    doomed = (lives[1].id, "M")
    real = live.run_live

    def run_live(arm: Any, life: Life, *args: Any, **kwargs: Any) -> ArmRun:
        target.run = (life.id, arm)
        return real(arm, life, *args, **kwargs)

    inner, doomed_calls = fakes.FakeSparkChat(), [0]

    def spark(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        if getattr(target, "run", None) == doomed:
            doomed_calls[0] += 1
            if doomed_calls[0] > 5:
                raise _Crash("the process dies mid-run")
        return inner(messages, max_tokens)

    first: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "run_live", run_live)
        with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
            with pytest.raises(_Crash):
                pilot.run_lives(data, lives, pilot.DEV_ARMS, session, dataclasses.replace(backend, spark=spark),
                                first.append, parallel=1)
    finished = {(life.id, arm) for life in lives for arm in pilot.DEV_ARMS if data.run(life.id, arm)}
    assert doomed not in finished and finished
    assert ledger.open_holds() == ()  # the failed night settled what it priced
    again: list[str] = []
    with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
        pilot.run_lives(data, lives, pilot.DEV_ARMS, session, backend, again.append, parallel=4)
    rerun = {(life.id, arm) for life in lives for arm in pilot.DEV_ARMS if f"ran {arm} on {life.id}:" in "\n".join(again)}
    assert rerun == {(life.id, arm) for life in lives for arm in pilot.DEV_ARMS} - finished
    (discarded,) = data.discarded()
    assert (discarded["life"], discarded["arm"]) == doomed and discarded["usd"] > 0
    spent = sum(SpendJournal(path).spent() for path in data.spend.rglob("*.jsonl") if path.name != "discarded.jsonl")
    assert ledger.open_holds() == () and ledger.used("baseten_mind_usd") == pytest.approx(spent)


def test_runs_go_in_parallel_each_life_arm_in_order(tmp_path: Path) -> None:
    data = DataDir(tmp_path / "data")
    lives = _lives(data, "development", 4)
    running, peak = [0], [0]
    lock = threading.Lock()
    real = live.run_live

    def run_live(*args: Any, **kwargs: Any) -> ArmRun:
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        try:
            threading.Event().wait(0.05)
            return real(*args, **kwargs)
        finally:
            with lock:
                running[0] -= 1

    backend = backends.fake_backend()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(batch, "run_live", run_live)
        with pilot.serving(backend, fakes.SETTINGS, fakes.SERVE_S, lambda _line: None) as session:
            runs = pilot.run_lives(data, lives, pilot.DEV_ARMS, session, backend, lambda _line: None, parallel=3)
    assert peak[0] == 3 and len(runs) == 8
    assert [(r.life_id, r.arm) for r in runs] == [(life.id, arm) for life in lives for arm in pilot.DEV_ARMS]
