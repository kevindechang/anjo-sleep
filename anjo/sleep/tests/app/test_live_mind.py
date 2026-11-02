# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""MIND's live pass after a judge's reply: judges only, capped a day, never during a night, one at a
time with a rerun coalesced, its record encrypted and free of conversation words. Extraction and the
MIND step are fakes: no Spark Open, Glimmer or ledger call."""
from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.app import judge_cohort, live_mind
from anjo.sleep.mind.inquiry.steps import Budget, RunStart, Step, WorkspaceEvent
from anjo.sleep.model_api import ModelApiError

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
JOINED = NOW - timedelta(hours=1)


class FakeMind:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.noticed: list[int] = []
        self.budgets: list[Budget] = []

    def notice_live(self, call: Any, fresh: list[Any], budget: Budget) -> dict[str, Any]:
        if self.fail:
            raise ModelApiError("down")
        assert call.night == call.as_of.date() == NOW.date() and call.actions_left == 0
        self.noticed.append(len(fresh))
        self.budgets.append(budget)
        return {"new": len(fresh), "questions_opened": 1}

    def results(self, call: Any) -> tuple[Any, ...]:
        return ()

    def questions(self, call: Any) -> tuple[Any, ...]:
        return ()


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("ANJO_SLEEP_PRICES", "0.1,0.4,1.0,1.95")
    judge_cohort.add_judge("judge-1", now=JOINED)
    return "judge-1"


def ports(mind: FakeMind, windows: list[tuple[datetime, datetime]], *, glimmer: bool = False,
          night: bool = False, spent: int = 0) -> live_mind.PassPorts:
    def extract(_uid: str, start: datetime, end: datetime) -> list[Any]:
        windows.append((start, end))
        return []

    return live_mind.PassPorts(extract=extract, mind=lambda _g: mind, glimmer_up=lambda: glimmer,
                               night_running=lambda: night,
                               budget=lambda _uid, _day: Budget(tokens_used=spent, token_cap=12_000))


def test_a_pass_reads_what_was_said_since_the_last_one_and_records_codes_only(judge: str) -> None:
    from anjo.sleep.app.storage import sleep_dir

    windows: list[tuple[datetime, datetime]] = []
    mind = FakeMind()
    done = live_mind.run_pass(judge, ports=ports(mind, windows), now=NOW)
    assert done["state"] == "ok" and done["thinker"] == "spark-open"
    assert done["notice"] == {"new": 0, "questions_opened": 1}
    assert done["choice"]["action"] is None and done["choice"]["considered"] == []
    later = NOW + timedelta(minutes=5)
    live_mind.run_pass(judge, ports=ports(mind, windows, glimmer=True), now=later)
    assert windows == [(JOINED, NOW), (NOW, later)]
    assert live_mind.extracted_until(judge) == later
    record = live_mind.load_record(judge)
    assert record is not None and record["thinker"] == "glimmer" and record["passes"] == 2
    raw = (sleep_dir(judge) / live_mind.RECORD_FILE).read_text()
    assert "questions_opened" not in raw  # encrypted at rest


def test_only_judges_and_never_during_a_night(judge: str) -> None:
    windows: list[tuple[datetime, datetime]] = []
    assert live_mind.run_pass("someone-else", ports=ports(FakeMind(), windows), now=NOW) == {"state": "not_a_judge"}
    assert live_mind.run_pass(judge, ports=ports(FakeMind(), windows, night=True), now=NOW) == {"state": "night_running"}
    assert windows == [] and live_mind.load_record(judge) is None


def test_passes_are_capped_a_day(judge: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(judge_cohort.LIVE_PER_DAY_ENV, "1")
    windows: list[tuple[datetime, datetime]] = []
    assert live_mind.run_pass(judge, ports=ports(FakeMind(), windows), now=NOW)["state"] == "ok"
    assert live_mind.run_pass(judge, ports=ports(FakeMind(), windows), now=NOW) == {
        "state": "capped", "reason": "live_spent_today"}


def test_a_pass_gets_its_own_budget_and_skips_once_it_is_spent(judge: str) -> None:
    mind, windows = FakeMind(), list[tuple[datetime, datetime]]()
    assert live_mind.run_pass(judge, ports=ports(mind, windows, spent=4000), now=NOW)["state"] == "ok"
    assert mind.budgets == [Budget(tokens_used=4000, token_cap=12_000)]
    assert live_mind.run_pass(judge, ports=ports(mind, windows, spent=12_000), now=NOW) == {"state": "budget_spent"}
    assert len(mind.budgets) == 1 and len(windows) == 1  # nothing extracted, no pass claimed


def _start(night: Any, decider: str | None) -> RunStart:
    from anjo.sleep.mind.inquiry import kernel
    from anjo.sleep.mind.inquiry.types import FULL_RULES

    return RunStart(night, NOW, "notice", FULL_RULES, "priority", None, Budget(), False, kernel.DENYLIST_VERSION,
                    decider=decider)


def _event(seq: int, entry: Any, tokens: int = 0) -> WorkspaceEvent:
    code = None if entry is not None else "malformed"  # a thinker reply that held no operation
    return WorkspaceEvent(Step(seq, NOW, entry, code=code, tokens=tokens), "before", "after")


def test_the_live_budget_counts_only_todays_live_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.night.mind import LIVE_DECIDER

    today, yesterday = NOW.date(), NOW.date() - timedelta(days=1)
    events = [
        _event(0, _start(yesterday, LIVE_DECIDER)), _event(1, None, 900),  # yesterday's live pass
        _event(2, _start(today, LIVE_DECIDER)), _event(3, None, 1500), _event(4, None, 500),  # today's
        _event(5, _start(today, None)), _event(6, None, 7000),  # tonight's own run
        _event(7, _start(today, LIVE_DECIDER)), _event(8, None, 1000),  # another today
    ]
    assert live_mind.live_budget_of(events, today, 12_000) == Budget(tokens_used=3000, token_cap=12_000)
    assert judge_cohort.setting(judge_cohort.LIVE_TOKENS_ENV) == 12_000
    monkeypatch.setenv(judge_cohort.LIVE_TOKENS_ENV, "2000")
    assert judge_cohort.setting(judge_cohort.LIVE_TOKENS_ENV) == 2000


def test_a_budget_that_cant_be_read_fails_the_pass_without_claiming_it(judge: str) -> None:
    def unreadable(_uid: str, _day: Any) -> Budget:
        raise ValueError("workspace.jsonl line 3 can't be read")

    windows: list[tuple[datetime, datetime]] = []
    broken = replace(ports(FakeMind(), windows), budget=unreadable)
    assert live_mind.run_pass(judge, ports=broken, now=NOW) == {"state": "failed", "error": "ValueError"}
    assert windows == [] and judge_cohort.counters().count(NOW, f"live:{judge}") == 0


def test_the_default_budget_reads_the_judges_own_log(judge: str) -> None:
    assert live_mind.PassPorts.__dataclass_fields__["budget"].default is live_mind.live_budget
    assert live_mind.live_budget(judge, NOW.date()) == Budget(token_cap=12_000)  # an empty log: all of it


def test_a_pass_holds_the_judges_lock_from_its_first_read_to_its_last_append(judge: str) -> None:
    seen: list[tuple[str, bool]] = []

    class Watching(FakeMind):
        def notice_live(self, call: Any, fresh: list[Any], budget: Budget) -> dict[str, Any]:
            seen.append(("notice", live_mind.wait_for_live_pass(judge, 0.0)))
            return super().notice_live(call, fresh, budget)

    def night_running() -> bool:
        seen.append(("night check", live_mind.wait_for_live_pass(judge, 0.0)))
        return False

    windows: list[tuple[datetime, datetime]] = []
    watched = replace(ports(Watching(), windows), night_running=night_running)
    assert live_mind.run_pass(judge, ports=watched, now=NOW)["state"] == "ok"
    assert seen == [("night check", False), ("notice", False)]  # both under the lock
    assert live_mind.wait_for_live_pass(judge, 0.0) and live_mind.live_pass_state(judge) == "idle"


def test_a_pass_skips_while_a_night_holds_its_lock(judge: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_mind, "LIVE_WAIT_S", 0.1)
    windows: list[tuple[datetime, datetime]] = []
    with live_mind.live_pass_guard(judge):
        assert live_mind.live_pass_state(judge) == "running"
        assert live_mind.run_pass(judge, ports=ports(FakeMind(), windows), now=NOW) == {"state": "busy"}
    assert windows == [] and judge_cohort.counters().count(NOW, f"live:{judge}") == 0


def test_the_lock_holds_across_processes(judge: str) -> None:
    import subprocess
    import sys

    path = live_mind._lock_path(judge)
    code = ("import fcntl, sys, time\n"
            "f = open(sys.argv[1], 'a+b'); fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(60)")
    holder = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        assert live_mind.wait_for_live_pass(judge, 0.2) is False
    finally:
        holder.kill()  # its own pid: the OS frees its lock
        holder.wait(timeout=10)
    assert live_mind.wait_for_live_pass(judge, 2.0) is True


def test_the_revision_grows_with_every_pass_and_every_step(judge: str) -> None:
    assert live_mind.revision(judge) == 0
    live_mind.run_pass(judge, ports=ports(FakeMind(), []), now=NOW)
    assert live_mind.revision(judge) == 1
    assert live_mind.revision(judge, steps=5) == 6


def test_a_failed_pass_is_recorded_and_keeps_its_cursor(judge: str) -> None:
    windows: list[tuple[datetime, datetime]] = []
    done = live_mind.run_pass(judge, ports=ports(FakeMind(fail=True), windows), now=NOW)
    assert (done["state"], done["error"]) == ("failed", "ModelApiError")
    assert live_mind.extracted_until(judge) == NOW  # what was extracted stays extracted


def test_one_pass_at_a_time_and_a_rerun_is_coalesced() -> None:
    gate, ran = threading.Event(), []
    threads: list[threading.Thread] = []

    def run(uid: str) -> dict[str, Any]:
        ran.append(uid)
        gate.wait(5)
        return {}

    def spawn(work: Any) -> None:
        thread = threading.Thread(target=work)
        threads.append(thread)
        thread.start()

    runner = live_mind.LiveRunner(run, spawn)
    assert runner.schedule("judge-1") is True
    assert runner.schedule("judge-1") is False
    assert runner.schedule("judge-1") is False  # still one rerun, not two
    gate.set()
    for thread in threads:
        thread.join(5)
    assert ran == ["judge-1", "judge-1"] and not runner.busy("judge-1")


def test_a_pause_holds_passes_until_the_night_is_done() -> None:
    ran: list[str] = []

    def run(uid: str) -> dict[str, Any]:
        ran.append(uid)
        return {}

    runner = live_mind.LiveRunner(run, lambda work: work())
    with runner.paused("judge-1"):
        assert runner.schedule("judge-1") is False
        assert ran == []
    assert ran == ["judge-1"]
    with runner.paused("judge-1"):
        pass
    assert ran == ["judge-1"]  # nothing was asked for during that one


def test_a_record_that_cant_be_read_reads_as_none(judge: str) -> None:
    from anjo.sleep.app.storage import sleep_dir

    path: Path = sleep_dir(judge) / live_mind.RECORD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not encrypted json")
    assert live_mind.load_record(judge) is None and live_mind.extracted_until(judge) is None


def test_the_real_step_is_the_nights_own_with_spark_while_glimmer_sleeps(judge: str) -> None:
    from anjo.sleep.night.mind import MindNight
    from anjo.sleep.spark_inquiry import SparkResolver, SparkThinker

    mind = live_mind._mind(False)
    assert isinstance(mind, MindNight)
    models = live_mind._models(False)(judge, lambda chat: chat)
    assert isinstance(models.thinker, SparkThinker) and isinstance(models.resolver, SparkResolver)
    assert models.model == live_mind.SPARK_THINKER
