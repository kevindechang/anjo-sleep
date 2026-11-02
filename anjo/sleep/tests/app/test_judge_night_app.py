# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Judges' "Sleep now" on the sleep host (2026-11-02): the two routes, the caps and the budget behind
them, the night records the view polls, the dispatcher, and the night's own process on a fake night.
No process is started and nothing is paid: spawning, the night and serving are fakes."""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from anjo.sleep.app import judge_child, judge_stages
from anjo.sleep.app import judge_night as jn
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.night.holds import usd
from anjo.sleep.night import judge
from anjo.sleep.night.judge import JudgeOutcome
from anjo.sleep.types import CheckpointRef, NightResult

JUDGE, OTHER = "judge-a", "judge-b"
NOW = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)


class Caps:
    def __init__(self, verdict: jn.StartVerdict = "ok") -> None:
        self.verdict: jn.StartVerdict = verdict
        self.asked: list[str] = []

    def try_start_night(self, user_id: str) -> jn.StartVerdict:
        self.asked.append(user_id)
        return self.verdict


class Spawner:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.dead: set[int] = set()

    def spawn(self, night_id: str) -> int:
        self.started.append(night_id)
        return 1000 + len(self.started)

    def alive(self, pid: int) -> bool:
        return pid not in self.dead


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    store, book, spawner, caps = jn.JobStore(tmp_path / "judge_nights"), Ledger(tmp_path / "budget.jsonl"), Spawner(), Caps()
    monkeypatch.setattr(jn, "job_store", lambda: store)
    monkeypatch.setattr(jn, "ledger", lambda: book)
    monkeypatch.setattr(jn, "_spawn", spawner.spawn)
    monkeypatch.setattr(jn, "_alive", spawner.alive)
    for env in (jn.CONCURRENCY_ENV, jn.BUDGET_ENV, jn.NIGHT_USD_ENV, jn.SPARK_ENV):
        monkeypatch.delenv(env, raising=False)
    jn.set_judge_ports(jn.JudgePorts(is_judge=lambda uid: uid in (JUDGE, OTHER), caps=caps))
    yield {"store": store, "book": book, "spawner": spawner, "caps": caps, "root": tmp_path}
    jn.set_judge_ports(None)


@pytest.fixture
def api(host: dict[str, Any]) -> Iterator[Any]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.dashboard.auth import get_current_user_id
    from anjo.sleep.app.judge_night_api import router

    who = {"user": JUDGE}
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_current_user_id] = lambda: who["user"]
    with TestClient(app) as client:
        client.who = who
        yield client


# ── the routes ───────────────────────────────────────────────────────────────


def test_nobody_is_a_judge_until_the_judge_api_says_so(api: Any) -> None:
    jn.set_judge_ports(None)
    assert api.post("/api/sleep/me/sleep-now").status_code == 404
    assert api.get("/api/sleep/me/nights/jn-0123456789abcdef").status_code == 404


def test_a_judge_starts_a_night_and_polls_it(api: Any, host: dict[str, Any]) -> None:
    started = api.post("/api/sleep/me/sleep-now")
    assert started.status_code == 200
    night_id = started.json()["night_id"]
    assert host["spawner"].started == [night_id] and host["caps"].asked == [JUDGE]
    view = api.get(f"/api/sleep/me/nights/{night_id}").json()
    assert view["status"] == "running" and view["phase"] == 1 and view["adapter_live"] is False
    assert [stage["name"] for stage in view["stages"]] == list(judge_stages.STAGES)
    assert all(stage["state"] == "pending" for stage in view["stages"]) and "note" not in view
    assert set(view["stages"][0]) == {"name", "state", "started_at", "ended_at", "summary"}


def test_a_second_start_while_one_runs_is_409_with_the_running_nights_id(api: Any) -> None:
    night_id = api.post("/api/sleep/me/sleep-now").json()["night_id"]
    again = api.post("/api/sleep/me/sleep-now")
    assert again.status_code == 409
    assert again.json()["night_id"] == night_id and again.json()["detail"] == jn.RUNNING


@pytest.mark.parametrize(("verdict", "status", "detail"), [
    ("per_judge", 429, jn.PER_JUDGE), ("global", 429, jn.GLOBAL), ("running", 409, jn.RUNNING)])
def test_the_caps_refuse_in_plain_words(api: Any, host: dict[str, Any], verdict: jn.StartVerdict, status: int,
                                        detail: str) -> None:
    host["caps"].verdict = verdict
    refused = api.post("/api/sleep/me/sleep-now")
    assert refused.status_code == status and refused.json() == {"detail": detail}
    assert host["spawner"].started == [] and host["store"].all() == []


def test_a_judge_never_sees_another_judges_night(api: Any) -> None:
    night_id = api.post("/api/sleep/me/sleep-now").json()["night_id"]
    api.who["user"] = OTHER
    assert api.get(f"/api/sleep/me/nights/{night_id}").status_code == 404
    assert api.get("/api/sleep/me/nights/../../etc").status_code == 404


# ── budget and alerts ────────────────────────────────────────────────────────


def test_the_hard_cap_refuses_a_night_before_any_cap_is_counted(api: Any, host: dict[str, Any],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jn.BUDGET_ENV, "15")
    _spent(host["book"], "judge-night:earlier", 6.0)
    refused = api.post("/api/sleep/me/sleep-now")  # $6 spent + a $10 hold would pass $15
    assert refused.status_code == 429 and refused.json()["detail"] == jn.BUDGET
    assert host["caps"].asked == []


def test_a_night_holds_its_budget_when_admitted_so_queued_nights_count(
    host: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(jn.BUDGET_ENV, "25")
    monkeypatch.setenv(jn.CONCURRENCY_ENV, "1")
    first, second = (jn.start_night(uid, now=lambda: NOW) for uid in ("j1", "j2"))
    assert isinstance(first, str) and isinstance(second, str)
    assert sorted(job["status"] for job in host["store"].all()) == ["queued", "running"]
    assert sorted(held.job for held in host["book"].open_holds()) == sorted(
        [jn.hold_key(first), jn.hold_key(second)])
    third = jn.start_night("j3", now=lambda: NOW)  # $20 held, nothing settled: a third $10 would pass $25
    assert isinstance(third, jn.Refused) and third.status == 429 and third.detail == jn.BUDGET
    assert host["caps"].asked == ["j1", "j2"]


@pytest.mark.parametrize("verdict", ["per_judge", "global", "running"])
def test_a_night_the_caps_refuse_lets_its_hold_go(host: dict[str, Any], verdict: jn.StartVerdict) -> None:
    host["caps"].verdict = verdict
    assert isinstance(jn.start_night(JUDGE, now=lambda: NOW), jn.Refused)
    assert host["book"].open_holds() == () and host["book"].used(jn.POOL) == 0.0


def test_the_budget_settings_never_pass_the_pool() -> None:
    assert jn.budget_usd({}) == CAPS["judge_usd"] == 100.0
    with pytest.raises(jn.JudgeSettingError):
        jn.budget_usd({jn.BUDGET_ENV: "150"})
    assert jn.night_usd({}) == 10.0 and jn.concurrency({}) == 2


def test_spend_alerts_fire_once_at_70_and_90_percent(tmp_path: Path) -> None:
    book, path = Ledger(tmp_path / "budget.jsonl"), tmp_path / "judge_alerts.jsonl"
    assert jn.check_alerts(book, path, 100.0, NOW) == []
    _spent(book, "judge-night:a", 75.0)
    assert jn.check_alerts(book, path, 100.0, NOW) == [0.7]
    assert jn.check_alerts(book, path, 100.0, NOW) == []
    _spent(book, "judge-night:b", 20.0)
    assert jn.check_alerts(book, path, 100.0, NOW) == [0.9]
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["threshold"] for line in lines] == [0.7, 0.9] and lines[1]["used_usd"] == 95.0


def test_the_night_process_runs_mind_s_loop_and_the_judges_spark_provider() -> None:
    env = jn.child_env({"PATH": "/bin", jn.SPARK_ENV: " Baseten "})
    assert env["ANJO_SLEEP_MIND_LOOP"] == "1" and env["ANJO_SLEEP_SPARK"] == "baseten" and env["PATH"] == "/bin"
    assert "ANJO_SLEEP_SPARK" not in jn.child_env({})
    host = jn.child_env({"ANJO_SLEEP_SPARK": "baseten"})
    assert host["ANJO_SLEEP_SPARK"] == "baseten"  # unset judges' setting: the host's own stands


# ── dispatch and records ─────────────────────────────────────────────────────


def test_nights_beyond_the_concurrency_wait_queued_and_start_as_room_frees(host: dict[str, Any]) -> None:
    store, spawner = host["store"], host["spawner"]
    for i, uid in enumerate(("j1", "j2", "j3")):
        at = NOW + timedelta(seconds=i)
        assert isinstance(jn.start_night(uid, now=lambda: at), str)
    assert [(job["user_id"], job["status"]) for job in store.all()] == [
        ("j1", "running"), ("j2", "running"), ("j3", "queued")]
    spawner.dead.add(1001)  # the first night's process died
    jn.pump(store, now=lambda: NOW)
    statuses = {job["user_id"]: job for job in store.all()}
    assert statuses["j1"]["status"] == "failed" and statuses["j1"]["error"] == jn.DIED
    assert statuses["j3"]["status"] == "running" and len(spawner.started) == 3


def test_a_night_whose_process_died_settles_its_hold_at_the_full_bound(host: dict[str, Any]) -> None:
    store, book, spawner = host["store"], host["book"], host["spawner"]
    night_id = jn.start_night(JUDGE, now=lambda: NOW)
    assert isinstance(night_id, str)
    book.reserve(f"{jn.HOLD_PREFIX}{night_id}", (jn.POOL,), usd(10.0))
    spawner.dead.add(1001)
    jn.pump(store, now=lambda: NOW)
    assert book.open_holds() == () and book.used(jn.POOL) == pytest.approx(10.0)


def test_each_night_reads_new_chat_and_is_dated_after_the_last(host: dict[str, Any]) -> None:
    first = jn.plan(JUDGE, [], None, NOW)
    assert first["night"] == "2026-11-03" and first["window"][0] == (NOW - timedelta(days=7)).isoformat()
    second = jn.plan(JUDGE, [first], date(2026, 11, 3), NOW + timedelta(minutes=20))
    assert second["night"] == "2026-11-04" and second["window"][0] == first["as_of"]
    third = jn.plan(JUDGE, [first, second], date(2026, 11, 3), NOW + timedelta(minutes=40))
    assert third["night"] == "2026-11-05"


# ── the stages ───────────────────────────────────────────────────────────────


def test_steps_fold_into_the_contracts_stages_with_plain_summaries() -> None:
    stages, at = judge_stages.empty(), "2026-11-03T15:00:00+00:00"
    label = {"t-1": "her climbing trip"}.get
    stages = judge_stages.fold(stages, "extract", "running", {}, at)
    assert stages["extract"]["state"] == "running" and judge_stages.phase(stages) == 1
    stages = judge_stages.fold(stages, "extract", "ok", {"items": 3, "new_since_last_training": 3,
                                                          "training_night": True}, at)
    stages = judge_stages.fold(stages, "notice", "ok", {"new": 3, "questions_opened": 2}, at)
    stages = judge_stages.fold(stages, "choose", "ok", {"action": "revisit_thread", "target_id": "t-1"}, at, label)
    stages = judge_stages.fold(stages, "brief", "ok", {}, at)  # no stage of its own
    stages = judge_stages.fold(stages, "practice", "running", {}, at)
    assert stages["extract"]["summary"] == "Read 3 things, 3 new"
    assert stages["mind_notice"]["summary"] == "Opened 2 questions"
    assert stages["mind_choose"]["summary"] == "Chose to study: her climbing trip"
    assert stages["mind_inquire"]["state"] == "skipped"
    assert judge_stages.phase(stages) == 2
    stages = judge_stages.fold(stages, "gates", "ok", {"gates": {
        "general": {"passed": True}, "collapse": {"passed": False}, "safety": {"passed": True, "skipped": True}}}, at)
    assert stages["gate"]["summary"] == "Gate held it back: collapse"
    stages = judge_stages.fold(stages, "train", "ok", {"receipt": {"steps": 8, "runner": "modal_warm"}}, at)
    assert stages["train"]["summary"] == "Updated her weights: 8 steps on a warm GPU"


def test_an_inquiry_inside_choose_settles_its_own_stage_and_refine_retitles_the_note() -> None:
    at = "2026-11-03T15:00:00+00:00"
    stages = judge_stages.fold(judge_stages.empty(), "choose", "ok",
                               {"action": "inquire", "inquiry": {"cycles": 1, "research_calls": 2}}, at)
    assert stages["mind_inquire"] == {"state": "done", "started_at": at, "ended_at": at,
                                      "summary": "Asked 2 research questions"}
    stages = judge_stages.fold(stages, "note", "ok", {"outcome": "held", "note_id": "n-1"}, at)
    stages = judge_stages.fold(stages, "refine", "ok", {"outcome": "refined"}, at)
    assert stages["note"]["state"] == "done" and stages["note"]["summary"] == "Rewritten by tonight's weights"


def test_a_closed_night_leaves_nothing_running_or_pending_but_serve() -> None:
    at = "2026-11-03T15:00:00+00:00"
    stages = judge_stages.fold(judge_stages.empty(), "practice", "running", {}, at)
    closed = judge_stages.close(stages, at, failed=True)
    assert closed["practice"]["state"] == "failed" and closed["train"]["state"] == "skipped"
    assert closed["serve"]["state"] == "pending"


# ── the night's own process ──────────────────────────────────────────────────


class Notes:
    def __init__(self) -> None:
        self.queued: list[tuple[str, str, str]] = []

    def queue(self, user_id: str, message: str, source: str) -> None:
        self.queued.append((user_id, message, source))

    def push(self, user_id: str, message: str) -> bool:
        raise AssertionError("a judge's note is never pushed")


def _result(status: str, root: Path, ckpt: CheckpointRef | None = None, cost: float = 0.8) -> NightResult:
    return NightResult(status=status, failed_at="train" if status == "failed" else None,  # type: ignore[arg-type]
                       checkpoint=ckpt, gates=MappingProxyType({}), choices=(), cost_usd=cost,
                       cost_parts=MappingProxyType({"training": cost}), log_path=root / "log.jsonl")


def _ckpt() -> CheckpointRef:
    return CheckpointRef(id="p-aaaaaaaaaaaa-n1", person_id=JUDGE, night=date(2026, 11, 3), uri="modal://adapters/x",
                         parent_id=None, rendering="r/low")


def _spent(book: Ledger, job: str, amount: float) -> None:
    held = book.reserve(job, (jn.POOL,), usd(amount))
    assert held is not None
    held.settle(amount)


def _queued(host: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(jn, "checkpoint_registry", lambda: type("R", (), {"last_promoted": lambda self, uid: None})())
    with host["store"].locked():
        night_id: str = jn.new_job(JUDGE, host["store"], NOW)["night_id"]
    return night_id


def _night(steps: list[tuple[str, str, dict[str, Any]]], result: NightResult | None, *,
           raises: BaseException | None = None) -> Any:
    def execute(*, observer: Any, **kwargs: Any) -> JudgeOutcome:
        for step, state, receipt in steps:
            observer(JUDGE, step, state, receipt)
        if raises is not None:
            raise raises
        assert result is not None
        return JudgeOutcome(result, warm_usd=0.2)

    return execute


class Busy(RuntimeError):
    """The fake lock's ``LiveBusy``."""


class Live:
    """``live_mind``'s wait, guard and cursor, as the night calls them: each call kept in ``log``."""

    def __init__(self, log: list[str] | None = None, *, waited: bool = True, busy: bool = False,
                 cursor: datetime | None = None) -> None:
        self.log = [] if log is None else log
        self.waited, self.busy, self.cursor = waited, busy, cursor

    def wait(self, user_id: str, timeout_s: float) -> bool:
        self.log.append(f"wait {user_id} {timeout_s:g}")
        return self.waited

    @contextmanager
    def guard(self, user_id: str, *, timeout_s: float = 0.0) -> Iterator[None]:
        if self.busy:
            raise Busy(user_id)
        self.log.append(f"guard {user_id} {timeout_s:g}")
        try:
            yield
        finally:
            self.log.append("guard released")

    def ports(self) -> judge_child.LivePorts:
        return judge_child.LivePorts(wait=self.wait, guard=self.guard, busy=Busy, cursor=lambda uid: self.cursor)


def _run(host: dict[str, Any], night_id: str, execute: Any, notes: Notes, monkeypatch: pytest.MonkeyPatch,
         served: list[str] | None = None, live: Live | None = None) -> int:
    held: list[Any] = []

    def wiring(notifier: Any, *args: Any) -> Any:
        held.append(notifier)
        return lambda trainer, options: None

    monkeypatch.setattr(judge_child, "judge_wiring", wiring)
    monkeypatch.setattr(judge_child, "target_label", lambda uid: lambda target_id: None)

    def capture(**kwargs: Any) -> JudgeOutcome:
        notifier = held[-1]  # the night's notifier, as the wiring was handed it
        notifier.queue(JUDGE, "Phase one's note.", "sleep")
        notifier.queue(JUDGE, "The rewritten note.", "sleep")
        outcome: JudgeOutcome = execute(**kwargs)
        return outcome

    ensure = (lambda ckpt: served.append(ckpt.id)) if served is not None else None
    return judge_child.run(night_id, store=host["store"], book=host["book"],
                           environ={"ANJO_DATA_DIR": str(host["root"])}, execute=capture, notifier=notes, ensure=ensure,
                           live=(live or Live()).ports())


def test_a_promoted_night_shows_its_steps_gives_one_note_and_goes_live(host: dict[str, Any],
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    night_id, notes, served = _queued(host, monkeypatch), Notes(), list[str]()
    steps: list[tuple[str, str, dict[str, Any]]] = [("extract", "running", {}), ("extract", "ok", {"items": 2, "new_since_last_training": 2}),
             ("practice", "ok", {"scenes": 8}), ("promote", "ok", {"promoted": "p-aaaaaaaaaaaa-n1"})]
    code = _run(host, night_id, _night(steps, _result("promoted", host["root"], _ckpt())), notes, monkeypatch, served)
    job = host["store"].get(night_id)
    assert code == 0 and job["status"] == "done" and job["phase"] == 2
    assert notes.queued == [(JUDGE, "The rewritten note.", "sleep")]  # once, the last
    assert job["note"] == "The rewritten note."
    assert served == ["p-aaaaaaaaaaaa-n1"] and job["adapter_live"] is True
    assert job["stages"]["serve"]["state"] == "done" and job["stages"]["serve"]["summary"] == judge_child.LIVE
    assert job["stages"]["train"]["state"] == "skipped"  # never reached in this fake night
    assert host["book"].open_holds() == () and host["book"].used(jn.POOL) == pytest.approx(1.0)  # 0.8 + 0.2 warm


def test_a_rejected_night_serves_nothing_new(host: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    night_id, served = _queued(host, monkeypatch), list[str]()
    code = _run(host, night_id, _night([], _result("rejected", host["root"])), Notes(), monkeypatch, served)
    job = host["store"].get(night_id)
    assert code == 0 and served == [] and job["adapter_live"] is False
    assert job["stages"]["serve"]["state"] == "skipped"


def test_a_night_that_breaks_after_it_began_settles_its_whole_hold(host: dict[str, Any],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    night_id = _queued(host, monkeypatch)
    execute = _night([("extract", "running", {})], None, raises=RuntimeError("boom"))
    assert _run(host, night_id, execute, Notes(), monkeypatch) == 1
    job = host["store"].get(night_id)
    assert job["status"] == "failed" and job["error"] == "RuntimeError"
    assert job["stages"]["extract"]["state"] == "failed"
    assert host["book"].used(jn.POOL) == pytest.approx(jn.DEFAULT_NIGHT_USD)


def test_the_nights_process_settles_the_hold_its_admission_took(host: dict[str, Any],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jn, "checkpoint_registry", lambda: type("R", (), {"last_promoted": lambda self, uid: None})())
    night_id = jn.start_night(JUDGE, now=lambda: NOW)
    assert isinstance(night_id, str)
    [admitted] = host["book"].open_holds()
    assert admitted.job == jn.hold_key(night_id) and admitted.bound.amount == pytest.approx(jn.DEFAULT_NIGHT_USD)
    assert _run(host, night_id, _night([], _result("rejected", host["root"])), Notes(), monkeypatch) == 0
    assert host["book"].open_holds() == () and host["book"].settlements() == {jn.hold_key(night_id): pytest.approx(1.0)}


def test_a_night_with_no_budget_left_fails_before_it_starts(host: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    night_id = _queued(host, monkeypatch)
    _spent(host["book"], "judge-night:earlier", 95.0)
    assert _run(host, night_id, _night([], None, raises=AssertionError("never runs")), Notes(), monkeypatch) == 1
    assert host["store"].get(night_id)["error"] == jn.BUDGET


def test_a_serving_failure_leaves_the_night_done_and_the_adapter_for_her_next_reply(
    host: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.sleep.serving.client import ServingError

    recorder = judge_child.Recorder(host["store"], _queued(host, monkeypatch), lambda target_id: None)

    def down(ckpt: CheckpointRef) -> None:
        raise ServingError("asleep")

    assert judge_child.serve(_ckpt(), recorder, down) is False


def test_the_dispatcher_names_an_entry_the_app_has() -> None:
    module, _, name = jn.ENTRY.partition(":")
    assert module == judge_child.__name__ and callable(getattr(judge_child, name))


def test_the_child_names_the_night_by_its_records_fields_and_the_engine_builds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, end = NOW - timedelta(days=7), NOW
    record = {"user_id": JUDGE, "night": "2026-11-03", "window": [start.isoformat(), end.isoformat()],
              "as_of": end.isoformat()}
    handed: dict[str, Any] = {}

    def execute_judge(**kwargs: Any) -> str:
        handed.update(kwargs)
        return "ran"

    monkeypatch.setattr(judge, "execute_judge", execute_judge)
    ran: object = judge.execute_record(job=judge_child._job(record), wiring="wiring")
    assert ran == "ran"
    assert handed == {"job": judge.JudgeNight(user=JUDGE, night=date(2026, 11, 3), window=(start, end), as_of=end),
                      "wiring": "wiring"}


# ── live MIND passes and the night ───────────────────────────────────────────


class Port:
    """A MIND port that answers available and keeps, for each call, whether the judge's live-pass lock
    was held then (the page's own reading of it)."""

    available = True

    def __init__(self, user_id: str) -> None:
        self.user_id, self.calls = user_id, list[tuple[str, bool]]()

    def think(self) -> str:
        from anjo.sleep.app import live_mind

        self.calls.append(("think", live_mind.live_pass_state(self.user_id) == "running"))
        return "thought"


def test_mind_is_on_only_once_the_judges_lock_was_had_and_each_call_holds_it() -> None:
    from anjo.sleep.app import live_mind

    rested: list[bool] = []
    lock = judge_child.MindLock(JUDGE, on_rest=lambda: rested.append(True))
    port = Port(JUDGE)
    mind = judge_child.LockedMind(port, lock)
    assert lock.state == "untried" and live_mind.live_pass_state(JUDGE) == "idle"
    assert mind.available and lock.state == "open"  # decided at MIND's first use, the lock had
    assert live_mind.live_pass_state(JUDGE) == "idle"  # between MIND's calls the page sees no pass
    assert mind.think() == "thought" and mind.think() == "thought"
    assert port.calls == [("think", True), ("think", True)] and rested == []
    assert live_mind.live_pass_state(JUDGE) == "idle"


def test_a_pass_holding_the_lock_past_both_waits_leaves_mind_resting_and_never_unlocked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from anjo.sleep.app import live_mind

    rested: list[bool] = []
    lock = judge_child.MindLock(JUDGE, timeout_s=0.05, on_rest=lambda: rested.append(True))
    port = Port(JUDGE)
    mind = judge_child.LockedMind(port, lock)
    with live_mind.live_pass_guard(JUDGE):  # a live pass stuck in its model call
        assert mind.available is False and lock.state == "resting"
    assert mind.available is False  # decided once: not tried again later in the night
    with pytest.raises(judge_child.MindResting):
        mind.think()
    assert port.calls == [] and rested == [True] and "MIND rests tonight" in caplog.text


def test_a_lock_that_fails_on_the_fake_too_rests_mind() -> None:
    log: list[str] = []
    lock = judge_child.MindLock(JUDGE, Live(log, waited=False, busy=True).ports())
    assert judge_child.LockedMind(Port(JUDGE), lock).available is False
    assert log == [f"wait {JUDGE} {judge_child.PASS_WAIT_S:g}"] and lock.state == "resting"


def test_a_resting_mind_shows_its_stages_skipped_in_plain_words(host: dict[str, Any],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    night_id = _queued(host, monkeypatch)
    recorder = judge_child.Recorder(host["store"], night_id, lambda target_id: None)
    recorder.observe(JUDGE, "notice", "running", {})
    recorder.rest_mind()
    recorder.observe(JUDGE, "notice", "ok", {"noop": True})
    recorder.observe(JUDGE, "choose", "ok", {"action": "hold_note"})
    recorder.observe(JUDGE, "self_predict", "ok", {"noop": True, "why": "MIND's loop is off"})
    stages = host["store"].get(night_id)["stages"]
    for name in judge_stages.RESTED:
        assert stages[name]["state"] == "skipped" and stages[name]["summary"] == judge_stages.MIND_RESTED
    assert stages["mind_choose"]["state"] == "done"  # her chooser ran, without MIND's questions


def _engine_night(tmp_path: Path, lock: judge_child.MindLock) -> tuple[Any, Any, Any]:
    from anjo.sleep.night.deps import NightSettings, NoInquiry, NoNotice
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.tests.night_fakes import NIGHT, World, thread

    seen: list[tuple[str, bool]] = []

    def held() -> bool:
        from anjo.sleep.app import live_mind

        return live_mind.live_pass_state(JUDGE) == "running"

    class Notice(NoNotice):
        available = True

        def notice(self, call: Any, new: Any) -> Any:
            seen.append(("notice", held()))
            return {"new": len(new), "questions_opened": 0}

    class Inquiry(NoInquiry):
        available = True

        def questions(self, call: Any) -> Any:
            seen.append(("questions", held()))
            return ()

        def settle(self, call: Any) -> Any:
            seen.append(("settle", held()))
            return []

    world = World(tmp_path / "world")
    world.plan(JUDGE, NIGHT)
    world.targets(JUDGE).upsert(thread(JUDGE))
    deps = world.deps(settings=NightSettings(token_cap=16384, min_new_examples=1, mind_inquiry=True))
    deps = replace(deps, notice=judge_child.LockedMind(Notice(), lock), inquiry=judge_child.LockedMind(Inquiry(), lock))
    done = run_night_batch_with(deps, {JUDGE: NIGHT})[JUDGE]
    return done, seen, world


def test_the_engine_runs_mind_only_under_the_judges_lock(tmp_path: Path) -> None:
    done, seen, _ = _engine_night(tmp_path, judge_child.MindLock(JUDGE))
    assert done.status != "failed" and seen and all(locked for _, locked in seen)
    assert {name for name, _ in seen} >= {"notice", "settle", "questions"}


def test_a_night_whose_mind_rests_runs_on_without_it(tmp_path: Path) -> None:
    from anjo.sleep.app import live_mind

    rested: list[bool] = []
    lock = judge_child.MindLock(JUDGE, timeout_s=0.05, on_rest=lambda: rested.append(True))
    with live_mind.live_pass_guard(JUDGE):  # held through the whole night: a pass that never ends
        done, seen, _ = _engine_night(tmp_path, lock)
    assert seen == [] and rested == [True]  # no MIND call ran, locked or not
    assert done.status != "failed"  # the rest of the night went on


# ── the night's extraction after the live passes' ────────────────────────────


class Inner:
    def __init__(self) -> None:
        self.windows: list[tuple[datetime, datetime]] = []

    def extract_day(self, user_id: str, day: date, window: tuple[datetime, datetime],
                    as_of: datetime | None = None) -> list[Any]:
        self.windows.append(window)
        return []


def test_a_judges_extraction_reads_only_after_the_live_passes_cursor() -> None:
    start, end = NOW - timedelta(days=7), NOW
    inner, live = Inner(), Live()
    after = judge_child.AfterLive(inner, live.ports())
    after.extract_day(JUDGE, date(2026, 11, 3), (start, end))
    live.cursor = NOW - timedelta(hours=1)
    after.extract_day(JUDGE, date(2026, 11, 3), (start, end))
    live.cursor = (NOW - timedelta(minutes=30)).replace(tzinfo=None)  # a cursor without a zone reads as UTC
    after.extract_day(JUDGE, date(2026, 11, 3), (start, end))
    live.cursor = NOW + timedelta(minutes=5)  # a pass after the night was asked for read all of it
    assert after.extract_day(JUDGE, date(2026, 11, 3), (start, end)) == []
    assert inner.windows == [(start, end), (NOW - timedelta(hours=1), end), (NOW - timedelta(minutes=30), end)]
    assert live.log.count(f"wait {JUDGE} {judge_child.PASS_WAIT_S:g}") == 4  # a pass under way ends first


def test_a_live_pass_then_a_judges_night_extract_each_turn_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import Counter

    from anjo.sleep.app import judge_cohort, live_mind
    from anjo.sleep.app.history_source import HistorySource
    from anjo.sleep.app.storage import evidence_store
    from anjo.sleep.mind.inquiry.steps import Budget
    from anjo.sleep.tests.app.test_history_source import EchoSparkOpen, _message
    from anjo.sleep.tests.app.test_live_mind import FakeMind

    monkeypatch.setenv("ANJO_SLEEP_PRICES", "0.1,0.4,1.0,1.95")
    joined = NOW - timedelta(hours=3)
    judge_cohort.add_judge(JUDGE, now=joined)
    chat = EchoSparkOpen()
    source = HistorySource(chat=chat)
    said = ["I sketch every layout on paper first.", "Grids feel like a cage to me.", "Type is where I start."]
    _message(JUDGE, "user", said[0], at=joined + timedelta(minutes=10))
    _message(JUDGE, "assistant", "Paper slows the first idea down.", at=joined + timedelta(minutes=11))
    _message(JUDGE, "user", said[1], at=joined + timedelta(minutes=20))
    passed_at = joined + timedelta(minutes=30)
    passed = live_mind.run_pass(JUDGE, now=passed_at, ports=live_mind.PassPorts(
        extract=lambda uid, start, end: source.record_day(uid, end.date(), (start, end), as_of=end)[0],
        mind=lambda glimmer: FakeMind(), glimmer_up=lambda: False, night_running=lambda: False,
        budget=lambda uid, day: Budget(tokens_used=0, token_cap=12_000)))
    assert passed["state"] == "ok" and live_mind.extracted_until(JUDGE) == passed_at
    _message(JUDGE, "user", said[2], at=passed_at + timedelta(minutes=10))
    night_end = passed_at + timedelta(minutes=20)
    night = judge_child.AfterLive(source)  # live_mind's own wait and cursor
    found = night.extract_day(JUDGE, date(2026, 11, 4), (NOW - timedelta(days=7), night_end), night_end)
    read = Counter(m["text"] for call in chat.calls for m in json.loads(call[-1]["content"])["messages"])
    assert read == Counter(said)  # each of the judge's turns went through one extraction, once
    assert [item.text for item in found] == [f"They said: {said[2]}"]
    stored = [item.text for item in evidence_store(JUDGE).stored(date(2026, 11, 4))]
    assert sorted(stored) == sorted(f"They said: {text}" for text in said)  # no item stored twice


# ── the wake's before and after: one real pair from tonight's collapse gate ──


def _pair(i: int, *, base_dragged: bool = False, dragged: bool = False) -> Any:
    from anjo.sleep.gates.collapse import CollapsePair

    return CollapsePair(prompt_id=f"c{i:02d}", prompt=f"everyday question {i}?", base_reply=f"base answer {i}",
                        reply=f"adapter answer {i}", base_dragged=base_dragged, dragged=dragged)


def _gated(status: str, root: Path, *, collapse: Any) -> NightResult:
    gates = {} if collapse is None else {"collapse": collapse}
    return replace(_result(status, root, _ckpt() if status == "promoted" else None), gates=MappingProxyType(gates))


def _paired_run(host: dict[str, Any], monkeypatch: pytest.MonkeyPatch, result: NightResult,
                pairs: list[Any] | None) -> list[dict[str, Any]]:
    """A night whose collapse gate hands ``pairs`` over (None: the gate never ran); what was saved."""
    night_id, saved, sinks = _queued(host, monkeypatch), list[dict[str, Any]](), list[Any]()
    monkeypatch.setattr(judge_child, "judge_wiring", lambda notifier, *args: sinks.append(args[-1]) or (lambda t, o: None))
    monkeypatch.setattr(judge_child, "target_label", lambda uid: lambda target_id: None)
    monkeypatch.setattr(judge_child, "save_before_after", lambda user_id, **kw: saved.append({"user": user_id, **kw}))

    def execute(**kwargs: Any) -> JudgeOutcome:
        if pairs is not None:
            sinks[-1](_ckpt(), pairs)
        return JudgeOutcome(result, warm_usd=0.0)

    judge_child.run(night_id, store=host["store"], book=host["book"], environ={"ANJO_DATA_DIR": str(host["root"])},
                    execute=execute, notifier=Notes(), ensure=lambda ckpt: None, live=Live().ports())
    return saved


def test_the_wake_gets_the_pair_the_gate_scored_most_for_tonights_adapter(host: dict[str, Any],
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.types import GateResult

    passed = GateResult("collapse", True, 0.0, 5.0)
    pairs = [_pair(0), _pair(1, dragged=True), _pair(2, base_dragged=True), _pair(3, base_dragged=True)]
    saved = _paired_run(host, monkeypatch, _gated("promoted", host["root"], collapse=passed), pairs)
    assert saved == [{"user": JUDGE, "question": "everyday question 2?", "before": "base answer 2",
                      "after": "adapter answer 2", "night": "2026-11-03", "before_model": judge_child.BASE_MODEL,
                      "after_model": "p-aaaaaaaaaaaa-n1", "promoted": True}]


def test_with_no_score_between_them_the_first_pair_is_kept_and_a_rejection_says_so(
        host: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.types import GateResult

    held = GateResult("collapse", False, 20.0, 5.0, "dragged")
    saved = _paired_run(host, monkeypatch, _gated("rejected", host["root"], collapse=held), [_pair(0), _pair(1)])
    assert [(s["question"], s["promoted"], s["after_model"]) for s in saved] == [
        ("everyday question 0?", False, "p-aaaaaaaaaaaa-n1")]


@pytest.mark.parametrize("case", ["skipped", "never_ran", "no_pairs"])
def test_no_gate_no_pair(host: dict[str, Any], monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    from anjo.sleep.types import GateResult

    gate = {"skipped": GateResult("collapse", False, 0.0, 5.0, "off", skipped=True), "never_ran": None,
            "no_pairs": GateResult("collapse", True, 0.0, 5.0)}[case]
    result = _gated("rejected", host["root"], collapse=gate)
    pairs = None if case == "never_ran" else [] if case == "no_pairs" else [_pair(0)]
    assert _paired_run(host, monkeypatch, result, pairs) == []
