# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Judges' "Sleep now" on the sleep host (2026-11-02): the two routes, the caps and the budget behind
them, the night records the view polls, the dispatcher, and the night's own process on a fake night.
No process is started and nothing is paid: spawning, the night and serving are fakes."""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
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


class Live:
    """``live_mind``'s wait and guard, as the night calls them: each call kept in ``log`` (with the night's
    steps, when the test's night writes them there too)."""

    def __init__(self, log: list[str] | None = None, *, waited: bool = True) -> None:
        self.log = [] if log is None else log
        self.waited = waited

    def wait(self, user_id: str, timeout_s: float) -> bool:
        self.log.append(f"wait {user_id} {timeout_s:g}")
        return self.waited

    @contextmanager
    def guard(self, user_id: str, *, timeout_s: float = 0.0) -> Iterator[None]:
        self.log.append(f"guard {user_id} {timeout_s:g}")
        try:
            yield
        finally:
            self.log.append("guard released")

    def ports(self) -> judge_child.LivePorts:
        return judge_child.LivePorts(wait=self.wait, guard=self.guard)


def _run(host: dict[str, Any], night_id: str, execute: Any, notes: Notes, monkeypatch: pytest.MonkeyPatch,
         served: list[str] | None = None, live: Live | None = None) -> int:
    held: list[Any] = []

    def wiring(notifier: Any) -> Any:
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


def _logged_night(log: list[str], steps: list[tuple[str, str, dict[str, Any]]], *,
                  raises: BaseException | None = None) -> Any:
    def execute(*, observer: Any, **kwargs: Any) -> JudgeOutcome:
        for step, state, receipt in steps:
            log.append(f"{step} {state}")
            observer(JUDGE, step, state, receipt)
        if raises is not None:
            raise raises
        return JudgeOutcome(_result("rejected", Path("/nonexistent")), warm_usd=0.0)

    return execute


PHASE_ONE: list[tuple[str, str, dict[str, Any]]] = [
    ("extract", "running", {}), ("extract", "ok", {}), ("notice", "running", {}), ("notice", "ok", {}),
    ("choose", "running", {}), ("homework", "running", {"background": True}), ("choose", "ok", {}),
    ("brief", "running", {}), ("brief", "ok", {}), ("homework", "running", {}), ("note", "running", {})]


def test_the_night_waits_for_a_live_pass_and_holds_its_lock_around_the_mind_steps(
    host: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    night_id = _queued(host, monkeypatch)
    assert _run(host, night_id, _logged_night(log, PHASE_ONE), Notes(), monkeypatch, live=Live(log)) == 0
    wait = judge_child.PASS_WAIT_S
    assert log[:6] == ["extract running", "extract ok", "notice running", f"wait {JUDGE} {wait:g}",
                       f"guard {JUDGE} {wait:g}", "notice ok"]  # extraction runs without it
    released = log.index("guard released")
    assert log[released - 1] == "brief running"  # the background study's start inside choose kept it
    assert log.count("guard released") == 1 and log.count(f"wait {JUDGE} {wait:g}") == 1


def test_a_night_that_breaks_during_its_mind_steps_lets_the_lock_go(host: dict[str, Any],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []
    night_id = _queued(host, monkeypatch)
    execute = _logged_night(log, PHASE_ONE[:3], raises=RuntimeError("boom"))
    assert _run(host, night_id, execute, Notes(), monkeypatch, live=Live(log)) == 1
    assert log[-1] == "guard released" and log.count("guard released") == 1


def test_a_live_pass_still_running_after_the_wait_leaves_the_night_waiting_for_its_lock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[str] = []
    guard = judge_child.LiveGuard(JUDGE, Live(log, waited=False).ports())
    guard.observe("notice", "running", {})
    assert guard.held and log[1] == f"guard {JUDGE} {judge_child.PASS_WAIT_S:g}" and "still ran" in caplog.text
    guard.release()
    assert log[-1] == "guard released"


def test_a_night_takes_the_judges_real_live_pass_lock_and_the_page_sees_it_only_around_mind() -> None:
    from anjo.sleep.app import live_mind

    guard = judge_child.LiveGuard(JUDGE)  # live_mind's own wait and lock
    guard.observe("extract", "running", {})
    assert live_mind.live_pass_state(JUDGE) == "idle"
    guard.observe("notice", "running", {})
    assert guard.held and live_mind.live_pass_state(JUDGE) == "running"
    assert live_mind.wait_for_live_pass(JUDGE, 0.0) is False  # a live pass now waits, then skips
    guard.observe("brief", "running", {})
    assert not guard.held and live_mind.live_pass_state(JUDGE) == "idle"


def test_a_lock_a_stuck_pass_holds_is_reported_once_and_mind_runs_on() -> None:
    from anjo.sleep.app import live_mind

    guard = judge_child.LiveGuard(JUDGE, timeout_s=0.05)
    with live_mind.live_pass_guard(JUDGE):  # a live pass that never lets go
        with pytest.raises(live_mind.LiveBusy):  # the night's observer logs it and the night goes on
            guard.observe("notice", "running", {})
        guard.observe("choose", "running", {})  # not tried again mid-MIND
    assert not guard.held
    guard.release()
