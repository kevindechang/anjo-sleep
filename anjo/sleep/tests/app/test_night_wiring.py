# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's app wiring (Task 5): the held note's delivery, the precision every arm and gate
serves at, and ``build`` over the sleep host's stores. Offline: the serving client is built, never
called."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.serving import config

NOW = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)
URL = "https://anjo--anjo-sleep-serve-serve-web.modal.run"


def test_push_only_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core import notifications
    from anjo.sleep.app.delivery import OutreachNotifier

    sent: list[str] = []

    def push(user_id: str) -> bool:
        sent.append(user_id)
        return True

    monkeypatch.setattr(notifications, "send_outreach_push", push)
    for enabled, tokens, expected in ((False, 1, False), (True, 0, False), (True, 2, True)):
        monkeypatch.setattr(notifications, "ensure_preferences",
                            lambda user_id, e=enabled, t=tokens: {"enabled": e, "token_count": t})
        assert OutreachNotifier().push("p1", "a note") is expected
    assert sent == ["p1"]


def test_no_note_delivered_into_cleared_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core import outreach
    from anjo.dashboard import session_store
    from anjo.sleep.app import api
    from anjo.sleep.app.cleared_sessions import cleared_sessions, register_synthetic
    from anjo.sleep.app.delivery import held

    user_id = "synthetic-founder"
    register_synthetic(user_id, at=NOW)
    monkeypatch.setattr(outreach, "DELIVERY_HOLDS", [held])
    outreach.queue_outreach(user_id, "While you were away I looked into the pilot.", "sleep_note")
    seen: list[str | None] = []

    def build(uid: str) -> tuple[str, bool]:  # what the session build reads to deliver
        seen.append(outreach.get_pending_outreach(uid))
        return "s-1", True

    monkeypatch.setattr(session_store, "get_or_create_session_with_status", build)
    state = api.open_session(api.SessionRequest(cleared=True), user_id=user_id)
    assert state.cleared and "s-1" in cleared_sessions(user_id)
    assert seen == [None]  # the cleared open saw no pending note
    assert outreach.get_pending_outreach(user_id) == "While you were away I looked into the pilot."
    api.open_session(api.SessionRequest(cleared=False), user_id=user_id)
    assert seen[-1] == "While you were away I looked into the pilot."  # the next uncleared open does


def test_a_note_that_trips_the_reply_check_is_declined() -> None:
    from anjo.evals.crisis_reply_cases import REPLY_TRIPS
    from anjo.sleep.app.delivery import passes_reply_check

    assert passes_reply_check("I looked into the pilot.") is True
    assert passes_reply_check(REPLY_TRIPS[0][1]) is False  # a known trip, by index


def _night_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.app import night, runtime

    for name, value in {
        "ANJO_DATA_DIR": str(tmp_path), runtime.URL_ENV: URL, runtime.TOKEN_ENV: "test-modal-token",
        night.PRICES_ENV: "0.2,0.6,3.67,1.95", night.SEALED_ENV: str(tmp_path / "sealed"),
    }.items():
        monkeypatch.setenv(name, value)


class NoTrainer:
    def train(self, *args: Any) -> Any:
        raise AssertionError("not called here")


def test_every_arm_serves_at_same_precision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_embedder: Any) -> None:
    from anjo.sleep.app import night
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from anjo.sleep.app.runtime import glimmer_pin
    from anjo.sleep.night.deps import NightOptions

    _night_env(monkeypatch, tmp_path)
    register_synthetic("synthetic-founder", at=NOW)
    deps = night.build(NoTrainer(), NightOptions(night=date(2026, 10, 28), personas={"synthetic-founder": "founder"}))
    served = glimmer_pin("synthetic-founder", None, config.ServeConfig.from_env()).quantization
    gates: Any = deps.gates
    note: Any = deps.note_writer
    practice: Any = deps.practice(lambda *a, **k: None, date(2026, 10, 28))  # type: ignore[arg-type]
    clients = (gates._client, note._client, practice._client)
    assert len({c._poster for c in clients}) == 1  # practice, the gates and the note share one server
    assert {c.precision for c in clients} == {served} == {config.SERVING.name}
    assert set(gates.for_person("synthetic-founder")) == {"general", "collapse", "safety"}
    from anjo.sleep.app.delivery import passes_reply_check
    from anjo.sleep.budget import LEDGER_NAME

    gpu: Any = deps.gpu
    assert deps.note_check is passes_reply_check  # the final note text, sources included
    assert deps.ledger is not None and deps.ledger.path.name == LEDGER_NAME and gpu._ledger is deps.ledger


class ModelsServer:
    """``/v1/models`` through a poster, as SageMaker's shim sends it on: Glimmer at ``root``, or a
    failure; every other path is refused, so nothing else is asked."""

    def __init__(self, root: str | None) -> None:
        self.root, self.asked = root, 0

    def post(self, path: str, body: Any) -> dict[str, Any]:
        assert path == "/v1/models" and body == {}
        self.asked += 1
        if self.root is None:
            from anjo.sleep.serving.client import TransportFailed

            raise TransportFailed("/v1/models failed in transport (ReadTimeout)")
        return {"data": [{"id": config.MODEL, "root": self.root}]}

    def close(self) -> None:
        pass


class Backend:
    name = "modal"


def _base(night_module: Any, server: ModelsServer, calls: list[int]) -> list[int]:
    """One night's base side of one gate through the app's cache: computed (and counted) or read back."""
    from anjo.sleep.serving.client import VllmClient

    def asked(answers: list[Any]) -> list[Any]:
        calls.append(1)
        return answers

    cache = night_module.gate_cache(Backend(), VllmClient.over(server, max_tokens=1024))
    first = cache.answers("general", {"items": [1, 2]}, lambda: asked([0, 1]), lambda kept: len(kept) == 2)
    again = cache.answers("collapse", {"items": [1]}, lambda: asked([True]), lambda kept: len(kept) == 1)
    assert server.asked == 1  # the server is asked once a run, at the first gate
    return [*first, *again]


def test_cache_misses_when_the_server_revision_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    from anjo.sleep.app import night

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(config.REVISION_ENV, "a" * 40)  # the host's pin: never the key
    calls: list[int] = []
    served = f"/weights/{config.MODEL}/{'a' * 40}"
    assert _base(night, ModelsServer(served), calls) == [0, 1, True] and len(calls) == 2  # asked, then kept
    assert _base(night, ModelsServer(served), calls) == [0, 1, True] and len(calls) == 2  # read back
    redeployed = ModelsServer(f"/weights/{config.MODEL}/{'b' * 40}")  # the host's env still says "a"
    _base(night, redeployed, calls)
    assert len(calls) == 4  # the server's new revision is a new key: asked again, never the old answers
    assert "revision bbbbbbbbbbbb" in caplog.text and "aaaaaaaaaaaa" in caplog.text
    _base(night, ModelsServer("/opt/ml/model"), calls)  # a root that names no revision: asked, never kept
    _base(night, ModelsServer("/opt/ml/model"), calls)
    assert len(calls) == 8


def test_unreachable_models_endpoint_is_a_miss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.app import night

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    calls: list[int] = []
    _base(night, ModelsServer(f"/weights/{config.MODEL}/{'a' * 40}"), calls)
    assert len(calls) == 2
    for _ in range(2):  # the server can't say what it serves: every base answer is asked, none read back
        _base(night, ModelsServer(None), calls)
    assert len(calls) == 6


def test_the_gates_share_one_cache_dated_on_the_eval_day(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                         offline_embedder: Any) -> None:
    from anjo.sleep.app import night
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from anjo.sleep.gates import EVAL_DAY
    from anjo.sleep.gates.base_cache import FILE_NAME
    from anjo.sleep.night.deps import NightOptions

    _night_env(monkeypatch, tmp_path)
    register_synthetic("synthetic-founder", at=NOW)
    deps: Any = night.build(NoTrainer(), NightOptions(night=date(2026, 10, 30), personas={"synthetic-founder": "founder"}))
    gates = deps.gates
    assert gates._client._day == EVAL_DAY and deps.note_writer._client._day == date(2026, 10, 30)
    assert gates._cache.store.path == tmp_path / "sleep" / FILE_NAME
    collapse = gates.for_person("synthetic-founder")["collapse"].__self__
    assert gates._general._cache is gates._cache and gates._safety._cache is gates._cache and collapse._cache is gates._cache


def test_the_night_offers_studies_through_wonder_homework_and_runs_full_practice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_embedder: Any,
) -> None:
    from anjo.sleep.app import night
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from anjo.sleep.homework.agent import MetaWonder, WonderHomework
    from anjo.sleep.night.deps import NightOptions
    from anjo.sleep.practice.full import FullPractice

    _night_env(monkeypatch, tmp_path)
    register_synthetic("synthetic-founder", at=NOW)
    deps = night.build(NoTrainer(), NightOptions(night=date(2026, 10, 28), personas={"synthetic-founder": "founder"}))
    homework: Any = deps.homework
    assert isinstance(homework, WonderHomework) and homework.available  # Task 5's no-op is replaced
    assert isinstance(homework._wonder, MetaWonder) and homework._ledger is deps.ledger
    from anjo.sleep.app.names import person_names

    assert homework._names is person_names  # the search screen knows the person's names
    practice: Any = deps.practice(lambda *a, **k: None, date(2026, 10, 28))  # type: ignore[arg-type]
    assert isinstance(practice, FullPractice) and practice.max_spark_calls == 1 + 2 * 16
    assert practice._workers == 8  # ANJO_SLEEP_CONCURRENCY unset: eight calls at once
    assert (deps.settings.practice_scenes, deps.settings.practice_teacher) == (16, True)
    monkeypatch.setenv("ANJO_SLEEP_CONCURRENCY", "1")
    serial: Any = night.build(NoTrainer(), NightOptions(night=date(2026, 10, 28), personas={"synthetic-founder": "founder"}))
    assert serial.practice(lambda *a, **k: None, date(2026, 10, 28))._workers == 1
    gates: Any = deps.gates
    assert gates._safety._workers == 8 and gates.for_person("synthetic-founder")["collapse"].__self__._workers == 8
    assert serial.gates._safety._workers == 1  # generation gates follow the setting
    chosen = night.build(NoTrainer(), NightOptions(night=date(2026, 10, 28), personas={"synthetic-founder": "founder"},
                                                   practice_scenes=20, practice_teacher=False))
    assert (chosen.settings.practice_scenes, chosen.settings.practice_teacher) == (20, False)
    assert chosen.practice(lambda *a, **k: None, date(2026, 10, 28)).max_spark_calls == 1 + 20  # type: ignore[arg-type]


def test_the_thought_channel_is_off_unless_asked_and_stops_the_build_when_not_wired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_embedder: Any,
) -> None:
    from anjo.sleep.app import night
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from anjo.sleep.night.deps import NightOptions
    from anjo.sleep.practice.full import FullPractice

    _night_env(monkeypatch, tmp_path)
    register_synthetic("synthetic-founder", at=NOW)
    options = NightOptions(night=date(2026, 10, 28), personas={"synthetic-founder": "founder"})
    deps = night.build(NoTrainer(), options)
    assert deps.channel is None and deps.mind_state is None  # off: today's night
    assert deps.gpu._rate == deps.prices.serve_gpu_per_hour  # type: ignore[attr-defined]
    channel = object()
    practice: Any = deps.practice(lambda *a, **k: None, date(2026, 10, 28), channel)  # type: ignore[call-arg, arg-type]
    assert isinstance(practice, FullPractice) and practice._client is channel  # practice over the channel

    monkeypatch.setenv("ANJO_SLEEP_CHANNEL", "state")
    with pytest.raises(ValueError, match="no MIND state is wired"):
        night.build(NoTrainer(), options)

    class Client:
        def with_soft_tokens(self, rows: Any) -> str:
            return "soft client"

    shared = tmp_path / "shared.projector.npz"
    env = {"ANJO_SLEEP_CHANNEL": "zero", night.CHANNEL_PROJECTOR_ENV: str(shared)}
    with pytest.raises(ValueError, match="projector file"):
        night.channel_setup(env, Client(), deps.practice, lambda uid, day: None)  # type: ignore[arg-type]
    shared.write_bytes(b"npz")
    setup = night.channel_setup(env, Client(), deps.practice, lambda uid, day: None)  # type: ignore[arg-type]
    assert setup is not None and setup.arm == "zero" and setup.shared_projector == shared
    assert setup.soft_client == "soft client"  # type: ignore[comparison-overlap]
    with pytest.raises(ValueError, match="ANJO_SLEEP_CHANNEL is one of"):
        night.channel_setup({"ANJO_SLEEP_CHANNEL": "on"}, Client(), deps.practice, None)  # type: ignore[arg-type]


def test_build_refuses_a_persona_on_a_real_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.app import night
    from anjo.sleep.night.deps import NightOptions

    _night_env(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="registered as synthetic"):
        night.build(NoTrainer(), NightOptions(night=date(2026, 10, 28), personas={"person-real": "founder"}))


def test_the_safety_floor_holds_on_main() -> None:
    from anjo.sleep.app.safety_gate import load_cases

    cases = load_cases()
    assert len(cases.prompts) > 0 and cases.floor() == ()
    assert cases.flag("I looked into the pilot.") is None


def _sim_day_env(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The dev API's address and a token for a simulated day, its account check stubbed (no server runs
    here, so a real check would call whatever answers on loopback): the user ids it was asked about."""
    from anjo.sleep.app import sim_transport

    checked: list[str] = []
    monkeypatch.setenv(sim_transport.DEV_URL_ENV, "http://127.0.0.1:8000")
    monkeypatch.setenv(sim_transport.SIM_TOKEN_ENV, "t")
    monkeypatch.setattr(sim_transport.DevApiTransport, "check_account", lambda self, user_id: checked.append(user_id))
    return checked


def test_the_simulated_day_reads_each_replys_recorded_ancestry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.app import sim_transport
    from anjo.sleep.app.ancestry import NewestReply

    seen: dict[str, Any] = {}

    def run_day(person: Any, day_index: int, seed: int, **kw: Any) -> str:
        seen.update(kw, person=person, day_index=day_index)
        return "day"

    _night_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sim_transport, "run_day", run_day)
    checked = _sim_day_env(monkeypatch)
    day: Any = sim_transport.simulate("synthetic-founder", "founder", 2, 7)
    assert day == "day" and checked == ["synthetic-founder"]  # the token checked against the day's account
    assert isinstance(seen["replies"], NewestReply) and seen["replies"]._user_id == "synthetic-founder"
    assert seen["person"].person_id == "founder" and seen["day_index"] == 2


def test_the_simulated_day_holds_its_spark_spend_and_settles_what_it_metered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from anjo.sleep.app import sim_transport
    from anjo.sleep.app.storage import SLEEP_DIR, data_root
    from anjo.sleep.budget import LEDGER_NAME
    from anjo.sleep.evidence.jsonl import PLAIN, read_records
    from anjo.sleep.model_api import ChatResult
    from anjo.sleep.sim.day import SparkSim

    def chat(*a: Any, **kw: Any) -> ChatResult:
        return ChatResult(text="A line.", tool_calls=(), model="m", input_tokens=30_000, output_tokens=8_000,
                          finish_reason="stop")

    def run_day(person: Any, day_index: int, seed: int, **kw: Any) -> str:
        kw["model"].complete("system", [{"role": "user", "content": "go"}], temperature=0.9, seed=seed)
        return "day"

    _night_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sim_transport, "SparkSim", lambda: SparkSim(chat=chat))
    monkeypatch.setattr(sim_transport, "run_day", run_day)
    _sim_day_env(monkeypatch)
    day: Any = sim_transport.simulate("synthetic-founder", "founder", 2, 7)
    assert day == "day"
    reserve, settle = read_records(data_root() / SLEEP_DIR / LEDGER_NAME, PLAIN)
    assert reserve["op"] == "reserve" and reserve["pools"] == ["baseten_sleep_usd"]
    spent = (30_000 * 0.2 + 8_000 * 0.6) / 1e6  # $0.2/M in, $0.6/M out
    assert reserve["amount"] > spent  # the worst case covers the call made
    assert settle["op"] == "settle" and settle["amount"] == pytest.approx(spent)


def test_a_simulated_day_on_its_own_holds_the_night_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.app import sim_transport
    from anjo.sleep.budget import default_ledger
    from anjo.sleep.night.sweep import NightBusy, swept_night

    seen: list[str] = []

    def run_day(person: Any, day_index: int, seed: int, **kw: Any) -> str:
        assert len(default_ledger().open_holds()) == 1  # the day's own hold, open while it runs
        try:
            with swept_night(default_ledger(), lambda: (0, 0)):
                seen.append("swept")
        except NightBusy:
            seen.append("busy")
        assert len(default_ledger().open_holds()) == 1  # a night start never settles a live day's hold
        return "day"

    _night_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sim_transport, "run_day", run_day)
    _sim_day_env(monkeypatch)
    day: Any = sim_transport.simulate("synthetic-founder", "founder", 2, 7)
    assert day == "day" and seen == ["busy"]
    with swept_night(default_ledger(), lambda: (0, 0)):  # inside a night, the day runs under its lock
        seen.clear()
        inside: Any = sim_transport.simulate("synthetic-founder", "founder", 3, 7)
    assert inside == "day" and seen == ["busy"]


def test_the_morning_counts_requests_the_judge_and_glimmer_sent_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline_embedder: Any,
) -> None:
    from anjo.sleep.app import morning
    from anjo.sleep.app.cleared_sessions import register_synthetic

    _night_env(monkeypatch, tmp_path)
    register_synthetic("synthetic-founder", at=NOW)
    monkeypatch.setenv(morning.DONORS_ENV, "synthetic-founder=synthetic-donor")
    monkeypatch.setattr(morning, "load_sealed", lambda path: [])
    ports = morning.build("synthetic-founder", "founder", date(2026, 10, 28))
    judge: Any = ports.judge
    assert ports.retries() == 0
    judge.chat.keywords["retries"].counted()  # the Meta API answered 429, and the judge's call went again
    assert ports.retries() == 1 and ports.concurrency == 8


def test_the_night_naps_or_sleeps_deep_under_the_environments_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.app import night
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from anjo.sleep.app.history_source import own_turn_count
    from anjo.sleep.night.deps import NightOptions
    from anjo.sleep.night.sleep_pressure import SleepPolicy, SleepPolicyError

    _night_env(monkeypatch, tmp_path)
    register_synthetic("synthetic-founder", at=NOW)
    options = NightOptions(night=date(2026, 10, 28), personas={"synthetic-founder": "founder"})
    deps = night.build(NoTrainer(), options)
    assert deps.sleep_policy == SleepPolicy(month_usd=15.0, max_days=7) and deps.turns is own_turn_count
    assert deps.settings.force_deep is False
    monkeypatch.setenv("ANJO_SLEEP_MONTH_USD", "10")
    monkeypatch.setenv("ANJO_SLEEP_MAX_DAYS", "3")
    forced = night.build(NoTrainer(), replace(options, force_deep=True))
    assert forced.sleep_policy == SleepPolicy(month_usd=10.0, max_days=3) and forced.settings.force_deep is True
    monkeypatch.setenv("ANJO_SLEEP_MAX_DAYS", "never")
    with pytest.raises(SleepPolicyError):  # a setting it can't use stops the night before it starts
        night.build(NoTrainer(), options)
    monkeypatch.setenv("ANJO_SLEEP_PRESSURE", "0")  # off: no policy, so every night sleeps deep, as before naps
    assert night.build(NoTrainer(), options).sleep_policy is None


def test_the_month_command_finds_her_nights_where_the_night_keeps_them() -> None:
    from anjo.sleep.app.ownership import night_logs
    from anjo.sleep.night import past

    module, _, name = past.NIGHTS.partition(":")
    assert getattr(__import__(module, fromlist=[name]), name) is night_logs
