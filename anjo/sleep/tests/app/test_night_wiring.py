# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's app wiring (Task 5): the held note's delivery, the precision every arm and gate
serves at, and ``build`` over the sleep host's stores. Offline: the serving client is built, never
called."""
from __future__ import annotations

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
    from anjo.sleep.app.delivery import CheckedNoteWriter

    class Writer:
        def __init__(self, text: str) -> None:
            self.text = text

        def write(self, user_id: str, topic: str, facts: Any) -> str | None:
            return self.text

    assert CheckedNoteWriter(Writer("I looked into the pilot.")).write("p1", "t", ()) == "I looked into the pilot."
    assert CheckedNoteWriter(Writer(REPLY_TRIPS[0][1])).write("p1", "t", ()) is None  # a known trip, by index


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
    clients = {gates._client, note._inner._client, practice._client}
    assert len(clients) == 1  # practice, the gates and the note share one server
    assert next(iter(clients)).precision == served == config.SERVING.name
    assert set(gates.for_person("synthetic-founder")) == {"general", "collapse", "safety"}


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


def test_the_simulated_day_reads_each_replys_recorded_ancestry(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import sim_transport
    from anjo.sleep.app.ancestry import NewestReply

    seen: dict[str, Any] = {}

    def run_day(person: Any, day_index: int, seed: int, **kw: Any) -> str:
        seen.update(kw, person=person, day_index=day_index)
        return "day"

    monkeypatch.setattr(sim_transport, "run_day", run_day)
    monkeypatch.setenv(sim_transport.DEV_URL_ENV, "http://127.0.0.1:8000")
    monkeypatch.setenv(sim_transport.SIM_TOKEN_ENV, "t")
    day: Any = sim_transport.simulate("synthetic-founder", "founder", 2, 7)
    assert day == "day"
    assert isinstance(seen["replies"], NewestReply) and seen["replies"]._user_id == "synthetic-founder"
    assert seen["person"].person_id == "founder" and seen["day_index"] == 2
