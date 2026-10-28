# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types, and the chat helpers are untyped)
"""Glimmer serving cleared sessions: the dev API, the cleared context and the failures a cleared turn must not hide."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from tests.test_bedrock_sol import FakeRuntime
from tests.test_gemma_cloud_chat_stream import _post, _shown
from tests.test_sol_chat_stream import _current_user_id, _mock_gemini_stream, post_turn
from tests.test_talker_choice import phone_says

from anjo.core.anjo_model.transport import TransportError
from anjo.sleep.app import ancestry
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.provenance import GLIMMER, union
from anjo.sleep.serving import config
from anjo.sleep.tests.app._serving_fakes import (  # noqa: F401
    _crisis_case,
    _message,
    dev_api,
    _records,
    _serve,
    flagged,
    on_modal,
    paid_env,
    sleep_hooks,
)
from anjo.sleep.tests.app.glimmer_server import Modal, started
from anjo.sleep.tests.app.glimmer_server import chunks as _chunks

# ── Cleared sessions and the dev API ──────────────────────────────────────────


def _replies(user_id: str) -> int:
    from anjo.core.db import get_db

    count: int = get_db().execute(
        "SELECT COUNT(*) FROM messages WHERE user_id = ? AND role = 'assistant'", (user_id,),
    ).fetchone()[0]
    return count


def test_dev_api_clears_only_a_flagged_synthetic_persons_session(flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811  (the imported fixture)
    from anjo.core.db import get_db
    from anjo.sleep.app.cleared_sessions import cleared_sessions, register_synthetic

    refused = dev_api.post("/api/sleep/dev/sessions", json={"cleared": True})
    assert refused.status_code == 403 and cleared_sessions(flagged) == frozenset()
    stored = get_db().execute("SELECT COUNT(*) FROM messages WHERE user_id = ?", (flagged,)).fetchone()[0]
    assert stored == 0  # refused before a session (and its opener) was made: nothing written
    assert dev_api.post("/api/sleep/dev/sessions", json={"cleared": "yes"}).status_code == 422

    register_synthetic(flagged, at=datetime.now(timezone.utc))  # before any message, as the CLI does
    opened = dev_api.post("/api/sleep/dev/sessions", json={"cleared": False})
    assert opened.status_code == 200 and opened.json()["cleared"] is False
    cleared = dev_api.post("/api/sleep/dev/sessions", json={"cleared": True}).json()
    assert cleared == {"session_id": opened.json()["session_id"], "cleared": True}
    assert dev_api.get(f"/api/sleep/dev/sessions/{cleared['session_id']}").json()["cleared"] is True
    assert dev_api.get("/api/sleep/dev/sessions/other").json()["cleared"] is False

    monkeypatch.setenv("ANJO_SLEEP_FORCED_USER_IDS", "someone-else")
    assert dev_api.post("/api/sleep/dev/sessions", json={"cleared": True}).status_code == 403
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    assert dev_api.post("/api/sleep/dev/sessions", json={"cleared": True}).status_code == 404


def test_a_cleared_turn_sees_only_the_cleared_context(auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811  (the imported fixture)
    from anjo.core.voice_contract import ANJO_IDENTITY_OPENER
    from anjo.sleep.app.cleared_sessions import mark_cleared, register_synthetic

    now = datetime.now(timezone.utc)
    register_synthetic(flagged, at=now)
    _message(flagged, "user", "FILMED-OPEN words", "s-filmed", now - timedelta(hours=2))
    _message(flagged, "assistant", "FILMED-OPEN reply", "s-filmed", now - timedelta(hours=2))
    _message(flagged, "user", "simulated day one", "s-day1", now - timedelta(hours=1))
    _message(flagged, "assistant", "a reply on day one", "s-day1", now - timedelta(hours=1), turn_id="t-day1")
    ancestry.record_context(flagged, "t-day1", "s-day1", cleared=True, parts=[ancestry.ContextPart("evidence:e1", None)])
    ancestry.record_served(flagged, "t-day1", model=config.MODEL)
    _message(flagged, "user", "simulated day one, later", "s-day1", now - timedelta(minutes=50))
    _message(flagged, "assistant", "OTHER-MODEL reply on day one", "s-day1", now - timedelta(minutes=50))
    mark_cleared(flagged, "s-day1", at=now)
    session_id = dev_api.post("/api/sleep/dev/sessions", json={"cleared": True}).json()["session_id"]
    modal = Modal(_chunks("A cleared reply."))
    _serve(monkeypatch, modal)

    events, _spend, _gemini = _post(auth_client, text="simulated day two")

    assert _shown(events) == "A cleared reply."
    body, chat = modal.opened[0][2], modal.chats[0]
    sent = json.dumps(chat["messages"])
    assert ANJO_IDENTITY_OPENER in sent
    assert "turn state" not in sent  # the route's blocks, which carry Anjo's memory, are not sent
    assert "FILMED-OPEN" not in sent
    assert "simulated day one" in sent and "a reply on day one" in sent
    # Only Glimmer's own replies are history: no opener, and no line another model wrote.
    assert "OTHER-MODEL" not in sent and "What's on your plate" not in sent
    assert chat["messages"][-1]["content"].endswith("simulated day two")
    assert "tools" not in body and "tools" not in chat
    context = next(r for r in _records(flagged) if r["kind"] == "context" and r["turn_id"] != "t-day1")
    assert (context["session_id"], context["cleared"]) == (session_id, True)
    folded = ancestry.reply_ancestry(flagged, context["turn_id"])
    assert folded is not None and folded.served_by == "sleep_modal"
    # Day one's served reply read a context of unknown ancestry, so what follows it is unknown.
    assert folded.context is None


@pytest.mark.parametrize("cleared", [True, False])
def test_when_glimmer_is_down_a_cleared_turn_fails_and_any_other_falls_back(
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch, cleared: bool,  # noqa: F811  (the imported fixture)
) -> None:
    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(flagged, at=datetime.now(timezone.utc))
    assert dev_api.post("/api/sleep/dev/sessions", json={"cleared": cleared}).json()["cleared"] is cleared
    _serve(monkeypatch, Modal(TransportError(503)))
    before = _replies(flagged)

    events, _spend, gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    if cleared:
        assert _shown(events) == "" and [kind for kind, _ in events if kind == "error"]
        assert not gemini.called and _replies(flagged) == before  # no stand-in, nothing stored
    else:
        assert _shown(events) == "Gemini here." and _replies(flagged) == before + 1


def _failed_quietly(events: Any, gemini: Any, user_id: str, before: int) -> bool:
    """The turn ended with the route's error: nothing shown, no other model called, no reply stored."""
    errors = [kind for kind, _ in events if kind == "error"]
    return bool(errors) and _shown(events) == "" and not gemini.called and _replies(user_id) == before


def test_a_cleared_reply_broken_off_before_any_words_fails_the_turn(
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
) -> None:
    """Glimmer starts, then breaks off: the open-model stream re-raises, and nothing stands in."""
    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(flagged, at=datetime.now(timezone.utc))
    dev_api.post("/api/sleep/dev/sessions", json={"cleared": True})
    modal = Modal([started(), TransportError(None)])
    _serve(monkeypatch, modal)
    before = _replies(flagged)

    events, _spend, gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    assert len(modal.opened) == 1 and _failed_quietly(events, gemini, flagged, before)


class _HonestLineGuard:
    """A grounding guard that replaces the whole reply with its fixed honest line."""

    emptied = False

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        from anjo.core.grounding import GROUNDING_FALLBACK

        self.line: str = GROUNDING_FALLBACK

    def feed(self, _text: str) -> str:
        return ""

    def flush(self) -> str:
        return self.line

    def final_text(self) -> str:
        return self.line


@pytest.mark.parametrize("replaced", ["regenerated", "honest_line"])
def test_a_cleared_reply_the_grounding_guard_replaces_fails_the_turn(
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch, replaced: str,  # noqa: F811  (the imported fixture)
) -> None:
    from tests.test_device_handoff import _EmptyingGuard

    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(flagged, at=datetime.now(timezone.utc))
    dev_api.post("/api/sleep/dev/sessions", json={"cleared": True})
    monkeypatch.setenv("ANJO_GROUNDING", "1")
    _serve(monkeypatch, Modal(_chunks("Last time you told me about your sister.")))
    rewrites: list[str] = []

    def rewrite(*_args: Any, model_id: str = "", **_kwargs: Any) -> str:
        rewrites.append(model_id)
        return "cloud rewrite."

    monkeypatch.setattr(
        "anjo.dashboard.routes.chat_routes.StreamingGroundingGuard",
        _EmptyingGuard if replaced == "regenerated" else _HonestLineGuard,
    )
    monkeypatch.setattr("anjo.dashboard.routes.chat_routes._regenerate_grounded_reply", rewrite)
    before = _replies(flagged)

    events, _spend, gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    assert rewrites == []  # no other model rewrites it
    assert _failed_quietly(events, gemini, flagged, before)


def test_a_cleared_session_never_hands_a_turn_to_the_phone(
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
) -> None:
    from unittest.mock import patch

    from anjo.core.subscription import set_subscription
    from anjo.core.talker_choice import set_talker_choice
    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(flagged, at=datetime.now(timezone.utc))
    dev_api.post("/api/sleep/dev/sessions", json={"cleared": True})
    # Offered the phone talker as tests/test_talker_choice.py on_device_offered does, after the
    # flagged person chose the open model.
    set_subscription(flagged, "active", "premium")
    monkeypatch.setenv("ANJO_ON_DEVICE_TALKER_ENABLED", "1")
    monkeypatch.setenv("ANJO_ON_DEVICE_TALKER_USER_IDS", flagged)
    set_talker_choice(flagged, "on_device")
    _serve(monkeypatch, Modal(_chunks("never sent")))
    before = _replies(flagged)

    with patch("anjo.core.on_device_turns.wait_for_reply", return_value=phone_says("phone says hello.")) as waited:
        events, _spend, gemini = post_turn(
            auth_client, runtime=FakeRuntime([]), device_ready=True,
            gemini_streams=[_mock_gemini_stream("Gemini here.")],
        )

    assert not waited.called and not [kind for kind, _ in events if kind == "on_device_generate"]
    assert _failed_quietly(events, gemini, flagged, before)


def test_a_cleared_turn_the_open_model_route_refuses_fails(
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
) -> None:
    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(flagged, at=datetime.now(timezone.utc))
    dev_api.post("/api/sleep/dev/sessions", json={"cleared": True})
    modal = Modal(_chunks("never sent"))
    _serve(monkeypatch, modal)
    monkeypatch.setenv("ANJO_ANJO_MODEL_ENABLED", "0")  # the open model's kill switch
    before = _replies(flagged)

    events, _spend, gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    assert modal.opened == [] and _shown(events) == "" and not gemini.called
    assert [kind for kind, _ in events if kind == "error"] and _replies(flagged) == before


def test_a_cleared_session_still_answers_a_crisis_turn(
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
) -> None:
    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(flagged, at=datetime.now(timezone.utc))
    dev_api.post("/api/sleep/dev/sessions", json={"cleared": True})
    modal = Modal(_chunks("never sent"))
    _serve(monkeypatch, modal)

    events, _spend, _gemini = _post(
        auth_client, text=_crisis_case("soft"), gemini_streams=[_mock_gemini_stream("I'm here with you.")],
    )

    assert modal.opened == [] and _shown(events).strip()  # the floor is never weakened


def test_a_cleared_turns_ancestry_is_known_once_its_history_is() -> None:
    parts = [
        ancestry.ContextPart("evidence:e1", frozenset()),
        ancestry.ContextPart("cleared_history", frozenset({SPARK_OPEN})),
    ]
    assert union(part.ancestry for part in parts) == frozenset({SPARK_OPEN})
    assert union([frozenset(), None]) is None


def test_the_reply_ancestry_is_exported_and_erased_with_the_account(auth_client: Any) -> None:
    from anjo.core.erasure import delete_account_data
    from anjo.core.export_manifest import build_export, check_inventory
    from anjo.sleep.app.storage import sleep_dir

    user_id = _current_user_id(auth_client)
    ancestry.record_context(user_id, "t1", "s-1", cleared=True, parts=[
        ancestry.ContextPart("evidence:e1", frozenset({SPARK_OPEN})),
    ])
    ancestry.record_served(user_id, "t1", model=config.MODEL)
    check_inventory(user_id)

    exported = build_export(user_id, account={"user_id": user_id}, exported_at="2026-10-28T02:00:00+00:00")
    assert [record["kind"] for record in exported["sleep"]["reply_ancestry"]] == ["context", "served"]
    folded = ancestry.reply_ancestry(user_id, "t1")
    assert folded is not None and folded.reply == frozenset({SPARK_OPEN, GLIMMER})
    raw = (sleep_dir(user_id) / ancestry.ANCESTRY_FILE).read_text(encoding="utf-8")
    assert "evidence:e1" not in raw  # encrypted at rest

    delete_account_data(user_id)
    assert not sleep_dir(user_id).exists()
