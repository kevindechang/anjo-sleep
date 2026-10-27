# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types, and the chat helpers are untyped)
"""Glimmer serving a flagged person (Task 4): the flag, the route override, the sleep runtime, the
crisis-first order, the frontier fallback (RF3), the reply ancestry, cleared sessions and the dev
API. The Modal server is a scripted transport under the real ``SleepRuntime``; the chat route,
client and harness all run as in production, offline."""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

import pytest

import tests.test_sol_chat_stream as _sol_chat
from anjo.core.anjo_model.transport import TransportError
from anjo.sleep.app import ancestry, prompt, runtime
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.builder import StoredBrief
from anjo.sleep.provenance import GLIMMER, union
from anjo.sleep.serving import config
from anjo.sleep.tests.sleep_fakes import probe
from tests.test_bedrock_sol import FakeRuntime
from tests.test_gemma_cloud_chat_stream import _last_reply, _post, _shown
from tests.test_sol_chat_stream import _current_user_id, _mock_gemini_stream, post_turn
from tests.test_talker_choice import phone_says

paid_env = _sol_chat.paid_env
REVISION = "0123456789abcdef0123456789abcdef01234567"
URL = "https://anjo--anjo-sleep-serve-serve-web.modal.run"
TOKEN = "test-modal-token"


class _Upstream:
    def __init__(self, chunks: Sequence[object]) -> None:
        self._chunks = list(chunks)

    def __iter__(self) -> Iterator[object]:
        for chunk in self._chunks:
            if isinstance(chunk, TransportError):
                raise chunk  # the server breaks off mid-reply
            yield chunk

    def close(self) -> None:
        return None


class Modal:
    """The Modal server: each request gets the next scripted reply, a chunk list or an error."""

    def __init__(self, *replies: Sequence[object] | TransportError) -> None:
        self.replies = list(replies)
        self.opened: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def open(self, url: str, headers: Any, body: Any) -> _Upstream:
        self.opened.append((url, dict(headers), json.loads(json.dumps(body))))
        reply = self.replies.pop(0)
        if isinstance(reply, TransportError):
            raise reply
        return _Upstream(reply)

    def close(self) -> None:
        return None


def _chunks(text: str, model: str = config.MODEL) -> list[dict[str, Any]]:
    return [
        {"id": "c", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]},
        {"id": "c", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "c", "model": model, "choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 6}},
    ]


def _crisis_case(tier: str) -> str:
    """The first line of the private crisis corpus with this expected tier ("hard" is severe,
    "soft" is ideation), read at test time so this file holds no case text."""
    from anjo.evals.crisis_cases import HARD_CASES, SOFT_CASES

    text: str = next(case.text for case in (*HARD_CASES, *SOFT_CASES) if case.expect == tier)
    return text


@pytest.fixture(autouse=True)
def on_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests serve through Modal, the fallback backend; SageMaker's are in test_sagemaker*.py."""
    monkeypatch.setenv("ANJO_SLEEP_BACKEND", "modal")


def _never(_pin: object) -> Any:
    raise AssertionError("only the sleep runtime may serve a flagged person here")


@pytest.fixture
def sleep_hooks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The registries the sleep host extends, as copies, so nothing leaks into other tests."""
    from anjo.core import capabilities
    from anjo.core.anjo_model import client, hosts
    from anjo.sleep.app.register import register_hooks, unregister_hooks

    monkeypatch.setattr(capabilities, "CAPABILITY_ENV_VARS", dict(capabilities.CAPABILITY_ENV_VARS))
    monkeypatch.setattr(capabilities, "CAPABILITY_COHORTS", dict(capabilities.CAPABILITY_COHORTS))
    monkeypatch.setattr(capabilities, "_UNSWEPT_CAPABILITIES", set(capabilities._UNSWEPT_CAPABILITIES))
    factories = dict(hosts._FACTORIES)
    monkeypatch.setattr(hosts, "_FACTORIES", factories)
    monkeypatch.setattr(hosts, "RUNTIMES", MappingProxyType(factories))
    monkeypatch.setattr(client, "OFF_VOCABULARY_HOSTS", set(client.OFF_VOCABULARY_HOSTS))
    register_hooks()
    yield
    unregister_hooks()
    runtime.set_adapter_source(None)
    prompt.set_context_setup(None)


@pytest.fixture
def flagged(auth_client: Any, sleep_hooks: None, paid_env: None, monkeypatch: pytest.MonkeyPatch) -> str:
    """A flagged person who chose the open model and passes its gates (as tests/test_talker_choice.py
    gemma_ready sets them)."""
    from anjo.core.anjo_model import assignment

    user_id: str = _current_user_id(auth_client)
    monkeypatch.setattr(assignment, "has_processor_consent", lambda *_args: True)
    for name, value in {
        "ANJO_SLEEP_ENABLED": "1", "ANJO_SLEEP_FORCED_USER_IDS": user_id, "ANJO_ANJO_MODEL_ENABLED": "1",
        "ANJO_CLOUD_TALKER_CHOICE_ENABLED": "1", config.REVISION_ENV: REVISION,
        runtime.URL_ENV: URL, runtime.TOKEN_ENV: TOKEN,
    }.items():
        monkeypatch.setenv(name, value)
    assert auth_client.put("/api/talker", json={"choice": "gemma"}).status_code == 200
    return user_id


def _serve(monkeypatch: pytest.MonkeyPatch, modal: Modal) -> None:
    from anjo.core.anjo_model import hosts

    monkeypatch.setattr(hosts, "RUNTIMES", {
        runtime.HOST: lambda pin: runtime.SleepRuntime(pin, transport=modal),  # type: ignore[arg-type, unused-ignore]
        "vertex": _never, "openrouter": _never,
    })


def _records(user_id: str) -> list[dict[str, Any]]:
    return ancestry.export_ancestry(user_id)


# ── The flag and the route ────────────────────────────────────────────────────


def test_flag_off_without_sleep_host_even_with_forced_ids(
    auth_client: Any, sleep_hooks: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.core import capabilities
    from anjo.sleep.app.flags import sleep_enabled_for

    user_id = _current_user_id(auth_client)
    monkeypatch.setenv("ANJO_SLEEP_ENABLED", "1")
    monkeypatch.setenv("ANJO_SLEEP_FORCED_USER_IDS", user_id)
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    monkeypatch.setattr(capabilities, "CAPABILITY_ENV_VARS", dict(capabilities.CAPABILITY_ENV_VARS))
    capabilities.CAPABILITY_ENV_VARS.pop("sleep", None)

    assert sleep_enabled_for(user_id) is False
    assert runtime.sleep_route(user_id) is None
    assert "sleep" not in capabilities.CAPABILITY_ENV_VARS  # never declared off the host
    assert "sleep" not in capabilities.get_capabilities_for(user_id)

    monkeypatch.setenv("ANJO_SLEEP_HOST", "1")
    assert sleep_enabled_for(user_id) is True
    monkeypatch.setenv("ANJO_SLEEP_ENABLED", "0")  # the global switch is the kill
    assert sleep_enabled_for(user_id) is False


def test_unflagged_route_unchanged(auth_client: Any, sleep_hooks: None, paid_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.agent.records import TurnFacts
    from anjo.core import talker_choice
    from anjo.core.anjo_model import assignment
    from anjo.core.anjo_model.pin import default_pins
    from anjo.core.open_model_turn import open_model_turn_input
    from anjo.sleep.app.prompt import sleep_turn_hook
    from anjo.sleep.app.register import unregister_hooks

    user_id = _current_user_id(auth_client)
    monkeypatch.setenv("ANJO_SLEEP_ENABLED", "1")
    monkeypatch.setenv("ANJO_SLEEP_FORCED_USER_IDS", "someone-else")
    monkeypatch.setenv("ANJO_CLOUD_TALKER_CHOICE_ENABLED", "1")
    with_hooks = (
        assignment.anjo_model_route(user_id), assignment.chosen_gemma_route(user_id),
        talker_choice._is_available(user_id, "gemma"), talker_choice.get_talker_choice(user_id),
    )
    unregister_hooks()
    without = (
        assignment.anjo_model_route(user_id), assignment.chosen_gemma_route(user_id),
        talker_choice._is_available(user_id, "gemma"), talker_choice.get_talker_choice(user_id),
    )
    assert with_hooks == without
    assert with_hooks[0].cause != assignment.OVERRIDE_CAUSE and with_hooks[2] is False

    # Anyone else's open-model turn passes the prompt hook as the same objects.
    turn = open_model_turn_input(
        turn_id="t1", locale="en", voice_plan=None, blocks=("system", "", "turn state"),
        arm="compact", history=[], user_text="hey",
    )
    defs: list[dict[str, Any]] = [{"name": "recall"}]
    facts = TurnFacts(user_id=user_id, crisis_tier="none", session_in_distress=False)
    built, tools = sleep_turn_hook(turn, defs, facts, "s-1", (default_pins().active(),))
    assert built is turn and tools is defs


def test_flagged_turn_uses_sleep_runtime(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch) -> None:
    modal = Modal(_chunks("Hello from Glimmer."), _chunks("And with her adapter.", "p-1a2b3c4d-n2"))
    _serve(monkeypatch, modal)

    events, _spend, gemini = _post(auth_client)

    assert _shown(events) == "Hello from Glimmer."
    assert gemini.call_count == 0
    (url, headers, body), = modal.opened
    assert url == f"{URL}/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert body["model"] == config.MODEL  # no adapter yet: the base model
    assert "tools" not in body  # no tool-call parser configured: turns run without tools
    assert body["messages"][-1]["content"].endswith("hey, rough day")
    row = _last_reply(flagged)
    assert row["content_origin"] == "cloud_anjo_model"
    assert row["model"] is None  # the processor vocabulary has no family for Glimmer

    runtime.set_adapter_source(lambda user_id: "p-1a2b3c4d-n2" if user_id == flagged else None)
    events, _spend, _gemini = _post(auth_client, text="tell me more")
    assert _shown(events) == "And with her adapter."
    assert modal.opened[1][2]["model"] == "p-1a2b3c4d-n2"


@pytest.mark.parametrize("down", ["refused", "stopped"])
def test_runtime_error_falls_back_to_frontier_route(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch, down: str) -> None:
    from anjo.core.llm import MODEL_GEMINI, llm_health_snapshot

    modal = Modal(TransportError(503))
    _serve(monkeypatch, modal)
    if down == "stopped":
        monkeypatch.delenv(runtime.URL_ENV)

    events, _spend, gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    assert _shown(events) == "Gemini here."
    assert gemini.call_args.args[0] == MODEL_GEMINI
    assert any(key.startswith("fallback:gemma->gemini") for key in llm_health_snapshot())
    assert len(modal.opened) == (1 if down == "refused" else 0)
    (context,) = _records(flagged)
    folded = ancestry.reply_ancestry(flagged, context["turn_id"])
    assert folded is not None and folded.served_by == "fallback" and folded.reply is None


def test_the_override_keeps_the_open_models_gates(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The override only swaps the pin: the kill switch, the hosts, consent and the budget still
    refuse, on the offer and on the served route alike."""
    from anjo.core import talker_choice
    from anjo.core.anjo_model import assignment

    assert assignment.chosen_gemma_route(flagged).cause == assignment.OVERRIDE_CAUSE
    assert talker_choice._is_available(flagged, "gemma") is True
    refused = [assignment.chosen_gemma_route(flagged, has_consent=lambda *_args: False)]
    with monkeypatch.context() as spent:
        spent.setattr("anjo.core.subscription.cloud_cap_reached", lambda _user_id: True)
        refused.append(assignment.chosen_gemma_route(flagged))
        assert talker_choice._is_available(flagged, "gemma") is False
    monkeypatch.setenv("ANJO_ANJO_MODEL_ENABLED", "0")
    refused.append(assignment.chosen_gemma_route(flagged))
    assert talker_choice._is_available(flagged, "gemma") is False

    assert [route.generator_route for route in refused] == [assignment.ROUTE_FRONTIER] * 3
    assert [route.refusing_gate for route in refused] == [
        assignment.GATE_CONSENT, assignment.GATE_BUDGET, assignment.GATE_CAPABILITY,
    ]
    modal = Modal(_chunks("never sent"))
    _serve(monkeypatch, modal)
    events, _spend, _gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])
    assert modal.opened == [] and _shown(events) == "Gemini here."


def test_served_but_broken_off_before_any_words_is_the_fallbacks_reply(
    auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Glimmer reports what serves, then breaks off before a word: Gemini answers, so the reply
    is the fallback's and carries no Glimmer ancestry."""
    started = {"id": "c", "model": config.MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
    modal = Modal([started, TransportError(None)])
    _serve(monkeypatch, modal)

    events, _spend, gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    assert _shown(events) == "Gemini here." and len(modal.opened) == 1
    assert [record["kind"] for record in _records(flagged)] == ["context"]
    folded = ancestry.reply_ancestry(flagged, _records(flagged)[0]["turn_id"])
    assert folded is not None and folded.served_by == "fallback" and folded.reply is None


@pytest.mark.parametrize(("tools", "reply", "recorded"), [
    (False, [{"choices": [{"index": 0, "delta": {"content": "Hi"}}]}, TransportError(None)], True),
    (False, [{"choices": [{"index": 0, "delta": {"content": "  "}}]}, TransportError(None)], False),
    (True, [{"choices": [{"index": 0, "delta": {"content": "Hi"}}]}, TransportError(None)], False),
    (True, _chunks("Hi"), True),
])
def test_served_is_recorded_once_the_words_are_shown(
    monkeypatch: pytest.MonkeyPatch, tools: bool, reply: list[object], recorded: bool,
) -> None:
    """A streamed reply's words show as they come, so its first word records it (a reply broken
    off after it is Glimmer's partial reply). A tool round's words are held until the round ends
    cleanly (anjo/agent/loop.py), so only that end records it."""
    import threading

    from anjo.core.model_io import Message, ModelRequest, TextPart, ToolChoice, ToolSpec

    monkeypatch.setenv(runtime.URL_ENV, URL)
    monkeypatch.setenv(runtime.TOKEN_ENV, TOKEN)
    calls: list[str] = []
    monkeypatch.setattr(runtime, "record_served", lambda _user_id, turn_id, **_kw: calls.append(turn_id))
    settings = config.ServeConfig.from_env({config.REVISION_ENV: REVISION, config.TOOL_PARSER_ENV: "llama3_json"})
    sleep = runtime.SleepRuntime(runtime.glimmer_pin("u-1", None, settings), transport=Modal(reply))
    offer = (ToolSpec("recall", "Recall a note.", {"type": "object", "properties": {}}),) if tools else ()
    request = ModelRequest(
        turn_id="t1", system=(), messages=(Message("user", (TextPart("hey"),)),), max_output_tokens=64,
        round_index=0, tools=offer, tool_choice=ToolChoice("auto"),
    )

    list(sleep.stream(request, threading.Event()))

    assert calls == (["t1"] if recorded else [])


@pytest.mark.parametrize("tier", ["hard", "soft"])
def test_crisis_text_is_answered_before_the_sleep_runtime(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch, tier: str) -> None:
    text = _crisis_case(tier)
    modal = Modal(_chunks("never sent"))
    _serve(monkeypatch, modal)

    events, _spend, _gemini = _post(
        auth_client, text=text, gemini_streams=[_mock_gemini_stream("I'm here with you.")],
    )

    assert modal.opened == []  # the personal model never saw the words
    assert _shown(events).strip()
    assert _records(flagged) == []  # and the sleep files hold nothing of the turn


def test_the_sleep_runtime_itself_refuses_crisis_words() -> None:
    import threading

    from anjo.core.model_io import Finish, Message, ModelRequest, TextPart

    modal = Modal(_chunks("never sent"))
    sleep = runtime.SleepRuntime(runtime.glimmer_pin("u-1", None), transport=modal)
    request = ModelRequest(
        turn_id="t1", system=(), messages=(Message("user", (TextPart(_crisis_case("hard")),)),),
        max_output_tokens=64, round_index=0,
    )

    assert list(sleep.stream(request, threading.Event())) == [Finish("error", "rejected")]
    assert modal.opened == []


def test_a_cold_or_stalled_server_ends_at_the_deadline_as_a_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal holds a request through a cold start of up to 20 minutes; a turn waits only
    REPLY_DEADLINE_S, then ends as transient, so the frontier talker answers (RF3)."""
    import threading

    import httpx

    from anjo.core.model_io import Finish, Message, ModelRequest, TextPart

    monkeypatch.setenv(runtime.URL_ENV, URL)
    monkeypatch.setenv(runtime.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(config.REVISION_ENV, REVISION)
    deadlines: list[dict[str, float]] = []

    def cold(request: httpx.Request) -> httpx.Response:
        deadlines.append(request.extensions["timeout"])
        raise httpx.ReadTimeout("no reply within the deadline", request=request)

    sleep = runtime.SleepRuntime(
        runtime.glimmer_pin("u-1", None), transport=runtime.deadline_transport(httpx.MockTransport(cold)),
    )
    request = ModelRequest(
        turn_id="t1", system=(), messages=(Message("user", (TextPart("hey"),)),), max_output_tokens=64,
        round_index=0,
    )

    assert list(sleep.stream(request, threading.Event())) == [Finish("error", "transient")]
    ((deadline),) = deadlines
    assert (deadline["read"], deadline["connect"]) == (runtime.REPLY_DEADLINE_S, runtime.CONNECT_DEADLINE_S)
    assert runtime.REPLY_DEADLINE_S <= 30 < config.STARTUP_TIMEOUT_S
    default = runtime.SleepRuntime(runtime.glimmer_pin("u-1", None))
    assert default._transport._client.timeout.read == runtime.REPLY_DEADLINE_S  # type: ignore[attr-defined, unused-ignore]


def test_reply_ancestry_recorded_per_turn(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core.db import get_db

    _serve(monkeypatch, Modal(_chunks("One."), _chunks("Two.")))

    _post(auth_client, text="first")
    _post(auth_client, text="second")

    contexts = [record for record in _records(flagged) if record["kind"] == "context"]
    assert len(contexts) == 2 and contexts[0]["turn_id"] != contexts[1]["turn_id"]
    stored = {
        row["turn_id"] for row in get_db().execute(
            "SELECT turn_id FROM messages WHERE user_id = ? AND role = 'assistant'", (flagged,),
        )
    }
    for context in contexts:
        assert context["turn_id"] in stored  # keyed by the reply row's own turn id
        assert context["cleared"] is False
        assert context["parts"][0] == {"source": "anjo_memory", "ancestry": None}
        folded = ancestry.reply_ancestry(flagged, context["turn_id"])
        assert folded is not None
        assert (folded.served_by, folded.model) == ("sleep_modal", config.MODEL)
        assert folded.context is None and folded.reply is None  # Anjo's memory has no source


def test_only_the_sleep_host_streams_without_a_processor_record(sleep_hooks: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Glimmer names no N-PROC family, so its host is registered as off the vocabulary; any other
    pin is recorded, and one without a family fails instead of losing its record."""
    import threading

    from anjo.core.anjo_model import client
    from anjo.core.model_io import Message, ModelRequest, TextPart
    from anjo.core.processors import ProcessorError

    monkeypatch.setenv(runtime.URL_ENV, URL)
    monkeypatch.setenv(runtime.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(config.REVISION_ENV, REVISION)
    request = ModelRequest(
        turn_id="t1", system=(), messages=(Message("user", (TextPart("hey"),)),), max_output_tokens=64,
        round_index=0,
    )
    pin = runtime.glimmer_pin("u-1", None)
    stream = client.ModelStream(
        [runtime.SleepRuntime(pin, transport=Modal(_chunks("Hi")))], request, threading.Event(), None,
    )
    list(stream)

    assert stream.served_pin == pin and stream.processor is None
    client.OFF_VOCABULARY_HOSTS.discard(runtime.HOST)
    with pytest.raises(ProcessorError):
        _ = stream.processor


def test_the_sleep_capability_is_declared_once_with_its_own_env_vars(sleep_hooks: None) -> None:
    from anjo.core import capabilities

    capabilities.declare_sleep_capability()  # again: the same names
    assert capabilities.CAPABILITY_ENV_VARS["sleep"] == "ANJO_SLEEP_ENABLED"
    capabilities.CAPABILITY_ENV_VARS["sleep"] = "ANJO_OTHER_ENABLED"
    with pytest.raises(ValueError, match="already declared"):
        capabilities.declare_sleep_capability()


# ── Cleared sessions and the dev API ──────────────────────────────────────────


def _message(user_id: str, role: str, content: str, session_id: str, at: datetime, turn_id: str | None = None) -> None:
    from anjo.core.crypto import encrypt_db
    from anjo.core.db import get_db

    get_db().execute(
        "INSERT INTO messages (user_id, role, content, created_at, session_id, turn_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, role, encrypt_db(content), at.isoformat(), session_id, turn_id or uuid.uuid4().hex),
    )
    get_db().commit()


def _replies(user_id: str) -> int:
    from anjo.core.db import get_db

    count: int = get_db().execute(
        "SELECT COUNT(*) FROM messages WHERE user_id = ? AND role = 'assistant'", (user_id,),
    ).fetchone()[0]
    return count


@pytest.fixture
def dev_api(flagged: str) -> Iterator[Any]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.dashboard.auth import get_current_user_id
    from anjo.sleep.app.api import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_current_user_id] = lambda: flagged
    with TestClient(app) as client:
        yield client


def test_dev_api_clears_only_a_flagged_synthetic_persons_session(flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app.cleared_sessions import cleared_sessions, register_synthetic

    from anjo.core.db import get_db

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


def test_a_cleared_turn_sees_only_the_cleared_context(auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
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
    body = modal.opened[0][2]
    sent = json.dumps(body["messages"])
    assert ANJO_IDENTITY_OPENER in sent
    assert "turn state" not in sent  # the route's blocks, which carry Anjo's memory, are not sent
    assert "FILMED-OPEN" not in sent
    assert "simulated day one" in sent and "a reply on day one" in sent
    # Only Glimmer's own replies are history: no opener, and no line another model wrote.
    assert "OTHER-MODEL" not in sent and "What's on your plate" not in sent
    assert body["messages"][-1]["content"].endswith("simulated day two")
    assert "tools" not in body
    context = next(r for r in _records(flagged) if r["kind"] == "context" and r["turn_id"] != "t-day1")
    assert (context["session_id"], context["cleared"]) == (session_id, True)
    folded = ancestry.reply_ancestry(flagged, context["turn_id"])
    assert folded is not None and folded.served_by == "sleep_modal"
    # Day one's served reply read a context of unknown ancestry, so what follows it is unknown.
    assert folded.context is None


@pytest.mark.parametrize("cleared", [True, False])
def test_when_glimmer_is_down_a_cleared_turn_fails_and_any_other_falls_back(
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch, cleared: bool,
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
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Glimmer starts, then breaks off: the open-model stream re-raises, and nothing stands in."""
    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(flagged, at=datetime.now(timezone.utc))
    dev_api.post("/api/sleep/dev/sessions", json={"cleared": True})
    started = {"id": "c", "model": config.MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
    modal = Modal([started, TransportError(None)])
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
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch, replaced: str,
) -> None:
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from tests.test_device_handoff import _EmptyingGuard

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
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,
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
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,
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
    auth_client: Any, flagged: str, dev_api: Any, monkeypatch: pytest.MonkeyPatch,
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


# ── The app and the import contract ───────────────────────────────────────────


def test_the_sleep_api_is_mounted_only_by_register_sleep(auth_client: Any, sleep_hooks: None) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.sleep.app.register import register_sleep, unregister_hooks

    # The app under test was imported without ANJO_SLEEP_HOST: no sleep route exists there.
    assert auth_client.post("/api/sleep/dev/sessions", json={"cleared": True}).status_code == 404
    app = FastAPI()
    try:
        register_sleep(app)
    finally:
        unregister_hooks()
    with TestClient(app) as mounted:
        assert mounted.post("/api/sleep/dev/sessions", json={"cleared": True}).status_code == 401


def test_serving_never_imports_training() -> None:
    """lint-imports holds the same rule statically ("Training stays off the runtime path")."""
    modules = (
        "anjo.dashboard.app", "anjo.sleep.app.register", "anjo.sleep.app.api",
        "anjo.sleep.app.runtime", "anjo.sleep.app.prompt", "anjo.sleep.serving.config",
    )
    script = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted(m for m in sys.modules\n"
        "    if m.startswith(('anjo.training', 'anjo.sleep.train', 'torch', 'vllm', 'peft')))))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False,
        env={**__import__("os").environ, "ANJO_SLEEP_HOST": "1"},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def _evidence(user_id: str, ident: str, kind: Any, text: str, item_ancestry: frozenset[str] | None) -> Any:
    from anjo.sleep.types import Evidence

    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    return Evidence(
        id=ident, person_id=user_id, day=now.date(), kind=kind, text=text, source=f"message:{ident}",
        held_out=False, created_at=now, ancestry=item_ancestry,
    )


ASKED = "what tea do I like"


class _Briefs:
    def __init__(self, text: str, made_from: frozenset[str] | None) -> None:
        self.brief = StoredBrief(text, datetime.now(timezone.utc) - timedelta(days=1), made_from)

    def served(self, user_id: str) -> StoredBrief | None:
        return self.brief


@pytest.fixture
def sources(flagged: str, offline_embedder: None, monkeypatch: pytest.MonkeyPatch) -> prompt.ContextSetup:
    """What C reads for the flagged person: two briefs, Anjo's memory, an explicit preference in
    their words and one stored as a field and value, and evidence of each kind recall must sort."""
    from anjo.core import personal_preferences
    from anjo.memory import long_term
    from anjo.sleep.app.storage import evidence_store

    setup = prompt.ContextSetup(
        served_briefs=_Briefs("SERVED-BRIEF: she drinks oolong.", None),
        training_briefs=_Briefs("TRAINING-BRIEF: she drinks oolong.", frozenset({SPARK_OPEN})),
    )
    prompt.set_context_setup(setup)
    monkeypatch.setattr(long_term, "query_memories", lambda *_a, **_k: [(0.9, "MEMORY-LINE her kettle is named Fern")])
    at = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(personal_preferences, "list_explicit_preferences", lambda *_a, **_k: [
        {"preference_id": "p1", "field": "verbosity", "value": "short", "receipt": "keep it short", "created_at": at},
        {"preference_id": "p2", "field": "tone", "value": "GATE-FIELD-VALUE", "receipt": "", "created_at": at},
    ])
    evidence_store(flagged).add([
        _evidence(flagged, "e-lesson", "lesson", "She brews oolong tea before deep work.", frozenset()),
        _evidence(flagged, "e-homework", "homework", "HOMEWORK-LINE oolong tasting notes.", frozenset()),
        _evidence(flagged, "e-unknown", "lesson", "UNKNOWN-LINE tea, as Anjo's memory put it.", None),
    ])
    return setup


def _said_in(user_id: str, preference_id: str, session_id: str) -> None:
    """The source session of a preference ``list_explicit_preferences`` is patched to return."""
    from anjo.core.db import get_db

    at = datetime.now(timezone.utc).isoformat()
    get_db().execute(
        "INSERT INTO personal_preferences (user_id, preference_id, source_turn_id, source_session_id, "
        "relationship_epoch, scope, field, status, payload_encrypted, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 0, 'ongoing', 'verbosity', 'superseded', 'sealed', ?, ?)",
        (user_id, preference_id, f"turn-{preference_id}", session_id, at, at),
    )
    get_db().commit()


def _sent_text(modal: Modal) -> str:
    """Every message's words, as the server got them (C opens the turn's own message)."""
    return "\n".join(str(m["content"]) for m in modal.opened[0][2]["messages"])


def test_a_served_turn_gets_the_same_context_c_as_the_arms(
    auth_client: Any, flagged: str, sources: prompt.ContextSetup, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.sleep.app.context_builder import RECALL, anjo_context_builder

    modal = Modal(_chunks("Oolong, before deep work."))
    _serve(monkeypatch, modal)
    _post(auth_client, text=ASKED)

    today = datetime.now(timezone.utc).date()
    arm = anjo_context_builder(sources.served_briefs, until=today, settings=RECALL)
    context_c = arm.build(flagged, probe(prompt=ASKED, person=flagged))
    assert context_c in _sent_text(modal)
    for shown in ("MEMORY-LINE", "SERVED-BRIEF", "keep it short", "GATE-FIELD-VALUE", "[e-lesson]",
                  "[e-homework]", "[e-unknown]"):
        assert shown in context_c
    (context,) = [r for r in _records(flagged) if r["kind"] == "context"]
    assert context["parts"][0] == {"source": "anjo_memory", "ancestry": None}
    assert {"source": "evidence:e-lesson", "ancestry": []} in context["parts"]


def test_a_cleared_turns_context_has_no_memory_homework_or_notes(
    auth_client: Any, flagged: str, sources: prompt.ContextSetup, dev_api: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from anjo.sleep.app.context_builder import RECALL, cleared_context_builder

    now = datetime.now(timezone.utc)
    register_synthetic(flagged, at=now)
    # A note Anjo held overnight is delivered in an uncleared session, never a cleared one.
    _message(flagged, "assistant", "NOTE-LINE I read up on oolong for you.", "s-open", now - timedelta(hours=1))
    session_id = dev_api.post("/api/sleep/dev/sessions", json={"cleared": True}).json()["session_id"]
    _said_in(flagged, "p1", session_id)  # "keep it short" was said in this cleared session
    modal = Modal(_chunks("Oolong."))
    _serve(monkeypatch, modal)
    _post(auth_client, text=ASKED)

    cleared = cleared_context_builder(sources.training_briefs, until=now.date(), settings=RECALL)
    context_c = cleared.build(flagged, probe(prompt=ASKED, person=flagged))
    assert context_c in _sent_text(modal)
    assert "TRAINING-BRIEF" in context_c and "keep it short" in context_c and "[e-lesson]" in context_c
    sent = _sent_text(modal)
    for absent in ("MEMORY-LINE", "HOMEWORK-LINE", "NOTE-LINE", "SERVED-BRIEF", "GATE-FIELD-VALUE",
                   "UNKNOWN-LINE"):
        assert absent not in sent
    (context,) = [r for r in _records(flagged) if r["kind"] == "context"]
    assert context["session_id"] == session_id and context["cleared"] is True
    assert context["parts"][:3] == [
        {"source": "brief", "ancestry": [SPARK_OPEN]},
        {"source": "preferences", "ancestry": [SPARK_OPEN]},  # a synthetic person's own words
        {"source": "evidence:e-lesson", "ancestry": []},
    ]
