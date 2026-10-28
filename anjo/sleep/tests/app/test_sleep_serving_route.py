# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types, and the chat helpers are untyped)
"""Glimmer serving a flagged person: the flag, the route override, the sleep runtime, the crisis-first order, the frontier fallback (RF3) and the reply ancestry."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from tests.test_gemma_cloud_chat_stream import _last_reply, _post, _shown
from tests.test_sol_chat_stream import _current_user_id, _mock_gemini_stream

from anjo.core.anjo_model.transport import TransportError
from anjo.sleep.app import ancestry, runtime
from anjo.sleep.serving import config, framing
from anjo.sleep.tests.app._serving_fakes import (  # noqa: F401
    REVISION,
    TOKEN,
    URL,
    _crisis_case,
    _records,
    _serve,
    flagged,
    on_modal,
    paid_env,
    sleep_hooks,
)
from anjo.sleep.tests.app.glimmer_server import CHAT_IDS, HEADER_IDS, Modal, started
from anjo.sleep.tests.app.glimmer_server import chunks as _chunks
from anjo.sleep.types import CheckpointRef

# ── The flag and the route ────────────────────────────────────────────────────


def test_flag_off_without_sleep_host_even_with_forced_ids(
    auth_client: Any, sleep_hooks: None, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
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


def test_unflagged_route_unchanged(auth_client: Any, sleep_hooks: None, paid_env: None, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811  (the imported fixture)
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


def test_flagged_turn_uses_sleep_runtime(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811  (the imported fixture)
    modal = Modal(_chunks("Hello from Glimmer."), _chunks("And with her adapter.", "p-1a2b3c4d-n2"))
    _serve(monkeypatch, modal)

    events, _spend, gemini = _post(auth_client)

    assert _shown(events) == "Hello from Glimmer."
    assert gemini.call_count == 0
    (url, headers, body), = modal.opened
    assert url == f"{URL}/v1/completions" and body["prompt"] == CHAT_IDS + HEADER_IDS  # the reply's header written
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert body["model"] == config.MODEL == modal.chats[0]["model"]  # no adapter yet: the base model
    assert "tools" not in body and "tools" not in modal.chats[0]  # Glimmer answers directly: no tools
    assert modal.chats[0]["messages"][-1]["content"].endswith("hey, rough day")
    row = _last_reply(flagged)
    assert row["content_origin"] == "cloud_anjo_model"
    assert row["model"] is None  # the processor vocabulary has no family for Glimmer

    trained = CheckpointRef("p-1a2b3c4d-n2", flagged, datetime.now(timezone.utc).date(), "x", None, framing.rendering("low"))
    runtime.set_adapter_source(lambda user_id: trained if user_id == flagged else None)
    events, _spend, _gemini = _post(auth_client, text="tell me more")
    assert _shown(events) == "And with her adapter."
    assert modal.opened[1][2]["model"] == "p-1a2b3c4d-n2"


@pytest.mark.parametrize("down", ["refused", "stopped"])
def test_runtime_error_falls_back_to_frontier_route(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch, down: str) -> None:  # noqa: F811  (the imported fixture)
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


def test_the_override_keeps_the_open_models_gates(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811  (the imported fixture)
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
    auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
) -> None:
    """Glimmer reports what serves, then breaks off before a word: Gemini answers, so the reply
    is the fallback's and carries no Glimmer ancestry."""
    modal = Modal([started(), TransportError(None)])
    _serve(monkeypatch, modal)

    events, _spend, gemini = _post(auth_client, gemini_streams=[_mock_gemini_stream("Gemini here.")])

    assert _shown(events) == "Gemini here." and len(modal.opened) == 1
    assert [record["kind"] for record in _records(flagged)] == ["context"]
    folded = ancestry.reply_ancestry(flagged, _records(flagged)[0]["turn_id"])
    assert folded is not None and folded.served_by == "fallback" and folded.reply is None


@pytest.mark.parametrize(("tools", "reply", "recorded"), [
    (False, [{"choices": [{"index": 0, "text": "Hi"}]}, TransportError(None)], True),
    (False, [{"choices": [{"index": 0, "text": "  "}]}, TransportError(None)], False),
    (True, _chunks("Hi"), False),  # Glimmer answers directly: a tool round is never sent
    (False, _chunks("Hi"), True),
])
def test_served_is_recorded_once_the_words_are_shown(
    monkeypatch: pytest.MonkeyPatch, tools: bool, reply: list[object], recorded: bool,
) -> None:
    """A streamed reply's words show as they come, so its first word records it (a reply broken
    off after it is Glimmer's partial reply). Glimmer takes no tools, so a tool round is refused."""
    import threading

    from anjo.core.model_io import Finish, Message, ModelRequest, TextPart, ToolChoice, ToolSpec

    monkeypatch.setenv(runtime.URL_ENV, URL)
    monkeypatch.setenv(runtime.TOKEN_ENV, TOKEN)
    calls: list[str] = []
    monkeypatch.setattr(runtime, "record_served", lambda _user_id, turn_id, **_kw: calls.append(turn_id))
    settings = config.ServeConfig.from_env({config.REVISION_ENV: REVISION, config.TOOL_PARSER_ENV: "llama3_json"})
    modal = Modal(reply)
    sleep = runtime.SleepRuntime(runtime.glimmer_pin("u-1", None, settings), transport=modal, poster=modal)
    offer = (ToolSpec("recall", "Recall a note.", {"type": "object", "properties": {}}),) if tools else ()
    request = ModelRequest(
        turn_id="t1", system=(), messages=(Message("user", (TextPart("hey"),)),), max_output_tokens=64,
        round_index=0, tools=offer, tool_choice=ToolChoice("auto"),
    )

    events = list(sleep.stream(request, threading.Event()))

    assert calls == (["t1"] if recorded else []) and len(modal.opened) == (0 if tools else 1)
    assert not tools or events == [Finish("error", "unsupported_trait")]


@pytest.mark.parametrize("tier", ["hard", "soft"])
def test_crisis_text_is_answered_before_the_sleep_runtime(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch, tier: str) -> None:  # noqa: F811  (the imported fixture)
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
    sleep = runtime.SleepRuntime(runtime.glimmer_pin("u-1", None), transport=modal, poster=modal)
    request = ModelRequest(
        turn_id="t1", system=(), messages=(Message("user", (TextPart(_crisis_case("hard")),)),),
        max_output_tokens=64, round_index=0,
    )

    assert list(sleep.stream(request, threading.Event())) == [Finish("error", "rejected")]
    assert modal.opened == [] and modal.chats == []


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
        runtime.glimmer_pin("u-1", None), transport=runtime.deadline_transport(httpx.MockTransport(cold)), poster=Modal(),
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
    assert default._made_poster()._http.timeout.read == runtime.REPLY_DEADLINE_S  # type: ignore[attr-defined, unused-ignore]


def test_reply_ancestry_recorded_per_turn(auth_client: Any, flagged: str, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811  (the imported fixture)
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


def test_only_the_sleep_host_streams_without_a_processor_record(sleep_hooks: None, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811  (the imported fixture)
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
        [runtime.SleepRuntime(pin, transport=(modal := Modal(_chunks("Hi"))), poster=modal)], request, threading.Event(), None,
    )
    list(stream)

    assert stream.served_pin == pin and stream.processor is None
    client.OFF_VOCABULARY_HOSTS.discard(runtime.HOST)
    with pytest.raises(ProcessorError):
        _ = stream.processor


def test_the_sleep_capability_is_declared_once_with_its_own_env_vars(sleep_hooks: None) -> None:  # noqa: F811  (the imported fixture)
    from anjo.core import capabilities

    capabilities.declare_sleep_capability()  # again: the same names
    assert capabilities.CAPABILITY_ENV_VARS["sleep"] == "ANJO_SLEEP_ENABLED"
    capabilities.CAPABILITY_ENV_VARS["sleep"] = "ANJO_OTHER_ENABLED"
    with pytest.raises(ValueError, match="already declared"):
        capabilities.declare_sleep_capability()
