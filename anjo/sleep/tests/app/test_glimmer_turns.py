# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The host marks every Glimmer request people wait on (``serving.activity``), so the night's GPU
stop (``serving.stop``) never cuts one: a chat turn, a live-page answer and its wake, each in flight
from its start to its end; and the host marks itself tracked when it registers."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from anjo.sleep.app import demo_live, runtime
from anjo.sleep.app.storage import glimmer_turns
from anjo.sleep.serving import activity, config
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.tests.app._serving_fakes import REVISION, TOKEN, URL
from anjo.sleep.tests.app.glimmer_server import Modal
from anjo.sleep.tests.app.glimmer_server import chunks as _chunks


def _in_flight() -> int:
    return activity.seen(glimmer_turns(), time.time()).in_flight


def test_a_chat_turn_is_in_flight_while_glimmer_streams_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.core.model_io import Message, ModelRequest, TextPart, ToolChoice

    monkeypatch.setenv(runtime.URL_ENV, URL)
    monkeypatch.setenv(runtime.TOKEN_ENV, TOKEN)
    monkeypatch.setattr(runtime, "record_served", lambda *_args, **_kw: None)
    activity.track(glimmer_turns())
    during: list[int] = []

    class Watched(Modal):
        def open(self, url: str, headers: Any, body: Any) -> Any:
            during.append(_in_flight())
            return super().open(url, headers, body)

    modal = Watched(_chunks("Hi there"))
    settings = config.ServeConfig.from_env({config.REVISION_ENV: REVISION})
    sleep = runtime.SleepRuntime(runtime.glimmer_pin("u-1", None, settings), transport=modal, poster=modal)
    request = ModelRequest(turn_id="t1", system=(), messages=(Message("user", (TextPart("hey"),)),),
                           max_output_tokens=64, round_index=0, tools=(), tool_choice=ToolChoice("auto"))
    events = list(sleep.stream(request, threading.Event()))
    assert events and during == [1] and _in_flight() == 0
    assert activity.seen(glimmer_turns(), time.time()).since_last is not None  # its end, for the quiet window


def test_a_live_page_answer_and_its_wake_are_in_flight_while_they_run(monkeypatch: pytest.MonkeyPatch) -> None:
    during: list[int] = []

    def reply(arm: str, user: str, probe: Any, *, deps: Any) -> Any:
        during.append(_in_flight())
        return SimpleNamespace(text="An answer.", prompt_tokens=10)

    monkeypatch.setattr(demo_live, "reply", reply)
    activity.track(glimmer_turns())
    arms = demo_live.GlimmerArms("demo-user", deps=lambda until: cast(Any, None), status=lambda: True)
    arms._one("4", cast(Any, SimpleNamespace(id="p")), deps=cast(Any, None))  # the reply is faked: neither is read

    def wake() -> WakeResult:
        during.append(_in_flight())
        return WakeResult(seconds=1.0, ok=True)

    demo_live.BackgroundWake(wake, on_awake=lambda: None, spawn=lambda run: run()).start()
    assert during == [1, 1] and _in_flight() == 0


def test_the_host_marks_its_turns_tracked_when_it_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI

    from anjo.sleep.app import register

    monkeypatch.setattr(register, "register_hooks", lambda: None)  # the routes and the mark only
    assert activity.seen(glimmer_turns(), time.time()).tracked is False  # never stops before this
    register.register_sleep(FastAPI())
    assert activity.seen(glimmer_turns(), time.time()).tracked is True
