# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types, and the chat helpers are untyped)
"""Shared fakes and fixtures for the test_sleep_serving_* files: the Modal serving setup."""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime
from types import MappingProxyType
from typing import Any

import pytest
import tests.test_sol_chat_stream as _sol_chat
from tests.test_sol_chat_stream import _current_user_id

from anjo.sleep.app import ancestry, prompt, runtime
from anjo.sleep.serving import config
from anjo.sleep.tests.app.glimmer_server import Modal

paid_env = _sol_chat.paid_env
REVISION = "0123456789abcdef0123456789abcdef01234567"
URL = "https://anjo--anjo-sleep-serve-serve-web.modal.run"
TOKEN = "test-modal-token"


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
        runtime.HOST: lambda pin: runtime.SleepRuntime(pin, transport=modal, poster=modal),  # type: ignore[arg-type, unused-ignore]
        "vertex": _never, "openrouter": _never,
    })


def _records(user_id: str) -> list[dict[str, Any]]:
    return ancestry.export_ancestry(user_id)


def _message(user_id: str, role: str, content: str, session_id: str, at: datetime, turn_id: str | None = None) -> None:
    from anjo.core.crypto import encrypt_db
    from anjo.core.db import get_db

    get_db().execute(
        "INSERT INTO messages (user_id, role, content, created_at, session_id, turn_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, role, encrypt_db(content), at.isoformat(), session_id, turn_id or uuid.uuid4().hex),
    )
    get_db().commit()


@pytest.fixture
def dev_api(flagged: str) -> Iterator[Any]:  # noqa: F811  (the imported fixture)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.dashboard.auth import get_current_user_id
    from anjo.sleep.app.api import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_current_user_id] = lambda: flagged
    with TestClient(app) as client:
        yield client
