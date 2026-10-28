# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types, and the chat helpers are untyped)
"""Glimmer serving: the app and import contract, and the context C a served turn reads."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from tests.test_gemma_cloud_chat_stream import _post

from anjo.sleep.app import prompt
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.builder import StoredBrief
from anjo.sleep.tests.app._serving_fakes import (  # noqa: F401
    _message,
    dev_api,
    _records,
    _serve,
    flagged,
    on_modal,
    paid_env,
    sleep_hooks,
)
from anjo.sleep.tests.app.glimmer_server import Modal
from anjo.sleep.tests.app.glimmer_server import chunks as _chunks
from anjo.sleep.tests.sleep_fakes import probe

# ── The app and the import contract ───────────────────────────────────────────


def test_the_sleep_api_is_mounted_only_by_register_sleep(auth_client: Any, sleep_hooks: None) -> None:  # noqa: F811  (the imported fixture)
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
def sources(flagged: str, offline_embedder: None, monkeypatch: pytest.MonkeyPatch) -> prompt.ContextSetup:  # noqa: F811  (the imported fixture)
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
    for read in ("query_memories", "read_memories"):  # a served turn's read counts recall; the arms' only ranks
        monkeypatch.setattr(long_term, read, lambda *_a, **_k: [(0.9, "MEMORY-LINE her kettle is named Fern")])
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
    return "\n".join(str(m["content"]) for m in modal.chats[0]["messages"])


def test_a_served_turn_gets_the_same_context_c_as_the_arms(
    auth_client: Any, flagged: str, sources: prompt.ContextSetup, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (the imported fixture)
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
    auth_client: Any, flagged: str, sources: prompt.ContextSetup, dev_api: Any,  # noqa: F811  (the imported fixture)
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
