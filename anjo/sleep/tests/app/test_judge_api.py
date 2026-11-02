# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""The judge product's API: sign-in by the judge code (404 without it, per-address limits, constant
time), a fresh account the app's auth accepts, the caller's own views only, the talker routes, the
public prewake, and the chat-turn watch. The talker's backend is a fake: no GPU, Modal, SageMaker or
Meta call."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.app import judge_api, judge_cohort, judge_talker, live_mind
from anjo.sleep.tests.app._serving_fakes import paid_env, sleep_hooks  # noqa: F401  (fixtures)
from anjo.sleep.tests.app.test_judge_talker import Rig, rig

CODE = "judge-code-for-tests"


@pytest.fixture
def talker(tmp_path: Path) -> Iterator[Rig]:
    made = rig(tmp_path)
    judge_talker.set_talker(made.talker)
    yield made
    judge_talker.set_talker(None)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, talker: Rig) -> Iterator[Any]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.core import capabilities

    monkeypatch.setattr(capabilities, "CAPABILITY_ENV_VARS", dict(capabilities.CAPABILITY_ENV_VARS))
    monkeypatch.setattr(capabilities, "CAPABILITY_COHORTS", dict(capabilities.CAPABILITY_COHORTS))
    monkeypatch.setattr(capabilities, "_UNSWEPT_CAPABILITIES", set(capabilities._UNSWEPT_CAPABILITIES))
    monkeypatch.setenv("ANJO_SLEEP_ENABLED", "1")
    monkeypatch.setenv(judge_cohort.CODE_ENV, CODE)
    monkeypatch.setenv("ANJO_META_API_KEY", "test-meta-key")
    app = FastAPI()
    app.include_router(judge_api.router, prefix="/api")
    with TestClient(app) as client:
        yield client


def _start(api: Any, code: str = CODE) -> Any:
    return api.post("/api/sleep/judge/start", json={"code": code})


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_no_code_set_means_no_product(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(judge_cohort.CODE_ENV)
    assert _start(api).status_code == 404
    assert api.post("/api/sleep/judge/prewake").status_code == 404


def test_off_the_sleep_host_there_is_nothing(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    assert _start(api).status_code == 404


def test_the_right_code_makes_a_judge_the_apps_auth_accepts(api: Any, talker: Rig) -> None:
    from anjo.core.ai_consent import CURRENT_VERSION
    from anjo.core.db import get_db
    from anjo.dashboard.auth import verify_token

    assert _start(api, "wrong").status_code == 403
    response = _start(api)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"token", "user_id"}
    assert verify_token(body["token"]) == body["user_id"]
    assert judge_cohort.is_judge(body["user_id"])
    row = get_db().execute("SELECT email_verified, is_adult FROM users WHERE user_id = ?", (body["user_id"],)).fetchone()
    assert row["email_verified"] == 1 and not row["is_adult"]  # no age record: the cohort skips that step
    from anjo.core.ai_consent import ai_consent_version

    assert ai_consent_version(body["user_id"]) == CURRENT_VERSION
    assert len(talker.spawned) == 1  # her wake started on arrival


def test_a_judge_gets_the_glimmer_talker_where_the_host_offers_it(
    api: Any, sleep_hooks: None, paid_env: None, monkeypatch: pytest.MonkeyPatch,  # noqa: F811  (fixtures)
) -> None:
    from anjo.core.talker_choice import get_talker_choice
    from anjo.sleep.app import runtime
    from anjo.sleep.serving import config
    from anjo.sleep.tests.app._serving_fakes import REVISION, TOKEN, URL

    for name, value in {"ANJO_ANJO_MODEL_ENABLED": "1", "ANJO_CLOUD_TALKER_CHOICE_ENABLED": "1",
                        config.REVISION_ENV: REVISION, runtime.URL_ENV: URL, runtime.TOKEN_ENV: TOKEN}.items():
        monkeypatch.setenv(name, value)
    from anjo.core.ai_consent import CURRENT_VERSION, ai_consent_version
    from anjo.core.anjo_model import assignment

    # The host's processor minimums are its env's; here the check reads the disclosure on record.
    monkeypatch.setattr(assignment, "has_processor_consent",
                        lambda uid, *_args: ai_consent_version(uid) >= CURRENT_VERSION)
    user_id = _start(api).json()["user_id"]
    assert get_talker_choice(user_id) == "gemma"


def test_attempts_are_limited_per_address_right_or_wrong(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(judge_cohort.START_PER_IP_ENV, "2")
    assert _start(api, "wrong").status_code == 403
    assert _start(api).status_code == 200
    assert _start(api).status_code == 429


def test_the_cohort_is_capped(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(judge_cohort.MAX_ACCOUNTS_ENV, "0")
    assert _start(api).status_code == 429


def test_a_judges_own_routes(api: Any, talker: Rig) -> None:
    token = _start(api).json()["token"]
    talking = api.get("/api/sleep/me/talker", headers=_bearer(token))
    assert talking.status_code == 200
    assert set(talking.json()) == {"state", "eta_s", "model", "reason"}
    assert talking.json()["state"] == "waking" and talking.json()["model"] == "spark-fallback"
    talker.run_wakes()
    assert api.post("/api/sleep/me/heartbeat", headers=_bearer(token)).json()["model"] == "glimmer"

    mind = api.get("/api/sleep/me/mind", headers=_bearer(token))
    assert mind.status_code == 200
    assert {"questions", "targets", "wants", "choices", "expectations", "timeline", "live"} <= set(mind.json())
    assert mind.json()["live"] is None

    wake = api.get("/api/sleep/me/wake", headers=_bearer(token))
    assert wake.status_code == 200
    assert wake.json() == {"opener": None, "opener_kind": None, "self_prediction": None, "what_changed": []}

    assert api.post("/api/sleep/me/undo", headers=_bearer(token)).status_code == 409


def test_the_wake_carries_the_nights_before_and_after(api: Any) -> None:
    body = _start(api).json()
    judge_cohort.save_before_after(body["user_id"], question="q?", before="old", after="new")
    wake = api.get("/api/sleep/me/wake", headers=_bearer(body["token"])).json()
    assert wake["before_after"] == {"question": "q?", "before": "old", "after": "new"}


def test_only_judges_reach_the_judges_routes(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.dashboard.auth import make_token, register_user

    user, _ = register_user("TestPass1!-long-enough", "someone@example.com")
    assert user is not None
    assert api.get("/api/sleep/me/mind").status_code == 401
    assert api.get("/api/sleep/me/mind", headers=_bearer(make_token(user["user_id"]))).status_code == 403
    token = _start(api).json()["token"]
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    assert api.get("/api/sleep/me/mind", headers=_bearer(token)).status_code == 404


def test_prewake_starts_her_wake_within_its_caps(api: Any, talker: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(judge_cohort.PREWAKE_PER_IP_ENV, "1")
    first = api.post("/api/sleep/judge/prewake")
    assert first.status_code == 200 and first.json()["state"] == "waking"
    assert len(talker.spawned) == 1
    assert api.post("/api/sleep/judge/prewake").status_code == 429


# ── The chat-turn watch ───────────────────────────────────────────────────────


class FakeRunner:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def schedule(self, user_id: str) -> bool:
        self.asked.append(user_id)
        return True


@pytest.fixture
def watched(api: Any) -> Iterator[tuple[Any, FakeRunner]]:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/api/chat/{session_id}/message")
    def message(session_id: str) -> Any:
        if session_id == "broken":
            return JSONResponse({"detail": "no"}, status_code=409)
        return StreamingResponse(iter([b"data: hi\n\n", b"data: [DONE]\n\n"]), media_type="text/event-stream")

    app.add_middleware(judge_api.TurnWatch)
    fake = FakeRunner()
    live_mind.set_runner(fake)  # type: ignore[arg-type, unused-ignore]  # a fake with its one method
    with TestClient(app) as client:
        yield client, fake
    live_mind.set_runner(None)


def test_a_judges_answered_turn_asks_for_a_live_pass(api: Any, watched: tuple[Any, FakeRunner], talker: Rig) -> None:
    from anjo.dashboard.auth import make_token, register_user

    client, fake = watched
    body = _start(api).json()
    assert client.post("/api/chat/s1/message", headers=_bearer(body["token"])).status_code == 200
    assert fake.asked == [body["user_id"]]
    assert client.post("/api/chat/broken/message", headers=_bearer(body["token"])).status_code == 409
    user, _ = register_user("TestPass1!-long-enough", "someone@example.com")
    assert user is not None
    assert client.post("/api/chat/s1/message", headers=_bearer(make_token(user["user_id"]))).status_code == 200
    assert client.post("/api/chat/s1/message").status_code == 200
    assert fake.asked == [body["user_id"]]  # nobody else's turn, and no failed one


def test_register_sleep_wires_the_judge_product(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI

    from anjo.core.anjo_model import assignment, client, hosts
    from anjo.sleep.app import register, spark_talker

    monkeypatch.setattr(assignment, "ROUTE_OVERRIDES", list(assignment.ROUTE_OVERRIDES))
    factories = dict(hosts._FACTORIES)
    monkeypatch.setattr(hosts, "_FACTORIES", factories)
    from types import MappingProxyType

    monkeypatch.setattr(hosts, "RUNTIMES", MappingProxyType(factories))
    monkeypatch.setattr(client, "OFF_VOCABULARY_HOSTS", set(client.OFF_VOCABULARY_HOSTS))
    app = FastAPI()
    try:
        register.register_sleep(app)
        assert assignment.ROUTE_OVERRIDES[0] is judge_talker.judge_route
        assert spark_talker.HOST in hosts.RUNTIMES and spark_talker.HOST in client.OFF_VOCABULARY_HOSTS
        assert judge_api.PUBLIC_PREFIX in app.state.public_prefixes
        assert any(m.cls is judge_api.TurnWatch for m in app.user_middleware)
        register.register_sleep(app)  # again: nothing doubles
        assert sum(m.cls is judge_api.TurnWatch for m in app.user_middleware) == 1
        assert assignment.ROUTE_OVERRIDES.count(judge_talker.judge_route) == 1
    finally:
        register.unregister_hooks()
    assert judge_talker.judge_route not in assignment.ROUTE_OVERRIDES
