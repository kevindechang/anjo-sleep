# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Who reaches the live page (Task 13a), on the real web app through all of its middleware: an
anonymous visitor reaches the demo page and the demo API on the sleep host, and nothing else of
Sleep's; no path trick widens that; with the host flag unset nothing opens. And the address the
demo counts a visitor by is the one the proxy forwards, never one the visitor names."""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote

import pytest

from anjo.sleep.demo import words
from anjo.sleep.tests.app.test_demo_page import (  # noqa: F401  (account is a fixture)
    ACCOUNT,
    HEADER,
    VISIT,
    FakeServing,
    ScriptedModel,
    _arms,
    _behind_caddy,
    _write_starters,
    account,
)

PUBLIC = ("/sleep/demo", "/api/sleep/demo")


@pytest.fixture
def sleep_app(_app: Any, account: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:  # noqa: F811  (the imported fixture)
    """The session's web app with Sleep registered as ``app.py`` does on the sleep host (hooks
    aside), its demo on fakes; put back as it was afterwards."""
    from anjo.sleep.app import demo_api, register
    from anjo.sleep.app.demo_live import build_service

    routes, state = list(_app.router.routes), dict(_app.state._state)
    monkeypatch.setattr(register, "register_hooks", lambda: None)  # the routes and the public paths only
    register.register_sleep(_app)
    serving = FakeServing()
    service = build_service(serving=serving, arms=_arms(ScriptedModel(), serving))
    _app.dependency_overrides[demo_api.demo_service] = lambda: service
    try:
        yield _app
    finally:
        _app.dependency_overrides.pop(demo_api.demo_service, None)
        _app.router.routes[:] = routes
        _app.state._state.clear()
        _app.state._state.update(state)


def _raw(app: Any, raw_path: str, method: str = "GET") -> int:
    """One request with ``raw_path`` exactly as sent: no client normalizes it on the way. The
    server decodes it as uvicorn does (``%2f`` becomes ``/`` in the path)."""
    sent: dict[str, int] = {}
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "https", "path": unquote(raw_path), "raw_path": raw_path.encode(), "query_string": b"",
        "root_path": "", "headers": [(b"host", b"testserver")], "client": ("8.8.8.8", 4242),
        "server": ("testserver", 443),
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]

    asyncio.run(app(scope, receive, send))
    return sent["status"]


def test_an_anonymous_visitor_reaches_the_page_and_its_api(sleep_app: Any, client: Any) -> None:
    _write_starters()
    page = client.get("/sleep/demo", follow_redirects=False)
    assert page.status_code == 200 and 'id="chat"' in page.text
    for asset in ("/sleep/demo/app.js", "/sleep/demo/demo.css"):
        assert client.get(asset, follow_redirects=False).status_code == 200, asset
    assert client.get("/api/sleep/demo/page").json()["brief"]
    visit = client.post(VISIT, json={"adult": True})
    assert visit.status_code == 200, visit.text
    headers = {HEADER: visit.json()["token"]}
    pair = client.post("/api/sleep/demo/starters/credit", headers=headers)
    assert pair.status_code == 200 and pair.json()["left"]["text"]
    ask = client.post("/api/sleep/demo/ask", json={"text": "Where does the title card go?"}, headers=headers)
    assert ask.status_code == 200 and ask.json()["kind"] == "pair"
    vote = client.post("/api/sleep/demo/vote", json={"pair_id": ask.json()["pair_id"], "choice": "tie"}, headers=headers)
    assert vote.status_code == 200
    assert "set-cookie" not in page.headers and "set-cookie" not in visit.headers  # no account, no session


def test_every_other_sleep_route_still_refuses_anonymous_visitors(sleep_app: Any, client: Any) -> None:
    for method, path in (("GET", "/api/sleep/nights"), ("GET", "/api/sleep/nights/2026-10-30"),
                         ("GET", "/api/sleep/what-changed"), ("POST", "/api/sleep/undo"),
                         ("POST", "/api/sleep/dev/sessions"), ("GET", "/api/sleep/dev/sessions/s1"),
                         ("GET", "/api/sleep/demonstration"), ("GET", "/api/sleep/demo-admin")):
        response = client.request(method, path, follow_redirects=False)
        assert response.status_code == 401, (method, path, response.status_code)
    for path in ("/sleep", "/sleep/nights", "/sleep/demox", "/sleep/demo-admin"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code in (302, 404) and 'id="chat"' not in response.text, path


@pytest.mark.parametrize("raw", [
    "/api/sleep/demo/../nights", "/api/sleep/demo/%2e%2e/nights", "/api/sleep/demo/%2E%2E/nights",
    "/api/sleep/demo/..%2fnights", "/api/sleep/demo/..%2Fwhat-changed", "/api/sleep/demo//../nights",
    "//api/sleep/demo/../nights", "/api/sleep/demo/./../nights", "/api/sleep/demo/page/../../nights",
    "/api/sleep/demo;/../nights", "/api/sleep/demo/page;/../../nights", "/api/sleep/demo%2f..%2fnights",
    "/API/SLEEP/DEMO/../nights", "/api/sleep/DEMO/page", "/Api/sleep/demo/page", "/api/sleep/demo/.",
    "/api/sleep/demo/..", "/api/sleep/demo/%2e", "/sleep/demo/../../api/sleep/nights",
    "/sleep/demo/%2e%2e/%2e%2e/api/sleep/nights", "/api/sleep/demo/../dev/sessions/s1",
])
def test_path_tricks_open_nothing_else(sleep_app: Any, raw: str) -> None:
    assert _raw(sleep_app, raw) in (302, 401, 404), raw
    assert _raw(sleep_app, raw, "POST") in (302, 401, 404, 405), raw


def test_the_public_paths_are_exactly_the_demo(sleep_app: Any) -> None:
    assert tuple(sleep_app.state.public_prefixes) == PUBLIC
    assert _raw(sleep_app, "/api/sleep/demo/page") == 200 and _raw(sleep_app, "/sleep/demo") == 200
    assert _raw(sleep_app, "/api/sleep/demo/page;") == 404  # past the gate, but no route of anyone else's
    assert _raw(sleep_app, "/api/sleep/demo") == 404 and _raw(sleep_app, "/api/sleep/demo/") in (401, 404)


def test_with_the_host_unset_the_app_opens_nothing(_app: Any, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import register
    from anjo.sleep.app.flags import SleepHostOff

    # The app under test was imported without ANJO_SLEEP_HOST: no public path, no demo route.
    assert getattr(_app.state, "public_prefixes", ()) == ()
    assert client.get("/api/sleep/demo/page").status_code == 401
    assert client.post(VISIT, json={"adult": True}).status_code == 401
    assert client.get("/sleep/demo", follow_redirects=False).status_code == 404
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    routes = len(_app.router.routes)
    with pytest.raises(SleepHostOff):
        register.register_sleep(_app)
    assert getattr(_app.state, "public_prefixes", ()) == () and len(_app.router.routes) == routes


# ── the visitor's address ─────────────────────────────────────────────────────


def test_anjo_dashboard_keeps_uvicorns_proxy_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dev host runs ``anjo-dashboard`` behind Caddy on loopback. Caddy sets X-Forwarded-For
    (never X-Real-IP), and uvicorn's proxy headers, on by default and trusting loopback, make it
    the request's client. The demo counts that address and nothing else."""
    import uvicorn

    from anjo.dashboard import app as web

    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    ran: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **options: ran.update(options, target=target))
    web.run()
    target = ran.pop("target")
    config = uvicorn.Config(target, **{k: v for k, v in ran.items() if k != "reload"})
    assert config.proxy_headers is True
    assert "127.0.0.1" in str(config.forwarded_allow_ips) and ran["host"] == "127.0.0.1"


def test_the_demo_counts_the_forwarded_address_never_a_named_one(account: str) -> None:  # noqa: F811  (the imported fixture)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.sleep.app.demo_api import demo_service, router
    from anjo.sleep.app.demo_live import build_service
    from anjo.sleep.demo.limits import Limits

    service = build_service(serving=FakeServing(), arms=_arms(ScriptedModel(), FakeServing()))
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[demo_service] = lambda: service
    client = TestClient(_behind_caddy(app))
    per_client = Limits().visits_per_client_per_day
    for n in range(per_client):  # a visitor naming a new X-Real-IP each time is still one address
        spoof = {"X-Forwarded-For": "8.8.8.8", "X-Real-IP": f"9.9.9.{n + 1}"}
        assert client.post(VISIT, json={"adult": True}, headers=spoof).status_code == 200
    spent = client.post(VISIT, json={"adult": True}, headers={"X-Forwarded-For": "8.8.8.8", "X-Real-IP": "9.9.9.99"})
    assert spent.status_code == 429 and spent.json()["detail"] == words.COPY["visits_spent"]
    # A forged X-Forwarded-For reaches the app only as a prefix of Caddy's: the address Caddy saw wins.
    forged = client.post(VISIT, json={"adult": True}, headers={"X-Forwarded-For": "1.1.1.1, 8.8.8.8"})
    assert forged.status_code == 429
    assert client.post(VISIT, json={"adult": True}, headers={"X-Forwarded-For": "8.8.4.4"}).status_code == 200

    # Were the proxy headers ever off, every visitor would arrive as Caddy's loopback address, and
    # a named X-Real-IP (which Caddy passes through untouched) still picks no bucket: they all share one.
    async def no_rewrite(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await app({**scope, "client": ("127.0.0.1", 50124)} if scope["type"] == "http" else scope, receive, send)

    bare = TestClient(no_rewrite)
    named = [bare.post(VISIT, json={"adult": True}, headers={"X-Real-IP": f"9.9.8.{n + 1}"}).status_code
             for n in range(per_client + 1)]
    assert named == [200] * per_client + [429]
    assert ACCOUNT  # the film student's account served the live side throughout
