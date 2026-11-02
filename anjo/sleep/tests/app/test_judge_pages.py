# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The judge product's pages on the sleep host: served only there, public, every file named, no
inline script and nothing from another host, the main site's own stylesheets linked, and the fake
API reachable only behind ``?mock=1``."""
from __future__ import annotations

import re
from typing import Any

import pytest

PAGES = ("/sleep/app", "/sleep/app/live")


@pytest.fixture
def mounted(monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.sleep.app import register

    monkeypatch.setenv("ANJO_SLEEP_HOST", "1")
    app = FastAPI()
    monkeypatch.setattr(register, "register_hooks", lambda: None)  # the routes only
    register.register_sleep(app)
    with TestClient(app) as client:
        yield client


def _scripts(html: str) -> list[str]:
    return re.findall(r"<script\b[^>]*>", html)


def test_pages_and_files_are_served_only_on_the_sleep_host(mounted: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app.judge_pages import ASSETS

    for path in PAGES:
        page = mounted.get(path)
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html"), path
        assert page.headers["cache-control"] == "no-store"
    for name, media in ASSETS.items():
        found = mounted.get(f"/sleep/app/{name}")
        assert found.status_code == 200 and found.headers["content-type"] == media, name
        assert found.content.strip(), name
    for path in ("/sleep/app/judge_pages.py", "/sleep/app/index.html", "/sleep/app/..%2Fdemo.css", "/sleep/app/x.js"):
        assert mounted.get(path).status_code == 404, path
    monkeypatch.delenv("ANJO_SLEEP_HOST")
    for path in (*PAGES, "/sleep/app/live.js"):
        assert mounted.get(path).status_code == 404, path


def test_the_pages_are_public_and_nothing_else_new_is(mounted: Any) -> None:
    public = mounted.app.state.public_prefixes
    assert "/sleep/app" in public and "/sleep/demo" in public
    assert not any(p.startswith("/api/sleep/me") for p in public)


def test_no_inline_script_no_handlers_and_nothing_from_elsewhere(mounted: Any) -> None:
    from anjo.sleep.app.judge_pages import APP_DIR, ASSETS

    for path in PAGES:
        html = mounted.get(path).text
        tags = _scripts(html)
        assert tags, path
        for tag in tags:  # every script is an external, same-origin file: the CSP nonce covers it
            src = re.search(r'\bsrc="([^"]+)"', tag)
            assert src and src.group(1).startswith("/"), tag
        assert len(re.findall(r"<script\b[^>]*>\s*</script>", html)) == len(tags), path  # no inline body
        assert not re.search(r"\son[a-z]+\s*=", html), path  # no inline handlers
        assert "javascript:" not in html
        assert "http://" not in html
        assert not re.findall(r"""(?:src|href)="https?://(?!anjo\.love)""", html), path
        assert '<link rel="stylesheet" href="/static/site.css' in html, path
        assert '<link rel="stylesheet" href="/static/home.css' in html, path
        assert 'name="color-scheme" content="dark"' in html
        for ref in re.findall(r'(?:src|href)="/sleep/app/([^"?#]+)"', html):
            assert ref in ASSETS, ref
    for name in ASSETS:
        text = (APP_DIR / name).read_text()
        assert "http://" not in text and "https://" not in text and "@import" not in text, name
        if name.endswith(".js"):
            for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
                assert banned not in text, (name, banned)
            for banned in ("localStorage", "window.confirm", "alert(", "prompt("):
                assert banned not in text, (name, banned)
        if name.endswith(".css"):
            assert "url(" not in text, name  # the type comes from the site's own stylesheet
            assert "prefers-reduced-motion" in text, name
        assert len(text.splitlines()) < 800, name


def test_mock_mode_needs_the_query_flag() -> None:
    """The fake API loads only from api.js, only when the page's own URL says ``mock=1``."""
    from anjo.sleep.app.judge_pages import APP_DIR, ASSETS

    importers = [n for n in ASSETS if n.endswith(".js") and "mock.js" in (APP_DIR / n).read_text() and n != "mock.js"]
    assert importers == ["api.js"]
    api = (APP_DIR / "api.js").read_text()
    assert 'get("mock") === "1"' in api and 'import("./mock.js")' in api
    for page in ("index.html", "live.html"):
        assert "mock.js" not in (APP_DIR / page).read_text()


def test_the_pages_call_the_contract_routes() -> None:
    from anjo.sleep.app.judge_pages import APP_DIR

    text = "".join((APP_DIR / n).read_text() for n in ("api.js", "landing.js", "live.js", "chat.js", "sse.js",
                                                         "panel.js", "sleep.js", "wake.js"))
    for route in ("/api/sleep/judge/start", "/api/sleep/me/mind", "/api/sleep/me/talker", "/api/sleep/me/sleep-now",
                  "/api/sleep/me/nights/", "/api/sleep/me/wake", "/api/sleep/me/undo", "/api/chat/start",
                  "/message", "platform=web", "Authorization", "Bearer", "sessionStorage", "Idempotency-Key"):
        assert route in text, route
    words = "".join((APP_DIR / n).read_text() for n in ("copy.js", "index.html", "live.html")).lower()
    assert "companion" not in words
    assert "glimmer" in words and "a personal intelligence that grows with you" in words
    for page in ("index.html", "live.html"):
        assert "Built with Muse Spark Open" in (APP_DIR / page).read_text(), page
