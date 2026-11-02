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
        assert "mock.js" not in (APP_DIR / page).read_text() and "mock_thought" not in (APP_DIR / page).read_text()
    fixture_importers = [n for n in ASSETS if n.endswith(".js") and "mock_thought.js" in (APP_DIR / n).read_text()]
    assert fixture_importers == ["mock.js"]  # the synthetic life loads only inside the fake API


def test_the_pages_call_the_contract_routes() -> None:
    from anjo.sleep.app.judge_pages import APP_DIR

    text = "".join((APP_DIR / n).read_text() for n in ("api.js", "landing.js", "live.js", "chat.js", "sse.js",
                                                         "panel.js", "sleep.js", "wake.js", "thought.js", "morning.js"))
    for route in ("/api/sleep/judge/start", "/api/sleep/me/mind", "/api/sleep/me/talker", "/api/sleep/me/sleep-now",
                  "/api/sleep/me/nights/", "/api/sleep/me/wake", "/api/sleep/me/undo", "/api/chat/start",
                  "/api/sleep/me/thought", "/dispute", "/hide",
                  "/message", "platform=web", "Authorization", "Bearer", "sessionStorage", "Idempotency-Key"):
        assert route in text, route
    words = "".join((APP_DIR / n).read_text() for n in ("copy.js", "index.html", "live.html")).lower()
    assert "companion" not in words
    assert "glimmer" in words and "a personal intelligence that grows with you" in words
    for page in ("index.html", "live.html"):
        assert "Built with Muse Spark Open" in (APP_DIR / page).read_text(), page


# ── Her own line of thought ───────────────────────────────────────────────────


def _fixture() -> dict[str, Any]:
    """The ``?mock=1`` thought view: the engine's own JSON for a synthetic life, inside a module."""
    import json

    from anjo.sleep.app.judge_pages import APP_DIR

    text = (APP_DIR / "mock_thought.js").read_text()
    body = text[text.index("Object.freeze(") + len("Object.freeze("):text.rindex(");")]
    data: dict[str, Any] = json.loads(body)
    return data


def test_the_mock_thought_view_has_every_field_the_page_reads() -> None:
    from anjo.sleep.app.judge_pages import APP_DIR

    view = _fixture()
    code = (APP_DIR / "thought.js").read_text() + (APP_DIR / "morning.js").read_text()
    assert view["version"] == 1 and view["credence_note"]
    assert view["hunches"] and view["settled"] and view["question"] and view["morning"]
    for hunch in view["hunches"]:
        for key in ("id", "plain", "exact", "odds", "due", "disputed", "about", "claim"):
            assert key in hunch, key
    for key in ("scored", "right", "hit_rate", "brier", "enough", "bins", "ran_out_of_time"):
        assert key in view["scorecard"], key
        assert key in code, key
    assert view["scorecard"]["enough"] is False  # the fixture shows "too few bets yet"
    question = view["question"]
    assert {"question_id", "plain", "exact", "ask", "choices"} <= set(question)
    assert question["choices"][-1]["id"].endswith(".else")
    morning = view["morning"]
    for key in ("plain", "since", "empty", "settled", "new_hunches", "shifts", "findings"):
        assert key in morning and key in code, key
    shown = {h["id"] for h in view["hunches"]} | {s["id"] for s in view["settled"]}
    assert set(morning["settled"]) | set(morning["new_hunches"]) <= shown  # the note points at items on the page
    for key in ("credence_note", "due", "odds", "disputed", "question_id", "choices", "now_leading", "statement"):
        assert key in code, key


def test_dispute_and_hide_show_at_once_and_roll_back() -> None:
    from anjo.sleep.app.judge_pages import APP_DIR

    code = (APP_DIR / "thought.js").read_text()
    act = code[code.index("async function act("):code.index("function focusAfter(")]
    assert act.index("render()") < act.index("await api.hide") < act.index("catch (error)")  # shown first
    assert "view = before" in act and 'setAttribute("role", "alert")' in code  # put back, and said beside it
    assert "NOTE_MAX = 500" in code  # the engine's own limit on a note
    copy = (APP_DIR / "copy.js").read_text()
    assert 'hide: "Hide this"' in copy and "forget" not in copy.lower()  # a hide is not an erase
    assert 'credence_note' in code and "score_few" in code  # her odds are hers, and too few bets says so


@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node is not installed")
def test_the_thought_helpers_change_nothing_in_place() -> None:
    """The pure helpers, run in node: dates on the viewer's calendar, a new view per change, the
    answer phrased as the judge's own message."""
    import json
    import subprocess

    from anjo.sleep.app.judge_pages import APP_DIR

    script = f"""
globalThis.window = {{ location: {{ search: "" }}, matchMedia: () => ({{ matches: false }}) }};
const t = await import({json.dumps((APP_DIR / "thought.js").as_uri())});
const {{ THOUGHT }} = await import({json.dumps((APP_DIR / "mock_thought.js").as_uri())});
const today = new Date(2026, 10, 2);
const disputed = t.withDisputed(THOUGHT, "p7");
const hidden = t.withoutHunch(THOUGHT, "p7");
const q = THOUGHT.question;
console.log(JSON.stringify({{
  due: t.dueWords(THOUGHT.hunches[0], today), later: t.dueWords(THOUGHT.hunches[1], today),
  disputed: [disputed.hunches[0].disputed, THOUGHT.hunches[0].disputed],
  hidden: [hidden.hunches.length, THOUGHT.hunches.length, hidden.hidden],
  answer: t.answerText(q, q.choices[0]), other: t.isElse(q.choices[2]), first: t.isElse(q.choices[0]),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["due"] == "Due Nov 3 · tomorrow" and out["later"] == "Due Nov 5 · in 3 days"
    assert out["disputed"] == [True, False] and out["hidden"] == [1, 2, 1]
    assert out["answer"] == "On \u201cWill Mira enter the March festival?\u201d: She submits for March"
    assert out["other"] is True and out["first"] is False
