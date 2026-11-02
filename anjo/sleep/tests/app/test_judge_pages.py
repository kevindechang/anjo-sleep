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


# ── MIND's live pass, after each reply ────────────────────────────────────────


def test_a_reply_starts_a_watch_for_the_live_pass_not_one_early_refresh() -> None:
    """The server runs MIND's live pass after the reply's stream ends, so a refresh at that moment
    reads old state: the page polls until ``revision`` moves and ``live_pass`` stops running."""
    from anjo.sleep.app.judge_pages import APP_DIR, ASSETS

    live = (APP_DIR / "live.js").read_text()
    reply = live[live.index("onReply: () => {"):live.index("talker.check();", live.index("onReply: () => {"))]
    assert "livePass.watch()" in reply and "panel.refresh()" not in reply
    assert "onSend: () => livePass.mark()" in live  # the revision from before the message
    watch = (APP_DIR / "livepass.js").read_text()
    assert "export const POLL_MS = 2000;" in watch and "export const CAP_MS = 60000;" in watch
    assert "export const SLOW_MS = 10000;" in watch and "export const SLOW_CAP_MS = 180000;" in watch
    assert '"visibilitychange"' in watch and "document.hidden" in watch  # paused while the tab is hidden
    assert "thought.isOpen() ? thought.refresh()" in watch  # her line of thought too, when it's open
    assert "livepass.js" in ASSETS
    html = (APP_DIR / "live.html").read_text()
    assert 'id="live-pass" role="status"' in html
    assert "Thinking about what you said" in (APP_DIR / "copy.js").read_text()
    mock = (APP_DIR / "mock.js").read_text()
    assert "revision:" in mock and 'live_pass: passing(' in mock  # the fake API answers with both fields
    for name in ("panel.js", "thought.js"):  # a running pass alone redraws nothing (marks and forms stay)
        assert "contentOf(" in (APP_DIR / name).read_text(), name


@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node is not installed")
def test_the_watch_stops_only_once_the_pass_has_landed() -> None:
    import json
    import subprocess

    from anjo.sleep.app.judge_pages import APP_DIR

    cases = [  # (answer, what the watch does next)
        ({"base": 3, "revision": 3, "livePass": "running", "sawRunning": True, "elapsed": 4000}, "poll"),
        ({"base": 3, "revision": 4, "livePass": "running", "sawRunning": True, "elapsed": 4000}, "poll"),
        ({"base": 3, "revision": 3, "livePass": "running", "sawRunning": True, "elapsed": 60000}, "slow"),
        ({"base": 3, "revision": 3, "livePass": "running", "sawRunning": True, "elapsed": 179000}, "slow"),
        ({"base": 3, "revision": 3, "livePass": "running", "sawRunning": True, "elapsed": 180000}, "stop"),
        ({"base": 3, "revision": 4, "livePass": "idle", "sawRunning": True, "elapsed": 6000}, "stop"),
        ({"base": 3, "revision": 3, "livePass": "idle", "sawRunning": True, "elapsed": 8000}, "final"),
        ({"base": 3, "revision": 3, "livePass": "idle", "sawRunning": False, "elapsed": 2000}, "poll"),
        ({"base": 3, "revision": 3, "livePass": "idle", "sawRunning": False, "elapsed": 10000}, "stop"),
        ({"base": None, "revision": 0, "livePass": "idle", "sawRunning": False, "elapsed": 2000}, "stop"),
        ({"elapsed": 2000, "fresh": 0}, "poll"),
        ({"elapsed": 4000, "fresh": 2}, "stop"),
        ({"elapsed": 60000, "fresh": 0}, "stop"),
        ({"base": "a1", "revision": "b2", "livePass": "idle", "sawRunning": False, "elapsed": 2000}, "stop"),
    ]
    script = f"""
const L = await import({json.dumps((APP_DIR / "livepass.js").as_uri())});
console.log(JSON.stringify({json.dumps([c for c, _ in cases])}.map((c) => L.nextStep(c))));
"""
    done = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [want for _, want in cases]


_WATCH_SIM = """
let now = 0;
const timers = [];
globalThis.window = {
  setTimeout: (f, ms) => { const t = { at: now + ms, f }; timers.push(t); return t; },
  clearTimeout: (t) => { const i = timers.indexOf(t); if (i >= 0) timers.splice(i, 1); },
};
Date.now = () => now;
const listeners = {};
const words = { textContent: "" };
const line = { hidden: true, querySelector: () => words };
const tab = { dataset: {} };
globalThis.document = {
  hidden: false, addEventListener: (name, f) => { listeners[name] = f; },
  getElementById: (id) => (id === "live-pass" ? line : tab),
};
const L = await import(LIVEPASS);
const { COPY } = await import(COPYJS);

async function until(t) {
  for (;;) {
    timers.sort((a, b) => a.at - b.at);
    const next = timers[0];
    if (!next || next.at > t) { now = t; return; }
    timers.shift();
    now = next.at;
    await next.f();
  }
}

// One watch after a reply. ``answer(t)`` is what /me/mind says at time t.
async function watch(answer, script = async () => until(400000)) {
  timers.length = 0;
  now = 0;
  const reads = [];
  let slowShown = false;
  const panel = {
    revision: () => 5,
    refresh: async () => { reads.push(now); if (words.textContent === COPY.live_slow) slowShown = true; return answer(now); },
  };
  const lp = L.createLivePass({ panel, thought: { isOpen: () => false } });
  lp.mark();
  lp.watch();
  await script();
  return { reads, hidden: line.hidden, pending: timers.length, slowShown };
}

const running = (t, end) => (t < end ? { revision: 5, livePass: "running" } : { revision: 6, livePass: "idle" });
const out = {};
out.slow = await watch((t) => running(t, 75000));             // a 75 s pass: seen in the background
out.endless = await watch(() => ({ revision: 5, livePass: "running" }));  // never lands: gives up at 3 min
out.away = await watch((t) => running(t, 75000), async () => {  // hidden from 30 s to 150 s
  await until(30000);
  document.hidden = true;
  await until(150000);
  document.hidden = false;
  await listeners.visibilitychange();
  await until(400000);
});
let late = 0;  // idle before its revision shows: one final read catches it
out.final = await watch((t) => (t < 10000 ? { revision: 5, livePass: "running" }
  : late++ === 0 ? { revision: 5, livePass: "idle" } : { revision: 6, livePass: "idle" }));
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node is not installed")
def test_a_slow_pass_is_still_seen_and_a_hidden_tab_pauses_the_checks() -> None:
    """The watch on a fake clock: a pass that runs 75 s (Spark may take 120 s) is caught by the
    10 s background check after the 2 s minute; one that never lands ends at 3 minutes; a hidden
    tab makes no requests and checks again as it shows; idle before the revision gets a final read."""
    import json
    import subprocess

    from anjo.sleep.app.judge_pages import APP_DIR

    script = _WATCH_SIM.replace("LIVEPASS", json.dumps((APP_DIR / "livepass.js").as_uri())).replace(
        "COPYJS", json.dumps((APP_DIR / "copy.js").as_uri()))
    done = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    slow = out["slow"]
    assert slow["reads"][:3] == [2000, 4000, 6000] and 60000 in slow["reads"]
    assert slow["reads"][-3:] == [60000, 70000, 80000]  # every 10 s once the minute is up, until it lands
    assert slow["slowShown"] and slow["hidden"] and slow["pending"] == 0
    endless = out["endless"]
    assert endless["reads"][-1] == 180000 and endless["hidden"] and endless["pending"] == 0
    away = out["away"]
    assert not [t for t in away["reads"] if 30000 < t < 150000]  # nothing while hidden
    assert 150000 in away["reads"] and away["reads"][-1] == 150000 and away["hidden"]
    final = out["final"]
    assert final["reads"][-2:] == [10000, 10000] and final["hidden"] and final["pending"] == 0


def test_the_judge_pages_call_anjo_her() -> None:
    """AGENTS.md: Anjo has her own state. The pages never call her "it"."""
    from anjo.sleep.app.judge_pages import APP_DIR

    index = (APP_DIR / "index.html").read_text()
    assert "She learns while she sleeps." in index and "It learns" not in index
    copy = "\n".join(line for line in (APP_DIR / "copy.js").read_text().splitlines() if not line.lstrip().startswith("//"))
    for text in (index, (APP_DIR / "live.html").read_text(), copy):
        assert not re.search(r"\b(its|itself)\b", text, re.IGNORECASE)
    for phrase in ("What she's working out about you", "Her bet on herself"):
        assert phrase in (APP_DIR / "live.html").read_text(), phrase
