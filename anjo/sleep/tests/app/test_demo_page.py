# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The live page on the sleep host (Task 13a, minimal step): the routes, the 18+ visit, Anjo's own
crisis checks, the blind pair and the vote, the limits, and that a visit writes nothing of the
visitor's: no history, no memory, no evidence."""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.demo.pairs import OPEN_PER_VISIT
from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs

ACCOUNT = "demo-film-student"
SERVE_RATE = 2.81  # the serving GPU's list price an hour (ANJO_SLEEP_PRICES' fourth value)
VISIT = "/api/sleep/demo/visits"
ASK = "/api/sleep/demo/ask"
VOTE = "/api/sleep/demo/vote"
HEADER = "X-Demo-Visit"


@dataclass
class ScriptedModel:
    """A ``ModelClient``: each call's reply comes from ``reply(n)``, n counting calls from 0."""

    reply: Any = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def complete(self, adapter, messages, *, temperature, seed=None):
        with self.lock:
            n = len(self.calls)
            self.calls.append({"adapter": adapter, "messages": messages})
        text = self.reply(n) if self.reply else f"a plain reply {n}"
        return ChatResult(text=text, tool_calls=(), model="m", input_tokens=700 + n, output_tokens=40,
                          finish_reason="stop")


@dataclass
class FakeServing:
    """The serving backend: counts every status request and every wake."""

    name: str = "sagemaker"
    up: bool = True
    pings: int = 0
    wakes: int = 0

    def ping(self) -> bool:
        self.pings += 1
        return self.up

    def wake(self) -> WakeResult:
        self.wakes += 1
        return WakeResult(1.0, True)

    def sleep_now(self) -> None:
        raise AssertionError("the page never puts Glimmer to sleep")

    def client(self) -> Any:
        raise AssertionError("the page reaches Glimmer only through the arms")

    def upload_adapter(self, folder: Path, name: str) -> None:
        raise AssertionError("the page never uploads an adapter")


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> str:
    from anjo.sleep.app.cleared_sessions import register_synthetic

    register_synthetic(ACCOUNT, at=datetime(2026, 10, 27, tzinfo=timezone.utc))
    monkeypatch.setenv("ANJO_SLEEP_DEMO_USER_ID", ACCOUNT)
    monkeypatch.setenv("ANJO_SLEEP_PRICES", f"0.1,0.4,2.0,{SERVE_RATE}")
    monkeypatch.delenv("ANJO_SLEEP_DEMO_BASELINE", raising=False)
    monkeypatch.delenv("ANJO_SLEEP_DEMO_WAKE", raising=False)
    monkeypatch.delenv("ANJO_SLEEP_DEMO_LIVE_HOURS", raising=False)
    return ACCOUNT


def _arms(model: ScriptedModel, serving: FakeServing, *, contexts: Any = None) -> Any:
    """The live arms over ``arms.reply`` with a fake model: the real registry and evidence store."""
    from anjo.sleep.app.demo_live import GlimmerArms, glimmer_status
    from anjo.sleep.app.storage import checkpoint_registry, evidence_store
    from anjo.sleep.probes.arms import ArmDeps

    def deps(until):
        return ArmDeps(model=model, adapters=checkpoint_registry(), contexts=contexts or FakeContexts(),
                       evidence=evidence_store, logprobs=FakeLogprobs(lambda adapter, messages: ""),
                       donors={}, until=until)

    return GlimmerArms(ACCOUNT, deps=deps, status=lambda: glimmer_status(serving))


def _client(service: Any) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.sleep.app.demo_api import demo_service, page_router, router

    app = FastAPI()
    app.include_router(page_router)
    app.include_router(router, prefix="/api")
    app.dependency_overrides[demo_service] = lambda: service
    return TestClient(app)


def _behind_caddy(app: Any) -> Any:
    """``app`` as the dev host serves it: Caddy on loopback in front, uvicorn's proxy headers."""
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    proxied = ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1")

    async def caddy(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope = {**scope, "client": ("127.0.0.1", 50123)}
        await proxied(scope, receive, send)

    return caddy


def _page(model: ScriptedModel | None = None, serving: FakeServing | None = None, **arms: Any) -> Any:
    from anjo.sleep.app.demo_live import build_service

    serving = serving or FakeServing()
    return _client(build_service(serving=serving, arms=_arms(model or ScriptedModel(), serving, **arms)))


def _visit(client: Any) -> dict[str, str]:
    response = client.post(VISIT, json={"adult": True})
    assert response.status_code == 200, response.text
    return {HEADER: response.json()["token"]}


def _demo_dir() -> Path:
    from anjo.sleep.app.demo_live import demo_dir

    return demo_dir()


def _write_starters(texts: Any = None) -> None:
    from anjo.sleep.demo.starters import Side, generate, to_json
    from anjo.sleep.types import CheckpointRef

    product, baseline = texts or (lambda s: f"P:{s.id}", lambda s: f"B:{s.id}")
    ckpt = CheckpointRef("p-film-n3", ACCOUNT, datetime(2026, 10, 30).date(), "uri://x", None)
    starters = generate(words.SCENES, lambda s: (Side(product(s), 1200, 0.004), Side(baseline(s), 1180, 0.0039)),
                        person="film_student", checkpoint=ckpt, baseline="2C",
                        now=datetime(2026, 10, 31, tzinfo=timezone.utc))
    _demo_dir().mkdir(parents=True, exist_ok=True)
    (_demo_dir() / "starter_pairs.json").write_text(to_json(starters))


def _case(expect: str) -> str:
    from anjo.evals.crisis_cases import ALL_CASES

    return next(case.text for case in ALL_CASES if case.expect == expect)


# ── the page ──────────────────────────────────────────────────────────────────


def test_page_served_only_with_sleep_host(unmounted_app: Any, auth_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.sleep.app import register

    # The app under test was imported without ANJO_SLEEP_HOST: no page and no demo route exist there.
    assert auth_client.get("/sleep/demo").status_code == 404
    assert auth_client.get("/api/sleep/demo/page").status_code == 404
    app = FastAPI()
    monkeypatch.setattr(register, "register_hooks", lambda: None)  # the routes only
    register.register_sleep(app)
    with TestClient(app) as mounted:
        page = mounted.get("/sleep/demo")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "app.js" in page.text and "demo.css" in page.text
        assert mounted.get("/sleep/demo/app.js").headers["content-type"].startswith("text/javascript")
        assert mounted.get("/sleep/demo/demo.css").status_code == 200
        assert mounted.get("/sleep/demo/timeline.js").headers["content-type"].startswith("text/javascript")
        assert mounted.get("/sleep/demo/fonts/geist-400.ttf").status_code == 404  # the type is the site's own
        assert mounted.get("/sleep/demo/register.py").status_code == 404
        monkeypatch.delenv("ANJO_SLEEP_HOST")
        for path in ("/sleep/demo", "/sleep/demo/app.js", "/api/sleep/demo/page"):
            assert mounted.get(path).status_code == 404, path
        assert mounted.post(VISIT, json={"adult": True}).status_code == 404


def test_the_page_loads_nothing_from_elsewhere_and_sets_text_only_as_text() -> None:
    from anjo.sleep.demo.page import ASSETS, WEB_DIR, page_html

    page = page_html()
    scripts = [(WEB_DIR / name).read_text() for name in ASSETS if name.endswith(".js")]
    styles = [(WEB_DIR / name).read_text() for name in ASSETS if name.endswith(".css")]
    assert len(scripts) == 7 and len(styles) == 3  # app, model, nights, mind, night, timeline and dailies; demo, mind and dailies css
    for text in (page, *scripts, *styles):
        assert "http://" not in text and "https://" not in text and "//cdn" not in text
        assert "@import" not in text
    for style in styles:  # type, colour and spacing come from the main site's stylesheet, linked first
        assert "url(" not in style and "@font-face" not in style
    assert page.index('href="/static/site.css"') < page.index("demo.css")
    assert "--paper: var(--canvas)" in styles[0] and "--ink: var(--text)" in styles[0]  # roles mapped, not copied
    for script in scripts:
        assert "innerHTML" not in script and "insertAdjacentHTML" not in script and "outerHTML" not in script
        assert "eval(" not in script and "localStorage" not in script and "document.write" not in script
    assert "prefers-reduced-motion" in styles[0]
    assert '<meta name="color-scheme" content="dark"' in page  # dark only, as the site is


def test_the_task9_controls_reference_their_routes_and_ask_inline() -> None:
    """The person's own controls (Task 9) call the three routes, never a browser dialog, and put
    every result in the page's own live line."""
    from anjo.sleep.demo.page import ASSETS, WEB_DIR, page_html

    page = page_html()
    scripts = {name: (WEB_DIR / name).read_text() for name in ASSETS if name.endswith(".js")}
    assert "model.js" in scripts
    model = scripts["model.js"]
    for endpoint in ("/api/sleep/nights", "/api/sleep/what-changed", "/api/sleep/undo", "/api/sleep/delete"):
        assert endpoint in model, endpoint
    for banned in ("window.confirm", "alert(", "prompt("):
        assert all(banned not in script for script in scripts.values()), banned
    assert 'id="yours-result"' in page and 'aria-live="polite"' in page
    assert 'id="yours-undo-step"' in page and 'id="yours-delete-step"' in page
    assert "textContent" in model and "innerHTML" not in model


@pytest.mark.parametrize("status", [200, 401, 403, 500])
def test_model_controls_visibility_and_retry_after_initial_failure(status: int) -> None:
    """Execute the real script: server failures expose the error and controls; auth failures hide them."""
    from anjo.sleep.demo.page import WEB_DIR

    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const status = Number(process.argv[1]);
const nodes = new Map();
function byId(id) {
  if (!nodes.has(id)) nodes.set(id, {
    hidden: id === 'yours', textContent: '', value: '', listeners: {},
    addEventListener(event, listener) { this.listeners[event] = listener; },
    replaceChildren() {}, append() {},
  });
  return nodes.get(id);
}
let nights = 0;
const copy = { yours_loading: 'Loading', yours_nights_empty: 'Empty',
  yours_changes_empty: 'No changes', yours_unavailable: 'Sign in' };
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {
  window: { anjoDemo: { copy, byId, element() {} } },
  fetch: async (path) => {
    const code = path.endsWith('/nights') && ++nights === 1 ? status : 200;
    return { ok: code === 200, status: code,
      json: async () => code === 200 ? [] : { detail: 'Inline failure' } };
  },
});
(async () => {
  await new Promise(setImmediate);
  const unavailable = status === 401 || status === 403;
  assert.equal(byId('yours').hidden, unavailable);
  assert.equal(byId('yours-retry').hidden, status !== 500);
  if (status === 500) {
    assert.equal(byId('yours-nights-state').textContent, 'Inline failure');
    assert.equal(byId('yours-delete').hidden, false);
    assert.equal(typeof byId('yours-delete').listeners.click, 'function');
    await byId('yours-retry').listeners.click();
    assert.equal(nights, 2);
    assert.equal(byId('yours-nights-state').textContent, 'Empty');
    assert.equal(byId('yours-retry').hidden, true);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(["node", "-e", script, str(status), str(WEB_DIR / "model.js")],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_brief_served_with_chat(account: str) -> None:
    from anjo.sleep.app.demo_live import DEMO_PERSONA
    from anjo.sleep.probes.judge_brief import judge_brief

    client = _page()
    body = client.get("/api/sleep/demo/page").json()
    assert body["brief"] == judge_brief(DEMO_PERSONA)
    assert [scene["id"] for scene in body["scenes"]] == [scene.id for scene in words.SCENES]
    assert body["pregenerated"] is None and not any(scene["ready"] for scene in body["scenes"])
    page = client.get("/sleep/demo").text
    assert 'id="brief"' in page and 'id="chat"' in page  # one page: the brief stays beside the chat
    assert page.index('id="brief"') < page.index('id="chat"')


def test_page_requires_adult_confirmation(account: str) -> None:
    client = _page()
    for body, status in (({}, 403), ({"adult": False}, 403), ({"adult": "yes"}, 422), ({"adult": 1}, 422)):
        response = client.post(VISIT, json=body)
        assert response.status_code == status, body
        assert "token" not in response.text
    assert client.post(VISIT, json={}).json()["detail"] == words.COPY["adult_required"]
    no_visit = {HEADER: "v1.forged-forged-forged.1.abc"}
    for headers in ({}, no_visit):
        assert client.post(ASK, json={"text": "hello"}, headers=headers).status_code == 401
        assert client.post("/api/sleep/demo/starters/nobody-to-tell", headers=headers).status_code == 401
        assert client.post(VOTE, json={"pair_id": "p", "choice": "tie"}, headers=headers).status_code == 401
    counters = json.loads((_demo_dir() / "counters.json").read_text()) if (_demo_dir() / "counters.json").exists() else {}
    assert counters.get("visits", 0) == 0  # a refused confirmation counts no visit
    headers = _visit(client)
    assert client.post(ASK, json={"text": "How do I light the pier at dusk?"}, headers=headers).json()["kind"] == "pair"


# ── Anjo's crisis checks ──────────────────────────────────────────────────────


def test_demo_crisis_text_answered_before_model(account: str) -> None:
    from anjo.core.crisis_render import ideation_line, severe_reply
    from anjo.core.db import get_db

    model, serving = ScriptedModel(), FakeServing()
    client = _page(model, serving)
    headers = _visit(client)
    severe = client.post(ASK, json={"text": _case("hard")}, headers=headers).json()
    assert severe == {"kind": "crisis", "text": severe_reply("en", None)}
    ideation = client.post(ASK, json={"text": _case("soft")}, headers=headers).json()
    assert ideation == {"kind": "crisis", "text": ideation_line("en", None)}
    assert model.calls == [] and serving.pings == 0  # no model saw them, and Glimmer wasn't asked
    counted = get_db().execute(
        "SELECT tier, SUM(count) FROM crisis_referral_ledger WHERE surface = 'sleep_demo' GROUP BY tier"
    ).fetchall()
    assert {row[0]: row[1] for row in counted} == {"severe": 1, "ideation": 1}  # counts, no words


def test_replies_pass_reply_crisis_check(account: str) -> None:
    from anjo.core.crisis_render import severe_reply
    from anjo.evals.crisis_reply_cases import REPLY_TRIPS

    tripping = next(text for language, text, _reason in REPLY_TRIPS if language == "en")
    client = _page(ScriptedModel(reply=lambda n: tripping if n == 0 else "Hold the shot on the rope."))
    headers = _visit(client)
    pair = client.post(ASK, json={"text": "What do I do with the deck scene?"}, headers=headers).json()
    sides = [pair["left"], pair["right"]]
    assert sorted(side["withheld"] for side in sides) == [False, True]
    withheld = next(side for side in sides if side["withheld"])
    assert withheld["text"] == severe_reply("en", None) and tripping not in json.dumps(pair)
    assert pair["votable"] is False
    vote = client.post(VOTE, json={"pair_id": pair["pair_id"], "choice": "left"}, headers=headers)
    assert vote.status_code == 409 and vote.json()["detail"] == words.COPY["not_votable"]

    _write_starters((lambda s: tripping, lambda s: f"B:{s.id}"))
    starter = client.post("/api/sleep/demo/starters/nobody-to-tell", headers=headers).json()
    assert tripping not in json.dumps(starter) and starter["votable"] is False


# ── the blind pair and the vote ───────────────────────────────────────────────


def test_side_order_randomized_and_revealed_after_pick(account: str) -> None:
    from fastapi.testclient import TestClient

    from anjo.sleep.app.demo_live import build_service

    _write_starters()
    serving = FakeServing()
    client = TestClient(_behind_caddy(_client(build_service(serving=serving, arms=_arms(ScriptedModel(), serving))).app),
                        headers={"X-Forwarded-For": "8.8.8.8"})
    lefts, pairs = set(), []
    for n in range(24):  # under the 30-a-minute request limit; one order every time has odds 2**-23
        if n % OPEN_PER_VISIT == 0:  # a visit holds this many unvoted pairs at most
            headers = _visit(client)
        pair = client.post("/api/sleep/demo/starters/nobody-to-tell", headers=headers).json()
        assert set(pair) == {"pair_id", "left", "right", "votable", "source"}
        assert pair["source"] == {"kind": "pre-generated", "checkpoint": "p-film-n3", "night": "2026-10-30",
                                  "generated_at": "2026-10-31T00:00:00+00:00"}
        shown = json.dumps({key: value for key, value in pair.items() if key != "pair_id"})  # the id is random
        assert '"4"' not in shown and "2C" not in shown and "1200" not in shown
        lefts.add(pair["left"]["text"])
        pairs.append((pair, headers))
    assert lefts == {"P:nobody-to-tell", "B:nobody-to-tell"}  # the server shuffles every pair
    # An address votes once a scene a day: one pair of each order, the second from another address.
    product_first = next((p, h) for p, h in pairs if p["left"]["text"] == "P:nobody-to-tell")
    baseline_first = next((p, h) for p, h in pairs if p["left"]["text"] == "B:nobody-to-tell")
    for (pair, headers), address in ((product_first, "8.8.8.8"), (baseline_first, "1.1.1.1")):
        reveal = client.post(VOTE, json={"pair_id": pair["pair_id"], "choice": "left"},
                             headers={**headers, "X-Forwarded-For": address}).json()
        product_left = pair["left"]["text"] == "P:nobody-to-tell"
        assert reveal["left"]["arm"] == ("4" if product_left else "2C")
        assert reveal["right"]["arm"] == ("2C" if product_left else "4")
        product = reveal["left"] if product_left else reveal["right"]
        assert (product["prompt_tokens"], product["cost_usd"]) == (1200, 0.004)
        assert product["label"] == words.COPY["arm_4"] and reveal["choice"] == "left"


def test_vote_stores_no_message_text(account: str) -> None:
    question = "MARKER-QUESTION-7f3 about the gulls"
    client = _page(ScriptedModel(reply=lambda n: f"MARKER-REPLY-{n} keep the gulls"))
    headers = _visit(client)
    first = client.post(ASK, json={"text": question}, headers=headers).json()
    second = client.post(ASK, json={"text": question}, headers=headers).json()
    assert client.post(VOTE, json={"pair_id": first["pair_id"], "choice": "neither"}, headers=headers).status_code == 200
    assert client.post(VOTE, json={"pair_id": second["pair_id"], "choice": "tie"}, headers=headers).status_code == 200
    again = client.post(VOTE, json={"pair_id": second["pair_id"], "choice": "left"}, headers=headers)
    assert again.status_code == 409
    stored = (_demo_dir() / "votes.jsonl").read_text()
    rows = [json.loads(line) for line in stored.splitlines()]
    assert [row["choice"] for row in rows] == ["neither", "tie"]
    assert all(set(row) == {"choice", "left", "right", "at"} for row in rows)
    assert all({row["left"], row["right"]} == {"4", "2C"} for row in rows)
    assert "MARKER" not in stored and "gulls" not in stored


def test_vote_offers_only_the_four_choices(account: str) -> None:
    client = _page()
    headers = _visit(client)
    pair = client.post(ASK, json={"text": "Should the title fade up?"}, headers=headers).json()
    bad = client.post(VOTE, json={"pair_id": pair["pair_id"], "choice": "both"}, headers=headers)
    assert bad.status_code == 422
    page = client.get("/sleep/demo").text
    for choice in ("left", "right", "tie", "neither"):
        assert f'value="{choice}"' in page


# ── the limits ────────────────────────────────────────────────────────────────


def test_demo_rate_limited(account: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every cap through HTTP, from public addresses as Caddy forwards them."""
    from datetime import timedelta

    from fastapi.testclient import TestClient

    from anjo.sleep.app import demo_api
    from anjo.sleep.app.demo_live import build_service
    from anjo.sleep.demo.limits import Limits

    clock = [datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(demo_api, "_now", lambda: clock[0])
    limits = Limits(visits_per_client_per_day=2, visits_per_day=4, calls_per_client_per_minute=2,
                    asks_per_client_per_day=2, calls_per_day=10)
    serving, model = FakeServing(), ScriptedModel()
    client = TestClient(_behind_caddy(_client(build_service(serving=serving, arms=_arms(model, serving),
                                                            limits=limits)).app))

    def visit(address: str) -> Any:
        return client.post(VISIT, json={"adult": True}, headers={"X-Forwarded-For": address})

    def ask(address: str, token: Any) -> str:
        sent = {"X-Forwarded-For": address, HEADER: token.json()["token"]}
        answer = client.post(ASK, json={"text": "Where does the title go?"}, headers=sent)
        return "ok" if answer.status_code == 200 else next(k for k, v in words.COPY.items() if v == answer.json()["detail"])

    a, b, c = "8.8.8.8", "1.1.1.1", "9.9.9.9"
    a_visit = visit(a)
    assert visit(a).status_code == 200 and visit(a).json()["detail"] == words.COPY["visits_spent"]
    b_visit, c_visit = visit(b), visit(c)
    assert visit("4.4.4.4").json()["detail"] == words.COPY["resting_today"]  # the day's visits, for everyone
    assert [ask(a, a_visit), ask(a, a_visit), ask(b, b_visit)] == ["ok", "slow_down", "ok"]  # a's minute, not b's
    clock[0] += timedelta(seconds=61)
    assert [ask(a, a_visit), ask(b, b_visit)] == ["ok", "ok"]
    clock[0] += timedelta(seconds=61)
    assert [ask(a, a_visit), ask(c, c_visit), ask(b, b_visit)] == ["asks_spent", "ok", "resting_today"]
    assert len(model.calls) == 2 * 5  # both sides of each answered question; no refused one called anything
    counters = (_demo_dir() / "counters.json").read_text()
    assert not any(address in counters for address in (a, b, c, "127.0.0.1"))  # keyed buckets only


def test_the_limiter_fails_closed_on_the_host(account: str) -> None:
    client = _page()
    _demo_dir().mkdir(parents=True, exist_ok=True)
    (_demo_dir() / "counters.json").write_text("{torn")
    refused = client.post(VISIT, json={"adult": True})
    assert refused.status_code == 503 and refused.json()["detail"] == words.COPY["resting"]


# ── stateless: nothing of the visitor's is written ────────────────────────────


def _snapshot(root: Path) -> dict[str, str]:
    """Every file under the data root but the demo's own counters and votes, the budget ledger its warm
    time is debited to and the host's Glimmer turn marks (times only, ``serving.activity``), by content hash."""
    from anjo.sleep.app.storage import glimmer_turns

    demo, ledger, turns = _demo_dir(), str(_ledger().path), glimmer_turns()  # the ledger and its lock file
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file() and demo not in path.parents and turns not in path.parents
        and not str(path).startswith(ledger) and not path.name.endswith(("-wal", "-shm", "-journal"))
    }


def _real_context() -> Any:
    """Context C from Anjo's own builder as the demo and the arms build it: served briefs, Anjo's
    memory, preferences and recall over the real stores. Only the embedders are stand-ins (the
    ``offline_embedder`` fixture, and one for recall)."""
    from anjo.sleep.app.context_builder import RECALL, anjo_context_builder
    from anjo.sleep.app.storage import brief_store
    from anjo.sleep.context.brief_store import ServedBriefs

    return anjo_context_builder(ServedBriefs(brief_store), until=datetime(2026, 10, 31).date(), settings=RECALL,
                                embed=lambda text: [1.0, 0.0])


MEMORY = "She is cutting the ferry short and keeps the gulls in."


def _seed_memory() -> None:
    from anjo.memory import long_term

    long_term.store_memory("m-ferry", MEMORY, "calm", 0.1, ["film"], 0.6, ACCOUNT, "s-ferry", "familiar",
                           confidence=0.9)


def _memory_rows() -> str:
    """Every row of her memory store, both collections, as stored: ids, documents, metadata
    (``recall_count`` among it) and vectors."""
    from anjo.memory import long_term

    out = []
    for collection in long_term._get_collections(ACCOUNT):
        got = collection.get(include=["metadatas", "documents", "embeddings"])
        out.append(json.dumps({"ids": got["ids"], "documents": got["documents"], "metadatas": got["metadatas"],
                               "embeddings": [[float(x) for x in vector] for vector in got["embeddings"]]},
                              sort_keys=True))
    return "\n".join(out)


def _db_rows() -> dict[str, str]:
    """Every table's rows, by content: a changed row shows, not only an added one."""
    from anjo.core.db import get_db

    db = get_db()
    tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {table: hashlib.sha256(repr(sorted(repr(tuple(row)) for row in db.execute(f'SELECT * FROM "{table}"'))).encode()).hexdigest()  # noqa: S608
            for table in tables}


def _seed_evidence() -> Any:
    from anjo.sleep.app.storage import evidence_store
    from anjo.sleep.tests.sleep_fakes import ev

    store = evidence_store(ACCOUNT)
    store.add([ev("e1", "She keeps the diesel hum under the deck scenes.", person=ACCOUNT)])
    return store


def test_demo_chat_never_writes_evidence(account: str, offline_embedder: None) -> None:
    from anjo.sleep.app.storage import EVIDENCE_FILE, sleep_dir

    store = _seed_evidence()
    before_items = store.stored(datetime(2026, 12, 31).date())
    before = (sleep_dir(ACCOUNT) / EVIDENCE_FILE).read_bytes()
    model = ScriptedModel()
    client = _page(model, contexts=_real_context())
    headers = _visit(client)
    pair = client.post(ASK, json={"text": "MARKER-EVIDENCE: what should I cut?"}, headers=headers).json()
    assert pair["kind"] == "pair" and len(model.calls) == 2
    assert client.post(VOTE, json={"pair_id": pair["pair_id"], "choice": "right"}, headers=headers).status_code == 200
    assert (sleep_dir(ACCOUNT) / EVIDENCE_FILE).read_bytes() == before
    assert store.stored(datetime(2026, 12, 31).date()) == before_items


def test_page_writes_no_history_or_memory(account: str, offline_embedder: None) -> None:
    from anjo.core import paths

    _seed_evidence()
    _seed_memory()
    root = Path(paths.DATA_ROOT)
    rows, memory, files = _db_rows(), _memory_rows(), _snapshot(root)

    model = ScriptedModel()
    client = _page(model, contexts=_real_context())
    headers = _visit(client)
    for text in ("MARKER-HISTORY one about the ferry short", "MARKER-HISTORY two about the ferry short"):
        pair = client.post(ASK, json={"text": text}, headers=headers).json()
        assert pair["kind"] == "pair"
        client.post(VOTE, json={"pair_id": pair["pair_id"], "choice": "tie"}, headers=headers)

    assert any(MEMORY in json.dumps(call["messages"]) for call in model.calls)  # her memory was read
    assert _memory_rows() == memory  # and not one of its rows moved: no recall count, nothing added
    assert _db_rows() == rows  # no message, session, memory, surface or evidence row, added or changed
    assert _snapshot(root) == files  # no file outside the demo's counters, votes and warm time changed
    from anjo.sleep.app.storage import glimmer_turns

    for path in [*_demo_dir().iterdir(), *(p for p in glimmer_turns().rglob("*") if p.is_file())]:
        assert "MARKER" not in path.read_text()


def test_an_arms_run_reads_her_memory_without_a_trace(account: str, offline_embedder: None) -> None:
    """The probes' arms read C through the same builder: an arms run leaves her memory as it was."""
    from anjo.sleep.app.storage import checkpoint_registry, evidence_store
    from anjo.sleep.probes.arms import ArmDeps, reply
    from anjo.sleep.types import Probe

    _seed_memory()
    rows, memory = _db_rows(), _memory_rows()
    model = ScriptedModel()
    deps = ArmDeps(model=model, adapters=checkpoint_registry(), contexts=_real_context(), evidence=evidence_store,
                   logprobs=FakeLogprobs(lambda adapter, messages: ""), donors={}, until=datetime(2026, 10, 31).date())
    question = Probe("p-arms", ACCOUNT, "new_work", "How long should the ferry short run?", None, None, ())
    for arm in ("4", "2C"):
        reply(arm, ACCOUNT, question, deps=deps)
    assert len(model.calls) == 2 and all(MEMORY in json.dumps(call["messages"]) for call in model.calls)
    assert _memory_rows() == memory and _db_rows() == rows


def test_only_a_served_turn_counts_its_recall(account: str, offline_embedder: None) -> None:
    """A served turn's reply reaches the person, so its read counts as Anjo's recall, as chat's own
    read does; the builder the arms and the demo use counts nothing."""
    from anjo.memory import long_term
    from anjo.sleep.app import prompt
    from anjo.sleep.types import Probe

    def recalled() -> int:
        semantic, _ = long_term._get_collections(ACCOUNT)
        return int(semantic.get(ids=["m-ferry"], include=["metadatas"])["metadatas"][0].get("recall_count", 0))

    _seed_memory()
    question = Probe("p-served", ACCOUNT, "new_work", "the ferry short", None, None, ())
    _real_context().compose(ACCOUNT, question)
    assert recalled() == 0
    prompt.set_context_setup(None)
    try:
        prompt._builder(cleared=False).compose(ACCOUNT, question)
    finally:
        prompt.set_context_setup(None)
    assert recalled() == 1


# ── the GPU ───────────────────────────────────────────────────────────────────


def test_gpu_never_woken_by_default(account: str) -> None:
    from anjo.sleep.app.demo_live import glimmer_status

    asleep = FakeServing(up=False)
    client = _page(serving=asleep)
    headers = _visit(client)
    body = client.get("/api/sleep/demo/page")
    assert body.status_code == 200 and asleep.pings == 0  # the page loads with no request to Glimmer
    answer = client.post(ASK, json={"text": "Are you there?"}, headers=headers).json()
    assert answer == {"kind": "asleep", "waking": False}
    assert asleep.wakes == 0 and asleep.pings == 1
    modal = FakeServing(name="modal", up=True)
    assert glimmer_status(modal) is False and modal.pings == 0  # on Modal any request starts a container


def _ledger() -> Any:
    from anjo.sleep.app.storage import SLEEP_DIR, data_root
    from anjo.sleep.budget import LEDGER_NAME, Ledger

    return Ledger(data_root() / SLEEP_DIR / LEDGER_NAME)


def _window(*, until_s: float, beat_s: float = 0.0) -> None:
    """A keep-warm window as ``serving.wake`` writes it: ends ``until_s`` from now, last answered
    ping ``beat_s`` ago."""
    import time

    from anjo.sleep.app.storage import data_root
    from anjo.sleep.serving.wake import WarmWindow, _write_window, window_path

    now = time.time()
    _write_window(window_path(data_root()), WarmWindow(now + until_s, now - beat_s))


def test_waking_on_demand_is_opt_in_capped_and_paid_for(account: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import timedelta

    from anjo.sleep.app.demo_live import build_service
    from anjo.sleep.demo.limits import Limits, Refused
    from anjo.sleep.demo.service import Asleep
    from anjo.sleep.serving.sagemaker import settings

    asleep, now = FakeServing(up=False), datetime.now(timezone.utc)
    off = build_service(serving=asleep, arms=_arms(ScriptedModel(), asleep), spawn=lambda run: run())
    assert off.ask("c", off.start_visit("c", True, now), "wake?", now) == Asleep(waking=False)
    assert asleep.wakes == 0 and _ledger().used("modal_usd") == 0
    monkeypatch.setenv("ANJO_SLEEP_DEMO_WAKE", "1")
    on = build_service(serving=asleep, arms=_arms(ScriptedModel(), asleep), spawn=lambda run: run())
    token = on.start_visit("c", True, now)
    cap = Limits().wakes_per_day
    asked = [on.ask("c", token, f"wake {n}", now + timedelta(minutes=n)) for n in range(cap)]
    assert asked == [Asleep(waking=True)] * cap and asleep.wakes == cap
    with pytest.raises(Refused) as spent:
        on.ask("c", token, "once more", now + timedelta(minutes=cap))
    assert spent.value.reason == "live_spent" and asleep.wakes == cap
    # The first wake bought its wake and SageMaker's idle window; each later one only the minute it added.
    bought = settings.WAKE_TOTAL_TIMEOUT_S + settings.IDLE_MINUTES * 60 + 60 * (cap - 1)
    assert _ledger().used("modal_usd") == pytest.approx(SERVE_RATE * bought / 3600)


def test_answers_buy_their_warm_time_from_the_gpu_pool(account: str) -> None:
    from anjo.sleep.serving import client
    from anjo.sleep.serving.sagemaker import settings

    client_ = _page()
    headers = _visit(client_)
    assert client_.post(ASK, json={"text": "Should the gulls open the film?"}, headers=headers).json()["kind"] == "pair"
    first = SERVE_RATE * (client.REQUEST_TIMEOUT_S + settings.IDLE_MINUTES * 60) / 3600
    assert _ledger().used("modal_usd") == pytest.approx(first)  # the answer and SageMaker's idle window after it
    assert client_.post(ASK, json={"text": "And the ending?"}, headers=headers).json()["kind"] == "pair"
    assert first <= _ledger().used("modal_usd") < first + SERVE_RATE * 60 / 3600  # only the seconds it added
    assert not _ledger().open_holds()  # each purchase is settled as it is made


def test_without_a_gpu_price_nothing_is_asked_live(account: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_SLEEP_PRICES")
    model = ScriptedModel()
    client = _page(model)
    refused = client.post(ASK, json={"text": "Is she there?"}, headers=_visit(client))
    assert refused.status_code == 429 and refused.json()["detail"] == words.COPY["live_spent"]
    assert model.calls == [] and _ledger().used("modal_usd") == 0


def test_inside_a_keep_warm_window_she_answers_on_either_backend_without_waking(account: str) -> None:
    for name in ("modal", "sagemaker"):
        _window(until_s=3600)
        serving, model = FakeServing(name=name, up=True), ScriptedModel()
        client = _page(model, serving)
        pair = client.post(ASK, json={"text": f"Where does the {name} title go?"}, headers=_visit(client)).json()
        assert pair["kind"] == "pair" and len(model.calls) == 2, name
        assert serving.wakes == 0 and serving.pings == (0 if name == "modal" else 1), name
        assert client.get("/api/sleep/demo/page").json()["live"] == "open"
    assert _ledger().used("modal_usd") == 0  # the window's owner pays for it, not the demo


def test_on_modal_outside_a_window_she_stays_asleep_and_unasked(account: str) -> None:
    for window in (None, {"until_s": 3600.0, "beat_s": 400.0}, {"until_s": -1.0}):  # none, stale pings, ended
        if window is not None:
            _window(**window)
        modal = FakeServing(name="modal", up=True)
        client = _page(serving=modal)
        answer = client.post(ASK, json={"text": "Are you awake?"}, headers=_visit(client)).json()
        assert answer == {"kind": "asleep", "waking": False} and modal.pings == 0 and modal.wakes == 0, window


def test_past_the_warm_cap_the_page_runs_on_starters_and_says_so(account: str) -> None:
    from anjo.sleep.app.demo_live import build_service
    from anjo.sleep.demo.limits import Limits

    _write_starters()
    serving, model = FakeServing(), ScriptedModel()
    tight = build_service(serving=serving, arms=_arms(model, serving), limits=Limits(warm_minutes_per_day=30))
    client = _client(tight)  # an answer keeps SageMaker up 35 minutes: more than the day allows
    assert client.get("/api/sleep/demo/page").json()["live"] == "spent"
    headers = _visit(client)
    refused = client.post(ASK, json={"text": "One more cut?"}, headers=headers)
    assert refused.status_code == 429 and refused.json()["detail"] == words.COPY["live_spent"]
    assert model.calls == [] and serving.pings == 0  # not even asked: on SageMaker a ping keeps her up
    assert client.post("/api/sleep/demo/starters/nobody-to-tell", headers=headers).status_code == 200
    assert "COPY-PENDING(kevin)" in Path(words.__file__).read_text() and words.COPY["live_spent"]


def test_outside_the_live_hours_nothing_is_bought(account: str, monkeypatch: pytest.MonkeyPatch) -> None:
    hour = datetime.now(timezone.utc).hour
    monkeypatch.setenv("ANJO_SLEEP_DEMO_LIVE_HOURS", f"{(hour + 2) % 24}-{(hour + 4) % 24}")
    model = ScriptedModel()
    client = _page(model)
    assert client.get("/api/sleep/demo/page").json()["live"] == "closed"
    refused = client.post(ASK, json={"text": "Now?"}, headers=_visit(client))
    assert refused.status_code == 429 and refused.json()["detail"] == words.COPY["live_closed"]
    assert model.calls == [] and _ledger().used("modal_usd") == 0
    _window(until_s=3600)  # a keep-warm window someone else holds opens it: nothing to buy
    assert client.get("/api/sleep/demo/page").json()["live"] == "open"


# ── the starter pairs ─────────────────────────────────────────────────────────


def test_starter_pairs_load_without_gpu_and_are_labeled(account: str) -> None:
    from anjo.sleep.app.demo_pairs import write_starters
    from anjo.sleep.demo.starters import LABEL, load

    model, serving = ScriptedModel(reply=lambda n: f"pre-generated reply {n}"), FakeServing()
    out = _demo_dir() / "starter_pairs.json"
    write_starters(out, _arms(model, serving), checkpoint=None, baseline="2C", serve_usd_per_hour=3.6,
                   now=datetime(2026, 10, 31, tzinfo=timezone.utc))
    starters = load(out)
    assert starters is not None and starters.label == LABEL and starters.checkpoint is None
    assert len(model.calls) == 2 * len(words.SCENES)

    serving.up = False  # asleep: the starters still load
    client = _page(model, serving)
    body = client.get("/api/sleep/demo/page").json()
    assert all(scene["ready"] for scene in body["scenes"])
    assert body["pregenerated"]["label"] == LABEL
    headers = _visit(client)
    pair = client.post(f"/api/sleep/demo/starters/{words.SCENES[0].id}", headers=headers).json()
    assert pair["source"]["kind"] == LABEL and pair["left"]["text"].startswith("pre-generated reply")
    assert len(model.calls) == 2 * len(words.SCENES) and serving.pings == 0 and serving.wakes == 0
