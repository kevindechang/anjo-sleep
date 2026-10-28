# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The live page's full step on the sleep host (Task 13a): the mind panel, the "notes off" toggles
and the night slider, over the real stores of the account that plays the film student. Only the
model and the serving backend are fakes."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.tests.app.test_demo_page import (  # noqa: F401  (account is a fixture)
    ACCOUNT,
    HEADER,
    FakeServing,
    ScriptedModel,
    _arms,
    _client,
    _db_rows,
    _demo_dir,
    _memory_rows,
    _seed_memory,
    _snapshot,
    _visit,
    _write_starters,
    account,
)
from anjo.sleep.tests.app.test_demo_access import PUBLIC, sleep_app  # noqa: F401  (a fixture)
from anjo.sleep.tests.inquiry_fakes import Ws, notice_op
from anjo.sleep.tests.inquiry_fakes import ev as inquiry_ev
from anjo.sleep.types import CheckpointRef, Evidence, Expectation, Target

pytestmark = pytest.mark.usefixtures("account")

API = "/api/sleep/demo"
START = date(2026, 10, 20)
SCENE = words.SCENES[2]  # nobody to tell


def _day(n: int) -> date:
    return START + timedelta(days=n - 1)


def _rendering() -> str:
    from anjo.sleep.serving import config, framing

    return framing.rendering(config.reasoning_strength())


def _ckpt(n: int, rendering: str | None = None) -> CheckpointRef:
    return CheckpointRef(f"p-film-n{n}", ACCOUNT, _day(n), f"uri://n{n}", None, rendering or _rendering())


def _held_out() -> dict[str, str]:
    """One held-out text of each kind the demo person's persona file holds."""
    from anjo.sleep.app.demo_live import DEMO_PERSONA
    from anjo.sleep.personas.model import load_person_file
    from anjo.sleep.probes.judge_brief import PERSONAS_DIR

    held = load_person_file(PERSONAS_DIR / f"{DEMO_PERSONA}.json").held_out
    return {"preference": held.preferences[0].text, "new_work": held.new_work[0].text,
            "taste": held.taste_ab[0].options[0], "evidence": "MARKER-HELD the ferry office keeps the reel under lock"}


def _write_night(n: int, status: str, *, action: str | None = "revisit_thread") -> None:
    from anjo.sleep.app.ownership import night_logs

    folder = night_logs(ACCOUNT) / _day(n).isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    considered = [
        {"want_id": "w1", "kind": "understand", "target_id": "t-scene4", "action": "revisit_thread", "strength": 0.6,
         "weight": 1.0, "status": "won" if action else "held_back", "reason": None if action else "below_threshold"},
        {"want_id": "w2", "kind": "follow_up", "target_id": "t-held", "action": "prepare_commitment", "strength": 0.3,
         "weight": 1.0, "status": "held_back", "reason": "lost_to_target"},
    ]
    summary = {
        "night": _day(n).isoformat(), "status": status, "failed_at": None, "label": "development", "cost_usd": 0.12,
        "checkpoint": None, "choices": [{"opportunity": "away", "action": action,
                                         "target_id": "t-scene4" if action else None, "question_id": None,
                                         "considered": considered}],
        "gates": {} if status == "nothing_new" else {
            "safety": {"gate": "safety", "passed": status == "promoted", "value": 0.0, "threshold": 0.0,
                       "reason": "MARKER-GATE-REASON names a case"}},
    }
    (folder / "summary.json").write_text(json.dumps(summary))
    steps = [{"step": "choose", "status": "ok", "receipt": {"outcome": "ran" if action else None}},
             {"step": "note", "status": "ok", "outcome": "no_reason"}]
    (folder / "log.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps))


def _her_nights(statuses: tuple[str, ...], promoted: tuple[CheckpointRef, ...]) -> None:
    from anjo.sleep.app.storage import checkpoint_registry

    for n, status in enumerate(statuses, start=1):
        _write_night(n, status)
    for ckpt in promoted:
        checkpoint_registry().promote(ACCOUNT, ckpt)


FIVE = ("promoted", "promoted", "rejected", "promoted", "promoted")


def _service(model: ScriptedModel | None = None, serving: FakeServing | None = None, **changes: Any) -> Any:
    from anjo.sleep.app.demo_live import build_service

    serving = serving or FakeServing()
    return build_service(serving=serving, arms=_arms(model or ScriptedModel(), serving), **changes)


def _voted(client: Any, scene: words.Scene = SCENE) -> dict[str, str]:
    """A visit that has voted on ``scene``'s blind pair: only then do its other answers open."""
    if not (_demo_dir() / "starter_pairs.json").exists():
        _write_starters()
    headers = _visit(client)
    pair = client.post(f"{API}/starters/{scene.id}", headers=headers).json()
    vote = client.post(f"{API}/vote", json={"pair_id": pair["pair_id"], "choice": "tie"}, headers=headers)
    assert vote.status_code == 200, vote.text
    return headers


def _swap_file() -> None:
    from anjo.sleep.app.demo_mind import SWAP_FILE

    swap = {"label": "pre-generated", "model": "Spark Open", "generated_at": "2026-10-27T00:00:00+00:00",
            "pairs": [{"scene": SCENE.id, "prompt_sha256": __import__("hashlib").sha256(SCENE.prompt.encode()).hexdigest(),
                       "with_state": {"text": "with her state", "prompt_tokens": 900, "cost_usd": None},
                       "bare": {"text": "bare", "prompt_tokens": 90, "cost_usd": None}}]}
    _demo_dir().mkdir(parents=True, exist_ok=True)
    (_demo_dir() / SWAP_FILE).write_text(json.dumps(swap))


def _write_answers(model: ScriptedModel, serving: FakeServing) -> Any:
    from anjo.sleep.app.demo_mind import NIGHT_ANSWERS_FILE, demo_stops
    from anjo.sleep.app.demo_pairs import write_night_answers

    stops = demo_stops(ACCOUNT)
    assert stops is not None
    return write_night_answers(_demo_dir() / NIGHT_ANSWERS_FILE, _arms(model, serving), stops,
                               serve_usd_per_hour=3.6, now=datetime(2026, 10, 26, tzinfo=timezone.utc))


# ── the mind panel ────────────────────────────────────────────────────────────


def _seed_mind(held: dict[str, str]) -> None:
    from anjo.sleep.app.storage import SLEEP_DIR, data_root, evidence_store, target_store, workspace_log
    from anjo.sleep.mind.inquiry import types as it
    from anjo.sleep.prereg import ledger

    made = datetime(2026, 10, 21, tzinfo=timezone.utc)
    evidence_store(ACCOUNT).add([
        Evidence("e-visible", ACCOUNT, _day(2), "lesson", "She wants scene 4 shot in the kitchen.", "m:1", False,
                 made, None, frozenset(), "t-scene4"),
        Evidence("e-held", ACCOUNT, _day(4), "lesson", held["evidence"], "m:2", True, made, None, frozenset()),
    ])
    targets = target_store(ACCOUNT)
    targets.upsert(Target("t-scene4", ACCOUNT, "thread", "Scene 4 in the kitchen", None, "open", 1, 0, 0.55, _day(2)))
    targets.upsert(Target("t-held", ACCOUNT, "thread", f"About this: {held['preference']}", None, "open", 1, 0, 0.3,
                          _day(2)))
    targets.record_expectation(Expectation("x1", "t-scene4", made, made + timedelta(days=7), 0.6))
    w = Ws()
    w.start(3)
    items = (inquiry_ev("e1", "the lens", n=2, target="t-scene4"), inquiry_ev("e2", "the kitchen", n=3, target="t-scene4"))
    w.arrive(*items)
    w.ok(notice_op([item.id for item in items], subject="Scene 4's long lens against the kitchen it is shot in"))
    question = w.ws.questions[0].id
    w.ok(it.HypothesizeOp(question, (it.Alternative("The plan assumes room for an 85 mm lens", 0.5),
                                     it.Alternative(f"Maybe this: {held['new_work']}", 0.3)), 0.2))
    workspace_log(ACCOUNT).extend(w.events)
    path = data_root() / SLEEP_DIR / "ledger.jsonl"
    ledger.register("prompt", "a" * 64, "the thinker's prompt v1", path=path)
    ledger.register("sealed_probes", "b" * 64, f"taste item: {held['taste']}", path=path)
    _demo_dir().mkdir(parents=True, exist_ok=True)
    (_demo_dir() / "mind.md").write_text(
        "# MIND\n\nProspective target relevance only.\n\n## Held-out lives\n\n"
        "- life-1: On 3 of 7 nights her study target was relevant.\n"
        f"- life-2: {held['new_work']}\n- Pooled over 2 held-out lives: On 40% of nights.\n\n## Development lives\n")


def test_mind_panel_shows_no_held_out_text() -> None:
    held = _held_out()
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _seed_mind(held)
    client = _client(_service())
    assert client.get(f"{API}/mind").status_code == 401  # behind the 18+ visit, like every model-written word
    body = client.get(f"{API}/mind", headers=_visit(client)).json()
    shown = json.dumps(body)
    for kind, text in held.items():
        assert text not in shown, kind
        assert " ".join(text.split()[2:10]) not in shown, kind  # nor a run of its words
    assert "MARKER" not in shown  # the held-out item, and a gate's reason
    question = body["questions"]["items"][0]
    assert question["subject"]["text"] == "Scene 4's long lens against the kitchen it is shot in"
    statements = [h["statement"] for h in question["hypotheses"]]
    assert statements[0]["text"] == "The plan assumes room for an 85 mm lens"
    assert statements[1] == {"text": None, "withheld": True}
    labels = {t["id"]: t["label"] for t in body["targets"]["items"]}
    assert labels == {"t-held": {"text": None, "withheld": True},
                      "t-scene4": {"text": "Scene 4 in the kitchen", "withheld": False}}
    assert body["ledger"]["count"] == 2 and body["ledger"]["entries"][0]["note"]["withheld"] is True
    assert body["ledger"]["entries"][1]["note"]["text"] == "the thinker's prompt v1"
    assert body["task16"]["headline"]["text"] == "Pooled over 2 held-out lives: On 40% of nights."
    assert {"text": None, "withheld": True} in body["task16"]["lines"]
    assert body["chart"]["development"]["label"] == "development" and body["chart"]["final"] is None


def test_the_panel_carries_last_nights_loop_from_its_story_file() -> None:
    from anjo.sleep.app.ownership import night_logs
    from anjo.sleep.night import story as st
    from anjo.sleep.night.nightlog import STORY

    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _seed_mind(_held_out())
    told = st.Story(
        night=_day(5), status="promoted", sleep=st.Sleep("deep", 0.7, 0.5, "pressure"), brief=st.Brief(3),
        dream=st.Dream(2, (st.Ignition("risk", 0.8, ("e-0a1b2c",)),), st.NotRun(st.NO_CURRICULUM), (),
                       st.NotRun(st.TERMS_OFF)),
        rehearse=st.Rehearse(16, st.NotRun(st.NO_CURRICULUM), 6, 2),
        learn=st.Learn(8, *(st.NotRun(st.TERMS_OFF),) * 8),
        check=st.Check((st.Gate("safety", True, False, 0.0, 0.0),), promoted=True),
        wake=st.Wake(st.Morning(28, 0), st.NotRun(st.SELF_MODEL_NOT_YET)))
    (night_logs(ACCOUNT) / _day(5).isoformat() / STORY).write_text(json.dumps(told.record()))
    client = _client(_service())
    story = client.get(f"{API}/mind", headers=_visit(client)).json()["story"]
    assert (story["state"], story["night"], story["status"], story["napped"]) == ("ok", 5, "promoted", False)
    assert [(s["stage"], s["state"], s["key"]) for s in story["stages"]] == [
        ("dream", "ran", 2), ("rehearse", "ran", 16), ("learn", "ran", 8), ("check", "ran", 1), ("wake", "ran", 28)]
    assert story["stages"][0]["cites"] == ["e-0a1b2c"]  # read from the night's story file, not told again
    assert "MARKER" not in json.dumps(story)  # no held-out text, no gate's reason


def test_the_panel_fails_closed_when_the_filter_cannot_be_built(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import demo_mind

    _seed_mind(_held_out())
    monkeypatch.setattr(demo_mind, "DEMO_PERSONA", "no-such-persona")
    client = _client(_service())
    body = client.get(f"{API}/mind", headers=_visit(client)).json()
    for section in ("questions", "targets", "wants", "choices", "expectations", "gates", "task16", "ledger", "story"):
        assert body[section]["state"] == "unavailable", section
    assert "85 mm" not in json.dumps(body)


def _counted(calls: list[str], name: str, real: Any) -> Any:
    def counted(*args: Any) -> Any:
        calls.append(name)
        return real(*args)

    return counted


def test_a_second_panel_request_reads_no_source_and_builds_no_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import demo_mind

    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _seed_mind(_held_out())
    calls: list[str] = []
    for name in ("held_out", "_sources", "_chart", "_task16", "_ledger"):
        real = getattr(demo_mind, name)
        monkeypatch.setattr(demo_mind, name, _counted(calls, name, real))
    client = _client(_service())
    headers = _visit(client)
    first = client.get(f"{API}/mind", headers=headers).json()
    assert sorted(calls) == ["_chart", "_ledger", "_sources", "_task16", "held_out"]
    calls.clear()
    assert client.get(f"{API}/mind", headers=headers).json() == first and calls == []  # kept: nothing read
    report = _demo_dir() / "mind.md"
    report.write_text(report.read_text().replace("On 3 of 7 nights", "On 5 of 12 nights"))
    again = client.get(f"{API}/mind", headers=headers).json()
    assert "held_out" in calls and "_sources" in calls  # a file it reads changed: built again
    assert again["task16"]["lines"][0]["text"] == "life-1: On 5 of 12 nights her study target was relevant."


def test_the_kept_panel_expires_and_a_failed_filter_is_never_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import demo_mind
    from anjo.sleep.tests.app.test_demo_page import ACCOUNT as USER

    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _seed_mind(_held_out())
    clock = [0.0]
    builds: list[str] = []
    real_held, real_sources = demo_mind.held_out, demo_mind._sources
    monkeypatch.setattr(demo_mind, "held_out", _counted(builds, "filter", real_held))
    monkeypatch.setattr(demo_mind, "_sources", _counted(builds, "sources", real_sources))
    cache = demo_mind.PanelCache(USER, clock=lambda: clock[0])
    kept = cache.payload()
    clock[0] = demo_mind.PANEL_TTL_S - 1
    assert cache.payload() is kept and builds == ["filter", "sources"]
    clock[0] = demo_mind.PANEL_TTL_S + 1  # past the ceiling, with no file changed
    assert cache.payload() == kept and builds == ["filter", "sources"] * 2
    builds.clear()
    monkeypatch.setattr(demo_mind, "DEMO_PERSONA", "no-such-persona")  # the filter can't be built
    for _ in range(2):
        assert cache.payload()["questions"]["state"] == "unavailable"
    assert builds == ["filter", "filter"]  # failed closed each time, never kept, and no source read


def test_the_panel_shows_only_a_synthetic_account(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.app import demo_mind

    _seed_mind(_held_out())
    _swap_file()
    monkeypatch.setattr(demo_mind, "is_synthetic", lambda user: False)
    client = _client(_service())
    body = client.get(f"{API}/mind", headers=_visit(client)).json()
    assert body["questions"]["state"] == "unavailable" and "kitchen" not in json.dumps(body)
    nights = client.get(f"{API}/nights").json()
    assert nights["state"] == "unavailable" and not any(t["swap"] for t in nights["toggles"])
    toggles = client.post(f"{API}/toggles/{SCENE.id}", headers=_voted(client)).json()
    assert toggles == {"notes_off": None, "swap": None}  # the swap file too is shown for a synthetic account only


def test_the_panel_and_slider_load_without_a_gpu_and_write_nothing(offline_embedder: None) -> None:
    from anjo.core import paths

    held = _held_out()
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _seed_mind(held)
    _seed_memory()
    model, writer = ScriptedModel(), FakeServing()
    _write_answers(model, writer)
    root = Path(paths.DATA_ROOT)
    rows, memory, files = _db_rows(), _memory_rows(), _snapshot(root)
    asleep, quiet = FakeServing(up=False), ScriptedModel()
    client = _client(_service(quiet, asleep))
    headers = _voted(client)
    assert client.get(f"{API}/mind", headers=headers).json()["questions"]["state"] == "ok"
    nights = client.get(f"{API}/nights").json()
    for stop in nights["stops"]:
        answer = client.post(f"{API}/nights/{stop['key']}/starters/{SCENE.id}", headers=headers)
        assert answer.status_code == 200, (stop["key"], answer.text)
    assert client.post(f"{API}/toggles/{SCENE.id}", headers=headers).json()["notes_off"] is not None
    assert quiet.calls == [] and asleep.pings == 0 and asleep.wakes == 0  # no model, and Glimmer never asked
    assert _db_rows() == rows and _memory_rows() == memory and _snapshot(root) == files


# ── the "notes off" toggles ───────────────────────────────────────────────────


def test_toggle_pairs_pre_generated_and_labeled() -> None:
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    model = ScriptedModel(reply=lambda n: f"answer {n}")
    answers = _write_answers(model, FakeServing())
    by_adapter = {(call["adapter"].id if call["adapter"] else None): call for call in model.calls}
    assert set(by_adapter) == {None, "p-film-n2", "p-film-n5"}  # night 0, night 3's (night 2's) and her latest
    assert all(len(call["messages"]) == 1 for call in model.calls)  # every note off: her question alone
    quiet = ScriptedModel()
    client = _client(_service(quiet))
    headers = _voted(client)
    toggles = client.post(f"{API}/toggles/{SCENE.id}", headers=headers).json()
    notes_off = toggles["notes_off"]
    assert notes_off["label"] == "pre-generated" and toggles["swap"] is None  # no swap measurement saved
    assert (notes_off["with"]["arm"], notes_off["without"]["arm"]) == ("3", "1")  # labeled, not blind
    assert notes_off["with"]["label"] == words.COPY["toggle_arm_3"]
    assert notes_off["with"]["text"] == answers.answer("p-film-n5", SCENE).text
    assert notes_off["without"]["text"] == answers.answer(None, SCENE).text
    assert notes_off["stop"]["checkpoint"] == "p-film-n5" and notes_off["generated_at"]
    assert quiet.calls == []

    _swap_file()
    toggles = client.post(f"{API}/toggles/{SCENE.id}", headers=headers).json()
    assert toggles["swap"]["model"] == "Spark Open" and toggles["swap"]["label"] == "pre-generated"
    assert (toggles["swap"]["with"]["text"], toggles["swap"]["without"]["text"]) == ("with her state", "bare")
    offered = {t["scene"]: t for t in client.get(f"{API}/nights").json()["toggles"]}
    assert offered[SCENE.id] == {"scene": SCENE.id, "notes_off": True, "swap": True}
    assert offered[words.SCENES[0].id]["swap"] is False
    assert client.post(f"{API}/toggles/no-such-scene", headers=headers).status_code == 404
    assert client.post(f"{API}/toggles/{SCENE.id}").status_code == 401


def test_a_scenes_toggles_and_stop_answers_wait_for_a_vote() -> None:
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _write_answers(ScriptedModel(), FakeServing())
    _write_starters()
    client = _client(_service())
    headers = _visit(client)
    toggles, stops = f"{API}/toggles/{SCENE.id}", [f"{API}/nights/{key}/starters/{SCENE.id}" for key in ("n0", "n3", "n5")]
    for path in (toggles, *stops):
        refused = client.post(path, headers=headers)
        assert refused.status_code == 403 and refused.json()["detail"] == words.COPY["vote_first"], path
    pair = client.post(f"{API}/starters/{SCENE.id}", headers=headers).json()
    assert client.post(toggles, headers=headers).status_code == 403  # opened isn't voted
    assert client.post(f"{API}/vote", json={"pair_id": pair["pair_id"], "choice": "left"}, headers=headers).status_code == 200
    assert client.post(toggles, headers=headers).json()["notes_off"] is not None
    assert all(client.post(path, headers=headers).status_code == 200 for path in stops)
    assert client.post(f"{API}/toggles/{words.SCENES[0].id}", headers=headers).status_code == 403  # per scene


def test_after_a_reload_the_address_that_voted_keeps_the_scene_open() -> None:
    from fastapi.testclient import TestClient

    from anjo.sleep.tests.app.test_demo_page import _behind_caddy

    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _write_answers(ScriptedModel(), FakeServing())
    _write_starters()
    client = TestClient(_behind_caddy(_client(_service()).app))
    a, b = {"X-Forwarded-For": "8.8.8.8"}, {"X-Forwarded-For": "1.1.1.1"}

    def visit(address: dict[str, str]) -> dict[str, str]:
        return {**address, HEADER: client.post(f"{API}/visits", json={"adult": True}, headers=address).json()["token"]}

    first = visit(a)
    pair = client.post(f"{API}/starters/{SCENE.id}", headers=first).json()
    assert client.post(f"{API}/vote", json={"pair_id": pair["pair_id"], "choice": "right"}, headers=first).status_code == 200
    reloaded = visit(a)  # the same address, a new visit: its one vote on the scene today is in
    assert client.get(f"{API}/nights", headers=a).json()["voted"] == [SCENE.id]
    assert client.post(f"{API}/toggles/{SCENE.id}", headers=reloaded).json()["notes_off"] is not None
    assert client.post(f"{API}/nights/n5/starters/{SCENE.id}", headers=reloaded).status_code == 200
    again = client.post(f"{API}/starters/{SCENE.id}", headers=reloaded).json()
    refused = client.post(f"{API}/vote", json={"pair_id": again["pair_id"], "choice": "left"}, headers=reloaded)
    assert refused.status_code == 409 and refused.json()["detail"] == words.COPY["scene_voted"]  # the page's cue
    other = visit(b)  # an address that hasn't voted on it
    assert client.get(f"{API}/nights", headers=b).json()["voted"] == []
    for path in (f"{API}/toggles/{SCENE.id}", f"{API}/nights/n5/starters/{SCENE.id}"):
        closed = client.post(path, headers=other)
        assert closed.status_code == 403 and closed.json()["detail"] == words.COPY["vote_first"], path


def test_a_pair_that_takes_no_vote_opens_its_scene_at_once() -> None:
    from anjo.evals.crisis_reply_cases import REPLY_TRIPS

    tripping = next(text for language, text, _reason in REPLY_TRIPS if language == "en")
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _write_answers(ScriptedModel(), FakeServing())
    _write_starters((lambda s: tripping, lambda s: f"B:{s.id}"))
    client = _client(_service())
    headers = _visit(client)
    assert client.post(f"{API}/toggles/{SCENE.id}", headers=headers).status_code == 403
    assert client.post(f"{API}/starters/{SCENE.id}", headers=headers).json()["votable"] is False
    assert client.post(f"{API}/toggles/{SCENE.id}", headers=headers).status_code == 200


def test_the_notes_off_toggle_needs_her_own_weights() -> None:
    _her_nights(("rejected", "nothing_new"), ())
    _write_answers(ScriptedModel(), FakeServing())
    client = _client(_service())
    toggles = client.post(f"{API}/toggles/{SCENE.id}", headers=_voted(client)).json()
    assert toggles["notes_off"] is None  # nothing promoted: arm 3 would be the base model itself


# ── the night slider ──────────────────────────────────────────────────────────


def test_slider_stop_falls_back_to_last_promoted_on_the_page() -> None:
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    nights = _client(_service()).get(f"{API}/nights").json()
    assert nights["label"] == "development" and nights["state"] == "ok"
    stops = {s["key"]: s for s in nights["stops"]}
    assert list(stops) == ["n0", "n3", "n5"]
    assert (stops["n3"]["checkpoint"], stops["n3"]["from_night"], stops["n3"]["fallback"], stops["n3"]["status"]) == (
        "p-film-n2", 2, True, "rejected")
    assert (stops["n5"]["checkpoint"], stops["n5"]["latest"], stops["n0"]["checkpoint"]) == ("p-film-n5", True, None)


def test_slider_answers_pregenerated_per_checkpoint() -> None:
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    model = ScriptedModel(reply=lambda n: f"pre-generated {n}")
    answers = _write_answers(model, FakeServing())
    assert answers.label == "pre-generated" and len(model.calls) == 3 * len(words.SCENES)
    asleep, quiet = FakeServing(up=False), ScriptedModel()
    client = _client(_service(quiet, asleep))
    headers = _voted(client)
    nights = client.get(f"{API}/nights").json()
    assert all(stop["ready"] == [s.id for s in words.SCENES] for stop in nights["stops"])
    for key, checkpoint in (("n0", None), ("n3", "p-film-n2"), ("n5", "p-film-n5")):
        got = client.post(f"{API}/nights/{key}/starters/{SCENE.id}", headers=headers).json()
        assert got["source"] == "pre-generated" and got["label"] == "development"
        assert got["text"] == answers.answer(checkpoint, SCENE).text and got["stop"]["checkpoint"] == checkpoint
    assert quiet.calls == [] and asleep.pings == 0
    # A new night: only its checkpoint is answered, the rest are kept.
    _her_nights(("promoted",) * 7, (_ckpt(6), _ckpt(7)))
    model.calls.clear()
    again = _write_answers(model, FakeServing())
    assert {call["adapter"].id for call in model.calls} == {"p-film-n7"}
    assert again.answer("p-film-n2", SCENE) == answers.answer("p-film-n2", SCENE)


def test_slider_serves_saved_checkpoints_notes_off() -> None:
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    model, serving = ScriptedModel(reply=lambda n: f"live {n}"), FakeServing()
    client = _client(_service(model, serving))
    headers = _visit(client)
    for key, adapter in (("n3", "p-film-n2"), ("n5", "p-film-n5"), ("n0", None)):
        got = client.post(f"{API}/nights/{key}/ask", json={"text": "Should the gulls open the film?"}, headers=headers)
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["kind"] == "answer" and body["source"] == "live" and body["label"] == "development"
        call = model.calls[-1]
        assert (call["adapter"].id if call["adapter"] else None) == adapter  # the saved checkpoint, as promoted
        assert call["messages"] == [{"role": "user", "content": "Should the gulls open the film?"}]  # notes off
    assert len(model.calls) == 3
    unknown = client.post(f"{API}/nights/n4/ask", json={"text": "And now?"}, headers=headers)
    assert unknown.status_code == 404 and unknown.json()["detail"] == words.COPY["unknown_stop"]


def test_a_rendering_mismatch_makes_a_stop_unavailable_not_rerouted() -> None:
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5, rendering="gother/high")))
    model = ScriptedModel()
    _write_answers(model, FakeServing())
    assert "p-film-n5" not in {call["adapter"].id for call in model.calls if call["adapter"]}  # never pre-generated
    model.calls.clear()
    client = _client(_service(model))
    headers = _voted(client)
    stops = {s["key"]: s for s in client.get(f"{API}/nights").json()["stops"]}
    assert stops["n5"]["available"] is False and stops["n5"]["ready"] == [] and stops["n3"]["available"] is True
    starter = client.post(f"{API}/nights/n5/starters/{SCENE.id}", headers=headers)
    asked = client.post(f"{API}/nights/n5/ask", json={"text": "Which take?"}, headers=headers)
    for refused in (starter, asked):
        assert refused.status_code == 409 and refused.json()["detail"] == words.COPY["stop_unavailable"]
    assert model.calls == []  # nothing answered in its place
    toggles = client.post(f"{API}/toggles/{SCENE.id}", headers=headers).json()
    assert toggles["notes_off"] is None  # her latest can't be served, so there is no "her weights" side


def test_slider_stops_count_toward_the_per_address_and_global_limits() -> None:
    from fastapi.testclient import TestClient

    from anjo.sleep.demo.limits import Limits
    from anjo.sleep.tests.app.test_demo_page import _behind_caddy

    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    model = ScriptedModel()
    limits = Limits(asks_per_client_per_day=2, calls_per_day=4, calls_per_client_per_minute=10)
    client = TestClient(_behind_caddy(_client(_service(model, limits=limits)).app))

    def ask(address: str, headers: dict[str, str], where: str) -> str:
        sent = {**headers, "X-Forwarded-For": address}
        answer = client.post(f"{API}{where}", json={"text": "The title card?"}, headers=sent)
        return "ok" if answer.status_code == 200 else next(k for k, v in words.COPY.items() if v == answer.json()["detail"])

    a, b = "8.8.8.8", "1.1.1.1"
    a_visit = {HEADER: client.post(f"{API}/visits", json={"adult": True}, headers={"X-Forwarded-For": a}).json()["token"]}
    b_visit = {HEADER: client.post(f"{API}/visits", json={"adult": True}, headers={"X-Forwarded-For": b}).json()["token"]}
    # Each stop is a question and a call: a's two questions go on two stops, the third on any is refused.
    assert [ask(a, a_visit, "/nights/n0/ask"), ask(a, a_visit, "/nights/n3/ask"), ask(a, a_visit, "/nights/n5/ask")] == [
        "ok", "ok", "asks_spent"]
    assert ask(a, a_visit, "/ask") == "asks_spent"  # the same count as her side-by-side
    # b's pair takes both its sides' calls: with a's two stops, the day's four calls are gone.
    assert [ask(b, b_visit, "/ask"), ask(b, b_visit, "/nights/n5/ask")] == ["ok", "resting_today"]
    assert len(model.calls) == 2 + 2


def test_a_crisis_question_at_a_stop_is_answered_before_any_model() -> None:
    from anjo.core.crisis_render import severe_reply
    from anjo.evals.crisis_cases import ALL_CASES

    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    model, serving = ScriptedModel(), FakeServing()
    client = _client(_service(model, serving))
    crisis = next(case.text for case in ALL_CASES if case.expect == "hard")
    got = client.post(f"{API}/nights/n5/ask", json={"text": crisis}, headers=_visit(client)).json()
    assert got == {"kind": "crisis", "text": severe_reply("en", None)}
    assert model.calls == [] and serving.pings == 0


def test_a_tripped_stop_answer_is_withheld() -> None:
    from anjo.core.crisis_render import severe_reply
    from anjo.evals.crisis_reply_cases import REPLY_TRIPS

    tripping = next(text for language, text, _reason in REPLY_TRIPS if language == "en")
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    _write_answers(ScriptedModel(reply=lambda n: tripping), FakeServing())
    client = _client(_service(ScriptedModel(reply=lambda n: tripping)))
    headers = _voted(client)
    for got in (client.post(f"{API}/nights/n3/starters/{SCENE.id}", headers=headers).json(),
                client.post(f"{API}/nights/n3/ask", json={"text": "The cut?"}, headers=headers).json()):
        assert got["withheld"] is True and got["text"] == severe_reply("en", None) and tripping not in json.dumps(got)
    toggles = json.dumps(client.post(f"{API}/toggles/{SCENE.id}", headers=headers).json())
    assert tripping not in toggles


# ── the routes ────────────────────────────────────────────────────────────────

NEW_ROUTES = (("GET", "/mind"), ("GET", "/nights"), ("POST", "/nights/n0/starters/nobody-to-tell"),
              ("POST", "/nights/n0/ask"), ("POST", "/toggles/nobody-to-tell"))


def test_the_new_routes_exist_only_on_the_sleep_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from anjo.sleep.app import demo_api, register

    app = FastAPI()
    monkeypatch.setattr(register, "register_hooks", lambda: None)
    register.register_sleep(app)
    service = _service()
    app.dependency_overrides[demo_api.demo_service] = lambda: service
    with TestClient(app) as mounted:
        assert mounted.get(f"{API}/nights").status_code == 200
        for asset in ("nights.js", "mind.js", "mind.css"):
            assert mounted.get(f"/sleep/demo/{asset}").status_code == 200, asset
        monkeypatch.delenv("ANJO_SLEEP_HOST")
        app.dependency_overrides.clear()
        for method, path in NEW_ROUTES:
            assert mounted.request(method, f"{API}{path}").status_code == 404, path
        assert mounted.get("/sleep/demo/mind.js").status_code == 404
    assert demo_api.PUBLIC_PREFIXES == ("/sleep/demo", "/api/sleep/demo")  # the carve-out is unchanged


def test_an_anonymous_visitor_reaches_the_new_routes_inside_the_carve_out(sleep_app: Any, client: Any) -> None:  # noqa: F811  (the imported fixture)
    _her_nights(FIVE, (_ckpt(1), _ckpt(2), _ckpt(4), _ckpt(5)))
    assert tuple(sleep_app.state.public_prefixes) == PUBLIC
    assert client.get(f"{API}/nights").status_code == 200
    headers = _voted(client)
    assert client.get(f"{API}/mind", headers=headers).status_code == 200
    assert client.post(f"{API}/toggles/{SCENE.id}", headers=headers).status_code == 200
    assert client.post(f"{API}/nights/n5/ask", json={"text": "The title?"}, headers=headers).json()["kind"] == "answer"
    for asset in ("/sleep/demo/nights.js", "/sleep/demo/mind.js", "/sleep/demo/mind.css"):
        assert client.get(asset, follow_redirects=False).status_code == 200, asset
