# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The live page's full step (Task 13a), its engine: the night slider's stops, the answers
pre-generated per checkpoint, the swap toggle's file, the held-out filter and the mind panel's
views. The routes and the app's reads are tested in ``tests/app/test_demo_mind.py``."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.demo.heldout import SHARED_RUN, HeldOutText, persona_texts
from anjo.sleep.demo.night_answers import LABEL, NightAnswersError, from_json, generate, load, to_json
from anjo.sleep.demo.panel import MindSources, PanelError, mind_view, night_record
from anjo.sleep.demo.record import chart_view, ledger_view, task16_view
from anjo.sleep.demo.slider import DEVELOPMENT, NightMark, Pinned, latest_stop, plan_stops
from anjo.sleep.demo.starters import Side
from anjo.sleep.demo.toggles import SwapError, swap_from_json
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.prereg import ledger
from anjo.sleep.probes.judge_brief import PERSONAS_DIR
from anjo.sleep.reports.results import Night, Stat
from anjo.sleep.types import CheckpointRef, Expectation, Target

START = date(2026, 10, 28)
RENDER = "gabc123/low"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
HELD = "She would rather the score stay silent until the last ferry crossing at dawn."


def _day(n: int) -> date:
    return START + timedelta(days=n - 1)


def _ckpt(n: int, rendering: str | None = RENDER) -> CheckpointRef:
    return CheckpointRef(f"p-film-n{n}", "film", _day(n), f"uri://n{n}", None, rendering)


def _nights(*statuses: str) -> list[NightMark]:
    return [NightMark(_day(n), status) for n, status in enumerate(statuses, start=1)]


# ── the night slider's stops ──────────────────────────────────────────────────


def test_slider_stop_falls_back_to_last_promoted() -> None:
    statuses = ("promoted", "promoted", "rejected", "promoted", "promoted", "failed", "nothing_new", "promoted")
    stops = plan_stops(_nights(*statuses), [_ckpt(n) for n in (1, 2, 4, 5, 8)], rendering=RENDER)
    assert [stop.key for stop in stops] == ["n0", "n3", "n7", "n8"]
    base, third, seventh, latest = stops
    assert (base.night, base.checkpoint, base.from_night, base.status, base.fallback) == (0, None, 0, None, False)
    assert third.checkpoint is not None and seventh.checkpoint is not None and latest.checkpoint is not None
    # A rejected or nothing_new night has no checkpoint of its own: its stop shows the last promoted one.
    assert (third.night, third.status, third.checkpoint.id, third.from_night, third.fallback) == (
        3, "rejected", "p-film-n2", 2, True)
    assert (seventh.night, seventh.status, seventh.checkpoint.id, seventh.from_night, seventh.fallback) == (
        7, "nothing_new", "p-film-n5", 5, True)
    assert (latest.night, latest.checkpoint.id, latest.fallback, latest.latest) == (8, "p-film-n8", False, True)
    assert latest_stop(stops) == latest and not any(stop.latest for stop in stops[:-1])
    assert all(stop.available for stop in stops) and DEVELOPMENT == "development"

    # Nothing promoted by night 3: the stop is the base model, and says so.
    early = plan_stops(_nights("nothing_new", "rejected", "rejected", "promoted"), [_ckpt(4)], rendering=RENDER)
    assert [(s.key, s.from_night, s.checkpoint.id if s.checkpoint else None, s.fallback) for s in early] == [
        ("n0", 0, None, False), ("n3", 0, None, True), ("n4", 4, "p-film-n4", False)]


def test_stops_are_the_nights_she_reached_with_real_numbers() -> None:
    three = plan_stops(_nights("promoted", "promoted", "promoted"), [_ckpt(n) for n in (1, 2, 3)], rendering=RENDER)
    assert [(s.key, s.latest) for s in three] == [("n0", False), ("n3", True)]  # night 3 is also her latest
    assert [s.key for s in plan_stops(_nights("promoted"), [_ckpt(1)], rendering=RENDER)] == ["n0", "n1"]
    none = plan_stops([], [], rendering=RENDER)
    assert [s.key for s in none] == ["n0"] and latest_stop(none) is None
    # A night run again is one night: its last run's status stands.
    rerun = [*_nights("promoted", "rejected"), NightMark(_day(2), "promoted")]
    stops = plan_stops(rerun, [_ckpt(1), _ckpt(2)], rendering=RENDER)
    assert [(s.key, s.status, s.fallback) for s in stops] == [("n0", None, False), ("n2", "promoted", False)]


def test_a_checkpoint_under_another_rendering_is_unavailable_never_replaced() -> None:
    history = [_ckpt(1), _ckpt(2, rendering="gother/high"), _ckpt(3, rendering=None)]
    stops = plan_stops(_nights("promoted", "promoted", "promoted"), history, rendering=RENDER)
    assert [(s.key, s.available) for s in stops] == [("n0", True), ("n3", False)]
    assert stops[1].checkpoint == history[2]  # its own checkpoint, unavailable: never night 1's instead


def test_a_pinned_registry_serves_one_checkpoint() -> None:
    pinned = Pinned("film", _ckpt(3))
    assert pinned.last_promoted("film") == _ckpt(3) and pinned.last_promoted("donor") is None
    assert Pinned("film", None).last_promoted("film") is None


# ── answers pre-generated per checkpoint ──────────────────────────────────────


def _answer(calls: list[tuple[str | None, str]]) -> Any:
    def answer(checkpoint: CheckpointRef | None, scene: words.Scene) -> Side:
        calls.append((checkpoint.id if checkpoint else None, scene.id))
        return Side(f"{checkpoint.id if checkpoint else 'base'}:{scene.id}", 400, 0.001)

    return answer


def test_night_answers_are_generated_per_checkpoint_kept_and_labeled(tmp_path: Path) -> None:
    calls: list[tuple[str | None, str]] = []
    first = generate(words.SCENES, [None, _ckpt(2), _ckpt(5)], _answer(calls), keep=None, person="film_student", now=NOW)
    assert first.label == LABEL == "pre-generated" and len(calls) == 3 * len(words.SCENES)
    scene = words.SCENES[0]
    assert first.answer(None, scene) == Side(f"base:{scene.id}", 400, 0.001)
    assert first.answer("p-film-n5", scene) is not None and first.answer("p-film-n9", scene) is None
    path = tmp_path / "night_answers.json"
    path.write_text(to_json(first))
    assert load(path) == first and load(tmp_path / "missing.json") is None

    # A new checkpoint is answered; the ones already answered are kept; a retired one is dropped.
    calls.clear()
    again = generate(words.SCENES, [None, _ckpt(5), _ckpt(8)], _answer(calls), keep=first, person="film_student", now=NOW)
    assert calls == [("p-film-n8", s.id) for s in words.SCENES]
    assert again.answer("p-film-n2", scene) is None and again.answer("p-film-n5", scene) == first.answer("p-film-n5", scene)
    changed = words.Scene(scene.id, scene.title, scene.prompt + " And the gulls?")
    assert again.answer(None, changed) is None  # an answer is to the words it was asked
    calls.clear()
    generate((changed, *words.SCENES[1:]), [None], _answer(calls), keep=again, person="film_student", now=NOW)
    assert calls == [(None, scene.id)]

    broken: Any
    for broken in ({**json.loads(to_json(first)), "label": "live"}, {"label": LABEL}, []):
        with pytest.raises(NightAnswersError):
            from_json(json.dumps(broken))


def test_the_swap_toggle_reads_only_a_labeled_file() -> None:
    scene = words.SCENES[1]
    raw = {"label": "pre-generated", "model": "Spark Open", "generated_at": "2026-11-01T00:00:00+00:00",
           "pairs": [{"scene": scene.id, "prompt_sha256": __import__("hashlib").sha256(scene.prompt.encode()).hexdigest(),
                      "with_state": {"text": "with", "prompt_tokens": 900, "cost_usd": None},
                      "bare": {"text": "bare", "prompt_tokens": 80, "cost_usd": None}}]}
    swaps = swap_from_json(json.dumps(raw))
    found = swaps.pair(scene)
    assert found is not None and (found.with_state.text, found.bare.text) == ("with", "bare")
    assert swaps.pair(words.SCENES[0]) is None
    with pytest.raises(SwapError):
        swap_from_json(json.dumps({**raw, "label": "live"}))


# ── the held-out filter ───────────────────────────────────────────────────────


def test_the_filter_withholds_any_text_carrying_held_out_words() -> None:
    held = HeldOutText([HELD, "   "])
    assert held.shown("Her open thread: the ferry edit") == "Her open thread: the ferry edit"
    for carrying in (
        f"A plan. {HELD}",  # the whole item
        "Maybe the score stay silent until the last ferry crossing",  # a run of SHARED_RUN words
        "Silent score until the final dawn ferry crossing, she would rather",  # most of its words, reworded
    ):
        assert held.carries(carrying) and held.shown(carrying) is None, carrying
    assert SHARED_RUN == 8 and held.shown(None) is None
    assert HeldOutText([]).shown("anything at all") == "anything at all"


def test_persona_texts_are_every_held_out_text_and_nothing_visible() -> None:
    person = load_person_file(PERSONAS_DIR / "film_student.json")
    texts = persona_texts(person)
    held = person.held_out
    for text in (*(p.text for p in held.preferences), *(w.text for w in held.new_work),
                 *(t.prompt for t in held.taste_ab), *(o for t in held.taste_ab for o in t.options)):
        assert text in texts
    assert not any(p.text in texts for p in person.visible.preferences)


# ── the mind panel ────────────────────────────────────────────────────────────


def _workspace(statement: str = "The plan assumes room for an 85 mm lens") -> it.Workspace:
    q = it.Question("q1", ("t-scene4",), "contradiction", ("e1", "e2"), "active", NOW - timedelta(days=2),
                    "Scene 4's long lens against the kitchen it is shot in")
    parked = it.Question("q2", ("t-sound",), "gap", ("e3",), "parked", NOW - timedelta(days=3), "The mix has no room booked")
    closed = it.Question("q3", ("t-sound",), "gap", ("e4",), "concluded", NOW - timedelta(days=3), "A closed matter")
    hypotheses = (it.Hypothesis("q1.h1", "q1", statement, 0.5), it.Hypothesis("q1.h2", "q1", "She shoots through the window", 0.3),
                  it.Hypothesis(it.catch_all_id("q1"), "q1", it.CATCH_ALL, 0.2))
    prediction = it.Prediction("q1.p1", "q1", "The courtyard is open on Thursday",
                               {"q1.h1": 0.2, "q1.h2": 0.9, "q1.else": 0.5}, NOW - timedelta(days=1), NOW + timedelta(days=6),
                               None, ())
    return it.Workspace(questions=(q, parked, closed), hypotheses=hypotheses, predictions=(prediction,))


def _target(id: str, label: str, *, kind: str = "thread", status: str = "open", confidence: float | None = 0.4) -> Target:
    return Target(id, "film", kind, label, NOW + timedelta(days=3) if kind == "commitment" else None,  # type: ignore[arg-type]
                  status, 3, 1, confidence, START)  # type: ignore[arg-type]


def _summary(n: int, status: str, *, action: str | None = "revisit_thread", considered: int = 2) -> dict[str, Any]:
    candidates = [{"want_id": "w-win", "kind": "understand", "target_id": "t-scene4", "action": action or "revisit_thread",
                   "strength": 0.6, "weight": 1.0, "status": "won" if action else "held_back",
                   "reason": None if action else "below_threshold"},
                  {"want_id": "w-lost", "kind": "follow_up", "target_id": "t-sound", "action": "prepare_commitment",
                   "strength": 0.3, "weight": 1.0, "status": "held_back", "reason": "lost_to_target"}][:considered]
    gates = {} if status == "nothing_new" else {
        "general": {"gate": "general", "passed": status == "promoted", "value": 0.4, "threshold": 1.0,
                    "reason": "SECRET-REASON names a case"},
        "safety": {"gate": "safety", "passed": True, "value": 0.0, "threshold": 0.0, "reason": None}}
    return {"night": _day(n).isoformat(), "status": status, "failed_at": None, "gates": gates,
            "checkpoint": None if status != "promoted" else {"id": f"p-film-n{n}"},
            "choices": [{"opportunity": "away", "action": action, "target_id": "t-scene4" if action else None,
                         "question_id": None, "considered": candidates}]}


def _steps(outcome: str | None = "ran", note: str = "no_reason") -> list[dict[str, Any]]:
    return [{"step": "choose", "status": "ok", "receipt": {"outcome": outcome}}, {"step": "note", "status": "ok", "outcome": note}]


def _sources(**changes: Any) -> MindSources:
    nights = (night_record(_summary(1, "promoted"), _steps()), night_record(_summary(2, "nothing_new", action=None), _steps(None)),
              night_record(_summary(3, "rejected"), _steps("ran", "held")))
    base = {"nights": nights, "workspace": _workspace(),
            "targets": (_target("t-scene4", "Scene 4 in the kitchen"), _target("t-sound", "Book the mix", kind="commitment"),
                        _target("t-old", "An old thread", status="closed")),
            "expectations": (Expectation("x1", "t-scene4", NOW - timedelta(days=9), NOW - timedelta(days=2), 0.62),
                             Expectation("x2", "t-sound", NOW - timedelta(days=1), NOW + timedelta(days=6), 0.4),
                             Expectation("x3", "t-old", NOW - timedelta(days=9), NOW - timedelta(days=2), 0.3, "censored")),
            "returns": {"t-scene4": (NOW - timedelta(days=5),)}}
    return MindSources(**{**base, **changes})


def _text(item: dict[str, Any]) -> str | None:
    assert set(item) == {"text", "withheld"}
    text: str | None = item["text"]
    return text


def test_mind_view_shows_her_state_plainly() -> None:
    view = mind_view(_sources(), HeldOutText([]))
    assert view["as_of"] == datetime.combine(_day(4), datetime.min.time(), timezone.utc).isoformat()
    questions = view["questions"]["items"]
    assert [q["id"] for q in questions] == ["q1", "q2"]  # open, active and parked; never a closed one
    first = questions[0]
    assert _text(first["subject"]) == "Scene 4's long lens against the kitchen it is shot in"
    assert [(_text(h["statement"]), h["credence"], h["catch_all"]) for h in first["hypotheses"]][-1] == (
        it.CATCH_ALL, 0.2, True)
    assert first["predictions"][0]["given"] == [0.2, 0.9, 0.5] and first["predictions"][0]["outcome"] is None
    targets = view["targets"]["items"]
    assert [(t["id"], t["kind"], t["confidence"]) for t in targets] == [("t-sound", "commitment", 0.4), ("t-scene4", "thread", 0.4)]
    wants = view["wants"]
    assert wants["night"] == 3 and [(w["kind"], w["status"], w["reason"]) for w in wants["items"]] == [
        ("understand", "won", None), ("follow_up", "held_back", "lost_to_target")]
    choices = view["choices"]["items"]
    assert [(c["night"], c["action"], c["outcome"], c["note"]) for c in choices] == [
        (3, "revisit_thread", "ran", "held"), (2, None, None, "no_reason"), (1, "revisit_thread", "ran", "no_reason")]
    assert _text(choices[0]["target"]) == "Scene 4 in the kitchen" and len(choices[0]["beat"]) == 1
    assert choices[1]["beat"][0]["reason"] == "below_threshold"  # a night she chose nothing, and what held back
    outcomes = {e["id"]: e["outcome"] for e in view["expectations"]["items"]}
    assert outcomes == {"x1": "returned", "x2": "open", "x3": "censored"}
    gates = view["gates"]["items"]
    assert [(g["night"], [x["gate"] for x in g["gates"]]) for g in gates] == [
        (3, ["general", "safety"]), (2, []), (1, ["general", "safety"])]
    assert "SECRET-REASON" not in json.dumps(view)  # a gate's reason can name a safety case


def test_mind_view_withholds_held_out_text_and_marks_what_cannot_be_read() -> None:
    view = mind_view(_sources(workspace=_workspace(f"Perhaps: {HELD}")), HeldOutText([HELD]))
    shown = json.dumps(view)
    assert "silent until the last ferry" not in shown
    statements = view["questions"]["items"][0]["hypotheses"]
    assert statements[0]["statement"] == {"text": None, "withheld": True}
    unread = mind_view(MindSources(), HeldOutText([]))
    assert {key: unread[key]["state"] for key in ("questions", "targets", "wants", "choices", "expectations", "gates")} == {
        key: "unavailable" for key in ("questions", "targets", "wants", "choices", "expectations", "gates")}
    empty = mind_view(MindSources(nights=(), workspace=it.Workspace(), targets=(), expectations=()), HeldOutText([]))
    assert empty["questions"]["state"] == "none" and empty["choices"]["state"] == "none" and empty["as_of"] is None


def test_a_night_record_is_ids_codes_and_numbers_only() -> None:
    record = night_record(_summary(3, "rejected"), _steps("declined", "note_declined"))
    assert (record.night, record.status, record.outcome, record.note) == (_day(3), "rejected", "declined", "note_declined")
    assert [(g.gate, g.passed) for g in record.gates] == [("general", False), ("safety", True)]
    with pytest.raises(PanelError):
        night_record({"status": "promoted"}, [])


# ── the chart, the Task 16 result and the ledger ──────────────────────────────


def _night(n: int, label: str, score: float) -> Night:
    return Night("film", _day(n).isoformat(), label, "promoted", 0.0, {},
                 {"1": {"knowledge": Stat(0.5)}, "3": {"knowledge": Stat(score), "overall": Stat(score)}}, {})


def test_chart_labels_development_and_final_apart() -> None:
    view = chart_view([_night(1, "development", 0.52), _night(2, "development", 0.58)], [], name="Mara")
    assert view["state"] == "ok" and view["final"] is None and view["dimensions"] == ["knowledge"]
    life = view["development"]["lives"][0]
    assert life["name"] == "Mara" and [(n["night"], n["scores"]["3"]["knowledge"]) for n in life["nights"]] == [(1, 0.52), (2, 0.58)]
    final = chart_view([], [[_night(1, "final", 0.61)], [_night(1, "development", 0.4)]], name="Mara")
    assert final["development"] is None and len(final["final"]["lives"]) == 1  # a life with no final night isn't one
    assert chart_view([], [], name="Mara")["state"] == "none"


def test_task16_result_is_shown_line_by_line_or_not_yet() -> None:
    assert task16_view(None, HeldOutText([]))["state"] == "none"
    text = f"# MIND\n\n## Held-out lives\n\n- Pooled: On 40% of nights her state studied something relevant.\n- {HELD}\n"
    view = task16_view(text, HeldOutText([HELD]))
    assert _text(view["headline"]) == "Pooled: On 40% of nights her state studied something relevant."
    assert {"text": None, "withheld": True} in view["lines"] and "ferry crossing" not in json.dumps(view)


def test_the_ledger_shows_its_hashes_and_whether_its_chain_holds(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    for n, note in enumerate(("sealed_probes film_student.jsonl", HELD)):
        ledger.register("sealed_probes", f"{n}" * 64, note, path=path, now=NOW)
    entries = ledger.read_entries(path)
    view = ledger_view(entries, pushed=entries[-1].entry_hash, held=HeldOutText([HELD]))
    assert (view["state"], view["verified"], view["count"], view["in_sync"]) == ("ok", True, 2, True)
    assert [e["seq"] for e in view["entries"]] == [1, 0] and view["entries"][1]["sha256"] == "0" * 64
    assert view["entries"][0]["note"] == {"text": None, "withheld": True}
    tampered = [entries[0], ledger.LedgerEntry(**{**entries[1].__dict__, "sha256": "f" * 64})]
    assert ledger_view(tampered, pushed=None, held=HeldOutText([]))["verified"] is False
    assert ledger_view([], pushed=None, held=HeldOutText([]))["state"] == "none"


def test_every_word_the_new_scripts_read_is_in_the_one_copy_module() -> None:
    import re

    from anjo.sleep.demo.page import WEB_DIR

    for name in ("nights.js", "mind.js"):
        script = (WEB_DIR / name).read_text()
        read = set(re.findall(r"copy\.([a-zA-Z0-9_]+)", script))
        assert read and read <= set(words.COPY), (name, sorted(read - set(words.COPY)))
    for prefix, codes in (("night_status", ("promoted", "rejected", "nothing_new", "failed")),
                          ("reason", ("unknown_priority_class", "action_not_offered", "candidate_limit",
                                      "below_threshold", "budget", "lost_to_target", "note_declined")),
                          ("chart_arm", ("1", "2C", "2F", "3", "3x", "4")), ("toggle_arm", ("1", "3", "swap", "bare")),
                          ("gate", ("general", "collapse", "safety"))):
        assert all(f"{prefix}_{code}" in words.COPY for code in codes), prefix
