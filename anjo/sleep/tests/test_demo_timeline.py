# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""How she got here (``demo/timeline.py``) and the return moment (``demo/return_view.py``): the
records of a quiet window turned into ordered nodes, and her own message read from them. Fakes
only, no network; the page's script is run under node against a fake DOM."""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.page import WEB_DIR
from anjo.sleep.demo.return_view import HeldNote, ReturnSources, return_view
from anjo.sleep.demo.timeline import DecisionRecord, TimelineSources, timeline_view
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.tests.inquiry_fakes import Ws, ev, night, notice_op, ref
from anjo.sleep.types import Result

QUESTION = "Scene 4's long lens against the kitchen it is shot in"
SEARCH = "film stock reciprocity failure at night"
CLAIM = "She names the courtyard instead of the kitchen"


def _window() -> tuple[Ws, list[Any], str]:
    """A quiet window: the person's items, then a question, its hypotheses, a web search, a
    prediction resolved right, and a night's decision."""
    item = ev("e-1", "The kitchen can't take an 85 mm lens.", n=1)
    seen = ev("e-2", "The courtyard is locked on Thursday.", n=3)
    resolver = ev("e-3", "The courtyard door was open on Friday.", n=3)
    w = Ws()
    w.start(4)
    w.arrive(item, seen, at=night(4))
    w.ok(notice_op([item.id, seen.id], subject=QUESTION), at=night(4))
    question = w.ws.questions[0].id
    w.ok(it.HypothesizeOp(question, (it.Alternative("The plan assumes room for an 85 mm lens", 0.5),
                                    it.Alternative("She will shoot through the courtyard window", 0.3)), 0.2),
         at=night(4))
    w.ok(it.InvestigateOp(question, "world_query", SEARCH), at=night(4))
    w.ok(it.Answered(question, SEARCH, "Reciprocity failure starts around one second."), at=night(4))
    ids = [h.id for h in w.ws.hypotheses_of(question)]
    given = {ids[0]: 0.2, ids[1]: 0.8, ids[2]: 0.5}
    w.ok(it.PredictOp(question, CLAIM, given, 2), at=night(4))
    prediction = w.ws.predictions[0].id
    w.ok(it.Resolved(prediction, "true", (ref(resolver),)), at=night(5))
    return w, [item, seen, resolver], question


def _view(w: Ws, evidence: list[Any], **changes: Any) -> dict[str, Any]:
    as_of = datetime.combine(night(5).date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    sources = TimelineSources(events=tuple(w.events), evidence=tuple(evidence), results=(), as_of=as_of,
                              decisions=(DecisionRecord(night(5).date(), night(5), "revisit_thread", "t-1", "ran",
                                                        "promoted", 0.7),), **changes)
    return timeline_view(sources, HeldOutText([]))


def test_the_quiet_window_is_ordered_and_cites_every_logged_record() -> None:
    w, evidence, _question = _window()
    view = _view(w, evidence)
    assert view["state"] == "ok" and view["quiet_days"] == 3
    assert view["counts"] == {"question": 1, "credence": 2, "prediction": 1, "resolution": 1, "search": 1,
                              "decision": 1}
    nodes = view["items"]
    assert [node["at"] for node in nodes] == sorted(node["at"] for node in nodes)  # the log's order, kept
    assert all(node["record"]["event"] for node in nodes)
    question = next(node for node in nodes if node["kind"] == "question")
    assert (question["origin"], question["code"], question["status"]) == ("her", "contradiction", "open")
    assert question["detail"]["subject"] == {"text": QUESTION, "withheld": False}
    assert [cite["id"] for cite in question["cites"]] == ["e-1", "e-2"]
    search = next(node for node in nodes if node["kind"] == "search")
    assert search["tool"] == "world_query" and search["detail"]["query"]["text"] == SEARCH
    assert search["detail"]["answer"]["text"] == "Reciprocity failure starts around one second."
    resolution = next(node for node in nodes if node["kind"] == "resolution")
    assert (resolution["verdict"], resolution["outcome"]) == ("right", "true")
    assert [cite["id"] for cite in resolution["seen"]] == ["e-3"]
    credence = [node for node in nodes if node["kind"] == "credence"]
    assert credence and any(move["before"] is not None for node in credence for move in node["moves"])
    decision = next(node for node in nodes if node["kind"] == "decision")
    assert decision["confidence"] == 0.7 and decision["record"]["event"] == "night-2026-10-06-choose"


def test_a_node_kind_this_build_does_not_render_becomes_no_node() -> None:
    w, evidence, _question = _window()
    view = _view(w, evidence)
    assert all(node["kind"] in {"question", "credence", "prediction", "resolution", "search", "decision"}
               for node in view["items"])
    assert "unknown" not in json.dumps(view)


def test_the_filter_withholds_her_words_and_the_page_learns_so() -> None:
    w, evidence, _question = _window()
    as_of = datetime.combine(night(5).date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    view = timeline_view(TimelineSources(events=tuple(w.events), evidence=tuple(evidence), as_of=as_of),
                         HeldOutText([QUESTION]))
    question = next(node for node in view["items"] if node["kind"] == "question")
    assert question["detail"]["subject"] == {"text": None, "withheld": True}


def test_no_events_unavailable_and_no_quiet_days_none() -> None:
    assert timeline_view(TimelineSources(), HeldOutText([])) == {"state": "unavailable"}
    as_of = datetime(2026, 10, 7, tzinfo=timezone.utc)
    assert timeline_view(TimelineSources(events=(), evidence=(), as_of=as_of), HeldOutText([])) == {
        "state": "none", "quiet_days": 0, "from": None, "to": as_of.isoformat(), "items": [], "counts": {}}


# ── the return moment ────────────────────────────────────────────────────────


def _last() -> datetime:
    return datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def test_a_held_note_speaks_first_with_its_night_and_quiet_days() -> None:
    note = HeldNote("I looked into the lens and left the rest until you're back.", id="n-1", night=_last().date())
    view = return_view(ReturnSources(note=note, evidence=(ev("e-0", "I am rebuilding the sorting line.", n=0),),
                                     as_of=_last() + timedelta(days=3)), HeldOutText([]))
    assert view["state"] == "ok" and view["kind"] == "held_note"
    assert view["text"] == note.text and view["quiet_days"] == 3
    assert view["record"] == {"id": "n-1", "kind": "night_note", "night": "2026-10-01", "at": None}


def test_a_withheld_note_is_said_to_be_held_back_never_guessed() -> None:
    note = HeldNote(QUESTION)
    view = return_view(ReturnSources(note=note, events=(), as_of=_last()), HeldOutText([QUESTION]))
    assert view == {"state": "withheld", "quiet_days": 0}


def test_without_a_note_the_share_mind_raised_is_her_message() -> None:
    w = Ws()
    w.start(1)
    item = ev("e-1", "The kitchen can't take an 85 mm lens.", n=1)
    other = ev("e-2", "The courtyard is locked on Thursday.", n=3)
    late = ev("e-3", "The courtyard door was open on Friday.", n=4)
    w.arrive(item, other, late, at=night(1))
    w.ok(notice_op([item.id, other.id]), at=night(1))
    question = w.ws.questions[0].id
    w.ok(it.HypothesizeOp(question, (it.Alternative("The plan assumes room for an 85 mm lens", 0.9),
                                    it.Alternative("She will shoot through the courtyard window", 0.1)), 0.0),
         at=night(1))
    w.found(question, late)
    ids = [h.id for h in w.ws.hypotheses_of(question)]
    w.ok(it.AssessOp(question, late.id, {ids[0]: 4.0, ids[1]: 1.0, ids[2]: 1.0}), at=night(1))
    leader = kernel.leader(w.ws.hypotheses_of(question))
    finding = "The courtyard window is the only place with room for the lens."
    w.ok(it.ConcludeOp(question, leader.id, "resolution", finding, (item.id,)), at=night(1))
    found = w.ws.findings[0]
    result = Result(f"{found.id}.{found.kind}", "t-1", None, 1, night(2), False)
    view = return_view(ReturnSources(events=tuple(w.events), results=(result,), evidence=(item,),
                                     as_of=night(2)), HeldOutText([]))
    assert view["state"] == "ok" and view["kind"] == "share" and view["text"] == finding
    assert view["record"]["id"] == found.id and view["record"]["event"] == "seq-6"
    assert view["sources"] == [{"id": "e-1", "url": "message:e-1"}]


def test_nothing_waiting_is_none_and_an_unreadable_source_unavailable() -> None:
    assert return_view(ReturnSources(events=(), results=(), as_of=_last()), HeldOutText([])) == {
        "state": "none", "quiet_days": 0}
    assert return_view(ReturnSources(), HeldOutText([])) == {"state": "unavailable", "quiet_days": 0}


# ── the page's script ────────────────────────────────────────────────────────

NODE = shutil.which("node")


def test_timeline_js_reads_only_words_in_the_copy_and_draws_with_no_html() -> None:
    script = (WEB_DIR / "timeline.js").read_text()
    read = set(__import__("re").findall(r"copy\.([a-zA-Z0-9_]+)", script))
    assert read and read <= set(words.COPY), sorted(read - set(words.COPY))
    assert "innerHTML" not in script and "eval(" not in script and "localStorage" not in script
    assert "http://" not in script and "https://" not in script


def _payload() -> dict[str, Any]:
    """Two days, handed out of order: the script must order by the log, then group by day."""
    base = {"record": {"event": "", "seq": None, "night": None}, "text": ""}
    return {
        "state": "ok", "quiet_days": 2, "from": "2026-10-01T12:00:00+00:00",
        "to": "2026-10-03T00:00:00+00:00",
        "items": [
            {**base, "kind": "decision", "at": "2026-10-02T00:00:00+00:00", "day": "2026-10-02", "seq": None,
             "record": {"event": "night-2026-10-02-choose", "seq": None, "night": "2026-10-02"},
             "action": "revisit_thread", "outcome": "ran", "status": "promoted", "confidence": 0.6,
             "target": {"text": "Scene 4", "withheld": False}},
            {**base, "kind": "search", "at": "2026-10-01T22:00:00+00:00", "day": "2026-10-01", "seq": 5,
             "record": {"event": "seq-5", "seq": 5, "night": "2026-10-01"}, "tool": "world_query",
             "detail": {"query": {"text": SEARCH, "withheld": False}, "answer": {"text": "It falls off.", "withheld": False},
                        "items": []}},
            {**base, "kind": "question", "at": "2026-10-01T21:00:00+00:00", "day": "2026-10-01", "seq": 3,
             "record": {"event": "seq-3", "seq": 3, "night": "2026-10-01"}, "code": "gap", "status": "open",
             "origin": "person", "detail": {"subject": {"text": QUESTION, "withheld": False}}, "cites": []},
            {**base, "kind": "mystery", "at": "2026-10-01T23:00:00+00:00", "day": "2026-10-01", "seq": 9,
             "record": {"event": "seq-9", "seq": 9, "night": None}},
        ],
        "counts": {"question": 1, "credence": 0, "prediction": 0, "resolution": 0, "search": 1, "decision": 1},
    }


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_timeline_js_orders_nodes_groups_them_by_day_and_links_each_event(tmp_path: Path) -> None:
    harness = tmp_path / "harness.js"
    harness.write_text((Path(__file__).parent / "timeline_js_harness.js").read_text())
    run = subprocess.run([str(NODE), str(harness), str(WEB_DIR / "timeline.js"), json.dumps(dict(words.COPY)),
                          json.dumps(_payload())], capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 0, run.stderr
    drawn = json.loads(run.stdout)
    assert [group["day"] for group in drawn["groups"]] == ["2026-10-01", "2026-10-02"]
    assert [kind for kind in drawn["groups"][0]["kinds"]] == ["question", "search"]  # seq 3 before 5; the unknown kind is skipped
    assert drawn["groups"][0]["events"] == ["seq-3", "seq-5"]
    assert drawn["groups"][1]["events"] == ["night-2026-10-02-choose"]
    assert drawn["links"] == ["#tl-question-0-seq-3", "#tl-search-1-seq-5", "#tl-decision-0-night-2026-10-02-choose"]
    assert SEARCH in drawn["text"] and QUESTION in drawn["text"]
    assert words.COPY["timeline_plain_question"].split("{")[0].strip() in drawn["text"]
