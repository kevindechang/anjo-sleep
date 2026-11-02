"""Her own line of thought, for the person (thought.py): pure views over the workspace log."""
import json
from dataclasses import dataclass, replace
from datetime import timedelta, timezone

import pytest

from anjo.sleep.mind import thought
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.tests.inquiry_fakes import Ws, ev, hypothesize, night, ref

E = tuple(ev(f"e{n}", f"item {n}", n=n, target="t1", model=None) for n in range(8))
SHOOT, EDIT = "Will the Friday shoot happen?", "Is the edit still due Friday?"
COURTYARD, CREW, KITCHEN = "The courtyard opens by Thursday", "The crew cancels", "The kitchen is free"
CUT, WRAP = "She sends the cut on Friday", "They wrap by Sunday"


@dataclass
class Life:
    w: Ws
    p: dict[str, float]  # her odds when she made each bet, by claim
    q1: str
    q2: str

    @property
    def events(self) -> tuple[it.WorkspaceEvent, ...]:
        return tuple(self.w.events)


def _given(w: Ws, qid: str, *ps: float) -> dict[str, float]:
    return dict(zip(w.ids(qid), ps, strict=True))


def _bet(w: Ws, p: dict[str, float], qid: str, claim: str, ps: tuple[float, ...], days: int, n: int) -> None:
    given = _given(w, qid, *ps)
    w.ok(it.PredictOp(qid, claim, given, days), at=night(n))
    p[claim] = sum(h.credence * given[h.id] for h in w.ws.hypotheses_of(qid))


def _question(w: Ws, subject: str) -> str:
    return next(q.id for q in w.ws.questions if q.subject == subject)


def life() -> Life:
    """Two questions over six nights: bets settled right, wrong and at exactly .5, one censored,
    one still open, a resolution finding and a close call."""
    w, p = Ws(), {}
    w.start(1)
    w.arrive(E[0], E[1], at=night(1))
    w.ok(it.NoticeOp("contradiction", SHOOT, ("t1",), ("e0", "e1")), at=night(1))
    q1 = _question(w, SHOOT)
    w.ok(hypothesize(q1, 0.5, 0.3, catch_all=0.2), at=night(1))
    _bet(w, p, q1, COURTYARD, (0.9, 0.2, 0.5), 3, 1)  # .61
    _bet(w, p, q1, CREW, (0.95, 0.7, 0.6), 3, 1)  # .805
    _bet(w, p, q1, KITCHEN, (0.6, 0.3, 0.55), 3, 1)  # exactly .5
    w.start(2)
    w.arrive(E[2], at=night(2))
    w.ok(it.NoticeOp("gap", EDIT, ("t1",), ("e2",)), at=night(2))
    q2 = _question(w, EDIT)
    w.ok(hypothesize(q2, 0.42, 0.40, catch_all=0.18), at=night(2))
    _bet(w, p, q2, CUT, (0.8, 0.3, 0.5), 9, 2)
    w.start(3)
    w.arrive(E[3], at=night(3))
    by_claim = {x.claim: x.id for x in w.ws.predictions}
    for claim, outcome in ((COURTYARD, "true"), (CREW, "false"), (KITCHEN, "true")):
        w.ok(it.Resolved(by_claim[claim], outcome, (ref(E[3]),)), at=night(3))
    w.arrive(E[4], at=night(3))
    w.ok(it.Retrieved(q1, "evidence_search", "courtyard", (ref(E[4]),)), at=night(3))
    w.ok(it.AssessOp(q1, "e4", _given(w, q1, 4.0, 0.25, 0.25)), at=night(3))
    w.ok(it.ConcludeOp(q1, w.ids(q1)[0], "resolution", "The shoot goes ahead in the courtyard", ("e4",)),
         at=night(3))
    w.start(4)
    _bet(w, p, q2, WRAP, (0.7, 0.2, 0.5), 1, 4)
    w.start(6)
    w.ok(it.Resolved({x.claim: x.id for x in w.ws.predictions}[WRAP], "censored", ()), at=night(6))
    return Life(w, p, q1, q2)


def view(life_: Life, marks=(), **kw):
    return thought.thought_view(life_.events, marks, now=kw.pop("now", night(6)), **kw)


def test_hunches_are_her_open_bets_with_her_implied_odds_soonest_first() -> None:
    L = life()
    v = view(L)
    assert [h["claim"]["text"] for h in v["hunches"]] == [CUT]
    h = v["hunches"][0]
    pred = next(x for x in L.w.ws.predictions if x.claim == CUT)
    assert h["p"] == pytest.approx(sum(x.credence * pred.p_true_given[x.id] for x in L.w.ws.hypotheses_of(L.q2)))
    assert h["about"]["text"] == EDIT
    assert h["due"] == (night(2) + timedelta(days=9)).date().isoformat()
    assert h["due_in_days"] == 5
    assert h["odds"] == thought.odds(h["p"]) == "about 5 in 10"
    assert h["disputed"] is False
    assert h["id"] == pred.id and h["plain"] and h["exact"] == CUT
    assert v["credence_note"] == thought.CREDENCE_NOTE


def test_settled_bets_are_scored_right_wrong_and_no_call_with_their_odds_at_the_bet() -> None:
    L = life()
    got = {s["claim"]["text"]: s for s in view(L)["settled"]}
    assert set(got) == {COURTYARD, CREW, KITCHEN}
    assert got[COURTYARD]["p_at_bet"] == pytest.approx(L.p[COURTYARD]) == pytest.approx(0.61)
    assert got[COURTYARD]["verdict"] == "right"  # .61, came true
    assert got[CREW]["verdict"] == "wrong"  # .805, did not happen
    assert got[KITCHEN]["verdict"] == "no_call"  # exactly .5
    assert all(s["settled_at"] == night(3).isoformat() and s["plain"] for s in got.values())


def test_scorecard_counts_censored_apart_and_says_when_there_are_too_few_bets() -> None:
    L = life()
    card = view(L)["scorecard"]
    assert (card["scored"], card["right"], card["ran_out_of_time"]) == (3, 1, 1)
    assert card["hit_rate"] == pytest.approx(1 / 3)
    expected = ((L.p[COURTYARD] - 1) ** 2 + L.p[CREW] ** 2 + (L.p[KITCHEN] - 1) ** 2) / 3
    assert card["brier"] == pytest.approx(expected)
    assert card["enough"] is False
    assert sum(b["n"] for b in card["bins"]) == 3
    assert sum(b["came_true"] for b in card["bins"]) == 2


def test_an_empty_log_shows_nothing_and_scores_nothing() -> None:
    v = thought.thought_view((), (), now=night(1))
    assert v["hunches"] == [] and v["settled"] == [] and v["question"] is None and v["morning"] is None
    assert v["scorecard"]["scored"] == 0 and v["scorecard"]["hit_rate"] is None and v["scorecard"]["brier"] is None


def test_bets_settled_before_a_wipe_still_count() -> None:
    L = life()
    L.w.start(7, wipe=True)
    v = view(L, now=night(7))
    assert v["scorecard"]["scored"] == 3
    assert v["hunches"] == []


def test_the_one_question_is_her_close_call_with_something_else_last() -> None:
    L = life()
    q = view(L)["question"]
    assert q["question_id"] == q["id"] == L.q2 and q["why"] == "close_call"
    assert q["ask"]["text"] == EDIT and q["plain"]
    assert [c["id"] for c in q["choices"]][-1] == it.catch_all_id(L.q2)
    assert q["choices"][-1]["text"]["text"] == thought.SOMETHING_ELSE
    credences = [c["credence"] for c in q["choices"][:-1]]
    assert credences == sorted(credences, reverse=True)


def test_the_morning_note_covers_only_what_happened_since_the_last_visit() -> None:
    L = life()
    m = view(L, since=night(2) + timedelta(hours=1))["morning"]
    assert m["empty"] is False and m["plain"]
    assert len(m["settled"]) == 3 and m["new_hunches"] == []
    assert [f["kind"] for f in m["findings"]] == ["resolution"]
    assert all(f["id"] and f["plain"] for f in m["findings"])
    shift = next(s for s in m["shifts"] if s["question_id"] == L.q1)
    assert shift["from"] == pytest.approx(0.5) and shift["to"] > 0.65 and shift["id"] and shift["plain"]
    later = view(L, since=night(5))["morning"]
    assert later["settled"] == [] and later["findings"] == [] and later["shifts"] == []
    assert later["empty"] is True


def test_dispute_flags_a_hunch_and_hide_removes_an_item_everywhere() -> None:
    L = life()
    open_id = view(L)["hunches"][0]["id"]
    v = view(L, (thought.dispute(L.events, open_id, at=night(6), note="It moved"),))
    assert v["hunches"][0]["disputed"] is True
    hidden = view(L, (thought.hide(L.events, L.q2, at=night(6)),), since=night(1) + timedelta(hours=1))
    assert hidden["hunches"] == [] and hidden["question"] is None
    assert all(s["question_id"] != L.q2 for s in hidden["settled"])
    assert all(f["question_id"] != L.q2 for f in hidden["morning"]["findings"])
    assert hidden["hidden"] == 1
    assert hidden["scorecard"] == view(L)["scorecard"]  # hiding never rewrites her record


def test_marks_refuse_unknown_ids_and_bad_records() -> None:
    L = life()
    with pytest.raises(ValueError):
        thought.dispute(L.events, "nope", at=night(6))
    good = thought.hide(L.events, L.q1, at=night(6))
    assert thought.Mark.from_json(json.loads(json.dumps(good.to_json()))) == good
    for bad in ({"kind": "forget", "item_id": L.q1, "at": night(6).isoformat()},
                {"kind": "hide", "item_id": "", "at": night(6).isoformat()},
                {"kind": "hide", "item_id": L.q1, "at": "yesterday"},
                {"kind": "hide", "item_id": L.q1, "at": "2026-10-07T23:00:00"},
                {"kind": "dispute", "item_id": L.q1, "at": night(6).isoformat(), "note": 3},
                ["hide", L.q1]):
        with pytest.raises(ValueError):
            thought.Mark.from_json(bad)


def test_every_free_text_goes_through_the_filter() -> None:
    L = life()

    def withhold(text: str) -> dict[str, object]:
        return {"text": "", "withheld": True}

    blob = json.dumps(view(L, since=night(1) + timedelta(hours=1), text=withhold))
    for phrase in ("Friday", "courtyard", "crew", "kitchen", "edit", "Sunday", "shoot"):
        assert phrase not in blob


@pytest.mark.parametrize("field", ["before", "after"])
def test_a_tampered_log_is_refused(field: str) -> None:
    L = life()
    events = list(L.events)
    events[3] = replace(events[3], **{field: "0" * 64})
    with pytest.raises(ValueError):
        thought.thought_view(tuple(events), (), now=night(6))


def test_steps_after_the_night_clock_are_left_out() -> None:
    from anjo.sleep.mind import thought_demo as demo

    events, now = demo.demo_events(), demo._night(4)
    upto = tuple(e for e in events if e.step.at <= now)
    assert len(upto) < len(events)
    assert thought.thought_view(events, (), now=now, since=demo.SINCE) == thought.thought_view(
        upto, (), now=now, since=demo.SINCE)


def test_a_new_hunch_is_listed_even_past_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    L = life()
    monkeypatch.setattr(thought, "MAX_HUNCHES", 0)
    v = view(L, now=night(4), since=night(3) + timedelta(hours=1))
    wrap = next(x.id for x in L.w.ws.predictions if x.claim == WRAP)
    assert v["morning"]["new_hunches"] == [wrap]
    assert [h["id"] for h in v["hunches"]] == [wrap]


def test_due_dates_are_read_on_the_night_clock_s_own_zone() -> None:
    L = life()
    horizon = night(2) + timedelta(days=9)  # CUT settles at 23:00 UTC
    now = (horizon - timedelta(hours=7)).astimezone(timezone(timedelta(hours=8)))  # already the next day there
    h = view(L, now=now)["hunches"][0]
    assert h["due_in_days"] == 0
    assert h["due"] == now.date().isoformat()


def test_a_finding_revised_overnight_shows_once_as_its_latest_words_with_a_new_id() -> None:
    L = life()
    first = view(L, since=night(2) + timedelta(hours=1))["morning"]["findings"]
    L.w.start(7)
    h = L.w.ids(L.q2)[0]
    for said in ("She sends a rough cut first", "She sends the cut late"):
        L.w.ok(it.ConcludeOp(L.q2, h, "suspicion", said, ("e2",)), at=night(7))
    found = view(L, now=night(7), since=night(6))["morning"]["findings"]
    assert [f["statement"]["text"] for f in found] == ["She sends the cut late"]
    assert found[0]["id"] not in {f["id"] for f in first}


def test_the_view_is_plain_json() -> None:
    v = view(life(), since=night(1) + timedelta(hours=1))
    assert json.loads(json.dumps(v)) == v
    assert v["version"] == thought.VERSION


def test_odds_words_cover_the_ends() -> None:
    assert [thought.odds(p) for p in (0.02, 0.5, 0.71, 0.97)] == [
        "less than 1 in 10", "about 5 in 10", "about 7 in 10", "more than 9 in 10"]


def test_the_demo_life_fills_every_section() -> None:
    from anjo.sleep.mind import thought_demo

    v = thought_demo.demo_view()
    assert v["hunches"] and v["settled"] and v["question"] and v["morning"] and not v["morning"]["empty"]
    assert v["scorecard"]["scored"] >= 3
    assert json.loads(json.dumps(v)) == v
