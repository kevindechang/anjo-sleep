# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""Her own line of thought for the judge it is about, on the real engine (``mind.thought``) over the
synthetic demo life (``mind.thought_demo``) written to the judge's own workspace log: the view on the
night clock, a visit that survives the page's polling, dispute and hide stored as the engine's own
marks in the person's encrypted file (a dispute's note also kept as their own words in their
evidence), 404 for an id not in her log, and 500 for a log or marks file that can't be read whole or
doesn't replay. No model call."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import pytest

from anjo.sleep.app import judge_thought
from anjo.sleep.mind import thought, thought_demo
from anjo.sleep.tests.app.test_judge_api import _bearer, _start, api, talker  # noqa: F401  (fixtures)

DEMO_NOW = thought_demo.NOW  # the demo's last night, 23:00 UTC
WALL = DEMO_NOW + timedelta(days=1)


def _judge(api: Any, events: Any = None) -> tuple[str, str]:  # noqa: F811
    from anjo.sleep.app.storage import workspace_log

    body = _start(api).json()
    workspace_log(body["user_id"]).extend(thought_demo.demo_events() if events is None else events)
    return body["token"], body["user_id"]


def _ids(view: dict[str, Any]) -> list[str]:
    return [h["id"] for h in view["hunches"]]


def test_the_view_is_the_engines_over_their_own_log(api: Any) -> None:  # noqa: F811
    token, _user_id = _judge(api)
    seen = api.get("/api/sleep/me/thought", headers=_bearer(token))
    assert seen.status_code == 200
    body = seen.json()
    expected = thought.thought_view(thought_demo.demo_events(), (), now=datetime.fromisoformat(body["as_of"]))
    polled = {key: body.pop(key) for key in ("revision", "live_pass")}
    assert body == expected and len(body["hunches"]) == 2 and body["question"] is not None
    assert polled == {"revision": len(thought_demo.demo_events()), "live_pass": "idle"}  # no live pass run yet
    assert body["morning"] is None  # a first visit: nothing to catch up on


def test_disputes_and_hides_are_the_engines_marks_kept_encrypted(api: Any) -> None:  # noqa: F811
    from anjo.sleep.app.storage import evidence_store, sleep_dir

    token, user_id = _judge(api)
    first = api.get("/api/sleep/me/thought", headers=_bearer(token)).json()
    hunch, question = first["hunches"][0], first["question"]["question_id"]

    disputed = api.post(f"/api/sleep/me/thought/{hunch['id']}/dispute",
                        json={"note": "  I already asked my tutor.  "}, headers=_bearer(token))
    assert disputed.status_code == 200
    assert disputed.json() == {"kind": "dispute", "item_id": hunch["id"], "at": disputed.json()["at"],
                               "note": "I already asked my tutor."}
    after = api.get("/api/sleep/me/thought", headers=_bearer(token)).json()
    assert next(h for h in after["hunches"] if h["id"] == hunch["id"])["disputed"] is True
    assert after["revision"] == first["revision"] + 1  # a mark moves the revision the page polls
    assert api.get("/api/sleep/me/mind", headers=_bearer(token)).json()["revision"] == after["revision"]
    said = [i for i in evidence_store(user_id).stored(datetime.max.date()) if i.source == f"thought:dispute:{hunch['id']}"]
    assert len(said) == 1 and said[0].kind == "correction" and said[0].text == "I already asked my tutor."
    assert said[0].ancestry == frozenset() and said[0].model is None and not said[0].held_out

    assert api.post(f"/api/sleep/me/thought/{question}/dispute", headers=_bearer(token)).status_code == 200
    assert api.get("/api/sleep/me/thought", headers=_bearer(token)).json()["question"]["disputed"] is True

    assert api.post(f"/api/sleep/me/thought/{hunch['question_id']}/hide", headers=_bearer(token)).status_code == 200
    hidden = api.get("/api/sleep/me/thought", headers=_bearer(token)).json()
    assert hunch["id"] not in _ids(hidden) and hidden["hidden"] == 1
    assert hidden["scorecard"] == first["scorecard"]  # hiding never rewrites her record

    raw = (sleep_dir(user_id) / judge_thought.MARKS_FILE).read_text()
    assert "dispute" not in raw and "tutor" not in raw  # encrypted at rest
    stored = judge_thought.load_marks(user_id)
    assert [m["kind"] for m in stored] == ["dispute", "dispute", "hide"]
    assert [thought.Mark.from_json(m).to_json() for m in stored] == stored  # the engine's own records


def test_a_note_is_capped_at_the_engines_limit_with_a_422_never_a_404(api: Any) -> None:  # noqa: F811
    from anjo.sleep.app.storage import evidence_store

    token, user_id = _judge(api)
    hunch = api.get("/api/sleep/me/thought", headers=_bearer(token)).json()["hunches"][0]["id"]
    too_long = api.post(f"/api/sleep/me/thought/{hunch}/dispute", json={"note": "x" * (thought.MAX_NOTE + 1)},
                        headers=_bearer(token))
    assert too_long.status_code == 422
    assert judge_thought.load_marks(user_id) == [] and evidence_store(user_id).stored(datetime.max.date()) == []
    padded = api.post(f"/api/sleep/me/thought/{hunch}/dispute", json={"note": "  " + "y" * (thought.MAX_NOTE - 4) + "  "},
                      headers=_bearer(token))
    assert padded.status_code == 200 and padded.json()["note"] == "y" * (thought.MAX_NOTE - 4)
    with pytest.raises(judge_thought.NoteTooLong):  # past the route: still never a 404
        judge_thought.dispute(user_id, hunch, "z" * (thought.MAX_NOTE + 1))


def test_an_id_not_in_her_log_is_404(api: Any) -> None:  # noqa: F811
    token, user_id = _judge(api)
    assert api.post("/api/sleep/me/thought/p999/hide", headers=_bearer(token)).status_code == 404
    assert api.post("/api/sleep/me/thought/p999/dispute", json={"note": "no"}, headers=_bearer(token)).status_code == 404
    assert judge_thought.load_marks(user_id) == []


def test_a_log_that_doesnt_replay_is_500_never_404(api: Any) -> None:  # noqa: F811
    events = list(thought_demo.demo_events())
    events[-1] = replace(events[-1], after="0" * 64)
    token, user_id = _judge(api, events)
    assert api.get("/api/sleep/me/thought", headers=_bearer(token)).status_code == 500
    assert api.post("/api/sleep/me/thought/p999/hide", headers=_bearer(token)).status_code == 500
    assert judge_thought.load_marks(user_id) == []


def test_a_marks_file_that_cant_be_read_is_500(api: Any) -> None:  # noqa: F811
    from anjo.sleep.app.storage import sleep_dir

    token, user_id = _judge(api)
    (sleep_dir(user_id) / judge_thought.MARKS_FILE).write_text("not a record\n")
    assert api.get("/api/sleep/me/thought", headers=_bearer(token)).status_code == 500


def test_the_view_runs_on_the_night_clock(api: Any) -> None:  # noqa: F811
    _token, user_id = _judge(api)
    early = DEMO_NOW - timedelta(days=2)  # a wall clock behind the night that logged
    seen = judge_thought.view(user_id, now=early)
    assert seen["as_of"] == DEMO_NOW.isoformat()  # nothing she logged is left out
    assert {k: v for k, v in seen.items() if k not in ("revision", "live_pass")} == thought.thought_view(
        thought_demo.demo_events(), (), now=DEMO_NOW)
    assert judge_thought.view(user_id, now=WALL)["as_of"] == WALL.isoformat()


def test_a_visit_survives_polling_and_the_next_one_starts_where_its_clock_ended() -> None:
    first = judge_thought.visit("judge-1", WALL)
    again = judge_thought.visit("judge-1", WALL + timedelta(minutes=10))
    later = judge_thought.visit("judge-1", WALL + timedelta(minutes=20), clock=WALL + timedelta(days=1))
    back = judge_thought.visit("judge-1", WALL + timedelta(hours=8))
    still = judge_thought.visit("judge-1", WALL + timedelta(hours=8, minutes=5))
    assert (first, again, later) == (None, None, None)
    assert back == still == WALL + timedelta(days=1)  # the night clock the last visit read


def test_a_return_visit_gets_the_morning_note(api: Any) -> None:  # noqa: F811
    from anjo.sleep.app.storage import workspace_log

    events = thought_demo.demo_events()
    upto = tuple(e for e in events if e.step.at <= thought_demo.SINCE)
    _token, user_id = _judge(api, upto)
    assert judge_thought.view(user_id, now=thought_demo.SINCE)["morning"] is None
    workspace_log(user_id).extend(events[len(upto):])  # two more nights while they were away
    back = judge_thought.view(user_id, now=WALL)
    expected = thought.thought_view(events, (), now=WALL, since=thought_demo.SINCE)
    assert back["morning"] == expected["morning"] and back["morning"] is not None


def _live_start(events: tuple[Any, ...], at: datetime) -> tuple[Any, ...]:
    """``events`` and one live pass's ``RunStart`` after them, applied as the loop logs it."""
    from anjo.sleep.mind.inquiry import kernel
    from anjo.sleep.mind.inquiry import types as it
    from anjo.sleep.night.mind import LIVE_DECIDER

    ws = kernel.fold(events)
    start = it.RunStart(at.date(), at, "notice", it.FULL_RULES, "priority", None, it.Budget(), False,
                        kernel.DENYLIST_VERSION, decider=LIVE_DECIDER)
    result = kernel.apply(ws, it.Step(ws.seq, at, start), rules=it.FULL_RULES)
    assert isinstance(result, it.Applied)
    return (*events, result.event)


def test_a_night_logged_within_the_gap_starts_a_new_visit(api: Any) -> None:  # noqa: F811
    """Sign in, sleep, wake ten minutes later: the morning note covers the night."""
    from anjo.sleep.app.storage import workspace_log

    events = thought_demo.demo_events()
    upto = tuple(e for e in events if e.step.at <= thought_demo.SINCE)
    _token, user_id = _judge(api, upto)
    assert judge_thought.view(user_id, now=thought_demo.SINCE)["morning"] is None  # signed in
    workspace_log(user_id).extend(events[len(upto):])  # nights 4 and 5
    woke = judge_thought.view(user_id, now=thought_demo.SINCE + timedelta(minutes=10))
    expected = thought.thought_view(events, (), now=DEMO_NOW, since=thought_demo.SINCE)
    assert woke["morning"] is not None and woke["morning"] == expected["morning"]
    polled = judge_thought.view(user_id, now=thought_demo.SINCE + timedelta(minutes=20))
    assert polled["morning"] == woke["morning"]  # the same visit: polling keeps the note


def test_a_night_logged_behind_the_last_visits_clock_still_counts_from_its_start(api: Any) -> None:  # noqa: F811
    from anjo.sleep.app.storage import workspace_log

    events = thought_demo.demo_events()
    upto = tuple(e for e in events if e.step.at <= thought_demo.SINCE)
    _token, user_id = _judge(api, upto)
    judge_thought.view(user_id, now=WALL)  # a wall clock past every night the demo logs
    workspace_log(user_id).extend(events[len(upto):])
    night4 = next(e.step.at for e in events[len(upto):])
    woke = judge_thought.view(user_id, now=WALL + timedelta(minutes=10))
    expected = thought.thought_view(events, (), now=WALL + timedelta(minutes=10), since=night4 - timedelta(microseconds=1))
    assert woke["morning"] == expected["morning"] and woke["morning"]["settled"]


def test_a_live_pass_after_a_reply_is_no_night() -> None:
    events = thought_demo.demo_events()
    assert judge_thought.visit("judge-1", WALL, events=events) is None
    live = _live_start(events, WALL + timedelta(minutes=1))
    assert judge_thought.visit("judge-1", WALL + timedelta(minutes=5), events=live) is None  # the same visit
