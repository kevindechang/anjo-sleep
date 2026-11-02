# mypy: disallow-untyped-decorators=False, disallow-untyped-calls=False
# (pyproject's follow_imports = "skip" hides pytest's own types)
"""Her own line of thought for the judge it is about: the view over their own log and marks, a
visit that survives the page's polling, dispute and hide stored as the person's own encrypted marks
(a dispute's note also kept as their own words in their evidence), 404 for anything their view
doesn't show, and 503 until the engine lands. The engine is a fake shaped like its schema."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from anjo.sleep.app import judge_thought
from anjo.sleep.tests.app.test_judge_api import _bearer, _start, api, talker  # noqa: F401  (fixtures)

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FakeMark:
    kind: str
    item_id: str
    at: datetime
    note: str | None = None


class FakeEngine:
    """``thought_view``, ``dispute`` and ``hide`` as the schema states them; one hunch ``p3`` on
    question ``q1``, one settled bet ``p1``, and the question ``q2`` it asks."""

    Mark = FakeMark  # the engine's own Mark, a dataclass

    def __init__(self) -> None:
        self.views: list[dict[str, Any]] = []

    def thought_view(self, events: Sequence[Any], marks: Sequence[FakeMark], *, now: datetime,
                     since: datetime | None = None) -> dict[str, Any]:
        assert all(isinstance(m, FakeMark) for m in marks)
        hidden = {m.item_id for m in marks if m.kind == "hide"}
        disputed = {m.item_id for m in marks if m.kind == "dispute"}
        self.views.append({"since": since, "now": now})
        hunches = [{"id": "p3", "question_id": "q1", "disputed": "p3" in disputed}]
        return {
            "version": 1, "as_of": now.isoformat(),
            "hunches": [h for h in hunches if h["id"] not in hidden],
            "settled": [{"id": "p1", "question_id": "q1", "outcome": "true"}],
            "question": {"question_id": "q2"},
            "morning": None if since is None else {"since": since.isoformat(), "empty": True},
            "hidden": len(hidden),
        }

    def dispute(self, item_id: str, *, at: datetime, note: str | None = None) -> FakeMark:
        return FakeMark("dispute", item_id, at, note)

    def hide(self, item_id: str, *, at: datetime) -> FakeMark:
        return FakeMark("hide", item_id, at)


@pytest.fixture
def engine() -> Iterator[FakeEngine]:
    fake = FakeEngine()
    judge_thought.set_engine(fake)
    yield fake
    judge_thought.set_engine(None)


def test_without_the_engine_the_routes_say_so(api: Any) -> None:  # noqa: F811
    token = _start(api).json()["token"]
    assert judge_thought.engine() is None  # not on this branch yet
    assert api.get("/api/sleep/me/thought", headers=_bearer(token)).status_code == 503
    assert api.post("/api/sleep/me/thought/p3/hide", headers=_bearer(token)).status_code == 503


def test_the_view_is_their_own_and_disputes_and_hides_are_marks(api: Any, engine: FakeEngine) -> None:  # noqa: F811
    from anjo.sleep.app.storage import evidence_store, sleep_dir

    body = _start(api).json()
    token, user_id = body["token"], body["user_id"]
    seen = api.get("/api/sleep/me/thought", headers=_bearer(token))
    assert seen.status_code == 200 and [h["id"] for h in seen.json()["hunches"]] == ["p3"]

    disputed = api.post("/api/sleep/me/thought/p3/dispute", json={"note": "  The courtyard is closed for good.  "},
                        headers=_bearer(token))
    assert disputed.status_code == 200
    assert disputed.json()["kind"] == "dispute" and disputed.json()["note"] == "The courtyard is closed for good."
    assert api.get("/api/sleep/me/thought", headers=_bearer(token)).json()["hunches"][0]["disputed"] is True
    said = [item for item in evidence_store(user_id).stored(datetime.max.date()) if item.source == "thought:dispute:p3"]
    assert len(said) == 1 and said[0].kind == "correction" and said[0].text == "The courtyard is closed for good."
    assert said[0].ancestry == frozenset() and said[0].model is None and not said[0].held_out

    assert api.post("/api/sleep/me/thought/q2/hide", headers=_bearer(token)).status_code == 200
    assert api.post("/api/sleep/me/thought/p3/hide", headers=_bearer(token)).status_code == 200
    after = api.get("/api/sleep/me/thought", headers=_bearer(token)).json()
    assert after["hunches"] == [] and after["hidden"] == 2
    raw = (sleep_dir(user_id) / judge_thought.MARKS_FILE).read_text()
    assert "dispute" not in raw and "courtyard" not in raw  # encrypted at rest
    assert [m["kind"] for m in judge_thought.load_marks(user_id)] == ["dispute", "hide", "hide"]


def test_only_what_their_view_shows_can_be_marked(api: Any, engine: FakeEngine) -> None:  # noqa: F811
    token = _start(api).json()["token"]
    assert api.post("/api/sleep/me/thought/p9/hide", headers=_bearer(token)).status_code == 404
    assert api.post("/api/sleep/me/thought/p9/dispute", json={}, headers=_bearer(token)).status_code == 404
    assert api.post("/api/sleep/me/thought/p1/dispute", headers=_bearer(token)).status_code == 200  # no note: no evidence


def test_a_visit_survives_polling_and_the_next_one_starts_from_its_end() -> None:
    first = judge_thought.visit("judge-1", NOW)
    again = judge_thought.visit("judge-1", NOW + timedelta(minutes=10))
    later = judge_thought.visit("judge-1", NOW + timedelta(minutes=20))
    back = judge_thought.visit("judge-1", NOW + timedelta(hours=8))
    still = judge_thought.visit("judge-1", NOW + timedelta(hours=8, minutes=5))
    assert (first, again, later) == (None, None, None)
    assert back == still == NOW + timedelta(minutes=20)


def test_a_mark_without_its_engine_reader_round_trips_as_a_record() -> None:
    record = judge_thought._record({"kind": "hide", "item_id": "p3", "at": NOW, "note": None})
    assert record == {"kind": "hide", "item_id": "p3", "at": NOW.isoformat()}
    assert judge_thought._marks_for(object(), [record]) == [record]
