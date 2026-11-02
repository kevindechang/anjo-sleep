"""A synthetic week of her thinking, for the thought panel's ``?mock=1`` data and its tests.

Mira (synthetic) is finishing a short film. Over five nights MIND notices three things:
- whether Saturday's courtyard shoot survives the rain forecast;
- whether the edit makes Friday's deadline;
- whether Mira enters the March festival.

On them MIND makes five bets. Two come true as she expected, one fails against her odds, and two
are still open. She reaches one conclusion and changes her mind twice. Every step goes through
``kernel.apply``, exactly as the night loop logs it, so the view reads a real log. No model is
called.

``python -m anjo.sleep.mind.thought_demo > thought_mock.json`` writes the view as JSON.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final

from anjo.sleep.mind import thought
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.types import Evidence

PERSON: Final = "demo-mira"
DAY0: Final = date(2026, 10, 26)
TARGET: Final = "t-film"


def _night(n: int) -> datetime:
    return datetime.combine(DAY0 + timedelta(days=n), time(23, 0), tzinfo=timezone.utc)


NOW: Final = _night(5)
SINCE: Final = _night(3) + timedelta(hours=1)  # her last visit: the morning after night 3


def _item(n: int, day: int, text: str) -> Evidence:
    at = datetime.combine(DAY0 + timedelta(days=day), time(12, 0), tzinfo=timezone.utc)
    return Evidence(id=f"m{n}", person_id=PERSON, day=at.date(), kind="lesson", text=text, source=f"message:m{n}",
                    held_out=False, created_at=at, model=None, ancestry=frozenset(), target_id=TARGET)


ITEMS: Final = (
    _item(0, 0, "I booked the courtyard for Saturday's shoot."),
    _item(1, 1, "The forecast says rain all Saturday."),
    _item(2, 2, "The edit is due to my tutor next Friday."),
    _item(3, 2, "I haven't started the colour grade."),
    _item(4, 3, "We're scouting the kitchen tomorrow, just in case."),
    _item(5, 4, "Shot it all on Saturday, in the kitchen in the end."),
    _item(6, 5, "Still grading. I'll probably need the weekend."),
    _item(7, 5, "Thinking about entering the festival in March."),
    _item(8, 5, "If it rains next time we'll just shoot in the kitchen again."),
)


class _Life:
    """Steps applied one at a time, as the loop logs them; a refused step is a bug in the demo."""

    def __init__(self) -> None:
        self.ws: it.Workspace = it.Workspace()
        self.events: list[it.WorkspaceEvent] = []

    def ok(self, entry: Any, n: int) -> None:
        step = it.Step(self.ws.seq, _night(n), entry, None, None, None, 0)
        result = kernel.apply(self.ws, step, rules=it.FULL_RULES)
        if not isinstance(result, it.Applied):
            raise RuntimeError(f"the demo's step {step.seq} was refused: {result.code}")
        self.ws = result.ws
        self.events.append(result.event)

    def night(self, n: int, *items: Evidence) -> None:
        self.ok(it.RunStart(_night(n).date(), _night(n), "inquiry", it.FULL_RULES, "priority", None, it.Budget(),
                            False, None), n)
        self.ok(it.Retrieved(None, "new_evidence", "", tuple(_ref(i) for i in items)), n)

    def question(self, n: int, kind: it.QuestionKind, subject: str, origin: Sequence[Evidence],
                 alternatives: Sequence[tuple[str, float]], catch_all: float) -> str:
        self.ok(it.NoticeOp(kind, subject, (TARGET,), tuple(i.id for i in origin)), n)
        qid: str = next(q.id for q in self.ws.questions if q.subject == subject)
        self.ok(it.HypothesizeOp(qid, tuple(it.Alternative(s, p) for s, p in alternatives), catch_all), n)
        return qid

    def bet(self, n: int, qid: str, claim: str, given: Sequence[float], days: int) -> str:
        ids = [h.id for h in self.ws.hypotheses_of(qid)]
        self.ok(it.PredictOp(qid, claim, dict(zip(ids, given, strict=True)), days), n)
        pid: str = next(p.id for p in self.ws.predictions if p.claim == claim)
        return pid

    def ids(self, qid: str) -> list[str]:
        return [h.id for h in self.ws.hypotheses_of(qid)]


def _ref(item: Evidence) -> it.EvidenceRef:
    return it.EvidenceRef(id=item.id, day=item.day, created_at=item.created_at, target_id=item.target_id)


def demo_events() -> tuple[it.WorkspaceEvent, ...]:
    m, (e0, e1, e2, e3, e4, e5, e6, e7, e8) = _Life(), ITEMS
    m.night(1, e0, e1)
    shoot = m.question(1, "contradiction", "Will Saturday's courtyard shoot survive the rain?", (e0, e1),
                       (("It goes ahead outside", 0.45), ("It moves indoors to the kitchen", 0.35)), 0.20)
    backup = m.bet(1, shoot, "Mira mentions a backup location before Friday", (0.3, 0.9, 0.5), 3)
    happens = m.bet(1, shoot, "The shoot happens on Saturday", (0.9, 0.8, 0.3), 4)
    m.night(2, e2, e3)
    edit = m.question(2, "gap", "Will the edit be ready for Friday?", (e2, e3),
                      (("It's ready on Friday", 0.42), ("She asks her tutor for more time", 0.40)), 0.18)
    m.bet(2, edit, "Mira asks her tutor for more time this week", (0.1, 0.9, 0.5), 6)
    m.night(3, e4)
    m.ok(it.Resolved(backup, "true", (_ref(e4),)), 3)
    grade = m.bet(3, edit, "Mira says the grade is done by Wednesday", (0.9, 0.3, 0.6), 2)
    m.night(4, e5)
    m.ok(it.Resolved(happens, "true", (_ref(e5),)), 4)
    m.night(5, e6, e7, e8)
    m.ok(it.Retrieved(shoot, "evidence_search", "kitchen shoot", (_ref(e8),)), 5)
    m.ok(it.AssessOp(shoot, e8.id, dict(zip(m.ids(shoot), (0.25, 4.0, 0.5), strict=True))), 5)
    m.ok(it.ConcludeOp(shoot, m.ids(shoot)[1], "resolution", "Bad weather moves her shoots indoors, not later",
                       (e8.id,)), 5)
    m.ok(it.Resolved(grade, "false", (_ref(e6),)), 5)
    festival = m.question(5, "gap", "Will Mira enter the March festival?", (e7,),
                          (("She submits for March", 0.40), ("She waits for the autumn round", 0.38)), 0.22)
    m.bet(5, festival, "Mira looks up the festival's entry fee", (0.8, 0.4, 0.5), 5)
    return tuple(m.events)


def demo_view() -> dict[str, Any]:
    return thought.thought_view(demo_events(), (), now=NOW, since=SINCE)


def main() -> None:
    print(json.dumps(demo_view(), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
