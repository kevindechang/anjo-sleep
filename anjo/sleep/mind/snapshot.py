"""Build the state the wants are derived from (Task 7a). Pure: ``as_of`` is always passed in.

The snapshot decides what her stores may pull on, so an unchanged topic never comes back:

- a thread passes only with new evidence from the person (``revision - acted_revision`` above 0),
  or while an open or active question on it continues. A result of hers waiting to be said is its
  own record: it reaches ``results`` without reopening the thread;
- a commitment passes while it is open;
- a research result passes only with a verified source, and an evidence result only on an open
  commitment due within seven days that cites at least two of the person's own items. A research
  result that is already consumed stays, because a result with a verified source closes the
  commitment's preparation for good. Nothing else is a change: no model rates its own work.

A ledger entry (``Commitment``) lends its date to the commitment target with the same id when the
target has none. An entry with no target is not acted on: nothing records that a note was held
for it, so it would come back every night.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Final

from anjo.sleep.types import Commitment, QuestionView, Result, Target, WantSnapshot

EVIDENCE_WINDOW: Final = timedelta(days=7)
MIN_USER_CITES: Final = 2
LIVE_QUESTION: Final = frozenset({"open", "active"})
CLOSED_QUESTION: Final = frozenset({"parked", "concluded", "retracted", "abandoned"})


def build(
    targets: list[Target],
    results: list[Result],
    as_of: datetime,
    last_message_at: datetime | None,
    intentions: tuple[Commitment, ...] = (),
    questions: tuple[QuestionView, ...] = (),
) -> WantSnapshot:
    dated = {c.id: c.check_after for c in intentions if c.check_after is not None}
    open_targets = {t.id: _dated(t, dated) for t in targets if t.status == "open"}
    admitted = tuple(
        sorted(
            (r for r in results if _admitted(r, open_targets.get(r.target_id), as_of)),
            key=lambda r: r.id,
        ),
    )
    continuing = continuing_targets(questions)
    kept = tuple(
        sorted(
            (t for t in open_targets.values() if _passes(t, continuing)),
            key=lambda t: t.id,
        ),
    )
    return WantSnapshot(
        as_of=as_of,
        last_message_at=last_message_at,
        targets=kept,
        results=admitted,
        intentions=tuple(sorted(intentions, key=lambda c: c.id)),
        questions=tuple(sorted(questions, key=lambda q: q.id)),
    )


def continuing_targets(questions: tuple[QuestionView, ...]) -> frozenset[str]:
    """Targets of an open or active question whose continuation holds. A parked, concluded,
    retracted or abandoned question continues nothing, whatever its flag says."""
    return frozenset(
        target_id
        for q in questions
        if q.status in LIVE_QUESTION and q.continues
        for target_id in q.target_ids
    )


def close_marks(question: QuestionView, targets: list[Target]) -> tuple[tuple[str, int], ...]:
    """The ``mark_acted`` calls a question's closing owes: each of its targets at its current
    revision, so only the person's new evidence wakes them. An open or active question owes none."""
    if question.status not in CLOSED_QUESTION:
        return ()
    by_id = {t.id: t for t in targets}
    return tuple((tid, by_id[tid].revision) for tid in question.target_ids if tid in by_id)


def _dated(target: Target, ledger: dict[str, datetime]) -> Target:
    if target.kind != "commitment" or target.check_after is not None or target.id not in ledger:
        return target
    return replace(target, check_after=ledger[target.id])


def _passes(target: Target, continuing: frozenset[str]) -> bool:
    if target.kind == "commitment":
        return True
    return target.unacted > 0 or target.id in continuing


def _admitted(result: Result, target: Target | None, as_of: datetime) -> bool:
    if target is None:
        return False
    if result.basis == "research":
        return result.verified_sources >= 1
    if result.consumed or target.kind != "commitment" or target.check_after is None:
        return False
    ahead = target.check_after - as_of
    return timedelta(0) <= ahead <= EVIDENCE_WINDOW and result.user_cites >= MIN_USER_CITES
