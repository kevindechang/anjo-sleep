"""The return moment: her own message, from her records, when the person comes back after quiet days.

The held note has one outlet today: a finished night queues it as the person's pending outreach
(``note.hold`` → ``core.outreach.queue_outreach``), and it surfaces as the next open's first bubble.
The live demo has no account to open, so this module reads the same pending record for the demo's
synthetic account, and falls back to the share MIND raised from its own state: an unconsumed result
on a finding made since the person last wrote. Nothing here is written, consumed or shown twice.

Both kinds carry their exact words: a note's final text (sources and held-back line included), or a
finding's statement. No sentence of fixed copy becomes her message; the page's labels live in
``words.COPY`` and are never part of ``text``. When the held-out filter withholds the words, the
moment says so instead of showing them. The record each message came from travels with it, and the
mind view's timeline is where its working out is read.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.timeline import quiet_days, quiet_start
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import Step, Workspace, WorkspaceEvent
from anjo.sleep.types import Evidence, Result


@dataclass(frozen=True)
class HeldNote:
    """The note waiting for the person's next open, as the night left it in outreach."""

    text: str
    id: str | None = None
    night: date | None = None
    at: datetime | None = None


@dataclass(frozen=True)
class ReturnSources:
    """What the return moment may read. ``None``: that source couldn't be read."""

    note: HeldNote | None = None
    events: Sequence[WorkspaceEvent] | None = None
    evidence: Sequence[Evidence] | None = None
    results: Sequence[Result] | None = None
    as_of: datetime | None = None


def finding_step(events: Sequence[WorkspaceEvent], finding_id: str) -> Step | None:
    """The logged step that last set ``finding_id``; None when the log no longer holds it."""
    final, before = None, Workspace()
    for index, event in enumerate(events):
        after = kernel.fold(events[:index + 1])
        was = next((f for f in before.findings if f.id == finding_id), None)
        now = next((f for f in after.findings if f.id == finding_id), None)
        if now is not None and now != was:
            final = event.step
        before = after
    return final


def _share(sources: ReturnSources, held: HeldOutText, last: datetime | None) -> dict[str, Any] | None:
    """The newest share still waiting on an open target, with the finding that raised it."""
    if sources.events is None or sources.results is None:
        return None
    waiting = sorted(
        (r for r in sources.results
         if not r.consumed and (last is None or r.created_at >= last)),
        key=lambda r: (r.created_at, r.id), reverse=True,
    )
    for result in waiting:
        finding = next((f for f in kernel.fold(sources.events).findings
                        if result.id == f"{f.id}.{f.kind}"), None)
        if finding is None:
            continue
        statement = held.item(finding.statement)
        if statement is None or statement["text"] is None:
            continue
        items = {item.id: item for item in sources.evidence or ()}
        step = finding_step(sources.events, finding.id)
        return {
            "kind": "share",
            "text": statement["text"],
            "record": {"id": finding.id, "question": finding.question_id, "kind": finding.kind,
                       "event": None if step is None else f"seq-{step.seq}",
                       "seq": None if step is None else step.seq,
                       "at": None if step is None else step.at.isoformat()},
            "sources": [{"id": cite, "url": None if items.get(cite) is None else items[cite].source}
                        for cite in finding.cites],
        }
    return None


def return_view(sources: ReturnSources, held: HeldOutText) -> dict[str, Any]:
    """Her message, the record behind it, and how many quiet days it answers."""
    last = quiet_start(sources.evidence, sources.as_of)
    days = quiet_days(last, sources.as_of)
    if sources.note is not None:
        note = held.item(sources.note.text)
        if note is not None and note["text"] is not None:
            return {
                "state": "ok", "quiet_days": days, "kind": "held_note", "text": note["text"],
                "record": {"id": sources.note.id, "kind": "night_note",
                           "night": None if sources.note.night is None else sources.note.night.isoformat(),
                           "at": None if sources.note.at is None else sources.note.at.isoformat()},
                "sources": [],
            }
    share = _share(sources, held, last)
    if share is not None:
        return {"state": "ok", "quiet_days": days, **share}
    if sources.note is None and sources.events is None:
        return {"state": "unavailable", "quiet_days": days}
    if sources.note is not None:
        return {"state": "withheld", "quiet_days": days}
    return {"state": "none", "quiet_days": days}
