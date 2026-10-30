"""How she got here (the mind view's timeline): a quiet window of her logged work, in order.

The page's centre is one moment: the person came back after days without writing, and she had
already worked something out. This module turns the records that moment rests on into nodes a
person can read:

- **questions she opened**, with whether the day's evidence raised them (the notice pass) or her own
  inquiry cycle did;
- **credences that moved**, before → after, with the item or resolved prediction that moved them;
- **predictions**, committed → resolved, and whether the outcome made the claim right or wrong;
- **web searches she ran**, with the query and what came back (her own records' search is not a web
  search and is left out);
- **each night's decision**, with the confidence she gave it when the record carries one.

Each node names the logged record it comes from (``record.event``: the workspace step's seq, or a
night's ``choose`` step). ``plain`` fields are for the line a person reads first; every free text
(question, hypothesis, claim, query, answer, evidence, target label) is passed through the held-out
filter and arrives as ``{"text": ..., "withheld": ...}``, so exact words only show on expand.

Pure: records in, payload out. No clock, file, model or database read. The fold is tolerant of step
kinds this build does not render: they produce no node and never fail the timeline.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final

from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.evidence.targets import PERSON_KINDS
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import (
    Answered,
    HypothesizeOp,
    InvestigateOp,
    Retrieved,
    Resolved,
    RunStart,
    Update,
    Workspace,
    WorkspaceEvent,
)
from anjo.sleep.types import Evidence, Result

WEB_TOOLS: Final[frozenset[str]] = frozenset({"research", "world_query"})  # the outer searches she runs
CREDENCE_PLACES: Final = 6


@dataclass(frozen=True)
class DecisionRecord:
    """One night's choice, already reduced to ids, codes and numbers by ``day.panel``.

    ``at`` is the night's own clock (the end of its day), so a decision sorts after the day it acted
    on; ``night`` is the date the page groups it under.
    """

    night: date
    at: datetime
    action: str | None
    target_id: str | None
    outcome: str | None
    status: str
    confidence: float | None = None


@dataclass(frozen=True)
class TimelineSources:
    """What the timeline reads. ``None``: that source couldn't be read.

    ``events`` is MIND's workspace log, not the folded workspace: a wipe keeps the ids of everything
    before it, and the timeline must be able to cite them.
    """

    events: Sequence[WorkspaceEvent] | None = None
    evidence: Sequence[Evidence] | None = None
    results: Sequence[Result] | None = None
    decisions: Sequence[DecisionRecord] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    as_of: datetime | None = None


def quiet_start(evidence: Sequence[Evidence] | None, as_of: datetime | None) -> datetime | None:
    """When the person last wrote: their newest own item on or before ``as_of``. Held-out items
    never set it, as they never reach her."""
    if evidence is None or as_of is None:
        return None
    own = [item.created_at for item in evidence
           if item.kind in PERSON_KINDS and not item.held_out and item.created_at <= as_of]
    return max(own) if own else None


def quiet_days(start: datetime | None, as_of: datetime | None) -> int:
    """Whole days between the person's last item and the end of her latest night."""
    if start is None or as_of is None:
        return 0
    return max(0, (as_of.date() - start.date()).days)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _evidence_items(evidence: Sequence[Evidence] | None) -> dict[str, Evidence]:
    return {item.id: item for item in evidence or ()}


def _cite(item_id: str, held: HeldOutText, items: Mapping[str, Evidence]) -> dict[str, Any]:
    """One cited record: its id always, its words and source only through the filter."""
    item = items.get(item_id)
    return {"id": item_id, "text": None if item is None else held.item(item.text),
            "source": None if item is None else item.source}


def _moves(before: Workspace, after: Workspace, update: Update, held: HeldOutText) -> list[dict[str, Any]]:
    """The hypotheses whose credence this one update moved, each with its before and after."""
    old = {h.id: h.credence for h in before.hypotheses}
    moves = []
    for h in after.hypotheses_of(update.question_id):
        previous = old.get(h.id)
        if previous is None or abs(previous - h.credence) > 10 ** -CREDENCE_PLACES:
            moves.append({"id": h.id, "statement": held.item(h.statement), "before": previous,
                          "after": h.credence, "catch_all": h.is_catch_all})
    return moves


def _verdict(outcome: str | None) -> str:
    """Right, wrong, or no verdict: what a resolution means without dressing up an unfinished one."""
    if outcome == "true":
        return "right"
    if outcome == "false":
        return "wrong"
    return "open"


def _search_node(base: dict[str, Any], tool: str, query: str, held: HeldOutText) -> dict[str, Any]:
    return {**base, "kind": "search", "tool": tool,
            "detail": {"query": held.item(query), "answer": None, "items": []}}


def _decisions(decisions: Sequence[DecisionRecord], start: datetime, end: datetime,
               labels: Mapping[str, str], held: HeldOutText) -> list[dict[str, Any]]:
    """Each night's choice in the quiet days; a night is kept by its date, not its end-of-day clock."""
    kept = [d for d in decisions if start.date() <= d.night <= end.date()]
    return [{
        "kind": "decision",
        "at": d.at.isoformat(),
        "day": d.night.isoformat(),
        "night": d.night.isoformat(),
        "seq": None,
        "record": {"event": f"night-{d.night.isoformat()}-choose", "seq": None, "night": d.night.isoformat()},
        "action": d.action,
        "outcome": d.outcome,
        "status": d.status,
        "confidence": d.confidence,
        "target": held.item(labels.get(d.target_id or "")),
    } for d in sorted(kept, key=lambda d: (d.night, d.action or ""))]


def _nodes(sources: TimelineSources, held: HeldOutText) -> tuple[list[dict[str, Any]], int]:
    """Every rendered step in the quiet window, and the day count the window covers."""
    events, as_of = sources.events, sources.as_of
    start = quiet_start(sources.evidence, as_of)
    if events is None or as_of is None or start is None:
        return [], quiet_days(start, as_of)
    items = _evidence_items(sources.evidence)
    nodes: list[dict[str, Any]] = []
    searches: dict[tuple[str, str], dict[str, Any]] = {}
    before = Workspace()
    call, night = "inquiry", None
    for index, event in enumerate(events):
        after = kernel.fold(events[:index + 1])
        step = event.step
        entry = step.entry
        if isinstance(entry, RunStart):
            call, night = entry.call, entry.night
        if start <= step.at <= as_of:
            base = {"at": step.at.isoformat(), "day": step.at.date().isoformat(), "seq": step.seq,
                    "night": None if night is None else night.isoformat(),
                    "record": {"event": f"seq-{step.seq}", "seq": step.seq,
                               "night": None if night is None else night.isoformat()}}
            for q in after.questions:
                if any(old.id == q.id for old in before.questions):
                    continue
                nodes.append({**base, "kind": "question", "code": q.kind, "status": q.status,
                              "origin": "person" if call == "notice" else "her",
                              "detail": {"subject": held.item(q.subject)},
                              "cites": [_cite(source, held, items) for source in q.origin_evidence]})
            for update in after.updates:
                if any(old == update for old in before.updates):
                    continue
                moves = _moves(before, after, update, held)
                if moves:
                    nodes.append({**base, "kind": "credence", "source": update.source,
                                  "moved_by": update.ref, "moves": moves,
                                  "evidence": [_cite(item, held, items) for item in update.evidence_ids]})
            if isinstance(entry, HypothesizeOp) and not any(
                    h.question_id == entry.question_id for h in before.hypotheses):
                question = after.question(entry.question_id)
                hypotheses = after.hypotheses_of(entry.question_id)
                if question is not None and hypotheses:
                    nodes.append({**base, "kind": "credence", "source": "hypothesized",
                                  "moved_by": entry.question_id,
                                  "moves": [{"id": h.id, "statement": held.item(h.statement), "before": None,
                                             "after": h.credence, "catch_all": h.is_catch_all}
                                            for h in hypotheses],
                                  "evidence": [_cite(item, held, items) for item in question.origin_evidence]})
            for prediction in after.predictions:
                old = next((p for p in before.predictions if p.id == prediction.id), None)
                if old is None:
                    nodes.append({**base, "kind": "prediction", "detail": {"claim": held.item(prediction.claim)},
                                  "given": dict(prediction.p_true_given),
                                  "horizon": prediction.horizon.isoformat(), "outcome": prediction.outcome})
                elif old.outcome != prediction.outcome:
                    seen = entry.seen if isinstance(entry, Resolved) and entry.prediction_id == prediction.id else ()
                    nodes.append({**base, "kind": "resolution", "detail": {"claim": held.item(prediction.claim)},
                                  "outcome": prediction.outcome, "verdict": _verdict(prediction.outcome),
                                  "seen": [_cite(item.id, held, items) for item in seen]})
            if isinstance(entry, InvestigateOp) and entry.tool in WEB_TOOLS:
                node = _search_node(base, entry.tool, entry.query, held)
                nodes.append(node)
                searches[(entry.question_id, entry.query)] = node
            elif isinstance(entry, Answered):
                # A world query answers in its own logged step: attach it to the search that asked.
                node = searches.get((entry.question_id, entry.query))
                if node is None:
                    node = _search_node(base, "world_query", entry.query, held)
                    nodes.append(node)
                    searches[(entry.question_id, entry.query)] = node
                node["detail"]["answer"] = held.item(entry.answer)
            elif isinstance(entry, Retrieved) and entry.tool in WEB_TOOLS and entry.question_id is not None:
                node = searches.get((entry.question_id, entry.query))
                if node is None:  # its INVESTIGATE ran before the quiet days: the result still shows
                    node = _search_node(base, entry.tool, entry.query, held)
                    nodes.append(node)
                    searches[(entry.question_id, entry.query)] = node
                node["detail"]["items"] = [_cite(item.id, held, items) for item in entry.items]
        before = after
    nodes.extend(_decisions(sources.decisions, start, as_of, sources.labels, held))
    nodes.sort(key=lambda node: (node["at"], -1 if node["seq"] is None else node["seq"]))
    return nodes, quiet_days(start, as_of)


def timeline_view(sources: TimelineSources, held: HeldOutText) -> dict[str, Any]:
    """The timeline's payload: a flat, ordered list of nodes; the page groups them by day."""
    try:
        nodes, days = _nodes(sources, held)
    except (ValueError, KeyError):
        return {"state": "unavailable"}
    if sources.events is None or sources.as_of is None:
        return {"state": "unavailable"}
    if not nodes:
        return {"state": "none", "quiet_days": days, "from": None, "to": sources.as_of.isoformat(),
                "items": [], "counts": {}}
    start = quiet_start(sources.evidence, sources.as_of)
    counts = Counter(node["kind"] for node in nodes)
    return {
        "state": "ok",
        "quiet_days": days,
        "from": None if start is None else start.isoformat(),
        "to": sources.as_of.isoformat(),
        "items": nodes,
        "counts": {key: counts.get(key, 0) for key in
                   ("question", "credence", "prediction", "resolution", "search", "decision")},
    }
