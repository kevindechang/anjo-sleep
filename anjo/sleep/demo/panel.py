"""Her mind panel (Task 13a, full step): her state as the page shows it, built on the server.

``mind_view`` turns what the app read into the panel's sections, each ``ok``, ``none`` (nothing to
show yet) or ``unavailable`` (its source couldn't be read):

- ``questions``: MIND's open, active and parked questions (Task 17), with their hypotheses, their
  credences (a heuristic, and labeled so on the page) and their predictions;
- ``targets``: her open targets, with her heuristic confidence that she understood each;
- ``wants``: the wants her latest night considered, as its receipt recorded them, the winner first;
- ``choices``: what each night chose and what it beat, nights she chose nothing included, with what
  became of the winner and of the night's note, and how the night slept (a nap or a deep sleep,
  its pressure, threshold and reason);
- ``expectations``: her expectations (Task 16) and how they resolved: the stored outcome, else what
  the person's side shows so far (the topic came back, still open, or past its horizon and
  waiting for the scorer, whose held-out half the page never reads);
- ``gates``: each night's gates, their values and thresholds. A gate's ``reason`` is never shown:
  it can name the safety case that tripped;
- ``story``: her latest night as one loop, dream → rehearse → learn → check → wake
  (``night_view.story_view``), from the night's ``story.json``, or told from its log for a night
  from before stories (``night.story.tell``). A story that can't be read or told makes only this
  section unavailable.

Every free text (a question, a hypothesis, a prediction's claim, a target's label) passes the
held-out filter (``heldout.HeldOutText``) and is sent as ``{"text": ..., "withheld": ...}``. Ids,
codes and numbers are sent as they are. Time is the night's clock: ``as_of`` is the end of her
latest night, virtual in an accelerated life, so nothing is resolved by the server's own date.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final

from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.night_view import story_view
from anjo.sleep.demo.timeline import DecisionRecord, TimelineSources, timeline_view
from anjo.sleep.mind.inquiry.types import LIVE_STATUSES, Question, Workspace, WorkspaceEvent
from anjo.sleep.night.sleep_pressure import SleepMark, SleepRecordError, sleep_mark
from anjo.sleep.night.story import Story, StoryError, story_from, tell
from anjo.sleep.types import Evidence, Expectation, Result, Target

SHOWN_STATUSES: Final = (*sorted(LIVE_STATUSES), "parked")  # a question's statuses the panel shows


class PanelError(ValueError):
    """A night's record lacks what the panel needs."""


@dataclass(frozen=True)
class GateMark:
    gate: str
    passed: bool
    value: float | None
    threshold: float | None
    skipped: bool = False  # switched off: it judged nothing, and is never shown as passed


@dataclass(frozen=True)
class Candidate:
    want_id: str
    kind: str
    target_id: str | None
    action: str | None
    strength: float | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class NightRecord:
    """One night as its log recorded it: ids, codes and numbers only. ``outcome`` is what became
    of the chosen action (the choose step's receipt), ``note`` the note step's outcome, ``sleep`` a
    nap or a deep sleep with its pressure and reason (None for a night from before naps), ``story``
    the night as one loop (None when it can't be read or told)."""

    night: date
    status: str
    action: str | None
    target_id: str | None
    candidates: tuple[Candidate, ...]
    gates: tuple[GateMark, ...]
    outcome: str | None
    note: str | None
    sleep: SleepMark | None = None
    story: Story | None = None
    confidence: float | None = None  # the chooser's own confidence, when its record carries one


@dataclass(frozen=True)
class MindSources:
    """What the app read. ``None``: that source couldn't be read, so its sections are unavailable.
    ``returns``: when the person's own evidence came back to each target (``expect.returning``).
    ``events`` is MIND's whole log (a wipe keeps the ids the timeline cites); ``evidence`` and
    ``results`` are what its searches and shares are read against."""

    nights: Sequence[NightRecord] | None = None
    workspace: Workspace | None = None
    targets: Sequence[Target] | None = None
    expectations: Sequence[Expectation] | None = None
    returns: Mapping[str, Sequence[datetime]] = field(default_factory=dict)
    events: Sequence[WorkspaceEvent] | None = None
    evidence: Sequence[Evidence] | None = None
    results: Sequence[Result] | None = None


# ── reading a night ───────────────────────────────────────────────────────────


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _candidate(raw: Any) -> Candidate:
    if not isinstance(raw, dict) or not isinstance(raw.get("want_id"), str):
        raise PanelError("a considered want is not a record")
    return Candidate(raw["want_id"], str(raw.get("kind", "")), _text(raw.get("target_id")), _text(raw.get("action")),
                     _number(raw.get("strength")), str(raw.get("status", "")), _text(raw.get("reason")))


def _gates(raw: Any) -> tuple[GateMark, ...]:
    if not isinstance(raw, dict):
        raise PanelError("a night's gates are not a record")
    return tuple(GateMark(str(name), bool(gate.get("passed")) and not gate.get("skipped"), _number(gate.get("value")),
                          _number(gate.get("threshold")), bool(gate.get("skipped")))
                 for name, gate in sorted(raw.items()) if isinstance(gate, dict))


def _step(steps: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    return next((step for step in steps if step.get("step") == name), {})


def _confidence(receipt: Any, summary_choice: Mapping[str, Any], step: Mapping[str, Any]) -> float | None:
    """The chooser's confidence in this decision, wherever the record keeps it: the receipt's own
    field, the winning candidate's, or the step's. None for a record written before it had one."""
    chosen = next((c for c in summary_choice.get("considered") or []
                   if isinstance(c, dict) and c.get("status") == "won"), {})
    for raw in (receipt.get("confidence") if isinstance(receipt, dict) else None,
                chosen.get("confidence") if isinstance(chosen, dict) else None,
                step.get("confidence")):
        value = _number(raw)
        if value is not None:
            return value
    return None


def _story(night: date, summary: Mapping[str, Any], steps: Sequence[Mapping[str, Any]],
           raw: Mapping[str, Any] | None) -> Story | None:
    """The night's ``story.json``, else (none, or one an older schema wrote) its story told from its
    log; None when neither can be read."""
    if raw is not None:
        try:
            return story_from(raw)
        except StoryError:
            pass  # an older version, or a broken file: the log tells it again
    try:
        return tell(night, str(summary.get("status")), steps)
    except StoryError:
        return None


def night_record(
    summary: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], story: Mapping[str, Any] | None = None,
) -> NightRecord:
    """A night from its ``summary.json``, its ``log.jsonl`` steps (``night.nightlog``) and its
    ``story.json`` when it has one."""
    try:
        night = date.fromisoformat(str(summary["night"]))
    except (KeyError, ValueError):
        raise PanelError("a night's summary has no night") from None
    choices = summary.get("choices") or []
    if not isinstance(choices, list):
        raise PanelError("a night's choices are not a list")
    first = choices[0] if choices and isinstance(choices[0], dict) else {}  # step 4's choice
    choose = _step(steps, "choose")
    receipt = choose.get("receipt")
    try:
        sleep = sleep_mark(summary.get("sleep"))
    except SleepRecordError as error:
        raise PanelError(str(error)) from None
    return NightRecord(
        night=night, status=str(summary.get("status", "")), action=_text(first.get("action")),
        target_id=_text(first.get("target_id")),
        candidates=tuple(_candidate(c) for c in first.get("considered") or []),
        gates=_gates(summary.get("gates") or {}),
        outcome=_text(receipt.get("outcome")) if isinstance(receipt, dict) else None,
        note=_text(_step(steps, "note").get("outcome")), sleep=sleep, story=_story(night, summary, steps, story),
        confidence=_confidence(receipt, first, choose),
    )


def slept(record: NightRecord) -> dict[str, Any] | None:
    """How the night slept, as the page shows it; None for a night from before naps."""
    return None if record.sleep is None else record.sleep.view()


def numbered(nights: Sequence[NightRecord]) -> list[tuple[int, NightRecord]]:
    """Each night with its number, newest first; a night run again counts once, its last run."""
    last = {record.night: record for record in nights}
    return [(n, last[day]) for n, day in reversed(list(enumerate(sorted(last), start=1)))]


# ── the view ──────────────────────────────────────────────────────────────────


def _section(items: list[dict[str, Any]] | None, **extra: Any) -> dict[str, Any]:
    if items is None:
        return {"state": "unavailable", "items": [], **extra}
    return {"state": "ok" if items else "none", "items": items, **extra}


def _question(ws: Workspace, q: Question, held: HeldOutText) -> dict[str, Any]:
    hypotheses = ws.hypotheses_of(q.id)
    order = [h.id for h in hypotheses]
    return {
        "id": q.id, "kind": q.kind, "status": q.status, "subject": held.item(q.subject),
        "hypotheses": [{"id": h.id, "statement": held.item(h.statement), "credence": h.credence,
                        "catch_all": h.is_catch_all} for h in hypotheses],
        "predictions": [{"id": p.id, "claim": held.item(p.claim), "given": [p.p_true_given.get(h) for h in order],
                         "horizon": p.horizon.isoformat(), "outcome": p.outcome}
                        for p in ws.predictions if p.question_id == q.id],
    }


def _questions(ws: Workspace | None, held: HeldOutText) -> list[dict[str, Any]] | None:
    if ws is None:
        return None
    shown = sorted((q for q in ws.questions if q.status in SHOWN_STATUSES), key=lambda q: q.status == "parked")
    return [_question(ws, q, held) for q in shown]


def _targets(targets: Sequence[Target] | None, held: HeldOutText) -> list[dict[str, Any]] | None:
    if targets is None:
        return None
    open_ = sorted((t for t in targets if t.status == "open"),
                   key=lambda t: (t.kind != "commitment", t.check_after or datetime.max.replace(tzinfo=timezone.utc), t.id))
    return [{"id": t.id, "kind": t.kind, "label": held.item(t.label), "confidence": t.confidence,
             "due": None if t.check_after is None else t.check_after.isoformat()} for t in open_]


def candidate_view(c: Candidate, labels: Mapping[str, str], held: HeldOutText) -> dict[str, Any]:
    return {"kind": c.kind, "target": held.item(labels.get(c.target_id or "")), "action": c.action,
            "strength": c.strength, "status": c.status, "reason": c.reason}


def _wants(nights: Sequence[NightRecord] | None, labels: Mapping[str, str], held: HeldOutText) -> dict[str, Any]:
    if nights is None:
        return _section(None, night=None)
    if not nights:
        return _section([], night=None)
    number, latest = numbered(nights)[0]
    ranked = sorted(latest.candidates, key=lambda c: c.status != "won")
    return _section([candidate_view(c, labels, held) for c in ranked], night=number)


def _choices(nights: Sequence[NightRecord] | None, labels: Mapping[str, str], held: HeldOutText) -> list[dict[str, Any]] | None:
    if nights is None:
        return None
    return [{"night": n, "date": r.night.isoformat(), "status": r.status, "sleep": slept(r), "action": r.action,
             "target": held.item(labels.get(r.target_id or "")), "outcome": r.outcome, "note": r.note,
             "beat": [candidate_view(c, labels, held) for c in r.candidates if c.status != "won"]}
            for n, r in numbered(nights)]


def _gate_nights(nights: Sequence[NightRecord] | None) -> list[dict[str, Any]] | None:
    if nights is None:
        return None
    return [{"night": n, "date": r.night.isoformat(), "status": r.status,
             "gates": [{"gate": g.gate, "passed": g.passed, "value": g.value, "threshold": g.threshold,
                        "skipped": g.skipped} for g in r.gates]}
            for n, r in numbered(nights)]


def _decisions(nights: Sequence[NightRecord] | None) -> tuple[DecisionRecord, ...]:
    """Each night's choice for the timeline, dated by its night and clocked at the day's end."""
    if not nights:
        return ()
    return tuple(DecisionRecord(r.night, datetime.combine(r.night + timedelta(days=1), time.min, timezone.utc),
                                r.action, r.target_id, r.outcome, r.status, r.confidence)
                 for _, r in numbered(nights))


def _outcome(made: Expectation, returns: Sequence[datetime], as_of: datetime | None) -> str:
    """The stored outcome, else what the person's side shows by ``as_of``."""
    if made.outcome is not None:
        return made.outcome
    if any(made.made_at <= at < made.horizon and (as_of is None or at <= as_of) for at in returns):
        return "returned"
    return "waiting" if as_of is not None and as_of >= made.horizon else "open"


def _expectations(
    sources: MindSources, labels: Mapping[str, str], as_of: datetime | None, held: HeldOutText,
) -> list[dict[str, Any]] | None:
    if sources.expectations is None:
        return None
    return [{"id": e.id, "target": held.item(labels.get(e.target_id)), "made_at": e.made_at.isoformat(),
             "horizon": e.horizon.isoformat(), "p_needed": e.p_needed,
             "outcome": _outcome(e, sources.returns.get(e.target_id, ()), as_of)}
            for e in sorted(sources.expectations, key=lambda e: e.made_at, reverse=True)]


def _as_of(nights: Sequence[NightRecord] | None) -> datetime | None:
    if not nights:
        return None
    return datetime.combine(max(r.night for r in nights) + timedelta(days=1), time.min, timezone.utc)


def mind_view(sources: MindSources, held: HeldOutText) -> dict[str, Any]:
    """The panel's sections, ready to send: nothing in it carries held-out text."""
    labels = {t.id: t.label for t in sources.targets or ()}
    as_of = _as_of(sources.nights)
    return {
        "as_of": None if as_of is None else as_of.isoformat(),
        "questions": _section(_questions(sources.workspace, held)),
        "targets": _section(_targets(sources.targets, held)),
        "wants": _wants(sources.nights, labels, held),
        "choices": _section(_choices(sources.nights, labels, held)),
        "expectations": _section(_expectations(sources, labels, as_of, held)),
        "gates": _section(_gate_nights(sources.nights)),
        "story": story_view(None if sources.nights is None else [(n, r.story) for n, r in numbered(sources.nights)]),
        "timeline": timeline_view(TimelineSources(
            events=sources.events, evidence=sources.evidence, results=sources.results,
            decisions=_decisions(sources.nights), labels=labels, as_of=as_of), held),
    }
