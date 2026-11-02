"""Her own line of thought, shown to the person (the judge product's thought panel).

``thought_view`` turns her workspace log into plain JSON:

- ``hunches``: her open bets, with her odds now and when each settles;
- ``settled``: the bets that resolved, with her odds when she made them and whether she was right;
- ``scorecard``: her record on those bets (hit rate, Brier score, calibration bins). Censored bets
  are counted apart, and ``enough`` stays false until ``ENOUGH_SCORED`` bets have settled;
- ``question``: her one most useful question for the person, a close call between her top two
  alternatives first;
- ``morning``: what settled, what she newly bet on, where her lead moved and what she concluded,
  since the person's last visit.

It is pure, with no model and no clock: ``now`` is the night clock, as everywhere in MIND. The log
is replayed once through ``kernel.apply`` with every logged digest checked, so a tampered log is
refused, and a bet settled before a wipe still counts. Steps logged after ``now`` are checked too,
but left out of the view.

Her credences are her own heuristic estimates, never calibrated odds. ``CREDENCE_NOTE`` travels
with every view so the page can say so beside any percentage.

**The person's marks** (``dispute``, ``hide``) never edit her log, which is append-only and
digest-chained. The caller stores them in the person's own marks file and passes them back in.
``dispute`` keeps a hunch on screen, flagged. ``hide`` removes an item, or a question with all its
bets, from everything this view returns. It doesn't erase the item from her workspace, so the page
says "Hide this", not "Forget". Her scorecard counts hidden bets anyway: a person hides text from
the view, not her mistakes from her record.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Literal, cast, get_args

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import (
    FULL_RULES,
    LIVE_STATUSES,
    Applied,
    Hypothesis,
    Prediction,
    Resolved,
    RunStart,
    Workspace,
    WorkspaceEvent,
    catch_all_id,
)

VERSION: Final = 1
CREDENCE_NOTE: Final = "Her own estimates, not calibrated odds."
SOMETHING_ELSE: Final = "Something else"
MAX_HUNCHES: Final = 5
MAX_SETTLED: Final = 10
MAX_CHOICES: Final = 3  # her alternatives offered, before "Something else"
ENOUGH_SCORED: Final = 10  # settled bets before her odds are worth judging
CLOSE_CALL: Final = 0.15  # her top two this close: neither leads (wants.CONTRADICTION_GAP)
SHIFT_MIN: Final = 0.15  # a lead that moved this much since the last visit is news
BIN_EDGES: Final = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
MAX_NOTE: Final = 500  # characters of a dispute's note

MarkKind = Literal["dispute", "hide"]
Text = Callable[[str], Mapping[str, Any]]


def plain(text: str) -> dict[str, Any]:
    """The default text filter: everything shown as it is (a judge's own state)."""
    return {"text": text, "withheld": False}


def _words(shown: Mapping[str, Any]) -> str | None:
    """Her words as the filter let them through, or ``None`` when withheld."""
    words = shown.get("text")
    return None if shown.get("withheld") or not isinstance(words, str) or not words else words


def _day(at: datetime) -> str:
    return f"{at:%b} {at.day}"


def _pct(p: float) -> str:
    return f"{round(p * 100)}%"


# ── The person's marks ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mark:
    """One dispute or hide by the person, on a prediction id or a question id."""

    kind: MarkKind
    item_id: str
    at: datetime
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "item_id": self.item_id, "at": self.at.isoformat()}
        return out if self.note is None else {**out, "note": self.note}

    @classmethod
    def from_json(cls, raw: Any) -> Mark:
        """A stored mark, checked: a bad record is a ``ValueError``, never a silent skip."""
        if not isinstance(raw, Mapping):
            raise ValueError("a mark is a JSON object")
        kind, item_id, at, note = raw.get("kind"), raw.get("item_id"), raw.get("at"), raw.get("note")
        if kind not in get_args(MarkKind):
            raise ValueError(f"unknown mark kind {kind!r}")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("a mark names an item id")
        if not isinstance(at, str):
            raise ValueError("a mark's time is an ISO 8601 string")
        when = datetime.fromisoformat(at)
        if when.tzinfo is None:
            raise ValueError("a mark's time carries its zone")
        if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE):
            raise ValueError(f"a note is text of at most {MAX_NOTE} characters")
        return cls(cast(MarkKind, kind), item_id, when, note)


def dispute(events: Sequence[WorkspaceEvent], item_id: str, *, at: datetime, note: str | None = None) -> Mark:
    """The person says a hunch or question is wrong: it stays on screen, flagged."""
    return _mark(events, "dispute", item_id, at, note)


def hide(events: Sequence[WorkspaceEvent], item_id: str, *, at: datetime) -> Mark:
    """The person hides a hunch, or a question with all its bets, from this view."""
    return _mark(events, "hide", item_id, at, None)


def _mark(events: Sequence[WorkspaceEvent], kind: MarkKind, item_id: str, at: datetime, note: str | None) -> Mark:
    if item_id not in _replay(events, None, None).known:
        raise ValueError(f"no hunch or question {item_id!r} in her log")
    return Mark.from_json(Mark(kind, item_id, at, note).to_json())


# ── Replaying her log ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Bet:
    id: str
    question_id: str
    claim: str
    p_at_bet: float


@dataclass(frozen=True)
class _Settled:
    bet: _Bet
    outcome: Literal["true", "false"]
    at: datetime


@dataclass(frozen=True)
class _Found:
    id: str
    question_id: str
    kind: str
    statement: str
    p: float
    at: datetime


@dataclass(frozen=True)
class _Replay:
    final: Workspace
    settled: tuple[_Settled, ...]
    censored: int
    found: tuple[_Found, ...]
    then: Workspace | None  # her state at ``since``
    known: frozenset[str]  # every prediction and question id the log ever held


def implied(ws: Workspace, prediction: Prediction) -> float:
    """Her probability that the claim comes true: each alternative's credence times the claim's
    probability under it."""
    given = prediction.p_true_given
    fallback = given.get(catch_all_id(prediction.question_id), 0.5)
    return math.fsum(h.credence * given.get(h.id, fallback) for h in ws.hypotheses_of(prediction.question_id))


def _replay(events: Sequence[WorkspaceEvent], since: datetime | None, now: datetime | None) -> _Replay:
    """Every step replayed with both its digests checked; only the steps up to ``now`` (all of them
    when ``None``) enter the view."""
    ws, current, rules = Workspace(), Workspace(), FULL_RULES
    bets: dict[str, _Bet] = {}
    settled: list[_Settled] = []
    found: list[_Found] = []
    censored, then = 0, (Workspace() if since is not None else None)
    known: set[str] = set()
    for event in events:
        step = event.step
        if isinstance(step.entry, RunStart):
            rules = step.entry.rules
        result = kernel.apply(ws, step, rules=rules)
        if not isinstance(result, Applied) or (result.event.before, result.event.after) != (event.before, event.after):
            raise ValueError(f"step {step.seq} doesn't replay to the digests her log holds")
        before, ws = ws, result.ws
        if now is not None and step.at > now:
            continue  # checked, but after the night clock: not in this view
        for p in ws.predictions:
            if p.id not in bets:
                bets[p.id] = _Bet(p.id, p.question_id, p.claim, implied(ws, p))
        known.update(bets, (q.id for q in ws.questions))
        entry = step.entry
        if isinstance(entry, Resolved) and entry.outcome in ("true", "false"):
            settled.append(_Settled(bets[entry.prediction_id], entry.outcome, step.at))
        censored += isinstance(entry, Resolved) and entry.outcome == "censored"
        found.extend(_new_findings(before, ws, step.at, step.seq))
        if since is not None and step.at <= since:
            then = ws
        current = ws
    return _Replay(current, tuple(settled), censored, tuple(found), then, frozenset(known))


def _new_findings(before: Workspace, after: Workspace, at: datetime, seq: int) -> list[_Found]:
    """Findings this step reached or reworded, each id naming the finding and the step."""
    out = []
    for f in after.findings:
        old = before.finding_of(f.question_id)
        if old is None or (old.kind, old.statement) != (f.kind, f.statement):
            out.append(_Found(f"{f.id}@{seq}", f.question_id, f.kind, f.statement, f.p, at))
    return out


def _latest(found: Sequence[_Found]) -> list[_Found]:
    """One entry per question, its last words, in the order she reached them."""
    last = {f.question_id: f for f in found}
    return sorted(last.values(), key=lambda f: found.index(f))


# ── The view ──────────────────────────────────────────────────────────────────


def thought_view(
    events: Sequence[WorkspaceEvent], marks: Sequence[Mark], *, now: datetime,
    since: datetime | None = None, text: Text = plain,
) -> dict[str, Any]:
    """Her line of thought as plain JSON (module docstring). ``ValueError`` on a log whose
    digests don't replay."""
    r = _replay(events, since, now)
    hidden = frozenset(m.item_id for m in marks if m.kind == "hide" and m.item_id in r.known)
    disputed = frozenset(m.item_id for m in marks if m.kind == "dispute")
    subjects = {q.id: q.subject for q in r.final.questions}
    shown = [p for p in r.final.predictions if p.pending and not _is_hidden(p.id, p.question_id, hidden)]
    shown.sort(key=lambda p: (p.horizon, p.id))
    fresh = [p.id for p in shown if since is not None and since < p.committed_at <= now]
    listed = [p for n, p in enumerate(shown) if n < MAX_HUNCHES or p.id in fresh]  # a new one is never cut
    hunches = [_hunch(r.final, p, subjects, disputed, now, text) for p in listed]
    settled = [s for s in reversed(r.settled) if not _is_hidden(s.bet.id, s.bet.question_id, hidden)]
    return {
        "version": VERSION,
        "as_of": now.isoformat(),
        "credence_note": CREDENCE_NOTE,
        "hunches": hunches,
        "settled": [_settled(s, text) for s in settled[:MAX_SETTLED]],
        "scorecard": _scorecard(r.settled, r.censored),
        "question": _question(r.final, hidden, disputed, text),
        "morning": None if since is None else _morning(r, since, now, hidden, fresh, text),
        "hidden": len(hidden),
    }


def _is_hidden(prediction_id: str, question_id: str, hidden: frozenset[str]) -> bool:
    return prediction_id in hidden or question_id in hidden


def odds(p: float) -> str:
    """Her probability in plain words, to the nearest tenth."""
    tenths = round(p * 10)
    if tenths <= 0:
        return "less than 1 in 10"
    if tenths >= 10:
        return "more than 9 in 10"
    return f"about {tenths} in 10"


def _hunch(ws: Workspace, p: Prediction, subjects: Mapping[str, str], disputed: frozenset[str],
           now: datetime, text: Text) -> dict[str, Any]:
    chance, claim = implied(ws, p), text(p.claim)
    words = _words(claim)
    said = odds(chance).capitalize()
    line = f"{said}: {words}." if words else f"{said}: a hunch kept back here."
    due = p.horizon.astimezone(now.tzinfo)  # dates on the night clock's own zone
    return {
        "id": p.id,
        "question_id": p.question_id,
        "plain": f"{line} I'll know by {_day(due)}.",
        "exact": words or "",
        "about": text(subjects.get(p.question_id, "")),
        "claim": claim,
        "p": chance,
        "odds": odds(chance),
        "due": due.date().isoformat(),
        "due_in_days": (due.date() - now.date()).days,
        "disputed": p.id in disputed or p.question_id in disputed,
    }


_CALLED: Final = {"right": "so I called it", "wrong": "so I got this one wrong", "no_call": "a coin flip"}


def _settled(s: _Settled, text: Text) -> dict[str, Any]:
    claim = text(s.bet.claim)
    words = _words(claim) or "One of my bets"
    happened = "it happened" if s.outcome == "true" else "it didn't happen"
    verdict = _verdict(s.bet.p_at_bet, s.outcome)
    return {
        "id": s.bet.id,
        "question_id": s.bet.question_id,
        "plain": f"{words}: {happened}. I had it at {odds(s.bet.p_at_bet)}, {_CALLED[verdict]}.",
        "exact": _words(claim) or "",
        "claim": claim,
        "p_at_bet": s.bet.p_at_bet,
        "outcome": s.outcome,
        "verdict": verdict,
        "settled_at": s.at.isoformat(),
    }


def _verdict(p: float, outcome: str) -> Literal["right", "wrong", "no_call"]:
    if math.isclose(p, 0.5):
        return "no_call"
    return "right" if (p > 0.5) == (outcome == "true") else "wrong"


def _scorecard(settled: Sequence[_Settled], censored: int) -> dict[str, Any]:
    scored = len(settled)
    right = sum(_verdict(s.bet.p_at_bet, s.outcome) == "right" for s in settled)
    brier = math.fsum((s.bet.p_at_bet - (s.outcome == "true")) ** 2 for s in settled) / scored if scored else None
    bins = []
    for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:], strict=False):
        inside = [s for s in settled if (lo < s.bet.p_at_bet <= hi) or (lo == 0.0 and s.bet.p_at_bet == 0.0)]
        if inside:
            bins.append({"from": lo, "to": hi, "n": len(inside), "came_true": sum(s.outcome == "true" for s in inside)})
    return {
        "scored": scored,
        "right": right,
        "hit_rate": right / scored if scored else None,
        "brier": brier,
        "enough": scored >= ENOUGH_SCORED,
        "bins": bins,
        "ran_out_of_time": censored,
    }


def _ranked(ws: Workspace, question_id: str) -> list[Hypothesis]:
    return sorted((h for h in ws.hypotheses_of(question_id) if not h.is_catch_all), key=lambda h: -h.credence)


def _alternative(h: Hypothesis, text: Text) -> Mapping[str, Any]:
    return plain(SOMETHING_ELSE) if h.is_catch_all else text(h.statement)


def _question(ws: Workspace, hidden: frozenset[str], disputed: frozenset[str], text: Text) -> dict[str, Any] | None:
    """Her close call with the smallest gap, else her least settled question with no leader."""
    best: tuple[tuple[int, float], dict[str, Any]] | None = None
    for q in ws.questions:
        ranked = _ranked(ws, q.id)
        if q.status not in LIVE_STATUSES or q.id in hidden or len(ranked) < 2:
            continue
        gap = ranked[0].credence - ranked[1].credence
        why = "close_call" if gap <= CLOSE_CALL else "uncertain" if ranked[0].credence < 0.5 else None
        if why is None:
            continue
        key = (0, gap) if why == "close_call" else (1, ranked[0].credence)
        if best is None or key < best[0]:
            best = (key, _ask(ws, q.id, q.subject, why, ranked, q.id in disputed, text))
    return None if best is None else best[1]


_WHY: Final = {"close_call": "I'm torn between two answers.", "uncertain": "I don't have a good guess yet."}


def _ask(ws: Workspace, question_id: str, subject: str, why: str, ranked: Sequence[Hypothesis],
         disputed: bool, text: Text) -> dict[str, Any]:
    ask = text(subject)
    words = _words(ask)
    lead = f"Quick question: {words}" if words else "I have a question I'm keeping back here."
    return {"id": question_id, "question_id": question_id, "plain": f"{lead} {_WHY[why]}", "exact": words or "",
            "ask": ask, "why": why, "choices": _choices(ws, question_id, ranked, text), "disputed": disputed}


def _choices(ws: Workspace, question_id: str, ranked: Sequence[Hypothesis], text: Text) -> list[dict[str, Any]]:
    rest = [h for h in ws.hypotheses_of(question_id) if h.is_catch_all]
    return [{"id": h.id, "text": _alternative(h, text), "credence": h.credence}
            for h in [*ranked[:MAX_CHOICES], *rest]]


def _morning(r: _Replay, since: datetime, now: datetime, hidden: frozenset[str],
             fresh: Sequence[str], text: Text) -> dict[str, Any]:
    window = [s for s in r.settled if since < s.at <= now and not _is_hidden(s.bet.id, s.bet.question_id, hidden)]
    found = _latest([f for f in r.found if since < f.at <= now and f.question_id not in hidden])
    morning = {
        "since": since.isoformat(),
        "settled": [s.bet.id for s in window],
        "new_hunches": list(fresh),
        "shifts": _shifts(r.then or Workspace(), r.final, hidden, text),
        "findings": [_finding(f, text) for f in found],
    }
    empty = not any(morning[k] for k in ("settled", "new_hunches", "shifts", "findings"))
    return {**morning, "empty": empty, "plain": _summary(morning, window)}


_FOUND: Final = {"resolution": "I worked it out: {}.", "suspicion": "I suspect {}.",
                 "retraction": "I take back what I thought: {}."}
_FOUND_KEPT: Final = {"resolution": "I worked something out.", "suspicion": "I have a suspicion.",
                      "retraction": "I took back something I thought."}


def _finding(f: _Found, text: Text) -> dict[str, Any]:
    statement = text(f.statement)
    words = _words(statement)
    line = _FOUND.get(f.kind, "{}.").format(words) if words else _FOUND_KEPT.get(f.kind, "I noted something.")
    return {"id": f.id, "question_id": f.question_id, "kind": f.kind, "plain": line, "exact": words or "",
            "statement": statement, "p": f.p}


def _summary(morning: Mapping[str, Any], window: Sequence[_Settled]) -> str:
    if not any(morning[k] for k in ("settled", "new_hunches", "shifts", "findings")):
        return "Nothing new since you were last here."
    right = sum(_verdict(s.bet.p_at_bet, s.outcome) == "right" for s in window)
    parts = []
    if window:
        parts.append(f"{len(window)} of my bets settled ({right} right)")
    if morning["new_hunches"]:
        parts.append(f"{len(morning['new_hunches'])} new hunch{'es' if len(morning['new_hunches']) > 1 else ''}")
    if morning["shifts"]:
        parts.append(f"I changed my mind {len(morning['shifts'])} time{'s' if len(morning['shifts']) > 1 else ''}")
    if morning["findings"]:
        parts.append(f"{len(morning['findings'])} conclusion{'s' if len(morning['findings']) > 1 else ''}")
    return "Since you were last here: " + ", ".join(parts) + "."


def _shifts(then: Workspace, now: Workspace, hidden: frozenset[str], text: Text) -> list[dict[str, Any]]:
    """Each question she held at the last visit whose lead changed, or moved by ``SHIFT_MIN``."""
    out = []
    held = {q.id for q in then.questions}
    for q in now.questions:
        alternatives = now.hypotheses_of(q.id)
        if q.id not in held or q.id in hidden or not alternatives:
            continue
        before = {h.id: h.credence for h in then.hypotheses_of(q.id)}
        lead = max(alternatives, key=lambda h: h.credence)
        lead_before = max(before, key=lambda k: before[k]) if before else None
        moved = abs(lead.credence - before.get(lead.id, 0.0))
        if lead.id != lead_before or moved >= SHIFT_MIN:
            out.append(_shift(q.id, q.subject, lead, before.get(lead.id, 0.0), lead.id != lead_before, text))
    return out


def _shift(question_id: str, subject: str, lead: Hypothesis, was: float, changed: bool, text: Text) -> dict[str, Any]:
    about, leading = text(subject), _alternative(lead, text)
    on, now = _words(about), _words(leading)
    verb = "I changed my mind" if changed else f"I'm surer now: {_pct(was)} to {_pct(lead.credence)}"
    line = f"{verb} on \u201c{on}\u201d. I now lean to \u201c{now}\u201d." if on and now else f"{verb} on a question kept back here."
    return {"id": f"shift:{question_id}", "question_id": question_id, "plain": line, "exact": now or "",
            "about": about, "now_leading": leading, "from": was, "to": lead.credence}
