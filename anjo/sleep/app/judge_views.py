"""What a judge's own pages read: ``GET /api/sleep/me/mind`` and ``GET /api/sleep/me/wake``.

**me/mind** is the live demo panel's view (``demo.panel.mind_view``) over the caller's own stores,
with an empty held-out filter: a judge has no persona file and no sealed tests, and the caller only
ever reads their own account. When MIND's last live pass (``live_mind``) is newer than their last
night, its wants replace the night's (``wants.live`` is true) and a ``live`` block says when it ran,
on which thinker, what notice opened and what the chooser would pick.

**me/wake** is what Anjo opens with after a night: ``opener`` is the held note waiting for their next
open (or the share MIND raised: ``demo_mind.return_payload``), else a line from the want her chooser
picked; ``self_prediction`` is her latest bet about her own practice (``mind.self_predict``): its
claim, the probability she gave it when she made it (her credences then, over P(claim | each
alternative)), and how it resolved; ``what_changed`` is the plain lines since their first day
(``ownership.what_changed``); ``before_after`` is the pair the night saved
(``judge_cohort.save_before_after``: one prompt from that night's collapse gate, answered by base
Glimmer and by the night's adapter), when there is one, with ``before_model``, ``after_model`` and
whether the gate ``promoted`` the adapter when the night recorded them.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Final

from anjo.sleep.app.judge_cohort import joined_at, load_before_after
from anjo.sleep.app.live_mind import load_record
from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.panel import Candidate, candidate_view, mind_view

logger = logging.getLogger(__name__)

NO_HELD_OUT: Final = HeldOutText(())
OPEN_STATES: Final = frozenset({"ok"})
# COPY-PENDING(kevin): the opener when no note waits, built from the want her chooser picked.
WANT_OPENERS: Final = {
    "inquire": "One thing I keep turning over: {target}. Can I ask you about it?",
    "hold_note": "I've been holding something about {target} for you.",
    "study": "While you were away I wanted to understand {target} better.",
}
WANT_OPENER_ANY: Final = "Something I want to settle with you: {target}."


def _sources(user_id: str) -> Any:
    from anjo.sleep.app import demo_mind

    return demo_mind._sources(user_id)  # the live page's own reads, over this account


def _labels(sources: Any) -> dict[str, str]:
    return {t.id: t.label for t in sources.targets or ()}


def _when(raw: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def _live_is_newer(record: dict[str, Any], sources: Any) -> bool:
    at = _when(record.get("at"))
    if at is None or record.get("state") != "ok" or not isinstance(record.get("choice"), dict):
        return False
    nights = sources.nights or ()
    if not nights:
        return True
    last = max(n.night for n in nights)
    return at >= datetime.combine(last + timedelta(days=1), datetime.min.time(), timezone.utc)


def _candidates(choice: dict[str, Any]) -> list[Candidate]:
    rows = choice.get("considered") or []
    return [Candidate(str(r.get("want_id", "")), str(r.get("kind", "")), r.get("target_id"), r.get("action"),
                      r.get("strength"), str(r.get("status", "")), r.get("reason"))
            for r in rows if isinstance(r, dict)]


def mind_payload(user_id: str) -> dict[str, Any]:
    sources = _sources(user_id)
    view = mind_view(sources, NO_HELD_OUT)
    record = load_record(user_id)
    if record is None:
        return {**view, "live": None}
    labels = _labels(sources)
    live: dict[str, Any] = {"at": record.get("at"), "state": record.get("state"), "thinker": record.get("thinker"),
                            "questions_opened": (record.get("notice") or {}).get("questions_opened", 0),
                            "new_items": (record.get("notice") or {}).get("new", 0)}
    if _live_is_newer(record, sources):
        choice = record["choice"]
        ranked = sorted(_candidates(choice), key=lambda c: c.status != "won")
        items = [candidate_view(c, labels, NO_HELD_OUT) for c in ranked]
        view["wants"] = {"state": "ok" if items else "none", "items": items, "night": None, "live": True}
        live["choice"] = {"action": choice.get("action"),
                          "target": NO_HELD_OUT.item(labels.get(choice.get("target_id") or ""))}
    return {**view, "live": live}


# ── me/wake ───────────────────────────────────────────────────────────────────


def _opener(user_id: str, sources: Any) -> tuple[str | None, str | None]:
    from anjo.sleep.app.demo_mind import return_payload

    back = return_payload(user_id, sources, NO_HELD_OUT)
    if back.get("state") in OPEN_STATES and isinstance(back.get("text"), str):
        return back["text"], str(back.get("kind") or "held_note")
    record = load_record(user_id) or {}
    choice = record.get("choice") if isinstance(record.get("choice"), dict) else None
    won = None
    if choice is not None and choice.get("target_id"):
        won = (choice.get("action"), choice["target_id"])
    elif sources.nights:
        latest = max(sources.nights, key=lambda n: n.night)
        if latest.target_id:
            won = (latest.action, latest.target_id)
    if won is None:
        return None, None
    label = _labels(sources).get(won[1])
    if not label:
        return None, None
    return WANT_OPENERS.get(str(won[0]), WANT_OPENER_ANY).format(target=label), "want"


def self_prediction(events: Any) -> dict[str, Any] | None:
    """Her latest bet about her own practice: claim, the probability she gave it before it resolved,
    and its outcome (None while pending)."""
    from anjo.sleep.mind import self_predict
    from anjo.sleep.mind.inquiry import kernel
    from anjo.sleep.mind.inquiry.steps import Resolved

    if not events:
        return None
    ws = kernel.fold(events)
    mine = [p for p in ws.predictions if self_predict.is_self_prediction(ws, p)]
    if not mine:
        return None
    bet = mine[-1]
    resolved_at = next((n for n, e in enumerate(events)
                        if isinstance(e.step.entry, Resolved) and e.step.entry.prediction_id == bet.id), None)
    before = ws if resolved_at is None else kernel.fold(events[:resolved_at])
    p = sum(h.credence * float(bet.p_true_given.get(h.id, 0.0)) for h in before.hypotheses_of(bet.question_id))
    return {"claim": bet.claim, "p": round(p, 3), "outcome": bet.outcome}


def _what_changed(user_id: str) -> list[dict[str, Any]]:
    from anjo.sleep.app.ownership import what_changed
    from anjo.sleep.ownership.changes import ChangesError

    joined = joined_at(user_id)
    since = joined.date() if joined is not None else date.today() - timedelta(days=7)
    try:
        return [{"night": c.night.isoformat(), "kind": c.kind, "line": c.line} for c in what_changed(user_id, since)]
    except ChangesError as error:
        logger.error("a judge's changes can't be read: %s", type(error).__name__)
        return []


def wake_payload(user_id: str) -> dict[str, Any]:
    sources = _sources(user_id)
    opener, kind = _opener(user_id, sources)
    out: dict[str, Any] = {
        "opener": opener, "opener_kind": kind,
        "self_prediction": self_prediction(sources.events),
        "what_changed": _what_changed(user_id),
    }
    pair = load_before_after(user_id)
    if pair is not None:
        out["before_after"] = {"question": pair["question"], "before": pair["before"], "after": pair["after"],
                               **{k: pair[k] for k in ("before_model", "after_model", "promoted") if k in pair}}
    return out
