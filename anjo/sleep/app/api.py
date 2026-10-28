# mypy: disallow-subclassing-any=False, disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" leaves fastapi and pydantic untyped here)
"""Anjo Sleep's dev API (Task 4), mounted only on the sleep host (``register.register_sleep``).

``POST /api/sleep/dev/sessions`` with ``{"cleared": true}`` opens the person's live chat session
and marks it cleared (Task 3's ``mark_cleared``), for the persona simulator's
``ChatTransport.send(session_id, text, cleared=True)``: the simulator opens a session here, posts
its messages to ``/api/chat/<session_id>/message`` and ends it with the existing chat route. Only
a flagged, synthetic person may clear a session; anyone else gets 403 and nothing is written.
``{"cleared": false}`` opens the session unchanged. ``GET /api/sleep/dev/sessions/<id>`` says
whether a session is cleared. A cleared open holds pending outreach back (``delivery.no_delivery``),
so a held note waits for the next uncleared open.

The person's own routes (Task 9), same user only (the caller is the account's own session; no route
takes another user's id) and on the same flagged-user check:

- ``GET /api/sleep/nights`` and ``GET /api/sleep/nights/<date>``: each finished night's status, gates
  by name, steps and the chooser's action code, never a conversation word;
- ``GET /api/sleep/what-changed?since=<date>``: plain lines for what changed since then;
- ``POST /api/sleep/undo``: the previous version serves again (409 when there is none).
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
from typing import Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool

from anjo.dashboard.auth import get_current_user_id
from anjo.sleep.app.cleared_sessions import NotSynthetic, cleared_sessions, is_synthetic, mark_cleared
from anjo.sleep.app.delivery import no_delivery
from anjo.sleep.app.flags import sleep_enabled_for, sleep_host
from anjo.sleep.app.ownership import person_results
from anjo.sleep.app.ownership import undo as undo_night
from anjo.sleep.app.ownership import what_changed as changes_since
from anjo.sleep.ownership.changes import ChangesError, NightView, night_views
from anjo.sleep.ownership.undo import NothingToUndo

PREFIX: Final = "/sleep"
DEFAULT_SINCE_DAYS: Final = 7
UNREADABLE: Final = "A night record can't be read."
NOTHING_TO_UNDO: Final = "There is no earlier version to go back to."
NOT_FLAGGED: Final = "Anjo Sleep isn't on for this account."
NOT_SYNTHETIC: Final = "Only a synthetic person's session can be cleared."

router = APIRouter(prefix=PREFIX)


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cleared: StrictBool = False


class SessionState(BaseModel):
    session_id: str
    cleared: bool


def flagged_user(user_id: str = Depends(get_current_user_id)) -> str:
    if not sleep_host():
        raise HTTPException(status_code=404, detail="Not Found")
    if not sleep_enabled_for(user_id):
        raise HTTPException(status_code=403, detail=NOT_FLAGGED)
    return user_id


@router.post("/dev/sessions")
def open_session(body: SessionRequest, user_id: str = Depends(flagged_user)) -> SessionState:
    from anjo.dashboard.session_store import get_or_create_session_with_status

    if body.cleared and not is_synthetic(user_id):
        raise HTTPException(status_code=403, detail=NOT_SYNTHETIC)  # before a session is made
    with no_delivery(user_id) if body.cleared else nullcontext():  # a note waits for an uncleared open
        session_id, _created = get_or_create_session_with_status(user_id)
    if body.cleared:
        try:
            mark_cleared(user_id, session_id, at=datetime.now(timezone.utc))
        except NotSynthetic:
            raise HTTPException(status_code=403, detail=NOT_SYNTHETIC) from None
    return SessionState(session_id=session_id, cleared=session_id in cleared_sessions(user_id))


@router.get("/dev/sessions/{session_id}")
def session_state(session_id: str, user_id: str = Depends(flagged_user)) -> SessionState:
    return SessionState(session_id=session_id, cleared=session_id in cleared_sessions(user_id))


# ── The person's own nights (Task 9) ──────────────────────────────────────────


class GateOut(BaseModel):
    gate: str
    passed: bool
    skipped: bool = False  # switched off that night: judged nothing, never passed


class StepOut(BaseModel):
    step: str
    status: str
    uses: str
    system: str | None


class NightOut(BaseModel):
    night: date
    status: str
    failed_at: str | None
    checkpoint: str | None
    gates: list[GateOut]
    steps: list[StepOut]
    action: str | None
    note: str | None


class ChangeOut(BaseModel):
    night: date
    kind: str
    line: str


class UndoOut(BaseModel):
    checkpoint: str
    night: date


def _night_out(view: NightView) -> NightOut:
    return NightOut(
        night=view.night, status=view.status, failed_at=view.failed_at, checkpoint=view.checkpoint,
        gates=[GateOut(gate=name, passed=passed, skipped=name in view.skipped) for name, passed in view.gates],
        steps=[StepOut(step=s.step, status=s.status, uses=s.uses, system=s.system) for s in view.steps],
        action=view.action, note=view.note,
    )


def _views(user_id: str) -> list[NightView]:
    try:
        return night_views(person_results(user_id))
    except ChangesError:
        raise HTTPException(status_code=500, detail=UNREADABLE) from None


@router.get("/nights")
def list_nights(user_id: str = Depends(flagged_user)) -> list[NightOut]:
    return [_night_out(view) for view in _views(user_id)]


@router.get("/nights/{night}")
def one_night(night: date, user_id: str = Depends(flagged_user)) -> NightOut:
    """The latest run on that date (a night run again is listed again in the list above)."""
    runs = [view for view in _views(user_id) if view.night == night]
    if not runs:
        raise HTTPException(status_code=404, detail="No night on that date.")
    return _night_out(runs[-1])


@router.get("/what-changed")
def what_changed(since: date | None = None, user_id: str = Depends(flagged_user)) -> list[ChangeOut]:
    start = since or datetime.now(timezone.utc).date() - timedelta(days=DEFAULT_SINCE_DAYS)
    try:
        lines = changes_since(user_id, start)
    except ChangesError:
        raise HTTPException(status_code=500, detail=UNREADABLE) from None
    return [ChangeOut(night=c.night, kind=c.kind, line=c.line) for c in lines]


@router.post("/undo")
def undo(user_id: str = Depends(flagged_user)) -> UndoOut:
    try:
        restored = undo_night(user_id)
    except NothingToUndo:
        raise HTTPException(status_code=409, detail=NOTHING_TO_UNDO) from None
    return UndoOut(checkpoint=restored.id, night=restored.night)
