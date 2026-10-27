# mypy: disallow-subclassing-any=False, disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" leaves fastapi and pydantic untyped here)
"""Anjo Sleep's dev API (Task 4), mounted only on the sleep host (``register.register_sleep``).

``POST /api/sleep/dev/sessions`` with ``{"cleared": true}`` opens the person's live chat session
and marks it cleared (Task 3's ``mark_cleared``), for the persona simulator's
``ChatTransport.send(session_id, text, cleared=True)``: the simulator opens a session here, posts
its messages to ``/api/chat/<session_id>/message`` and ends it with the existing chat route. Only
a flagged, synthetic person may clear a session; anyone else gets 403 and nothing is written.
``{"cleared": false}`` opens the session unchanged. ``GET /api/sleep/dev/sessions/<id>`` says
whether a session is cleared.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool

from anjo.dashboard.auth import get_current_user_id
from anjo.sleep.app.cleared_sessions import NotSynthetic, cleared_sessions, is_synthetic, mark_cleared
from anjo.sleep.app.flags import sleep_enabled_for, sleep_host

PREFIX: Final = "/sleep/dev"
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


@router.post("/sessions")
def open_session(body: SessionRequest, user_id: str = Depends(flagged_user)) -> SessionState:
    from anjo.dashboard.session_store import get_or_create_session_with_status

    if body.cleared and not is_synthetic(user_id):
        raise HTTPException(status_code=403, detail=NOT_SYNTHETIC)  # before a session is made
    session_id, _created = get_or_create_session_with_status(user_id)
    if body.cleared:
        try:
            mark_cleared(user_id, session_id, at=datetime.now(timezone.utc))
        except NotSynthetic:
            raise HTTPException(status_code=403, detail=NOT_SYNTHETIC) from None
    return SessionState(session_id=session_id, cleared=session_id in cleared_sessions(user_id))


@router.get("/sessions/{session_id}")
def session_state(session_id: str, user_id: str = Depends(flagged_user)) -> SessionState:
    return SessionState(session_id=session_id, cleared=session_id in cleared_sessions(user_id))
