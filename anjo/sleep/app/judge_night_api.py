# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" leaves fastapi untyped here)
"""The judges' night routes (2026-11-02), under ``/api/sleep/me``, Bearer only, judges only:

- ``POST /sleep-now``: ``{night_id}``; 429 at a cap or the judges' budget, 409 while a night runs (its
  body carries that night's ``night_id``, so the page can go back to watching it);
- ``GET /nights/{night_id}``: the night view's record (``judge_night.night_view``), polled every 2 s.

Anyone who isn't a judge (``judge_night.judge_ports``) gets 404 from both, and a judge asking for
another judge's night gets 404 too.
"""
from __future__ import annotations

from typing import Any, Final

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from anjo.dashboard.auth import get_current_user_id
from anjo.sleep.app import judge_night as jn

PREFIX: Final = "/sleep/me"
router = APIRouter(prefix=PREFIX)


def judge_user(user_id: str = Depends(get_current_user_id)) -> str:
    if not jn.judge_ports().is_judge(user_id):
        raise HTTPException(status_code=404, detail="Not Found")
    return user_id


@router.post("/sleep-now", response_model=None)
def sleep_now(user_id: str = Depends(judge_user)) -> dict[str, str] | JSONResponse:
    started = jn.start_night(user_id)
    if isinstance(started, jn.Refused):
        body: dict[str, Any] = {"detail": started.detail}
        if started.night_id is not None:
            body["night_id"] = started.night_id
        return JSONResponse(status_code=started.status, content=body)
    return {"night_id": started}


@router.get("/nights/{night_id}")
def one_night(night_id: str, user_id: str = Depends(judge_user)) -> dict[str, Any]:
    store = jn.job_store()
    jn.pump(store)
    job = store.get(night_id)
    if job is None or job["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="No such night.")
    return jn.night_view(job)
