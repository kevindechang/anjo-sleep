# mypy: disallow-subclassing-any=False, disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" leaves fastapi and pydantic untyped here)
"""The judge product's API on the sleep host (Meta hackathon judging, Nov 2026), mounted by
``register.register_sleep``.

Public (``PUBLIC_PREFIX``), each 404 unless this is the sleep host and ``ANJO_SLEEP_JUDGE_CODE`` is set:

- ``POST /api/sleep/judge/start`` ``{code}`` -> ``{token, user_id}``: a fresh account in the judge
  cohort (``judge_cohort``) and a Bearer token the app's auth accepts. At most
  ``ANJO_SLEEP_JUDGE_START_PER_IP`` attempts (default 20) from one address a day, right or wrong,
  counted before the code is compared (in constant time). The account is made with the app's own
  helpers: ``register_user`` with an unroutable address and a random password nobody keeps (the
  token lives in that browser), its email marked verified (no inbox behind it), the starter credits
  every signup gets, the AI disclosure in force recorded as accepted (the judge landing carries
  it), and the Glimmer talker chosen. No age record is written: the cohort skips the age step
  (owner decision, 2026-11-02), and the chat's crisis floor and age lock run as for everyone. It also starts
  Glimmer's wake (``judge_talker``).
- ``POST /api/sleep/judge/prewake`` -> the talker's status: "Grow your own" was clicked, so Glimmer
  may start waking while the code is typed. Debounced with every other wake, at most
  ``ANJO_SLEEP_JUDGE_PREWAKE_PER_IP`` (6) from one address and ``ANJO_SLEEP_JUDGE_PREWAKE_PER_DAY`` (60)
  in all a day; it holds her warm ``ANJO_SLEEP_JUDGE_PREWAKE_MINUTES`` (5) unless a judge arrives.

The caller's own (Bearer; 404 off the sleep host, 401 without a token, 403 outside the cohort). Each
one counts as the judge being present (``judge_talker.arrive``), so a cold Glimmer starts waking:

- ``GET /api/sleep/me/mind``: their MIND state (``judge_views.mind_payload``);
- ``GET /api/sleep/me/wake``: what Anjo opens with after a night (``judge_views.wake_payload``);
- ``POST /api/sleep/me/undo``: the previous version serves again (``api.undo``'s; 409 when none);
- ``GET /api/sleep/me/talker``: ``{state: warm|waking|cold, eta_s, model, reason, fallback_via}`` (``model``
  is ``glimmer`` or ``spark-fallback``; ``fallback_via`` names what answers under that label);
- ``POST /api/sleep/me/heartbeat``: the page's every 60 s while its tab shows; returns the talker's
  status. With no heartbeat (or any other judge call) from any judge for
  ``ANJO_SLEEP_JUDGE_IDLE_MINUTES``, Glimmer's GPU stops (``judge_talker``);
- ``GET /api/sleep/me/thought``, ``POST /api/sleep/me/thought/{id}/dispute`` (``{note?}``, at most
  ``thought.MAX_NOTE`` = 500 characters, 422 over it) and ``POST /api/sleep/me/thought/{id}/hide``:
  her own line of thought and the person's marks on it (``judge_thought``; 404 for an id not in her
  log, 500 for a log that can't be read whole or doesn't replay to its digests).

``TurnWatch`` sees each judge's chat message (``POST /api/chat/{sid}/message``): presence as it
starts, and MIND's live pass (``live_mind``) once it has answered.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import secrets
from typing import Any, Final

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from anjo.dashboard.auth import get_current_user_id
from anjo.sleep.app.flags import sleep_host
from anjo.sleep.app.judge_cohort import (
    CODE_ENV,
    MAX_ACCOUNTS_ENV,
    PREWAKE_PER_DAY_ENV,
    PREWAKE_PER_IP_ENV,
    START_PER_IP_ENV,
    CapReached,
    Check,
    add_judge,
    counters,
    is_judge,
    judge_ids,
    setting,
)
from anjo.sleep.mind.thought import MAX_NOTE

logger = logging.getLogger(__name__)

PREFIX: Final = "/sleep"
PUBLIC_PREFIX: Final = "/api/sleep/judge"
KEY_INFO: Final = b"anjo-sleep-judge-limits-v1"
EMAIL_DOMAIN: Final = "judges.invalid"  # RFC 2606: never routable, so no mail is ever sent to one
TALKER: Final = "gemma"  # the talker choice the sleep route serves Glimmer under
WRONG_CODE: Final = "That code isn't right."
NOT_A_JUDGE: Final = "This account isn't a judge account."
REFUSED: Final = {
    "judges_full": "The judge accounts are all taken.",
    "start_spent_today": "Too many tries from here today.",
    "prewake_spent_today": "Too many tries from here today.",
    "counters_unreadable": "Try again in a moment.",
}

router = APIRouter(prefix=PREFIX)


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1, max_length=200)


class StartOut(BaseModel):
    token: str
    user_id: str


def _code() -> str | None:
    code = os.environ.get(CODE_ENV, "").strip()
    return code or None


def _product_on() -> str:
    """The judge code, or 404: the product is off without it, and everywhere but the sleep host."""
    code = _code()
    if not sleep_host() or code is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return code


def _client(request: Request) -> str:
    from anjo.core.crypto import _get_key
    from anjo.sleep.demo.visits import client_key

    key: str = client_key(request.client.host if request.client else "", bytes(_get_key(KEY_INFO)))
    return key


def _refuse(refused: CapReached) -> HTTPException:
    return HTTPException(status_code=refused.status, detail=REFUSED.get(refused.reason, refused.reason))


def _talker() -> Any:
    from anjo.sleep.app.judge_talker import talker

    return talker()


def _arrive(**lease: float) -> None:
    from anjo.core.best_effort import best_effort

    with best_effort("judge presence", type_only=True):  # presence never fails the judge's request
        _talker().arrive(**lease)


def _make_account() -> str:
    """A fresh account through the app's own helpers (see the module's note); its id."""
    from anjo.core.ai_consent import CURRENT_VERSION, record_ai_consent
    from anjo.core.credits import grant_initial_credits
    from anjo.dashboard.auth import force_verify_email, register_user

    user: dict[str, Any] | None = None
    for _ in range(2):  # a clash on a random address is all but impossible; once more covers it
        user, _err = register_user(secrets.token_urlsafe(32), f"judge-{secrets.token_hex(8)}@{EMAIL_DOMAIN}")
        if user is not None:
            break
    if user is None:
        raise HTTPException(status_code=500, detail="The account couldn't be made. Try again.")
    user_id = str(user["user_id"])
    force_verify_email(user_id)
    grant_initial_credits(user_id)
    record_ai_consent(user_id, CURRENT_VERSION)
    return user_id


def _choose_glimmer(user_id: str) -> None:
    from anjo.core.talker_choice import set_talker_choice

    try:
        set_talker_choice(user_id, TALKER)
    except ValueError as refused:  # the chat route's own fallback answers until it is offered
        logger.warning("a judge's Glimmer talker wasn't set: %s", refused)


@router.post("/judge/start")
def judge_start(body: StartRequest, request: Request) -> StartOut:
    code = _product_on()
    try:
        counters().claim(None, Check("start", _client(request), setting(START_PER_IP_ENV), "start_spent_today"))
    except CapReached as refused:
        raise _refuse(refused) from None
    if not hmac.compare_digest(body.code.strip().encode(), code.encode()):
        raise HTTPException(status_code=403, detail=WRONG_CODE)
    if len(judge_ids()) >= setting(MAX_ACCOUNTS_ENV):
        raise _refuse(CapReached(429, "judges_full"))
    from anjo.dashboard.auth import make_token

    user_id = _make_account()
    try:
        add_judge(user_id)
    except CapReached as refused:
        raise _refuse(refused) from None
    _choose_glimmer(user_id)
    _arrive()
    return StartOut(token=make_token(user_id), user_id=user_id)


@router.post("/judge/prewake")
def judge_prewake(request: Request) -> dict[str, Any]:
    _product_on()
    try:
        counters().claim(None, Check("prewake", None, setting(PREWAKE_PER_DAY_ENV), "prewake_spent_today"),
                         Check("prewake", _client(request), setting(PREWAKE_PER_IP_ENV), "prewake_spent_today"))
    except CapReached as refused:
        raise _refuse(refused) from None
    talker = _talker()
    _arrive(lease_s=talker.settings.prewake_s)
    status: dict[str, Any] = talker.status().view()
    return status


# ── The caller's own ──────────────────────────────────────────────────────────


def judge_user(user_id: str = Depends(get_current_user_id)) -> str:
    if not sleep_host():
        raise HTTPException(status_code=404, detail="Not Found")
    if not is_judge(user_id):
        raise HTTPException(status_code=403, detail=NOT_A_JUDGE)
    _arrive()
    return user_id


@router.get("/me/mind")
def my_mind(user_id: str = Depends(judge_user)) -> dict[str, Any]:
    from anjo.sleep.app.judge_views import mind_payload
    from anjo.sleep.demo.panel import PanelError

    try:
        payload: dict[str, Any] = mind_payload(user_id)
    except PanelError:
        raise HTTPException(status_code=500, detail="Your MIND record can't be read.") from None
    return payload


@router.get("/me/wake")
def my_wake(user_id: str = Depends(judge_user)) -> dict[str, Any]:
    from anjo.sleep.app.judge_views import wake_payload
    from anjo.sleep.demo.panel import PanelError

    try:
        payload: dict[str, Any] = wake_payload(user_id)
    except PanelError:
        raise HTTPException(status_code=500, detail="Your night record can't be read.") from None
    return payload


@router.post("/me/undo")
def my_undo(user_id: str = Depends(judge_user)) -> Any:
    from anjo.sleep.app import api

    return api.undo(user_id)


@router.get("/me/talker")
def my_talker(user_id: str = Depends(judge_user)) -> dict[str, Any]:
    status: dict[str, Any] = _talker().status().view()
    return status


@router.post("/me/heartbeat")
def my_heartbeat(user_id: str = Depends(judge_user)) -> dict[str, Any]:
    status: dict[str, Any] = _talker().status().view()
    return status


# ── Her own line of thought (judge_thought) ──────────────────────────────────

NOT_SHOWN: Final = "There's nothing with that id to change."
UNREADABLE: Final = "Her line of thought can't be read right now."


TOO_LONG: Final = f"A note can be at most {MAX_NOTE} characters."


class DisputeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(default=None, max_length=MAX_NOTE)  # the engine's own limit: longer is a 422


def _thought(run: Any) -> dict[str, Any]:
    from anjo.sleep.app.judge_thought import LogUnreadable, NoteTooLong, UnknownItem

    try:
        done: dict[str, Any] = run()
    except LogUnreadable:
        raise HTTPException(status_code=500, detail=UNREADABLE) from None
    except NoteTooLong:
        raise HTTPException(status_code=422, detail=TOO_LONG) from None
    except UnknownItem:
        raise HTTPException(status_code=404, detail=NOT_SHOWN) from None
    return done


@router.get("/me/thought")
def my_thought(user_id: str = Depends(judge_user)) -> dict[str, Any]:
    from anjo.sleep.app import judge_thought

    return _thought(lambda: judge_thought.view(user_id))


@router.post("/me/thought/{item_id}/dispute")
def dispute_thought(item_id: str, body: DisputeRequest | None = None,
                    user_id: str = Depends(judge_user)) -> dict[str, Any]:
    from anjo.sleep.app import judge_thought

    return _thought(lambda: judge_thought.dispute(user_id, item_id, None if body is None else body.note))


@router.post("/me/thought/{item_id}/hide")
def hide_thought(item_id: str, user_id: str = Depends(judge_user)) -> dict[str, Any]:
    from anjo.sleep.app import judge_thought

    return _thought(lambda: judge_thought.hide(user_id, item_id))


# ── Each judge's chat turn ────────────────────────────────────────────────────

CHAT_MESSAGE: Final = re.compile(r"^/api/chat/[^/]+/message$")


def _judge_of(scope: Any) -> str | None:
    from anjo.core.best_effort import best_effort
    from anjo.dashboard.auth import verify_request_token

    found: str | None = None
    with best_effort("judge turn lookup", type_only=True):  # the turn goes on as anyone's
        user_id = verify_request_token(Request(scope))  # memoized on the scope: the auth middleware reuses it
        found = user_id if user_id and is_judge(user_id) else None
    return found


def _answered(user_id: str) -> None:
    from anjo.core.best_effort import best_effort
    from anjo.sleep.app.live_mind import runner

    with best_effort("judge live MIND", type_only=True):
        runner().schedule(user_id)


class TurnWatch:
    """Pure ASGI: a judge's chat message counts as presence when it starts, and asks for MIND's
    live pass once it has answered (status 200, its stream ended). Every other request passes by
    untouched; a judge's turn itself is never changed or held up."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST" or not CHAT_MESSAGE.match(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        from starlette.concurrency import run_in_threadpool

        user_id = await run_in_threadpool(_judge_of, scope)
        if user_id is None:
            await self.app(scope, receive, send)
            return
        await run_in_threadpool(_arrive)
        seen: dict[str, Any] = {}

        async def watched(message: Any) -> None:
            if message.get("type") == "http.response.start":
                seen["status"] = message.get("status")
            elif message.get("type") == "http.response.body" and not message.get("more_body", False):
                seen["done"] = True
            await send(message)

        await self.app(scope, receive, watched)
        if seen.get("status") == 200 and seen.get("done"):
            _answered(user_id)
