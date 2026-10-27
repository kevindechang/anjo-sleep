# mypy: disallow-subclassing-any=False, disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" leaves fastapi and pydantic untyped here)
"""The live demo page's routes (Task 13a, minimal step), mounted only on the sleep host.

``register.register_sleep`` mounts ``page_router`` at the root and ``router`` under ``/api``, and
every route answers 404 unless ``ANJO_SLEEP_HOST=1``:

- ``GET /sleep/demo``, ``/sleep/demo/app.js``, ``/sleep/demo/demo.css``: the page and its two files;
- ``GET /api/sleep/demo/page``: her brief (``judge_brief``, the panel beside the chat), the starter
  scenes and whether their pre-generated pairs are ready. No GPU, no model;
- ``POST /api/sleep/demo/visits`` with ``{"adult": true}``: the visit token; anything else is 403;
- with that token in ``X-Demo-Visit``: ``POST /api/sleep/demo/starters/<scene>`` (a pre-generated
  pair), ``POST /api/sleep/demo/ask`` (her own question, live) and ``POST /api/sleep/demo/vote``.

The page is public: there is no account, no cookie and no stored message. ``PUBLIC_PREFIXES`` are
the only paths ``register_sleep`` opens to anonymous visitors (the app's auth middleware reads them
from ``app.state.public_prefixes``); every other Sleep route still needs a signed-in person.

**The visitor's address.** The demo's own limits are the ones that hold (``demo.limits``; the app's
rate-limit middleware covers ``/api/*`` too). They count ``request.client``, the address uvicorn's
proxy headers take from the X-Forwarded-For that Caddy sets on the dev host (``anjo-dashboard``
leaves them on, trusting loopback). Caddy sets no X-Real-IP, so the demo never reads one: a visitor
can't name their own bucket. If the proxy headers were ever off, every visitor would arrive as
Caddy's loopback address and share one bucket (``visits.client_key``), which fails closed. A
refusal's ``detail`` is the visitor's text from ``words.COPY``.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Final

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from anjo.sleep.app.flags import sleep_host
from anjo.sleep.demo.limits import Refused
from anjo.sleep.demo.page import asset, page_html
from anjo.sleep.demo.pairs import Reveal, SideFacts
from anjo.sleep.demo.service import MAX_CHARS, Asleep, DemoService, Fixed, PairView, SideView
from anjo.sleep.demo.visits import client_key
from anjo.sleep.demo.words import COPY

PREFIX: Final = "/sleep/demo"
PUBLIC_PREFIXES: Final = (PREFIX, f"/api{PREFIX}")  # the page, then its API: nothing else is public
NO_STORE: Final = {"Cache-Control": "no-store"}

router = APIRouter(prefix=PREFIX)
page_router = APIRouter(prefix=PREFIX)
_lock = threading.Lock()
_service: DemoService | None = None


def _require_host() -> None:
    if not sleep_host():
        raise HTTPException(status_code=404, detail="Not Found")


def demo_service() -> DemoService:
    """One service per process: its limits, open pairs and visits live in it."""
    global _service
    _require_host()
    with _lock:
        if _service is None:
            from anjo.sleep.app.demo_live import build_service

            try:
                _service = build_service()
            except (RuntimeError, ValueError):
                raise HTTPException(status_code=503, detail=COPY["resting"]) from None
        return _service


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _client(request: Request, service: DemoService) -> str:
    """The visitor's bucket: the proxied client address (see the module's note), never a header."""
    return client_key(request.client.host if request.client else "", service.config.secret)


def _refusal(refused: Refused) -> HTTPException:
    return HTTPException(status_code=refused.status, detail=COPY[refused.reason])


class VisitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adult: StrictBool = False


class AskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr = Field(max_length=MAX_CHARS * 4)


class VoteIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: StrictStr = Field(max_length=64)
    choice: StrictStr = Field(max_length=16)


def _side_out(view: SideView) -> dict[str, Any]:
    return {"text": view.text, "withheld": view.withheld}


def _pair_out(pair: PairView) -> dict[str, Any]:
    source = pair.source
    return {
        "pair_id": pair.pair_id, "left": _side_out(pair.left), "right": _side_out(pair.right),
        "votable": pair.votable,
        "source": {"kind": source.kind, "checkpoint": source.checkpoint, "night": source.night,
                   "generated_at": source.generated_at},
    }


def _facts_out(facts: SideFacts) -> dict[str, Any]:
    return {"arm": facts.arm, "label": COPY[f"arm_{facts.arm}"], "prompt_tokens": facts.prompt_tokens,
            "cost_usd": facts.cost_usd}


def _reveal_out(reveal: Reveal) -> dict[str, Any]:
    return {"choice": reveal.vote.choice, "left": _facts_out(reveal.left), "right": _facts_out(reveal.right)}


# ── the page ──────────────────────────────────────────────────────────────────


@page_router.get("", response_class=HTMLResponse)
def page() -> HTMLResponse:
    _require_host()
    return HTMLResponse(page_html(), headers=NO_STORE)


@page_router.get("/{name}")
def page_file(name: str) -> Response:
    _require_host()
    found = asset(name)
    if found is None:
        raise HTTPException(status_code=404, detail="Not Found")
    content, media_type = found
    return Response(content, media_type=media_type, headers=NO_STORE)


# ── the API ───────────────────────────────────────────────────────────────────


@router.get("/page")
def page_data(request: Request, service: DemoService = Depends(demo_service)) -> dict[str, Any]:
    from anjo.sleep.app.demo_live import DEMO_PERSONA
    from anjo.sleep.probes.judge_brief import judge_brief

    try:
        scenes, source = service.scenes(_client(request, service), _now())
    except Refused as refused:
        raise _refusal(refused) from None
    return {
        "brief": judge_brief(DEMO_PERSONA),
        "scenes": [{"id": v.scene.id, "title": v.scene.title, "prompt": v.scene.prompt, "ready": v.ready}
                   for v in scenes],
        "pregenerated": None if source is None else {
            "label": source.kind, "checkpoint": source.checkpoint, "night": source.night,
            "generated_at": source.generated_at},
    }


@router.post("/visits")
def start_visit(body: VisitIn, request: Request, service: DemoService = Depends(demo_service)) -> dict[str, str]:
    try:
        return {"token": service.start_visit(_client(request, service), body.adult, _now())}
    except Refused as refused:
        raise _refusal(refused) from None


@router.post("/starters/{scene_id}")
def starter(
    scene_id: str, request: Request, x_demo_visit: str = Header(default="", max_length=256),
    service: DemoService = Depends(demo_service),
) -> dict[str, Any]:
    try:
        return _pair_out(service.starter(_client(request, service), x_demo_visit, scene_id, _now()))
    except Refused as refused:
        raise _refusal(refused) from None


@router.post("/ask")
def ask(
    body: AskIn, request: Request, x_demo_visit: str = Header(default="", max_length=256),
    service: DemoService = Depends(demo_service),
) -> dict[str, Any]:
    try:
        answer = service.ask(_client(request, service), x_demo_visit, body.text, _now())
    except Refused as refused:
        raise _refusal(refused) from None
    if isinstance(answer, Fixed):
        return {"kind": "crisis", "text": answer.text}
    if isinstance(answer, Asleep):
        return {"kind": "asleep", "waking": answer.waking}
    return {"kind": "pair", **_pair_out(answer)}


@router.post("/vote")
def vote(
    body: VoteIn, request: Request, x_demo_visit: str = Header(default="", max_length=256),
    service: DemoService = Depends(demo_service),
) -> dict[str, Any]:
    try:
        reveal = service.vote(_client(request, service), x_demo_visit, body.pair_id, body.choice, _now())
    except Refused as refused:
        raise _refusal(refused) from None
    return _reveal_out(reveal)
