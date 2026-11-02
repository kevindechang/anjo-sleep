# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" leaves fastapi untyped here)
"""The judge product's pages, mounted only on the sleep host: static files from ``anjo/sleep/web/app``.

- ``GET /sleep/app``: the landing, with the two ways in (Teo's Anjo, or the judge code);
- ``GET /sleep/app/live``: the product view (chat, the MIND panel, Sleep now, the night, the wake);
- ``GET /sleep/app/<file>``: their scripts and stylesheets, named in ``ASSETS`` and nothing else.

The pages call the judge API (``/api/sleep/judge/*``, ``/api/sleep/me/*``) and the existing chat
routes; nothing here reads a person's data. Every route answers 404 unless ``ANJO_SLEEP_HOST=1``.
The pages carry no inline script (the app's CSP adds a nonce to every ``<script>`` tag, and these
are all external, same-origin files) and load nothing from any other host. Their look is the main
site's own front-end code: ``/static/site.css`` and ``/static/home.css``.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from anjo.sleep.app.flags import sleep_host

PREFIX: Final = "/sleep/app"
PUBLIC_PREFIXES: Final = (PREFIX,)  # the pages only; the judge API opens its own sign-in route
APP_DIR: Final = Path(__file__).parents[1] / "web" / "app"
NO_STORE: Final = {"Cache-Control": "no-store"}
PAGES: Final[Mapping[str, str]] = MappingProxyType({"": "index.html", "live": "live.html"})
_JS: Final = "text/javascript; charset=utf-8"
_CSS: Final = "text/css; charset=utf-8"
ASSETS: Final[Mapping[str, str]] = MappingProxyType({
    "app.css": _CSS,       # the site has no pattern for these: forms, the chat, the live layout
    "panel.css": _CSS,     # the MIND panel, the night's stages and the wake card
    "copy.js": _JS,        # every word the pages say, in one place
    "dom.js": _JS,         # small DOM helpers: text in through textContent only
    "api.js": _JS,         # requests, the Bearer token, plain errors, and the ?mock=1 switch
    "mock.js": _JS,        # canned answers for development, loaded only with ?mock=1
    "sse.js": _JS,         # the chat stream over POST (ported from mobile/lib/sse.ts)
    "landing.js": _JS,     # the judge code
    "live.js": _JS,        # the product view: tabs, the talker label, wiring
    "chat.js": _JS,
    "panel.js": _JS,       # the MIND panel
    "sleep.js": _JS,       # Sleep now and the night view
    "wake.js": _JS,        # the wake view and undo
    "thought.css": _CSS,   # her line of thought, the side panel's tabs, "since your last visit"
    "thought.js": _JS,     # her line of thought: hunches (dispute, hide), her scorecard, one question
    "morning.js": _JS,     # "since your last visit", in the wake and on the next visit
    "livepass.js": _JS,    # after a reply, polls MIND until its live pass has landed
    "mock_thought.js": _JS,  # her line of thought for ?mock=1 (a synthetic life), loaded only by mock.js
})

page_router = APIRouter(prefix=PREFIX)


def _require_host() -> None:
    if not sleep_host():
        raise HTTPException(status_code=404, detail="Not Found")


def _page(name: str) -> HTMLResponse:
    _require_host()
    return HTMLResponse((APP_DIR / PAGES[name]).read_text(encoding="utf-8"), headers=NO_STORE)


@page_router.get("", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    return _page("")


@page_router.get("/{name}")
def app_file(name: str) -> Response:
    """A page by its name, or one of its files; any other name is a 404, so nothing walks the folder."""
    if name in PAGES:
        return _page(name)
    _require_host()
    media = ASSETS.get(name)
    if media is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return Response((APP_DIR / name).read_bytes(), media_type=media, headers=NO_STORE)
