# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Fakes for homework's tests (Task 7): a fake public web behind ``httpx.MockTransport`` with its own
DNS, and a scripted Muse Wonder. Nothing here touches the network or a real resolver."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from anjo.sleep.homework.tools import FetchFailed, SafeFetcher, SearchHit
from anjo.sleep.model_api import ChatResult, ModelApiError, ToolCall

PUBLIC = "93.184.216.34"
DOCS = "93.184.216.35"
STUDY_URL = "https://example.org/sorting-study"
STUDY_QUOTE = "Optical sorters miss black film unless it is lit from below"
STUDY_HTML = (
    "<html><head><title>Sorting film plastics</title><script>var x = 'hidden script';</script></head>"
    f"<body><h1>Film plastics</h1><p>{STUDY_QUOTE}, a 2025 trial found.</p>"
    "<p>Near-infrared cameras read most clear films in a single pass on a slow belt.</p></body></html>"
)
GUIDE_URL = "https://docs.example.org/infeed-guide"
GUIDE_QUOTE = "Keep the infeed belt under one metre per second"
GUIDE_HTML = f"<html><head><title>Infeed guide</title></head><body><p>{GUIDE_QUOTE} for film.</p></body></html>"


Handler = Callable[[httpx.Request], httpx.Response]


@dataclass
class FakeWeb:
    """A tiny web: ``dns`` maps a host to its addresses, ``pages`` maps (host, path) to a handler.
    ``requests`` is every request that reached the transport, ``lookups`` every host resolved."""

    dns: dict[str, list[str]] = field(default_factory=lambda: {
        "example.org": [PUBLIC], "docs.example.org": [DOCS], "intranet.example.org": ["10.0.0.5"],
        "loop.example.org": ["127.0.0.1"], "rebind.example.org": [PUBLIC, "127.0.0.1"],
        "meta.example.org": ["169.254.169.254"], "nat64.example.org": ["64:ff9b::7f00:1"],
    })
    pages: dict[tuple[str, str], Handler] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    lookups: list[str] = field(default_factory=list)

    def resolve(self, host: str, port: int) -> list[str]:
        self.lookups.append(host)
        if host not in self.dns:
            raise FetchFailed("no such host")
        return list(self.dns[host])

    def html(self, url: str, body: str, *, headers: dict[str, str] | None = None) -> None:
        parsed = httpx.URL(url)
        self.pages[(parsed.host, parsed.path)] = lambda request: httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8", **(headers or {})}, content=body.encode(),
        )

    def redirect(self, url: str, location: str, *, headers: dict[str, str] | None = None) -> None:
        parsed = httpx.URL(url)
        self.pages[(parsed.host, parsed.path)] = lambda request: httpx.Response(
            302, headers={"location": location, **(headers or {})},
        )

    def respond(self, url: str, handler: Handler) -> None:
        parsed = httpx.URL(url)
        self.pages[(parsed.host, parsed.path)] = handler

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.headers["host"]
        handler = self.pages.get((host, request.url.path))
        if handler is None:
            return httpx.Response(404, headers={"content-type": "text/plain"}, content=b"not found")
        return handler(request)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def fetcher(self, **kw: Any) -> SafeFetcher:
        return SafeFetcher(resolve=self.resolve, transport=self.transport(), **kw)


def study_web() -> FakeWeb:
    web = FakeWeb()
    web.html(STUDY_URL, STUDY_HTML)
    web.html(GUIDE_URL, GUIDE_HTML)
    return web


@dataclass
class FakeSearch:
    hits: list[SearchHit] = field(default_factory=lambda: [
        SearchHit("Sorting film plastics", STUDY_URL, "A trial of optical sorters on film."),
        SearchHit("Infeed guide", GUIDE_URL, "Belt speeds for film."),
    ])
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, k: int) -> list[SearchHit]:
        self.queries.append(query)
        return self.hits[:k]


def tool(name: str, call_id: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def final(claims: Sequence[Any], title: str = "Sorting film plastics") -> str:
    return json.dumps({"title": title, "claims": list(claims)})


GOOD_CLAIMS = [
    {"section": "What the trials show", "text": "Optical sorters miss black film unless it is lit from below.",
     "sources": [{"url": STUDY_URL, "quote": STUDY_QUOTE}]},
    {"section": "Running the line", "text": "Run the infeed belt slower than one metre per second for film.",
     "sources": [{"url": GUIDE_URL, "quote": GUIDE_QUOTE}]},
]


def research_script(claims: Sequence[Any] = GOOD_CLAIMS) -> list[Any]:
    """Search, fetch both pages, then answer."""
    return [
        (tool("web_search", "c1", query="optical sorting film plastics"),),
        (tool("fetch_page", "c2", url=STUDY_URL), tool("fetch_page", "c3", url=GUIDE_URL)),
        final(claims),
    ]


@dataclass
class FakeWonder:
    """A scripted Wonder: each call takes the next step, tool calls or a final text. ``fail_at`` makes
    that call (1-based) raise like a provider error."""

    script: list[Any]  # each step a tuple of ToolCalls or a final text
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_at: int | None = None
    tokens_in: int = 2000
    tokens_out: int = 400

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, max_tokens: int,
             timeout_s: float) -> ChatResult:
        self.calls.append({"messages": json.loads(json.dumps(messages)), "tools": tools, "max_tokens": max_tokens})
        if self.fail_at == len(self.calls):
            raise ModelApiError("Meta Model API chat with muse-wonder-1.0 returned HTTP 503")
        step = self.script[min(len(self.calls), len(self.script)) - 1]
        if tools is None and not isinstance(step, str):  # a forced final: answer with the last step
            step = next(s for s in reversed(self.script) if isinstance(s, str))
        if isinstance(step, str):
            return ChatResult(step, (), "muse-wonder-1.0", self.tokens_in, self.tokens_out, "stop")
        return ChatResult("", step, "muse-wonder-1.0", self.tokens_in, self.tokens_out, "tool_calls")


def chunks(parts: Sequence[bytes], on_chunk: Callable[[], None] | None = None) -> Iterator[bytes]:
    for part in parts:
        if on_chunk is not None:
            on_chunk()
        yield part
