"""Homework (Task 7): Muse Wonder studies the target her chooser picked and writes a sourced artifact.

``do_homework`` runs the study. Her chooser picked the target and the action (Task 7a); what the
choice beat lives in the night's receipt, never in Wonder's output. Wonder plans and runs the study
with two tools, ``web_search`` and ``fetch_page`` (``tools.py``), then answers with JSON claims, each
citing pages and a quote from each.

**Verified sources.** A cited source is verified only when its URL is a page this study fetched
(as asked or after its redirects) and its quote, at least ``MIN_QUOTE_CHARS`` long, is in that
page's text once both are normalized (case, whitespace, typographic quotes and dashes). Unverified
sources are dropped, a claim left without one is removed, and the artifact links only verified
pages, under their final URL. Links or brackets inside a claim's own text are neutralized.

**Spend.** Each Wonder call holds its worst case on ``meta_usd`` first (``night/holds.py``):
``MAX_INPUT_TOKENS`` in and ``MAX_OUTPUT_TOKENS`` out at Wonder's list price. The request is kept
inside its input bound (a token is at least one byte of the request's JSON, plus
``MESSAGE_OVERHEAD`` per message), trimming the oldest page text first. The hold is settled at the
reply's metered cost, or at its whole bound when the call failed, and every call is debited from
the night's ``WonderMeter``, whose dollars are what is left of the rolling monthly cap. The study
plans only the calls the meter can cover at their worst case, and its last call offers no tools, so
it always ends with an answer. When not one call fits, ``WonderSpent`` is raised before any call.

``WonderHomework`` is the night's homework port: it reads the target's own evidence, runs the
study and returns the artifact as a ``homework`` evidence item whose ancestry includes Wonder, a
hosted model, with a ``research`` result.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Protocol

import httpx

from anjo.sleep import meta_api
from anjo.sleep.context.template import render
from anjo.sleep.evidence.targets import PERSON_KINDS
from anjo.sleep.homework.tools import FetchFailed, FetchRefused, Page, PageFetcher, WebSearch
from anjo.sleep.model_api import ChatResult, ToolCall
from anjo.sleep.night.deps import StudyOutcome
from anjo.sleep.night.holds import held, usd
from anjo.sleep.night.meter import WonderMeter, WonderSpent
from anjo.sleep.provenance import union
from anjo.sleep.budget import Ledger
from anjo.sleep.types import Evidence, Result, Target

PROMPT_PATH: Final = Path(__file__).parent / "prompts" / "homework_v1.md"
WONDER_ID: Final = meta_api.PINNED_MODELS["muse-wonder"]
WONDER_IN_PER_M: Final = 4.0  # list price, $ per million input tokens (§3.2)
WONDER_OUT_PER_M: Final = 20.0
MAX_INPUT_TOKENS: Final = 64_000  # one call's input bound, for the hold's worst case
MAX_OUTPUT_TOKENS: Final = 4_096
MESSAGE_OVERHEAD: Final = 16  # tokens a chat template adds per message, at most
CALL_TIMEOUT_S: Final = 180.0
STUDY_ACTIONS: Final = frozenset({"revisit_thread", "prepare_commitment", "inquire"})
ACTIONS: Final[Mapping[str, str]] = {
    "revisit_thread": "revisit a thread the person is working through, and find what would help them move it forward",
    "prepare_commitment": "prepare them for something they committed to, and find what would help them do it well",
    "inquire": "answer an open question about this topic",
}
SEARCH_HITS: Final = 5
MAX_TOOLS_PER_TURN: Final = 4
MAX_FETCHES: Final = 20
PAGE_CHARS: Final = 6_000  # page text shown to Wonder; verification reads the whole fetched page
EVIDENCE_CHARS: Final = 12_000
ITEM_CHARS: Final = 600
QUERY_CHARS: Final = 200
ARGUMENT_CHARS: Final = 1_000
MIN_QUOTE_CHARS: Final = 20
MAX_CLAIMS: Final = 24
CLAIM_CHARS: Final = 600
TITLE_CHARS: Final = 120
DEFAULT_SECTION: Final = "Findings"
TRIMMED: Final = "[page text trimmed to keep the request inside its bound]"
FINAL_TURN: Final = (
    "No tools are left. Answer now with the JSON described in the instructions, citing only pages you "
    "fetched, with quotes copied word for word from them."
)
TOOL_SPECS: Final[list[dict[str, Any]]] = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": f"Search the public web: up to {SEARCH_HITS} results, each a title, URL and snippet. "
                       "Topic terms only, never the person's names, places or dates.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "The text of one public http or https page.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]
_URL_IN_TEXT: Final = re.compile(r"(?:https?|ftp|file|javascript|data):\S*", re.IGNORECASE)
_TYPOGRAPHY: Final = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'", "“": '"', "”": '"', "„": '"',
    "″": '"', "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", " ": " ",
})


class WonderClient(Protocol):
    """Muse Wonder's chat with tools (``MetaWonder``; a scripted fake in tests)."""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, max_tokens: int,
             timeout_s: float) -> ChatResult: ...


class StoredEvidence(Protocol):
    """What the port reads of a person's evidence (``EvidenceStore.stored``)."""

    def stored(self, until: date) -> list[Evidence]: ...


class MetaWonder:
    """``WonderClient`` on the Meta Model API: ``meta_api.chat`` with ``muse-wonder``, sent as its
    pinned id, the output capped. ``transport`` is for tests."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, max_tokens: int,
             timeout_s: float) -> ChatResult:
        return meta_api.chat("muse-wonder", messages, tools, timeout_s, max_tokens=max_tokens,
                             transport=self._transport)


@dataclass(frozen=True)
class Source:
    """A verified source: a fetched page's final URL, its title and the quotes found on it."""

    url: str
    title: str
    quotes: tuple[str, ...]


@dataclass(frozen=True)
class Homework:
    """A finished study. ``calls_used`` counts Wonder calls, a failed one included; the rest are
    the receipt's counts."""

    target_id: str
    artifact_md: str
    sources: tuple[Source, ...]
    calls_used: int
    claims: int = 0
    dropped_claims: int = 0
    dropped_sources: int = 0
    searches: int = 0
    fetched: int = 0
    refused: int = 0


def wonder_usd(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * WONDER_IN_PER_M + tokens_out * WONDER_OUT_PER_M) / 1e6


def call_worst_usd() -> float:
    """The most one homework call can cost."""
    return wonder_usd(MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS)


# ── the tools ─────────────────────────────────────────────────────────────────


def _url_key(url: str) -> str | None:
    try:
        return str(httpx.URL(url.strip()).copy_with(fragment=None))
    except (httpx.InvalidURL, TypeError, ValueError):
        return None


@dataclass
class _Tools:
    """The study's two tools, and every page it fetched (keyed by asked and final URL)."""

    search: WebSearch
    fetcher: PageFetcher
    pages: dict[str, Page] = field(default_factory=dict)
    searches: int = 0
    fetched: int = 0
    refused: int = 0

    def run(self, calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
        """The assistant turn that asked for ``calls`` (at most ``MAX_TOOLS_PER_TURN``) and a reply
        to each."""
        kept = calls[:MAX_TOOLS_PER_TURN]
        asked = {"role": "assistant", "content": None, "tool_calls": [
            {"id": call.id, "type": "function", "function": {
                "name": call.name, "arguments": call.arguments if len(call.arguments) <= ARGUMENT_CHARS else "{}"}}
            for call in kept
        ]}
        return [asked, *({"role": "tool", "tool_call_id": call.id, "content": self._one(call)} for call in kept)]

    def _one(self, call: ToolCall) -> str:
        if call.name not in ("web_search", "fetch_page"):
            return f"unknown tool {call.name[:40]!r}: the only tools are web_search and fetch_page"
        try:
            arguments = json.loads(call.arguments)
        except ValueError:
            arguments = None
        key = "query" if call.name == "web_search" else "url"
        value = arguments.get(key) if isinstance(arguments, dict) else None
        if not isinstance(value, str) or not value.strip():
            return f"error: {call.name} takes a JSON object with a {key!r} string"
        return self._search(value) if call.name == "web_search" else self._fetch(value)

    def _search(self, query: str) -> str:
        self.searches += 1
        try:
            hits = self.search.search(" ".join(query.split())[:QUERY_CHARS], SEARCH_HITS)
        except (FetchFailed, FetchRefused) as error:
            return f"search failed: {error}"
        return json.dumps([{"title": h.title[:TITLE_CHARS], "url": h.url, "snippet": h.snippet[:300]} for h in hits])

    def _fetch(self, url: str) -> str:
        if self.fetched >= MAX_FETCHES:
            return f"skipped: this study already fetched {MAX_FETCHES} pages"
        self.fetched += 1
        try:
            page = self.fetcher.fetch(url.strip())
        except FetchRefused as error:
            self.refused += 1
            return f"refused: {error}"
        except FetchFailed as error:
            return f"failed: {error}"
        for key in (_url_key(url), _url_key(page.final_url)):
            if key is not None:
                self.pages[key] = page
        return f"{page.title}\n{page.final_url}\n\n{page.text[:PAGE_CHARS]}"


# ── the request ───────────────────────────────────────────────────────────────


def _line(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def _opening(target: Target, action: str, evidence: Sequence[Evidence], calls: int) -> list[dict[str, Any]]:
    said: list[str] = []
    used = 0
    for item in sorted(evidence, key=lambda e: (e.created_at, e.id), reverse=True):
        entry = f"- ({item.kind}) {_line(item.text, ITEM_CHARS)}"
        if used + len(entry) > EVIDENCE_CHARS:
            break
        said.append(entry)
        used += len(entry)
    brief = (f"Study: {action}: {ACTIONS[action]}.\nTopic ({target.kind}): {_line(target.label, 300)}\n\n"
             "What the person said about it, newest first:\n" + ("\n".join(said) or "- (nothing yet)"))
    system = render(PROMPT_PATH.read_text(encoding="utf-8"), {"CALLS": str(calls)})
    return [{"role": "system", "content": system}, {"role": "user", "content": brief}]


def request_size(messages: Sequence[Mapping[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    """An upper bound on a request's input tokens: its JSON bytes and each message's overhead."""
    return len(json.dumps({"messages": list(messages), "tools": tools}).encode()) + MESSAGE_OVERHEAD * len(messages)


def _fit(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """``messages`` inside ``MAX_INPUT_TOKENS``: the oldest page text goes first, then the oldest
    tool calls' arguments."""
    fitted = list(messages)
    stubs: tuple[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
        ("tool", lambda m: {**m, "content": TRIMMED}),
        ("assistant", lambda m: {**m, "tool_calls": [
            {**c, "function": {**c["function"], "arguments": "{}"}} for c in m.get("tool_calls") or ()]}),
    )
    for role, stub in stubs:
        for index, message in enumerate(fitted):
            if request_size(fitted, tools) <= MAX_INPUT_TOKENS:
                return fitted
            if message["role"] == role and index > 1:
                fitted[index] = stub(message)
    if request_size(fitted, tools) > MAX_INPUT_TOKENS:
        raise ValueError("a homework request can't fit inside its input bound")
    return fitted


# ── the answer ────────────────────────────────────────────────────────────────


def _normal(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY).casefold().split())


def _plain(text: object, limit: int) -> str:
    """Wonder's words as one line of plain text: no links, no brackets, no markdown block marks."""
    flat = _line(_URL_IN_TEXT.sub("(link removed)", str(text)), limit)
    return flat.replace("[", "(").replace("]", ")").lstrip("#>-*+ ").strip()


def _answer(text: str) -> dict[str, Any] | None:
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for candidate in (body, body[body.find("{"): body.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


@dataclass
class _Verified:
    sources: dict[str, Source] = field(default_factory=dict)
    lines: dict[str, list[str]] = field(default_factory=dict)
    claims: int = 0
    dropped_claims: int = 0
    dropped_sources: int = 0


def _verify(cited: object, pages: Mapping[str, Page], normal: dict[str, str]) -> tuple[Page, str] | None:
    if not isinstance(cited, dict) or not isinstance(cited.get("url"), str) or not isinstance(cited.get("quote"), str):
        return None
    key = _url_key(cited["url"])
    page = pages.get(key) if key is not None else None
    quote = _line(cited["quote"], 2_000)
    wanted = _normal(quote).strip("\"'.,;: ")
    if page is None or len(wanted) < MIN_QUOTE_CHARS:
        return None
    if page.final_url not in normal:
        normal[page.final_url] = _normal(page.text)
    return (page, quote) if wanted in normal[page.final_url] else None


def _href(url: str) -> str:
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _claim(raw: object, pages: Mapping[str, Page], normal: dict[str, str], out: _Verified) -> None:
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        out.dropped_claims += 1
        return
    found = [_verify(cited, pages, normal) for cited in raw["sources"]]
    kept = [f for f in found if f is not None]
    out.dropped_sources += len(found) - len(kept)
    text = _plain(raw.get("text", ""), CLAIM_CHARS)
    if not kept or not text:
        out.dropped_claims += 1
        return
    links: dict[str, str] = {}
    for page, quote in kept:
        prior = out.sources.get(page.final_url)
        title = _plain(page.title, TITLE_CHARS) or (httpx.URL(page.final_url).host or page.final_url)
        quotes = prior.quotes if prior else ()
        out.sources[page.final_url] = Source(page.final_url, title, quotes if quote in quotes else (*quotes, quote))
        links[page.final_url] = f"[{title}]({_href(page.final_url)})"
    section = _plain(raw.get("section") or DEFAULT_SECTION, 80) or DEFAULT_SECTION
    out.lines.setdefault(section, []).append(f"- {text} ({'; '.join(links.values())})")
    out.claims += 1


def _artifact(target: Target, text: str, tools: _Tools, calls: int) -> Homework:
    data = _answer(text) or {}
    found = data.get("claims")
    raw_claims: list[Any] = found if isinstance(found, list) else []
    out = _Verified(dropped_claims=max(0, len(raw_claims) - MAX_CLAIMS))
    normal: dict[str, str] = {}
    for raw in raw_claims[:MAX_CLAIMS]:
        _claim(raw, tools.pages, normal, out)
    title = _plain(data.get("title") or "", TITLE_CHARS) or _plain(target.label, TITLE_CHARS) or "Homework"
    lines = [f"# {title}"]
    for section, claims in out.lines.items():
        lines += ["", f"## {section}", "", *claims]
    return Homework(
        target_id=target.id, artifact_md="\n".join(lines) + "\n", sources=tuple(out.sources.values()),
        calls_used=calls, claims=out.claims, dropped_claims=out.dropped_claims,
        dropped_sources=out.dropped_sources, searches=tools.searches, fetched=tools.fetched, refused=tools.refused,
    )


# ── the study ─────────────────────────────────────────────────────────────────


def _affordable(meter: WonderMeter, worst: float) -> int:
    """How many more calls the meter covers, each at its worst case."""
    return min(meter.calls_left, int(meter.usd_left // worst))


def _check(user_id: str, target: Target, action: str, evidence: Sequence[Evidence]) -> None:
    if action not in STUDY_ACTIONS:
        raise ValueError(f"{action!r} is not a study action ({', '.join(sorted(STUDY_ACTIONS))})")
    if target.person_id != user_id or any(item.person_id != user_id for item in evidence):
        raise ValueError("homework never reads another person's target or evidence")


@dataclass(frozen=True)
class _Call:
    wonder: WonderClient
    meter: WonderMeter
    ledger: Ledger | None
    job: str
    timeout_s: float

    def __call__(self, number: int, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResult:
        """One Wonder call, its worst case held on ``meta_usd`` and debited at its metered cost, or
        at its bound when it failed."""
        worst = call_worst_usd()
        spent = [worst]
        with held(self.ledger, f"{self.job}:c{number}", ("meta_usd",), usd(worst), lambda: spent[0]):
            try:
                reply = self.wonder.chat(messages, tools, max_tokens=MAX_OUTPUT_TOKENS, timeout_s=self.timeout_s)
                spent[0] = wonder_usd(reply.input_tokens, reply.output_tokens)
            finally:
                self.meter.debit(spent[0])
        return reply


def do_homework(
    user_id: str, night: date, target: Target, action: str, evidence: Sequence[Evidence], *,
    wonder: WonderClient, search: WebSearch, fetcher: PageFetcher, meter: WonderMeter,
    ledger: Ledger | None = None, timeout_s: float = CALL_TIMEOUT_S,
) -> Homework:
    """Wonder studies ``target`` for ``action`` from the person's ``evidence`` on it (module
    docstring). Raises ``WonderSpent`` when not one call fits the meter, and lets a failed call's
    ``ModelApiError`` through after debiting it."""
    _check(user_id, target, action, evidence)
    worst = call_worst_usd()
    planned = _affordable(meter, worst)
    if planned < 1:
        raise WonderSpent("Wonder's calls or dollars left can't cover one homework call at its worst case")
    call = _Call(wonder, meter, ledger, f"night:homework:{user_id}:{night.isoformat()}:{target.id}", timeout_s)
    tools = _Tools(search, fetcher)
    messages = _opening(target, action, evidence, planned)
    used, text = 0, ""
    while (left := min(planned - used, _affordable(meter, worst))) >= 1:
        offered = None if left == 1 else TOOL_SPECS
        if offered is None:
            messages = [*messages, {"role": "user", "content": FINAL_TURN}]
        messages = _fit(messages, offered)
        used += 1
        reply = call(used, messages, offered)
        if offered is None or not reply.tool_calls:
            text = reply.text
            break
        messages = [*messages, *tools.run(reply.tool_calls)]
    return _artifact(target, text, tools, used)


# ── the night's port ──────────────────────────────────────────────────────────


def _digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


class WonderHomework:
    """The night's ``Homework`` port (``night/deps.py``): the target's own evidence in, the artifact
    out as ``homework`` evidence with Wonder in its ancestry, and a ``research`` result."""

    available: bool = True

    def __init__(
        self, *, wonder: WonderClient, search: WebSearch, fetcher: PageFetcher,
        evidence: Callable[[str], StoredEvidence], ledger: Ledger | None = None,
    ) -> None:
        self._wonder = wonder
        self._search = search
        self._fetcher = fetcher
        self._evidence = evidence
        self._ledger = ledger

    def study(self, user_id: str, target: Target, night: date, meter: WonderMeter, *, action: str,
              at: datetime) -> StudyOutcome:
        own = [item for item in self._evidence(user_id).stored(night)
               if item.target_id == target.id and item.kind in PERSON_KINDS and not item.held_out]
        homework = do_homework(user_id, night, target, action, own, wonder=self._wonder, search=self._search,
                               fetcher=self._fetcher, meter=meter, ledger=self._ledger)
        items: tuple[Evidence, ...] = ()
        if homework.sources:
            items = (Evidence(
                id=f"hw-{_digest(user_id, night.isoformat(), target.id, homework.artifact_md)}", person_id=user_id,
                day=night, kind="homework", text=homework.artifact_md, source=homework.sources[0].url,
                held_out=False, created_at=at, model=WONDER_ID,
                ancestry=union([frozenset({WONDER_ID}), *(item.ancestry for item in own)]), target_id=target.id,
            ),)
        result = Result(
            id=f"res-{_digest(user_id, night.isoformat(), target.id, target.revision)}", target_id=target.id,
            homework_id=items[0].id if items else None, verified_sources=len(homework.sources), created_at=at,
            consumed=False, basis="research",
        )
        receipt = {
            "verified_sources": len(homework.sources), "wonder_calls": homework.calls_used, "claims": homework.claims,
            "dropped_claims": homework.dropped_claims, "dropped_sources": homework.dropped_sources,
            "searches": homework.searches, "fetched": homework.fetched, "refused": homework.refused,
        }
        return StudyOutcome(result=result, evidence=items, receipt=receipt)
