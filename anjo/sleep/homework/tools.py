"""Homework's two tools (Task 7): search the web and fetch one public page. Nothing else.

**The fetch tool** (``SafeFetcher``) reads public pages only:

- **Schemes and URLs:** http and https on their default ports, with no credentials in the URL.
  Hosts named for private networks (``localhost``, ``*.internal``, ``*.local``, the cloud metadata
  names) are refused before any lookup.
- **Addresses:** it resolves the host itself and refuses the URL when any address is private,
  loopback, link-local, multicast, reserved, unspecified, carrier-grade NAT or a cloud metadata
  address, IPv4 embedded in IPv6 (mapped, 6to4, Teredo, NAT64) included. It then connects to the
  address it checked, sending the host name only in ``Host`` and as the TLS server name, so a second
  lookup can't swap in a private address. Redirects are followed by hand, at most five, and every
  hop is checked the same way.
- **Limits:** at most ``max_bytes`` of body are read (a compressed reply is inflated only up to the
  same limit), and the whole fetch, redirects included, stops at ``timeout_s``. The lookup and each
  body read wait at most the time left, on a worker thread, so neither a stalled resolver nor a
  server that drips one byte at a time can outlast the limit. A body in an unknown charset is read
  as UTF-8.
- **Nothing private is sent:** each hop is one bare request on a transport, with no client, so no
  cookie jar, no environment proxy or ``.netrc`` and no ``Authorization``. Its only headers are
  ``Host``, ``User-Agent``, ``Accept`` and ``Accept-Encoding: identity``.

A refused URL raises ``FetchRefused``; a page that couldn't be read (DNS, transport, timeout, an
HTTP error) raises ``FetchFailed``. Neither message carries a transport error's text.

**The search tool** is a port (``WebSearch``). ``WikipediaSearch`` is the keyless default: it reads
Wikipedia's public search API through a ``SafeFetcher`` that accepts JSON.
"""
from __future__ import annotations

import codecs
import ipaddress
import json
import socket
import time
import zlib
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Final, Protocol, runtime_checkable
from urllib.parse import quote

import httpx

DEFAULT_MAX_BYTES: Final = 1_000_000
DEFAULT_TIMEOUT_S: Final = 15.0
MAX_REDIRECTS: Final = 5
USER_AGENT: Final = "AnjoSleepHomework/1.0 (public pages only)"
TEXT_TYPES: Final = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
JSON_TYPES: Final = frozenset({"application/json"})
DEFAULT_PORTS: Final = {"http": 80, "https": 443}
REDIRECTS: Final = frozenset({301, 302, 303, 307, 308})
BLOCKED_NAMES: Final = frozenset({
    "localhost", "metadata", "metadata.google.internal", "metadata.goog", "instance-data",
    "instance-data.ec2.internal", "kubernetes", "kubernetes.default",
})
BLOCKED_SUFFIXES: Final = (
    ".localhost", ".local", ".internal", ".localdomain", ".home.arpa", ".lan", ".intranet", ".corp",
)
# Cloud metadata endpoints, refused even where the address would otherwise count as public.
METADATA: Final = frozenset({
    ipaddress.ip_address(a) for a in (
        "169.254.169.254", "169.254.170.2", "100.100.100.200", "168.63.129.16", "192.0.0.192",
        "fd00:ec2::254",
    )
})
_NAT64: Final = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL: Final = ipaddress.ip_network("64:ff9b:1::/48")
_SKIPPED_TAGS: Final = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_BREAKS: Final = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"})

Resolver = Callable[[str, int], Sequence[str]]
Clock = Callable[[], float]


class FetchRefused(ValueError):
    """A URL, an address or a reply the fetch tool won't take: never a public page."""


class FetchFailed(RuntimeError):
    """A public page that couldn't be read: DNS, transport, timeout, an HTTP error or too many
    redirects."""


@dataclass(frozen=True)
class Page:
    """A fetched page. ``url`` is what was asked for and ``final_url`` where the text came from
    after redirects. ``truncated`` says the body stopped at the size limit."""

    url: str
    final_url: str
    title: str
    text: str
    truncated: bool
    links: tuple[str, ...] = ()  # the page's http(s) links, absolute, without fragments


@dataclass(frozen=True)
class Raw:
    """A fetched body before it becomes text."""

    url: str
    final_url: str
    content_type: str
    body: bytes
    charset: str | None
    truncated: bool


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


@runtime_checkable
class WebSearch(Protocol):
    """The search tool's backend."""

    def search(self, query: str, k: int) -> list[SearchHit]:
        """At most ``k`` public results for ``query``."""


@runtime_checkable
class PageFetcher(Protocol):
    """The fetch tool's backend: ``SafeFetcher`` in the night, a fake web in tests."""

    def fetch(self, url: str) -> Page:
        """The page's text, or ``FetchRefused`` / ``FetchFailed``."""


# ── addresses ─────────────────────────────────────────────────────────────────


def _embedded(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    if address in _NAT64:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def public_address(text: str) -> bool:
    """Whether ``text`` is a public unicast address a homework fetch may connect to."""
    try:
        address = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    if address in METADATA:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address in _NAT64_LOCAL:
            return False
        inner = _embedded(address)
        if inner is not None:
            return public_address(str(inner))
    return address.is_global and not (
        address.is_multicast or address.is_reserved or address.is_loopback or address.is_link_local
        or address.is_private or address.is_unspecified
    )


def system_resolver(host: str, port: int) -> list[str]:
    """Every address the system resolves ``host`` to."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError):
        raise FetchFailed("the host name didn't resolve") from None
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


@dataclass(frozen=True)
class _Target:
    url: httpx.URL  # as asked, with the host name
    host: str  # the ASCII host name, or the address for a literal
    port: int
    literal: bool


def _checked(url: str | httpx.URL) -> _Target:
    """The URL's parts, or ``FetchRefused`` when its scheme, credentials, port or host name can't
    be fetched. Addresses are checked after resolution (``_address``)."""
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, TypeError, ValueError):
        raise FetchRefused("not a URL") from None
    if parsed.scheme not in DEFAULT_PORTS:
        raise FetchRefused("only http and https pages are fetched")
    if parsed.userinfo:
        raise FetchRefused("a URL with credentials is refused")
    default = DEFAULT_PORTS[parsed.scheme]
    if parsed.port not in (None, default):
        raise FetchRefused("only the scheme's default port is fetched")
    try:
        host = parsed.raw_host.decode("ascii").strip("[]").rstrip(".").lower()
    except UnicodeDecodeError:
        raise FetchRefused("not a host name") from None
    if not host:
        raise FetchRefused("a URL without a host is refused")
    if host in BLOCKED_NAMES or host.endswith(BLOCKED_SUFFIXES):
        raise FetchRefused("a private network name is refused")
    literal = _is_address(host)
    if literal and not public_address(host):
        raise FetchRefused("a private, loopback, link-local, multicast or metadata address is refused")
    return _Target(url=parsed.copy_with(fragment=None), host=host, port=default, literal=literal)


def _is_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


# ── the fetcher ───────────────────────────────────────────────────────────────


def _mime(header: str) -> tuple[str, str | None]:
    parts = [part.strip() for part in header.split(";")]
    charset = next((p.split("=", 1)[1].strip('"\' ') for p in parts[1:] if p.lower().startswith("charset=")), None)
    return parts[0].lower(), charset


def _inflate(body: bytes, encoding: str, limit: int) -> tuple[bytes, bool]:
    """``body`` decompressed up to ``limit`` bytes, and whether it was cut there."""
    wbits = {"gzip": 16 + zlib.MAX_WBITS, "x-gzip": 16 + zlib.MAX_WBITS, "deflate": zlib.MAX_WBITS}.get(encoding)
    if wbits is None:
        raise FetchRefused(f"a reply encoded as {encoding!r} is refused")
    inflater = zlib.decompressobj(wbits)
    try:
        out = inflater.decompress(body, limit)
    except zlib.error:
        raise FetchFailed("a compressed reply couldn't be read") from None
    return out, bool(inflater.unconsumed_tail)


class SafeFetcher:
    """``PageFetcher`` for public pages (module docstring). ``resolve``, ``transport`` and ``clock``
    are for tests: a fake DNS, an ``httpx.MockTransport`` and a fake clock."""

    def __init__(
        self, *, resolve: Resolver = system_resolver, transport: httpx.BaseTransport | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES, timeout_s: float = DEFAULT_TIMEOUT_S,
        max_redirects: int = MAX_REDIRECTS, clock: Clock = time.monotonic, types: frozenset[str] = TEXT_TYPES,
    ) -> None:
        if max_bytes < 1 or timeout_s <= 0 or max_redirects < 0:
            raise ValueError("the fetch limits are positive")
        self._resolve = resolve
        self._transport = transport
        self._max_bytes = max_bytes
        self._timeout_s = timeout_s
        self._max_redirects = max_redirects
        self._clock = clock
        self._types = types

    def fetch(self, url: str) -> Page:
        raw = self.get(url)
        text = decode(raw.body, raw.charset)
        if raw.content_type == "text/plain":
            title, body, hrefs = "", " ".join(text.split()), []
        else:
            parsed = _parse(text)
            title, body, hrefs = parsed.title_text(), parsed.body_text(), parsed.links
        return Page(url=raw.url, final_url=raw.final_url, title=title, text=body[: self._max_bytes],
                    truncated=raw.truncated, links=_absolute(raw.final_url, hrefs))

    def get(self, url: str) -> Raw:
        """The body at ``url`` after at most ``max_redirects`` checked redirects."""
        deadline = self._clock() + self._timeout_s
        target = _checked(url)
        for _ in range(self._max_redirects + 1):
            self._in_time(deadline)
            location, raw = self._hop(target, self._address(target, deadline), deadline, url)
            if raw is not None:
                return raw
            assert location is not None
            target = _checked(target.url.join(location))
        raise FetchFailed(f"more than {self._max_redirects} redirects")

    def _in_time(self, deadline: float) -> None:
        if self._clock() > deadline:
            raise FetchFailed(f"the fetch passed its {self._timeout_s:g}s time limit")

    def _address(self, target: _Target, deadline: float) -> str:
        """The checked address to connect to: every address the host resolves to must be public.
        The lookup runs on a worker thread so a stalled resolver can't outlast the time limit."""
        if target.literal:
            return target.host
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            lookup = pool.submit(self._resolve, target.host, target.port)
            found = list(lookup.result(timeout=max(0.001, deadline - self._clock())))
        except FutureTimeout:
            raise FetchFailed(f"the fetch passed its {self._timeout_s:g}s time limit") from None
        finally:
            pool.shutdown(wait=False)
        if not found:
            raise FetchFailed("the host name didn't resolve")
        if not all(public_address(address) for address in found):
            raise FetchRefused("the host resolves to a private, loopback, link-local, multicast or metadata address")
        return found[0]

    def _request(self, target: _Target, address: str, deadline: float) -> httpx.Request:
        left = max(0.001, deadline - self._clock())
        host_header = f"[{target.host}]" if ":" in target.host else target.host
        return httpx.Request(
            "GET", target.url.copy_with(host=address),
            headers={"Host": host_header, "User-Agent": USER_AGENT, "Accept-Encoding": "identity",
                     "Accept": ", ".join(sorted(self._types))},
            extensions={"timeout": {"connect": left, "read": left, "write": left, "pool": left},
                        "sni_hostname": target.host},
        )

    def _hop(self, target: _Target, address: str, deadline: float, asked: str) -> tuple[str | None, Raw | None]:
        """One request: a redirect's location, or the body."""
        transport = self._transport or httpx.HTTPTransport(retries=0)
        try:
            response = transport.handle_request(self._request(target, address, deadline))
            try:
                if response.status_code in REDIRECTS and "location" in response.headers:
                    return response.headers["location"], None
                return None, self._body(response, target, deadline, asked)
            finally:
                response.close()
        except httpx.TimeoutException:
            raise FetchFailed(f"the fetch passed its {self._timeout_s:g}s time limit") from None
        except httpx.HTTPError as error:
            raise FetchFailed(f"the page couldn't be read ({type(error).__name__})") from None
        finally:
            if self._transport is None:
                transport.close()

    def _body(self, response: httpx.Response, target: _Target, deadline: float, asked: str) -> Raw:
        if response.status_code >= 400:
            raise FetchFailed(f"the page answered HTTP {response.status_code}")
        if response.status_code in REDIRECTS:
            raise FetchFailed("a redirect without a location")
        content_type, charset = _mime(response.headers.get("content-type", ""))
        if content_type not in self._types:
            raise FetchRefused(f"a {content_type or 'typeless'} reply is not a page this tool reads")
        stream = response.stream  # the raw bytes, never decoded here: a compressed body is inflated below, bounded
        if not isinstance(stream, httpx.SyncByteStream):
            raise FetchFailed("the page couldn't be read (no byte stream)")
        body, truncated = self._read(stream, deadline)
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            body, cut = _inflate(body, encoding, self._max_bytes)
            truncated = truncated or cut
        return Raw(url=asked, final_url=str(target.url), content_type=content_type, body=body,
                   charset=charset, truncated=truncated)

    def _read(self, chunks: Iterable[bytes], deadline: float) -> tuple[bytes, bool]:
        """The body up to ``max_bytes``. Each read waits on a worker thread for at most the time
        left; past the deadline the hop's ``finally`` closes the response, which ends the read."""
        kept = bytearray()
        stream = iter(chunks)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            while True:
                self._in_time(deadline)
                try:
                    chunk = pool.submit(next, stream, None).result(timeout=max(0.001, deadline - self._clock()))
                except FutureTimeout:
                    raise FetchFailed(f"the fetch passed its {self._timeout_s:g}s time limit") from None
                if chunk is None:
                    return bytes(kept), False
                room = self._max_bytes - len(kept)
                kept += chunk[:room]
                if len(chunk) > room:
                    return bytes(kept), True
        finally:
            pool.shutdown(wait=False)


# ── page text ─────────────────────────────────────────────────────────────────


MAX_LINKS: Final = 300


def decode(body: bytes, charset: str | None) -> str:
    """``body`` as text in its declared charset, or as UTF-8 when that charset is unknown."""
    try:
        codecs.lookup(charset or "utf-8")
    except LookupError:
        charset = None
    return body.decode(charset or "utf-8", errors="replace")


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.links: list[str] = []
        self._skip = 0
        self._in_title = False

    def title_text(self) -> str:
        return " ".join("".join(self.title).split())

    def body_text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and len(self.links) < MAX_LINKS:
            href = dict(attrs).get("href")
            if href:
                self.links.append(href.strip())
        if tag == "title":
            self._in_title = True
        elif tag in _SKIPPED_TAGS:
            self._skip += 1
        elif tag in _BREAKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIPPED_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.parts.append(data)


def _parse(html: str) -> _Text:
    parser = _Text()
    parser.feed(html)
    parser.close()
    return parser


def html_text(html: str) -> tuple[str, str]:
    """A page's title and its visible text (no scripts or styles), whitespace collapsed per line."""
    parser = _parse(html)
    return parser.title_text(), parser.body_text()


def _absolute(base: str, hrefs: Iterable[str]) -> tuple[str, ...]:
    """``hrefs`` resolved against ``base``: http(s) only, without fragments, each once."""
    found: dict[str, None] = {}
    for href in hrefs:
        try:
            url = httpx.URL(base).join(href).copy_with(fragment=None)
        except (httpx.InvalidURL, TypeError, ValueError):
            continue
        if url.scheme in DEFAULT_PORTS:
            found[str(url)] = None
    return tuple(found)


# ── search ────────────────────────────────────────────────────────────────────


WIKIPEDIA_API: Final = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE: Final = "https://en.wikipedia.org/wiki/"


class WikipediaSearch:
    """``WebSearch`` over Wikipedia's public search API: keyless, read through a ``SafeFetcher``."""

    def __init__(self, fetcher: SafeFetcher | None = None) -> None:
        self._fetcher = fetcher if fetcher is not None else SafeFetcher(types=JSON_TYPES)

    def search(self, query: str, k: int) -> list[SearchHit]:
        url = (f"{WIKIPEDIA_API}?action=query&list=search&format=json&utf8=1"
               f"&srlimit={max(1, min(k, 10))}&srsearch={quote(query)}")
        raw = self._fetcher.get(url)
        try:
            found = json.loads(decode(raw.body, raw.charset))["query"]["search"]
            return [
                SearchHit(title=str(hit["title"]), url=WIKIPEDIA_PAGE + quote(str(hit["title"]).replace(" ", "_")),
                          snippet=html_text(str(hit.get("snippet", "")))[1])
                for hit in found
            ][:k]
        except (ValueError, KeyError, TypeError):
            raise FetchFailed("the search reply couldn't be read") from None
