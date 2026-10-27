# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Homework's fetch tool (Task 7): public pages only. It refuses private, loopback, link-local,
multicast and metadata addresses after DNS resolution and again on every redirect, refuses other
schemes and credentials, keeps to its size and time limits and never sends cookies or keys. The
web is a fake behind ``httpx.MockTransport``; nothing reaches a real resolver."""
from __future__ import annotations

import gzip

import httpx
import pytest

from anjo.sleep.homework.tools import FetchFailed, FetchRefused, SafeFetcher, html_text, public_address
from anjo.sleep.tests.homework_fakes import (
    PUBLIC,
    STUDY_HTML,
    STUDY_QUOTE,
    STUDY_URL,
    FakeWeb,
    chunks,
    study_web,
)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/admin", "http://[::1]/", "http://10.1.2.3/",
    "http://192.168.1.1/", "http://172.16.0.9/", "http://169.254.169.254/latest/meta-data/",
    "http://100.100.100.200/", "http://168.63.129.16/", "http://[fd00:ec2::254]/",
    "http://[::ffff:127.0.0.1]/", "http://224.0.0.1/", "http://0.0.0.0/",
    "http://metadata.google.internal/computeMetadata/v1/", "http://printer.local/",
    "http://loop.example.org/", "http://intranet.example.org/", "http://meta.example.org/",
    "http://nat64.example.org/", "http://rebind.example.org/",
])
def test_homework_fetch_refuses_private_and_loopback_hosts(url: str) -> None:
    web = study_web()
    with pytest.raises(FetchRefused):
        web.fetcher().fetch(url)
    assert web.requests == []


@pytest.mark.parametrize("url", [
    "ftp://example.org/file", "file:///etc/passwd", "gopher://example.org/", "javascript:alert(1)",
    "https://user:secret@example.org/", "https://token@example.org/", "https://example.org:8443/",
    "http://example.org:443/", "//example.org/x", "",
])
def test_fetch_refuses_other_schemes_credentials_and_ports(url: str) -> None:
    web = study_web()
    with pytest.raises(FetchRefused):
        web.fetcher().fetch(url)
    assert web.requests == []


@pytest.mark.parametrize("location", [
    "http://intranet.example.org/admin", "http://169.254.169.254/latest/meta-data/",
    "http://127.0.0.1:80/", "http://[::1]/", "file:///etc/passwd", "https://user:pw@example.org/x",
    "http://loop.example.org/",
])
def test_fetch_refuses_redirect_to_private_host(location: str) -> None:
    web = study_web()
    web.redirect("https://example.org/start", location)
    with pytest.raises(FetchRefused):
        web.fetcher().fetch("https://example.org/start")
    assert [r.url.host for r in web.requests] == [PUBLIC]  # the redirect was never followed


def test_fetch_follows_a_public_redirect_and_pins_the_checked_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_META_API_KEY", "meta-key-should-never-leave")
    web = study_web()
    web.redirect("https://example.org/old", STUDY_URL, headers={"set-cookie": "session=abc; Path=/"})
    page = web.fetcher().fetch("https://example.org/old")
    assert page.final_url == STUDY_URL and page.title == "Sorting film plastics"
    assert STUDY_QUOTE in page.text and "hidden script" not in page.text
    for request in web.requests:
        assert request.url.host == PUBLIC  # connected to the address that was checked
        assert request.headers["host"] == "example.org"
        assert request.extensions["sni_hostname"] == "example.org"
        assert "cookie" not in request.headers and "authorization" not in request.headers
        assert "meta-key-should-never-leave" not in str(request.headers)
    assert web.lookups == ["example.org", "example.org"]  # resolved again on the redirect


def test_fetch_stops_after_five_redirects() -> None:
    web = study_web()
    for i in range(7):
        web.redirect(f"https://example.org/r{i}", f"https://example.org/r{i + 1}")
    with pytest.raises(FetchFailed, match="redirects"):
        web.fetcher().fetch("https://example.org/r0")
    assert len(web.requests) == 6


def test_fetch_stops_reading_at_its_size_limit() -> None:
    web = FakeWeb()
    body = [b"<p>" + b"word " * 200 + b"</p>"] * 50  # about 50 KB
    web.respond(STUDY_URL, lambda request: httpx.Response(
        200, headers={"content-type": "text/html"}, content=chunks(body)))
    page = web.fetcher(max_bytes=4096).fetch(STUDY_URL)
    assert page.truncated and len(page.text) <= 4096


def test_fetch_keeps_a_compressed_reply_inside_its_size_limit() -> None:
    web = FakeWeb()
    bomb = gzip.compress(b"<p>" + b"a" * 2_000_000 + b"</p>")
    web.respond(STUDY_URL, lambda request: httpx.Response(
        200, headers={"content-type": "text/html", "content-encoding": "gzip"}, content=bomb))
    page = web.fetcher(max_bytes=4096).fetch(STUDY_URL)
    assert page.truncated and len(page.text) <= 4096
    assert web.requests[0].headers["accept-encoding"] == "identity"


def test_fetch_stops_at_its_time_limit() -> None:
    web = FakeWeb()
    now = [0.0]

    def tick() -> None:
        now[0] += 6.0

    web.respond(STUDY_URL, lambda request: httpx.Response(
        200, headers={"content-type": "text/html"}, content=chunks([b"<p>slow</p>"] * 10, tick)))
    with pytest.raises(FetchFailed, match="time"):
        web.fetcher(timeout_s=15.0, clock=lambda: now[0]).fetch(STUDY_URL)


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "application/octet-stream", ""])
def test_fetch_refuses_what_is_not_a_text_page(content_type: str) -> None:
    web = FakeWeb()
    web.respond(STUDY_URL, lambda request: httpx.Response(200, headers={"content-type": content_type}, content=b"%PDF"))
    with pytest.raises(FetchRefused):
        web.fetcher().fetch(STUDY_URL)


def test_fetch_reports_an_http_error_as_a_failure() -> None:
    web = FakeWeb()
    with pytest.raises(FetchFailed, match="404"):
        web.fetcher().fetch("https://example.org/missing")


def test_a_transport_error_is_a_failure_without_its_text() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection to 93.184.216.34 refused, secret detail")

    web = FakeWeb()
    web.respond(STUDY_URL, broken)
    with pytest.raises(FetchFailed) as caught:
        web.fetcher().fetch(STUDY_URL)
    assert "secret detail" not in str(caught.value)


def test_public_address_unwraps_embedded_ipv4() -> None:
    assert public_address(PUBLIC) and public_address("2606:2800:220:1::1")
    for address in ("::ffff:10.0.0.1", "2002:7f00:1::1", "64:ff9b::a00:1", "64:ff9b:1::1", "ff02::1",
                    "fe80::1", "239.1.2.3", "198.18.0.1", "not an address"):
        assert not public_address(address), address


def test_html_text_skips_scripts_and_keeps_the_title() -> None:
    title, text = html_text(STUDY_HTML)
    assert title == "Sorting film plastics"
    assert STUDY_QUOTE in text and "var x" not in text


def test_a_stalled_lookup_stops_at_the_time_limit() -> None:
    import threading
    import time

    released = threading.Event()

    def stalled(host: str, port: int) -> list[str]:
        released.wait(5.0)
        return [PUBLIC]

    web = study_web()
    fetcher = SafeFetcher(resolve=stalled, transport=web.transport(), timeout_s=0.05)
    started = time.monotonic()
    try:
        with pytest.raises(FetchFailed, match="time"):
            fetcher.fetch(STUDY_URL)
        assert time.monotonic() - started < 2.0  # it didn't wait for the resolver
    finally:
        released.set()
    assert web.requests == []


def test_an_unknown_charset_falls_back_to_utf8() -> None:
    web = FakeWeb()
    web.respond(STUDY_URL, lambda request: httpx.Response(
        200, headers={"content-type": "text/html; charset=bogus"}, content=STUDY_HTML.encode()))
    page = web.fetcher().fetch(STUDY_URL)
    assert STUDY_QUOTE in page.text


def test_the_search_reads_a_reply_with_an_unknown_charset() -> None:
    from anjo.sleep.homework.tools import JSON_TYPES, WikipediaSearch

    web = FakeWeb()
    web.dns["en.wikipedia.org"] = [PUBLIC]
    reply = {"query": {"search": [{"title": "Optical sorting", "snippet": "sorts <b>film</b>"}]}}
    web.respond("https://en.wikipedia.org/w/api.php", lambda request: httpx.Response(
        200, headers={"content-type": "application/json; charset=bogus"}, json=reply))
    [hit] = WikipediaSearch(web.fetcher(types=JSON_TYPES)).search("optical sorting", 3)
    assert hit.url == "https://en.wikipedia.org/wiki/Optical_sorting" and hit.snippet == "sorts film"


def test_a_slow_drip_stops_at_the_time_limit() -> None:
    import time

    web = FakeWeb()
    web.respond(STUDY_URL, lambda request: httpx.Response(
        200, headers={"content-type": "text/html"},
        content=chunks([b"<p>drip</p>"] * 6, lambda: time.sleep(0.35))))
    started = time.monotonic()
    with pytest.raises(FetchFailed, match="time"):
        web.fetcher(timeout_s=0.4).fetch(STUDY_URL)
    assert time.monotonic() - started <= 0.4 + 0.15  # never a whole extra read past the limit
