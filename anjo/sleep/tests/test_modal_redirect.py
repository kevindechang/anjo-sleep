# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Modal holds a web request at most 150 s, then answers 303 to the original URL with a query naming
the call: that result URL blocks until the call completes, up to 150 s a hop, and is fetched with a
GET. Mornings of 2026-10-24 and 10-25 failed on it: the poster didn't follow, and read the 303's empty
body as "not JSON". Now a night client's post follows Modal's 303 on the server's own host, inside the
request's one bound; any other redirect, and a 2xx that isn't JSON, is a ServingError naming its
status. Fakes only: no network."""
from __future__ import annotations

import httpx
import pytest

from anjo.sleep.serving import client as vllm
from anjo.sleep.serving import config
from anjo.sleep.serving.client import HttpPoster, ServingError, TransportFailed, VllmClient
from anjo.sleep.tests.test_adapter_load import ADAPTER, MESSAGES, LiveVllm, Waits

COMPLETIONS = "/v1/completions"
URL = "https://serve.example"
RESULT = f"{COMPLETIONS}?call=fc-1"  # Modal's result URL: the original one, a query naming the call
HOLD_S = 150.0  # Modal's limit on one web request
BODY = {"model": config.MODEL, "prompt": [1, 2]}


class Held(LiveVllm):
    """vLLM behind Modal's proxy. A completion takes ``work_s``; the proxy holds each request on it, the
    POST and every GET of ``location``, at most ``HOLD_S`` and then answers 303 to ``location``. A request
    whose read timeout ends first raises ReadTimeout there, and a new POST starts the work again. Time
    moves on ``waits``' clock. Every request is kept: method, URL, read timeout, bearer header."""

    def __init__(self, waits: Waits, work_s: float, location: str = RESULT, status: int = 303) -> None:
        super().__init__()
        self.waits, self.work_s, self.location, self.status = waits, work_s, location, status
        self.ready_at = 0.0
        self.result: httpx.Response | None = None
        self.seen: list[tuple[str, str, float, str | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        timeout = request.extensions["timeout"]["read"]
        self.seen.append((request.method, str(request.url), timeout, request.headers.get("authorization")))
        if request.url.path != COMPLETIONS:
            return super().__call__(request)
        if request.method == "POST":
            self.ready_at = self.waits.now + self.work_s
            self.result = super().__call__(request)  # vLLM's answer, once its work is done
        wait = min(self.ready_at - self.waits.now, HOLD_S)
        if wait > timeout:
            self.waits.now += timeout
            raise httpx.ReadTimeout("held past the read timeout", request=request)
        self.waits.now += max(wait, 0.0)
        if self.waits.now >= self.ready_at and self.result is not None:
            return self.result
        return httpx.Response(self.status, headers={"location": self.location})


def _poster(fake: Held, waits: Waits) -> HttpPoster:
    return HttpPoster(URL, "tok", transport=httpx.MockTransport(fake), clock=waits.clock)


def test_a_303_is_followed_to_its_result() -> None:
    waits = Waits()
    fake = Held(waits, work_s=200.0)  # past Modal's 150 s: answered 303, then served from the result URL
    got = _poster(fake, waits).post(COMPLETIONS, BODY, timeout_s=300.0)
    assert got["choices"][0]["text"] == "hello" and waits.now == 200.0
    assert fake.seen == [("POST", f"{URL}{COMPLETIONS}", 300.0, "Bearer tok"),
                         ("GET", f"{URL}{RESULT}", 150.0, "Bearer tok")]  # a GET, on what is left of the 300 s


def test_a_303_past_the_bound_gives_up_inside_it() -> None:
    waits = Waits()
    fake = Held(waits, work_s=1000.0)  # Modal answers 303 every 150 s, for as long as anyone asks
    with pytest.raises(TransportFailed, match="HTTP 303"):
        _poster(fake, waits).post(COMPLETIONS, BODY, timeout_s=300.0)
    assert waits.now == 300.0 and [m for m, *_ in fake.seen] == ["POST", "GET"]

    waits = Waits()
    fake = Held(waits, work_s=1000.0)
    with pytest.raises(TransportFailed):
        _poster(fake, waits).post(COMPLETIONS, BODY, timeout_s=200.0)  # the second hop's wait is cut to fit
    assert waits.now == 200.0 and fake.seen[-1][2] == 50.0


def test_a_night_completion_past_150_s_is_read() -> None:
    """The seed arm's morning: one completion took past 150 s under load. Its warm first attempt still
    gives up at ``WARM_TIMEOUT_S`` and goes again; the second follows the 303 and reads the reply, and
    every attempt ends inside the one 300 s bound."""
    waits = Waits()
    fake = Held(waits, work_s=170.0)
    client = VllmClient(URL, "tok", transport=httpx.MockTransport(fake), clock=waits.clock, sleep=waits.sleep)
    assert client.complete(None, MESSAGES, temperature=1.0, seed=3).text == "hello"
    sent = [(m, t) for m, url, t, _ in fake.seen if COMPLETIONS in url]
    budget = vllm.REQUEST_TIMEOUT_S - vllm.WARM_TIMEOUT_S - vllm.BACKOFF_S[0]
    assert sent == [("POST", vllm.WARM_TIMEOUT_S), ("POST", budget), ("GET", budget - HOLD_S)]
    assert waits.now == vllm.WARM_TIMEOUT_S + vllm.BACKOFF_S[0] + 170.0 and client.retries() == 1

    waits = Waits()
    fake = Held(waits, work_s=1000.0)
    client = VllmClient(URL, "tok", transport=httpx.MockTransport(fake), clock=waits.clock, sleep=waits.sleep)
    with pytest.raises(TransportFailed):
        client.complete(None, MESSAGES, temperature=1.0, seed=3)
    assert waits.now == vllm.REQUEST_TIMEOUT_S  # never past the bound one request could always take


def test_a_redirect_off_the_server_is_refused_and_the_token_stays_home() -> None:
    waits = Waits()
    fake = Held(waits, work_s=200.0, location="https://elsewhere.example/v1/completions?call=fc-1")
    with pytest.raises(ServingError, match=f"^{COMPLETIONS} returned HTTP 303, a redirect it doesn't follow$"):
        _poster(fake, waits).post(COMPLETIONS, BODY, timeout_s=300.0)
    assert [url for _, url, *_ in fake.seen] == [f"{URL}{COMPLETIONS}"]  # nothing sent to the other host

    waits = Waits()
    fake = Held(waits, work_s=200.0, location="http://serve.example/v1/completions?call=fc-1")  # plain http
    with pytest.raises(ServingError, match="HTTP 303"):
        _poster(fake, waits).post(COMPLETIONS, BODY, timeout_s=300.0)
    assert len(fake.seen) == 1


def test_a_proxy_that_answers_303_at_once_is_not_followed_forever() -> None:
    def loop(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(303, headers={"location": RESULT})

    seen: list[str] = []
    with pytest.raises(ServingError, match="HTTP 303"):
        HttpPoster(URL, "tok", transport=httpx.MockTransport(loop)).post(COMPLETIONS, BODY)
    assert seen == ["POST"] + ["GET"] * vllm.RESULT_HOPS


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_any_other_redirect_is_a_clear_error(status: int) -> None:
    waits = Waits()
    fake = Held(waits, work_s=200.0, status=status)
    poster = _poster(fake, waits)
    with pytest.raises(ServingError, match=f"^{COMPLETIONS} returned HTTP {status}, a redirect it doesn't follow$"):
        poster.post(COMPLETIONS, BODY, timeout_s=300.0)
    assert len(fake.seen) == 1

    def load(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"location": vllm.LOAD_PATH})

    loader = HttpPoster(URL, "tok", transport=httpx.MockTransport(load))
    with pytest.raises(ServingError, match=f"HTTP {status}"):  # a redirect never counts as loaded
        loader.load_adapter(ADAPTER.id, f"{config.ADAPTERS_DIR}/{ADAPTER.id}")


def test_a_2xx_that_is_not_json_names_its_status() -> None:
    def page(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>a page</html>")

    with pytest.raises(ServingError, match=f"^{COMPLETIONS} sent a reply that is not JSON \\(HTTP 200\\)$"):
        HttpPoster(URL, "tok", transport=httpx.MockTransport(page)).post(COMPLETIONS, BODY)
