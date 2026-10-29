# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Modal holds a web request at most 150 s, then answers 303 to the original URL with a query naming
the call: that result URL blocks until the call completes, up to 150 s a hop, and is fetched with a
GET. Mornings of 2026-10-24 and 10-25 failed on it: the poster didn't follow, and read the 303's empty
body as "not JSON". Now a night client's post follows Modal's 303 on the server's own host, inside the
request's one bound; any other redirect, and a 2xx that isn't JSON, is a ServingError naming its
status. Once a 303 has come, the call is running on vLLM: a result GET that drops or meets a proxy
status is asked again, and the POST is never sent again. Fakes only: no network."""
from __future__ import annotations

import httpx
import pytest

from anjo.sleep.serving import client as vllm
from anjo.sleep.serving import config, redirect
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
    whose read timeout ends first raises ReadTimeout there, and a new POST starts the work again. The
    first GETs meet ``fails`` in turn, at once: a status the proxy sends, or a dropped connection. Time
    moves on ``waits``' clock. Every request is kept: method, URL, read timeout, bearer header."""

    def __init__(
        self, waits: Waits, work_s: float, location: str = RESULT, status: int = 303,
        fails: tuple[int | type[httpx.TransportError], ...] = (),
    ) -> None:
        super().__init__()
        self.waits, self.work_s, self.location, self.status, self.fails = waits, work_s, location, status, fails
        self.gets = 0
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
        else:
            self.gets += 1
            if self.gets <= len(self.fails):
                failure = self.fails[self.gets - 1]
                if isinstance(failure, int):
                    return httpx.Response(failure, text="<html>the proxy's own page</html>")
                raise failure("dropped", request=request)
        wait = min(self.ready_at - self.waits.now, HOLD_S)
        if wait > timeout:
            self.waits.now += timeout
            raise httpx.ReadTimeout("held past the read timeout", request=request)
        self.waits.now += max(wait, 0.0)
        if self.waits.now >= self.ready_at and self.result is not None:
            return self.result
        return httpx.Response(self.status, headers={"location": self.location})


def _poster(fake: Held, waits: Waits) -> HttpPoster:
    return HttpPoster(URL, "tok", transport=httpx.MockTransport(fake), clock=waits.clock, sleep=waits.sleep)


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


def _night(fake: Held, waits: Waits) -> VllmClient:
    return VllmClient(URL, "tok", transport=httpx.MockTransport(fake), clock=waits.clock, sleep=waits.sleep)


def _completions(fake: Held) -> list[tuple[str, float]]:
    return [(method, timeout) for method, url, timeout, _ in fake.seen if COMPLETIONS in url]


def test_a_120_s_completion_on_a_warm_server_is_not_sent_again() -> None:
    """The seed arm's mornings: warm completions took 128 s and 147 s under load. A warm first try now
    waits past Modal's 150 s hop, so a slow reply is read the first time, never cut and sent again."""
    waits = Waits()
    fake = Held(waits, work_s=120.0)
    client = _night(fake, waits)
    assert client.complete(None, MESSAGES, temperature=1.0, seed=3).text == "hello"
    assert _completions(fake) == [("POST", vllm.WARM_TIMEOUT_S)] and client.retries() == 0 and waits.now == 120.0
    assert HOLD_S < vllm.WARM_TIMEOUT_S < vllm.REQUEST_TIMEOUT_S


def test_a_completion_past_the_hop_is_followed_on_the_first_try() -> None:
    """Past 150 s Modal answers 303: the request isn't lost, so its result is waited for on the rest of
    the request's 300 s, never cut at the warm try's bound and generated again."""
    waits = Waits()
    fake = Held(waits, work_s=250.0)
    client = _night(fake, waits)
    assert client.complete(None, MESSAGES, temperature=1.0, seed=3).text == "hello"
    assert _completions(fake) == [("POST", vllm.WARM_TIMEOUT_S), ("GET", vllm.REQUEST_TIMEOUT_S - HOLD_S)]
    assert client.retries() == 0 and waits.now == 250.0

    waits = Waits()
    fake = Held(waits, work_s=1000.0)
    with pytest.raises(TransportFailed):
        _night(fake, waits).complete(None, MESSAGES, temperature=1.0, seed=3)
    assert waits.now == vllm.REQUEST_TIMEOUT_S  # never past the bound one request could always take


def _methods(fake: Held) -> list[str]:
    return [method for method, url, *_ in fake.seen if COMPLETIONS in url]


@pytest.mark.parametrize("failure", [httpx.RemoteProtocolError, httpx.ConnectError, 408, 502, 503, 504])
def test_a_result_get_that_fails_is_asked_again_and_never_posted_again(
    failure: int | type[httpx.TransportError],
) -> None:
    """Review of serve-303: a result GET that dropped or met a proxy status came back as a lost request
    or a proxy failure, and the night client sent the POST again while vLLM was still generating the
    first. Now the same result URL is asked again, after a pause, inside the bound."""
    waits = Waits()
    fake = Held(waits, work_s=200.0, fails=(failure,))
    client = _night(fake, waits)
    assert client.complete(None, MESSAGES, temperature=1.0, seed=3).text == "hello"
    assert _methods(fake) == ["POST", "GET", "GET"] and client.retries() == 0  # one POST in all
    assert waits.slept == [redirect.HOP_PAUSE_S] and waits.now == 200.0


def test_a_result_get_that_meets_502s_to_the_bound_fails_cleanly_after_one_post() -> None:
    waits = Waits()
    fake = Held(waits, work_s=200.0, fails=(502,) * 99)
    client = _night(fake, waits)
    with pytest.raises(TransportFailed, match="HTTP 502") as raised:
        client.complete(None, MESSAGES, temperature=1.0, seed=3)
    assert "proxy's own page" not in str(raised.value)
    assert _methods(fake).count("POST") == 1 and _methods(fake).count("GET") > 2 and client.retries() == 0
    assert HOLD_S < waits.now <= vllm.REQUEST_TIMEOUT_S  # paused between GETs, never past the 300 s
    assert max(waits.slept) == redirect.HOP_PAUSE_MAX_S


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
    assert seen == ["POST"] + ["GET"] * redirect.RESULT_HOPS


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


def test_long_scoring_reads_past_300_s_without_reposting() -> None:
    waits = Waits()
    fake = Held(waits, work_s=850.0)
    client = _night(fake, waits)
    body = {"model": config.MODEL, "prompt": [1] * 32000, "prompt_logprobs": 0, "max_tokens": 1}
    client._requests.post(COMPLETIONS, body)
    assert waits.now == 850.0 and _methods(fake).count("POST") == 1
    assert _methods(fake).count("GET") > 4 and client.retries() == 0


def test_long_generation_budget_scales_and_is_capped() -> None:
    waits = Waits()
    fake = Held(waits, work_s=850.0)
    client = _night(fake, waits)
    assert client.complete(None, MESSAGES, temperature=1.0, max_tokens=4000).text == "hello"
    assert waits.now == 850.0 and _methods(fake).count("POST") == 1
    assert vllm.result_timeout(COMPLETIONS, {"prompt": [1] * 100000, "max_tokens": 100000}) == vllm.MAX_RESULT_TIMEOUT_S
    assert vllm.result_timeout("/tokenize", {"prompt": [1] * 32000}) == vllm.REQUEST_TIMEOUT_S


def test_load_result_gets_its_cold_read_budget_without_reposting() -> None:
    waits = Waits()
    methods: list[str] = []
    ready = 850.0

    def serve(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        left = ready - waits.now
        timeout = request.extensions["timeout"]["read"]
        wait = min(left, HOLD_S)
        if wait > timeout:
            waits.now += timeout
            raise httpx.ReadTimeout("slow disk", request=request)
        waits.now += wait
        if waits.now < ready:
            return httpx.Response(303, headers={"location": f"{vllm.LOAD_PATH}?call=load"})
        return httpx.Response(200, text="loaded")

    client = VllmClient(URL, "tok", transport=httpx.MockTransport(serve), clock=waits.clock, sleep=waits.sleep)
    assert client._requests.load_adapter(ADAPTER.id, "/adapters/" + ADAPTER.id)
    assert waits.now == ready and methods.count("POST") == 1 and methods.count("GET") > 4


@pytest.mark.parametrize("once,timeout", [(True, 300.0), (False, 200.0)])
def test_extended_result_budgets_preserve_explicit_and_once_deadlines(once: bool, timeout: float) -> None:
    waits = Waits()
    fake = Held(waits, work_s=850.0)
    client = VllmClient(URL, "tok", transport=httpx.MockTransport(fake), timeout_s=timeout,
                        clock=waits.clock, sleep=waits.sleep)
    if once:
        client = client.once()
    with pytest.raises(vllm.ResultUnread):
        client.complete(None, MESSAGES, temperature=1.0, max_tokens=4000)
    assert waits.now == timeout and _methods(fake).count("POST") == 1


def test_long_scoring_failure_stops_at_its_bound_without_reposting() -> None:
    waits = Waits()
    fake = Held(waits, work_s=3000.0)
    client = _night(fake, waits)
    body = {"model": config.MODEL, "prompt": [1] * 32000, "prompt_logprobs": 0, "max_tokens": 1}
    with pytest.raises(vllm.ResultUnread):
        client._requests.post(COMPLETIONS, body)
    assert waits.now == vllm.result_timeout(COMPLETIONS, body)
    assert _methods(fake).count("POST") == 1
