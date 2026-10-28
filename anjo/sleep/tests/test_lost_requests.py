# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""One request lost on the way to Glimmer no longer fails a whole night (night 1's rerun, 2026-10-28:
the safety gate died on a ReadTimeout after 300 s for a request Modal never delivered).

The night's client sends a request lost in transport again, at most twice, with a short pause; a
status from the server is an answer and is never sent again. A warm server gets a shorter timeout
per attempt than a cold start does, and every attempt of one request still ends inside the 300 s
one request could always take. One client never has more requests in flight than one server
container takes. A chat turn and the live page send once. The night log counts what was sent
again, step by step. Fakes only: no network."""
from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.serving import client as vllm
from anjo.sleep.serving import config
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.tests.test_adapter_load import ADAPTER, MESSAGES, LiveVllm, Waits

COMPLETIONS = "/v1/completions"


class Lossy(LiveVllm):
    """vLLM behind a proxy that loses requests: the first ``lose`` requests to each path in ``paths``
    never come back (``error``), and each request's read timeout is kept."""

    def __init__(self, lose: int, paths: tuple[str, ...] = (COMPLETIONS,), error: type[httpx.TransportError] = httpx.ReadTimeout) -> None:
        super().__init__()
        self.lose, self.lost_paths, self.error = lose, paths, error
        self.timeouts: list[tuple[str, float]] = []
        self.seen: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":  # a liveness probe: kept apart, in ``probes``
            return super().__call__(request)
        path = request.url.path
        self.timeouts.append((path, request.extensions["timeout"]["read"]))
        self.seen[path] = self.seen.get(path, 0) + 1
        if path in self.lost_paths and self.seen[path] <= self.lose:
            raise self.error("lost on the way", request=request)
        return super().__call__(request)


class Hangs(Lossy):
    """``Lossy`` whose lost requests hang until their attempt's timeout, on ``waits``' clock."""

    def __init__(self, waits: Waits, lose: int, paths: tuple[str, ...] = (COMPLETIONS,)) -> None:
        super().__init__(lose=lose, paths=paths)
        self.waits = waits

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        lost = path in self.lost_paths and self.seen.get(path, 0) < self.lose
        if lost or (request.method == "GET" and not self.alive):
            self.waits.now += request.extensions["timeout"]["read"]
        return super().__call__(request)


def _client(fake: Any, waits: Waits) -> VllmClient:
    return VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake),
                      adapters_dir=config.ADAPTERS_DIR, clock=waits.clock, sleep=waits.sleep)


def test_transport_error_retries_then_succeeds() -> None:
    fake, waits = Lossy(lose=1, paths=(COMPLETIONS, vllm.LOAD_PATH)), Waits()
    client = _client(fake, waits)
    assert client.complete(ADAPTER, MESSAGES, temperature=1.0, seed=3).text == "hello"
    assert fake.seen[COMPLETIONS] == 2 and fake.seen[vllm.LOAD_PATH] == 2  # the load is sent again too
    assert waits.slept == [vllm.BACKOFF_S[0]] * 2
    assert client.retries() == 2 and client.on(date(2026, 11, 1)).retries() == 2  # its copies share the count
    assert fake.probes == []  # each went again unasked: only a request about to give up asks /v1/models


def test_transport_error_gives_up_after_three_attempts() -> None:
    fake, waits = Lossy(lose=99, error=httpx.ConnectError), Waits()
    client = _client(fake, waits)
    with pytest.raises(ServingError, match="failed in transport") as raised:
        client.complete(None, MESSAGES, temperature=1.0, seed=3)
    assert "hi" not in str(raised.value)
    assert vllm.ATTEMPTS == 3 and fake.seen[COMPLETIONS] == 3 and fake.seen["/tokenize"] == 2
    assert waits.slept == list(vllm.BACKOFF_S) and client.retries() == 2


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_status_error_is_not_retried(status: int) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(status, json={"error": "busy"})

    seen: list[str] = []
    waits = Waits()
    client = _client(answer, waits)
    with pytest.raises(ServingError, match=str(status)):
        client.complete(None, MESSAGES, temperature=1.0)
    assert seen == ["/tokenize"] and waits.slept == []  # the server answered: an answer is never sent again
    assert client.retries() == 0


def test_warm_attempt_timeout_is_shorter_than_cold() -> None:
    fake, waits = Lossy(lose=0), Waits()
    client = _client(fake, waits)
    client.complete(None, MESSAGES, temperature=1.0)
    first, *after = (timeout for _, timeout in fake.timeouts)
    assert first == vllm.REQUEST_TIMEOUT_S  # the first request after a wake may meet a cold start
    assert after == [vllm.WARM_TIMEOUT_S] * 2 and vllm.WARM_TIMEOUT_S < vllm.REQUEST_TIMEOUT_S
    waits.now += config.SCALEDOWN_S  # silent past Modal's scale-down: the container may be gone
    client.complete(None, MESSAGES, temperature=1.0)
    assert fake.timeouts[3][1] == vllm.REQUEST_TIMEOUT_S

    waits = Waits()
    warm = Hangs(waits, lose=99)  # the chat's ids came back; every completion is then lost
    with pytest.raises(ServingError, match="failed in transport"):
        _client(warm, waits).complete(None, MESSAGES, temperature=1.0)
    budget = vllm.REQUEST_TIMEOUT_S - vllm.WARM_TIMEOUT_S - vllm.BACKOFF_S[0]
    assert [t for path, t in warm.timeouts if path == COMPLETIONS] == [vllm.WARM_TIMEOUT_S, budget]
    assert waits.now == vllm.REQUEST_TIMEOUT_S  # every attempt of one request ends inside what one could always take

    waits = Waits()
    cold = Hangs(waits, lose=99, paths=("/tokenize",))  # the first request after a wake is lost: it had the whole 300 s
    with pytest.raises(ServingError, match="failed in transport"):
        _client(cold, waits).complete(None, MESSAGES, temperature=1.0)
    assert [t for _, t in cold.timeouts] == [vllm.REQUEST_TIMEOUT_S] and waits.now == vllm.REQUEST_TIMEOUT_S


def test_warm_timeout_then_cold_gets_the_remaining_budget() -> None:
    waits = Waits()
    fake = Hangs(waits, lose=1)  # one completion hangs at the warm timeout; the container may have restarted
    client = _client(fake, waits)
    assert client.complete(None, MESSAGES, temperature=1.0, seed=3).text == "hello"
    budget = vllm.REQUEST_TIMEOUT_S - vllm.WARM_TIMEOUT_S - vllm.BACKOFF_S[0]
    assert [t for path, t in fake.timeouts if path == COMPLETIONS] == [vllm.WARM_TIMEOUT_S, budget]
    assert budget > 2 * vllm.WARM_TIMEOUT_S and client.retries() == 1  # a cold start's patience, nearly all of it


def test_a_warm_client_meets_a_restarted_container() -> None:
    waits = Waits()
    fake = Hangs(waits, lose=0, paths=("/tokenize",))
    client = _client(fake, waits)
    client.complete(ADAPTER, MESSAGES, temperature=1.0, seed=1)  # loaded, and warm
    fake.timeouts.clear()
    fake.lose = fake.seen["/tokenize"] + 1  # the next request hangs while the container restarts
    fake.loaded.clear()  # and the new container has forgotten every load
    assert client.complete(ADAPTER, MESSAGES, temperature=1.0, seed=2).text == "hello"
    budget = vllm.REQUEST_TIMEOUT_S - vllm.WARM_TIMEOUT_S - vllm.BACKOFF_S[0]
    assert fake.timeouts == [("/tokenize", vllm.WARM_TIMEOUT_S), ("/tokenize", budget),  # then a 404: forgotten
                             (vllm.LOAD_PATH, vllm.WARM_TIMEOUT_S), ("/tokenize", vllm.WARM_TIMEOUT_S),
                             ("/tokenize", vllm.WARM_TIMEOUT_S), (COMPLETIONS, vllm.WARM_TIMEOUT_S)]
    assert len(fake.loads) == 2 and client.retries() == 1  # loaded again once, sent again once


def test_restart_mid_night_is_waited_out() -> None:
    """A warm server's container restarts mid-night: Modal holds every request, /v1/models included, for
    its cold start of about 3 minutes. The client asks /v1/models only when a request is about to give
    up, so the attempt after a lost one gets the rest of the 300 s and the night goes on."""
    waits = Waits()
    fake = Hangs(waits, lose=0, paths=("/tokenize",))
    client = _client(fake, waits)
    client.complete(None, MESSAGES, temperature=1.0)  # warm
    fake.lose = fake.seen["/tokenize"] + 1  # the next request is lost while the container restarts
    fake.alive = False  # and /v1/models would be held past LIVENESS_S as well
    assert client.complete(None, MESSAGES, temperature=1.0).text == "hello"  # the new container answers in time
    assert fake.probes == [] and client.retries() == 1
    assert client.complete(None, MESSAGES, temperature=1.0).text == "hello"  # never marked down


def test_wedge_is_marked_down_after_one_round() -> None:
    """Night 1c (2026-10-28): the container took requests and answered none, not even /v1/models, and
    each gate thread waited out 300 s per request. Now a request that gives up after its 300 s round asks
    /v1/models within ``LIVENESS_S``; with no answer it fails as ``ServerUnresponsive``, and every request
    the client and its copies send for the next ``DOWN_HOLD_S`` fails at once, without being sent."""
    waits = Waits()
    fake = Hangs(waits, lose=0, paths=("/tokenize", COMPLETIONS))
    client = _client(fake, waits)
    client.complete(None, MESSAGES, temperature=1.0)  # warm
    fake.lose, fake.alive = 99, False  # wedged: nothing comes back
    started = waits.now
    with pytest.raises(vllm.ServerUnresponsive, match="/v1/models") as raised:
        client.complete(None, MESSAGES, temperature=1.0)
    assert isinstance(raised.value, ServingError)
    assert waits.now - started == vllm.REQUEST_TIMEOUT_S + vllm.LIVENESS_S  # one round, then the question
    assert fake.probes == [vllm.LIVENESS_S] and waits.slept == [vllm.BACKOFF_S[0]] and client.retries() == 1

    sent = dict(fake.seen)
    for copy in (client, client.on(date(2026, 11, 1)), client.once(), client.with_load_wait(0)):
        with pytest.raises(vllm.ServerUnresponsive):
            copy.complete(ADAPTER, MESSAGES, temperature=1.0)
    assert fake.seen == sent and fake.probes == [vllm.LIVENESS_S]  # nothing sent
    assert waits.now - started == vllm.REQUEST_TIMEOUT_S + vllm.LIVENESS_S

    waits.now += vllm.DOWN_HOLD_S  # a minute on, a new container answers
    fake.lose, fake.alive = 0, True
    assert client.complete(ADAPTER, MESSAGES, temperature=1.0).text == "hello"


def test_a_cold_start_never_marks_the_server_down() -> None:
    """Before the server has answered (right after a wake, or silent past its scale-down window) a slow
    or lost request is a cold start's: the client doesn't ask /v1/models, and sends it again as before."""
    waits = Waits()
    fake = Lossy(lose=1, paths=("/tokenize",), error=httpx.ConnectError)
    fake.alive = False  # /v1/models wouldn't answer yet either
    client = _client(fake, waits)
    assert client.complete(None, MESSAGES, temperature=1.0).text == "hello"
    assert fake.probes == [] and client.retries() == 1

    waits.now += config.SCALEDOWN_S  # silent past scale-down: the next request may meet a new container
    fake.lose = fake.seen["/tokenize"] + 1
    assert client.complete(None, MESSAGES, temperature=1.0).text == "hello"
    assert fake.probes == [] and client.retries() == 2


def test_in_flight_never_exceeds_server_slots() -> None:
    class Busy(LiveVllm):
        def __init__(self) -> None:
            super().__init__()
            self.now = self.most = 0
            self.lock = threading.Lock()

        def __call__(self, request: httpx.Request) -> httpx.Response:
            with self.lock:
                self.now += 1
                self.most = max(self.most, self.now)
            try:
                time.sleep(0.01)
                return super().__call__(request)
            finally:
                with self.lock:
                    self.now -= 1

    fake = Busy()
    client = VllmClient("https://serve.example", "tok", transport=httpx.MockTransport(fake),
                        adapters_dir=config.ADAPTERS_DIR)
    copies = [client, client.on(date(2026, 11, 1)), client.with_load_wait(0)]
    threads = [threading.Thread(target=lambda i=i: copies[i % 3].complete(ADAPTER, MESSAGES, temperature=1.0, seed=i))
               for i in range(40)]  # more threads than any step runs, through every copy at once
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert 1 < fake.most <= config.CONCURRENT_REQUESTS


def test_night_log_counts_retries(tmp_path: Path) -> None:
    from anjo.sleep.night import daily
    from anjo.sleep.night.run import run_night_batch_with
    from anjo.sleep.sim.day import DayRun
    from anjo.sleep.tests.night_fakes import NIGHT, World, step
    from anjo.sleep.types import NightResult

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    reads = iter([0, 2, 2, 5])  # the client's running count, read as practice and the gates start and end
    result = run_night_batch_with(world.deps(retries=lambda: next(reads)), {"p1": NIGHT})["p1"]
    assert step(result.log_path, "practice")["retries"] == 2 and step(result.log_path, "gates")["retries"] == 3

    quiet = World(tmp_path / "quiet")
    quiet.plan("p1", NIGHT)
    calm = run_night_batch_with(quiet.deps(), {"p1": NIGHT})["p1"]
    assert "retries" not in step(calm.log_path, "practice") and "retries" not in step(calm.log_path, "gates")

    class Ports:
        def __init__(self) -> None:
            self.reads = iter([7, 8])

        def retries(self) -> int:
            return next(self.reads)

    at = datetime(2026, 10, 30, 2, 0, tzinfo=daily.TAIPEI)
    markers = tmp_path / "markers"
    daily.write_marker(markers, "founder", date(2026, 10, 29), 1, DayRun((at, at), 4, 0, ()))
    folder = tmp_path / "night"
    folder.mkdir()
    daily.run_night_phase("acct", "founder", at, markers=markers, morning=lambda *a: Ports(),
                          nights=lambda **kw: {"acct": NightResult("rejected", None, None, {}, (), 0.0, {}, folder / "log.jsonl")},
                          score=lambda *a, **kw: None)
    [morning] = [json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines()]
    assert morning["step"] == "morning" and morning["retries"] == 1  # a count, never what was sent
