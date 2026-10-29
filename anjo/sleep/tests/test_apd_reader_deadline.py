# mypy: disable-error-code="untyped-decorator"
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A generation call that never answers (MIND's data): a Gemini read ends at its own short
deadline, which really fires; a kill reaches a call in flight, its hold settles and no join waits on
it; 429s wait without spending the attempt's billed extra tries; each failed attempt is recorded as
it fails; and ``calls`` tells failed tries priced at their worst case from billed reads. Offline."""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import calls, core_splits, failures, fakes, gen_batch, generate, readers, retry
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.generate import GenerationError
from anjo.sleep.mind.apd.judging import Rate
from anjo.sleep.mind.apd.limiter import Limiter
from anjo.sleep.mind.apd.live import RunsFailed
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult, ModelApiError, RateLimited

SPEC = next(spec for spec in core_splits.core_specs("core-train") if spec.prevalence > 0)
GEMINI_RATE = Rate(0.75, 3.75)
VERTEX_429 = "Gemini on Vertex returned HTTP 429"


class NeverAnswers:
    """A gate reader whose calls never answer (until the test lets them go), recording each call's
    timeout; it answers the leak reads only once ``answer`` is set."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []
        self.entered, self.answer, self.release = threading.Event(), threading.Event(), threading.Event()
        self.inner = fakes.FakeGateReader()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        self.timeouts.append(timeout_s)
        self.entered.set()
        if not self.answer.is_set():
            self.release.wait(30.0)
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


@pytest.fixture
def never() -> Iterator[NeverAnswers]:
    chat = NeverAnswers()
    yield chat
    chat.release.set()  # the calls given up on end with the test


def _gemini_backend(tmp_path: Path, slept: list[float], **chats: Any) -> backends.Backend:
    return dataclasses.replace(backends.fake_backend(FIREWORKS), sleep=slept.append,
                               ledger=RunLedger(tmp_path / "budget.jsonl"), prices=cli.FAKE_PRICES,
                               rates=cli.FAKE_RATES, read_by=readers.GEMINI_READER, reader_rate=GEMINI_RATE,
                               **chats)


def _in_thread(call: Any) -> tuple[threading.Thread, list[BaseException]]:
    raised: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - the test reads what the call raised
            raised.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, raised


# ── A Gemini read's deadline ──────────────────────────────────────────────────


def test_a_gemini_read_that_never_answers_ends_at_its_deadline(tmp_path: Path, never: NeverAnswers,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a Gemini reader that never answers, under a short deadline
    monkeypatch.setattr(backends, "GEMINI_READ_TIMEOUT_S", 0.2)
    monkeypatch.setattr(backends, "CALL_GRACE_S", 0.1)
    slept: list[float] = []
    started = time.monotonic()

    # Act
    with pytest.raises(GenerationError, match="timed out"):
        gen_batch.generate_life(SPEC, _gemini_backend(tmp_path, slept, reader=never))

    # Assert: each try waited its deadline, not the Spark Open reply allowance; the call's tries
    # ended each attempt, the next after its cool-down; every hold settled
    assert time.monotonic() - started < 15.0
    assert set(never.timeouts) == {0.2} and len(never.timeouts) == generate.MAX_ATTEMPTS * retry.ATTEMPTS
    assert slept.count(retry.ATTEMPT_COOL_DOWN_S) == generate.MAX_ATTEMPTS - 1
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_a_gemini_read_is_capped_at_its_own_deadline_not_the_reply_allowance(tmp_path: Path) -> None:
    seen = NeverAnswers()
    seen.answer.set()
    gen_batch.generate_life(SPEC, _gemini_backend(tmp_path, [], reader=seen))
    assert set(seen.timeouts) == {backends.GEMINI_READ_TIMEOUT_S}
    assert backends.GEMINI_READ_TIMEOUT_S < retry.reply_timeout_s(retry.REPLY_MAX_TOKENS)


def test_gemini_chat_sends_low_thinking_and_its_deadline_to_vertex(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the real SDK, its requests caught by a mock transport
    genai, types = pytest.importorskip("google.genai"), pytest.importorskip("google.genai.types")
    sent: list[httpx.Request] = []

    def vertex(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "{\"flag\": false}"}]}}],
            "usageMetadata": {"promptTokenCount": 2539, "candidatesTokenCount": 8, "thoughtsTokenCount": 242}})

    client = genai.Client(vertexai=True, api_key="test",
                          http_options=types.HttpOptions(client_args={"transport": httpx.MockTransport(vertex)}))
    monkeypatch.setattr(backends, "_vertex", lambda: client)

    # Act
    reply = backends.gemini_chat("gemini-x", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                                 None, 120.0, temperature=0.0)

    # Assert: low thinking is on the wire, the thoughts are billed as output, and the read's
    # deadline is both the transport's and the server's
    (request,) = sent
    config = json.loads(request.content)["generationConfig"]
    assert {key.lower().replace("_", ""): value for key, value in config["thinkingConfig"].items()} == {
        "thinkinglevel": "LOW"}
    assert (reply.input_tokens, reply.output_tokens) == (2539, 250)
    assert request.extensions["timeout"]["read"] == 120.0 and request.headers["x-server-timeout"] == "120"


# ── A kill reaches a call in flight ───────────────────────────────────────────


def test_a_kill_reaches_a_read_in_flight_and_settles_its_hold(tmp_path: Path, never: NeverAnswers) -> None:
    # Arrange: a read in flight that would never answer
    halt, log = threading.Event(), calls.CallLog()
    backend = dataclasses.replace(_gemini_backend(tmp_path, [], reader=never), halt=halt)
    worker, raised = _in_thread(lambda: gen_batch.generate_life(SPEC, backend, log))
    assert never.entered.wait(10.0)

    # Act
    started = time.monotonic()
    halt.set()
    worker.join(5.0)

    # Assert: the attempt ended at once, the read given up on was priced at its worst case, and
    # the hold settled
    assert not worker.is_alive() and time.monotonic() - started < 2.0
    assert len(raised) == 1 and isinstance(raised[0], backends.Abandoned)
    (given_up,) = [call for call in log.calls if call.failed]
    assert given_up.kind == "leak" and given_up.usd == pytest.approx(GEMINI_RATE.worst)
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


def test_an_attempt_cut_short_by_a_kill_is_recorded_as_killed(tmp_path: Path, never: NeverAnswers) -> None:
    # Arrange: a life's read in flight that would never answer, the attempt's failures recorded
    halt, data = threading.Event(), DataDir(tmp_path / "pilot")
    log = calls.CallLog(on_failure=gen_batch.recorder(data, "generate-attempt", "the life"))
    backend = dataclasses.replace(_gemini_backend(tmp_path, [], reader=never), halt=halt)
    worker, raised = _in_thread(lambda: gen_batch.generate_life(SPEC, backend, log))
    assert never.entered.wait(10.0)

    # Act
    halt.set()
    worker.join(5.0)

    # Assert: the attempt the kill cut short has its record, marked killed, at the read in flight
    assert len(raised) == 1 and isinstance(raised[0], backends.Abandoned)
    (record,) = data.failed()
    assert (record["what"], record["attempt"], record["step"], record["error"], record["killed"]) == (
        "generate-attempt", 0, "leak reader", "Abandoned", True)
    assert "killed" in failures.line("the life", failures.AttemptFailure.from_json(record))


def test_a_read_given_up_on_keeps_its_limiter_place_until_it_really_ends(never: NeverAnswers,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: one place; a read past its deadline, still in flight
    monkeypatch.setattr(backends, "CALL_GRACE_S", 0.05)
    read = backends.bounded(never, None, 0.1, Limiter(1))
    leak = [{"role": "system", "content": "# leak_v1"}, {"role": "user", "content": "{}"}]
    with pytest.raises(ModelApiError, match="timed out"):
        read("gemini", leak, None, 5.0)

    # Act: a second read, while the first is still in flight
    second, raised = _in_thread(lambda: read("gemini", leak, None, 5.0))
    time.sleep(0.2)
    waited = len(never.timeouts)
    never.answer.set()
    never.release.set()  # the first read really ends, and its place frees
    second.join(5.0)

    # Assert: the second read reached the model only once the first had ended
    assert waited == 1
    assert len(never.timeouts) == 2 and raised == [] and not second.is_alive()


def test_reads_stuck_past_their_timeout_give_their_places_back_within_the_bound(
        never: NeverAnswers, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: two places, both taken by reads whose transport never times out
    monkeypatch.setattr(backends, "CALL_GRACE_S", 0.05)
    monkeypatch.setattr(backends, "LEAK_MARGIN_S", 0.1)
    log = calls.CallLog()
    read = backends.CountedChat(backends.bounded(never, None, 0.1, Limiter(2)), 10, None, log=log)
    leak = [{"role": "system", "content": "# leak_v1"}, {"role": "user", "content": "{}"}]
    for _ in range(2):
        with pytest.raises(ModelApiError, match="timed out"):
            read("gemini", leak, None, 5.0)

    # Act: a third read
    never.answer.set()
    started = time.monotonic()
    read("gemini", leak, None, 5.0)

    # Assert: it got a place within the transport timeout and the margin, not never, and each
    # stuck read whose place a timer freed is marked leaked
    assert time.monotonic() - started < 2.0
    marked = time.monotonic() + 2.0
    while not all(call.leaked for call in log.calls[:2]) and time.monotonic() < marked:
        time.sleep(0.02)
    assert [call.leaked for call in log.calls] == [True, True, False]


def test_a_read_that_ends_gives_its_place_back_once(never: NeverAnswers, monkeypatch: pytest.MonkeyPatch) -> None:
    # A read given up on that then ends frees its place itself, and its timer frees nothing more
    monkeypatch.setattr(backends, "CALL_GRACE_S", 0.05)
    monkeypatch.setattr(backends, "LEAK_MARGIN_S", 0.1)
    limit, log = Limiter(1), calls.CallLog()
    read = backends.CountedChat(backends.bounded(never, None, 0.1, limit), 10, None, log=log)
    leak = [{"role": "system", "content": "# leak_v1"}, {"role": "user", "content": "{}"}]
    with pytest.raises(ModelApiError, match="timed out"):
        read("gemini", leak, None, 5.0)
    never.answer.set()
    never.release.set()
    time.sleep(0.4)  # past the timer
    with limit.slot():
        assert not limit._slots.acquire(blocking=False)  # one place, never two
    assert not log.calls[0].leaked


def test_ctrl_c_with_a_call_that_never_answers_ends_without_a_stuck_join(
        tmp_path: Path, never: NeverAnswers, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: one life whose leak read never answers; Ctrl-C comes while the command waits on it
    real_backend = cli._fake_backend
    monkeypatch.setattr(cli, "_fake_backend", lambda data, provider: dataclasses.replace(
        real_backend(data, provider), reader=never))

    def interrupted(futures: Any) -> Iterator[Any]:
        assert never.entered.wait(10.0)
        raise KeyboardInterrupt
        yield  # a generator, as the results are

    monkeypatch.setattr(gen_batch, "_results", interrupted)
    data = tmp_path / "pilot"
    started = time.monotonic()

    # Act
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--fake", "--data", str(data), "--only", "1", "generate", "--split", "core-train"], out=lambda _: None)

    # Assert: it ended in moments, not when the read would have, with no worker left and every
    # hold settled
    assert time.monotonic() - started < 10.0
    assert not [t for t in threading.enumerate() if t.name.startswith(gen_batch.WORKER) and t.is_alive()]
    assert RunLedger(data / "fake-budget.jsonl").open_holds() == ()


# ── A 429 waits without spending the attempt's billed tries ──────────────────


class LimitedOnce:
    """A gate reader that answers 429 to the first try of every read, then answers it."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.lock, self.inner = threading.Lock(), fakes.FakeGateReader()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        key = json.dumps(messages, sort_keys=True)
        with self.lock:
            first = key not in self.seen
            self.seen.add(key)
        if first:
            raise RateLimited(VERTEX_429, retry_after_s=1.0)
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


def test_429s_on_more_reads_than_the_extra_tries_wait_within_one_attempt(tmp_path: Path) -> None:
    slept: list[float] = []
    reader = LimitedOnce()
    made = gen_batch.generate_life(SPEC, _gemini_backend(tmp_path, slept, reader=reader))
    assert made.attempts == 1 and retry.ATTEMPT_COOL_DOWN_S not in slept
    assert len(slept) == len(reader.seen) > backends.GEN_SPARK_RETRIES


def test_an_attempts_429_waits_stop_at_its_budget() -> None:
    # Each wait is taken from the attempt's budget; one that would pass it isn't waited
    budget = backends.WaitBudget(10.0)
    tries = backends.Retries(budget=budget, rand=lambda: 0.0)
    assert tries.wait(RateLimited(VERTEX_429, retry_after_s=8.0), 1) == 8.0 and budget.left_s == 2.0
    assert tries.wait(RateLimited(VERTEX_429, retry_after_s=8.0), 1) is None
    assert retry.RATE_LIMIT_BUDGET_S == 1200.0


def test_an_attempt_out_of_429_budget_fails_recorded_and_the_next_cools_down(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: every read answered 429 once, with a 30 s hint and no jitter, under a 100 s budget
    # an attempt
    monkeypatch.setattr(retry, "RATE_LIMIT_BUDGET_S", 100.0)
    monkeypatch.setattr(retry, "HINT_JITTER", 0.0)
    slept: list[float] = []
    data = DataDir(tmp_path / "pilot")

    class Hinted(LimitedOnce):
        def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                     temperature: float | None = None) -> ChatResult:
            try:
                return super().__call__(model, messages, tools, timeout_s, temperature=temperature)
            except RateLimited:
                raise RateLimited(VERTEX_429, retry_after_s=30.0) from None

    # Act
    with pytest.raises(RunsFailed):
        gen_batch.generate_lives(data, [SPEC], _gemini_backend(tmp_path, slept, reader=Hinted()), lambda _: None)

    # Assert: each attempt waited out 429s up to its budget, then failed and was recorded, and
    # each next attempt cooled down first
    waits = [wait for wait in slept if wait != retry.ATTEMPT_COOL_DOWN_S]
    assert slept.count(retry.ATTEMPT_COOL_DOWN_S) == generate.MAX_ATTEMPTS - 1
    assert waits == [30.0] * 3 * generate.MAX_ATTEMPTS
    records = [raw for raw in data.failed() if raw["what"] == "generate-attempt"]
    assert [(raw["attempt"], raw["error"], raw["rate_limits"]) for raw in records] == [
        (n, "RateLimited", 4) for n in range(generate.MAX_ATTEMPTS)]


class Scripted:
    """A gate reader that fails its first tries as scripted (a 429 or another error), then answers."""

    def __init__(self, *errors: str) -> None:
        self.errors, self.calls, self.inner = list(errors), 0, fakes.FakeGateReader()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        self.calls += 1
        if self.errors:
            error = self.errors.pop(0)
            raise RateLimited(error) if error == VERTEX_429 else ModelApiError(error)
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        return result


VERTEX_504 = "Gemini on Vertex returned HTTP 504"
LEAK = [{"role": "system", "content": "# leak_v1"}, {"role": "user", "content": "{}"}]


@pytest.mark.parametrize(("errors", "tries"), [
    ([VERTEX_504, VERTEX_429, VERTEX_504], 4),  # 2026-10-29: the 504 on try 3 ended the call at 3 of its 5
    ([VERTEX_504, VERTEX_429, VERTEX_504, VERTEX_429], 5),
    ([VERTEX_429, VERTEX_429, VERTEX_429, VERTEX_504], 5),
])
def test_a_5xx_after_a_429_is_tried_again_within_the_calls_five_tries(errors: list[str], tries: int) -> None:
    # Arrange
    chat = Scripted(*errors)
    counted = backends.CountedChat(chat, 1, None, backends.Retries(rand=lambda: 0.0, sleep=lambda _s: None))

    # Act
    counted("gemini", LEAK, None, 1.0)

    # Assert: answered on its last try; each 5xx spent one billed extra try, a 429 none
    assert chat.calls == tries and tries <= retry.RATE_LIMIT_ATTEMPTS
    assert counted.retries is not None
    assert counted.retries.left == backends.GEN_SPARK_RETRIES - errors.count(VERTEX_504)


@pytest.mark.parametrize(("errors", "tries"), [
    ([VERTEX_504] * 3, retry.ATTEMPTS),  # at most three tries of 5xx and timeouts, as ever
    ([VERTEX_429, VERTEX_504, VERTEX_504, VERTEX_504], 4),
    ([VERTEX_429, VERTEX_429, VERTEX_429, VERTEX_429, VERTEX_504], retry.RATE_LIMIT_ATTEMPTS),
])
def test_a_call_still_ends_at_three_5xx_or_five_tries_in_all(errors: list[str], tries: int) -> None:
    chat = Scripted(*errors, VERTEX_504)
    counted = backends.CountedChat(chat, 1, None, backends.Retries(rand=lambda: 0.0, sleep=lambda _s: None))
    with pytest.raises(ModelApiError, match="HTTP 504"):
        counted("gemini", LEAK, None, 1.0)
    assert chat.calls == tries


def test_a_429_never_spends_the_billed_extra_tries() -> None:
    tries = backends.Retries(left=0)
    assert tries.wait(RateLimited(VERTEX_429), 1) is not None and tries.left == 0
    assert tries.wait(RateLimited(VERTEX_429), retry.RATE_LIMIT_ATTEMPTS) is None  # its own call's tries still end
    assert tries.wait(ModelApiError("Gemini on Vertex timed out after 1s"), 1) is None  # that one is billed


# ── Each failed attempt recorded as it fails ──────────────────────────────────


def test_a_failed_attempt_is_recorded_before_its_life_ends(tmp_path: Path) -> None:
    # Arrange: the first attempt fails its leak gate; the second attempt's first write looks at
    # what is recorded so far
    data = DataDir(tmp_path / "pilot")
    flagged, peeked = list[bool](), list[int]()

    class FlagsOnce(fakes.FakeGateReader):
        def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                     temperature: float | None = None) -> ChatResult:
            if calls.kind_of(messages) == "leak" and not flagged:
                flagged.append(True)
                return ChatResult(json.dumps({"flag": True}), (), model, 10, 2, "stop")
            return super().__call__(model, messages, tools, timeout_s, temperature=temperature)

    class Peeks(fakes.FakeLifeChat):
        def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                     temperature: float | None = None) -> ChatResult:
            if flagged and not peeked:
                peeked.append(len(data.failed()))
            result: ChatResult = super().__call__(model, messages, tools, timeout_s, temperature=temperature)
            return result

    # Act
    gen_batch.generate_lives(data, [SPEC], _gemini_backend(tmp_path, [], reader=FlagsOnce(), life_chat=Peeks()),
                             lambda _: None)

    # Assert: attempt 1's failure was on disk while attempt 2 ran, and is recorded once
    assert peeked == [1]
    assert [raw["attempt"] for raw in data.failed()] == [0]


# ── ``calls`` tells failed tries from reads ───────────────────────────────────


def test_the_breakdown_tells_failed_tries_priced_at_their_worst_case_from_reads() -> None:
    read = calls.Call("leak", 0, 1, 1, 2539, 250, GEMINI_RATE.usd(2539, 250))
    failed = calls.Call("leak", 0, 1, 1, 0, 0, GEMINI_RATE.worst, failed=True)
    limited = calls.Call("leak", 0, 1, 1, 0, 0, 0.0, failed=True, rate_limited=True)
    lines = calls.breakdown({"life": [read, failed, failed, failed, limited, limited, limited]})
    (leak,) = [line for line in lines if line.startswith("  leak: ")]
    assert "$0.1687" in leak and "of it $0.1659 for 3 failed tries priced at their worst case" in leak
