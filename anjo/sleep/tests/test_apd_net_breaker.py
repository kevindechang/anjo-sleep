# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The network looking down stops a generation run (MIND's data). On 2026-10-29 the Mac lost its
network for 20 minutes and the run went through every remaining life, three attempts each, each
call failing in transport. Once ``NET_DOWN_ATTEMPTS`` attempts in a row across the run end on a
call that never connected (refused, its name not found, its network unreachable, or its connect
timing out: ``retry.connect_failed``), no new attempt starts, the lives and pairs it stops are not
recorded as failed, and the command says so and exits ``NETWORK_DOWN``. A slow reply (a read
timeout, a call's own deadline) or a dropped connection is no sign of a network that is down: it
starts the count again, as an attempt that passes or fails for any other reason does. Once tripped,
the breaker stays tripped. Every model is a fake: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import fakes, gen_batch, generate, retry
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult, ModelApiError, post_chat

WRITE = "# life_v1"
DOWN = "Fireworks chat with deepseek failed in transport (ConnectError)"  # model_api's own message, batch 7
REFUSED = "Fireworks chat with deepseek returned HTTP 400"  # a call failed for good: its life is skipped
SLOW = "Gemini on Vertex timed out after 120s"  # a reader's reply too slow: no sign of a network down


@dataclasses.dataclass
class Down:
    """The fakes' generation chat, every writer call of an attempt whose seed (``variation``) is in
    ``seeds`` failing in transport, or every writer call at all when ``seeds`` is ``None``; each
    writer call's seed kept."""

    seeds: frozenset[int] | None = None
    refused: frozenset[int] = frozenset()  # attempts whose writer is answered a 400 instead
    inner: fakes.FakeLifeChat = dataclasses.field(default_factory=fakes.FakeLifeChat)
    writes: list[int] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if str(messages[0]["content"]).startswith(WRITE):
            seed = json.loads(messages[-1]["content"])["variation"]
            self.writes.append(seed)
            if seed in self.refused:
                raise ModelApiError(REFUSED)
            if self.seeds is None or seed in self.seeds:
                raise ModelApiError(DOWN)
        return self.inner(model, messages, tools, timeout_s, temperature=temperature)


@dataclasses.dataclass
class SlowReader:
    """The fakes' gate reader, every call timing out on its reply as Vertex's port says it."""

    calls: int = 0

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        self.calls += 1
        raise ModelApiError(SLOW)


def _backend(root: Path, chat: Down, reader: SlowReader | None = None) -> backends.Backend:
    found = dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=chat, sleep=lambda _s: None,
                                ledger=RunLedger(root / "budget.jsonl"), prices=cli.FAKE_PRICES, rates=cli.FAKE_RATES)
    return found if reader is None else dataclasses.replace(found, reader=reader)


def _stopped_line() -> str:
    return (f"Stopped: 3 attempts in a row failed to connect (last: {DOWN}); the network looks down. "
            "What finished is kept; rerun to go on.")


def _tripped() -> gen_batch.Breaker:
    breaker = gen_batch.Breaker()
    for _ in range(gen_batch.NET_DOWN_ATTEMPTS):
        breaker.ended(gen_batch.CallFailed(DOWN))
    return breaker


@pytest.mark.parametrize(("message", "expected"), [
    (DOWN, True),  # refused, its name not found or its network unreachable: httpx's ConnectError
    ("Gemini on Vertex failed in transport (ConnectError)", True),
    ("Fireworks chat with deepseek failed in transport (ConnectTimeout)", True),  # no connection made in time
    ("Fireworks chat with deepseek failed in transport (ReadError)", False),  # connected, then dropped
    ("Fireworks chat with deepseek failed in transport (RemoteProtocolError)", False),
    ("Fireworks chat with deepseek timed out after 470s", False),  # a reply too slow
    (SLOW, False),
    ("a generation call timed out after 600s (its deadline)", False),  # our own deadline
    ("Fireworks chat with deepseek returned HTTP 429", False),
    ("Gemini on Vertex returned HTTP 503", False),
    (REFUSED, False),
    ("Fireworks chat with deepseek failed before a reply (InvalidURL)", False),
])
def test_connect_failed_is_a_connection_never_made_never_a_slow_or_dropped_reply(message: str, expected: bool) -> None:
    assert retry.connect_failed(ModelApiError(message)) is expected
    assert retry.connect_failed(gen_batch.CallFailed(message)) is expected  # an ended attempt keeps the message
    assert not expected or retry.transient(ModelApiError(message))  # every one is tried again within its call first


def test_a_connect_timeout_is_named_a_failed_connection_and_still_tried_again(monkeypatch: pytest.MonkeyPatch) -> None:
    # httpx's ConnectTimeout is a TimeoutException: model_api and the Vertex port name it apart from a slow reply
    def connect_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("down", request=request)

    monkeypatch.setenv("APD_BREAKER_TEST_KEY", "k")
    with pytest.raises(ModelApiError) as failed:
        post_chat(provider="Fireworks", url="https://example.invalid/v1/chat/completions", key_env="APD_BREAKER_TEST_KEY",
                  model="deepseek", messages=[{"role": "user", "content": "y"}], tools=None, timeout_s=5.0,
                  temperature=0.0, transport=httpx.MockTransport(connect_timeout))
    assert str(failed.value) == "Fireworks chat with deepseek failed in transport (ConnectTimeout)"
    assert retry.connect_failed(failed.value) and retry.transient(failed.value)
    vertex = backends._vertex_error(httpx.ConnectTimeout("down"), 120.0)
    assert str(vertex) == "Gemini on Vertex failed in transport (ConnectTimeout)" and retry.transient(vertex)
    assert str(backends._vertex_error(httpx.ReadTimeout("slow"), 120.0)) == SLOW  # a slow reply named as before


def test_the_breaker_trips_after_n_attempts_in_a_row_failed_to_connect() -> None:
    # Arrange
    breaker = gen_batch.Breaker()

    # Act: one short of the count, then the last
    for _ in range(gen_batch.NET_DOWN_ATTEMPTS - 1):
        breaker.ended(gen_batch.CallFailed(DOWN))
    breaker.check()  # still closed
    breaker.ended(gen_batch.CallFailed(DOWN))

    # Assert
    with pytest.raises(gen_batch.NetworkDown) as stopped:
        breaker.check()
    assert gen_batch.NET_DOWN_ATTEMPTS == 3
    assert str(stopped.value) == f"3 attempts in a row failed to connect (last: {DOWN})"


@pytest.mark.parametrize("other", [
    None,  # the attempt passed, or failed a rule or a gate
    gen_batch.CallFailed("Fireworks chat with deepseek returned HTTP 503"),  # busy, not down
    gen_batch.CallFailed(SLOW),  # slow, not down
    gen_batch.CallFailed("Fireworks chat with deepseek failed in transport (ReadError)"),  # dropped, not down
    ModelApiError(REFUSED),
])
def test_an_attempt_ending_any_other_way_starts_the_count_again(other: Exception | None) -> None:
    breaker = gen_batch.Breaker()
    for _ in range(gen_batch.NET_DOWN_ATTEMPTS - 1):
        breaker.ended(gen_batch.CallFailed(DOWN))
    breaker.ended(other)
    for _ in range(gen_batch.NET_DOWN_ATTEMPTS - 1):
        breaker.ended(gen_batch.CallFailed(DOWN))
    breaker.check()  # still closed: the count started again
    breaker.ended(gen_batch.CallFailed(DOWN))
    with pytest.raises(gen_batch.NetworkDown):
        breaker.check()


def test_once_tripped_it_stays_tripped_whatever_ends_after() -> None:
    # Arrange: tripped, then an attempt in flight on another thread passes
    breaker = _tripped()

    # Act
    breaker.ended(None)

    # Assert
    with pytest.raises(gen_batch.NetworkDown, match="3 attempts in a row failed to connect"):
        breaker.check()


def test_a_life_given_a_tripped_breaker_makes_no_call(tmp_path: Path) -> None:
    chat = Down(frozenset())  # healthy
    with pytest.raises(gen_batch.NetworkDown):
        gen_batch.generate_life(generate.development_specs()[0], _backend(tmp_path, chat), breaker=_tripped())
    assert chat.writes == []


def test_a_life_alone_without_a_run_breaker_is_skipped_as_before(tmp_path: Path) -> None:
    # generate_life called on its own (no run): three attempts failing in transport skip the life
    with pytest.raises(generate.GenerationError, match="failed in transport"):
        gen_batch.generate_life(generate.development_specs()[0], _backend(tmp_path, Down()))


def test_lives_passing_between_transport_failures_never_trip_it(tmp_path: Path) -> None:
    # Arrange: two lives, one at a time, each failing its first two attempts in transport and passing
    # its third; without a reset the fourth failure in transport would trip the breaker
    specs = generate.development_specs()[:2]
    chat = Down(frozenset(generate.attempt_seed(spec.seed, attempt) for spec in specs for attempt in (0, 1)))
    data, lines = DataDir(tmp_path / "data"), list[str]()

    # Act
    made = gen_batch.generate_lives(data, specs, _backend(tmp_path, chat), lines.append, parallel=1)

    # Assert
    assert len(made) == 2 and not any(line.startswith("SKIPPED") for line in lines)


def test_the_command_stops_with_its_own_code_and_line_and_records_no_life_as_failed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the network is down for every writer call
    chat, data = Down(), tmp_path / "pilot"
    monkeypatch.setattr(cli, "_fake_backend", lambda found, _provider: _backend(found.root, chat))
    lines: list[str] = []

    # Act
    code = cli.main(["--fake", "--data", str(data), "--only", "4", "generate", "--parallel", "1"], out=lines.append)

    # Assert: three attempts in all, the first life's, then the stop; no life skipped or recorded failed
    assert code == cli.NETWORK_DOWN and cli.NETWORK_DOWN not in (0, cli.REFUSED_CODE, cli.STOPPED, cli.BUSY, cli.CEILING)
    assert lines[-1] == _stopped_line()
    assert len(set(chat.writes)) == gen_batch.NET_DOWN_ATTEMPTS
    assert not any(line.startswith(("SKIPPED", "Before the stop")) for line in lines)
    recorded = DataDir(data).failed()
    assert [raw["what"] for raw in recorded] == ["generate-attempt"] * gen_batch.NET_DOWN_ATTEMPTS
    assert DataDir(data).lives("development") == ()
    assert RunLedger(data / "budget.jsonl").open_holds() == ()


def test_lives_skipped_before_the_stop_are_named_before_it(tmp_path: Path) -> None:
    # Arrange: the first life's writer refused for good (a 400: skipped at once), then the network down
    specs = generate.development_specs()[:2]
    chat = Down(refused=frozenset({generate.attempt_seed(specs[0].seed, 0)}))
    data, lines = DataDir(tmp_path / "data"), list[str]()

    # Act
    with pytest.raises(gen_batch.NetworkDown):
        gen_batch.generate_lives(data, specs, _backend(tmp_path, chat), lines.append, parallel=1)

    # Assert: the skip is recorded and named again once the run stops, the stopped life is not
    name = gen_batch.spec_name(specs[0])
    assert sum(line.startswith("SKIPPED") for line in lines) == 1
    assert lines[-1] == f"Before the stop, 1 failed and weren't saved ({name}); a rerun generates them again."
    assert [raw.get("life") for raw in data.failed() if raw["what"] == "generate"] == [name]


def test_a_slow_reader_never_trips_it_and_its_lives_are_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the writer healthy, every reader call timing out on its reply, as a slow Vertex would
    chat, reader, data = Down(frozenset()), SlowReader(), tmp_path / "pilot"
    monkeypatch.setattr(cli, "_fake_backend", lambda found, _provider: _backend(found.root, chat, reader))
    lines: list[str] = []

    # Act
    code = cli.main(["--fake", "--data", str(data), "--only", "4", "generate", "--parallel", "1"], out=lines.append)

    # Assert: the first life plants nothing, so no reader is asked and it passes; then nine attempts in
    # a row end on a reply too slow, three times the breaker's count, not one counted as the network
    # down: each of the three lives is skipped, and the run stops as a run with skipped lives does
    assert code == cli.STOPPED and reader.calls > 0
    assert sum(line.startswith("FAILED attempt") and line.endswith(SLOW) for line in lines) == 3 * generate.MAX_ATTEMPTS
    assert sum(line.startswith("SKIPPED") for line in lines) == 3
    assert not any("network looks down" in line or line.startswith("Before the stop") for line in lines)


def test_lives_side_by_side_stop_with_every_hold_settled_once_and_nothing_written(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: four lives at once, the network down for every writer call
    chat, data = Down(), tmp_path / "pilot"
    monkeypatch.setattr(cli, "_fake_backend", lambda found, _provider: _backend(found.root, chat))
    lines: list[str] = []

    # Act
    code = cli.main(["--fake", "--data", str(data), "--only", "4", "generate", "--parallel", "4"], out=lines.append)

    # Assert: the attempts begun before the trip at most; each hold settled or released once, and
    # no life, made record or skip written (call logs and attempt rows are kept, by design)
    assert code == cli.NETWORK_DOWN and lines[-1] == _stopped_line()
    assert gen_batch.NET_DOWN_ATTEMPTS <= len(set(chat.writes)) <= gen_batch.NET_DOWN_ATTEMPTS + 4 - 1
    events = read_records(data / "budget.jsonl", PLAIN)
    held = [event["job"] for event in events if event["op"] == "reserve"]
    ended = Counter(event["job"] for event in events if event["op"] in ("settle", "release"))
    assert held and len(set(held)) == len(held) and ended == Counter(held)
    assert RunLedger(data / "budget.jsonl").open_holds() == ()
    assert DataDir(data).lives("development") == () and not (data / "made").exists()
    assert {raw["what"] for raw in DataDir(data).failed()} == {"generate-attempt"}
    assert not any(line.startswith("SKIPPED") for line in lines)


def test_pairs_stop_once_attempts_in_a_row_fail_to_connect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the network down for every writer call of generate-pairs
    chat, data = Down(), tmp_path / "pilot"
    monkeypatch.setattr(cli, "_fake_backend", lambda found, _provider: _backend(found.root, chat))
    lines: list[str] = []

    # Act
    code = cli.main(["--fake", "--data", str(data), "generate-pairs", "--n", "3", "--parallel", "1"], out=lines.append)

    # Assert: three attempts in all, the first pair's, then the stop; no pair skipped or recorded failed
    assert code == cli.NETWORK_DOWN and lines[-1] == _stopped_line()
    assert len(set(chat.writes)) == gen_batch.NET_DOWN_ATTEMPTS
    assert not any(line.startswith("SKIPPED") for line in lines)
    assert [raw["what"] for raw in DataDir(data).failed()] == ["generate-pairs-attempt"] * gen_batch.NET_DOWN_ATTEMPTS
    assert DataDir(data).pairs() == ()
    assert RunLedger(data / "budget.jsonl").open_holds() == ()
