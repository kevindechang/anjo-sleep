# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The network looking down stops a generation run (MIND's data). On 2026-10-29 the Mac lost its
network for 20 minutes and the run went through every remaining life, three attempts each, each
call failing in transport. Once ``NET_DOWN_ATTEMPTS`` attempts in a row across the run end on a
call that failed in transport (a refused or dropped connection, or a timeout, never an HTTP status:
``retry.in_transport``), no new attempt starts, the lives it stops are not recorded as failed, and
the command says so and exits ``NETWORK_DOWN``. An attempt that passes, or fails for any other
reason, starts the count again. Every model is a fake: no network, no spend."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import fakes, gen_batch, generate, retry
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.spark_provider import FIREWORKS
from anjo.sleep.model_api import ChatResult, ModelApiError

WRITE = "# life_v1"
DOWN = "Fireworks chat with deepseek failed in transport (ConnectError)"  # model_api's own message, batch 7


@dataclasses.dataclass
class Down:
    """The fakes' generation chat, every writer call of an attempt whose seed (``variation``) is in
    ``seeds`` failing in transport, or every writer call at all when ``seeds`` is ``None``; each
    writer call's seed kept."""

    seeds: frozenset[int] | None = None
    inner: fakes.FakeLifeChat = dataclasses.field(default_factory=fakes.FakeLifeChat)
    writes: list[int] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if str(messages[0]["content"]).startswith(WRITE):
            seed = json.loads(messages[-1]["content"])["variation"]
            self.writes.append(seed)
            if self.seeds is None or seed in self.seeds:
                raise ModelApiError(DOWN)
        return self.inner(model, messages, tools, timeout_s, temperature=temperature)


def _backend(root: Path, chat: Down) -> backends.Backend:
    return dataclasses.replace(backends.fake_backend(FIREWORKS), life_chat=chat, sleep=lambda _s: None,
                               ledger=RunLedger(root / "budget.jsonl"), prices=cli.FAKE_PRICES, rates=cli.FAKE_RATES)


@pytest.mark.parametrize(("message", "expected"), [
    (DOWN, True),
    ("Gemini on Vertex failed in transport (ConnectError)", True),
    ("Fireworks chat with deepseek failed in transport (ReadError)", True),
    ("Fireworks chat with deepseek timed out after 470s", True),
    ("Fireworks chat with deepseek returned HTTP 429", False),
    ("Gemini on Vertex returned HTTP 503", False),
    ("Fireworks chat with deepseek returned HTTP 400", False),
    ("Fireworks chat with deepseek failed before a reply (InvalidURL)", False),
])
def test_in_transport_is_the_network_and_never_a_status(message: str, expected: bool) -> None:
    assert retry.in_transport(ModelApiError(message)) is expected
    assert retry.in_transport(gen_batch.CallFailed(message)) is expected  # an ended attempt keeps the message
    assert not expected or retry.transient(ModelApiError(message))  # every one is tried again within its call first


def test_the_breaker_trips_after_n_attempts_in_a_row_failed_in_transport() -> None:
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
    assert str(stopped.value) == f"3 attempts in a row failed in transport (last: {DOWN})"


@pytest.mark.parametrize("other", [
    None,  # the attempt passed, or failed a rule or a gate
    gen_batch.CallFailed("Fireworks chat with deepseek returned HTTP 503"),  # busy, not down
    ModelApiError("Fireworks chat with deepseek returned HTTP 400"),
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
    assert lines[-1] == (f"Stopped: 3 attempts in a row failed in transport (last: {DOWN}); the network looks down. "
                         "What finished is kept; rerun to go on.")
    assert len(set(chat.writes)) == gen_batch.NET_DOWN_ATTEMPTS
    assert not any(line.startswith("SKIPPED") for line in lines)
    recorded = DataDir(data).failed()
    assert [raw["what"] for raw in recorded] == ["generate-attempt"] * gen_batch.NET_DOWN_ATTEMPTS
    assert DataDir(data).lives("development") == ()
    assert RunLedger(data / "budget.jsonl").open_holds() == ()
