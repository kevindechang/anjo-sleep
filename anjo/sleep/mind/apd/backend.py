"""Where the APD pilot's calls go (Task 17): offline fakes (``--fake``) or the live services.

Live, and only after ``--live --yes``:

- **Glimmer:** Colab vLLM behind an SSH tunnel (``colab_vllm``), every CLI call under its timeout,
  each session on a throwaway key made in the data dir's ``ssh`` folder, after a ``preflight`` of
  the CLI, ``ssh`` and ``ssh-keygen``;
- **Spark Open** on the experiment's provider (``mind.spark_provider``: Fireworks by default, or
  Baseten): the life writer and extractor, the gates' readers, the screen, the resolver, 17b's
  labels and the intervention test's writer;
- **Muse Spark** (Meta's API): the gates' auditor and the APD judge;
- **Gemini** (Vertex, ``probes.second_judge.MODEL_ID``): the rescoring judge. Its SDK is imported
  only when a live judge first calls it, with Vertex's own application-default credentials. Gemini
  3.x can't turn thinking off, and 3.7 and 3.8 refuse ``minimal``, so it asks for ``low`` and
  reserves room for the thought tokens on top of the reply's cap; they are billed as output, so
  they are counted with it.

Every paid call is counted, priced and held through the shared budget ledger. Generation holds
each attempt's worst case before it starts (``GenCalls``, sized to the life's planted items by
``life_calls``): its Spark Open calls and ``GEN_SPARK_RETRIES`` extra tries on the provider's
pool, and its Muse Spark calls on ``meta_usd`` (a call past either count is refused), the most one
attempt can make: a writer and an extractor call a day, every leak window of both views and a
discoverability read for each planted item, an identify call for each and the script audit. A transient Spark Open failure is
tried again from the extra tries (``retry``), each try a call of its own, and a failed try is
priced at one call's worst case, so the hold stays a true bound. Every generation reply goes out
capped at ``retry.REPLY_MAX_TOKENS`` (``gen_chat``), the cap one call is held for. Side by side, an
attempt whose hold finds no room waits for the command's other attempts in flight to end
(``Flight``), and once the command is ending (``halt``) no attempt or call starts (``Halted``). A
core split's generator (``generators``) writes and extracts on Fireworks at its own list prices:
an attempt holds its writes at those and its gate reads at Spark Open's, on the same pool.
"""
from __future__ import annotations

import functools
import importlib
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from anjo.sleep import fireworks_api, meta_api
from anjo.sleep.budget import Ledger, Pool
from anjo.sleep.context.brief import ChatFn as BriefChat
from anjo.sleep.mind.apd import colab_vllm, fakes, retry
from anjo.sleep.mind.apd.ceiling import Ceiling
from anjo.sleep.mind.apd.colab_vllm import ColabSession, TunnelOpener
from anjo.sleep.mind.apd.generators import BY_MODEL, SPARK, Generator
from anjo.sleep.mind.apd.judging import MUSE, Rate, TextModel
from anjo.sleep.mind.apd.types import LIFE_DAYS, MAX_PREVALENCE
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import held, usd
from anjo.sleep.mind.spark_provider import BASETEN, SparkProvider
from anjo.sleep.probes import second_judge
from anjo.sleep.spark_inquiry import ChatFn
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train.runners.colab import COLAB_BIN, ColabCli



def item_reads(pre_surface_days: int) -> int:
    """The gates' Spark Open reads of one item read over ``pre_surface_days`` days (its surface
    day's index): every day and every two adjacent days in both views (``gates._leak``), and its
    discoverability read. A life that passes makes all of them."""
    if pre_surface_days < 1:
        raise ValueError("an item surfaces after at least one day")
    return 2 * (2 * pre_surface_days - 1) + 1


GEN_WRITE_CALLS: Final = 2 * LIFE_DAYS  # the writer and the extractor, once a day
GEN_READS_PER_ITEM: Final = item_reads(LIFE_DAYS)  # the hold's bound per item: more than any surface day's
GEN_SPARK_RETRIES: Final = 6  # extra tries one attempt holds for transient Spark Open failures, shared by its calls
JUDGE_TIMEOUT_S: Final = 120.0
JUDGE_MAX_TOKENS: Final = 400
GEMINI_LOCATION: Final = "global"
GEMINI_THINKING: Final = "low"  # the floor 3.7 and 3.8 accept on Vertex
GEMINI_THINKING_RESERVE: Final = 512  # thought tokens come out of max_output_tokens on 3.x
ROOM_POLL_S: Final = 0.1  # how often an attempt waiting for room looks for a kill


@dataclass(frozen=True)
class Backend:
    """Every model and machine the pilot reaches, and where its spend is held (``ledger`` and the
    prices: ``None`` with the fakes, which hold nothing). ``provider`` serves every Spark Open call
    (``spark``, ``life_chat`` and ``reader``), held on its pool. ``ceiling`` is the command's
    ``--max-usd``, checked before each new unit of work. Generating side by side, ``halt`` is set
    once the command is ending (no attempt or call starts after it) and ``flight`` counts the
    attempts holding, so one whose hold finds no room waits for them (``held_attempt``)."""

    cli: ColabCli
    tunnels: TunnelOpener
    wait: Callable[[ColabSession], bool]
    spark: ChatFn
    life_chat: BriefChat
    reader: BriefChat
    auditor: BriefChat
    muse: TextModel
    gemini: TextModel
    provider: SparkProvider = BASETEN
    ledger: Ledger | None = None
    prices: Prices | None = None
    rates: Mapping[str, Rate] | None = None
    ceiling: Ceiling | None = field(default=None, compare=False)
    fake: bool = False
    claude: Callable[[str], str] | None = None  # the fakes' stand-in for Claude's hand audit
    vm: fakes.FakeColab | None = field(default=None, compare=False)
    clock: Callable[[], float] = time.time  # the session's wall clock: its deadline and its CU
    sleep: Callable[[float], None] = time.sleep
    keygen: colab_vllm.KeyGen = colab_vllm.ssh_keygen  # each session's throwaway SSH key
    key_dir: Path | None = None  # where it is made and deleted (live: the data dir's ``ssh`` folder)
    preflight: Callable[[], None] = lambda: None  # live: the CLI, ``ssh`` and ``ssh-keygen`` are here
    halt: threading.Event | None = field(default=None, compare=False)
    flight: Flight | None = field(default=None, compare=False)
    generator: Generator = SPARK  # who writes and extracts a core split's lives and the pairs (``generators``)
    generator_rate: Rate | None = None  # its list prices, when it isn't Spark Open


def _text(complete: Callable[[str], str]) -> TextModel:
    def model(prompt: str) -> tuple[str, int, int]:
        return complete(prompt), len(prompt) // 4, 16
    return model


def fake_backend(provider: SparkProvider = BASETEN) -> Backend:
    """Everything offline: the fake VM, Glimmer, Spark Open (answering as ``provider``'s model), the
    gates and the judges."""
    vm = fakes.FakeColab()
    return Backend(
        cli=ColabCli(Path("/fake/colab"), run=vm), tunnels=fakes.fake_tunnels(vm),
        wait=functools.partial(colab_vllm.ready, clock=fakes.Ticks(), sleep=lambda _s: None),
        spark=fakes.FakeSparkChat(model=provider.model), life_chat=fakes.FakeLifeChat(), reader=fakes.FakeGateReader(),
        auditor=fakes.FakeGateAuditor(), muse=_text(fakes.fake_judge(strict=False)),
        gemini=_text(fakes.fake_judge(strict=True)), provider=provider, fake=True, claude=fakes.fake_judge(strict=False),
        vm=vm,
        clock=fakes.Ticks(1.0), sleep=lambda _s: None, keygen=fakes.fake_keygen,
    )


# ── Live ──────────────────────────────────────────────────────────────────────


def gen_chat(provider: SparkProvider) -> BriefChat:
    """Generation's calls: Muse Spark on Meta's API for the auditor, a core split's generator on
    Fireworks for its own id (``generators``), Spark Open on ``provider`` else, every reply capped at
    ``retry.REPLY_MAX_TOKENS``, the most one call is held for."""

    def chat(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
             temperature: float | None = None) -> ChatResult:
        cap = retry.REPLY_MAX_TOKENS
        if model == "muse-spark":
            return meta_api.chat("muse-spark", messages, tools, timeout_s, temperature=temperature, max_tokens=cap)
        if model in BY_MODEL:
            return fireworks_api.generate(model, messages, tools, timeout_s, temperature=temperature, max_tokens=cap,
                                          reasoning_effort=BY_MODEL[model].reasoning_effort)
        return provider.write(model, messages, tools, timeout_s, temperature=temperature, max_tokens=cap)

    return chat


def _muse(prompt: str) -> tuple[str, int, int]:
    result = meta_api.chat("muse-spark", [{"role": "user", "content": prompt}], None, JUDGE_TIMEOUT_S,
                           temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)
    return result.text, result.input_tokens, result.output_tokens


@functools.cache
def _vertex() -> Any:
    genai = importlib.import_module("google.genai")
    return genai.Client(vertexai=True, location=GEMINI_LOCATION)


def _gemini(prompt: str) -> tuple[str, int, int]:
    model, types = second_judge.MODEL_ID, importlib.import_module("google.genai.types")
    config = types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=JUDGE_MAX_TOKENS + GEMINI_THINKING_RESERVE,
        thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING))
    try:
        reply = _vertex().models.generate_content(model=model, contents=prompt, config=config)
    except Exception as error:  # noqa: BLE001 - the SDK's errors share no base; each is a failed call
        raise ModelApiError(f"Gemini on Vertex: {type(error).__name__}") from None
    usage = reply.usage_metadata
    out = int(usage.candidates_token_count or 0) + int(usage.thoughts_token_count or 0)  # both billed as output
    return reply.text or "", int(usage.prompt_token_count or 0), out


def live_backend(root: Path, ledger: Ledger, prices: Prices, rates: Mapping[str, Rate],
                 provider: SparkProvider) -> Backend:
    """The live services, Spark Open on ``provider``, held on ``ledger`` at ``prices`` and the
    judges' ``rates``."""
    chat = gen_chat(provider)
    return Backend(
        cli=ColabCli(), tunnels=colab_vllm.ssh_tunnel(root / "ssh"), wait=colab_vllm.ready, spark=provider.chat,
        life_chat=chat, reader=chat, auditor=chat, muse=_muse, gemini=_gemini, provider=provider,
        ledger=ledger, prices=prices, rates=rates, key_dir=root / "ssh",
        preflight=functools.partial(colab_vllm.preflight, COLAB_BIN),
    )


# ── Generation's holds ────────────────────────────────────────────────────────


class Halted(RuntimeError):
    """The command is ending (Ctrl-C, a signal, an error): no attempt or call starts."""


def halted(halt: threading.Event | None) -> None:
    if halt is not None and halt.is_set():
        raise Halted("the command is ending: no further attempt or call")


class Flight:
    """The attempts of one command taking or holding their holds, side by side. An attempt whose
    hold finds no room waits for one of them to end, and stops the run only when none is in flight."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._flying = 0
        self._ended = 0

    def board(self) -> int:
        """One more attempt in flight (counted while it reserves); how many have ended so far."""
        with self._changed:
            self._flying += 1
            return self._ended

    def land(self, *, held: bool) -> None:
        """An attempt out of flight; one that held has freed its room."""
        with self._changed:
            self._flying -= 1
            self._ended += int(held)
            self._changed.notify_all()

    def room(self, since: int, halt: threading.Event | None) -> bool:
        """After a hold found no room: once an attempt that held has ended since ``since``, True (try
        again); False at once when no other attempt is in flight. ``Halted`` if the command ends."""
        with self._changed:
            while self._ended == since:
                if self._flying == 0:
                    return False
                halted(halt)
                self._changed.wait(ROOM_POLL_S)
            return True


@dataclass
class Retries:
    """The extra tries one attempt holds for transient failures (a 429, a 5xx, a timeout or a
    reset), shared by its Spark Open calls; each waits its backoff (``retry.backoff``) first."""

    left: int = GEN_SPARK_RETRIES
    sleep: Callable[[float], None] = time.sleep
    rand: Callable[[], float] = random.random

    def take(self, error: ModelApiError, attempt: int) -> bool:
        """Whether try ``attempt``'s failure is tried again: transient, under ``retry.ATTEMPTS``
        tries of that call, and an extra try still held. Waits the backoff when it is."""
        if attempt >= retry.ATTEMPTS or not retry.transient(error) or self.left < 1:
            return False
        self.left -= 1
        self.sleep(retry.backoff(attempt, self.rand))
        return True


@dataclass
class CountedChat:
    """Generation's calls to one model family: no call past ``limit`` (the attempt's hold), each
    try priced at ``rate`` as it returns, or at one call's worst case when it fails (what it cost
    is unknown). A transient failure is tried again while ``retries`` allows. No try starts once
    ``halt`` is set (``Halted``)."""

    inner: BriefChat
    limit: int
    rate: Rate | None
    retries: Retries | None = None
    calls: int = 0
    usd: float = 0.0
    halt: threading.Event | None = None

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if self.calls >= self.limit:
            raise ModelApiError("past the calls this attempt holds")
        self.calls += 1
        attempt = 1
        while True:
            halted(self.halt)
            try:
                result = self.inner(model, messages, tools, timeout_s, temperature=temperature)
            except ModelApiError as error:
                self.usd += 0.0 if self.rate is None else self.rate.worst
                if self.retries is None or not self.retries.take(error, attempt):
                    raise
                attempt += 1
                continue
            if self.rate is not None:
                self.usd += self.rate.usd(result.input_tokens, result.output_tokens)
            return result


def spark_rate(prices: Prices | None) -> Rate | None:
    return None if prices is None else Rate(prices.spark_in_per_m, prices.spark_out_per_m)


@dataclass(frozen=True)
class GenCalls:
    """The most one generation attempt calls: the writer and extractor (``writes``) and the gates'
    readers (``reads``) on Spark Open, with ``retries`` extra tries they share, and the auditor
    (``muse``) on Muse Spark."""

    writes: int
    reads: int
    muse: int
    retries: int = GEN_SPARK_RETRIES

    @property
    def spark(self) -> int:
        return self.writes + self.reads + self.retries


def life_calls(prevalence: int) -> GenCalls:
    """One life's attempt with ``prevalence`` planted items: a writer and an extractor call a day,
    each item's reads and identify call, and the script audit."""
    if not 0 <= prevalence <= MAX_PREVALENCE:
        raise ValueError(f"prevalence must be 0 to {MAX_PREVALENCE}")
    return GenCalls(GEN_WRITE_CALLS, prevalence * GEN_READS_PER_ITEM, prevalence + 1)


WORST_LIFE: Final = life_calls(MAX_PREVALENCE)


def gen_holds(prices: Prices | None, rates: Mapping[str, Rate] | None, spark_pool: Pool = BASETEN.pool,
              calls: GenCalls = WORST_LIFE, write_rate: Rate | None = None) -> dict[Pool, float]:
    """One generation attempt's worst case on each pool: on ``spark_pool``, its writes at
    ``write_rate`` (a generator's, or Spark Open's when ``None``), its reads at Spark Open's, and
    each extra try at the dearer of the two."""
    spark, muse = spark_rate(prices), None if rates is None else rates[MUSE]
    writes = write_rate or spark
    on_pool = 0.0 if spark is None or writes is None else (
        calls.writes * writes.worst + calls.reads * spark.worst + calls.retries * max(writes.worst, spark.worst))
    return {spark_pool: on_pool, "meta_usd": muse.worst * calls.muse if muse else 0.0}


def write_rate(backend: Backend) -> Rate | None:
    """What the writer and extractor cost: the generator's prices, or Spark Open's."""
    if backend.generator.model is None:
        return spark_rate(backend.prices)
    if backend.prices is not None and backend.generator_rate is None:
        raise ValueError(f"--generator {backend.generator.name} needs its prices to be held")
    return backend.generator_rate


@contextmanager
def held_attempt(backend: Backend, key: str,
                 calls: GenCalls = WORST_LIFE) -> Iterator[tuple[CountedChat, CountedChat, CountedChat]]:
    """One attempt's writer, reader and auditor, at most ``calls``, its worst case held first on
    both pools (Spark Open's on the provider's, with the extra tries its Spark Open calls share).
    Nothing starts once ``backend.halt`` is set; a hold without room waits (``_room``)."""
    halted(backend.halt)
    retries = Retries(calls.retries, sleep=backend.sleep)
    writes = write_rate(backend)
    spark = CountedChat(backend.life_chat, calls.writes, writes, retries, halt=backend.halt)
    reader = CountedChat(backend.reader, calls.reads, spark_rate(backend.prices), retries, halt=backend.halt)
    auditor = CountedChat(backend.auditor, calls.muse, None if backend.rates is None else backend.rates[MUSE],
                          halt=backend.halt)
    if backend.ledger is None:
        yield spark, reader, auditor
        return
    pool, ledger = backend.provider.pool, backend.ledger
    worst = gen_holds(backend.prices, backend.rates, pool, calls, writes)

    def holds(stack: ExitStack) -> None:
        stack.enter_context(held(ledger, f"apd:gen:{key}:spark", (pool,), usd(worst[pool]),
                                 lambda: spark.usd + reader.usd))
        stack.enter_context(held(ledger, f"apd:gen:{key}:muse", ("meta_usd",), usd(worst["meta_usd"]),
                                 lambda: auditor.usd))

    with _room(backend, holds):
        yield spark, reader, auditor


@contextmanager
def _room(backend: Backend, take: Callable[[ExitStack], None]) -> Iterator[None]:
    """The attempt's holds, taken together. When one finds no room, any already taken is let go and,
    while other attempts of the command are in flight (``backend.flight``), it waits for one to end
    and tries again; with none, ``BudgetStop``."""
    flight = backend.flight or Flight()  # alone, nothing is in flight to wait for
    while True:
        halted(backend.halt)
        since, stack = flight.board(), ExitStack()
        try:
            take(stack)
        except BudgetStop:
            stack.close()
            flight.land(held=False)
            if flight.room(since, backend.halt):
                continue
            raise
        except BaseException:
            stack.close()
            flight.land(held=False)
            raise
        break
    try:
        with stack:
            yield
    finally:
        flight.land(held=True)
