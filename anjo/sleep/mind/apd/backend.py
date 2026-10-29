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
priced at one call's worst case, so the hold stays a true bound; a 429 waits longer (its hint, or
``retry.rate_limit_wait``) and is priced at nothing, as it is billed. The Muse auditor, whose hold
has no extra tries, tries again on a 429 only. A wait or a cool-down (``pause``) ends early once
the command is ending. Every generation reply goes out
capped at ``retry.REPLY_MAX_TOKENS`` (``gen_chat``), the cap one call is held for. Side by side, an
attempt whose hold finds no room waits for the command's other attempts in flight to end
(``Flight``), and once the command is ending (``halt``) no attempt or call starts (``Halted``). A
core split's generator (``generators``) writes and extracts on Fireworks at its own list prices,
held on the provider's pool. Its reader (``readers``) reads the gates: Spark Open, on that pool
too, or Gemini on Vertex (``gemini_chat``, through the judge's client at low thinking, every reply
capped at ``retry.REPLY_MAX_TOKENS`` with its thoughts inside the cap and counted as output), held
on ``vertex_mind_usd`` at its own prices. The attempt's extra tries are held on each pool it uses.
Every try is logged (``calls.CallLog``) when the caller asks.
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
from typing import TYPE_CHECKING, Any, Final

import httpx

from anjo.sleep import fireworks_api, meta_api
from anjo.sleep.budget import Ledger, Pool
from anjo.sleep.context.brief import ChatFn as BriefChat
from anjo.sleep.mind.apd import colab_vllm, fakes, retry
from anjo.sleep.mind.apd.calls import CallLog
from anjo.sleep.mind.apd.ceiling import Ceiling
from anjo.sleep.mind.apd.colab_vllm import ColabSession, TunnelOpener
from anjo.sleep.mind.apd.generators import BY_MODEL, SPARK, Generator
from anjo.sleep.mind.apd.judging import MUSE, Rate, TextModel
from anjo.sleep.mind.apd.readers import SPARK_READER, Reader
from anjo.sleep.mind.apd.types import LIFE_DAYS, MAX_PREVALENCE

if TYPE_CHECKING:  # the limiter waits on this module's halt, so it imports this one
    from anjo.sleep.mind.apd.limiter import Limiter
from anjo.sleep.model_api import ChatResult, ModelApiError, RateLimited
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
GEMINI_READ_TIMEOUT_S: Final = 120.0  # a Gemini gate read's whole wait: a verdict, not a Spark Open decode
CALL_GRACE_S: Final = 5.0  # past its own timeout, a generation call is given up on (``bounded``)
CALL_POLL_S: Final = 0.1  # how often a caller waiting on a call in flight looks for a kill
LEAK_MARGIN_S: Final = 30.0  # past its transport timeout, a call given up on gives its limiter place back anyway
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
    read_by: Reader = SPARK_READER  # who reads a core split's and the pairs' gates (``readers``), through ``reader``
    reader_rate: Rate | None = None  # its list prices, when it isn't Spark Open
    read_limit: Limiter | None = field(default=None, compare=False)  # the run's one limit on the reader's calls
    repair: bool = False  # a core split's failed attempt has the days a leak, missed origin or audit names rewritten
    keep_failed: bool = False  # a core split's failed attempt is kept under ``quarantine`` (``store``)
    keep_off: bool = False  # a core split's writer is given its life's topics (``generate.KEEP_OFF``)


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


def gemini_chat(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                temperature: float | None = None) -> ChatResult:
    """A gate read on Gemini (Vertex, the judge's client): the system prompt as its instruction, at
    low thinking, the reply capped at ``retry.REPLY_MAX_TOKENS`` with the thoughts inside the cap
    and counted as output. A 429, a 5xx, a timeout or a dropped connection fails as ``model_api``'s
    do, so ``retry.transient`` tries it again."""
    types = importlib.import_module("google.genai.types")
    system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
    config = types.GenerateContentConfig(
        system_instruction=system or None, temperature=temperature, max_output_tokens=retry.REPLY_MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING),
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    contents = "\n\n".join(str(m["content"]) for m in messages if m["role"] != "system")
    try:
        reply = _vertex().models.generate_content(model=model, contents=contents, config=config)
    except Exception as error:  # noqa: BLE001 - the SDK's errors share no base; each is a failed call
        raise _vertex_error(error, timeout_s) from None
    usage = reply.usage_metadata
    out = int(usage.candidates_token_count or 0) + int(usage.thoughts_token_count or 0)  # both billed as output
    return ChatResult(reply.text or "", (), str(model), int(usage.prompt_token_count or 0), out, None)


def _vertex_error(error: Exception, timeout_s: float) -> ModelApiError:
    """``model_api``'s message for the failure, the SDK's text left out. A 429 (or
    ``RESOURCE_EXHAUSTED``) is ``RateLimited``, with its ``RetryInfo`` delay when it gave one."""
    where = "Gemini on Vertex"
    code = getattr(error, "code", None)
    if code == 429 or getattr(error, "status", None) == "RESOURCE_EXHAUSTED":
        return RateLimited(f"{where} returned HTTP 429", retry_after_s=_retry_delay(getattr(error, "details", None)))
    if isinstance(code, int):
        return ModelApiError(f"{where} returned HTTP {code}")
    if isinstance(error, httpx.TimeoutException):
        return ModelApiError(f"{where} timed out after {timeout_s:g}s")
    if isinstance(error, httpx.HTTPError):
        return ModelApiError(f"{where} failed in transport ({type(error).__name__})")
    return ModelApiError(f"{where} failed before a reply ({type(error).__name__})")


def _retry_delay(details: object) -> float | None:
    """A Vertex error's ``RetryInfo`` delay in seconds (``"17s"``), the only field read from it."""
    body = details.get("error", details) if isinstance(details, dict) else None
    found = body.get("details") if isinstance(body, dict) else None
    for item in found if isinstance(found, list) else []:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                seconds = float(delay[:-1])
            except ValueError:
                return None
            return seconds if seconds > 0 else None
    return None


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


def pause(seconds: float, sleep: Callable[[float], None], halt: threading.Event | None) -> None:
    """Wait ``seconds``, then ``Halted`` if the command is ending. Live (``time.sleep``) the wait is
    on the halt itself, so a kill ends it at once."""
    if halt is not None and sleep is time.sleep:
        halt.wait(seconds)
    else:
        sleep(seconds)
    halted(halt)


class GivenUp:
    """A call given up on (past its deadline, or at a kill) whose thread may still be in flight.
    If its limiter place has to be freed for it rather than by its end, it ``leak``s: each hook
    runs (``CountedChat`` marks the call ``leaked`` in its log)."""

    def __init__(self) -> None:
        self._lock, self._leaked = threading.Lock(), False
        self._hooks: list[Callable[[], None]] = []

    def on_leak(self, hook: Callable[[], None]) -> None:
        with self._lock:
            if not self._leaked:
                self._hooks.append(hook)
                return
        hook()

    def leak(self) -> None:
        with self._lock:
            self._leaked, hooks, self._hooks = True, self._hooks, []
        for hook in hooks:
            hook()


class Abandoned(Halted):
    """A call given up on in flight once the command is ending: it may still be billed, so it is
    priced at its worst case (``CountedChat``)."""

    def __init__(self, message: str, given_up: GivenUp | None = None) -> None:
        super().__init__(message)
        self.given_up = given_up


class CallTimedOut(ModelApiError):
    """A call past its deadline, given up on (``bounded``): a timeout, which ``retry.transient``
    tries again."""

    def __init__(self, message: str, given_up: GivenUp) -> None:
        super().__init__(message)
        self.given_up = given_up


class _Place:
    """One call's place at ``limiter`` (none without one), given back once only: by the call's own
    thread when the call ends, or, once the caller gave up on it, by a timer ``LEAK_MARGIN_S`` past
    its transport timeout, so a thread whose transport never times out can't keep it forever."""

    def __init__(self, limiter: Limiter | None, halt: threading.Event | None) -> None:
        if limiter is not None:
            limiter.acquire(halt)
        self._limiter, self._lock = limiter, threading.Lock()

    def release(self) -> bool:
        """Give the place back, if it is still held; whether this call did."""
        with self._lock:
            limiter, self._limiter = self._limiter, None
        if limiter is not None:
            limiter.release()
        return limiter is not None

    def give_up(self, wait_s: float) -> GivenUp:
        """The caller gave up on the call: the place is freed ``wait_s`` and ``LEAK_MARGIN_S`` from
        now if the call hasn't ended by then (it ``leak``s)."""
        given_up = GivenUp()

        def free() -> None:
            if self.release():
                given_up.leak()

        if self._limiter is not None:  # still held: no timer for a call with no place
            timer = threading.Timer(wait_s + LEAK_MARGIN_S, free)
            timer.daemon = True
            timer.start()
        return given_up


def bounded(chat: BriefChat, halt: threading.Event | None, cap_s: float | None = None,
            limiter: Limiter | None = None) -> BriefChat:
    """``chat`` with a deadline that fires and a kill that reaches a call in flight. Each call runs
    on a daemon thread of its own, its timeout capped at ``cap_s``; the caller waits for its reply
    no longer than that timeout and ``CALL_GRACE_S`` (then a timeout, as ``model_api``'s, which
    ``retry.transient`` tries again) and only until ``halt`` is set (``Abandoned``). A thread given
    up on ends with its own transport's timeout and never holds the process open. With
    ``limiter``, the caller takes a place before the call (its wait there no part of the deadline)
    and the call's own thread gives it back once the call really ends, so a call given up on but
    still in flight keeps counting against the limit, for at most its transport timeout and
    ``LEAK_MARGIN_S`` more (``_Place``)."""

    def call(model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, timeout_s: float, *,
             temperature: float | None = None) -> ChatResult:
        wait = timeout_s if cap_s is None else min(timeout_s, cap_s)
        found: list[ChatResult | BaseException] = []
        done, place = threading.Event(), _Place(limiter, halt)

        def run() -> None:
            try:
                found.append(chat(model, messages, tools, wait, temperature=temperature))
            except BaseException as error:  # noqa: BLE001 - handed to the caller, raised there
                found.append(error)
            finally:
                try:
                    place.release()
                finally:
                    done.set()

        try:
            threading.Thread(target=run, name="apd-call", daemon=True).start()
        except BaseException:
            place.release()
            raise
        deadline = time.monotonic() + wait + CALL_GRACE_S
        while not done.wait(max(0.0, min(CALL_POLL_S, deadline - time.monotonic()))):
            if halt is not None and halt.is_set():
                raise Abandoned("the command is ending: a call in flight was given up on", place.give_up(wait))
            if time.monotonic() >= deadline:
                raise CallTimedOut(f"a generation call timed out after {wait:g}s (its deadline)", place.give_up(wait))
        if isinstance(found[0], BaseException):
            raise found[0]
        return found[0]

    return call


def cool_down(backend: Backend) -> None:
    """The wait before the attempt after one a call ended (``retry.ATTEMPT_COOL_DOWN_S``)."""
    pause(retry.ATTEMPT_COOL_DOWN_S, backend.sleep, backend.halt)


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
class WaitBudget:
    """What one attempt may still wait on 429s, in seconds, shared by its calls (``Retries``)."""

    left_s: float = retry.RATE_LIMIT_BUDGET_S


@dataclass
class Retries:
    """The extra tries one attempt holds for transient failures (a 5xx, a timeout or a reset),
    shared by its writer's and reader's calls and held at their worst case; each waits first
    (``wait``). A 429 is billed nothing and held nothing, so its tries never spend ``left``: the
    call's ``retry.RATE_LIMIT_ATTEMPTS`` tries in all and the attempt's ``budget`` of 429 waits
    bound them. With ``rate_limits_only`` (the auditor's, whose hold has no extra tries) only a 429
    is tried again."""

    left: int = GEN_SPARK_RETRIES
    sleep: Callable[[float], None] = time.sleep
    rand: Callable[[], float] = random.random
    halt: threading.Event | None = None
    rate_limits_only: bool = False
    budget: WaitBudget = field(default_factory=WaitBudget)

    def wait(self, error: ModelApiError, attempt: int) -> float | None:
        """The wait before trying again a call whose try ``attempt`` failed with ``error``;
        ``None``: it isn't tried again. A 429 waits its hint or ``retry.rate_limit_wait``, taken from
        the attempt's budget, up to ``retry.RATE_LIMIT_ATTEMPTS`` tries of the call in all; another
        transient failure ``retry.backoff``, up to ``retry.ATTEMPTS``, one extra try spent."""
        if retry.rate_limited(error):
            wait = retry.rate_limit_wait(attempt, retry.hint(error), self.rand)
            if attempt >= retry.RATE_LIMIT_ATTEMPTS or wait > self.budget.left_s:
                return None
            self.budget.left_s -= wait
            return wait
        if self.rate_limits_only or self.left < 1 or attempt >= retry.ATTEMPTS or not retry.transient(error):
            return None
        self.left -= 1
        return retry.backoff(attempt, self.rand)

    def pause(self, seconds: float) -> None:
        pause(seconds, self.sleep, self.halt)


@dataclass
class CountedChat:
    """Generation's calls to one model family: no call past ``limit`` (the attempt's hold), each
    try priced at ``rate`` as it returns, or at one call's worst case when it fails or is given up
    on in flight (``Abandoned``: what it cost is unknown). A transient failure is tried again while
    ``retries`` allows. No try starts once ``halt`` is set (``Halted``)."""

    inner: BriefChat
    limit: int
    rate: Rate | None
    retries: Retries | None = None
    calls: int = 0
    usd: float = 0.0
    halt: threading.Event | None = None
    log: CallLog | None = None  # each try, when the caller keeps them

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
            except Abandoned as error:
                self._priced(messages, 0, 0, 0.0 if self.rate is None else self.rate.worst, failed=True)
                self._watch(error.given_up)
                raise
            except ModelApiError as error:
                limited = retry.rate_limited(error)  # a 429 is billed nothing; another failure at worst
                wait = None if self.retries is None else self.retries.wait(error, attempt)
                self._priced(messages, 0, 0, 0.0 if limited or self.rate is None else self.rate.worst, failed=True,
                             rate_limited=limited, wait_s=wait)
                self._watch(error.given_up if isinstance(error, CallTimedOut) else None)
                if self.retries is None or wait is None:
                    raise
                self.retries.pause(wait)
                attempt += 1
                continue
            self._priced(messages, result.input_tokens, result.output_tokens,
                         0.0 if self.rate is None else self.rate.usd(result.input_tokens, result.output_tokens),
                         reply=result)
            return result

    def _watch(self, given_up: GivenUp | None) -> None:
        """The try just logged, given up on: marked ``leaked`` in the log if its place is freed for it."""
        if given_up is not None and self.log is not None:
            given_up.on_leak(functools.partial(self.log.mark_leaked, len(self.log.calls) - 1))

    def _priced(self, messages: list[dict[str, Any]], tokens_in: int, tokens_out: int, usd: float, *,
                failed: bool = False, reply: ChatResult | None = None, rate_limited: bool = False,
                wait_s: float | None = None) -> None:
        self.usd += usd
        if self.log is not None:
            self.log.add(messages, tokens_in, tokens_out, usd, failed=failed,
                         reply_chars=None if reply is None else len(reply.text),
                         finish=None if reply is None else reply.finish_reason, rate_limited=rate_limited, wait_s=wait_s)


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


def life_calls(prevalence: int, *, day_retries: int = 0) -> GenCalls:
    """One life's attempt with ``prevalence`` planted items: a writer and an extractor call a day,
    and ``day_retries`` more writer calls a day (``generate.day_retries``), each item's reads and
    identify call, and the script audit."""
    if not 0 <= prevalence <= MAX_PREVALENCE:
        raise ValueError(f"prevalence must be 0 to {MAX_PREVALENCE}")
    return GenCalls(GEN_WRITE_CALLS + day_retries * LIFE_DAYS, prevalence * GEN_READS_PER_ITEM, prevalence + 1)


WORST_LIFE: Final = life_calls(MAX_PREVALENCE)


def gen_holds(prices: Prices | None, rates: Mapping[str, Rate] | None, spark_pool: Pool = BASETEN.pool,
              calls: GenCalls = WORST_LIFE, write_rate: Rate | None = None, *,
              read_rate: Rate | None = None, read_pool: Pool | None = None) -> dict[Pool, float]:
    """One generation attempt's worst case on each pool: on ``spark_pool``, its writes at
    ``write_rate`` (a generator's, or Spark Open's when ``None``) and its reads at Spark Open's,
    each extra try at the dearer of the two; with a reader on a pool of its own (``read_pool``),
    its reads and as many extra tries at ``read_rate`` there, and the writes' extra tries apart."""
    spark, muse = spark_rate(prices), None if rates is None else rates[MUSE]
    writes = write_rate or spark
    holds: dict[Pool, float] = {"meta_usd": muse.worst * calls.muse if muse else 0.0}
    if read_pool is None or read_pool == spark_pool:
        reads = read_rate or spark
        holds[spark_pool] = 0.0 if reads is None or writes is None else (
            calls.writes * writes.worst + calls.reads * reads.worst + calls.retries * max(writes.worst, reads.worst))
        return holds
    holds[spark_pool] = 0.0 if writes is None else (calls.writes + calls.retries) * writes.worst
    holds[read_pool] = 0.0 if read_rate is None else (calls.reads + calls.retries) * read_rate.worst
    return holds


def write_rate(backend: Backend) -> Rate | None:
    """What the writer and extractor cost: the generator's prices, or Spark Open's."""
    if backend.generator.model is None:
        return spark_rate(backend.prices)
    if backend.prices is not None and backend.generator_rate is None:
        raise ValueError(f"--generator {backend.generator.name} needs its prices to be held")
    return backend.generator_rate


def read_rate(backend: Backend) -> Rate | None:
    """What a gate read costs: the reader's prices, or Spark Open's."""
    if backend.read_by.model is None:
        return spark_rate(backend.prices)
    if backend.prices is not None and backend.reader_rate is None:
        raise ValueError(f"--reader {backend.read_by.name} needs its prices to be held")
    return backend.reader_rate


def attempt_holds(backend: Backend, calls: GenCalls) -> dict[Pool, float]:
    """``gen_holds`` for one of ``backend``'s attempts: its generator's writes, its reader's reads."""
    pool = backend.read_by.pool_on(backend.provider)
    return gen_holds(backend.prices, backend.rates, backend.provider.pool, calls, write_rate(backend),
                     read_rate=read_rate(backend), read_pool=pool)


@contextmanager
def held_attempt(backend: Backend, key: str, calls: GenCalls = WORST_LIFE,
                 log: CallLog | None = None) -> Iterator[tuple[CountedChat, CountedChat, CountedChat]]:
    """One attempt's writer, reader and auditor, at most ``calls``, its worst case held first on
    every pool it uses (``attempt_holds``: the provider's, with the extra tries its calls share, the
    reader's own when it has one, and Muse Spark's), each try logged to ``log`` when given.
    Nothing starts once ``backend.halt`` is set, a read queued at the run's limiter
    (``read_limit``) included, and a call in flight is given up on (``bounded``); a Gemini read
    waits ``GEMINI_READ_TIMEOUT_S`` at most; a hold without room waits (``_room``)."""
    halted(backend.halt)
    halt, cap = backend.halt, None if backend.read_by.model is None else GEMINI_READ_TIMEOUT_S
    budget = WaitBudget(retry.RATE_LIMIT_BUDGET_S)  # the attempt's 429 waits, its auditor's included
    retries = Retries(calls.retries, sleep=backend.sleep, halt=halt, budget=budget)
    spark = CountedChat(bounded(backend.life_chat, halt), calls.writes, write_rate(backend), retries, halt=halt,
                        log=log)
    reads = bounded(backend.reader, halt, cap, backend.read_limit)
    reader = CountedChat(reads, calls.reads, read_rate(backend), retries, halt=halt, log=log)
    auditor = CountedChat(bounded(backend.auditor, halt), calls.muse,
                          None if backend.rates is None else backend.rates[MUSE],
                          Retries(calls.retries, sleep=backend.sleep, halt=halt, rate_limits_only=True, budget=budget),
                          halt=backend.halt, log=log)
    if backend.ledger is None:
        yield spark, reader, auditor
        return
    pool, ledger, worst = backend.provider.pool, backend.ledger, attempt_holds(backend, calls)
    read_pool = backend.read_by.pool_on(backend.provider)

    def holds(stack: ExitStack) -> None:
        on_pool = (lambda: spark.usd + reader.usd) if read_pool == pool else (lambda: spark.usd)
        stack.enter_context(held(ledger, f"apd:gen:{key}:spark", (pool,), usd(worst[pool]), on_pool))
        if read_pool != pool:
            stack.enter_context(held(ledger, f"apd:gen:{key}:reader", (read_pool,), usd(worst[read_pool]),
                                     lambda: reader.usd))
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
