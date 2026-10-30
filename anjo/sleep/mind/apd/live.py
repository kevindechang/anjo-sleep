"""The APD arms' live models (Tasks 17 and 17b): ``ArmModels`` on Glimmer on Colab vLLM and Spark Open.

- **The thinker:** base Glimmer, or a 17b adapter, on the serving session's own tunnel
  (``replay.served.arm_thinker``, through ``spend.colab_client``), at temperature 0, each prompt
  dated on the arms' ``NightClock``.
- **``world_query``:** the same base Glimmer, under ``world_query_v1.md``, at temperature 0 with no
  adapter, dated the same, as the night wires it (``app/mind_night.py``): the world query asks the
  thinker's own model.
- **The blind resolver:** Spark Open (``SparkResolver``) on the experiment's provider
  (``mind.spark_provider``). Every Spark Open call of a run goes through one ``SparkCalls``:
  counted against the run's hold on the provider's pool (``baseten_mind_usd`` or
  ``fireworks_mind_usd``; no call past it) and priced as it returns.
- **Provenance:** the model each world answer's and resolver verdict's reply named, as its API
  reported it, goes to ``ArmModels.reported``, which the arms pair with the step that logs it.
- **No run with a failed call:** a transient failure (a 429, a 5xx, a timeout or a reset) is
  tried again at most ``retry.ATTEMPTS`` times in all (``retry``: under the session's client for
  Glimmer, here for Spark Open, each try a call of its own, held and journaled). A Glimmer call
  (the thinker's or the world query's) or a Spark Open call that still fails stops its run there
  (``RunFailed``), and so does any Spark Open call once the session has stopped serving, so a run
  with a failed call is never saved and never judged. Only the refusal past a night's held calls
  stays inside the run, as the night's MIND step has it.

``SparkCalls`` holds each night's Spark Open worst case before that night's first call and settles
what it priced, as the night's MIND step does, and journals every call before it returns
(``journal``), so a night its process never closed is settled from the journal. A night refused
its hold waits for room (``room_wait``): for one of the other runs' nights to settle (runs go in
parallel), up to ``ROOM_WAIT_S``; on Baseten, for Sleep's open holds on the shared credit, which
has Sleep's priority, up to ``SHARED_WAIT_S``, then stops cleanly (``BudgetStop``); and when neither
holds what it needs (MIND's own cap binds, or nothing open would free enough), no room is coming
and it stops at once. Fireworks' pool shares nothing with Sleep, so there only other MIND nights
are waited on. The GPU is held once, by the serving session itself (``colab_vllm.serve``).
``Meter`` sums what a batch of runs used, for timing.
"""
from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

from anjo.sleep.budget import Ledger, Pool, Reservation
from anjo.sleep.mind.apd import retry
from anjo.sleep.mind.apd.arms import MindCore, run_arm
from anjo.sleep.mind.apd.colab_vllm import ColabSession
from anjo.sleep.mind.apd.journal import SpendJournal
from anjo.sleep.mind.apd.types import ArmModels, ArmName, ArmRun, Life, NightClock, Reported, WrittenPort
from anjo.sleep.mind.inquiry.types import Budget, ModelReply
from anjo.sleep.mind.replay.served import ServedThinker, arm_thinker
from anjo.sleep.mind.replay.spend import SPARK_POOL, colab_client
from anjo.sleep.mind.replay.types import CallEnd
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import hold, spark_usd, spark_worst_usd, usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.spark_inquiry import WORLD_MAX_TOKENS, WORLD_PROMPT, ChatFn, SparkResolver
from anjo.sleep.stops import BudgetStop
from anjo.sleep.types import CheckpointRef

TEMPERATURE: Final = 0.0
SEED: Final = 0
ROOM_POLL_S: Final = 5.0  # how often a night waiting for room looks again
ROOM_WAIT_S: Final = 3600.0  # and how long it waits on other MIND nights before it stops
SHARED_WAIT_S: Final = 600.0  # or on Sleep's open holds on the shared Baseten credit
SLEEP_POOL: Final = "baseten_sleep_usd"
_EPSILON: Final = 1e-9


class RunFailed(RuntimeError):
    """A model call of a run failed, or the session stopped under it: the run stops and is never
    saved. It is no ``ModelApiError``, so no port absorbs it."""


class RunsFailed(RunFailed):
    """A batch ended with runs that failed alone: each recorded (``DataDir.failed``), none saved,
    every other run kept; a rerun runs only those again."""


@dataclass(frozen=True)
class StrictThinker:
    """The arm's thinker (``ServedThinker``), stopping the run at its first failed call."""

    inner: ServedThinker

    @property
    def calls(self) -> list[CallEnd]:
        return self.inner.calls

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        reply = self.inner.propose(rendered, max_tokens)
        if self.inner.calls and self.inner.calls[-1].failed:
            raise RunFailed("a Glimmer thinker call failed")
        return reply


@dataclass(frozen=True)
class GlimmerChat:
    """Base Glimmer as a capped ``ChatFn`` on the session's client, dated on ``clock``. A failed
    request is a ``ModelApiError``, which ``GlimmerWorld`` turns into ``RunFailed``."""

    client: VllmClient
    clock: NightClock

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        if self.clock.day is None:
            raise ValueError("a Glimmer prompt is dated on its night, and none is set")
        try:
            return self.client.on(self.clock.day).complete(None, messages, temperature=TEMPERATURE, seed=SEED,
                                                           max_tokens=max_tokens)
        except ServingError as error:
            raise ModelApiError(f"base Glimmer: {error}") from None


@dataclass(frozen=True)
class GlimmerWorld:
    """``world_query`` on base Glimmer, under ``world_query_v1.md``. A failed call stops the run
    (``RunFailed``); a reply's model, as reported, goes to ``reported``."""

    chat: ChatFn
    reported: Reported

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        cap = min(max_tokens, WORLD_MAX_TOKENS)
        if cap <= 0:
            return ModelReply(text="", tokens=0, model=GLIMMER)
        try:
            result = self.chat([{"role": "system", "content": WORLD_PROMPT}, {"role": "user", "content": text}], cap)
        except ModelApiError as error:
            raise RunFailed(f"a Glimmer world query failed: {error}") from None
        if result.model:
            self.reported.saw("world", result.model)
        return ModelReply(text=result.text, tokens=result.output_tokens, model=result.model or GLIMMER)


@dataclass
class SparkCalls:
    """Every Spark Open call of one run, held a night at a time as the night's MIND step holds them:
    at a night's first call (``clock``), its worst case of ``per_night`` calls is held on ``pool``,
    the provider's (the night before settled at what it priced), and a call past that count
    is refused (the resolver then says undetermined). Each call is journaled with its real
    tokens before it returns. A transient failure is tried again (``retry``), ``wait``-ing its
    backoff, each try a call of its own under the night's count; a failed try is priced at one
    call's worst case, since what it cost is unknown, and journaled as failed. A call that still
    fails, or any call once ``serving`` has stopped, stops the run (``RunFailed``). Without a
    ledger nothing is held; without prices nothing is priced. ``total_calls`` and ``total_usd``
    count every night's."""

    chat: ChatFn
    prices: Prices | None
    ledger: Ledger | None = None
    key: str = "spark"
    pool: Pool = SPARK_POOL
    per_night: int = MIND_SPARK_CALLS
    clock: NightClock | None = None
    journal: SpendJournal | None = None
    wait: Callable[[float], None] = time.sleep
    rand: Callable[[], float] = random.random  # the backoff's jitter
    serving: ColabSession | None = None  # the run's session: no call once it stops
    night: date | None = None
    calls: int = 0  # this night's
    usd: float = 0.0  # this night's
    total_calls: int = 0
    total_usd: float = 0.0
    _held: Reservation | None = field(default=None, repr=False)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self._check_serving()
        self._turn()
        if self.calls >= self.per_night:
            raise ModelApiError("past the Spark Open calls this night holds")
        attempt = 1
        while True:
            try:
                return self._try(messages, max_tokens)
            except ModelApiError as error:
                if attempt >= retry.ATTEMPTS or not retry.transient(error):
                    raise RunFailed(f"a Spark Open call failed: {error}") from None
                if self.calls >= self.per_night:
                    raise RunFailed(f"a Spark Open call failed and the night holds no call to retry it: {error}") from None
            self.wait(retry.backoff(attempt, self.rand))
            attempt += 1

    def _try(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        """One try: a call of its own under the night's count, journaled before it returns."""
        self._check_serving()
        self.calls += 1
        self.total_calls += 1
        try:
            result = self.chat(messages, max_tokens)
        except ModelApiError:
            self._spent(0, 0, 0.0 if self.prices is None else spark_worst_usd(self.prices, 1), failed=True)
            raise
        self._spent(result.input_tokens, result.output_tokens,
                    0.0 if self.prices is None else spark_usd(self.prices, result.input_tokens, result.output_tokens))
        return result

    def _check_serving(self) -> None:
        if self.serving is not None and not self.serving.live:
            raise RunFailed(f"session {self.serving.session} stopped serving; no Spark Open call after it")

    def _spent(self, tokens_in: int, tokens_out: int, price: float, *, failed: bool = False) -> None:
        self.usd += price
        self.total_usd += price
        if self.journal is not None and self._held is not None:
            self.journal.call(self._held.job, tokens_in, tokens_out, price, failed=failed)

    def _turn(self) -> None:
        day = None if self.clock is None else self.clock.day
        if day == self.night and (self.ledger is None or self._held is not None):
            return
        self.close()
        self.night, self.calls, self.usd = day, 0, 0.0
        if self.ledger is not None:
            if self.prices is None:
                raise ValueError("a held run is priced")
            self._held = self._hold(self.ledger, self.prices, day)
            if self.journal is not None:
                self.journal.held(self._held.job)

    def _hold(self, ledger: Ledger, prices: Prices, day: date | None) -> Reservation:
        """The night's hold; while other holds take the room it needs, wait for them (``room_wait``)."""
        need, waited = usd(spark_worst_usd(prices, self.per_night)), 0.0
        while True:
            try:
                return hold(ledger, f"apd:{self.key}:{day}", (self.pool,), need)
            except BudgetStop:
                limit = room_wait(ledger, need.amount, self.pool)
                if limit is None or waited >= limit:
                    if limit == SHARED_WAIT_S:
                        raise BudgetStop(
                            f"Sleep's open Baseten holds left MIND's night no room for {SHARED_WAIT_S / 60:.0f} "
                            "minutes; nothing past it was held. Rerun once Sleep's night is done") from None
                    raise
            self.wait(ROOM_POLL_S)
            waited += ROOM_POLL_S

    def close(self) -> None:
        """Settle the open night's hold at what it priced, or release it."""
        held_now, self._held = self._held, None
        if held_now is None:
            return
        if self.usd > 0:
            held_now.settle(self.usd)
        else:
            held_now.release()
        if self.journal is not None:
            self.journal.closed(held_now.job)

    def reporting(self, reported: Reported, port: WrittenPort) -> ChatFn:
        """These calls, each reply's model going to ``reported`` as ``port``'s."""

        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            result = self(messages, max_tokens)
            if result.model:
                reported.saw(port, result.model)
            return result

        return chat


def room_wait(ledger: Ledger, need: float, pool: Pool = SPARK_POOL) -> float | None:
    """How long a night refused ``need`` on MIND's ``pool`` may wait for room: ``ROOM_WAIT_S``
    while other MIND nights hold the pool (one settles soon); on Baseten's pool, ``SHARED_WAIT_S``
    when only Sleep's open holds stand in the way (the shared credit binds, not MIND's own cap, and
    what Sleep holds would free enough); else ``None``, since no room is coming."""
    holds = ledger.open_holds()
    if any(pool in held.pools for held in holds):
        return ROOM_WAIT_S
    if pool != SPARK_POOL:  # only Baseten's credit is shared with Sleep
        return None
    sleep_held = sum(held.bound.amount for held in holds if SLEEP_POOL in held.pools)
    mind_room, shared_room = ledger.remaining(SPARK_POOL), ledger.remaining(SLEEP_POOL)
    if sleep_held <= 0 or mind_room + _EPSILON < shared_room or need > shared_room + sleep_held + _EPSILON:
        return None
    return SHARED_WAIT_S


def arm_models(session: ColabSession, spark: SparkCalls, *, adapter: CheckpointRef | None = None) -> ArmModels:
    """One run's models: Glimmer on ``session`` (``adapter`` for M+replay and M+shuffled) as the
    thinker and the world query, Spark Open through ``spark`` as the resolver, all on one night
    clock."""
    clock, reported = NightClock(), Reported()
    spark.clock, spark.serving = clock, session
    client = colab_client(session)
    return ArmModels(
        thinker=StrictThinker(arm_thinker(session, adapter, clock)),
        world=GlimmerWorld(GlimmerChat(client, clock), reported),
        resolver=SparkResolver(chat=spark.reporting(reported, "resolver")),
        clock=clock, reported=reported,
    )


@contextmanager
def held_run(spark: SparkCalls) -> Iterator[SparkCalls]:
    """A run's Spark Open calls, the last night's hold settled on the way out."""
    try:
        yield spark
    finally:
        spark.close()


@dataclass
class Meter:
    """What a batch of runs used, summed across their threads: Glimmer thinker calls, Spark Open
    calls and their price, and life-arm nights."""

    thinker_calls: int = 0
    spark_calls: int = 0
    spark_usd: float = 0.0
    nights: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, models: ArmModels, spark: SparkCalls, nights: int) -> None:
        thinker = len(models.thinker.calls) if isinstance(models.thinker, StrictThinker) else 0
        with self._lock:
            self.thinker_calls += thinker
            self.spark_calls += spark.total_calls
            self.spark_usd += spark.total_usd
            self.nights += nights


def run_live(
    arm: ArmName, life: Life, session: ColabSession, spark_chat: ChatFn, *, ledger: Ledger | None,
    prices: Prices | None, adapter: CheckpointRef | None = None, budget: Budget = Budget(),
    core: MindCore | None = None, brief: str | None = None, key: str | None = None,
    journal: SpendJournal | None = None, meter: Meter | None = None, wait: Callable[[float], None] = time.sleep,
    pool: Pool = SPARK_POOL,
) -> ArmRun:
    """One arm through ``life`` on the live models, its Spark Open calls held a night at a time on
    ``pool`` under ``key`` and journaled; ``wait`` is how a call waits for room or a retry."""
    spark = SparkCalls(spark_chat, prices, ledger, key or f"run:{life.id}:{arm}", pool=pool, journal=journal,
                       wait=wait)
    models = arm_models(session, spark, adapter=adapter)
    with held_run(spark):
        found = run_arm(arm, life, models, budget=budget, core=core, brief=brief)
    if meter is not None:
        meter.add(models, spark, len(life.days))
    return found
