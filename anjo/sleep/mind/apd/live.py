"""The APD arms' live models (Tasks 17 and 17b): ``ArmModels`` on Glimmer on Colab vLLM and Spark Open.

- **The thinker:** base Glimmer, or a 17b adapter, on the serving session's own tunnel
  (``replay.served.arm_thinker``, through ``spend.colab_client``), at temperature 0, each prompt
  dated on the arms' ``NightClock``.
- **``world_query``:** the same base Glimmer, under ``world_query_v1.md``, at temperature 0 with no
  adapter, dated the same, as the night wires it (``app/mind_night.py``): the world query asks the
  thinker's own model.
- **The screen and the blind resolver:** Spark Open (``mind.screen.SparkScreen``, with no names,
  since a synthetic life has none, and ``SparkResolver``). Every Spark Open call of a run, the
  screen's included, goes through one ``SparkCalls``: counted against the run's hold on
  ``baseten_mind_usd`` (no call past it) and priced as it returns.
- **Provenance:** the model each world answer's and resolver verdict's reply named, as its API
  reported it, goes to ``ArmModels.reported``, which the arms pair with the step that logs it.

``SparkCalls`` holds each night's Spark Open worst case before that night's first call and settles
what it priced, as the night's MIND step does. The GPU is held once, by the serving session itself
(``colab_vllm.serve``).
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

from anjo.sleep.budget import Ledger, Reservation
from anjo.sleep.mind.apd.arms import MindCore, run_arm
from anjo.sleep.mind.apd.colab_vllm import ColabSession
from anjo.sleep.mind.apd.types import ArmModels, ArmName, ArmRun, Life, NightClock, Reported, WrittenPort
from anjo.sleep.mind.inquiry.types import Budget, ModelReply
from anjo.sleep.mind.replay.served import arm_thinker
from anjo.sleep.mind.replay.spend import SPARK_POOL, colab_client
from anjo.sleep.mind.screen import SparkScreen
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import hold, spark_usd, spark_worst_usd, usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.spark_inquiry import WORLD_MAX_TOKENS, WORLD_PROMPT, ChatFn, SparkResolver
from anjo.sleep.types import CheckpointRef

logger = logging.getLogger(__name__)
TEMPERATURE: Final = 0.0
SEED: Final = 0


@dataclass(frozen=True)
class GlimmerChat:
    """Base Glimmer as a capped ``ChatFn`` on the session's client, dated on ``clock``. A failed
    request is a ``ModelApiError``, so the port charges its cap."""

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
    """``world_query`` on base Glimmer, under ``world_query_v1.md``. A failed call is empty text
    charged its cap; a reply's model, as reported, goes to ``reported``."""

    chat: ChatFn
    reported: Reported

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        cap = min(max_tokens, WORLD_MAX_TOKENS)
        if cap <= 0:
            return ModelReply(text="", tokens=0, model=GLIMMER)
        try:
            result = self.chat([{"role": "system", "content": WORLD_PROMPT}, {"role": "user", "content": text}], cap)
        except ModelApiError as error:
            logger.warning("APD's world query failed, charged its cap of %d tokens: %s", cap, error)
            return ModelReply(text="", tokens=cap, model=GLIMMER)
        if result.model:
            self.reported.saw("world", result.model)
        return ModelReply(text=result.text, tokens=result.output_tokens, model=result.model or GLIMMER)


@dataclass
class SparkCalls:
    """Every Spark Open call of one run, held a night at a time as the night's MIND step holds them:
    at a night's first call (``clock``), its worst case of ``per_night`` calls is held on
    ``baseten_mind_usd`` (the night before settled at what it priced), and a call past that count
    is refused (the port then refuses or says undetermined, as for a failed call). Without a
    ledger nothing is held; without prices nothing is priced."""

    chat: ChatFn
    prices: Prices | None
    ledger: Ledger | None = None
    key: str = "spark"
    per_night: int = MIND_SPARK_CALLS
    clock: NightClock | None = None
    night: date | None = None
    calls: int = 0  # this night's
    usd: float = 0.0  # this night's
    _held: Reservation | None = field(default=None, repr=False)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self._turn()
        if self.calls >= self.per_night:
            raise ModelApiError("past the Spark Open calls this night holds")
        self.calls += 1
        result = self.chat(messages, max_tokens)
        if self.prices is not None:
            self.usd += spark_usd(self.prices, result.input_tokens, result.output_tokens)
        return result

    def _turn(self) -> None:
        day = None if self.clock is None else self.clock.day
        if day == self.night and (self.ledger is None or self._held is not None):
            return
        self.close()
        self.night, self.calls, self.usd = day, 0, 0.0
        if self.ledger is not None:
            if self.prices is None:
                raise ValueError("a held run is priced")
            self._held = hold(self.ledger, f"apd:{self.key}:{day}", (SPARK_POOL,),
                              usd(spark_worst_usd(self.prices, self.per_night)))

    def close(self) -> None:
        """Settle the open night's hold at what it priced, or release it."""
        held_now, self._held = self._held, None
        if held_now is None:
            return
        if self.usd > 0:
            held_now.settle(self.usd)
        else:
            held_now.release()

    def reporting(self, reported: Reported, port: WrittenPort) -> ChatFn:
        """These calls, each reply's model going to ``reported`` as ``port``'s."""

        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            result = self(messages, max_tokens)
            if result.model:
                reported.saw(port, result.model)
            return result

        return chat


def arm_models(session: ColabSession, spark: SparkCalls, *, adapter: CheckpointRef | None = None) -> ArmModels:
    """One run's models: Glimmer on ``session`` (``adapter`` for M+replay and M+shuffled) as the
    thinker and the world query, Spark Open through ``spark`` as the screen and the resolver, all
    on one night clock."""
    clock, reported = NightClock(), Reported()
    spark.clock = clock
    client = colab_client(session)
    return ArmModels(
        thinker=arm_thinker(session, adapter, clock), world=GlimmerWorld(GlimmerChat(client, clock), reported),
        screen=SparkScreen(names=(), chat=spark), resolver=SparkResolver(chat=spark.reporting(reported, "resolver")),
        clock=clock, reported=reported,
    )


@contextmanager
def held_run(spark: SparkCalls) -> Iterator[SparkCalls]:
    """A run's Spark Open calls, the last night's hold settled on the way out."""
    try:
        yield spark
    finally:
        spark.close()


def run_live(
    arm: ArmName, life: Life, session: ColabSession, spark_chat: ChatFn, *, ledger: Ledger | None,
    prices: Prices | None, adapter: CheckpointRef | None = None, budget: Budget = Budget(),
    core: MindCore | None = None, brief: str | None = None,
) -> ArmRun:
    """One arm through ``life`` on the live models, its Spark Open calls held first."""
    spark = SparkCalls(spark_chat, prices, ledger, f"run:{life.id}:{arm}")
    models = arm_models(session, spark, adapter=adapter)
    with held_run(spark):
        return run_arm(arm, life, models, budget=budget, core=core, brief=brief)
