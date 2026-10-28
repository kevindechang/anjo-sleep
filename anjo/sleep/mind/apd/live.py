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

``run_live`` holds a run's Spark Open worst case before its first call (``held_run``) and settles
what it priced. The GPU is held once, by the serving session itself (``colab_vllm.serve``).
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final

from anjo.sleep.budget import Ledger
from anjo.sleep.mind.apd.arms import MindCore, run_arm
from anjo.sleep.mind.apd.colab_vllm import ColabSession
from anjo.sleep.mind.apd.types import ArmModels, ArmName, ArmRun, Life, NightClock, Reported, WrittenPort
from anjo.sleep.mind.inquiry.types import Budget, ModelReply
from anjo.sleep.mind.replay.served import arm_thinker
from anjo.sleep.mind.replay.spend import SPARK_POOL, colab_client
from anjo.sleep.mind.screen import SparkScreen
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import held, spark_usd, spark_worst_usd, usd
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
    """Every Spark Open call of one run: no call past ``limit``, the count its hold covers (the
    port then refuses or says undetermined, as for a failed call), and each priced as it returns
    (``prices`` ``None``: unpriced, with nothing held)."""

    chat: ChatFn
    prices: Prices | None
    limit: int
    calls: int = 0
    usd: float = 0.0

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        if self.calls >= self.limit:
            raise ModelApiError("past the Spark Open calls this run holds")
        self.calls += 1
        result = self.chat(messages, max_tokens)
        if self.prices is not None:
            self.usd += spark_usd(self.prices, result.input_tokens, result.output_tokens)
        return result

    def reporting(self, reported: Reported, port: WrittenPort) -> ChatFn:
        """These calls, each reply's model going to ``reported`` as ``port``'s."""

        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            result = self(messages, max_tokens)
            if result.model:
                reported.saw(port, result.model)
            return result

        return chat


def spark_calls(chat: ChatFn, prices: Prices | None, life: Life) -> SparkCalls:
    """A run's Spark Open calls: at most ``MIND_SPARK_CALLS`` a night, as a night's MIND step."""
    return SparkCalls(chat, prices, len(life.days) * MIND_SPARK_CALLS)


def arm_models(session: ColabSession, spark: SparkCalls, *, adapter: CheckpointRef | None = None) -> ArmModels:
    """One run's models: Glimmer on ``session`` (``adapter`` for M+replay and M+shuffled) as the
    thinker and the world query, Spark Open through ``spark`` as the screen and the resolver."""
    clock, reported = NightClock(), Reported()
    client = colab_client(session)
    return ArmModels(
        thinker=arm_thinker(session, adapter, clock), world=GlimmerWorld(GlimmerChat(client, clock), reported),
        screen=SparkScreen(names=(), chat=spark), resolver=SparkResolver(chat=spark.reporting(reported, "resolver")),
        clock=clock, reported=reported,
    )


@contextmanager
def held_run(ledger: Ledger | None, key: str, spark: SparkCalls) -> Iterator[None]:
    """A run's Spark Open worst case held on ``baseten_mind_usd`` before its first call, settled
    at what it priced. Without a ledger (tests and ``--fake``) nothing is held."""
    if ledger is None:
        yield
        return
    if spark.prices is None:
        raise ValueError("a held run is priced")
    with held(ledger, f"apd:{key}", (SPARK_POOL,), usd(spark_worst_usd(spark.prices, spark.limit)), lambda: spark.usd):
        yield


def run_live(
    arm: ArmName, life: Life, session: ColabSession, spark_chat: ChatFn, *, ledger: Ledger | None,
    prices: Prices | None, adapter: CheckpointRef | None = None, budget: Budget = Budget(),
    core: MindCore | None = None, brief: str | None = None,
) -> ArmRun:
    """One arm through ``life`` on the live models, its Spark Open calls held first."""
    spark = spark_calls(spark_chat, prices, life)
    models = arm_models(session, spark, adapter=adapter)
    with held_run(ledger, f"run:{life.id}:{arm}", spark):
        return run_arm(arm, life, models, budget=budget, core=core, brief=brief)
