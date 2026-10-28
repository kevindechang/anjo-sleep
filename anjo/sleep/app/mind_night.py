"""MIND's inquiry wired for the sleep host (Task 17); ``app/night.py`` builds it only with ``--mind``.

- **The thinker and ``world_query``:** base Glimmer on the night's vLLM server, in its own requests
  (``adapter=None``, never a personal adapter), each capped at the smaller of its ceiling and the
  loop's allowance. The server is woken before the first call (the GPU job wakes it later anyway); a
  nap makes no call, so it wakes nothing (``night/sleep_night.py``).
  A failed call is empty text charged its cap, as on Spark Open (``spark_inquiry.call``).
- **The screen and the blind resolver:** Spark Open (``app/screen.py``, ``SparkResolver``), every
  call through the step's wrapper, which counts it for the ``baseten_mind_usd`` hold and prices it.
- **``research``:** Muse Wonder (``WonderResearch``), on the night's Wonder meter, held on ``meta_usd``.
- **Her workspace:** ``storage.workspace_log``, exported and erased with her other sleep files.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from anjo.sleep.app.names import person_names
from anjo.sleep.app.screen import SparkScreen
from anjo.sleep.app.storage import workspace_log
from anjo.sleep.budget import Ledger
from anjo.sleep.homework.agent import MetaWonder
from anjo.sleep.homework.research import WonderResearch
from anjo.sleep.homework.tools import SafeFetcher, WikipediaSearch
from anjo.sleep.mind.inquiry.types import ModelReply
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import MindCall, Prices
from anjo.sleep.night.mind import MindModels, MindNight, SparkWrap
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving.client import ServingError, VllmClient, WakeResult
from anjo.sleep.spark_inquiry import THINK_MAX_TOKENS, WORLD_MAX_TOKENS, WORLD_PROMPT, ChatFn, SparkResolver, spark_open

logger = logging.getLogger(__name__)
TEMPERATURE: Final = 0.0


def glimmer_chat(client: VllmClient, *, seed: int, wake: Callable[[], WakeResult] | None) -> ChatFn:
    """Base Glimmer as a capped ``ChatFn``; a server that won't wake, or a failed request, is a
    ``ModelApiError``."""
    awake: list[bool] = []

    def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        if wake is not None and not awake:
            if not wake().ok:
                raise ModelApiError("base Glimmer didn't wake")
            awake.append(True)
        try:
            return client.complete(None, messages, temperature=TEMPERATURE, seed=seed, max_tokens=max_tokens)
        except ServingError as error:
            raise ModelApiError(f"base Glimmer: {error}") from None

    return chat


def _call(chat: ChatFn, messages: list[dict[str, Any]], cap: int) -> ModelReply:
    if cap <= 0:
        return ModelReply(text="", tokens=0, model=GLIMMER)
    try:
        result = chat(messages, cap)
    except ModelApiError as error:
        logger.warning("MIND's Glimmer call failed, charged its cap of %d tokens: %s", cap, error)
        return ModelReply(text="", tokens=cap, model=GLIMMER)
    return ModelReply(text=result.text, tokens=result.output_tokens, model=GLIMMER)


@dataclass(frozen=True)
class GlimmerThinker:
    """The ``Thinker`` on base Glimmer: the rendered workspace is the whole prompt."""

    chat: ChatFn

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        return _call(self.chat, [{"role": "user", "content": rendered}], min(max_tokens, THINK_MAX_TOKENS))


@dataclass(frozen=True)
class GlimmerWorldQuery:
    """``world_query`` on the thinker's own model, under ``world_query_v1.md``."""

    chat: ChatFn

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        messages = [{"role": "system", "content": WORLD_PROMPT}, {"role": "user", "content": text}]
        return _call(self.chat, messages, min(max_tokens, WORLD_MAX_TOKENS))


def build_mind(
    client: VllmClient, ledger: Ledger, prices: Prices, *, seed: int, wake: Callable[[], WakeResult] | None,
) -> MindNight:
    glimmer = glimmer_chat(client, seed=seed, wake=wake)

    def models(user_id: str, spark: SparkWrap) -> MindModels:
        return MindModels(
            thinker=GlimmerThinker(glimmer), world=GlimmerWorldQuery(glimmer),
            screen=SparkScreen(names=person_names(user_id), chat=spark(spark_open)),
            resolver=SparkResolver(chat=spark(spark_open)), model=GLIMMER,
        )

    def research(call: MindCall) -> WonderResearch:
        return WonderResearch(
            user_id=call.user_id, night=call.night, at=call.at, store=call.store, meter=call.wonder,
            wonder=MetaWonder(), search=WikipediaSearch(), fetcher=SafeFetcher(), ledger=ledger,
            names=tuple(person_names(call.user_id)),
        )

    return MindNight(models=models, log=workspace_log, prices=prices, research=research, ledger=ledger)
