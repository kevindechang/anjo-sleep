"""A critic for full practice that is not cleared to train (``roles``: ``critic=deepseek``).

``provenance`` traces a training row's ancestry through every model whose output its text derives
from, and what that model read. Practice's trained text is a candidate: Glimmer's, or the teacher's
(Spark Open), each written from F and the scene's message only. The critic reads the candidates and
answers with scores and the fixed category ``critic_v1`` asks for (the letters that miss a
constraint); ``full._marks`` reads those numbers and letters and nothing else, so no word of its reply
reaches a transcript, the teacher or a training row, and its model never enters an ancestry. A reply
cut at its cap is never read: its scene is dropped. Its verdict still chooses which candidate trains
and which is the pair's worse side: a preference label, not text.

DeepSeek V4.1 Flash runs on Fireworks' serverless API through the core splits' generator call
(``fireworks_api.generate``, ``mind.apd.generators.DEEPSEEK``: non-thinking, MIT licence, $0.30 and
$1.20 a million tokens), each call held first on ``fireworks_sleep_usd`` at its worst and settled at its
list price (``held_critic``).
"""
from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from anjo.sleep import fireworks_api
from anjo.sleep.budget import Ledger, Pool
from anjo.sleep.mind.apd.generators import DEEPSEEK
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, held, usd

Ask = Callable[[str], ChatResult]  # the critic prompt -> its reply, tokens and finish reason
POOL: Final[Pool] = "fireworks_sleep_usd"
MAX_TOKENS: Final = 512  # the critic's visible cap (its JSON); non-thinking, so nothing more to make room for
TIMEOUT_S: Final = 120.0
assert DEEPSEEK.model is not None and DEEPSEEK.list_prices is not None
DEEPSEEK_IN_PER_M, DEEPSEEK_OUT_PER_M = DEEPSEEK.list_prices


@dataclass(frozen=True)
class ThirdPartyCritic:
    """``ask`` takes the critic prompt and returns its reply; ``model`` is logged, never trained."""

    model: str
    ask: Ask
    usd_per_m_in: float
    usd_per_m_out: float
    pool: Pool = POOL

    def usd(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_per_m_in + tokens_out * self.usd_per_m_out) / 1e6

    @property
    def worst_usd(self) -> float:
        return self.usd(SPARK_MAX_IN_TOKENS, MAX_TOKENS)


def held_critic(critic: ThirdPartyCritic, ledger: Ledger | None, job: str) -> ThirdPartyCritic:
    """``critic`` with each call held on its pool at its worst and settled at what it used."""
    numbers = itertools.count(1)  # two calls on one prompt (a retest) hold apart

    def ask(prompt: str) -> ChatResult:
        spent = [0.0]
        key = f"{next(numbers)}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"
        with held(ledger, f"{job}:critic:{key}", (critic.pool,), usd(critic.worst_usd), lambda: spent[0]):
            reply = critic.ask(prompt)
            spent[0] = critic.usd(reply.input_tokens, reply.output_tokens)
        return reply

    return ThirdPartyCritic(critic.model, ask, critic.usd_per_m_in, critic.usd_per_m_out, critic.pool)


def deepseek_critic(generate: Callable[..., ChatResult] = fireworks_api.generate) -> ThirdPartyCritic:
    """DeepSeek V4.1 Flash as the critic: temperature 0, non-thinking, its reply capped at ``MAX_TOKENS``."""
    model = DEEPSEEK.model
    assert model is not None

    def ask(prompt: str) -> ChatResult:
        result: ChatResult = generate(model, [{"role": "user", "content": prompt}], None, TIMEOUT_S, temperature=0.0,
                                      max_tokens=MAX_TOKENS, reasoning_effort=DEEPSEEK.reasoning_effort)
        return result

    return ThirdPartyCritic(model, ask, DEEPSEEK_IN_PER_M, DEEPSEEK_OUT_PER_M)
