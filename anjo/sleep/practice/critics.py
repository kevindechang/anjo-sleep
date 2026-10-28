"""A critic for full practice that is not cleared to train (``roles``: ``critic=gemini``).

``provenance`` traces a training row's ancestry through every model whose output its text derives
from, and what that model read. Practice's trained text is a candidate: Glimmer's, or the teacher's
(Spark Open), each written from F and the scene's message only. The critic reads the candidates and
answers with scores and the letters that miss a constraint; ``full._marks`` reads those numbers and
letters and nothing else, so no word of its reply reaches a transcript, the teacher or a training row,
and its model never enters an ancestry. Its verdict still chooses which candidate trains and which is
the pair's worse side: a preference label, not text. Whether its provider's terms allow that use of a
label is a terms question this module can't answer (see the A/B's notes).

The critic is called through the APD judge's existing Vertex path (``mind.apd.backend``), each call
held first on ``vertex_usd`` at its worst and settled at its list price (``held``).
"""
from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from anjo.sleep.budget import Ledger
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, held, usd
from anjo.sleep.probes.second_judge import MODEL_ID as GEMINI_FLASH

TextModel = Callable[[str], tuple[str, int, int]]  # a prompt -> the reply, its tokens in and out
GEMINI_IN_PER_M: Final = 0.75  # Gemini 3.8 Flash on Vertex, list price (2026-10-28)
GEMINI_OUT_PER_M: Final = 3.75
WORST_OUT_TOKENS: Final = 400 + 512  # the APD path's cap: its reply cap and the thinking it reserves


@dataclass(frozen=True)
class ThirdPartyCritic:
    """``ask`` takes the critic prompt and returns its reply and tokens; ``model`` is logged, never trained."""

    model: str
    ask: TextModel
    usd_per_m_in: float
    usd_per_m_out: float

    def usd(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_per_m_in + tokens_out * self.usd_per_m_out) / 1e6

    @property
    def worst_usd(self) -> float:
        return self.usd(SPARK_MAX_IN_TOKENS, WORST_OUT_TOKENS)


def held_critic(critic: ThirdPartyCritic, ledger: Ledger | None, job: str) -> ThirdPartyCritic:
    """``critic`` with each call held on ``vertex_usd`` at its worst and settled at what it used."""

    numbers = itertools.count(1)  # two calls on one prompt (a retest) hold apart

    def ask(prompt: str) -> tuple[str, int, int]:
        spent = [0.0]
        key = f"{next(numbers)}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"
        with held(ledger, f"{job}:critic:{key}", ("vertex_usd",), usd(critic.worst_usd), lambda: spent[0]):
            text, tokens_in, tokens_out = critic.ask(prompt)
            spent[0] = critic.usd(tokens_in, tokens_out)
        return text, tokens_in, tokens_out

    return ThirdPartyCritic(critic.model, ask, critic.usd_per_m_in, critic.usd_per_m_out)


def gemini_critic(ask: TextModel) -> ThirdPartyCritic:
    return ThirdPartyCritic(GEMINI_FLASH, ask, GEMINI_IN_PER_M, GEMINI_OUT_PER_M)
