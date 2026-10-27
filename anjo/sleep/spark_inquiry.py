"""Spark Open behind MIND's model ports (Task 17): a ``Thinker``, ``world_query`` and the blind
``PredictionResolver``, on Baseten's OpenAI-compatible API (``baseten_api``).

The resolver always runs here (``resolve_prediction_v1.md``), and so does the screen
(``anjo.sleep.app.screen``, through ``call``). The live night's thinker is base Glimmer and APD's
is Glimmer on Colab; the Spark Open thinker and ``world_query`` serve development runs and the
thinker swap.

**Capped, and charged when they fail.** Every call is one attempt (``model_api`` doesn't retry),
capped at the smaller of its own ceiling (``*_MAX_TOKENS``) and the allowance the loop passes, and
it charges the reply's generated tokens. A call that fails returns empty text charged its whole
cap: its usage is unknown, and the night's token cap is hard, so a failure is never counted as
free. The failure is logged by its ``model_api`` message, which names the provider, the model and
the HTTP status and never a body or a key. A resolver whose call fails, or whose reply isn't one
of its three verdicts, says ``undetermined``.

**The resolver is blind by construction:** it is called with the claim and the evidence items
only, and its prompt names neither hypotheses nor questions.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep import baseten_api
from anjo.sleep.mind.inquiry.parse import field_value
from anjo.sleep.mind.inquiry.types import ModelReply, ResolverReply, Verdict
from anjo.sleep.model_api import ChatResult, ModelApiError, post_chat
from anjo.sleep.types import Evidence

logger = logging.getLogger(__name__)

PROMPTS: Final = Path(__file__).parent / "mind" / "inquiry" / "prompts"
RESOLVE_PROMPT: Final = (PROMPTS / "resolve_prediction_v1.md").read_text(encoding="utf-8")
WORLD_PROMPT: Final = (PROMPTS / "world_query_v1.md").read_text(encoding="utf-8")
TIMEOUT_S: Final = 120.0
THINK_MAX_TOKENS: Final = 1_500  # one JSON operation, or a notice pass's array
WORLD_MAX_TOKENS: Final = 600
RESOLVE_MAX_TOKENS: Final = 200  # reported apart from the cap (``InquiryRun.resolver_tokens``)
_VERDICTS: Final[dict[str, Verdict]] = {"true": "true", "false": "false", "undetermined": "undetermined"}

ChatFn = Callable[[list[dict[str, Any]], int], ChatResult]  # messages, max_tokens


def spark_open(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
    """One Spark Open call on Baseten at temperature 0, no tools, its reply capped."""
    return post_chat(
        provider="Baseten", url=baseten_api.URL, key_env=baseten_api.KEY_ENV, model=baseten_api.SPARK_OPEN,
        messages=messages, tools=None, timeout_s=TIMEOUT_S, temperature=0.0, transport=None, max_tokens=max_tokens,
    )


def call(chat: ChatFn, messages: list[dict[str, Any]], cap: int) -> ModelReply:
    """One capped call. No allowance left means no call; a failed call is empty text charged
    ``cap``."""
    if cap <= 0:
        return ModelReply(text="", tokens=0, model=baseten_api.SPARK_OPEN)
    try:
        result = chat(messages, cap)
    except ModelApiError as exc:
        logger.warning("MIND's Spark Open call failed, charged its cap of %d tokens: %s", cap, exc)
        return ModelReply(text="", tokens=cap, model=baseten_api.SPARK_OPEN)
    return ModelReply(text=result.text, tokens=result.output_tokens, model=result.model or baseten_api.SPARK_OPEN)


def resolver_messages(claim: str, evidence: Sequence[Evidence]) -> list[dict[str, Any]]:
    lines = "\n".join(f"- ({item.day.isoformat()}) {item.text}" for item in evidence)
    return [
        {"role": "system", "content": RESOLVE_PROMPT},
        {"role": "user", "content": f"Claim: {claim}\n\nEvidence:\n{lines}"},
    ]


def parse_verdict(text: str) -> Verdict:
    """``{"verdict": ...}`` anywhere in the reply, or the reply as one bare word; else
    ``undetermined``."""
    value = field_value(text, "verdict")
    word = value if isinstance(value, str) else text if value is None else ""
    return _VERDICTS.get(word.strip().strip(".").strip().casefold(), "undetermined")


@dataclass(frozen=True)
class SparkThinker:
    """The ``Thinker`` port on Spark Open: the rendered workspace is the whole prompt."""

    chat: ChatFn = spark_open

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        return call(self.chat, [{"role": "user", "content": rendered}], min(max_tokens, THINK_MAX_TOKENS))


@dataclass(frozen=True)
class SparkWorldQuery:
    """``world_query`` on Spark Open: general knowledge, under ``world_query_v1.md``."""

    chat: ChatFn = spark_open

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        messages = [{"role": "system", "content": WORLD_PROMPT}, {"role": "user", "content": text}]
        return call(self.chat, messages, min(max_tokens, WORLD_MAX_TOKENS))


@dataclass(frozen=True)
class SparkResolver:
    """The ``PredictionResolver`` port on Spark Open, under ``resolve_prediction_v1.md``."""

    chat: ChatFn = spark_open

    def resolve(self, claim: str, evidence: Sequence[Evidence]) -> ResolverReply:
        reply = call(self.chat, resolver_messages(claim, evidence), RESOLVE_MAX_TOKENS)
        return ResolverReply(verdict=parse_verdict(reply.text), tokens=reply.tokens)
