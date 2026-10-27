"""What every ``evidence_recall`` shares (Task 2): token estimate, eligibility and the budget.

``evidence_recall`` returns the top-k evidence items for a turn inside a token budget. It never
returns a held-out item, even from a store that leaks one. Two modes: ``serve`` (served context:
every stored item, homework and practice included) and ``sft`` (the inputs of a training
example: no ``practice`` item and no item without cleared ancestry, so a target never reaches its
own input and no hosted text enters training).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Final, Literal, Protocol

from anjo.sleep.provenance import may_train
from anjo.sleep.types import Evidence

RecallMode = Literal["serve", "sft"]
CHARS_PER_TOKEN: Final = 4  # a slight over-count for English, so budgets are never exceeded


def estimate_tokens(text: str) -> int:
    """A conservative token count for ``text``, with no tokenizer."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class RecallEngine(Protocol):
    """The ranker behind ``ContextBuilder.recall``: BM25 here, MiniLM in the app."""

    def recall(
        self, user_id: str, query: str, k: int, budget_tokens: int, *, mode: RecallMode = "serve",
    ) -> tuple[Evidence, ...]: ...


def eligible(
    items: Iterable[Evidence], mode: RecallMode, *, hosted_in_training: bool = False,
) -> list[Evidence]:
    """The items ``mode`` may return: never held-out, and in ``sft`` mode only cleared ones that
    are not practice."""
    kept = [item for item in items if not item.held_out]
    if mode == "serve":
        return kept
    return [
        item for item in kept
        if item.kind != "practice" and may_train(item.ancestry, hosted_in_training=hosted_in_training)
    ]


def fit_budget(ranked: Sequence[Evidence], k: int, budget_tokens: int) -> tuple[Evidence, ...]:
    """Walk ``ranked`` best first and keep up to ``k`` items whose texts fit ``budget_tokens``
    together. An item that doesn't fit is skipped, so a smaller, lower-ranked one can take its
    place."""
    chosen: list[Evidence] = []
    used = 0
    for item in ranked:
        if len(chosen) >= k:
            break
        cost = estimate_tokens(item.text)
        if used + cost <= budget_tokens:
            chosen.append(item)
            used += cost
    return tuple(chosen)


def order_key(score: float, item: Evidence) -> tuple[float, float, str]:
    """Sort key for ``sorted``: best score first, then newest, then id, so ties are stable."""
    return (-score, -item.created_at.timestamp(), item.id)
