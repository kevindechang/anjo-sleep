"""BM25 ``evidence_recall`` (Task 2): the lexical ranker, public parts only.

The app ranks with Anjo's MiniLM embedder instead (``anjo.sleep.app.recall``); this one serves
the standalone example (Task 15) and the tests. Okapi BM25 over the eligible items, ties broken
by newest then id, so the same store and query always give the same result.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import date
from typing import Final

from anjo.sleep.context.recall_common import RecallMode, eligible, fit_budget, order_key
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.types import Evidence

K1: Final = 1.5
B: Final = 0.75
_WORD = re.compile(r"\w+")


def _terms(text: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(text)]


def bm25_scores(items: list[Evidence], query: str) -> list[float]:
    """The BM25 score of each item for ``query``, in the order of ``items``."""
    docs = [Counter(_terms(item.text)) for item in items]
    lengths = [sum(doc.values()) for doc in docs]
    average = (sum(lengths) / len(lengths)) if lengths else 0.0
    wanted = set(_terms(query))
    count = len(docs)
    scores = [0.0] * count
    for term in wanted:
        holding = sum(1 for doc in docs if term in doc)
        if not holding:
            continue
        idf = math.log(1.0 + (count - holding + 0.5) / (holding + 0.5))
        for index, doc in enumerate(docs):
            frequency = doc.get(term, 0)
            if frequency:
                norm = 1.0 - B + B * (lengths[index] / average if average else 1.0)
                scores[index] += idf * frequency * (K1 + 1.0) / (frequency + K1 * norm)
    return scores


class Bm25Recall:
    """``RecallEngine`` over one evidence store per person."""

    def __init__(
        self,
        evidence: Callable[[str], EvidenceStore],
        *,
        until: date,
        hosted_in_training: bool = False,
    ) -> None:
        self._evidence = evidence
        self._until = until
        self._hosted_in_training = hosted_in_training

    def recall(
        self, user_id: str, query: str, k: int, budget_tokens: int, *, mode: RecallMode = "serve",
    ) -> tuple[Evidence, ...]:
        pool = eligible(
            self._evidence(user_id).stored(self._until), mode,
            hosted_in_training=self._hosted_in_training,
        )
        scores = bm25_scores(pool, query)
        ranked = sorted(
            ((score, item) for score, item in zip(scores, pool, strict=True) if score > 0.0),
            key=lambda pair: order_key(*pair),
        )
        return fit_budget([item for _, item in ranked], k, budget_tokens)
