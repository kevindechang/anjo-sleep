"""``evidence_recall`` ranked by Anjo's existing MiniLM embedder (Task 2).

One implementation per job: this wraps ``anjo.memory.embedder`` (the embedder Anjo already
ships) instead of adding a second embedding path. It ranks the same eligible items as the BM25
recall (``anjo.sleep.context.recall_bm25``), inside the same token budget, and never returns a
held-out item. The embedder is injected for tests; the default loads MiniLM on first use.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

from anjo.sleep.context.recall_common import RecallMode, eligible, fit_budget, order_key
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.types import Evidence

Embed = Callable[[str], Sequence[float]]


def _default_embed() -> Embed:
    from anjo.memory.embedder import embed_semantic

    embed: Embed = embed_semantic
    return embed


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class MiniLmRecall:
    """``RecallEngine`` over one evidence store per person. Vectors are unit length, so the dot
    product is the cosine. Item vectors are kept for the life of the object."""

    def __init__(
        self,
        evidence: Callable[[str], EvidenceStore],
        *,
        until: date,
        embed: Embed | None = None,
        hosted_in_training: bool = False,
    ) -> None:
        self._evidence = evidence
        self._until = until
        self._embed = embed if embed is not None else _default_embed()
        self._hosted_in_training = hosted_in_training
        self._vectors: dict[str, Sequence[float]] = {}

    def _vector(self, item: Evidence) -> Sequence[float]:
        if item.id not in self._vectors:
            self._vectors[item.id] = self._embed(item.text)
        return self._vectors[item.id]

    def recall(
        self, user_id: str, query: str, k: int, budget_tokens: int, *, mode: RecallMode = "serve",
    ) -> tuple[Evidence, ...]:
        pool = eligible(
            self._evidence(user_id).stored(self._until), mode,
            hosted_in_training=self._hosted_in_training,
        )
        if not pool:
            return ()
        wanted = self._embed(query)
        ranked = sorted(
            ((_dot(wanted, self._vector(item)), item) for item in pool),
            key=lambda pair: order_key(*pair),
        )
        return fit_budget([item for _, item in ranked], k, budget_tokens)
