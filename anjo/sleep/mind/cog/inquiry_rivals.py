"""The rival inquirers of MIND's cognition sprint (T3): never ask, ask at random, ask about the
most uncertain subject.

``MaxUncertainty`` is the total-uncertainty rival DESIGN.md says should fail the distractor test:
it scores a question by its predictor's uncertainty about the question's subject (the largest
``u`` among today's queries on that subject, or their binary entropy when ``u`` is None), scores
a question on a subject no query is about by the mean over all of today's queries, and counts a
topic (a distractor) as maximally uncertain, since nothing predicts it. It asks whenever a
question is offered.
"""
from __future__ import annotations

import math
import zlib
from collections.abc import Sequence

import numpy as np

from anjo.sleep.mind.cog.types import LifeMeta, Prediction, Predictor, Query, Question


class NeverAsk:
    name = "never"

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        return None


class RandomAsk:
    """Asks with probability ``p_ask`` a uniformly drawn question. Each draw is seeded by
    (seed, day, the day's question ids), so it is the same in any process and differs by life."""

    def __init__(self, seed: int = 0, p_ask: float = 0.3) -> None:
        if not 0.0 <= p_ask <= 1.0:
            raise ValueError(f"p_ask must be in [0, 1], got {p_ask}")
        self.seed, self.p_ask = seed, p_ask
        self.name = "random"

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        if not questions:
            return None
        ids = "\n".join(q.question_id for q in questions)
        rng = np.random.default_rng([self.seed, day, zlib.crc32(ids.encode("utf-8"))])
        if rng.random() >= self.p_ask:
            return None
        return questions[int(rng.integers(len(questions)))].question_id


def _uncertainty(pred: Prediction) -> float:
    if pred.u is not None:
        return float(pred.u)
    p = min(max(pred.p, 1e-12), 1 - 1e-12)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


class MaxUncertainty:
    name = "max_uncertainty"

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self.topics: frozenset[str] = frozenset()

    def begin(self, meta: LifeMeta) -> None:
        self.topics = frozenset(meta.topics)

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        if not questions:
            return None
        by_qid = {p.qid: p for p in self.predictor.predict(upcoming)} if upcoming else {}
        per_subject: dict[str, float] = {}
        for q in upcoming:
            if q.qid in by_qid:
                u = _uncertainty(by_qid[q.qid])
                per_subject[q.subject] = max(u, per_subject.get(q.subject, -math.inf))
        everything = [_uncertainty(p) for p in by_qid.values()]
        overall = sum(everything) / len(everything) if everything else 0.0

        def score(question: Question) -> float:
            if question.subject in self.topics:
                return math.inf
            return per_subject.get(question.subject, overall)

        return max(questions, key=score).question_id
