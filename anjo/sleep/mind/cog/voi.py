"""MIND's inquirer: ask the one question worth asking, by its information value, or stay silent.

For a question ``Q`` with options ``o`` and the upcoming queries ``j`` (the caller's; ``run.py``
passes today's), with ``p_j|o`` the core's prediction after observing the answer ``o`` (on a clone
of the person's state, ``MindPredictor.peek``; the live state never moves) and ``P(o)`` the answer
head's distribution:

    value(Q) = sum_j [ H(pbar_j) - sum_o P(o) H(p_j|o) ],   pbar_j = sum_o P(o) p_j|o   (nats)

``H`` is the binary entropy. This is the information gain: the mutual information between the
answer and each outcome under the core's own joint ``P(o) p(y_j | o)``. When the answer head agrees
with the predictive (``pbar_j = p_j``, the prediction now) it equals the expected reduction of the
predictive's entropy, which is the expected log-loss reduction of a calibrated core. We use the
mixture rather than ``p_j`` because the two heads are trained apart: with ``p_j`` an answer head
that disagrees with the predictive gives negative values (we measured -0.05 nats on the toy world);
the information gain is never negative, and is zero for an answer that moves no prediction (a
distractor's), however uncertain the answer itself is. The inquirer asks the question with the
largest ``value - lam * cost`` if its value is more than ``lam * cost``, else returns None.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from anjo.sleep.mind.cog.types import Event, Prediction, Query, Question

_FLOOR = 1e-6


def entropy(p: float) -> float:
    p = min(max(p, _FLOOR), 1 - _FLOOR)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


class Weighable(Protocol):
    """What the inquirer needs of a predictor (``MindPredictor`` has it): pure views of its state."""

    @property
    def live(self) -> Any: ...
    def peek(self, day: int, events: Sequence[Event], state: Any = None) -> Any: ...
    def predict_on(self, state: Any, queries: Sequence[Query]) -> tuple[list[Prediction], Any, Any]: ...
    def answer_probs(self, question: Question, state: Any = None) -> np.ndarray: ...


def _reply(day: int, question: Question, option: Sequence[str]) -> Event:
    return Event(day, "answer", question.subject, " ".join(option), tuple(option))


class MindInquirer:
    """An ``Inquirer`` over a ``MindPredictor``'s live state."""

    def __init__(self, predictor: Weighable, *, lam: float = 1.0, name: str = "mind-voi") -> None:
        if lam < 0:
            raise ValueError("lam is the price of a unit of cost, at least 0")
        self.predictor, self.lam, self.name = predictor, lam, name

    def values(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> dict[str, float]:
        """Each question's value in nats (questions without options are worth 0)."""
        pred = self.predictor
        live = pred.live
        out: dict[str, float] = {}
        for q in questions:
            if not q.options or not upcoming:
                out[q.question_id] = 0.0
                continue
            probs = np.asarray(pred.answer_probs(q, live), dtype=np.float64)
            after = np.array([[p.p for p in pred.predict_on(pred.peek(day, (_reply(day, q, o),), live), upcoming)[0]]
                              for o in q.options])  # (O, J)
            mixture = probs @ after
            cond = probs @ np.vectorize(entropy)(after)
            out[q.question_id] = float(sum(entropy(m) for m in mixture) - cond.sum())
        return out

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        if not questions or not upcoming:
            return None
        values = self.values(day, questions, upcoming)
        best = max(questions, key=lambda q: values[q.question_id] - self.lam * q.cost)
        return best.question_id if values[best.question_id] > self.lam * best.cost else None
