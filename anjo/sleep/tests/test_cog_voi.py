# mypy: disallow-untyped-decorators=False
"""The cog core's inquirer (``anjo.sleep.mind.cog.voi``): the value formula on a stub predictor,
then the tiny trained core (``test_cog_core.trained_tiny``) on the toy world: it asks about work,
not the weather; it stays silent when questions cost more than they are worth; and weighing a
question never changes the live state."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from anjo.sleep.mind.cog import core
from anjo.sleep.mind.cog.toy_life import ToyLife
from anjo.sleep.mind.cog.types import Event, Prediction, Query, Question
from anjo.sleep.mind.cog.voi import MindInquirer, entropy
from anjo.sleep.tests.test_cog_core import DEV, trained_tiny


# ── the formula ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Stub:
    """p = 0.5 before any answer; after the answer ("x", "hi") p = 0.9, after ("x", "lo") p = 0.1;
    an answer to the "noise" question leaves p at 0.5."""

    day: int = 0
    answer: tuple[str, ...] = ()


class StubPredictor:
    def __init__(self) -> None:
        self.live = _Stub()

    def peek(self, day: int, events: Sequence[Event], state: _Stub | None = None) -> _Stub:
        return _Stub(day, tuple(events[-1].tokens))

    def predict_on(self, state: _Stub, queries: Sequence[Query]) -> tuple[list[Prediction], object, object]:
        p = {("x", "hi"): 0.9, ("x", "lo"): 0.1}.get(state.answer, 0.5)
        return [Prediction(q.qid, p) for q in queries], None, None

    def answer_probs(self, question: Question, state: _Stub | None = None) -> np.ndarray:
        return np.array([0.75, 0.25]) if question.subject == "x" else np.array([0.5, 0.5])


def _q(qid: str, subject: str, cost: float, options: tuple[tuple[str, ...], ...]) -> Question:
    return Question(qid, 0, subject, "", (subject,), cost, options)


UPCOMING = tuple(Query(f"u{i}", 0, 1 + i, "run", "", ("will", "run"), "hidden") for i in range(3))


def test_value_is_the_information_gain_on_the_upcoming_queries() -> None:
    inq = MindInquirer(StubPredictor(), lam=1.0)  # type: ignore[arg-type]
    useful = _q("a", "x", 0.1, (("x", "hi"), ("x", "lo")))
    noise = _q("b", "noise", 0.0, (("n", "1"), ("n", "2")))
    values = inq.values(0, (useful, noise), UPCOMING)
    want = 3 * (entropy(0.75 * 0.9 + 0.25 * 0.1) - (0.75 * entropy(0.9) + 0.25 * entropy(0.1)))
    assert math.isclose(values["a"], want, rel_tol=1e-9)
    assert math.isclose(values["b"], 0.0, abs_tol=1e-12)
    assert inq.choose(0, (useful, noise), UPCOMING) == "a"
    assert MindInquirer(StubPredictor(), lam=want / 0.1 + 1.0).choose(0, (useful, noise), UPCOMING) is None  # type: ignore[arg-type]


def test_no_upcoming_queries_means_silence() -> None:
    inq = MindInquirer(StubPredictor(), lam=0.0)  # type: ignore[arg-type]
    assert inq.choose(0, (_q("a", "x", 0.0, (("x", "hi"), ("x", "lo"))),), ()) is None


def test_lam_must_not_be_negative() -> None:
    with pytest.raises(ValueError):
        MindInquirer(StubPredictor(), lam=-1.0)  # type: ignore[arg-type]


# ── on the trained core ──────────────────────────────────────────────────────


def _upcoming(life: ToyLife, day: int) -> tuple[Query, ...]:
    """What ``run.py`` passes as upcoming: today's queries."""
    return tuple(life.queries(day))


def _walk(lam: float) -> tuple[list[dict[str, float]], list[str | None]]:
    pred = core.MindPredictor(trained_tiny(), night="off")
    inq = MindInquirer(pred, lam=lam)
    values, chosen = [], []
    for seed in DEV[:12]:
        life = ToyLife(seed)
        pred.begin(life.meta)
        for d in range(life.meta.n_days - 3):
            pred.observe(d, life.events(d))
            if d % 4 == 1:
                questions, upcoming = life.questions(d), _upcoming(life, d)
                values.append({q.subject: v for q, v in zip(questions, inq.values(d, questions, upcoming).values(),
                                                           strict=True)})
                pick = inq.choose(d, questions, upcoming)
                chosen.append(None if pick is None else pick.split("-")[-2])
            pred.predict(life.queries(d))
            pred.end_day(d)
    return values, chosen


def test_the_inquirer_asks_about_work_not_the_weather() -> None:
    values, chosen = _walk(lam=0.0)
    work = np.array([v["work"] for v in values])
    weather = np.array([v["weather"] for v in values])
    assert work.mean() > 5 * weather.mean()
    assert np.mean(work > weather) > 0.8
    assert sum(c == "work" for c in chosen) > 0.8 * len(chosen)


def test_the_inquirer_stays_silent_when_questions_cost_too_much() -> None:
    _, chosen = _walk(lam=1e6)
    assert chosen and all(c is None for c in chosen)


def test_weighing_questions_never_changes_the_live_state() -> None:
    pred = core.MindPredictor(trained_tiny())
    life = ToyLife(DEV[0])
    pred.begin(life.meta)
    for d in range(10):
        pred.observe(d, life.events(d))
        pred.predict(life.queries(d))
        pred.end_day(d)
    pred.observe(10, life.events(10))
    before = pred.live
    want = pred.predict_on(before, _upcoming(life, 10))[0]
    MindInquirer(pred, lam=0.0).choose(10, life.questions(10), _upcoming(life, 10))
    assert pred.live is before
    assert pred.predict_on(pred.live, _upcoming(life, 10))[0] == want
