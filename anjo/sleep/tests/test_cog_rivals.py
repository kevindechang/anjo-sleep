"""The rival predictors (rivals.py) and rival inquirers (inquiry_rivals.py), on the fake life."""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import pytest

from anjo.sleep.mind.cog import score
from anjo.sleep.mind.cog.inquiry_rivals import MaxUncertainty, NeverAsk, RandomAsk
from anjo.sleep.mind.cog.rivals import (
    Counter,
    FeatureSpec,
    OnlineLogistic,
    PersonHistory,
    PopulationPrior,
    SurpriseLR,
    features,
    is_late,
    token_ints,
)
from anjo.sleep.mind.cog.run import run_life, run_many
from anjo.sleep.mind.cog.types import Event, LifeMeta, Prediction, Query, Question
from anjo.sleep.tests.test_cog_run import FakeLife, FreqPredictor

META = LifeMeta(person_id="p", n_days=40, start_weekday=0, activities=("run", "gym"),
                people=("sam",), topics=("weather",))


def _life(seed: int, n_days: int = 56, shift: bool = False) -> FakeLife:
    return FakeLife(seed, n_days=n_days, shift_day=n_days // 2 if shift else None, rotate=seed % 7,
                    idiosyncratic=seed % 4 == 0)


@pytest.fixture(scope="module")
def train() -> list[FakeLife]:
    return [_life(s) for s in range(40)]


@pytest.fixture(scope="module")
def population(train: list[FakeLife]) -> object:
    return OnlineLogistic.fit_population(train)


@pytest.fixture(scope="module")
def dev_runs(population: object) -> dict[str, tuple]:
    dev = [_life(s) for s in range(1000, 1020)]
    prior = population.spec.prior  # type: ignore[attr-defined]
    return {
        "population": run_many(dev, partial(PopulationPrior, prior)),
        "counter": run_many(dev, partial(Counter, prior)),
        "online_lr": run_many(dev, partial(OnlineLogistic, population)),
    }


# -- reading a life from fields and tokens --
def test_history_reads_plans_cancels_moods_and_late_messages() -> None:
    h = PersonHistory(META)
    h.add([
        Event(0, "plan", "run", "x", ("plan", "run"), value=2.0),
        Event(0, "plan", "gym", "x", ("plan", "gym", "day=3")),
        Event(1, "cancel", "gym", "x", ("cancel", "gym")),
        Event(1, "mood", "self", "x", ("mood", "low")),
        Event(1, "mood", "self", "x", ("mood",), value=4.0),
        Event(1, "message", "work", "x", ("message", "late")),
        Event(1, "message", "work", "x", ("message", "hour=23")),
        Event(1, "message", "sam", "x", ("message", "hour=14")),
        Event(2, "did", "run", "x", ("did", "run")),
        Event(2, "did", "swim", "x", ("did", "swim")),
    ])
    assert h.plans == {("run", 2): 0, ("gym", 3): 0}
    assert h.live_plan("gym", 3, 0) and not h.live_plan("gym", 3, 1)  # the cancel hit the next plan
    assert h.mood_mean(1, 1) == pytest.approx(3.0)
    assert h.total(h.late, 1, 1) == 2 and h.total(h.messages, 1, 1) == 3
    assert h.outcome("run", 2) == 1 and h.outcome("gym", 3) == 0 and h.outcome("run", 3) == 0
    assert h.outcome("run", 3, absent_is_negative=False) is None
    assert h.kept(3) == (1, 2) and h.kept(1) == (0, 0)
    assert "swim" in h.activities


def test_token_helpers() -> None:
    assert token_ints(("day=17", "12", "d:3", "run")) == [17, 12, 3]
    assert is_late(("Message", "LATE_night")) and is_late(("hour=2",)) and not is_late(("hour=12",))


def test_weekday_rates_count_absences_and_the_decayed_rate_forgets() -> None:
    h = PersonHistory(META)
    h.add([Event(d, "did", "run", "x", ("did", "run")) for d in (0, 7, 14)])
    assert h.rate("run", 0, 20) == (3, 3) and h.rate("run", 1, 20) == (0, 3)
    h.add([Event(d, "skipped", "run", "x", ("skipped", "run")) for d in (21, 28)])
    assert h.rate("run", 0, 28) == (3, 5)
    assert h.decayed_rate("run", 0, 28, 0.6) < 3 / 5 < h.decayed_rate("run", 0, 14, 0.6)


def test_features_see_nothing_after_the_asked_day() -> None:
    life = FakeLife(3, n_days=20)
    spec = FeatureSpec(vocab=("class", "gym", "run"))
    early, full = PersonHistory(life.meta), PersonHistory(life.meta)
    for d in range(20):
        full.add(life.events(d))
        if d <= 9:
            early.add(life.events(d))
    early.add([life.answer(life.questions(9)[0].question_id)])
    full.add([life.answer(life.questions(9)[0].question_id)])
    for a in ("run", "gym", "class"):
        np.testing.assert_array_equal(features(spec, early, a, 9, 11, "habit"),
                                      features(spec, full, a, 9, 11, "habit"))


def test_feature_names_match_the_vector_and_mark_the_personal_weights() -> None:
    spec = FeatureSpec(vocab=("gym", "run"))
    x = features(spec, PersonHistory(META), "run", 0, 1, "habit")
    assert len(x) == spec.size == len(spec.names)
    personal = {n for n, p in zip(spec.names, spec.personal) if p}
    assert {"counter_logit", "recent_logit", "act=run*late3", "act=<other>*mood3"} <= personal
    assert "late3" not in personal and "bias" not in personal


# -- the population prior and the counter --
def test_population_prior_knows_the_weekday_habit_and_backs_off(train: list[FakeLife]) -> None:
    table = PopulationPrior.fit_population([FakeLife(s, n_days=42) for s in range(20)])
    on = table.p("habit", "class", 0, 1)  # class is every weekday; the weekend is off
    off = table.p("habit", "class", 6, 1)
    assert on > 0.6 > 0.2 > off
    unknown = table.p("habit", "piano", 2, 1)
    assert unknown == pytest.approx(table.p("habit", "piano", 5, 3))


def test_counter_learns_a_person_unlike_the_population() -> None:
    table = PopulationPrior.fit_population([FakeLife(s, n_days=42) for s in range(20)])
    life = FakeLife(5, n_days=49, rotate=3)  # every habit three weekdays later
    counter = Counter(table)
    rec = run_life(life, counter)
    prior = run_life(life, PopulationPrior(table))
    late_c = [q for q in rec.queries if q.asked_day >= 28]
    late_p = [q for q in prior.queries if q.asked_day >= 28]
    loss = lambda qs: float(score.log_loss(np.array([q.p for q in qs]), np.array([q.outcome for q in qs])).mean())  # noqa: E731
    assert loss(late_c) < loss(late_p) - 0.1
    first, last = rec.queries[0].u, rec.queries[-1].u
    assert first is not None and last is not None and last < first


def test_counter_uses_the_kept_rate_for_plan_queries() -> None:
    counter = Counter(None, plan_strength=2.0)
    counter.begin(META)
    counter.observe(0, [Event(0, "plan", "run", "x", ("plan", "run"), value=1.0)])
    counter.observe(1, [Event(1, "did", "run", "x", ("did", "run"))])
    q = Query("q", 1, 4, "gym", "Will they keep their plan?", ("query", "gym", "plan"), "hidden")
    (pred,) = counter.predict([q])
    assert pred.p == pytest.approx((1 + 2 * 0.5) / (1 + 2))


# -- the logistic rivals --
def test_online_logistic_beats_the_counter_and_the_prior(dev_runs: dict[str, tuple]) -> None:
    ll = {name: score.summarize_run(recs)["log_loss"] for name, recs in dev_runs.items()}
    assert ll["online_lr"] < ll["counter"] - 0.01
    assert ll["counter"] < ll["population"] - 0.03


def test_online_logistic_uncertainty_shrinks_with_data(dev_runs: dict[str, tuple]) -> None:
    # u is parameter uncertainty only; whether it tracks the real error is T2's question, not a given
    qs = [q for r in dev_runs["online_lr"] for q in r.queries]
    week0 = np.mean([q.u for q in qs if q.asked_day < 7])
    later = np.mean([q.u for q in qs if q.asked_day >= 42])
    assert all(q.u is not None and q.u > 0 for q in qs) and later < week0


def test_online_logistic_predict_has_no_side_effects(population: object) -> None:
    life = FakeLife(7, n_days=10)
    a, b = OnlineLogistic(population), OnlineLogistic(population)  # type: ignore[arg-type]
    for m in (a, b):
        m.begin(life.meta)
    for d in range(10):
        for m in (a, b):
            m.observe(d, life.events(d))
        first = a.predict(life.queries(d))
        assert a.predict(life.queries(d)) == first == b.predict(life.queries(d))
        for m in (a, b):
            m.end_day(d)
    np.testing.assert_array_equal(a.m, b.m)


def test_a_busy_answer_lowers_the_run_after_the_population_saw_asks(population: object) -> None:
    life = FakeLife(11, n_days=14)
    q = Query("q", 6, 7, "run", "Will they run?", ("query", "run"), "hidden")
    ps = {}
    for word in ("calm", "busy"):
        m = OnlineLogistic(population)  # type: ignore[arg-type]
        m.begin(life.meta)
        for d in range(7):
            m.observe(d, life.events(d))
            if d < 6:
                m.end_day(d)
        m.observe(6, [Event(6, "answer", "work", "x", ("answer", "work", word))])
        ps[word] = m.predict([q])[0].p
    assert ps["busy"] < ps["calm"] - 0.05


def test_online_logistic_runs_without_a_population() -> None:
    rec = run_life(FakeLife(2, n_days=14), OnlineLogistic())
    assert all(0 < q.p < 1 and q.u is not None for q in rec.queries)


def test_surprise_raises_the_learning_rate_after_a_shift(population: object) -> None:
    life = FakeLife(9, n_days=56, shift_day=28)
    online = OnlineLogistic(population)  # type: ignore[arg-type]
    surprised = SurpriseLR(population, gain=5.0)  # type: ignore[arg-type]
    r_on, r_su = run_life(life, online), run_life(life, surprised)
    after = [i for i, q in enumerate(r_on.queries) if q.asked_day >= 28]
    assert max(c for d, c in surprised.cpp_trace) > surprised.hazard
    assert np.mean([r_su.queries[i].u for i in after]) > np.mean([r_on.queries[i].u for i in after])
    assert surprised.name == "surprise_lr" and np.trace(surprised.cov) > np.trace(online.cov)


def test_logistic_rivals_run_the_same_in_a_process_pool(population: object) -> None:
    lives = [_life(s, n_days=14) for s in (1, 2, 3)]
    factory = partial(SurpriseLR, population)
    assert run_many(lives, factory) == run_many(lives, factory, workers=2)


# -- the rival inquirers --
def _question(qid: str, subject: str) -> Question:
    return Question(qid, 0, subject, "?", ("ask", subject), 0.1, options=(("answer", subject, "x"),))


def _query(qid: str, subject: str) -> Query:
    return Query(qid, 0, 1, subject, "?", ("query", subject), "count")


class _Fixed:
    name = "fixed"

    def __init__(self, table: dict[str, tuple[float, float | None]]) -> None:
        self.table = table

    def predict(self, queries: list[Query]) -> list[Prediction]:
        return [Prediction(q.qid, *self.table[q.qid]) for q in queries]


def test_never_ask_never_asks() -> None:
    assert NeverAsk().choose(0, [_question("a", "work")], []) is None


def test_random_ask_is_seeded_by_day_and_questions() -> None:
    qs = [_question("a", "work"), _question("b", "weather")]
    assert all(RandomAsk(1, 0.0).choose(d, qs, []) is None for d in range(20))
    always = [RandomAsk(1, 1.0).choose(d, qs, []) for d in range(40)]
    assert set(always) == {"a", "b"}
    assert [RandomAsk(1, 0.5).choose(d, qs, []) for d in range(40)] == \
        [RandomAsk(1, 0.5).choose(d, qs, []) for d in range(40)]
    assert RandomAsk(1, 1.0).choose(0, [], []) is None
    with pytest.raises(ValueError):
        RandomAsk(0, 1.5)


def test_max_uncertainty_falls_for_the_distractor() -> None:
    inq = MaxUncertainty(_Fixed({"q1": (0.5, 0.9), "q2": (0.5, 0.1)}))
    inq.begin(META)
    qs = [_query("q1", "run"), _query("q2", "gym")]
    assert inq.choose(0, [_question("a", "gym"), _question("t", "weather")], qs) == "t"
    assert inq.choose(0, [_question("a", "gym"), _question("b", "run")], qs) == "b"
    assert inq.choose(0, [], qs) is None


def test_max_uncertainty_uses_entropy_without_u_and_the_mean_for_other_subjects() -> None:
    inq = MaxUncertainty(_Fixed({"q1": (0.5, None), "q2": (0.95, None)}))
    inq.begin(META)
    qs = [_query("q1", "run"), _query("q2", "gym")]
    assert inq.choose(0, [_question("a", "gym"), _question("b", "run")], qs) == "b"
    mean_h = (math.log(2) + -(0.95 * math.log(0.95) + 0.05 * math.log(0.05))) / 2
    assert inq.choose(0, [_question("a", "gym"), _question("w", "work")], qs) == "w"
    assert mean_h > -(0.95 * math.log(0.95) + 0.05 * math.log(0.05))


def test_max_uncertainty_spends_every_ask_on_distractors_in_the_fake() -> None:
    predictor = FreqPredictor()
    rec = run_life(FakeLife(4, n_days=10), predictor, MaxUncertainty(predictor))
    assert len(rec.asks) == 10 and all(a.is_distractor for a in rec.asks)
