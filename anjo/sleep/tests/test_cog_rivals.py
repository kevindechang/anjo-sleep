# mypy: disallow-untyped-decorators=False
"""The rival predictors (rivals.py) and rival inquirers (inquiry_rivals.py), on the fake life."""
from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial

import numpy as np
import pytest

from anjo.sleep.mind.cog import score
from anjo.sleep.mind.cog.inquiry_rivals import MaxUncertainty, NeverAsk, RandomAsk
from anjo.sleep.mind.cog.rival_features import RawFeatures, _answer_timing, raw_features, select_keys
from anjo.sleep.mind.cog.rival_history import stem
from anjo.sleep.mind.cog.rivals import (
    Cells,
    Counter,
    FeatureSpec,
    LogisticPopulation,
    OnlineLogistic,
    PersonHistory,
    PopulationPrior,
    SurpriseLR,
    _table,
    features,
    is_late,
    token_ints,
)
from anjo.sleep.mind.cog.run import LifeRecord, QueryRecord, run_life, run_many
from anjo.sleep.mind.cog.types import Event, LifeMeta, Prediction, Query, Question
from anjo.sleep.tests.test_cog_run import FakeLife, FreqPredictor

META = LifeMeta(person_id="p", n_days=40, start_weekday=0, activities=("run", "gym"),
                people=("sam",), topics=("weather",))
# The fake's habits hang on the weekday (on 1.5, off -2.5 in logit), unlike the cog world's,
# whose queries are only about days an activity can happen: here a weekday's own rate must count.
FAKE_WEEKDAY = 2.0


def _life(seed: int, n_days: int = 56, shift: bool = False) -> FakeLife:
    return FakeLife(seed, n_days=n_days, shift_day=n_days // 2 if shift else None, rotate=seed % 7,
                    idiosyncratic=seed % 4 == 0)


@pytest.fixture(scope="module")
def train() -> list[FakeLife]:
    return [_life(s) for s in range(40)]


Runs = dict[str, tuple[LifeRecord, ...]]


@pytest.fixture(scope="module")
def population(train: list[FakeLife]) -> LogisticPopulation:
    # an answer every day, so the 40 fake lives carry enough of them for the work answer's keys
    return OnlineLogistic.fit_population(train, weekday_strength=FAKE_WEEKDAY, min_groups=10, ask_rate=1.0)


@pytest.fixture(scope="module")
def dev_runs(population: LogisticPopulation) -> Runs:
    dev = [_life(s) for s in range(1000, 1020)]
    prior = population.spec.prior
    assert prior is not None
    return {
        "population": run_many(dev, partial(PopulationPrior, prior)),
        "counter": run_many(dev, partial(Counter, prior, weekday_strength=FAKE_WEEKDAY)),
        "online_lr": run_many(dev, partial(OnlineLogistic, population)),
    }


# -- reading a life from fields and tokens --
def test_history_reads_outcomes_plans_moods_late_messages_and_mentions() -> None:
    h = PersonHistory(META)
    h.add([
        Event(0, "plan", "run", "x", ("plan", "run"), value=2.0),
        Event(0, "plan", "gym", "x", ("plan", "gym", "day=3")),
        Event(1, "mood", "self", "x", ("mood", "low")),
        Event(1, "mood", "self", "x", ("mood",), value=4.0),
        Event(1, "message", "work", "x", ("message", "late")),
        Event(1, "message", "work", "x", ("message", "hour=23")),
        Event(1, "message", "sam", "x", ("message", "hour=14")),
        Event(1, "mention", "knee", "x", ("mention", "knee", "sore")),
        Event(2, "did", "run", "x", ("did", "run")),
        Event(2, "did", "swim", "x", ("did", "swim")),
        Event(3, "cancel", "gym", "x", ("cancel", "gym"), value=3.0),
    ])
    assert h.plans == {("run", 2): 0, ("gym", 3): 0}
    assert h.live_plan("gym", 3, 2) and not h.live_plan("gym", 3, 3)  # decided on its day
    assert h.mood_mean(1, 1) == pytest.approx(3.0)
    assert h.late_count(1, 1) == 2
    assert h.outcome("run", 2) == 1 and h.outcome("gym", 3) == 0
    assert h.outcome("run", 3) is None  # no event is no occurrence, not a miss
    assert h.kept(3) == (1, 2) and h.kept(1) == (0, 0)
    assert h.mentioned(2, 2) == ["knee"] and h.mentioned(3, 2) == []
    assert "swim" in h.activities


def test_token_helpers_and_stems() -> None:
    assert token_ints(("day=17", "12", "d:3", "run")) == [17, 12, 3]
    assert is_late(("Message", "LATE_night")) and is_late(("hour=2",)) and not is_late(("hour=12",))
    assert stem("dinner_sam", ("Sam", "Ann")) == "dinner_<friend>" and stem("run", ("Sam",)) == "run"


def test_rates_count_skips_and_cancels_and_the_decayed_rate_forgets() -> None:
    h = PersonHistory(META)
    h.add([Event(d, "did", "run", "x", ("did", "run")) for d in (0, 7, 14)])
    assert h.rate("run", 20, 0) == (3, 3) and h.rate("run", 20, 1) == (0, 0)
    h.add([Event(d, "skipped", "run", "x", ("skipped", "run")) for d in (21, 28)])
    assert h.rate("run", 28, 0) == (3, 5) and h.rate("run", 28) == (3, 5)
    assert h.decayed_rate("run", 28, 0.6) < 3 / 5 < h.decayed_rate("run", 14, 0.6)
    assert h.streak("run", 28) == 2 and h.streak("run", 14) == 0
    h.add([Event(35, "cancel", "run", "x", ("cancel", "run"), value=35.0)])
    assert h.rate("run", 35) == (3, 6) and h.counts("run", 35, 15) == (0, 3)


def test_features_see_nothing_after_the_asked_day() -> None:
    life = FakeLife(3, n_days=20)
    spec = FeatureSpec(vocab=("class", "gym", "run"), keys=("run|m7:weather", "ans:work/busy:a0"))
    early, full = PersonHistory(life.meta), PersonHistory(life.meta)
    for d in range(20):
        full.add(life.events(d))
        if d <= 9:
            early.add(life.events(d))
            for h in (early, full):
                h.note_queries(life.queries(d))
    early.add([life.answer(life.questions(9)[0].question_id)])
    full.add([life.answer(life.questions(9)[0].question_id)])
    for a in ("run", "gym", "class"):
        np.testing.assert_array_equal(features(spec, early, a, 9, 11, "habit"),
                                      features(spec, full, a, 9, 11, "habit"))
        assert raw_features(spec, early, a, 9, 11, "habit", "hidden").keys == \
            raw_features(spec, full, a, 9, 11, "habit", "hidden").keys


def test_feature_names_match_the_vector_and_mark_the_personal_weights() -> None:
    spec = FeatureSpec(vocab=("gym", "run"), keys=("ans:work/busy:a0",))
    hist = PersonHistory(META)
    hist.add([Event(0, "answer", "work", "x", ("answer", "work", "busy"))])
    x = features(spec, hist, "run", 0, 1, "habit")
    assert len(x) == spec.size == len(spec.names) and x[spec.names.index("ans:work/busy:a0")] == 1.0
    personal = {n for n, p in zip(spec.names, spec.personal) if p}
    assert {"counter_logit", "recent_logit", "streak", "act=run*late7", "act=<other>*mood3"} <= personal
    assert not {"late3", "bias", "act=run", "ans:work/busy:a0"} & personal


def test_a_count_query_has_no_regime_features_or_keys() -> None:
    spec = FeatureSpec(vocab=("gym", "run"))
    hist = PersonHistory(META)
    hist.add([Event(0, "message", "work", "x", ("message", "late")),
              Event(0, "mention", "knee", "x", ("mention", "knee", "sore"))])
    count = raw_features(spec, hist, "run", 0, 1, "habit", "count")
    hidden = raw_features(spec, hist, "run", 0, 1, "habit", "hidden")
    late = spec.dense_names.index("late1")
    assert count.dense[late] == 0.0 and hidden.dense[late] == 1.0
    assert count.keys == {} and hidden.keys["run|m2:knee"] == 1.0


def test_answer_timing_reads_counts_as_days_until_and_days_left() -> None:
    assert _answer_timing(("answer", "plans", "exam", "in=3"), 5, 7) == "u1"
    assert _answer_timing(("answer", "plans", "exam", "in=1"), 5, 7) == "u-1"  # already past
    assert _answer_timing(("answer", "plans", "injury", "left=4"), 5, 7) == "r2"
    assert _answer_timing(("answer", "plans", "injury", "left=4"), 5, 9) == "r0"
    assert _answer_timing(("answer", "work", "busy"), 5, 6) == "a0"
    assert _answer_timing(("answer", "work", "busy"), 5, 13) == "a2"


def test_key_selection_is_robust_to_one_person_carrying_a_key() -> None:
    def row(keys: dict[str, float]) -> RawFeatures:
        return RawFeatures(np.zeros(1), keys, 0.5, True)

    rows, ys, groups = [], [], []
    for life in range(30):
        for i in range(20):
            on = i % 2 == 0
            keys = {"real": 1.0} if on else {}
            if life == 0:
                keys["marker"] = 1.0  # one person, always doing it
            rows.append(row(keys))
            ys.append(float(on) if life % 3 else float(i % 4 == 0))
            groups.append(life)
    kept = select_keys(rows, ys, groups, z_min=3.0, min_groups=5)
    assert kept == ("real",)


def test_the_cross_fitted_table_leaves_out_its_lives() -> None:
    a: Cells = {("x",): [1.0, 2.0], (): [1.0, 2.0]}
    b: Cells = {("x",): [3.0, 4.0], (): [3.0, 4.0]}
    table = _table([a, b], 10.0, minus=[a])
    assert table.cells[("x",)] == (3.0, 4.0) and table.cells[()] == (3.0, 4.0)


# -- the population prior and the counter --
def test_population_prior_keys_on_family_and_stem_and_backs_off(train: list[FakeLife]) -> None:
    table = PopulationPrior.fit_population([FakeLife(s, n_days=42) for s in range(20)])
    assert table.p("count", "habit", "class") != table.p("hidden", "habit", "run")
    unknown = table.p("hidden", "habit", "piano", 2)
    assert unknown == pytest.approx(table.p("hidden", "habit", "piano", 5)) == \
        pytest.approx(table.p("hidden", "habit", "run", depth=3))


def test_counter_learns_a_person_unlike_the_population() -> None:
    table = PopulationPrior.fit_population([FakeLife(s, n_days=42) for s in range(20)])
    life = FakeLife(5, n_days=49, rotate=3)  # every habit three weekdays later
    counter = Counter(table, weekday_strength=FAKE_WEEKDAY)
    rec = run_life(life, counter)
    prior = run_life(life, PopulationPrior(table))
    late_c = [q for q in rec.queries if q.asked_day >= 28]
    late_p = [q for q in prior.queries if q.asked_day >= 28]

    def loss(qs: list[QueryRecord]) -> float:
        return float(score.log_loss(np.array([q.p for q in qs]), np.array([q.outcome for q in qs])).mean())

    assert loss(late_c) < loss(late_p) - 0.05
    first, last = rec.queries[0].u, rec.queries[-1].u
    assert first is not None and last is not None and last < first


def test_counter_counts_cancels_against_a_plan_and_shrinks_activity_then_weekday() -> None:
    counter = Counter(None, strength=2.0, weekday_strength=4.0)
    counter.begin(META)  # day 0 is a Monday
    counter.observe(0, [Event(0, "plan", "run", "x", ("plan", "run"), value=1.0)])
    counter.observe(1, [Event(1, "did", "run", "x", ("did", "run"))])
    counter.observe(2, [Event(2, "did", "run", "x", ("did", "run"))])
    counter.observe(3, [Event(3, "cancel", "run", "x", ("cancel", "run"), value=3.0)])
    q = Query("q", 3, 10, "run", "Will they keep their plan?", ("query", "plan", "run"), "hidden")
    (pred,) = counter.predict([q])
    p_run = (2 + 2 * 0.5) / (3 + 2)  # 2 of 3, cancels in the denominator
    assert pred.p == pytest.approx((0 + 4 * p_run) / (1 + 4))  # Thursdays: one cancel


# -- the logistic rivals --
def test_online_logistic_beats_the_counter_and_the_prior(dev_runs: Runs) -> None:
    ll = {name: score.summarize_run(recs)["log_loss"] for name, recs in dev_runs.items()}
    assert ll["online_lr"] < ll["counter"] - 0.01
    assert ll["counter"] < ll["population"] - 0.03


def test_online_logistic_uncertainty_shrinks_with_data(dev_runs: Runs) -> None:
    # u is parameter uncertainty only; whether it tracks the real error is T2's question, not a given
    qs = [q for r in dev_runs["online_lr"] for q in r.queries]
    week0 = np.mean([q.u for q in qs if q.asked_day < 7])
    later = np.mean([q.u for q in qs if q.asked_day >= 42])
    assert all(q.u is not None and q.u > 0 for q in qs) and later < week0


def test_online_logistic_predict_has_no_side_effects(population: LogisticPopulation) -> None:
    life = FakeLife(7, n_days=10)
    a, b = OnlineLogistic(population), OnlineLogistic(population)
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


def test_a_busy_answer_lowers_the_run_after_the_population_saw_asks(population: LogisticPopulation) -> None:
    life = FakeLife(11, n_days=14)
    q = Query("q", 6, 7, "run", "Will they run?", ("query", "run"), "hidden")
    ps = {}
    for word in ("calm", "busy"):
        m = OnlineLogistic(population)
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


def test_surprise_raises_the_learning_rate_after_a_shift(population: LogisticPopulation) -> None:
    life = FakeLife(9, n_days=56, shift_day=28)
    online = OnlineLogistic(population)
    surprised = SurpriseLR(population, gain=5.0)
    r_on, r_su = run_life(life, online), run_life(life, surprised)
    after = [i for i, q in enumerate(r_on.queries) if q.asked_day >= 28]
    assert max(c for d, c in surprised.cpp_trace) > surprised.hazard
    assert np.mean([r_su.queries[i].u for i in after]) > np.mean([r_on.queries[i].u for i in after])
    assert surprised.name == "surprise_lr" and np.trace(surprised.cov) > np.trace(online.cov)


def test_logistic_rivals_run_the_same_in_a_process_pool(population: LogisticPopulation) -> None:
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

    def predict(self, queries: Sequence[Query]) -> Sequence[Prediction]:
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
