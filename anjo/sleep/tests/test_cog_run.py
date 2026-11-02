"""The cognition sprint's run loop (run.py) and its CLI, on a small fake life.

``FakeLife`` here is also the fake world of the other ``test_cog_*`` files of the evaluation side:
three activities with weekday habits, a hidden calm/busy regime (a sticky two-state Markov chain)
that lowers two of them and shows only through late messages and low moods, next-day plans,
distractor topic mentions, two questions a day (work, a topic), an exact forward-filter oracle, and
an optional regime shift that moves the run to the weekend for good, and two ways to be unlike the
population: every habit rotated by some weekdays, or busy spells raising the gym instead of lowering it.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from anjo.sleep.mind.cog.run import (
    AskRecord,
    LifeRecord,
    QueryRecord,
    load_records,
    run_life,
    run_many,
    save_records,
)
from anjo.sleep.mind.cog.types import (
    Event,
    LifeMeta,
    Prediction,
    Query,
    Question,
    Resolution,
    Truth,
)

WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
ACTIVITIES = ("run", "gym", "class")
PEOPLE = ("mom", "sam")
TOPICS = ("weather", "football", "movies")
BUSY_EFFECT = {"run": -2.5, "gym": -1.5, "class": 0.0}
SCHEDULE = {"run": frozenset({0, 2, 4}), "gym": frozenset({1, 3}), "class": frozenset({0, 1, 2, 3, 4})}
SHIFTED_RUN = frozenset({5, 6})
BASE_ON, BASE_OFF, PLAN_BONUS = 1.5, -2.5, 1.5
STAY = 0.9
P_LATE = (0.1, 0.75)
P_LOW = (0.2, 0.7)
P_PLAN = 0.3
P_PLAN_CELL = P_PLAN / len(ACTIVITIES)
QUESTION_COST = 0.05
TRANSITION = np.array([[STAY, 1 - STAY], [1 - STAY, STAY]])


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FakeLife:
    """One fake person; every random draw is made at construction from ``seed``."""

    def __init__(self, seed: int, n_days: int = 28, shift_day: int | None = None, rotate: int = 0,
                 idiosyncratic: bool = False) -> None:
        rng = np.random.default_rng(seed)
        self.seed = seed
        self._rotate = rotate  # moves every habit by this many weekdays
        # against the population norm: a busy spell sends this person to the gym more, not less
        self._busy = {**BUSY_EFFECT, "gym": 2.0} if idiosyncratic else BUSY_EFFECT
        self.meta = LifeMeta(person_id=f"p{seed}", n_days=n_days, start_weekday=seed % 7,
                             activities=ACTIVITIES, people=PEOPLE, topics=TOPICS)
        self._shift_day = shift_day
        states = [int(rng.random() < 0.5)]
        for _ in range(1, n_days):
            states.append(states[-1] if rng.random() < STAY else 1 - states[-1])
        self.states = tuple(states)
        self._plans: dict[tuple[str, int], int] = {}
        for day in range(n_days - 1):
            if rng.random() < P_PLAN:
                self._plans[(ACTIVITIES[int(rng.integers(len(ACTIVITIES)))], day + 1)] = day
        self._outcomes = {(a, d): bool(rng.random() < self._p(a, d, states[d], (a, d) in self._plans))
                          for d in range(n_days) for a in ACTIVITIES}
        self._late = tuple(bool(rng.random() < P_LATE[s]) for s in states)
        self._low = tuple(bool(rng.random() < P_LOW[s]) for s in states)
        self._topic = tuple(TOPICS[int(rng.integers(len(TOPICS)))] for _ in range(n_days))
        self._likes = tuple(bool(rng.random() < 0.5) for _ in range(n_days))
        self._events = tuple(self._make_events(d) for d in range(n_days))
        self._filters = self._forward()
        self._queries = tuple(self._make_queries(d) for d in range(n_days))
        self._questions = tuple(self._make_questions(d) for d in range(n_days))
        self._by_qid = {q.qid: q for day in self._queries for q in day}
        self._by_question = {q.question_id: q for day in self._questions for q in day}

    # -- the generative model --
    def weekday(self, day: int) -> int:
        return (self.meta.start_weekday + day) % 7

    def scheduled(self, activity: str, day: int) -> bool:
        shifted = activity == "run" and self._shift_day is not None and day >= self._shift_day
        return (self.weekday(day) - self._rotate) % 7 in (SHIFTED_RUN if shifted else SCHEDULE[activity])

    def _p(self, activity: str, day: int, state: int, planned: bool) -> float:
        logit = (BASE_ON if self.scheduled(activity, day) else BASE_OFF)
        return _sigmoid(logit + self._busy[activity] * state + (PLAN_BONUS if planned else 0.0))

    def _make_events(self, day: int) -> tuple[Event, ...]:
        out: list[Event] = []
        for a in ACTIVITIES:
            if self._outcomes[(a, day)]:
                out.append(Event(day, "did", a, f"I did my {a} today.", ("did", a)))
            elif self.scheduled(a, day):
                out.append(Event(day, "skipped", a, f"I skipped {a} today.", ("skipped", a)))
        for (a, target), made in sorted(self._plans.items()):
            if made == day:
                out.append(Event(day, "plan", a, f"I plan to do {a} tomorrow.",
                                 ("plan", a, f"day={target}"), value=float(target)))
        low = self._low[day]
        out.append(Event(day, "mood", "self", "Feeling low today." if low else "Feeling good today.",
                         ("mood", "low" if low else "good"), value=2.0 if low else 4.0))
        if self._late[day]:
            out.append(Event(day, "message", "work", "Still on work email at 11pm.", ("message", "late")))
        else:
            out.append(Event(day, "message", "sam", "Texted Sam about lunch.", ("message", "day")))
        topic = self._topic[day]
        out.append(Event(day, "mention", topic, f"Saw something about the {topic}.", ("mention", topic)))
        return tuple(out)

    def _likelihood(self, day: int) -> np.ndarray:
        lik = np.ones(2)
        for s in (0, 1):
            for a in ACTIVITIES:
                p = self._p(a, day, s, (a, day) in self._plans)
                lik[s] *= p if self._outcomes[(a, day)] else 1 - p
            lik[s] *= P_LATE[s] if self._late[day] else 1 - P_LATE[s]
            lik[s] *= P_LOW[s] if self._low[day] else 1 - P_LOW[s]
        return lik

    def _forward(self) -> tuple[np.ndarray, ...]:
        belief, out = np.array([0.5, 0.5]), []
        for day in range(self.meta.n_days):
            prior = belief if day == 0 else belief @ TRANSITION
            post = prior * self._likelihood(day)
            belief = post / post.sum()
            out.append(belief)
        return tuple(out)

    def _make_queries(self, day: int) -> tuple[Query, ...]:
        out = []
        for h in (1, 2):
            t = day + h
            if t >= self.meta.n_days:
                continue
            for a in ACTIVITIES:
                planned = self._plans.get((a, t), day + 1) <= day
                text = (f"Will they keep their plan to do {a} on {WEEKDAY_NAMES[self.weekday(t)]}?"
                        if planned else f"Will they do {a} on {WEEKDAY_NAMES[self.weekday(t)]}?")
                tokens = ("query", a, f"day={t}", WEEKDAY_NAMES[self.weekday(t)]) + (("plan",) if planned else ())
                out.append(Query(f"s{self.seed}d{day}h{h}{a}", day, t, a, text, tokens,
                                 "count" if BUSY_EFFECT[a] == 0 else "hidden"))
        return tuple(out)

    def _make_questions(self, day: int) -> tuple[Question, ...]:
        topic = self._topic[day]
        return (
            Question(f"s{self.seed}d{day}work", day, "work", "How is work this week?", ("ask", "work"),
                     QUESTION_COST, options=(("answer", "work", "calm"), ("answer", "work", "busy"))),
            Question(f"s{self.seed}d{day}topic", day, topic, f"What do you make of the {topic}?",
                     ("ask", topic), QUESTION_COST,
                     options=(("answer", topic, "like"), ("answer", topic, "dislike"))),
        )

    # -- the Life protocol --
    def events(self, day: int) -> tuple[Event, ...]:
        return self._events[day]

    def queries(self, day: int) -> tuple[Query, ...]:
        return self._queries[day]

    def questions(self, day: int) -> tuple[Question, ...]:
        return self._questions[day]

    def answer(self, question_id: str) -> Event:
        q = self._by_question[question_id]
        if q.subject == "work":
            busy = self.states[q.day]
            return Event(q.day, "answer", "work", "Work is crazy busy." if busy else "Work is calm.",
                         q.options[busy])
        like = self._likes[q.day]
        return Event(q.day, "answer", q.subject, f"I {'like' if like else 'dislike'} the {q.subject}.",
                     q.options[0 if like else 1])

    def resolve(self, qid: str) -> Resolution:
        q = self._by_qid[qid]
        prior = self._filters[q.asked_day] @ np.linalg.matrix_power(TRANSITION, q.target_day - q.asked_day)
        known = (q.subject, q.target_day) in self._plans and q.target_day - q.asked_day == 1

        def p_of(s: int) -> float:
            if known:
                return self._p(q.subject, q.target_day, s, True)
            return (P_PLAN_CELL * self._p(q.subject, q.target_day, s, True)
                    + (1 - P_PLAN_CELL) * self._p(q.subject, q.target_day, s, False))

        oracle = float(sum(prior[s] * p_of(s) for s in (0, 1)))
        planned = (q.subject, q.target_day) in self._plans
        state_p = self._p(q.subject, q.target_day, self.states[q.target_day], planned)
        return Resolution(qid, self._outcomes[(q.subject, q.target_day)], oracle, state_p)

    def truth(self) -> Truth:
        if self._shift_day is None:
            return Truth(shift_days=())
        return Truth(shift_days=(self._shift_day,), params={"shift_subjects": ("run",)})


def make_fake_life(seed: int, population: str = "base", n_days: int = 28) -> FakeLife:
    """The CLI's ``make_life`` stand-in: every third seed has a shift halfway through."""
    return FakeLife(seed, n_days=n_days, shift_day=n_days // 2 if seed % 3 == 0 else None)


# -- small predictors and inquirers for the run loop --
class ConstPredictor:
    def __init__(self, p: float = 0.5, name: str = "const") -> None:
        self.p, self.name = p, name

    def begin(self, meta: LifeMeta) -> None: ...
    def observe(self, day: int, events: Sequence[Event]) -> None: ...
    def end_day(self, day: int) -> None: ...

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        return [Prediction(q.qid, self.p) for q in queries]


class FreqPredictor:
    """p = (times the subject was done + 1) / (days seen + 2): its answers depend on what it saw.

    Today counts as seen once observed (``days`` counts nights), hence the + 3."""
    name = "freq"

    def begin(self, meta: LifeMeta) -> None:
        self.done: dict[str, int] = {}
        self.days = 0
        self.answers = 0

    def observe(self, day: int, events: Sequence[Event]) -> None:
        for e in events:
            if e.kind == "did":
                self.done[e.subject] = self.done.get(e.subject, 0) + 1
            if e.kind == "answer":
                self.answers += 1

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        return [Prediction(q.qid, (self.done.get(q.subject, 0) + 1 + 0.1 * self.answers)
                           / (self.days + 3 + 0.2 * self.answers),
                           u=1.0 / (self.days + 1)) for q in queries]

    def end_day(self, day: int) -> None:
        self.days += 1


class RecordingPredictor(FreqPredictor):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def begin(self, meta: LifeMeta) -> None:
        super().begin(meta)
        self.calls.append(("begin", meta.person_id))

    def observe(self, day: int, events: Sequence[Event]) -> None:
        super().observe(day, events)
        self.calls.append(("observe", day, tuple(e.kind for e in events)))

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        self.calls.append(("predict", tuple(q.asked_day for q in queries)))
        return super().predict(queries)

    def end_day(self, day: int) -> None:
        super().end_day(day)
        self.calls.append(("end_day", day))


class AlwaysWork:
    """Asks the work question every time it may, and records what it was shown."""
    name = "always_work"

    def __init__(self, predictor: object | None = None) -> None:
        self.seen: list[tuple[int, tuple[str, ...], tuple[int, ...]]] = []

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        self.seen.append((day, tuple(q.question_id for q in questions), tuple(q.asked_day for q in upcoming)))
        work = [q for q in questions if q.subject == "work"]
        return work[0].question_id if work else (questions[0].question_id if questions else None)


class Observant:
    """An inquirer with the optional begin/observe hooks: it is told the life like a predictor."""
    name = "observant"

    def __init__(self, predictor: object | None = None) -> None:
        self.days: list[int] = []
        self.kinds: list[str] = []
        self.meta: LifeMeta | None = None

    def begin(self, meta: LifeMeta) -> None:
        self.meta = meta

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.days.append(day)
        self.kinds.extend(e.kind for e in events)

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        return questions[0].question_id if day == 2 else None


def make_always_work(predictor: object) -> AlwaysWork:
    return AlwaysWork(predictor)


# -- the fake itself --
def test_fake_life_is_deterministic_and_its_oracle_beats_a_coin() -> None:
    a, b = FakeLife(5), FakeLife(5)
    assert a.events(3) == b.events(3) and a.queries(3) == b.queries(3)
    losses, coin = [], []
    for d in range(a.meta.n_days):
        for q in a.queries(d):
            r = a.resolve(q.qid)
            assert 0 < r.oracle_p < 1 and q.target_day > q.asked_day
            p = r.oracle_p if r.outcome else 1 - r.oracle_p
            losses.append(-math.log(p))
            coin.append(math.log(2))
    assert np.mean(losses) < np.mean(coin) - 0.1


# -- run_life --
def test_run_life_calls_observe_ask_predict_night_in_order() -> None:
    life, pred = FakeLife(1, n_days=5), RecordingPredictor()
    run_life(life, pred, AlwaysWork())
    assert pred.calls[0] == ("begin", "p1")
    day0 = pred.calls[1:5]
    assert day0[0][0] == "observe" and day0[0][1] == 0 and "answer" not in day0[0][2]
    assert day0[1] == ("observe", 0, ("answer",))
    assert day0[2][0] == "predict" and set(day0[2][1]) == {0}
    assert day0[3] == ("end_day", 0)
    assert pred.calls[-1] == ("end_day", 4)


def test_the_inquirer_sees_only_todays_queries_and_questions() -> None:
    life, inq = FakeLife(2, n_days=6), AlwaysWork()
    run_life(life, FreqPredictor(), inq)
    for day, questions, asked_days in inq.seen:
        assert set(asked_days) <= {day}
        assert questions == tuple(q.question_id for q in life.questions(day))


def test_asks_are_recorded_with_cost_answer_and_distractor_flag() -> None:
    life = FakeLife(3, n_days=6)
    rec = run_life(life, FreqPredictor(), AlwaysWork(), max_asks_per_day=2)
    assert len(rec.asks) == 12  # two asks a day: the work question, then the topic one
    first, second = rec.asks[0], rec.asks[1]
    assert first.day == 0 and first.subject == "work" and not first.is_distractor
    assert first.cost == pytest.approx(QUESTION_COST)
    assert first.answer_tokens in (("answer", "work", "calm"), ("answer", "work", "busy"))
    assert second.is_distractor and second.subject in TOPICS
    assert rec.inquirer == "always_work" and rec.run == "freq+always_work"


def test_one_ask_a_day_by_default_and_silence_is_allowed() -> None:
    rec = run_life(FakeLife(3, n_days=6), FreqPredictor(), AlwaysWork())
    assert [a.day for a in rec.asks] == list(range(6))
    silent = run_life(FakeLife(3, n_days=6), FreqPredictor(), Observant())
    assert [a.day for a in silent.asks] == [2]


def test_inquirer_hooks_get_the_meta_events_and_answers() -> None:
    inq = Observant()
    run_life(FakeLife(4, n_days=4), FreqPredictor(), inq)
    assert inq.meta is not None and inq.meta.person_id == "p4"
    assert sorted(set(inq.days)) == [0, 1, 2, 3] and inq.kinds.count("answer") == 1


def test_query_records_carry_prediction_and_resolution() -> None:
    life = FakeLife(6, n_days=5)
    rec = run_life(life, FreqPredictor(), seed=6)
    expected = [q for d in range(5) for q in life.queries(d)]
    assert [r.qid for r in rec.queries] == [q.qid for q in expected]
    for r, q in zip(rec.queries, expected):
        res = life.resolve(q.qid)
        assert (r.family, r.subject, r.asked_day, r.target_day) == (q.family, q.subject, q.asked_day, q.target_day)
        assert (r.outcome, r.oracle_p, r.state_p) == (res.outcome, res.oracle_p, res.state_p)
        assert 0 < r.p < 1 and r.u is not None
    assert rec.seed == 6 and rec.person_id == "p6" and rec.topics == TOPICS and rec.asks == ()


def test_query_days_restricts_which_days_are_predicted() -> None:
    pred = RecordingPredictor()
    rec = run_life(FakeLife(7, n_days=8), pred, query_days={2, 5})
    assert {r.asked_day for r in rec.queries} == {2, 5}
    assert sum(1 for c in pred.calls if c[0] == "predict") == 2
    assert sum(1 for c in pred.calls if c[0] == "end_day") == 8


def test_shift_truth_is_recorded() -> None:
    rec = run_life(FakeLife(9, n_days=10, shift_day=5), FreqPredictor())
    assert rec.shift_days == (5,) and rec.shift_subjects == ("run",)
    plain = run_life(FakeLife(9, n_days=10), FreqPredictor())
    assert plain.shift_days == () and plain.shift_subjects is None


class _Missing(FreqPredictor):
    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        return super().predict(queries)[1:]


class _Bad(FreqPredictor):
    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        return [Prediction(q.qid, 1.0) for q in queries]


class _Unknown:
    name = "unknown"

    def choose(self, day: int, questions: Sequence[Question], upcoming: Sequence[Query]) -> str | None:
        return "no-such-question"


def test_a_predictor_must_answer_every_query_with_a_probability() -> None:
    with pytest.raises(ValueError, match="no prediction"):
        run_life(FakeLife(1, n_days=3), _Missing())
    with pytest.raises(ValueError, match="probability"):
        run_life(FakeLife(1, n_days=3), _Bad())


def test_an_unknown_question_id_fails_loudly() -> None:
    with pytest.raises(ValueError, match="not among"):
        run_life(FakeLife(1, n_days=3), FreqPredictor(), _Unknown())


def test_usage_is_copied_from_a_predictor_that_counts_it() -> None:
    class Counting(FreqPredictor):
        def begin(self, meta: LifeMeta) -> None:
            super().begin(meta)
            self.usage = {"prompt_chars": 0.0}

        def predict(self, queries: Sequence[Query]) -> list[Prediction]:
            self.usage = {"prompt_chars": self.usage["prompt_chars"] + 10}
            return super().predict(queries)

    rec = run_life(FakeLife(1, n_days=4), Counting())
    assert rec.usage == {"prompt_chars": 30.0}  # the last day has no queries, so no predict call


# -- run_many and records on disk --
def test_run_many_is_the_same_for_any_worker_count() -> None:
    seeds = [11, 12, 13, 14]
    one = run_many(seeds, FreqPredictor, inquirer_factory=make_always_work, make_life=make_fake_life)
    two = run_many(seeds, FreqPredictor, inquirer_factory=make_always_work, make_life=make_fake_life,
                   workers=2)
    assert one == two
    assert [r.seed for r in one] == seeds and all(len(r.asks) == 28 for r in one)


def test_run_many_takes_lives_too() -> None:
    lives = [FakeLife(21, n_days=4), FakeLife(22, n_days=4)]
    recs = run_many(lives, FreqPredictor)
    assert [r.person_id for r in recs] == ["p21", "p22"] and recs[0].seed is None


def test_run_many_needs_a_life_maker_for_seeds() -> None:
    with pytest.raises(ValueError, match="make_life"):
        run_many([1], FreqPredictor)


def test_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    recs = run_many([15, 16], FreqPredictor, inquirer_factory=make_always_work, make_life=make_fake_life)
    path = tmp_path / "deep" / "recs.jsonl"
    save_records(path, recs)
    assert len(path.read_text().splitlines()) == 2
    back = load_records(path)
    assert back == recs
    assert isinstance(back[0], LifeRecord) and isinstance(back[0].queries[0], QueryRecord)
    assert isinstance(back[0].asks[0], AskRecord) and isinstance(back[0].topics, tuple)


# -- the CLI --
def test_cli_eval_writes_records_and_a_report(tmp_path: Path) -> None:
    from anjo.sleep.mind.cog import cli

    out = tmp_path / "out"
    code = cli.main(["eval", "--predictor", "counter", "--seeds", "30-33", "--train-seeds", "0-5",
                     "--n-days", "21", "--out", str(out), "--inquirer", "random", "--set", "p_ask=0.5"],
                    make_life=make_fake_life)
    assert code == 0
    assert (out / "counter.jsonl").exists() and (out / "counter+random.jsonl").exists()
    report = json.loads((out / "report.json").read_text())
    assert set(report["runs"]) == {"counter", "counter+random"}
    assert report["runs"]["counter+random"]["t3"]["n_lives"] == 4
    again = tmp_path / "again.json"
    assert cli.main(["report", "--records", str(out / "counter.jsonl"), str(out / "counter+random.jsonl"),
                     "--out", str(again)]) == 0
    assert json.loads(again.read_text())["runs"].keys() == report["runs"].keys()


def test_cli_lists_its_registries() -> None:
    from anjo.sleep.mind.cog import cli

    assert {"population", "counter", "online_lr", "surprise_lr", "gemini"} <= set(cli.PREDICTORS)
    assert {"never", "random", "max_uncertainty", "gemini"} <= set(cli.INQUIRERS)


def test_cli_seed_ranges() -> None:
    from anjo.sleep.mind.cog import cli

    assert cli.parse_seeds("3-5,9") == (3, 4, 5, 9)
    with pytest.raises(ValueError):
        cli.parse_seeds("5-3")


def test_cli_rejects_an_unknown_predictor(tmp_path: Path) -> None:
    from anjo.sleep.mind.cog import cli

    with pytest.raises(SystemExit):
        cli.main(["eval", "--predictor", "nope", "--seeds", "1", "--out", str(tmp_path)],
                 make_life=make_fake_life)
