"""The cog world: simulated lives whose hidden structure only a real inference recovers.

Population-level tests run on the dev seeds (10000-10199) and take a few seconds each; the speed
test makes 1000 lives.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pytest

from anjo.sleep.mind.cog import render, world, world_filter
from anjo.sleep.mind.cog.types import EVENT_KINDS, QUERY_FAMILIES, Event, Query, Resolution

DEV = range(10000, 10200)
TRAIN = range(0, 100)
OUTCOME_KINDS = ("did", "skipped", "cancel")


@pytest.fixture(scope="module")
def lives() -> list[world.SimLife]:
    return [world.make_life(seed) for seed in DEV]


@pytest.fixture(scope="module")
def base_rates() -> dict[tuple[str, str, str], float]:
    """A population prior: the outcome rate per activity kind, query kind and family, on train lives."""
    hits: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for seed in TRAIN:
        life = world.make_life(seed)
        for _, q, r in _records([life]):
            hits[_key(q)].append(int(r.outcome))
    return {key: float(np.mean(v)) for key, v in hits.items()}


def _key(q: Query) -> tuple[str, str, str]:
    stem = "dinner" if q.subject.startswith(render.DINNER_PREFIX) else q.subject
    return stem, q.tokens[1], q.family


def _records(lives: list[world.SimLife]) -> Iterator[tuple[world.SimLife, Query, Resolution]]:
    for life in lives:
        for day in range(life.meta.n_days):
            for q in life.queries(day):
                yield life, q, life.resolve(q.qid)


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _table(lives: list[world.SimLife]) -> dict[str, np.ndarray]:
    rows = list(_records(lives))
    return {
        "oracle": np.array([r.oracle_p for _, _, r in rows]),
        "state": np.array([r.state_p for _, _, r in rows]),
        "y": np.array([float(r.outcome) for _, _, r in rows]),
        "hidden": np.array([q.family == "hidden" for _, q, _ in rows]),
        "ahead": np.array([q.target_day - q.asked_day for _, q, _ in rows]),
    }


# ── the contract ──────────────────────────────────────────────────────────────────────────────
def test_a_life_meets_the_contract() -> None:
    life = world.make_life(3)
    meta = life.meta
    assert meta.n_days == 84 and 0 <= meta.start_weekday <= 6
    assert len(meta.activities) == 6 and 3 <= len(meta.people) <= 5 and len(meta.topics) == 3
    assert world.POPULATIONS == ("base", "ood")
    qids: set[str] = set()
    for day in range(meta.n_days):
        for e in life.events(day):
            assert isinstance(e, Event) and e.day == day and e.kind in EVENT_KINDS
            assert e.kind != "answer" and e.tokens[0] == e.kind and e.text and e.subject
        for q in life.queries(day):
            assert q.asked_day == day < q.target_day <= meta.n_days - 1 and q.family in QUERY_FAMILIES
            assert q.subject in meta.activities and q.text.endswith("?") and q.qid not in qids
            qids.add(q.qid)
        for question in life.questions(day):
            assert question.day == day and question.options and question.text.endswith("?")
    assert life.queries(meta.n_days - 1) == ()
    with pytest.raises(ValueError):
        life.events(meta.n_days)
    with pytest.raises(KeyError):
        life.resolve("no such query")
    with pytest.raises(KeyError):
        life.answer("no such question")
    with pytest.raises(ValueError):
        world.make_life(1, population="mars")
    short = world.make_life(5, n_days=21)
    assert short.meta.n_days == 21 and short.queries(19)


def test_the_same_seed_makes_the_same_life() -> None:
    a, b, other = world.make_life(7), world.make_life(7), world.make_life(8)
    for day in range(a.meta.n_days):
        assert a.events(day) == b.events(day)
        assert a.queries(day) == b.queries(day)
        assert a.questions(day) == b.questions(day)
        for question in a.questions(day):
            assert a.answer(question.question_id) == b.answer(question.question_id)
        for q in a.queries(day):
            assert a.resolve(q.qid) == b.resolve(q.qid)
    assert a.truth() == b.truth() and a.meta == b.meta
    assert any(a.events(d) != other.events(d) for d in range(a.meta.n_days))
    assert world.make_life(7, population="ood").events(0) != a.events(0)


def test_every_day_has_five_to_fifteen_events(lives: list[world.SimLife]) -> None:
    counts = [len(life.events(d)) for life in lives for d in range(life.meta.n_days)]
    assert min(counts) >= 5 and max(counts) <= 15


def test_text_and_tokens_say_the_same(lives: list[world.SimLife]) -> None:
    for life in lives[:20]:
        for day in range(life.meta.n_days):
            for e in life.events(day):
                assert e.text in {render.event_text(e.tokens, c) for c in range(render.BANK_SPAN)}
                if e.kind in OUTCOME_KINDS:
                    assert e.tokens == (e.kind, e.subject, f"dow={(life.meta.start_weekday + day) % 7}")
            for q in life.queries(day):
                assert q.text in {render.query_text(q.tokens, c) for c in range(render.BANK_SPAN)}


# ── no leakage ────────────────────────────────────────────────────────────────────────────────
def test_no_query_leaks_its_outcome(lives: list[world.SimLife]) -> None:
    for life in lives[:40]:
        by_day = [life.events(d) for d in range(life.meta.n_days)]
        for day, events in enumerate(by_day):
            for e in events:
                if e.kind == "plan":
                    assert e.value is not None and e.value > day
                    assert e.tokens == ("plan", e.subject, f"in={int(e.value) - day}",
                                        f"dow={(life.meta.start_weekday + int(e.value)) % 7}")
                elif e.kind != "mood":
                    assert e.value is None
        for q in (q for d in range(life.meta.n_days) for q in life.queries(d)):
            r = life.resolve(q.qid)
            before = [e for d in range(q.asked_day + 1) for e in by_day[d]]
            assert not any(e.kind in OUTCOME_KINDS and e.day >= q.target_day for e in before)
            assert not any(e.value == q.target_day and e.kind != "plan" for e in before if e.kind != "mood")
            on_target = [e for e in by_day[q.target_day] if e.subject == q.subject and e.kind in OUTCOME_KINDS]
            assert len(on_target) == 1 and (on_target[0].kind == "did") == r.outcome
            if q.tokens[1] == "kept":
                assert on_target[0].kind in ("did", "cancel")
                assert any(e.kind == "plan" and e.subject == q.subject and e.value == q.target_day
                           for e in before)
            else:
                assert on_target[0].kind in ("did", "skipped")


def test_flipping_an_outcome_changes_nothing_before_it(lives: list[world.SimLife]) -> None:
    rng = np.random.default_rng(0)
    for life in lives[:8]:
        seed = int(life.meta.person_id.split("-")[1])
        occurred = np.argwhere(life._occ[10:]) + [10, 0]
        for day, a in occurred[rng.choice(len(occurred), 3, replace=False)]:
            activity = life.meta.activities[a]
            flipped = world.SimLife(seed, interventions=frozenset({(activity, int(day))}))
            for d in range(day):
                assert flipped.events(d) == life.events(d) and flipped.queries(d) == life.queries(d)
                for q in life.queries(d):
                    old, new = life.resolve(q.qid), flipped.resolve(q.qid)
                    assert (old.oracle_p, old.state_p) == (new.oracle_p, new.state_p)
                    if q.target_day != day or q.subject != activity:
                        continue
                    assert new.outcome != old.outcome
            was = [e.kind for e in life.events(int(day)) if e.subject == activity and e.kind in OUTCOME_KINDS]
            now = [e.kind for e in flipped.events(int(day)) if e.subject == activity and e.kind in OUTCOME_KINDS]
            assert len(was) == len(now) == 1 and (was[0] == "did") != (now[0] == "did")


# ── queries and questions ─────────────────────────────────────────────────────────────────────
def test_queries_per_day_and_family_shares(lives: list[world.SimLife]) -> None:
    per_day = [len(life.queries(d)) for life in lives for d in range(life.meta.n_days - 8)]
    assert max(per_day) <= 10 and np.mean([6 <= n <= 10 for n in per_day]) >= 0.95
    table = _table(lives)
    assert 0.3 <= table["hidden"].mean() <= 0.7
    assert np.mean(table["ahead"] <= 2) >= 0.4 and table["ahead"].max() == 7


def test_questions_have_options_and_true_answers(lives: list[world.SimLife]) -> None:
    for life in lives[:30]:
        truth = life.truth()
        for day in range(life.meta.n_days):
            questions = life.questions(day)
            assert 3 <= len(questions) <= 5
            assert {q.subject for q in questions} >= {"work", "plans"}
            mentioned = {e.subject: e.tokens[2] for e in life.events(day) if e.kind == "mention"}
            for question in questions:
                assert 0.05 <= question.cost <= 0.2
                assert question.options == render.answer_options(question.subject)
                answer = life.answer(question.question_id)
                assert answer.kind == "answer" and answer.day == day and answer.subject == question.subject
                assert answer.tokens in question.options
                assert answer.text in {render.event_text(answer.tokens, c) for c in range(render.BANK_SPAN)}
                if question.subject == "work":
                    assert answer.tokens[2] == truth.params["regime_path"][day]
                elif question.subject == "plans":
                    assert answer.tokens == truth.params["fact_path"][day]
                elif question.subject in mentioned:
                    assert answer.tokens[2] == mentioned[question.subject]


# ── references ────────────────────────────────────────────────────────────────────────────────
def test_the_oracle_is_calibrated(lives: list[world.SimLife]) -> None:
    table = _table(lives)
    for name in ("oracle", "state"):
        p, y = table[name], table["y"]
        bins = np.minimum((p * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            if mask.sum() >= 400:
                assert abs(p[mask].mean() - y[mask].mean()) < 0.025, (name, b, mask.sum())


def test_state_beats_oracle_beats_a_population_base_rate(
        lives: list[world.SimLife], base_rates: dict[tuple[str, str, str], float]) -> None:
    rows = list(_records(lives))
    table = _table(lives)
    prior = np.array([base_rates[_key(q)] for _, q, _ in rows])
    y, hidden = table["y"], table["hidden"]
    state, oracle = _log_loss(table["state"], y), _log_loss(table["oracle"], y)
    population = _log_loss(prior, y)
    assert state < oracle - 0.005 and oracle < population - 0.05
    assert _log_loss(table["state"][hidden], y[hidden]) < _log_loss(table["oracle"][hidden], y[hidden]) - 0.015


def _no_inference(life: world.SimLife) -> np.ndarray:
    """(A, n_ask, horizon): the true parameters and observed streaks, but the hidden state left at
    its day-0 prior: everything the oracle knows except what it infers."""
    person, n_ask = life._person, life.meta.n_days - 1
    streak = np.zeros(life._y.shape, dtype=int)
    current = np.zeros(life._y.shape[1], dtype=int)
    for d in range(n_ask + 1):
        streak[d] = current
        occurred = life._occ[d]
        current = np.where(occurred & life._y[d], 0, np.where(occurred, np.minimum(current + 1, 2), current))
    occ = world_filter.occurrence_probs(person, life._plan_made, n_ask, world.HORIZON)
    return world_filter.predictive(person, np.tile(person.init, (n_ask, 1)), np.arange(n_ask), occ,
                                   streak[1:], life._y[:n_ask].astype(float), world.HORIZON)


def test_inferring_the_hidden_state_is_worth_something(lives: list[world.SimLife]) -> None:
    """On hidden queries the oracle beats the same parameters without the filter: the edge a
    learned predictor can only get by inferring the hidden state."""
    oracle, blind, y = [], [], []
    for life in lives:
        guess = _no_inference(life)
        for _, q, r in _records([life]):
            if q.family == "hidden":
                a = life.meta.activities.index(q.subject)
                oracle.append(r.oracle_p)
                blind.append(guess[a, q.asked_day, q.target_day - q.asked_day - 1])
                y.append(float(r.outcome))
    gain = _log_loss(np.array(blind), np.array(y)) - _log_loss(np.array(oracle), np.array(y))
    assert gain > 0.03, gain


def test_count_queries_depend_on_nothing_hidden(lives: list[world.SimLife]) -> None:
    table = _table(lives)
    count = ~table["hidden"]
    assert np.allclose(table["oracle"][count], table["state"][count], atol=1e-12)
    assert np.mean(np.abs(table["oracle"] - table["state"])[table["hidden"]]) > 0.05


def _counter_predictions(life: world.SimLife, base_rates: dict[tuple[str, str, str], float],
                         strength: float = 6.0) -> list[tuple[Query, float]]:
    """A per-person counter: outcome rates per activity and weekday (plans: per activity)."""
    tally: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    out: list[tuple[Query, float]] = []
    for day in range(life.meta.n_days):
        for e in life.events(day):
            if e.kind not in OUTCOME_KINDS:
                continue
            weekday = -1 if _planned(life, e.subject) else (life.meta.start_weekday + day) % 7
            hit = tally[(e.subject, weekday)]
            hit[0] += int(e.kind == "did")
            hit[1] += 1
        for q in life.queries(day):
            weekday = -1 if q.tokens[1] == "kept" else (life.meta.start_weekday + q.target_day) % 7
            done, n = tally[(q.subject, weekday)]
            out.append((q, (done + strength * base_rates[_key(q)]) / (n + strength)))
    return out


def _planned(life: world.SimLife, activity: str) -> bool:
    return bool(life._person.planned[life.meta.activities.index(activity)])


def test_a_counter_is_near_the_oracle_only_on_count_queries(
        lives: list[world.SimLife], base_rates: dict[tuple[str, str, str], float]) -> None:
    gaps = {}
    for family in QUERY_FAMILIES:
        counter, oracle, y = [], [], []
        for life in lives:
            for q, p in _counter_predictions(life, base_rates):
                if q.family == family:
                    r = life.resolve(q.qid)
                    counter.append(p)
                    oracle.append(r.oracle_p)
                    y.append(float(r.outcome))
        gaps[family] = _log_loss(np.array(counter), np.array(y)) - _log_loss(np.array(oracle), np.array(y))
    assert gaps["count"] < 0.03 and gaps["hidden"] > 0.07 and gaps["hidden"] > 3 * gaps["count"], gaps


def test_the_oracle_matches_a_monte_carlo_of_the_world(lives: list[world.SimLife]) -> None:
    """Sampling the world forward from the filter (or the true state) gives the references."""
    rng = np.random.default_rng(1)
    checked = 0
    for life in lives[:60]:
        person = life._person
        if not (person.streak_w != 0).any() and not (person.trigger_src >= 0).any():
            continue
        far = [q for d in range(5, life.meta.n_days - 8) for q in life.queries(d)
               if q.target_day - q.asked_day >= 3 and q.family == "hidden"]
        for q in [far[i] for i in rng.choice(len(far), 2, replace=False)]:
            r = life.resolve(q.qid)
            for start, reference in ((life._alpha[q.asked_day], r.oracle_p),
                                     (np.eye(len(person.init))[life._z[q.asked_day]], r.state_p)):
                estimate = _monte_carlo(life, q, start, rng, n=20000)
                assert abs(estimate - reference) < 4 * math.sqrt(reference * (1 - reference) / 20000) + 0.003
            checked += 1
    assert checked >= 20


def _monte_carlo(life: world.SimLife, q: Query, start: np.ndarray, rng: np.random.Generator,
                 n: int) -> float:
    person, t, a = life._person, q.asked_day, life.meta.activities.index(q.subject)
    n_act = len(life.meta.activities)
    streak = np.zeros(n_act, dtype=int)
    for d in range(t + 1):  # replay the observed history into the streaks
        occurred = life._occ[d]
        streak = np.where(occurred & life._y[d], 0, np.where(occurred, np.minimum(streak + 1, 2), streak))
    streak = np.tile(streak, (n, 1))
    prev = np.tile(life._y[t].astype(float), (n, 1))
    z = _draw(rng, np.tile(start, (n, 1)))
    for d in range(t + 1, q.target_day + 1):
        z = _draw(rng, person.chain[z])
        occurs = np.zeros((n, n_act), dtype=bool)
        for b in range(n_act):
            occurs[:, b] = _occurs(life, b, d, t, rng, n)
        p = 1.0 / (1.0 + np.exp(-person.outcome_logits(d, z, streak, prev)))
        done = occurs & (rng.random((n, n_act)) < p)
        if d == q.target_day:
            return float(done[:, a].sum() / occurs[:, a].sum())
        streak = np.where(done, 0, np.where(occurs, np.minimum(streak + 1, 2), streak))
        prev = done.astype(float)
    raise AssertionError("unreachable")


def _draw(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    return (rng.random((len(probs), 1)) > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, probs.shape[1] - 1)


def _occurs(life: world.SimLife, b: int, d: int, t: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Whether activity ``b`` takes place on day ``d`` as seen on day ``t``: a plan made by ``t``
    is known; otherwise the plan process is resampled until it agrees with seeing no plan by ``t``."""
    person = life._person
    if not person.planned[b]:
        return np.full(n, bool(person.pattern[b, d]))
    if 0 <= life._plan_made[b, d] <= t:
        return np.ones(n, dtype=bool)
    if not person.pattern[b, d]:
        return np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    todo = np.ones(n, dtype=bool)
    lead_max = min(world.PLAN_LEAD_MAX, d)
    while todo.any():
        k = int(todo.sum())
        planned = rng.random(k) < person.plan_rate[b, d]
        made = d - (1 + rng.integers(0, lead_max, k))
        ok = ~(planned & (made <= t))
        idx = np.flatnonzero(todo)
        out[idx[ok]] = planned[ok]
        todo[idx[ok]] = False
    return out


# ── the hidden structure ──────────────────────────────────────────────────────────────────────
def test_distractor_mentions_are_independent_of_outcomes(lives: list[world.SimLife]) -> None:
    by_value: dict[tuple[str, str], list[int]] = defaultdict(list)
    counts, rates = [], []
    for life in lives:
        for day in range(life.meta.n_days - 1):
            outcomes = [int(e.kind == "did") for d in (day, day + 1) for e in life.events(d)
                        if e.kind in OUTCOME_KINDS]
            mentions = [e for e in life.events(day) if e.kind == "mention" and e.subject in life.meta.topics]
            for e in mentions:
                by_value[(render.topic_kind(e.subject), e.tokens[2])].extend(outcomes)
            if outcomes:
                counts.append(len(mentions))
                rates.append(np.mean(outcomes))
    overall = np.mean([o for v in by_value.values() for o in v])
    for key, outcomes in by_value.items():
        assert len(outcomes) > 2000 and abs(np.mean(outcomes) - overall) < 0.03, key
    assert abs(np.corrcoef(counts, rates)[0, 1]) < 0.03
    for kind, values in render.TOPIC_VALUES.items():
        n = np.array([len(by_value[(kind, v)]) for v in values], dtype=float)
        entropy = -np.sum(n / n.sum() * np.log(n / n.sum()))
        assert entropy > 0.97 * math.log(len(values))


def test_about_a_quarter_have_an_idiosyncratic_effect(lives: list[world.SimLife]) -> None:
    flags = []
    for life in lives:
        params = life.truth().params
        odd = set(params["idiosyncratic"])
        flags.append(bool(odd))
        for activity, spec in params["activities"].items():
            if spec["steady"]:
                continue
            for regime in ("busy", "crunch"):
                same = math.copysign(1, spec["regime_effects"][regime]) == math.copysign(
                    1, spec["population_mean"][regime])
                assert same != ((activity, "regime") in odd)
    assert 0.17 <= np.mean(flags) <= 0.33


def test_shifts_are_recorded_and_real(lives: list[world.SimLife]) -> None:
    n_shifts, faded = [], {"before": [], "after": []}
    for life in lives:
        truth = life.truth()
        shifts = truth.params["shifts"]
        assert truth.shift_days == tuple(s["day"] for s in shifts)
        assert all(0 < s["day"] < life.meta.n_days for s in shifts)
        n_shifts.append(len(shifts))
        for shift in shifts:
            assert shift["before"] != shift["after"]
            a, day = shift["activity"], shift["day"]
            seen = [(e.day, (life.meta.start_weekday + e.day) % 7, e.kind == "did")
                    for d in range(life.meta.n_days) for e in life.events(d)
                    if e.subject == a and e.kind in OUTCOME_KINDS]
            if shift["param"] == "weekdays":
                assert {w for d, w, _ in seen if d < day} <= set(shift["before"])
                assert {w for d, w, _ in seen if d >= day} <= set(shift["after"])
                assert {w for d, w, _ in seen if d >= day} & (set(shift["after"]) - set(shift["before"]))
            else:
                faded["before"].extend(done for d, _, done in seen if d < day)
                faded["after"].extend(done for d, _, done in seen if d >= day)
    assert set(n_shifts) == {0, 1, 2}
    assert np.mean(faded["after"]) < np.mean(faded["before"]) - 0.2


def test_hidden_facts_are_recorded_and_bite(lives: list[world.SimLife]) -> None:
    per_life, injured, healthy, hint_on, hint_off = [], [], [], [], []
    for life in lives:
        truth = life.truth()
        per_life.append(len(truth.hidden_facts))
        hurt_days = {d for f in truth.hidden_facts.values() if f["kind"] == "injury"
                     for d in range(f["start"], f["end"] + 1)}
        fact_days = {d for d, tokens in enumerate(truth.params["fact_path"]) if tokens[2] != "none"}
        sensitive = {a for a, spec in truth.params["activities"].items() if not spec["steady"]}
        for day in range(life.meta.n_days):
            events = life.events(day)
            for e in events:
                if e.kind in OUTCOME_KINDS and e.subject in {"run", "gym", "climbing"} & sensitive:
                    (injured if day in hurt_days else healthy).append(e.kind == "did")
            hinted = any(e.kind == "mention" and e.subject not in life.meta.topics for e in events)
            (hint_on if day in fact_days else hint_off).append(hinted)
    assert 1.0 <= np.mean(per_life) <= 4.0
    assert np.mean(injured) < 0.12 < np.mean(healthy)
    assert np.mean(hint_on) > 3 * np.mean(hint_off)


def test_the_ood_population_differs_from_base(lives: list[world.SimLife]) -> None:
    ood = [world.make_life(seed, population="ood") for seed in range(20000, 20100)]

    def run_days(group: list[world.SimLife]) -> float:
        hits = [(life.meta.start_weekday + e.day) % 7 in (1, 3, 5)
                for life in group for d in range(life.meta.n_days) for e in life.events(d)
                if e.subject == "run" and e.kind in OUTCOME_KINDS]
        return float(np.mean(hits))

    def odd_share(group: list[world.SimLife]) -> float:
        return float(np.mean([bool(life.truth().params["idiosyncratic"]) for life in group]))

    def gym_crunch(group: list[world.SimLife]) -> float:
        return float(np.mean([life.truth().params["activities"]["gym"]["population_mean"]["crunch"]
                              for life in group if "gym" in life.meta.activities]))

    assert run_days(lives) > 0.6 > 0.3 > run_days(ood)
    assert odd_share(ood) > odd_share(lives) + 0.15
    assert gym_crunch(lives) < 0 < gym_crunch(ood)
    assert ood[0].meta.person_id.startswith("ood-")


def test_a_thousand_lives_take_under_two_minutes() -> None:
    began = time.perf_counter()
    for seed in range(30000, 31000):
        world.make_life(seed)
    assert time.perf_counter() - began < 120.0
