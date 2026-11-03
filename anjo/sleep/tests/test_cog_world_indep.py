# mypy: disallow-untyped-decorators=False
"""The independent world of ordinary lives: interface, determinism, leakage, calibration, speed."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from anjo.sleep.mind.cog.types import EVENT_KINDS, QUERY_FAMILIES, Event, Life, Query
from anjo.sleep.mind.cog.world_indep import make_life

N_CAL = 200  # lives for the calibration checks
OUTCOME_KINDS = ("did", "skipped")


@pytest.fixture(scope="module")
def lives() -> list[Life]:
    return [make_life(seed) for seed in range(N_CAL)]


@pytest.fixture(scope="module")
def resolved(lives: list[Life]) -> dict[str, np.ndarray]:
    oracle, state, outcome, hidden, who = [], [], [], [], []
    for i, life in enumerate(lives):
        for day in range(life.meta.n_days):
            for q in life.queries(day):
                r = life.resolve(q.qid)
                oracle.append(r.oracle_p)
                state.append(r.state_p)
                outcome.append(float(r.outcome))
                hidden.append(q.family == "hidden")
                who.append(i)
    return {"oracle": np.array(oracle), "state": np.array(state), "y": np.array(outcome),
            "hidden": np.array(hidden), "life": np.array(who)}


def _all_queries(life: Life) -> list[Query]:
    return [q for d in range(life.meta.n_days) for q in life.queries(d)]


def _clustered_se(residual: np.ndarray, life: np.ndarray) -> float:
    """Standard error of the residuals' mean when queries about one life are correlated."""
    sums = np.bincount(life, weights=residual)
    return float(math.sqrt(np.sum(sums**2)) / len(residual))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---- interface ---------------------------------------------------------------------------


def test_meta_names_only() -> None:
    life = make_life(3)
    meta = life.meta
    assert meta.n_days == 84
    assert 0 <= meta.start_weekday <= 6
    assert 3 <= len(meta.activities) <= 7
    assert len(set(meta.activities)) == len(meta.activities)
    assert len(meta.people) >= 3 and len(meta.topics) >= 5
    assert all(isinstance(x, str) and x for x in (*meta.activities, *meta.people, *meta.topics))


def test_events_well_formed() -> None:
    for seed in range(5):
        life = make_life(seed)
        for day in range(life.meta.n_days):
            evs = life.events(day)
            assert isinstance(evs, tuple)
            for e in evs:
                assert isinstance(e, Event)
                assert e.day == day and e.kind in EVENT_KINDS and e.kind != "answer"
                assert e.text and e.tokens and e.tokens[0] == e.kind
                assert all(isinstance(t, str) for t in e.tokens)
        assert life.events(-1) == () and life.events(life.meta.n_days) == ()


def test_every_activity_reported_daily() -> None:
    life = make_life(11)
    for day in range(life.meta.n_days):
        reported = {e.subject for e in life.events(day) if e.kind in OUTCOME_KINDS}
        assert set(life.meta.activities) <= reported


def test_queries_well_formed() -> None:
    for seed in range(5):
        life = make_life(seed)
        n = life.meta.n_days
        for day in range(n):
            qs = life.queries(day)
            if day <= n - 8:
                assert 6 <= len(qs) <= 10
            assert len({q.qid for q in qs}) == len(qs)
            for q in qs:
                assert q.asked_day == day and day < q.target_day <= day + 7 and q.target_day < n
                assert q.family in QUERY_FAMILIES and q.text.endswith("?") and q.tokens
        assert life.queries(n - 1) == ()


def test_questions_and_answers() -> None:
    for seed in range(8):
        life = make_life(seed)
        for day in range(life.meta.n_days):
            qs = life.questions(day)
            assert 3 <= len(qs) <= 5
            assert len({q.question_id for q in qs}) == len(qs)
            for q in qs:
                assert q.day == day and 0.05 <= q.cost <= 0.2 and len(q.options) >= 2
                a = life.answer(q.question_id)
                assert a.kind == "answer" and a.day == day and a.text
                assert a.tokens in q.options


def test_unknown_ids_raise() -> None:
    life = make_life(1)
    with pytest.raises(KeyError):
        life.resolve("nope")
    with pytest.raises(KeyError):
        life.answer("nope")


# ---- determinism -------------------------------------------------------------------------


def _fingerprint(life: Life) -> tuple[object, ...]:
    n = life.meta.n_days
    evs = tuple(life.events(d) for d in range(n))
    qs = tuple(life.queries(d) for d in range(n))
    ques = tuple(life.questions(d) for d in range(n))
    ans = tuple(life.answer(q.question_id) for day in ques for q in day)
    res = tuple(life.resolve(q.qid) for day in qs for q in day)
    return life.meta, evs, qs, ques, ans, res


def test_deterministic_from_seed() -> None:
    a, b = make_life(42), make_life(42)
    assert _fingerprint(a) == _fingerprint(b)
    assert a.truth() == b.truth()
    assert _fingerprint(make_life(43))[1] != _fingerprint(a)[1]


def test_no_lookahead_prefix() -> None:
    """A shorter run of the same person is the same person: nothing on day d depends on later days."""
    short, full = make_life(9, n_days=40), make_life(9)
    for d in range(40):
        assert short.events(d) == full.events(d)
        assert short.questions(d) == full.questions(d)
    for d in range(40 - 8):
        assert short.queries(d) == full.queries(d)
        for q in short.queries(d):
            assert short.resolve(q.qid) == full.resolve(q.qid)


# ---- leakage -----------------------------------------------------------------------------


def _outcome_event(life: Life, q: Query) -> Event:
    matches = [e for e in life.events(q.target_day)
               if e.kind in OUTCOME_KINDS and e.subject == q.subject and e.tokens[1] == q.tokens[1]]
    assert len(matches) == 1, (q, matches)
    return matches[0]


def test_no_event_reveals_a_later_outcome() -> None:
    for seed in range(30):
        life = make_life(seed)
        for day in range(life.meta.n_days):
            for e in life.events(day):
                if e.kind in ("did", "skipped", "cancel"):
                    # an outcome is only ever reported on its own day
                    assert not any(t.startswith(("for=", "in=")) for t in e.tokens), e
        for q in _all_queries(life):
            r = life.resolve(q.qid)
            assert _outcome_event(life, q).kind == ("did" if r.outcome else "skipped")
            for d in range(q.asked_day + 1):
                for e in life.events(d):
                    same = e.subject == q.subject and len(e.tokens) > 1 and e.tokens[1] == q.tokens[1]
                    assert not (same and e.kind in ("did", "skipped", "cancel") and d == q.target_day)


# ---- families ----------------------------------------------------------------------------


def test_both_families_common(resolved: dict[str, np.ndarray]) -> None:
    share_hidden = resolved["hidden"].mean()
    assert 0.2 <= share_hidden <= 0.8


def test_families_per_life() -> None:
    for seed in range(20):
        qs = _all_queries(make_life(seed))
        share = sum(q.family == "count" for q in qs) / len(qs)
        assert 0.15 <= share <= 0.6


# ---- the oracle --------------------------------------------------------------------------


def _check_calibration(p: np.ndarray, y: np.ndarray, life: np.ndarray) -> None:
    edges = np.linspace(0.0, 1.0, 11)
    which = np.clip(np.digitize(p, edges) - 1, 0, 9)
    for b in range(10):
        sel = which == b
        n = int(sel.sum())
        if n < 300:
            continue
        gap = float(np.mean(y[sel] - p[sel]))
        se = _clustered_se(y[sel] - p[sel], life[sel])
        assert abs(gap) < 4.0 * se + 0.004, (b, n, gap, se)


def test_probabilities_in_open_interval(resolved: dict[str, np.ndarray]) -> None:
    for key in ("oracle", "state"):
        assert np.all((resolved[key] > 0) & (resolved[key] < 1))


def test_oracle_calibrated(resolved: dict[str, np.ndarray]) -> None:
    p, y, hidden, life = resolved["oracle"], resolved["y"], resolved["hidden"], resolved["life"]
    assert len(p) > 50_000
    _check_calibration(p, y, life)
    _check_calibration(p[hidden], y[hidden], life[hidden])
    _check_calibration(p[~hidden], y[~hidden], life[~hidden])


def test_state_p_calibrated(resolved: dict[str, np.ndarray]) -> None:
    _check_calibration(resolved["state"], resolved["y"], resolved["life"])


def test_oracle_is_the_mean_of_state_p(resolved: dict[str, np.ndarray]) -> None:
    """oracle_p = E[state_p | what was observed], so within any oracle bin the means agree."""
    p, s, life = resolved["oracle"], resolved["state"], resolved["life"]
    edges = np.linspace(0.0, 1.0, 21)
    which = np.clip(np.digitize(p, edges) - 1, 0, 19)
    for b in range(20):
        sel = which == b
        if int(sel.sum()) < 300:
            continue
        gap = float(np.mean(s[sel] - p[sel]))
        assert abs(gap) < 4.0 * _clustered_se(s[sel] - p[sel], life[sel]) + 0.001, (b, gap)


def _observed_features(life: Life, day: int) -> set[str]:
    """Things anyone reading days day-1 and day could see: mood, talk, reasons, messages."""
    out = set()
    for back in (0, 1):
        for e in life.events(day - back):
            if e.kind == "mood":
                out.add(f"mood{back}={int(e.value or 0)}")
            elif e.kind == "mention" and e.tokens[1] in ("health", "work", "school", "weather"):
                out.add(f"{e.tokens[1]}{back}={e.tokens[2]}")
            elif e.kind == "skipped" and e.tokens[-1].startswith("reason="):
                out.add(f"why{back}={e.tokens[-1]}")
            elif e.kind == "message" and e.tokens[2] in ("swamped", "invite", "doubt", "confirm", "deadline"):
                out.add(f"msg{back}={e.tokens[2]}")
    return out


def test_oracle_exact_given_what_was_seen(lives: list[Life]) -> None:
    """For any observable feature, E[state_p - oracle_p | feature] = 0. A filter whose likelihoods
    disagree with how the world draws events shows a gap after mood reports, reasons or talk."""
    gaps: dict[str, list[tuple[int, float]]] = {}
    for i, life in enumerate(lives):
        for day in range(1, life.meta.n_days):
            feats = _observed_features(life, day)
            for q in life.queries(day):
                if q.family == "hidden":
                    r = life.resolve(q.qid)
                    for f in feats:
                        gaps.setdefault(f, []).append((i, r.state_p - r.oracle_p))
    checked = 0
    for f, rows in gaps.items():
        if len(rows) < 500:
            continue
        who = np.array([w for w, _ in rows])
        gap = np.array([g for _, g in rows])
        z = abs(gap.mean()) / (_clustered_se(gap, who) + 1e-12)
        assert z < 4.5, (f, len(rows), gap.mean(), z)
        checked += 1
    assert checked >= 30


def test_oracle_beats_base_rate_and_state_beats_oracle(resolved: dict[str, np.ndarray]) -> None:
    y = resolved["y"]
    base = np.full_like(y, y.mean())
    ll_base, ll_oracle, ll_state = (_log_loss(base, y), _log_loss(resolved["oracle"], y),
                                    _log_loss(resolved["state"], y))
    assert ll_oracle < ll_base - 0.05
    assert ll_state <= ll_oracle + 1e-9
    hid = resolved["hidden"]
    assert _log_loss(resolved["state"][hid], y[hid]) < _log_loss(resolved["oracle"][hid], y[hid])


def test_count_queries_need_only_the_weekday(lives: list[Life]) -> None:
    for life in lives[:20]:
        for q in _all_queries(life):
            if q.family == "count":
                r = life.resolve(q.qid)
                assert r.oracle_p == r.state_p


# ---- truth -------------------------------------------------------------------------------


def test_truth(lives: list[Life]) -> None:
    n_shift = 0
    for life in lives:
        t = life.truth()
        assert list(t.shift_days) == sorted(set(t.shift_days))
        assert all(0 < d < life.meta.n_days for d in t.shift_days)
        assert t.params and t.hidden_facts
        n_shift += len(t.shift_days)
    assert n_shift > 0


def test_people_vary() -> None:
    archetypes = {make_life(s).truth().hidden_facts["archetype"] for s in range(60)}
    assert len(archetypes) >= 5


# ---- speed -------------------------------------------------------------------------------


def test_speed_500_lives() -> None:
    t0 = time.perf_counter()
    for seed in range(10_000, 10_500):
        life = make_life(seed)
        for d in range(life.meta.n_days):
            life.events(d)
            life.queries(d)
            life.questions(d)
    assert time.perf_counter() - t0 < 120.0
