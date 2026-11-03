"""Scoring of the cognition sprint (score.py): T1-T4 metrics, paired intervals and the report."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from anjo.sleep.mind.cog import score
from anjo.sleep.mind.cog.run import AskRecord, LifeRecord, QueryRecord, run_many
from anjo.sleep.tests.test_cog_run import FakeLife, FreqPredictor, make_always_work


def _q(qid: str, day: int, p: float, outcome: bool, *, oracle: float = 0.5, family: str = "count",
       subject: str = "run", u: float | None = None, target: int | None = None) -> QueryRecord:
    return QueryRecord(qid=qid, subject=subject, family=family, asked_day=day,
                       target_day=day + 1 if target is None else target, p=p, u=u, outcome=outcome,
                       oracle_p=oracle, state_p=oracle)


def _rec(run: str, queries: list[QueryRecord], *, seed: int = 0, asks: tuple[AskRecord, ...] = (),
         inquirer: str | None = None, predictor: str | None = None, shift_days: tuple[int, ...] = (),
         shift_subjects: tuple[str, ...] | None = None, n_days: int = 28) -> LifeRecord:
    return LifeRecord(run=run, predictor=predictor or run.split("+")[0], inquirer=inquirer, seed=seed,
                      person_id=f"p{seed}", n_days=n_days, start_weekday=0, topics=("weather",),
                      shift_days=shift_days, shift_subjects=shift_subjects, queries=tuple(queries),
                      asks=asks)


def _ask(day: int, cost: float = 0.1, distractor: bool = False) -> AskRecord:
    return AskRecord(day=day, question_id=f"a{day}", subject="weather" if distractor else "work",
                     cost=cost, is_distractor=distractor, answer_tokens=("answer",))


# -- the per-query scores --
def test_log_loss_and_brier_clip_and_score() -> None:
    p = np.array([0.8, 0.8, 1.0, 0.0])
    y = np.array([True, False, False, True])
    ll = score.log_loss(p, y)
    assert ll[0] == pytest.approx(-math.log(0.8)) and ll[1] == pytest.approx(-math.log(0.2))
    assert ll[2] == pytest.approx(-math.log(1e-4)) and ll[3] == pytest.approx(-math.log(1e-4))
    assert score.brier(p[:2], y[:2]).tolist() == pytest.approx([0.04, 0.64])


def test_ece_is_zero_when_calibrated_and_the_gap_otherwise() -> None:
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert score.ece(p, y) == pytest.approx(0.0)
    assert score.ece(np.full(4, 0.9), np.array([1, 0, 1, 0], dtype=bool)) == pytest.approx(0.4)


def test_aurc_rewards_uncertainty_that_ranks_the_losses() -> None:
    loss = np.array([0.1, 0.2, 0.9, 2.0])
    good = score.aurc(np.array([0.0, 1.0, 2.0, 3.0]), loss)
    bad = score.aurc(np.array([3.0, 2.0, 1.0, 0.0]), loss)
    flat = score.aurc(np.zeros(4), loss)
    assert good == pytest.approx(np.mean([0.1, 0.15, 0.4, 0.8]))
    assert good < flat < bad and flat == pytest.approx(loss.mean())


def test_spearman_handles_ties_and_constants() -> None:
    assert score.spearman(np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])) == pytest.approx(1.0)
    assert score.spearman(np.array([1, 2, 3, 4.0]), np.array([4, 3, 2, 1.0])) == pytest.approx(-1.0)
    assert score.spearman(np.array([1, 1, 2, 2.0]), np.array([1, 2, 3, 4.0])) == pytest.approx(0.8944, abs=1e-3)
    assert score.spearman(np.ones(3), np.array([1, 2, 3.0])) is None


# -- T1/T2 over a run --
def test_summary_scores_families_weeks_and_skips_aurc_without_u() -> None:
    recs = [_rec("a", [_q("x1", 0, 0.8, True, family="count"), _q("x2", 8, 0.4, False, family="hidden")]),
            _rec("a", [_q("y1", 1, 0.6, True, family="count")], seed=1)]
    s = score.summarize_run(recs)
    assert s["n_lives"] == 2 and s["n_queries"] == 3
    assert s["log_loss"] == pytest.approx(np.mean([-math.log(0.8), -math.log(0.6), -math.log(0.6)]))
    assert s["by_family"]["hidden"]["n"] == 1
    assert s["by_family"]["hidden"]["ece"] == pytest.approx(0.4)  # one query at 0.4 that did not happen
    assert s["by_family"]["count"]["log_loss"] == pytest.approx(np.mean([-math.log(0.8), -math.log(0.6)]))
    assert [w["week"] for w in s["learning_curve"]] == [0, 1]
    assert s["learning_curve"][1]["n"] == 1
    assert s["oracle_log_loss"] == pytest.approx(math.log(2))
    assert s["aurc"] is None and s["spearman_u_excess"] is None


def test_summary_t2_uses_u_against_excess_loss() -> None:
    qs = [_q(f"q{i}", i, p, o, u=u) for i, (p, o, u) in
          enumerate([(0.9, True, 0.1), (0.7, True, 0.2), (0.6, False, 0.3), (0.2, True, 0.4)])]
    s = score.summarize_run([_rec("a", qs)])
    assert s["spearman_u_excess"] == pytest.approx(1.0)
    assert s["aurc"]["aurc"] == pytest.approx(s["aurc"]["optimal"])
    assert s["aurc"]["aurc"] < s["aurc"]["random"]


# -- T3 --
def test_t3_counts_the_week_after_each_ask_minus_its_cost() -> None:
    base = _rec("m", [_q("a", 0, 0.5, True), _q("b", 3, 0.5, True), _q("c", 7, 0.5, True),
                      _q("d", 12, 0.5, True)])
    asked = _rec("m+inq", [_q("a", 0, 0.5, True), _q("b", 3, 0.8, True), _q("c", 7, 0.9, True),
                           _q("d", 12, 0.9, True)],
                 asks=(_ask(2, cost=0.05), _ask(3, cost=0.05, distractor=True)), inquirer="inq", predictor="m")
    t3 = score.t3_utility([asked], [base], window=7)
    # windows [2, 8] and [3, 9]: b (day 3) and c (day 7) count; a (day 0) and d (day 12) do not
    reduction = (math.log(2) + math.log(0.8)) + (math.log(2) + math.log(0.9))
    assert t3["loss_reduction_per_life"] == pytest.approx(reduction)
    assert t3["utility_per_life"] == pytest.approx(reduction - 0.1)
    assert t3["asks_per_life"] == 2 and t3["distractor_share"] == pytest.approx(0.5)
    assert t3["n_window_queries"] == 2 and t3["n_lives"] == 1


def test_t3_never_asking_is_worth_exactly_zero() -> None:
    base = _rec("m", [_q("a", 0, 0.5, True)])
    silent = _rec("m+never", [_q("a", 0, 0.5, True)], inquirer="never", predictor="m")
    t3 = score.t3_utility([silent], [base])
    assert t3["utility_per_life"] == 0.0 and t3["distractor_share"] is None


# -- T4 --
def _shift_life(gaps: list[float], *, seed: int = 0, shift: int = 10) -> LifeRecord:
    """A life with one query a day; after the shift its excess loss over the oracle is ``gaps``."""
    qs = []
    for d in range(shift + len(gaps)):
        gap = gaps[d - shift] if d >= shift else 0.0
        p = math.exp(-(math.log(2) + gap))  # loss = log 2 + gap, oracle loss = log 2
        qs.append(_q(f"q{d}", d, p, True, subject="run" if d % 2 else "gym"))
    return _rec("m", qs, seed=seed, shift_days=(shift,), shift_subjects=("run",), n_days=shift + len(gaps))


def test_t4_recovery_is_the_first_post_shift_week_within_margin() -> None:
    # weeks starting 0, 1, 2, 3 days after the shift have mean gaps 3/7, 2/7, 1/7, 0
    rec = _shift_life([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t4 = score.t4_recovery([rec], margin=0.05)
    assert t4 is not None
    assert t4["n_shifts"] == 1 and t4["recovered_share"] == 1.0
    assert t4["recovery_days_mean"] == 3
    assert t4["gap14_mean"] == pytest.approx(3 / 14)


def test_t4_censors_a_predictor_that_never_recovers_and_scores_untouched_subjects() -> None:
    rec = _shift_life([0.5] * 10)
    t4 = score.t4_recovery([rec], margin=0.05)
    assert t4 is not None
    assert t4["recovered_share"] == 0.0
    assert t4["recovery_days_mean"] == 4  # censored: four full weeks fit after the shift, none within margin
    untouched = t4["untouched"]
    assert untouched["post_excess"] == pytest.approx(0.5)  # gym queries after the shift
    assert untouched["pre_excess"] == pytest.approx(0.0) and untouched["forgetting"] == pytest.approx(0.5)


def test_t4_skips_recovery_when_no_full_week_follows_the_shift() -> None:
    t4 = score.t4_recovery([_shift_life([0.5] * 5)], margin=0.05)
    assert t4 is not None
    assert t4["n_shifts"] == 1 and t4["n_recovery"] == 0 and t4["recovery_days_mean"] is None
    assert t4["gap14_mean"] == pytest.approx(0.5)


def test_t4_is_none_without_shifts() -> None:
    assert score.t4_recovery([_rec("m", [_q("a", 0, 0.5, True)])]) is None


# -- paired intervals --
def test_paired_comparison_is_deterministic_and_pairs_on_common_queries() -> None:
    a = [_rec("a", [_q("x", 0, 0.9, True), _q("y", 1, 0.9, True)], seed=s) for s in range(5)]
    b = [_rec("b", [_q("x", 0, 0.5, True)], seed=s) for s in range(5)]
    c1 = score.compare(a, b, n_boot=500, seed=3)
    c2 = score.compare(a, b, n_boot=500, seed=3)
    assert c1 == c2 and c1["n_lives"] == 5 and c1["n_queries"] == 5
    diff = -math.log(0.9) - math.log(2)
    assert c1["log_loss_diff"]["mean"] == pytest.approx(diff)
    assert c1["log_loss_diff"]["ci"] == pytest.approx([diff, diff])


def test_paired_interval_covers_a_noisy_difference() -> None:
    rng = np.random.default_rng(0)
    a, b = [], []
    for s in range(40):
        pa = float(np.clip(0.7 + rng.normal(0, 0.1), 0.05, 0.95))
        a.append(_rec("a", [_q("x", 0, pa, True)], seed=s))
        b.append(_rec("b", [_q("x", 0, 0.7, True)], seed=s))
    c = score.compare(a, b, n_boot=1000, seed=0)
    lo, hi = c["log_loss_diff"]["ci"]
    assert lo < c["log_loss_diff"]["mean"] < hi


# -- the report --
def test_report_on_fake_lives_is_json_and_finds_t3_pairs() -> None:
    lives = [FakeLife(s, n_days=21, shift_day=10 if s % 2 else None) for s in range(4)]
    base = run_many(lives, FreqPredictor)
    asked = run_many(lives, FreqPredictor, inquirer_factory=make_always_work)
    report = score.build_report(score.group_runs([*base, *asked]), n_boot=200)
    json.dumps(report, allow_nan=False)
    assert set(report["runs"]) == {"freq", "freq+always_work"}
    assert report["runs"]["freq"]["t3"] is None
    t3 = report["runs"]["freq+always_work"]["t3"]
    assert t3["base_run"] == "freq" and t3["asks_per_life"] == 21 and t3["distractor_share"] == 0.0
    assert report["runs"]["freq"]["t4"]["n_shifts"] == 2
    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["by_week"][0]["week"] == 0


def test_report_with_a_reference_compares_everyone_to_it() -> None:
    runs = {name: [_rec(name, [_q("x", 0, p, True)], seed=s) for s in range(3)]
            for name, p in (("ref", 0.6), ("b", 0.7), ("c", 0.8))}
    report = score.build_report(runs, reference="ref", n_boot=100)
    assert [(c["a"], c["b"]) for c in report["comparisons"]] == [("b", "ref"), ("c", "ref")]
    with pytest.raises(ValueError, match="reference"):
        score.build_report(runs, reference="nope")


def test_summary_totals_usage_over_lives() -> None:
    a = _rec("g", [_q("x", 0, 0.6, True)])
    b = _rec("g", [_q("y", 0, 0.6, True)], seed=1)
    a = LifeRecord(**{**a.__dict__, "usage": {"calls": 2.0, "prompt_chars": 100.0}})
    b = LifeRecord(**{**b.__dict__, "usage": {"calls": 3.0}})
    assert score.summarize_run([a, b])["usage"] == {"calls": 5.0, "prompt_chars": 100.0}
