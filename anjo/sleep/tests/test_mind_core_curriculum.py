# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The dream curriculum: the night's dreams choose what she rehearses (pure, no torch)."""
from __future__ import annotations

import pytest

from anjo.sleep.mind.core import curriculum
from anjo.sleep.mind.core.types import Ignition, Predictions, SlotPrediction


def _slot(slot: int, risk: float, self_error: float, cites: tuple[str, ...]) -> SlotPrediction:
    return SlotPrediction(slot=slot, usage=0.9, risk=risk, hazard=(0.1,), missing=0.1, self_error=self_error,
                          choice=0.2, cites=cites)


SEEN = Predictions(speak=0.3, none=0.4, slots=(
    _slot(0, 0.10, 0.01, ("a1", "a2")),
    _slot(1, 0.60, 0.20, ("b1", "b2", "b3")),
    _slot(2, 0.30, 0.05, ("c1",)),
    _slot(3, 0.90, 0.30, ()),  # open, but cites nothing real: never rehearsed
))


def test_threads_rank_by_how_unsure_or_at_risk_she_is_and_what_her_dreams_ignited() -> None:
    ranked = curriculum.rank_threads(SEEN)
    assert [t.slot for t in ranked] == [1, 2, 0]
    assert ranked[0].score == pytest.approx(0.8)
    lit = curriculum.rank_threads(SEEN, (Ignition(slot=0, other=None, kind="risk", score=0.9, cited_event_ids=("a1",)),))
    assert [t.slot for t in lit] == [0, 1, 2]
    both = curriculum.rank_threads(SEEN, (Ignition(slot=2, other=0, kind="collision", score=0.7, cited_event_ids=("c1", "a1")),))
    assert [t.slot for t in both][:2] == [2, 0]


def test_the_budget_goes_to_the_top_threads_by_their_surprise() -> None:
    weights = {"b1": 3.0, "b2": 3.0, "b3": 3.0, "c1": 1.0}
    picks = curriculum.pick(SEEN, (), weights, budget=4, threads=2)
    assert [t.slot for t in picks.threads] == [1, 2]
    assert [t.share for t in picks.threads] == pytest.approx([0.75, 0.25])
    assert picks.evidence_ids == ("b1", "b2", "b3", "c1")
    small = curriculum.pick(SEEN, (), weights, budget=2, threads=2)
    assert small.evidence_ids == ("b1", "c1")  # every picked thread gets at least one rehearsal
    plain = curriculum.pick(SEEN, (), {}, budget=3, threads=3)
    assert len(plain.evidence_ids) == 3 and set(plain.evidence_ids) <= {"a1", "a2", "b1", "b2", "b3", "c1"}


def test_an_unspent_share_passes_to_the_threads_with_more_to_rehearse() -> None:
    picks = curriculum.pick(SEEN, (), {"c1": 9.0}, budget=4, threads=2)
    assert picks.evidence_ids == ("b1", "b2", "b3", "c1")


def test_nothing_open_picks_nothing() -> None:
    empty = Predictions(speak=0.1, none=0.9, slots=())
    assert curriculum.pick(empty, (), {}, budget=5).evidence_ids == ()
    with pytest.raises(ValueError, match="budget"):
        curriculum.pick(SEEN, (), {}, budget=0)


def test_picks_record_ids_and_numbers_only() -> None:
    picks = curriculum.pick(SEEN, (), {"b1": 2.0}, budget=3, threads=2)
    record = curriculum.record(picks)
    assert record["evidence_ids"] == list(picks.evidence_ids)
    assert {"slot", "score", "share", "cites"} <= set(record["threads"][0])


def _within(weights: dict[str, float]) -> bool:
    mean = sum(weights.values()) / len(weights)
    return mean == pytest.approx(1.0, abs=1e-9) and all(
        curriculum.WEIGHT_FLOOR - 1e-12 <= w <= curriculum.WEIGHT_CEILING + 1e-12 for w in weights.values())


@pytest.mark.parametrize("surprises", [
    {"huge": 1000.0, "b": 1.0, "c": 1.0, "d": 1.0},  # one startling event can't take the night
    {"a": 0.1, "b": 0.2, "c": 5.0, "d": 0.3, "e": 0.01},
    {"a": 0.0, "b": 0.0, "c": 3.0},  # a third surprised: the floor and the ceiling, exactly
    {"a": 0.0, "b": 0.0, "c": 0.0, "d": 5.0},  # under a third: the rest share what is left
    {"a": 0.0, "b": 2.0, "c": 4.0, "d": 6.0},
])
def test_surprise_weights_stay_bounded_and_average_one(surprises: dict[str, float]) -> None:
    weights = curriculum.bounded_weights(surprises)
    assert weights.keys() == surprises.keys() and _within(weights)
    ranked = sorted(surprises, key=surprises.__getitem__)
    assert [weights[k] for k in ranked] == sorted(weights.values())  # more surprise, never less weight
    assert weights == curriculum.bounded_weights(dict(reversed(surprises.items())))  # order-free


def test_one_huge_surprise_takes_the_ceiling_and_no_weight_reaches_zero() -> None:
    weights = curriculum.bounded_weights({"huge": 1e9, "b": 1.0, "c": 1.0, "d": 1.0})
    assert weights["huge"] == curriculum.WEIGHT_CEILING
    assert weights["b"] == weights["c"] == weights["d"] == pytest.approx(2 / 3)
    assert curriculum.bounded_weights({"a": 0.0, "b": 0.0, "c": 0.0, "d": 5.0})["a"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("surprises", [{}, {"only": 3.0}, {"a": 0.4, "b": 0.4}, {"a": 0.0, "b": 0.0}])
def test_one_event_equal_surprises_or_none_weigh_one(surprises: dict[str, float]) -> None:
    assert curriculum.bounded_weights(surprises) == dict.fromkeys(surprises, 1.0)


def test_a_surprise_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        curriculum.bounded_weights({"a": float("nan"), "b": 1.0})
