# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The MIND experiment's outcomes (Task 16): her expectations and their resolution, the frozen
relevance labels, and the report. Offline, fakes only."""
from __future__ import annotations

import dataclasses
import inspect
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from anjo.sleep.evidence.targets import expectation_record
from anjo.sleep.mind import expect
from anjo.sleep.mind.deciders import NightPicks, Note
from anjo.sleep.model_api import ChatResult
from anjo.sleep.personas.model import NewWork
from anjo.sleep.probes import relevance
from anjo.sleep.reports import mind
from anjo.sleep.types import Evidence, Expectation, Target

START = date(2026, 11, 3)
PROJECT = "Black film line"
COEFFICIENTS = expect.Coefficients(intercept=-0.4, understand=0.3, unacted=0.2, days=-0.05, base_rate=0.25, n=20)


def _at(day: int) -> datetime:
    return datetime.combine(START + timedelta(days=day), datetime.min.time(), tzinfo=timezone.utc)


def _thread(target_id: str = "t1", **changes: Any) -> Target:
    base = Target(
        id=target_id, person_id="synthetic-life", kind="thread", label="private words about a topic",
        check_after=None, status="open", revision=3, acted_revision=1, confidence=None, last_mentioned=START,
    )
    return replace(base, **changes)


def _night(day: int, studied: dict[str, str | None], notes: dict[str, Note | None] | None = None) -> NightPicks:
    study = {name: studied.get(name) for name in ("D0", "D1", "D2", "D3", "D4")}
    return NightPicks(night=START + timedelta(days=day), studied=study["D0"] is not None, study=study,
                      notes=notes or {"D0": None, "D1": None}, d1=())


def _labels(relevant: set[tuple[str, str]], targets: tuple[str, ...], items: tuple[str, ...]) -> relevance.FrozenLabels:
    pairs = {(t, i): (t, i) in relevant for t in targets for i in items}
    return relevance.freeze("p" * 64, pairs, frozenset())


def _life(
    life_id: str, nights: tuple[NightPicks, ...], arrivals: dict[str, int], relevant: set[tuple[str, str]], *,
    split: Any = "held_out", last_day: int = 20, expectations: tuple[Expectation, ...] = (),
    returns: dict[str, tuple[date, ...]] | None = None,
) -> mind.LifeRecord:
    picked = {t for n in nights for t in n.study.values() if t} | {e.target_id for e in expectations}
    labels = _labels(relevant, tuple(sorted(picked)), tuple(sorted(arrivals)))
    return mind.LifeRecord(
        life_id=life_id, split=split, nights=nights,
        arrivals=tuple(mind.Arrival(item, START + timedelta(days=day)) for item, day in sorted(arrivals.items())),
        labels=labels, last_day=START + timedelta(days=last_day), expectations=expectations, returns=returns or {},
    )


def _registered(*lives: mind.LifeRecord) -> dict[str, str]:
    return {life.life_id: life.labels.digest for life in lives}


def _report(*lives: mind.LifeRecord, **kwargs: Any) -> mind.MindReport:
    return mind.mind_report(lives, registered=_registered(*lives), coefficients=COEFFICIENTS, resamples=400, **kwargs)


class FakeSpark:
    """Muse Spark for the relevance pass: relevant only for the pairs given."""

    def __init__(self, relevant: set[tuple[str, str]], labels: dict[str, str], items: dict[str, str]) -> None:
        self.relevant, self.labels, self.items = relevant, labels, items
        self.prompts: list[str] = []
        self.temperatures: list[float | None] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        target = next(t for t, label in self.labels.items() if label in prompt)
        item = next(i for i, text in self.items.items() if text in prompt)
        verdict = {"relevant": (target, item) in self.relevant}
        return ChatResult(text=json.dumps(verdict), tool_calls=(), model=str(model), input_tokens=90,
                          output_tokens=6, finish_reason="stop")


# ── The spec's tests ──────────────────────────────────────────────────────────


def test_relevance_labels_blind_and_frozen_before_scoring() -> None:
    names = {"t1": "sorting black film", "t2": "pricing the pilot"}
    items = (NewWork("w1", 3, PROJECT, "A buyer asks about film from a farm"), NewWork("w2", 5, PROJECT, "A grant form"))
    # who picked which target is never an input: the same picks, swapped between deciders, give the same pairs
    one = relevance.pairs_to_label(["t1", "t2", "t1"], names, items)
    other = relevance.pairs_to_label(["t2", "t1"], names, items)
    assert one == other and len(one) == 4
    spark = FakeSpark({("t1", "w1")}, names, {w.id: w.text for w in items})
    labels = relevance.label_pairs(one, spark)
    assert len(spark.prompts) == 4 and spark.temperatures == [0.0] * 4  # each pair once
    for prompt in spark.prompts:
        assert not any(word in prompt for word in ("D0", "D1", "D2", "D3", "D4", "decider", "plain model"))
    assert labels.relevant("t1", "w1") and not labels.relevant("t2", "w1")
    # frozen: read-only, hashed, and a changed label no longer matches its hash
    with pytest.raises(TypeError):
        labels.labels[("t2", "w1")] = True  # type: ignore[index]
    with pytest.raises(ValueError):
        replace(labels, labels={**labels.labels, ("t2", "w1"): True})
    relevance.verify(labels, labels.digest)
    with pytest.raises(ValueError):
        relevance.verify(labels, "0" * 64)
    # scoring refuses labels that aren't the registered ones, and picks the labels never covered
    life = _life("h1", (_night(0, {"D0": "t1"}),), {"w1": 3}, {("t1", "w1")})
    with pytest.raises(ValueError, match="registered"):
        mind.mind_report([life], registered={"h1": "0" * 64}, coefficients=COEFFICIENTS, resamples=10)
    late = replace(life, nights=(*life.nights, _night(1, {"D0": "t9"})))
    with pytest.raises(ValueError, match="not labeled"):
        mind.mind_report([late], registered=_registered(late), coefficients=COEFFICIENTS, resamples=10)


def test_silence_counts_as_no_hit() -> None:
    nights = (_night(0, {"D0": "t1", "D1": None}), _night(1, {"D0": None, "D1": None}))
    life = _life("h1", nights, {"w1": 3}, {("t1", "w1")})
    assert mind.anticipation(life, "D0") == mind.Share(1, 2)  # a silent night stays in the denominator
    assert mind.anticipation(life, "D1") == mind.Share(0, 2)
    assert mind.silence(life, "D0") == mind.Share(1, 2) and mind.silence(life, "D1") == mind.Share(2, 2)
    report = _report(life)
    [row] = report.lives
    assert row.anticipation["D1"].share.rate == 0.0 and row.anticipation["D1"].interval == (0.0, 0.0)


def test_distinct_items_count_once_in_the_secondary() -> None:
    nights = tuple(_night(day, {"D0": "t1"}) for day in range(3))
    life = _life("h1", nights, {"w1": 3, "w2": 9, "w3": 15}, {("t1", "w1"), ("t1", "w2"), ("t1", "w3")})
    assert mind.anticipation(life, "D0") == mind.Share(3, 3)  # every night hits: w1 counts on all three
    distinct = mind.distinct_items(life, "D0")
    assert distinct == {"w1": START, "w2": START + timedelta(days=2)}  # each once, at the earliest night
    assert "w3" not in distinct  # past every night's 7-day horizon
    report = _report(life)
    assert report.lives[0].distinct["D0"] == 2 and report.pooled.distinct["D0"] == 2


def test_unresolved_horizon_is_censored() -> None:
    made = expect.expectation(_thread(), "understand", _at(6), COEFFICIENTS)
    assert made.horizon == _at(13)
    life_end = _at(11)  # the timeline's last day ends before the horizon
    early = (_at(7),)  # the person came back to it before the horizon, and it is still censored
    assert expect.resolve(made, returns=early, arrivals=(), life_end=life_end, as_of=life_end) == "censored"
    assert expect.resolve(made, returns=(), arrivals=(), life_end=life_end, as_of=_at(6)) == "censored"
    life = _life("h1", (), {}, set(), last_day=10, expectations=(made,), returns={"t1": (START + timedelta(days=7),)})
    demand = _report(life).demand
    assert (demand.n, demand.censored) == (0, 1) and demand.brier is None


def test_date_passing_alone_never_resolves_needed() -> None:
    plan = _thread("c1", kind="commitment", check_after=_at(5) + timedelta(days=3), revision=1, acted_revision=0)
    made = expect.expectation(plan, "follow_up", _at(5), COEFFICIENTS)
    assert made.horizon == plan.check_after  # its date comes before the seven days do
    end = _at(30)
    assert expect.resolve(made, returns=(), arrivals=(), life_end=end, as_of=_at(7)) is None
    assert expect.resolve(made, returns=(), arrivals=(), life_end=end, as_of=_at(9)) == "not_needed"
    assert expect.resolve(made, returns=(_at(7),), arrivals=(), life_end=end, as_of=_at(9)) == "needed"
    assert "target" not in inspect.signature(expect.resolve).parameters  # it can't read a date of the target
    # her own homework on the target, or a held-out item, is not the person returning to it
    item = Evidence(id="e1", person_id="synthetic-life", day=START, kind="homework", text="x", source="s",
                    held_out=False, created_at=_at(7), target_id="c1")
    person = replace(item, id="e2", kind="correction")
    assert expect.returning([item, replace(item, id="e3", kind="practice"), replace(person, held_out=True)], "c1") == ()
    assert expect.returning([item, person], "c1") == (person,)


def test_held_out_report_excludes_development_lives() -> None:
    hit = (_night(0, {"D0": "t1", "D1": "t1"}),)
    dev = _life("dev-founder", hit, {"w1": 2}, {("t1", "w1")}, split="development")
    held = _life("h1", (_night(0, {"D0": "t1", "D1": None}),), {"w1": 2}, set())
    report = _report(dev, held)
    assert report.pooled.lives == ("h1",)
    assert report.pooled.anticipation["D0"].share == mind.Share(0, 1)
    assert [(row.life_id, row.split) for row in report.lives] == [("dev-founder", "development"), ("h1", "held_out")]
    text = mind.render(report)
    held_part, dev_part = text.split("## Development lives")
    assert "dev-founder" not in held_part and "development" in dev_part.lower() and "dev-founder" in dev_part
    assert "Pooled over 1 held-out life" in held_part
    assert "significant" not in text.lower()


def test_expectation_fields_never_describe_the_person() -> None:
    assert [f.name for f in dataclasses.fields(Expectation)] == [
        "id", "target_id", "made_at", "horizon", "p_needed", "outcome",
    ]
    assert [f.name for f in dataclasses.fields(expect.Features)] == ["understand", "unacted", "days"]
    assert [f.name for f in dataclasses.fields(expect.Coefficients)] == [
        "intercept", "understand", "unacted", "days", "base_rate", "n",
    ]
    assert list(inspect.signature(expect.estimate).parameters) == ["coefficients", "features"]
    made = expect.expectation(_thread(), "understand", _at(2), COEFFICIENTS)
    text = json.dumps(expectation_record(made))
    assert "private words" not in text and "synthetic-life" not in text
    assert made.outcome is None and 0.0 <= made.p_needed <= 1.0
    with pytest.raises(ValueError):
        expect.expectation(_thread(), "share", _at(2), COEFFICIENTS)  # a note is not a study


# ── More of the outcomes ──────────────────────────────────────────────────────


def test_fit_learns_the_direction_and_refuses_held_out_rows() -> None:
    rows = [expect.FitRow(expect.Features(True, unacted, 7.0), unacted >= 3, "development") for unacted in range(6)] * 4
    fitted = expect.fit(rows)
    assert fitted.unacted > 0 and fitted.n == 24 and fitted.base_rate == pytest.approx(0.5)
    low, high = (expect.estimate(fitted, expect.Features(True, n, 7.0)) for n in (0, 5))
    assert low < 0.5 < high
    assert expect.fit(rows) == fitted  # deterministic
    with pytest.raises(ValueError, match="development"):
        expect.fit([*rows, replace(rows[0], split="held_out")])
    target = _thread()
    assert expect.fit_row(target, "understand", _at(1), "censored", "development") is None
    row = expect.fit_row(target, "understand", _at(1), "needed", "development")
    assert row == expect.FitRow(expect.Features(True, 2, 7.0), True, "development")


def test_demand_forecasting_brier_against_the_development_rate() -> None:
    needed = expect.expectation(_thread("t1"), "understand", _at(0), COEFFICIENTS)
    not_needed = expect.expectation(_thread("t2"), "understand", _at(1), COEFFICIENTS)
    nights = (_night(0, {"D0": "t1"}), _night(1, {"D0": "t2"}))
    life = _life("h1", nights, {"w1": 3}, {("t1", "w1")}, expectations=(needed, not_needed))
    demand = _report(life).demand
    p1, p2 = needed.p_needed, not_needed.p_needed
    assert demand.n == 2 and demand.censored == 0
    assert demand.brier == pytest.approx(((p1 - 1) ** 2 + p2 ** 2) / 2)
    assert demand.constant == pytest.approx(((0.25 - 1) ** 2 + 0.25 ** 2) / 2)


def test_contact_discipline_splits_by_whether_step_four_studied() -> None:
    unanchored = Note("t2", "post_result", anchored=False)
    anchored = Note("t1", "post_result", anchored=True)
    nights = (
        _night(0, {"D0": "t1"}, {"D0": anchored, "D1": unanchored}),
        _night(1, {"D0": None}, {"D0": None, "D1": unanchored}),
        _night(2, {"D0": None}, {"D0": None, "D1": None}),
    )
    life = _life("h1", nights, {}, set())
    d0, d1 = mind.contact(life, "D0"), mind.contact(life, "D1")
    assert d0.studied == mind.Rates(nights=1, notes=1, unanchored=0)
    assert d0.not_studied == mind.Rates(nights=2, notes=0, unanchored=0)
    assert d1.studied == mind.Rates(nights=1, notes=1, unanchored=1)
    assert d1.not_studied == mind.Rates(nights=2, notes=1, unanchored=1)


def test_wording_and_the_nights_both_studied() -> None:
    nights = (
        _night(0, {"D0": "t1", "D1": "t2", "D2": "t1"}),
        _night(1, {"D0": "t1", "D1": None}),
        _night(2, {"D0": None, "D1": "t1"}),
    )
    life = _life("h1", nights, {"w1": 4}, {("t1", "w1")})
    both = mind.both_studied(life)
    assert both == mind.Paired(d0=mind.Share(1, 1), d1=mind.Share(0, 1))
    report = _report(life, seed=7)
    text = mind.render(report)
    assert "On 67% of nights her state studied something relevant to work the synthetic person brought later; " \
           "a plain model given the same state, 33%" in text
    assert "stationary bootstrap" in text and "mean block 7" in text
    assert report == _report(life, seed=7)  # the interval is seeded
    bounds = report.pooled.anticipation["D0"].interval
    assert bounds is not None and 0.0 <= bounds[0] <= 2 / 3 <= bounds[1] <= 1.0


def test_stationary_bootstrap_keeps_constant_sequences() -> None:
    import random

    rng = random.Random(3)
    for n in (1, 5, 12):
        indices = mind.stationary_indices(n, 7.0, rng)
        assert len(indices) == n and all(0 <= i < n for i in indices)
    life = _life("h1", tuple(_night(day, {"D0": "t1"}) for day in range(4)), {"w1": 5}, {("t1", "w1")})
    assert _report(life).lives[0].anticipation["D0"].interval == (1.0, 1.0)
