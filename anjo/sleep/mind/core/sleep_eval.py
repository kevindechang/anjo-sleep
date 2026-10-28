"""MIND's nights, measured offline on generated lives (torch, CPU): does what her dreams pick make
the better rehearsal, and does consolidate-then-reset beat a fast memory that is never reset?

**Rehearsal.** Each night of each life, after the day's events, the same state is rehearsed three
ways at the same budget (``budget`` events replayed into the fast memory, ``Core.rehearse``):

- ``dream``: the curriculum's picks (``curriculum.pick``) from the core's predictions, that
  night's dream ignitions and the day's surprise weights;
- ``recent``: the newest events she has seen (today's practice seeds on the newest evidence);
- ``random``: events drawn uniformly from all she has seen;
- ``boundary``: the real events under the dream's decision boundaries (``Dream.boundaries``: where
  an imagined turn first flips her decision, fewest imagined steps first), topped up with the
  dream's picks: where she is fragile, before where she is at risk. It costs the same one dream.

Then the next day's events are observed from each rehearsed state, and from the un-rehearsed one
(``none``). The score is the surprise of the next day's events: all of them, and those of labelled
threads (``needed``: the threads that will fail or turn out fine, what she will need). Arms are
paired per life-night; each difference is reported with a bootstrap 95% interval over life-nights.
The life then goes on from the un-rehearsed state, so no arm's past leaks into another's.

**Ablations** (``ablations``), on the same lives, nights and budget, each night rehearsing then
consolidating: ``full`` (the dream's picks, shared by surprise), ``no_curriculum`` (the newest
events), ``no_sleep_link`` (the dream's picks, shared equally), ``core_reset`` (a fresh state each
day), and ``no_rehearsal`` (consolidation alone: the core's side of the live night, where
rehearsal trains Glimmer, not the core). They run here, on generated lives, never on a real
person's nights. Rehearsing into the core's own memory is a proxy: what the live night rehearses
trains Glimmer's adapter, which only a GPU run measures.

**Consolidation.** Long lives (``toy.chained``: several toy lives lived by one person, many
nights) run day by day with one operation each night (``evaluate.trace(nightly=...)``): ``never``
(the fast memory is never reset), ``reset`` (the fast memory is cleared, nothing kept),
``consolidate`` (``Core.consolidate``: the fast memory moves into the slow one, then resets). The
score is recall and false calls at the locked threshold, the planted problems' AUC
(``evaluate.separation``) and the risk's Brier score (``evaluate.calibration``).

**Measured** with the toy recipe ``cli`` trains (seeds 0 / 1 / 2), each choice made on validation
lives (``toy.lives(64, seed=2)``, ``toy.chained(seed=2)``) and test lives (seed 3) reported once:

- The dream's picks do not beat random rehearsal: dream minus random on the needed threads'
  next-day surprise is -0.002 / -0.002 / -0.006 on validation and -0.008 / +0.019 / +0.002 on
  test, every interval across zero. They do beat the newest events (what practice seeded on
  before), +0.062 / +0.054 / +0.020 on validation and +0.061 / +0.065 / +0.042 on test, every
  interval above zero, so the curriculum stays on (``night.CURRICULUM_ENV``).
- Rehearsing into the core's own memory is not chosen: problems caught on long lives, of 27 on
  validation, ``full`` 7 / 8 / 4 and ``no_rehearsal`` 11 / 14 / 10; on test, of 31, 9 / 9 / 5 and
  17 / 13 / 10. So the core's night consolidates only (``night``).
- Consolidating each night matches never resetting (validation, of 48: 22 / 18 / 13 against
  20 / 18 / 15; test, of 60: 31 / 22 / 17 against 31 / 23 / 19) and resetting loses nearly every
  problem (test 1 / 3 / 0). Nothing saturates over these lives, so the slow memory has no decay.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Final

import numpy as np

from anjo.sleep.mind.core import curriculum, evaluate
from anjo.sleep.mind.core.api import Core
from anjo.sleep.mind.core.evaluate import NightOp
from anjo.sleep.mind.core.state import CoreState
from anjo.sleep.mind.core.types import EventLatent, LabelledStream, days_of

ARMS: Final = ("none", "dream", "recent", "random", "boundary")
PRESSURE_SKIP: Final = 0.5  # the sampled lives skip a night's consolidation this often, so days pile up
REHEARSAL_BUDGET: Final = 8
BOOTSTRAP: Final = 2000
ABLATIONS: Final = ("full", "no_curriculum", "no_sleep_link", "core_reset", "no_rehearsal")


@dataclass(frozen=True)
class NightSample:
    """One life-night: the next day's mean surprise under each arm, over all its events and the
    labelled threads' (``None`` when the next day holds none)."""

    life_id: str
    night: date
    rehearsed: int  # events each rehearsing arm replayed
    all: Mapping[str, float]
    needed: Mapping[str, float | None]


@dataclass(frozen=True)
class Gain:
    """``arm`` against ``base``: the mean drop in next-day surprise (positive: ``arm`` predicts
    better), its bootstrap 95% interval, over ``n`` life-nights."""

    arm: str
    base: str
    scope: str
    mean: float
    low: float
    high: float
    n: int


def _surprises(core: Core, state: CoreState, events: Sequence[EventLatent]) -> list[float]:
    out = []
    for event in events:
        state, surprise = core.observe(state, event)
        out.append(surprise)
    return out


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _chosen(
    core: Core, state: CoreState, seen: Sequence[EventLatent], weights: Mapping[str, float], budget: int,
    rng: np.random.Generator,
) -> dict[str, list[EventLatent]]:
    by_id = {e.event_id: e for e in seen}
    dreamt = core.dream_states(state, evaluate.DREAM_STEPS, rng)
    picks = curriculum.pick(core.predict(state), dreamt.ignitions, weights, budget=budget)
    dream = [by_id[i] for i in picks.evidence_ids if i in by_id]
    n = len(dream)  # every arm replays as many events as the dream picked
    fragile = [i for b in sorted(dreamt.boundaries, key=lambda b: b.steps) for i in b.dream.rests_on]
    edge = [by_id[i] for i in dict.fromkeys([*fragile, *picks.evidence_ids]) if i in by_id][:n]
    drawn = sorted(int(i) for i in rng.choice(len(seen), size=min(n, len(seen)), replace=False))
    return {"dream": sorted(dream, key=lambda e: e.t), "recent": list(seen[-n:]) if n else [],
            "random": [seen[i] for i in drawn], "boundary": sorted(edge, key=lambda e: e.t)}


def night_samples(core: Core, stream: LabelledStream, *, budget: int, seed: int) -> list[NightSample]:
    rng = np.random.default_rng(seed)
    needed = {eid for label in stream.labels for eid in label.event_ids}
    by_day: dict[date, list[EventLatent]] = {}
    for event in stream.events:
        by_day.setdefault(event.t.date(), []).append(event)
    state, seen, out = core.init_state(stream.life_id), list[EventLatent](), []
    for day, after in zip(stream.days, stream.days[1:]):
        weights: dict[str, float] = {}
        if by_day.get(day):
            state, weights = core.surprise_weights(state, by_day[day])
            seen.extend(by_day[day])
        upcoming = by_day.get(after, [])
        if not seen or not upcoming:
            continue
        chosen = _chosen(core, state, seen, weights, budget, rng)
        if not chosen["dream"]:
            continue
        states = {"none": state, **{arm: core.rehearse(state, events) for arm, events in chosen.items()}}
        all_, need = {}, {}
        for arm, start in states.items():
            surprise = _surprises(core, start, upcoming)
            all_[arm] = math.fsum(surprise) / len(surprise)
            need[arm] = _mean([s for s, e in zip(surprise, upcoming, strict=True) if e.event_id in needed])
        out.append(NightSample(stream.life_id, day, len(chosen["dream"]), all_, need))
    return out


def gains(samples: Sequence[NightSample], *, seed: int, base: str = "random") -> tuple[Gain, ...]:
    """Each rehearsing arm against ``base`` (and against ``none``), over all events and the needed ones."""
    rng = np.random.default_rng(seed)
    out = []
    for scope in ("all", "needed"):
        for arm, other in (("dream", base), ("dream", "recent"), ("dream", "none"), ("recent", "none"),
                           ("random", "none"), ("boundary", base), ("boundary", "dream"), ("boundary", "none")):
            if arm == other:
                continue
            diffs = []
            for s in samples:
                values = s.all if scope == "all" else s.needed
                a, b = values[arm], values[other]
                if a is not None and b is not None:
                    diffs.append(b - a)
            if not diffs:
                continue
            arr = np.array(diffs)
            boots = np.array([rng.choice(arr, size=len(arr)).mean() for _ in range(BOOTSTRAP)])
            out.append(Gain(arm, other, scope, float(arr.mean()), float(np.quantile(boots, 0.025)),
                            float(np.quantile(boots, 0.975)), len(diffs)))
    return tuple(out)


def rehearsal_gain(
    core: Core, streams: Sequence[LabelledStream], *, budget: int = REHEARSAL_BUDGET, seed: int = 0,
) -> tuple[tuple[NightSample, ...], tuple[Gain, ...]]:
    samples = [s for n, stream in enumerate(streams) for s in night_samples(core, stream, budget=budget, seed=seed + n)]
    return tuple(samples), gains(samples, seed=seed)


@dataclass(frozen=True)
class ArmScore:
    arm: str
    recall: float
    false_per_100: float
    auc: float | None
    brier: float
    caught: int
    problems: int


def night_ops(core: Core, *, rank: int | None = None, residue: float = 0.0) -> dict[str, NightOp | None]:
    """The consolidation arms: never reset, reset without consolidating, consolidate then reset."""
    def reset(state: CoreState, weights: Mapping[str, float]) -> CoreState:
        return replace(state, memory=(residue * np.asarray(state.memory)).astype(np.float32),
                       momentum=np.zeros_like(state.momentum))

    return {"never": None, "reset": reset,
            "consolidate": lambda state, weights: core.consolidate(state, rank=rank, residue=residue)}


def _scored(core: Core, arm: str, runs: Sequence[evaluate.LifeTrace], streams: Sequence[LabelledStream]) -> ArmScore:
    head = evaluate.score(runs, streams, core.threshold)
    return ArmScore(arm, head.recall, head.false_per_100, evaluate.separation(runs, streams),
                    evaluate.calibration(runs).risk_brier, head.caught, head.problems)


def consolidation(
    core: Core, streams: Sequence[LabelledStream], ops: Mapping[str, NightOp | None],
) -> tuple[ArmScore, ...]:
    return tuple(_scored(core, arm, evaluate.traces(core, streams, nightly=op), streams) for arm, op in ops.items())


# ── The night's ablations, offline ───────────────────────────────────────────


def _night(
    core: Core, stream: LabelledStream, arm: str, *, budget: int, rng: np.random.Generator,
) -> NightOp:
    """One arm's night over one life: rehearse (the dream's picks, or the newest events), then
    consolidate. ``no_sleep_link`` shares the budget equally instead of by surprise."""
    by_id = {e.event_id: e for e in stream.events}

    def night(state: CoreState, weights: Mapping[str, float]) -> CoreState:
        seen = [e for e in stream.events if state.now is not None and days_of(e.t) <= state.now + 1e-9]
        if arm == "no_rehearsal":
            chosen = []
        elif arm == "no_curriculum":
            chosen = seen[-budget:]
        else:
            picks = curriculum.pick(core.predict(state), core.dream(state, evaluate.DREAM_STEPS, rng),
                                    {} if arm == "no_sleep_link" else weights, budget=budget)
            chosen = sorted((by_id[i] for i in picks.evidence_ids if i in by_id), key=lambda e: e.t)
        return core.consolidate(core.rehearse(state, chosen))

    return night


def ablations(
    core: Core, streams: Sequence[LabelledStream], *, budget: int = REHEARSAL_BUDGET, seed: int = 0,
) -> tuple[ArmScore, ...]:
    """The full night against each ablation, on the same lives and budget (``ABLATIONS``)."""
    out = []
    for arm in ABLATIONS:
        rng = np.random.default_rng(seed)
        runs = tuple(
            evaluate.trace(core, stream, reset_each_night=True) if arm == "core_reset"
            else evaluate.trace(core, stream, nightly=_night(core, stream, arm, budget=budget, rng=rng))
            for stream in streams
        )
        out.append(_scored(core, arm, runs, streams))
    return tuple(out)


# ── Sleep pressure ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PressureSample:
    """One night: what pressure reads, the days since the last consolidation, and what skipping
    tonight's consolidation costs (the next ``horizon`` days' mean surprise, skipped minus done)."""

    features: tuple[float, float, float]
    days: int
    cost: float


def _days(stream: LabelledStream) -> list[list[EventLatent]]:
    by_day: dict[date, list[EventLatent]] = {}
    for event in stream.events:
        by_day.setdefault(event.t.date(), []).append(event)
    return [by_day.get(day, []) for day in stream.days]


def _mean_surprise(core: Core, state: CoreState, days: Sequence[Sequence[EventLatent]]) -> float | None:
    found = _surprises(core, state, [e for day in days for e in day])
    return _mean(found)


def pressure_samples(core: Core, streams: Sequence[LabelledStream], *, horizon: int = 3, seed: int = 0) -> list[PressureSample]:
    """Each night of each life, with the life consolidating on a random half of its nights."""
    rng = np.random.default_rng(seed)
    out = []
    for stream in streams:
        days = _days(stream)
        state, since = core.init_state(stream.life_id), 0
        for n, events in enumerate(days[:-1]):
            for event in events:
                state, _ = core.observe(state, event)
            since += 1
            ahead = days[n + 1:n + 1 + horizon]
            slept = core.consolidate(state)
            skipped, done = _mean_surprise(core, state, ahead), _mean_surprise(core, slept, ahead)
            if skipped is not None and done is not None and state.observed:
                surprise, load, risk = core.pressure_features(state)
                out.append(PressureSample((surprise, load, risk), since, skipped - done))
            if rng.random() >= PRESSURE_SKIP:
                state, since = slept, 0
    return out


def fit_pressure(samples: Sequence[PressureSample]) -> tuple[tuple[float, float, float], dict[str, float]]:
    """Non-negative weights on the three features that predict what skipping costs, scaled so an
    ordinary day's cost (the mean over nights one day after a consolidation) is 1.0. Zero weights
    when skipping costs nothing on an ordinary day or the fit explains nothing (R^2 <= 0): then
    pressure has nothing to measure."""
    x = np.array([s.features for s in samples], dtype=np.float64)
    y = np.array([s.cost for s in samples], dtype=np.float64)
    active = list(range(x.shape[1]))
    coef = np.zeros(x.shape[1])
    while active:
        fit, *_ = np.linalg.lstsq(x[:, active], y, rcond=None)
        if (fit >= 0).all():
            coef[active] = fit
            break
        active = [a for a, c in zip(active, fit, strict=True) if c > 0] if (fit > 0).any() else []
    predicted = x @ coef
    total = float(((y - y.mean()) ** 2).sum())
    one_day = [p for p, s in zip(predicted, samples, strict=True) if s.days == 1]
    day_cost = float(np.mean(one_day)) if one_day else 0.0
    r2 = 1.0 - float(((y - predicted) ** 2).sum()) / total if total > 0 else 0.0
    weights = tuple(float(c / day_cost) for c in coef) if day_cost > 0 and r2 > 0 else (0.0, 0.0, 0.0)
    stats = {"r2": r2, "samples": float(len(samples)), "day_cost": day_cost, "mean_cost": float(y.mean())}
    return (weights[0], weights[1], weights[2]), stats
