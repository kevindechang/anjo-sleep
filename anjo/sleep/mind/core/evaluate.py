"""The core's offline evaluation, without the LLM (torch, CPU).

**The decision a day.** Each person-day the core observes the day's events and then decides, as
the night would: speak when its speak probability reaches the threshold, about the slot with the
top choice share, citing that slot's events. An intervention **catches** a failing thread when it
cites one of the thread's ``origin_ids`` on a day from the day the problem is knowable until the
day before the person raises it (APD's match rule); each thread is caught once, and a later
intervention on a thread already caught is a **repeat**. Any other intervention is **false**.

- ``headline``: recall of failing threads (overall and per family), and false interventions per
  100 person-days.
- ``lock_threshold``: the speak threshold with the best recall at no more than
  ``MAX_FALSE_PER_100`` false interventions per 100 person-days, on validation lives; the highest
  such threshold on ties. Test lives are then scored at it, untouched. ``lock_ignite`` locks the
  dream's own threshold the same way on the same lives: each night's ignitions
  (``dream_traces``, ``DREAM_STEPS`` steps, seeded) are scored as interventions. The two budgets
  are separate, so speak calls and ignitions together stay within twice ``MAX_FALSE_PER_100``.
- ``calibration``: Brier score and reliability bins of each open slot's risk against what happened
  to the thread it cites (it surfaced within the horizon), and of the self-error head against the
  risk's realized squared error.
- Ablations: ``reset_each_night`` starts each day from an empty state; a core trained with
  ``fast=False`` has no fast weights; ``memory=False`` is the no-memory baseline.
- ``transplant``: the counterfactual test. Two histories of one person differ only in how earlier
  bookings ended; the present is the same. With nothing but the state moved, the booking's slot
  must flip (speak about it after the history of cancellations, not after the deliveries), and the
  slot of an unrelated thread still open at the cut (a plan and a conflicting constraint, not yet
  raised: a thread able to fail), a slot citing neither the booking nor any event that differs
  between the histories, should keep its risk within ``tolerance`` on average; its mean risk and
  the largest shift are reported too, since the shared fast memory lets a little of the history
  reach every later write.
- ``separation``: the threshold-free AUC of each labelled thread's peak risk, problems against
  decoys.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np

from anjo.sleep.mind.core.api import Core
from anjo.sleep.mind.core.state import CoreState
from anjo.sleep.mind.core.toy import TransplantPair
from anjo.sleep.mind.core.types import EventLatent, LabelledStream, Predictions, ThreadLabel

MAX_FALSE_PER_100: Final = 1.0
BINS: Final = 10
DREAM_STEPS: Final = 8  # a night's dream, as the CLI's ``--dream`` runs it
LOCK_CANDIDATES: Final = 256  # thresholds tried when locking; quantiles of the calls beyond that


@dataclass(frozen=True)
class DayCall:
    day: date
    p_speak: float
    cites: tuple[str, ...]  # the chosen slot's, empty when no slot is open


@dataclass(frozen=True)
class SlotRecord:
    risk: float
    self_error: float
    outcome: float  # 1 when the thread the slot cites surfaced within the horizon


@dataclass(frozen=True)
class LifeTrace:
    life_id: str
    calls: tuple[DayCall, ...]
    slots: tuple[SlotRecord, ...]
    peaks: tuple[tuple[str, float], ...] = ()  # per label: the highest risk of a slot citing it while open


@dataclass(frozen=True)
class Headline:
    threshold: float
    recall: float
    false_per_100: float
    problems: int
    caught: int
    false: int
    repeats: int
    person_days: int
    by_family: tuple[tuple[str, int, int], ...] = ()  # (family, failing threads, caught)


@dataclass(frozen=True)
class Reliability:
    lower: float
    upper: float
    count: int
    predicted: float
    observed: float


@dataclass(frozen=True)
class Calibration:
    risk_brier: float
    risk_bins: tuple[Reliability, ...]
    self_error_mse: float
    self_error_bins: tuple[Reliability, ...]
    records: int


@dataclass(frozen=True)
class Transplant:
    pairs: int
    flipped: int
    relevant_gap: float  # mean risk of the booking's slot, cancellations minus deliveries
    mean_unrelated_shift: float
    max_unrelated_shift: float
    unrelated_found: int  # pairs whose unrelated thread holds a slot of its own in both histories
    tolerance: float
    mean_unrelated_risk: float = 0.0  # that slot's risk after the cancellations: how able to fail it looks


# ── Traces ────────────────────────────────────────────────────────────────────


def _surfaces(label: ThreadLabel, day: date, horizon: int) -> bool:
    return label.failed and label.knowable_at.date() <= day < label.outcome_at.date() <= date.fromordinal(
        day.toordinal() + horizon)


def _records(seen: Predictions, labels: Sequence[ThreadLabel], day: date, horizon: int) -> list[SlotRecord]:
    out = []
    for slot in seen.slots:
        if not slot.cites:
            continue
        cited = set(slot.cites)
        hit = any(_surfaces(label, day, horizon) and cited & set(label.event_ids) for label in labels)
        out.append(SlotRecord(slot.risk, slot.self_error, 1.0 if hit else 0.0))
    return out


def trace(core: Core, stream: LabelledStream, *, reset_each_night: bool = False) -> LifeTrace:
    state = core.init_state(stream.life_id)
    by_day: dict[date, list[EventLatent]] = {}
    for event in stream.events:
        by_day.setdefault(event.t.date(), []).append(event)
    calls, records = [], []
    peaks = {label.id: 0.0 for label in stream.labels}
    for day in stream.days:
        if reset_each_night:
            state = core.init_state(stream.life_id)
        for event in by_day.get(day, ()):
            state, _ = core.observe(state, event)
        seen = core.predict(state)
        decision = core.decide(state)
        cites = state.cited(decision.slot) if decision.slot is not None else ()
        calls.append(DayCall(day, seen.speak, cites))
        records.extend(_records(seen, stream.labels, day, core.config.horizon))
        for label in stream.labels:
            if label.knowable_at.date() <= day < label.outcome_at.date():
                held = [s.risk for s in seen.slots if set(s.cites) & set(label.event_ids)]
                peaks[label.id] = max([peaks[label.id], *held])
    return LifeTrace(stream.life_id, tuple(calls), tuple(records), tuple(peaks.items()))


def traces(core: Core, streams: Sequence[LabelledStream], *, reset_each_night: bool = False) -> tuple[LifeTrace, ...]:
    return tuple(trace(core, s, reset_each_night=reset_each_night) for s in streams)


# ── The headline ──────────────────────────────────────────────────────────────


_Matched = tuple[tuple[LabelledStream, tuple[tuple[DayCall, tuple[ThreadLabel, ...]], ...]], ...]


def _matched(runs: Sequence[LifeTrace], streams: Sequence[LabelledStream]) -> _Matched:
    """Each call with the failing threads it would catch (APD's match rule), whatever the threshold."""
    out = []
    for run, stream in zip(runs, streams, strict=True):
        failing = [label for label in stream.labels if label.failed]
        calls = tuple((call, tuple(label for label in failing if set(call.cites) & set(label.origin_ids)
                                   and label.knowable_at.date() <= call.day < label.outcome_at.date()))
                      for call in run.calls if call.cites)
        out.append((stream, calls))
    return tuple(out)


def _tally(matched: _Matched, threshold: float) -> Headline:
    caught = false = repeats = days = 0
    families: dict[str, list[int]] = {}
    for stream, calls in matched:
        days += len(stream.days)
        found: set[str] = set()
        for call, match in calls:
            if call.p_speak < threshold:
                continue
            fresh = [label for label in match if label.id not in found]
            if fresh:
                found.add(fresh[0].id)
                caught += 1
            elif match:
                repeats += 1
            else:
                false += 1
        for label in stream.labels:
            if label.failed:
                tally = families.setdefault(label.family, [0, 0])
                tally[0] += 1
                tally[1] += label.id in found
    problems = sum(n for n, _ in families.values())
    return Headline(
        threshold=threshold, recall=caught / problems if problems else 0.0,
        false_per_100=100.0 * false / days if days else 0.0, problems=problems, caught=caught, false=false,
        repeats=repeats, person_days=days, by_family=tuple((f, n, c) for f, (n, c) in sorted(families.items())),
    )


def score(runs: Sequence[LifeTrace], streams: Sequence[LabelledStream], threshold: float) -> Headline:
    return _tally(_matched(runs, streams), threshold)


def _lock(runs: Sequence[LifeTrace], val: Sequence[LabelledStream]) -> float:
    matched = _matched(runs, val)
    values: list[float] = sorted({call.p_speak for _, calls in matched for call, _ in calls})
    if len(values) > LOCK_CANDIDATES:
        cut: list[float] = np.quantile(values, np.linspace(0.0, 1.0, LOCK_CANDIDATES)).tolist()
        values = sorted(set(cut))
    best = (-1.0, 0.0)
    for threshold in [*values, 1.0 + 1e-6]:
        result = _tally(matched, threshold)
        if result.false_per_100 <= MAX_FALSE_PER_100 and (result.recall, threshold) > best:
            best = (result.recall, threshold)
    return best[1]


def lock_threshold(core: Core, val: Sequence[LabelledStream]) -> float:
    """The speak threshold locked on validation lives (see the module docstring)."""
    return _lock(traces(core, val), val)


def dream_traces(
    core: Core, streams: Sequence[LabelledStream], *, seed: int, steps: int = DREAM_STEPS,
) -> tuple[LifeTrace, ...]:
    """Each night's ignitions as calls (their score, their cites), every one of them (no threshold)."""
    rng = np.random.default_rng(seed)
    out = []
    for stream in streams:
        state, calls = core.init_state(stream.life_id), list[DayCall]()
        for day in stream.days:
            state = _run(core, state, [e for e in stream.events if e.t.date() == day])
            calls.extend(DayCall(day, i.score, i.cited_event_ids) for i in core.dream(state, steps, rng, ignite_at=0.0))
        out.append(LifeTrace(stream.life_id, tuple(calls), ()))
    return tuple(out)


def lock_ignite(core: Core, val: Sequence[LabelledStream], *, seed: int) -> float:
    """The dream's ignition threshold, locked on validation lives like the speak threshold."""
    return _lock(dream_traces(core, val, seed=seed), val)


def headline(
    core: Core, streams: Sequence[LabelledStream], threshold: float, *, reset_each_night: bool = False,
) -> Headline:
    return score(traces(core, streams, reset_each_night=reset_each_night), streams, threshold)


# ── Calibration ───────────────────────────────────────────────────────────────


def _bins(pairs: Sequence[tuple[float, float]], top: float) -> tuple[Reliability, ...]:
    out = []
    for n in range(BINS):
        lower, upper = top * n / BINS, top * (n + 1) / BINS
        inside = [(p, y) for p, y in pairs if lower <= p < upper or (n == BINS - 1 and p == upper)]
        if inside:
            out.append(Reliability(round(lower, 4), round(upper, 4), len(inside),
                                   math.fsum(p for p, _ in inside) / len(inside),
                                   math.fsum(y for _, y in inside) / len(inside)))
    return tuple(out)


def calibration(runs: Sequence[LifeTrace]) -> Calibration:
    records = [r for run in runs for r in run.slots]
    if not records:
        return Calibration(0.0, (), 0.0, (), 0)
    risk = [(r.risk, r.outcome) for r in records]
    errors = [(r.self_error, (r.outcome - r.risk) ** 2) for r in records]
    top = max(1.0, max(p for p, _ in errors))
    return Calibration(
        risk_brier=math.fsum((p - y) ** 2 for p, y in risk) / len(risk), risk_bins=_bins(risk, 1.0),
        self_error_mse=math.fsum((p - y) ** 2 for p, y in errors) / len(errors),
        self_error_bins=_bins(errors, top), records=len(records),
    )


# ── The counterfactual transplant ─────────────────────────────────────────────


def separation(
    runs: Sequence[LifeTrace], streams: Sequence[LabelledStream], select: Callable[[ThreadLabel], bool] = bool,
) -> float | None:
    """How well the risk ranks problems above their decoys, threshold-free: the chance a failing
    thread's peak risk while open beats a non-failing one's (ties count half; AUC). ``None`` when
    either side is empty."""
    labels = {label.id: label for stream in streams for label in stream.labels if select(label)}
    peaks = [(peak, labels[name].failed) for run in runs for name, peak in run.peaks if name in labels]
    pos, neg = [p for p, failed in peaks if failed], [p for p, failed in peaks if not failed]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _run(core: Core, state: CoreState, events: Sequence[EventLatent]) -> CoreState:
    for event in events:
        state, _ = core.observe(state, event)
    return state


def _risk_citing(core: Core, state: CoreState, ids: Sequence[str], excluding: Sequence[str] = ()) -> float | None:
    wanted, unwanted = set(ids), set(excluding)
    held = [s for s in core.predict(state).slots if wanted & set(s.cites) and not unwanted & set(s.cites)]
    return max(s.risk for s in held) if held else None


def transplant(
    core: Core, pairs: Sequence[TransplantPair], *, tolerance: float, threshold: float | None = None,
) -> Transplant:
    line = core.threshold if threshold is None else threshold
    flipped, gaps, shifts, risks = 0, [], [], []
    for pair in pairs:
        cancelled = _run(core, _run(core, core.init_state(pair.person), pair.problem), pair.now)
        delivered = _run(core, _run(core, core.init_state(pair.person), pair.benign), pair.now)

        def raises(state: CoreState) -> bool:
            decision = core.decide(state)
            return (decision.p_speak >= line and decision.slot is not None
                    and pair.relevant in state.cited(decision.slot))

        flipped += int(raises(cancelled) and not raises(delivered))
        high, low = _risk_citing(core, cancelled, (pair.relevant,)), _risk_citing(core, delivered, (pair.relevant,))
        gaps.append((high or 0.0) - (low or 0.0))
        moved = (pair.relevant, *pair.moved)
        a = _risk_citing(core, cancelled, pair.unrelated, excluding=moved)
        b = _risk_citing(core, delivered, pair.unrelated, excluding=moved)
        if a is not None and b is not None:
            shifts.append(abs(a - b))
            risks.append(a)
    return Transplant(
        pairs=len(pairs), flipped=flipped, relevant_gap=math.fsum(gaps) / max(len(gaps), 1),
        mean_unrelated_shift=math.fsum(shifts) / max(len(shifts), 1), max_unrelated_shift=max(shifts, default=0.0),
        unrelated_found=len(shifts), tolerance=tolerance, mean_unrelated_risk=math.fsum(risks) / max(len(risks), 1),
    )
