"""Her expectations (Task 16): how likely she thinks the person is to need a study before its horizon.

Engine core, and pure: every time is passed in, on the night's clock (virtual in accelerated runs).

- **The horizon** is the target's ``check_after`` or seven days, whichever comes first, whenever the
  target has a date. A date at or before the moment she chose ends the horizon there: the study
  resolves ``not_needed`` at once and nothing later can make it needed.
- **The estimator** is a logistic of the want kind (``follow_up`` or ``understand``), the target's
  unacted revisions and the signed days to its ``check_after``, unclipped. A target with no date (a
  thread, or a plan with none) reads ``undated_days``, which is part of the registered
  ``Coefficients``, so the fill is fixed with them. ``fit`` fits it on development lives only; its
  coefficients are registered before the freeze.
- **Resolution comes from the person's side only.** ``needed`` when, inside ``[made_at, horizon)``,
  the person's own evidence returns to the target or a held-out item the frozen relevance labels mark
  relevant arrives; ``not_needed`` once the horizon has passed without either; ``censored`` whenever
  the horizon runs past the life's last day, whatever happened before it, so early hits can't bias
  the score. ``resolve`` never sees the target, so no date of its can resolve anything.

An expectation is about the usefulness of her own action. Its fields name a target id, two times and
a probability; nothing here predicts the person's traits or behavior beyond whether a topic comes back.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal

from anjo.sleep.types import Evidence, Expectation, ExpectationOutcome, Target, WantKind

HORIZON: Final = timedelta(days=7)
THREAD_DAYS: Final = 7.0  # the default registered fill for a target with no date
STUDY_KINDS: Final = frozenset({"follow_up", "understand"})  # the wants a study comes from
# Evidence from the person: their turns, or their own stored corrections and preferences.
PERSON_KINDS: Final = frozenset({"lesson", "correction", "preference", "tone_request"})
RIDGE: Final = 1.0  # the fit's L2 penalty on every term but the intercept
MAX_ITERATIONS: Final = 100
TOLERANCE: Final = 1e-10

Split = Literal["development", "held_out"]


@dataclass(frozen=True)
class Coefficients:
    """The registered estimator: log-odds = ``intercept`` + ``understand`` (for an ``understand``
    want; 0 for ``follow_up``) + ``unacted`` x unacted revisions + ``days`` x signed days to its
    date, or x ``undated_days`` for a target with none. ``base_rate`` is the development lives'
    share of resolved expectations that were needed (the constant-rate forecast) and ``n`` how many
    resolved expectations the fit saw."""

    intercept: float
    understand: float
    unacted: float
    days: float
    base_rate: float
    n: int
    undated_days: float = THREAD_DAYS

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_rate <= 1.0:
            raise ValueError(f"base_rate must be in [0, 1], got {self.base_rate}")


@dataclass(frozen=True)
class Features:
    """What the estimator reads: about her study and its target, never about the person. ``days``
    is ``None`` for a target with no date; the coefficients say what that reads as."""

    understand: bool
    unacted: int
    days: float | None


@dataclass(frozen=True)
class FitRow:
    """One resolved expectation of a development life (censored ones are left out)."""

    features: Features
    needed: bool
    split: Split


def days_to_check(target: Target, made_at: datetime) -> float | None:
    """Signed days from ``made_at`` to the target's date (negative once it has passed), or ``None``
    when it has none."""
    if target.check_after is None:
        return None
    return (target.check_after - made_at) / timedelta(days=1)


def features(kind: WantKind, target: Target, made_at: datetime) -> Features:
    if kind not in STUDY_KINDS:
        raise ValueError(f"an expectation is made for a study, not a {kind} want")
    return Features(understand=kind == "understand", unacted=target.unacted, days=days_to_check(target, made_at))


def horizon(target: Target, made_at: datetime) -> datetime:
    """The target's date or seven days, whichever comes first. A date at or before ``made_at`` ends
    the horizon there, so the expectation resolves at once and never as needed."""
    week = made_at + HORIZON
    if target.check_after is None:
        return week
    return min(target.check_after, week)


def estimate(coefficients: Coefficients, features: Features) -> float:
    """``p_needed``: the logistic of the registered coefficients over ``features``."""
    return _logistic(_log_odds(coefficients, features))


def expectation_id(target_id: str, made_at: datetime) -> str:
    digest = hashlib.sha256(f"{target_id}\x1f{made_at.isoformat()}".encode()).hexdigest()
    return f"x-{digest[:16]}"


def expectation(target: Target, kind: WantKind, made_at: datetime, coefficients: Coefficients) -> Expectation:
    """The expectation she records when she chooses to study ``target`` for a ``kind`` want."""
    return Expectation(
        id=expectation_id(target.id, made_at), target_id=target.id, made_at=made_at,
        horizon=horizon(target, made_at), p_needed=estimate(coefficients, features(kind, target, made_at)),
    )


def returning(evidence: Iterable[Evidence], target_id: str) -> tuple[Evidence, ...]:
    """The person's own items on ``target_id``: never her homework or practice, never a held-out item."""
    return tuple(
        item for item in evidence
        if item.target_id == target_id and item.kind in PERSON_KINDS and not item.held_out
    )


def resolve(
    made: Expectation, *, returns: Sequence[datetime], arrivals: Sequence[datetime], life_end: datetime,
    as_of: datetime,
) -> ExpectationOutcome | None:
    """``made``'s outcome at ``as_of``, or ``None`` while it is still open. ``returns`` are when the
    person's own evidence came back to its target; ``arrivals`` are when held-out items labeled
    relevant to it arrived; ``life_end`` is the end of the life's last day."""
    if made.horizon > life_end:
        return "censored"
    if any(made.made_at <= moment < made.horizon and moment <= as_of for moment in (*returns, *arrivals)):
        return "needed"
    return "not_needed" if as_of >= made.horizon else None


def fit_row(
    target: Target, kind: WantKind, made_at: datetime, outcome: ExpectationOutcome | None, split: Split,
) -> FitRow | None:
    """The fit's row for a study she chose, or ``None`` when its expectation is censored or open."""
    if outcome not in ("needed", "not_needed"):
        return None
    return FitRow(features(kind, target, made_at), outcome == "needed", split)


def fit(rows: Sequence[FitRow], *, ridge: float = RIDGE, undated_days: float = THREAD_DAYS) -> Coefficients:
    """Fit the estimator on development lives' resolved expectations (Newton's method with an L2
    penalty, so it stays finite when one outcome dominates). A held-out row is refused. An undated
    row reads ``undated_days``, which the coefficients keep."""
    if not rows:
        raise ValueError("the estimator needs at least one resolved development expectation")
    if any(row.split != "development" for row in rows):
        raise ValueError("the estimator is fitted on development lives only")
    weights = [0.0, 0.0, 0.0, 0.0]
    for _ in range(MAX_ITERATIONS):
        step = _newton_step(rows, weights, ridge, undated_days)
        weights = [w - s for w, s in zip(weights, step, strict=True)]
        if max(abs(s) for s in step) < TOLERANCE:
            break
    intercept, understand, unacted, days = weights
    needed = sum(row.needed for row in rows)
    return Coefficients(intercept, understand, unacted, days, base_rate=needed / len(rows), n=len(rows),
                        undated_days=undated_days)


def _vector(features: Features, undated_days: float) -> tuple[float, float, float, float]:
    days = undated_days if features.days is None else features.days
    return 1.0, 1.0 if features.understand else 0.0, float(features.unacted), days


def _log_odds(coefficients: Coefficients, features: Features) -> float:
    terms = (coefficients.intercept, coefficients.understand, coefficients.unacted, coefficients.days)
    return math.fsum(w * x for w, x in zip(terms, _vector(features, coefficients.undated_days), strict=True))


def _logistic(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    tail = math.exp(z)
    return tail / (1.0 + tail)


def _newton_step(rows: Sequence[FitRow], weights: list[float], ridge: float, undated_days: float) -> list[float]:
    """H^-1 g for the penalized log-likelihood (H negative definite), so ``w - step`` climbs it."""
    size = len(weights)
    gradient = [-ridge * w if k else 0.0 for k, w in enumerate(weights)]
    hessian = [[-ridge if (k == j and k) else 0.0 for j in range(size)] for k in range(size)]
    for row in rows:
        x = _vector(row.features, undated_days)
        p = _logistic(math.fsum(w * v for w, v in zip(weights, x, strict=True)))
        for k in range(size):
            gradient[k] += (float(row.needed) - p) * x[k]
            for j in range(size):
                hessian[k][j] -= p * (1.0 - p) * x[k] * x[j]
    return _solve(hessian, gradient)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting on copies of ``matrix`` and ``vector``."""
    size = len(vector)
    rows = [[*matrix[k], vector[k]] for k in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-15:
            raise ValueError("the fit's system is singular: add development rows")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, size):
            factor = rows[r][col] / rows[col][col]
            rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col], strict=True)]
    solution = [0.0] * size
    for r in range(size - 1, -1, -1):
        known = math.fsum(rows[r][c] * solution[c] for c in range(r + 1, size))
        solution[r] = (rows[r][size] - known) / rows[r][r]
    return solution
