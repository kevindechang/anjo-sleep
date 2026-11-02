"""The rival predictors of MIND's cognition sprint (DESIGN.md, "Rivals").

- ``PopulationPrior``: P(happens | query family, kind, activity stem, target weekday, horizon)
  from train lives, backed off level by level to the global rate. ``Query.family`` is part of the
  query (types.py), so every rival may use it; it says whether the activity is a steady one.
- ``Counter``: a per-person Beta-Binomial: the activity's did / (did + skipped + cancelled) shrunk
  to the population prior, then its weekday's shrunk to that (``CounterSpec``). A plan's
  occurrences are its plan days, so for a plan this is the activity's own kept-rate.
- ``OnlineLogistic``: a per-person Bayesian logistic regression on ``rival_features``, its prior
  the population fit on train lives (standardized features, an L2 penalty, the prior table
  cross-fitted so no train life sees its own outcomes in it), updated online by a Laplace
  (extended Kalman) step per outcome, with prior variance only where a person may differ. ``u``
  is the predictive std of the logit. The strongest simple rival.
- ``SurpriseLR``: ``OnlineLogistic`` whose learning rate rises with surprise (T4's rival).

All read the life through ``PersonHistory`` (rival_history.py): event fields and tokens, never
text. An outcome is a ``did`` (1) or a ``skipped``/``cancel`` (0); a day with neither is no
occurrence, not a miss. Labels for online learning come only from the person's own later events,
never from ``resolve``; ``fit_population`` may use ``resolve`` because train lives are training data.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from anjo.sleep.mind.cog.rival_features import (
    P_MAX,
    P_MIN,
    CounterSpec,
    FeatureSpec,
    PriorKey,
    PriorTable,
    RawFeatures,
    assemble,
    counter_estimate,
    features,
    logit,
    prior_keys,
    raw_features,
    select_keys,
    sigmoid,
)
from anjo.sleep.mind.cog.rival_history import PersonHistory, is_late, is_plan_query, query_kind, stem, token_ints
from anjo.sleep.mind.cog.types import Event, Life, LifeMeta, Prediction, Query

__all__ = [
    "Counter", "CounterSpec", "FeatureSpec", "LogisticPopulation", "OnlineLogistic", "PersonHistory",
    "PopulationPrior", "PriorTable", "SurpriseLR", "features", "is_late", "is_plan_query", "query_kind",
    "token_ints",
]


def _clip(p: float) -> float:
    return min(max(p, P_MIN), P_MAX)


# -- the population prior --
Cells = dict[PriorKey, list[float]]


def _life_cells(life: Life) -> Cells:
    cells: Cells = defaultdict(lambda: [0.0, 0.0])
    hist = PersonHistory(life.meta)
    for d in range(life.meta.n_days):
        hist.add(life.events(d))
        for q in life.queries(d):
            y = float(life.resolve(q.qid).outcome)
            for key in prior_keys(q.family, query_kind(q, hist, d), hist.stem(q.subject), q.target_day - d):
                cells[key][0] += y
                cells[key][1] += 1.0
    return cells


def _table(cells: Iterable[Cells], smoothing: float, minus: Iterable[Cells] = ()) -> PriorTable:
    total: Cells = defaultdict(lambda: [0.0, 0.0])
    for sign, group in ((1.0, cells), (-1.0, minus)):
        for c in group:
            for key, (k, n) in c.items():
                total[key][0] += sign * k
                total[key][1] += sign * n
    return PriorTable({key: (k, n) for key, (k, n) in total.items() if n > 0.5}, smoothing)


class PopulationPrior:
    name = "population"

    def __init__(self, table: PriorTable) -> None:
        self.table = table

    @classmethod
    def fit_population(cls, train_lives: Iterable[Life], *, smoothing: float = 10.0) -> PriorTable:
        return _table([_life_cells(life) for life in train_lives], smoothing)

    def begin(self, meta: LifeMeta) -> None:
        self.hist = PersonHistory(meta)

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.hist.add(events)

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        return [Prediction(q.qid, _clip(self.table.p(
            q.family, query_kind(q, self.hist, q.asked_day), self.hist.stem(q.subject),
            q.target_day - q.asked_day))) for q in queries]

    def end_day(self, day: int) -> None: ...


# -- the per-person counter --
class Counter:
    name = "counter"

    def __init__(self, prior: PriorTable | None = None, *, strength: float = 6.0, count_strength: float = 100.0,
                 weekday_strength: float = 100.0) -> None:
        self.prior = prior
        self.spec = CounterSpec(strength, count_strength, weekday_strength)

    def begin(self, meta: LifeMeta) -> None:
        self.hist = PersonHistory(meta)

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.hist.add(events)

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        out = []
        for q in queries:
            d = q.asked_day
            p, n = counter_estimate(self.spec, self.prior, self.hist, q.family, query_kind(q, self.hist, d),
                                    q.subject, d, q.target_day)
            s = self.spec.count_strength if q.family == "count" else self.spec.strength
            out.append(Prediction(q.qid, _clip(p), u=math.sqrt(p * (1 - p) / (n + s + 1))))
        return out

    def end_day(self, day: int) -> None: ...


# -- the population fit of the logistic rivals --
@dataclass(frozen=True, eq=False)
class LogisticPopulation:
    """The population fit: weights on standardized features, and the standardization."""
    spec: FeatureSpec
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    n_examples: int = 0


def _fit_logistic(z: np.ndarray, y: np.ndarray, l2: float, iters: int = 50) -> np.ndarray:
    """L2-penalized logistic regression by Newton's method (the bias is not penalized)."""
    w = np.zeros(z.shape[1])
    penalty = np.full(z.shape[1], l2)
    penalty[0] = 1e-6
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
        grad = z.T @ (p - y) + penalty * w
        hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    out: np.ndarray = w
    return out


def _replay(life: Life, spec: FeatureSpec, ask_rate: float, rng: np.random.Generator) -> tuple[list[RawFeatures], list[float]]:
    """A train life's queries as (features, outcome), with random asks so answers get weights."""
    hist, rows, ys = PersonHistory(life.meta), [], []
    for d in range(life.meta.n_days):
        hist.add(life.events(d))
        questions = life.questions(d)
        if questions and rng.random() < ask_rate:
            hist.add([life.answer(questions[int(rng.integers(len(questions)))].question_id)])
        queries = life.queries(d)
        hist.note_queries(queries)
        for q in queries:
            rows.append(raw_features(spec, hist, q.subject, d, q.target_day, query_kind(q, hist, d), q.family))
            ys.append(float(life.resolve(q.qid).outcome))
    return rows, ys


def _mood_center(lives: Sequence[Life]) -> float:
    vals = [e.value for life in lives for d in range(life.meta.n_days) for e in life.events(d)
            if e.kind == "mood" and e.value is not None]
    return float(np.mean(vals)) if vals else 3.0


# -- the per-person online logistic regression --
class OnlineLogistic:
    """Per-person Bayesian logistic regression, its prior centred on the population weights.

    The prior is structured: variance ``prior_var`` on the weights that may be personal
    (``FeatureSpec.personal``) and ``shared_var`` on the rest, so a few weeks of one person's
    outcomes move what can be idiosyncratic and leave the population's reading of signals alone.

    Each night every activity's outcome that day (a did, skipped or cancel) is one training
    example, with the features of "will it happen tomorrow?" asked the day before (one per
    occurrence: the queries that targeted it would add correlated copies). The update is one
    Laplace step per example (Sherman-Morrison on the covariance). p is the probit-approximate
    predictive mean. ``u`` is the predictive std of the logit: the weights' posterior variance plus
    the stacked Counter estimate's own, ``w_c^2 / (n_eff p (1 - p))`` with ``w_c`` its weight and
    ``n_eff`` its occurrences plus pseudo-counts (the fit already priced that noise into p, so it
    enters u alone). ``predict`` changes no prediction, so an inquirer may call it; it notes the
    family of each subject it is asked about.
    """
    name = "online_lr"

    def __init__(self, population: LogisticPopulation | None = None, *, prior_var: float = 1e-3,
                 shared_var: float = 1e-6, name: str | None = None) -> None:
        self.population, self.prior_var, self.shared_var = population, prior_var, shared_var
        if name is not None:
            self.name = name

    @classmethod
    def fit_population(cls, train_lives: Iterable[Life], *, l2: float = 1.0, ask_rate: float = 0.25,
                       smoothing: float = 10.0, strength: float = 6.0, count_strength: float = 100.0,
                       weekday_strength: float = 100.0, folds: int = 5, z_min: float = 3.0, min_groups: int = 20,
                       max_keys: int = 150, seed: int = 0) -> LogisticPopulation:
        """The population weights. Each train life's features read a prior table fitted on the
        other folds, so its own outcomes never leak into the prior it is fitted against."""
        lives = list(train_lives)
        if not lives:
            raise ValueError("fit_population needs train lives with queries")
        vocab = tuple(sorted({stem(a, life.meta.people) for life in lives for a in life.meta.activities}))
        cells = [_life_cells(life) for life in lives]
        folds = max(1, min(folds, len(lives)))
        fold_tables = [_table(cells, smoothing, minus=cells[f::folds]) if folds > 1 else _table(cells, smoothing)
                       for f in range(folds)]
        spec = FeatureSpec(vocab=vocab, mood_center=_mood_center(lives), prior=_table(cells, smoothing),
                           counter=CounterSpec(strength, count_strength, weekday_strength))
        rng = np.random.default_rng(seed)
        rows: list[RawFeatures] = []
        ys: list[float] = []
        groups: list[int] = []
        for i, life in enumerate(lives):
            r, y = _replay(life, replace(spec, prior=fold_tables[i % folds]), ask_rate, rng)
            rows.extend(r)
            ys.extend(y)
            groups.extend([i] * len(r))
        if not rows:
            raise ValueError("fit_population needs train lives with queries")
        keys = select_keys(rows, ys, groups, z_min=z_min, min_groups=min_groups, max_keys=max_keys)
        spec = replace(spec, keys=keys)
        x = np.vstack([assemble(spec, r) for r in rows])
        mean, scale = x.mean(axis=0), x.std(axis=0)
        mean[0], scale[0] = 0.0, 1.0
        scale = np.where(scale < 1e-8, 1.0, scale)
        w = _fit_logistic((x - mean) / scale, np.array(ys), l2)
        return LogisticPopulation(spec, w, mean, scale, len(rows))

    def begin(self, meta: LifeMeta) -> None:
        self.hist = PersonHistory(meta)
        pop = self.population
        if pop is None:
            self.spec = FeatureSpec(vocab=tuple(sorted({stem(a, meta.people) for a in meta.activities})))
            self._mean, self._scale = np.zeros(self.spec.size), np.ones(self.spec.size)
            self.m = np.zeros(self.spec.size)
        else:
            self.spec, self._mean, self._scale = pop.spec, pop.mean, pop.scale
            self.m = pop.weights.copy()
        self.cov0 = np.diag(np.where(self.spec.personal, self.prior_var, self.shared_var))
        self.cov = self.cov0.copy()

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.hist.add(events)

    def _raw_z(self, subject: str, as_of: int, target: int, kind: str, family: str) -> tuple[RawFeatures, np.ndarray]:
        raw = raw_features(self.spec, self.hist, subject, as_of, target, kind, family)
        z: np.ndarray = (assemble(self.spec, raw) - self._mean) / self._scale
        return raw, z

    def _z(self, subject: str, as_of: int, target: int, kind: str, family: str) -> np.ndarray:
        return self._raw_z(subject, as_of, target, kind, family)[1]

    def _counter_var(self, raw: RawFeatures) -> float:
        """The stacked Counter estimate's variance, carried to the logit through its weights."""
        names = self.spec.dense_names
        i, j = names.index("counter_logit"), names.index("counter_logit*hidden")
        w = self.m[i] / self._scale[i] + (self.m[j] / self._scale[j] if raw.hidden else 0.0)
        pc = min(max(raw.counter_p, P_MIN), P_MAX)
        return float(w * w / (raw.counter_n * pc * (1 - pc)))

    def _predictive(self, z: np.ndarray) -> tuple[float, float]:
        mu, var = float(self.m @ z), max(float(z @ self.cov @ z), 0.0)  # rounding can dip below 0
        return sigmoid(mu / math.sqrt(1.0 + math.pi * var / 8.0)), var

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        self.hist.note_queries(queries)
        out = []
        for q in queries:
            raw, z = self._raw_z(q.subject, q.asked_day, q.target_day, query_kind(q, self.hist, q.asked_day),
                                 q.family)
            p, var = self._predictive(z)
            out.append(Prediction(q.qid, _clip(p), u=math.sqrt(var + self._counter_var(raw))))
        return out

    def end_day(self, day: int) -> None:
        self._learn(self._examples(day))

    def _examples(self, day: int) -> list[tuple[np.ndarray, float]]:
        """The day's occurrences of activities whose family a query has shown."""
        examples = []
        for a in self.hist.activities:
            y, family = self.hist.outcome(a, day), self.hist.families.get(a)
            if y is not None and family is not None:
                kind = "plan" if self.hist.live_plan(a, day, day - 1) else "habit"
                examples.append((self._z(a, day - 1, day, kind, family), float(y)))
        return examples

    def _learn(self, examples: Sequence[tuple[np.ndarray, float]]) -> None:
        for z, y in examples:
            sz = self.cov @ z
            p = sigmoid(float(self.m @ z))
            w = p * (1 - p)
            self.cov = self.cov - (w / (1.0 + w * float(z @ sz))) * np.outer(sz, sz)
            self.m = self.m + (self.cov @ z) * (y - p)


class SurpriseLR(OnlineLogistic):
    """``OnlineLogistic`` that learns faster when surprised (Nassar et al. 2010, simplified).

    Each night, before learning, the evidence for "the person changed" accumulates as a decaying
    CUSUM of log Bayes factors, change (outcome a coin flip, Nassar's uniform U) against no change:
    ``R_t = max(0, decay * R_{t-1} + sum_i log(0.5 / p_i(y_i)))``, with ``p_i(y_i)`` the predictive
    probability of each of the night's outcomes. The change-point probability is
    ``CPP_t = sigmoid(logit(hazard) + R_t)``, and its excess over the hazard,
    ``c_t = (CPP_t - hazard) / (1 - hazard)``, adds ``gain * c_t * Σ0`` (``Σ0`` the prior covariance)
    to the posterior covariance before the night's updates: process noise, so they move
    ``1 + gain * c_t * Σ0/Σ`` times as far, i.e. ``lr_t = base_t * (1 + k * CPP_t)`` in SGD terms.
    With no surprise (R = 0) it is exactly ``OnlineLogistic``.
    """
    name = "surprise_lr"

    def __init__(self, population: LogisticPopulation | None = None, *, hazard: float = 0.02,
                 gain: float = 1.0, decay: float = 0.7, prior_var: float = 1e-3, shared_var: float = 1e-6,
                 name: str | None = None) -> None:
        super().__init__(population, prior_var=prior_var, shared_var=shared_var, name=name)
        self.hazard, self.gain, self.decay = hazard, gain, decay

    def begin(self, meta: LifeMeta) -> None:
        super().begin(meta)
        self.evidence = 0.0
        self.cpp_trace: list[tuple[int, float]] = []

    def end_day(self, day: int) -> None:
        examples = self._examples(day)
        if examples:
            bayes = 0.0
            for z, y in examples:
                p, _ = self._predictive(z)
                bayes += math.log(0.5 / max(p if y else 1 - p, P_MIN))
            self.evidence = max(0.0, self.decay * self.evidence + bayes)
            cpp = sigmoid(logit(self.hazard) + self.evidence)
            excess = (cpp - self.hazard) / (1 - self.hazard)
            if excess > 0:
                self.cov = self.cov + self.gain * excess * self.cov0
            self.cpp_trace.append((day, cpp))
        self._learn(examples)
