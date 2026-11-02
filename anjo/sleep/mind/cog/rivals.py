"""The rival predictors of MIND's cognition sprint (DESIGN.md, "Rivals").

- ``PopulationPrior``: P(happens | query kind, activity, weekday, horizon) from train lives, backed
  off level by level to the global rate (activities are names, so the key is the name).
- ``Counter``: a per-person Beta-Binomial on (activity, weekday) outcomes seen so far, with the
  population prior as the Beta prior's mean (``strength`` pseudo-counts); a plan query uses the
  person's plan kept-rate instead.
- ``OnlineLogistic``: a per-person Bayesian logistic regression on recent-event features (the
  population prior's and the Counter's estimates stacked in, ``FeatureSpec.names``), its prior the
  population fit on train lives, updated online by a Laplace (extended Kalman) step per outcome,
  with prior variance only where a person may differ. ``u`` is the predictive std of the logit.
  The strongest simple rival.
- ``SurpriseLR``: ``OnlineLogistic`` whose learning rate rises with surprise (T4's rival).

All read the life through ``PersonHistory``: event fields (kind, subject, value, day) and tokens,
never text. Until world.py fixes a token schema these conventions are assumed:

- an outcome on day ``t``: a ``did`` event for the activity means it happened; ``skipped`` or a
  ``cancel`` aimed at ``t`` means it did not; no event means it did not (``absent_is_negative``);
- ``plan`` and ``cancel`` carry the planned day in ``value`` (else the first integer in a token,
  e.g. ``day=17``; a plan without one is for tomorrow, a cancel without one hits the next plan);
- ``mood`` carries its level in ``value`` (else a mood word token, ``MOOD_WORDS``);
- a ``message`` is late-night when a token contains ``late``, ``night`` or ``midnight``, or names an
  hour (``hour=23``) from 22:00 to 04:59;
- a query is a plan query when a token contains ``plan`` or ``kept``, or a live plan exists.

Labels for online learning come only from the person's own later events, never from ``resolve``;
``fit_population`` may use ``resolve`` because train lives are training data.
"""
from __future__ import annotations

import math
import zlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from anjo.sleep.mind.cog.types import Event, Life, LifeMeta, Prediction, Query

P_MIN, P_MAX = 1e-4, 1 - 1e-4
LATE_WORDS = ("late", "night", "midnight")
PLAN_WORDS = ("plan", "kept")
MOOD_WORDS: Mapping[str, float] = {
    "awful": 1.0, "terrible": 1.0, "bad": 2.0, "low": 2.0, "sad": 2.0, "stressed": 2.0,
    "tired": 2.5, "meh": 3.0, "ok": 3.0, "okay": 3.0, "fine": 3.0, "good": 4.0, "happy": 4.5,
    "great": 5.0,
}
LATE_HOURS = frozenset({22, 23, 0, 1, 2, 3, 4})
RECENT_ANSWER_DAYS = 7
PERSONAL_FEATURES = frozenset({"counter_logit", "recent_logit"})
RECENT_GAMMA = 0.6  # per week, for the decayed weekday rate


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def logit(p: float) -> float:
    p = min(max(p, P_MIN), P_MAX)
    return math.log(p / (1 - p))


def token_ints(tokens: Iterable[str]) -> list[int]:
    """Integers a token spells: ``17``, ``day=17``, ``day:17``."""
    out = []
    for tok in tokens:
        tail = tok.rsplit("=", 1)[-1].rsplit(":", 1)[-1]
        if tail.lstrip("-").isdigit():
            out.append(int(tail))
    return out


def is_late(tokens: Sequence[str]) -> bool:
    for tok in (t.lower() for t in tokens):
        if any(w in tok for w in LATE_WORDS):
            return True
        if tok.startswith("hour") and any(h in LATE_HOURS for h in token_ints([tok])):
            return True
    return False


def is_plan_query(query: Query) -> bool:
    return any(w in tok.lower() for tok in query.tokens for w in PLAN_WORDS)


class PersonHistory:
    """What one person has said so far, indexed by day; every read is "as of" a day."""

    def __init__(self, meta: LifeMeta) -> None:
        self.meta = meta
        self.activities: list[str] = list(meta.activities)
        self.did: dict[int, set[str]] = defaultdict(set)
        self.skipped: dict[int, set[str]] = defaultdict(set)
        self.cancels: dict[int, list[str]] = defaultdict(list)  # day said -> activities
        self.moods: dict[int, list[float]] = defaultdict(list)
        self.late: dict[int, int] = defaultdict(int)
        self.messages: dict[int, int] = defaultdict(int)
        self.plans: dict[tuple[str, int], int] = {}  # (activity, target day) -> day planned
        self.plan_cancels: dict[tuple[str, int], int] = {}  # (activity, target day) -> day cancelled
        self.answers: list[tuple[int, tuple[str, ...]]] = []

    def weekday(self, day: int) -> int:
        return (self.meta.start_weekday + day) % 7

    def add(self, events: Iterable[Event]) -> None:
        for e in events:
            self._add(e)

    def _activity(self, name: str) -> None:
        if name not in self.activities:
            self.activities.append(name)

    def _add(self, e: Event) -> None:
        d, a = e.day, e.subject
        if e.kind in ("did", "skipped"):
            self._activity(a)
            (self.did if e.kind == "did" else self.skipped)[d].add(a)
        elif e.kind == "plan":
            self._activity(a)
            planned = self._day_of(e)
            self.plans[(a, planned if planned is not None else d + 1)] = d
        elif e.kind == "cancel":
            self._activity(a)
            named = self._day_of(e)
            live = sorted(t for (pa, t) in self.plans if pa == a and t >= d)
            target = named if named is not None else (live[0] if live else d)
            self.plan_cancels[(a, target)] = d
            self.cancels[d].append(a)
        elif e.kind == "mood":
            level = e.value if e.value is not None else next(
                (MOOD_WORDS[t.lower()] for t in e.tokens if t.lower() in MOOD_WORDS), None)
            if level is not None:
                self.moods[d].append(float(level))
        elif e.kind == "message":
            self.messages[d] += 1
            self.late[d] += int(is_late(e.tokens))
        elif e.kind == "answer":
            self.answers.append((d, tuple(e.tokens)))

    @staticmethod
    def _day_of(e: Event) -> int | None:
        """The day a plan or cancel is for: its value, else the first later day a token names."""
        if e.value is not None:
            return int(e.value)
        later = [n for n in token_ints(e.tokens) if n > e.day]
        return later[0] if later else None

    # -- reads, as of a day --
    def outcome(self, activity: str, day: int, absent_is_negative: bool = True) -> int | None:
        if activity in self.did.get(day, ()):
            return 1
        if activity in self.skipped.get(day, ()) or (activity, day) in self.plan_cancels:
            return 0
        return 0 if absent_is_negative else None

    def count(self, store: Mapping[int, set[str]], activity: str, as_of: int, days: int) -> int:
        return sum(activity in store.get(j, ()) for j in range(as_of - days + 1, as_of + 1))

    def total(self, store: Mapping[int, int | set[str] | list[str]], as_of: int, days: int) -> int:
        """Events in the store over the last ``days`` days: a count, or how many items a day holds."""
        out = 0
        for j in range(as_of - days + 1, as_of + 1):
            v = store.get(j, 0)
            out += v if isinstance(v, int) else len(v)
        return out

    def cancel_count(self, activity: str, as_of: int, days: int) -> int:
        return sum(self.cancels.get(j, []).count(activity) for j in range(as_of - days + 1, as_of + 1))

    def rate(self, activity: str, weekday: int, as_of: int, absent_is_negative: bool = True) -> tuple[int, int]:
        """(times done, days counted) on this weekday up to ``as_of``."""
        k = n = 0
        for j in range((weekday - self.meta.start_weekday) % 7, as_of + 1, 7):
            y = self.outcome(activity, j, absent_is_negative)
            if y is not None:
                k, n = k + y, n + 1
        return k, n

    def decayed_rate(self, activity: str, weekday: int, as_of: int, gamma: float) -> float:
        """The weekday's rate with each earlier week weighted ``gamma`` times the next, a prior of
        one success in two: it forgets an old habit, which the plain counts never do."""
        k, n, weight = 1.0, 2.0, 1.0
        last = as_of - (self.weekday(as_of) - weekday) % 7
        for j in range(last, -1, -7):
            y = self.outcome(activity, j)
            k, n, weight = k + weight * float(y or 0), n + weight, weight * gamma
        return k / n

    def live_plan(self, activity: str, target: int, as_of: int) -> bool:
        made = self.plans.get((activity, target))
        cancelled = self.plan_cancels.get((activity, target))
        return made is not None and made <= as_of and not (cancelled is not None and cancelled <= as_of)

    def kept(self, as_of: int) -> tuple[int, int]:
        """(plans kept, plans due) for plans whose day has passed by ``as_of``."""
        k = n = 0
        for (a, t), made in self.plans.items():
            if made <= as_of and t <= as_of:
                k, n = k + int(a in self.did.get(t, ())), n + 1
        return k, n

    def mood_mean(self, as_of: int, days: int) -> float | None:
        vals = [v for j in range(as_of - days + 1, as_of + 1) for v in self.moods.get(j, ())]
        return sum(vals) / len(vals) if vals else None

    def recent_answers(self, as_of: int, days: int) -> list[tuple[int, tuple[str, ...]]]:
        return [(d, toks) for d, toks in self.answers if as_of - days < d <= as_of]

    def last_outcomes(self, activity: str, as_of: int, n: int) -> list[int]:
        """The activity's outcomes on its last ``n`` days with an event about it, oldest first."""
        days = sorted({j for j, s in self.did.items() if activity in s and j <= as_of}
                      | {j for j, s in self.skipped.items() if activity in s and j <= as_of}
                      | {t for (a, t), c in self.plan_cancels.items() if a == activity and c <= as_of and t <= as_of})
        return [int(self.outcome(activity, j) or 0) for j in days[-n:]]


def query_kind(query: Query, hist: PersonHistory, as_of: int) -> str:
    plan = is_plan_query(query) or hist.live_plan(query.subject, query.target_day, as_of)
    return "plan" if plan else "habit"


# -- the population prior --
PriorKey = tuple[str | int, ...]


@dataclass(frozen=True)
class PriorTable:
    cells: Mapping[PriorKey, tuple[float, float]]  # key -> (times happened, queries)
    smoothing: float = 10.0

    def p(self, kind: str, subject: str, weekday: int, horizon: int) -> float:
        est = 0.5
        for key in _prior_keys(kind, subject, weekday, horizon):
            k, n = self.cells.get(key, (0.0, 0.0))
            est = (k + self.smoothing * est) / (n + self.smoothing)
        return est


def _prior_keys(kind: str, subject: str, weekday: int, horizon: int) -> tuple[PriorKey, ...]:
    return ((), (kind,), (kind, subject), (kind, subject, weekday), (kind, subject, weekday, horizon))


class PopulationPrior:
    name = "population"

    def __init__(self, table: PriorTable) -> None:
        self.table = table

    @classmethod
    def fit_population(cls, train_lives: Iterable[Life], *, smoothing: float = 10.0) -> PriorTable:
        cells: dict[PriorKey, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for life in train_lives:
            hist = PersonHistory(life.meta)
            for d in range(life.meta.n_days):
                hist.add(life.events(d))
                for q in life.queries(d):
                    y = float(life.resolve(q.qid).outcome)
                    keys = _prior_keys(query_kind(q, hist, d), q.subject, hist.weekday(q.target_day),
                                       q.target_day - d)
                    for key in keys:
                        cells[key][0] += y
                        cells[key][1] += 1.0
        return PriorTable({k: (v[0], v[1]) for k, v in cells.items()}, smoothing)

    def begin(self, meta: LifeMeta) -> None:
        self.hist = PersonHistory(meta)

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.hist.add(events)

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        return [Prediction(q.qid, _clip(self.table.p(query_kind(q, self.hist, q.asked_day), q.subject,
                                                     self.hist.weekday(q.target_day), q.target_day - q.asked_day)))
                for q in queries]

    def end_day(self, day: int) -> None: ...


def _clip(p: float) -> float:
    return min(max(p, P_MIN), P_MAX)


# -- the per-person counter --
class Counter:
    name = "counter"

    def __init__(self, prior: PriorTable | None = None, *, strength: float = 4.0,
                 plan_strength: float = 4.0, absent_is_negative: bool = True) -> None:
        self.prior, self.strength, self.plan_strength = prior, strength, plan_strength
        self.absent_is_negative = absent_is_negative

    def begin(self, meta: LifeMeta) -> None:
        self.hist = PersonHistory(meta)

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.hist.add(events)

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        out = []
        for q in queries:
            d, wd = q.asked_day, self.hist.weekday(q.target_day)
            kind = query_kind(q, self.hist, d)
            p0 = self.prior.p(kind, q.subject, wd, q.target_day - d) if self.prior else 0.5
            if kind == "plan":
                (k, n), s = self.hist.kept(d), self.plan_strength
            else:
                (k, n), s = self.hist.rate(q.subject, wd, d, self.absent_is_negative), self.strength
            mean = (k + s * p0) / (n + s)
            out.append(Prediction(q.qid, _clip(mean), u=math.sqrt(mean * (1 - mean) / (n + s + 1))))
        return out

    def end_day(self, day: int) -> None: ...


# -- features of the logistic rivals --
@dataclass(frozen=True)
class FeatureSpec:
    vocab: tuple[str, ...]  # activity names; any other subject shares one "<other>" slot
    hash_dims: int = 16
    mood_center: float = 3.0
    prior: PriorTable | None = None  # the population prior, stacked in as a feature
    counter_strength: float = 4.0  # the Counter's pseudo-counts, for its stacked estimate

    @property
    def acts(self) -> tuple[str, ...]:
        return (*self.vocab, "<other>")

    @property
    def names(self) -> tuple[str, ...]:
        return ("bias", *(f"wd={w}" for w in range(7)), "horizon", *(f"act={a}" for a in self.acts),
                "did1", "did3", "did7", "skip1", "skip3", "skip7", "cancel7", "rate_logit", "rate_n",
                "prior_logit", "counter_logit", "recent_logit",
                "mood3", "mood7", "has_mood3", "late3", "late7", "msgs3", "all_did7", "all_skip7",
                "all_cancel7", "planned", "plan_query", "kept_logit",
                *(f"answer#{i}" for i in range(self.hash_dims)),
                *(f"act*answer#{i}" for i in range(self.hash_dims)), "answer_names_day",
                "answer_names_subject", *(f"act={a}*late3" for a in self.acts),
                *(f"act={a}*mood3" for a in self.acts))

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def personal(self) -> np.ndarray:
        """Which weights may differ from person to person: the trust in the person's own counts
        (which already carry their base rates) against their recent weeks, and each activity's
        response to late nights and mood (where a person can go against the population)."""
        mask: np.ndarray = np.array([n in PERSONAL_FEATURES or n.endswith(("*late3", "*mood3")) for n in self.names])
        return mask


def _hash(token: str, dims: int) -> int:
    return zlib.crc32(token.encode("utf-8")) % dims


def features(spec: FeatureSpec, hist: PersonHistory, subject: str, as_of: int, target: int,
             kind: str) -> np.ndarray:
    """The feature vector of "will ``subject`` happen on ``target``?", from events up to ``as_of``.

    ``kind`` is the query's kind ("plan" or "habit"). With a population prior in ``spec``, its
    estimate and the Counter's posterior on it (habit: this activity on this weekday; plan: the
    person's kept-rate) are features too, so the regression starts from both and can only add.
    """
    a, d = subject, as_of
    act = [0.0] * len(spec.acts)
    act[spec.vocab.index(a) if a in spec.vocab else len(spec.vocab)] = 1.0
    wd = [0.0] * 7
    wd[hist.weekday(target)] = 1.0
    k, n = hist.rate(a, hist.weekday(target), d)
    kept_k, kept_n = hist.kept(d)
    p0 = spec.prior.p(kind, a, hist.weekday(target), target - d) if spec.prior is not None else 0.5
    ck, cn = (kept_k, kept_n) if kind == "plan" else (k, n)
    counter = (ck + spec.counter_strength * p0) / (cn + spec.counter_strength)
    mood3, mood7 = hist.mood_mean(d, 3), hist.mood_mean(d, 7)
    m3 = 0.0 if mood3 is None else mood3 - spec.mood_center
    late3 = float(hist.total(hist.late, d, 3))
    answers, act_answers = [0.0] * spec.hash_dims, [0.0] * spec.hash_dims
    recent = hist.recent_answers(d, RECENT_ANSWER_DAYS)
    for _, toks in recent:
        for tok in toks:
            if not any(c.isdigit() for c in tok):
                answers[_hash(tok.lower(), spec.hash_dims)] += 1.0
                act_answers[_hash(f"{a}|{tok.lower()}", spec.hash_dims)] += 1.0
    names_day = any(target in token_ints(toks) for day, toks in hist.answers if day <= d)
    names_subject = any(a in toks for _, toks in recent)
    x: np.ndarray = np.array([
        1.0, *wd, (target - d) / 7.0, *act,
        hist.count(hist.did, a, d, 1), hist.count(hist.did, a, d, 3), hist.count(hist.did, a, d, 7),
        hist.count(hist.skipped, a, d, 1), hist.count(hist.skipped, a, d, 3),
        hist.count(hist.skipped, a, d, 7), hist.cancel_count(a, d, 7),
        logit((k + 1) / (n + 2)), float(n), logit(p0), logit(counter),
        logit(hist.decayed_rate(a, hist.weekday(target), d, RECENT_GAMMA)),
        m3, 0.0 if mood7 is None else mood7 - spec.mood_center, float(mood3 is not None),
        late3, float(hist.total(hist.late, d, 7)), float(hist.total(hist.messages, d, 3)),
        float(hist.total(hist.did, d, 7)), float(hist.total(hist.skipped, d, 7)),
        float(hist.total(hist.cancels, d, 7)),
        float(hist.live_plan(a, target, d)), float(kind == "plan"), logit((kept_k + 1) / (kept_n + 2)),
        *answers, *act_answers, float(names_day), float(names_subject),
        *(v * late3 for v in act), *(v * m3 for v in act),
    ], dtype=float)
    return x


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


def _replay(life: Life, spec: FeatureSpec, ask_rate: float, rng: np.random.Generator) -> tuple[list[np.ndarray], list[float]]:
    """A train life's queries as (features, outcome), with random asks so answers get weights."""
    hist, rows, ys = PersonHistory(life.meta), [], []
    for d in range(life.meta.n_days):
        hist.add(life.events(d))
        questions = life.questions(d)
        if questions and rng.random() < ask_rate:
            hist.add([life.answer(questions[int(rng.integers(len(questions)))].question_id)])
        for q in life.queries(d):
            rows.append(features(spec, hist, q.subject, d, q.target_day, query_kind(q, hist, d)))
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

    Each night every activity's outcome that day is one training example, with the features of
    "will it happen tomorrow?" asked the day before (one per activity-day: the queries that
    targeted it would add correlated copies). The update is one Laplace step per example
    (Sherman-Morrison on the covariance). p is the probit-approximate predictive mean; ``u`` is the
    predictive std of the logit. ``predict`` has no side effects, so an inquirer may call it.
    """
    name = "online_lr"

    def __init__(self, population: LogisticPopulation | None = None, *, prior_var: float = 0.03,
                 shared_var: float = 1e-4, absent_is_negative: bool = True, name: str | None = None) -> None:
        self.population, self.prior_var, self.shared_var = population, prior_var, shared_var
        self.absent_is_negative = absent_is_negative
        if name is not None:
            self.name = name

    @classmethod
    def fit_population(cls, train_lives: Iterable[Life], *, l2: float = 1.0, ask_rate: float = 0.25,
                       hash_dims: int = 16, smoothing: float = 10.0, counter_strength: float = 4.0,
                       seed: int = 0) -> LogisticPopulation:
        lives = list(train_lives)
        vocab = tuple(sorted({a for life in lives for a in life.meta.activities}))
        spec = FeatureSpec(vocab=vocab, hash_dims=hash_dims, mood_center=_mood_center(lives),
                           prior=PopulationPrior.fit_population(lives, smoothing=smoothing),
                           counter_strength=counter_strength)
        rng = np.random.default_rng(seed)
        rows: list[np.ndarray] = []
        ys: list[float] = []
        for life in lives:
            r, y = _replay(life, spec, ask_rate, rng)
            rows.extend(r)
            ys.extend(y)
        if not rows:
            raise ValueError("fit_population needs train lives with queries")
        x = np.vstack(rows)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        mean[0], scale[0] = 0.0, 1.0
        scale = np.where(scale < 1e-8, 1.0, scale)
        w = _fit_logistic((x - mean) / scale, np.array(ys), l2)
        return LogisticPopulation(spec, w, mean, scale, len(rows))

    def begin(self, meta: LifeMeta) -> None:
        self.hist = PersonHistory(meta)
        pop = self.population
        if pop is None:
            self.spec = FeatureSpec(vocab=tuple(meta.activities))
            self._mean, self._scale = np.zeros(self.spec.size), np.ones(self.spec.size)
            self.m = np.zeros(self.spec.size)
        else:
            self.spec, self._mean, self._scale = pop.spec, pop.mean, pop.scale
            self.m = pop.weights.copy()
        self.cov0 = np.diag(np.where(self.spec.personal, self.prior_var, self.shared_var))
        self.cov = self.cov0.copy()

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self.hist.add(events)

    def _z(self, subject: str, as_of: int, target: int, kind: str) -> np.ndarray:
        z: np.ndarray = (features(self.spec, self.hist, subject, as_of, target, kind) - self._mean) / self._scale
        return z

    def _predictive(self, z: np.ndarray) -> tuple[float, float]:
        mu, var = float(self.m @ z), max(float(z @ self.cov @ z), 0.0)  # rounding can dip below 0
        return sigmoid(mu / math.sqrt(1.0 + math.pi * var / 8.0)), var

    def predict(self, queries: Sequence[Query]) -> list[Prediction]:
        out = []
        for q in queries:
            z = self._z(q.subject, q.asked_day, q.target_day, query_kind(q, self.hist, q.asked_day))
            p, var = self._predictive(z)
            out.append(Prediction(q.qid, _clip(p), u=math.sqrt(var)))
        return out

    def end_day(self, day: int) -> None:
        self._learn(self._examples(day))

    def _examples(self, day: int) -> list[tuple[np.ndarray, float]]:
        examples = []
        for a in self.hist.activities:
            y = self.hist.outcome(a, day, self.absent_is_negative)
            if y is not None:
                kind = "plan" if self.hist.live_plan(a, day, day - 1) else "habit"
                examples.append((self._z(a, day - 1, day, kind), float(y)))
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
                 gain: float = 1.0, decay: float = 0.7, prior_var: float = 0.03, shared_var: float = 1e-4,
                 absent_is_negative: bool = True, name: str | None = None) -> None:
        super().__init__(population, prior_var=prior_var, shared_var=shared_var,
                         absent_is_negative=absent_is_negative, name=name)
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
