"""What the rivals compute from a ``PersonHistory``: the population prior table, the Counter's
estimate, and the logistic rivals' features.

The feature vector of "will ``subject`` happen on ``target``?", read as of ``as_of``, has a dense
block and sparse keys (``FeatureSpec.names``):

- base rates, for both families: the population prior's logit, the Counter's (and times
  ``hidden``), its evidence, an exponentially forgetting rate (it follows a fade), the horizon;
- the hidden family's regime signals (zero for a count query): late-night messages over 1/3/7/14
  days, mood over 1/3/7, the residual of the person's hidden-family outcomes over 3/7/14 days
  (``y`` minus each activity's own rate), the activity's streak of misses, those signals times
  the horizon, and per activity stem times late7, mood3 and resid7 (activities differ in how hard a
  busy spell hits them, and one in four people has one going against the population);
- keys, hidden family only, of which the fit keeps the informative ones (``select_keys``): a
  mention in the last 2 or 7 days (alone and per stem), another activity done the day before the
  target (per stem pair; its rate when that day is still ahead), and an answer from the last 14
  days with its timing at the target (``in=c`` as days until, ``left=r`` as days left, else age).
"""
from __future__ import annotations

import math
from collections import Counter as Tally
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from anjo.sleep.mind.cog.rival_history import PersonHistory, token_ints

P_MIN, P_MAX = 1e-4, 1 - 1e-4
OTHER = "<other>"
ANSWER_DAYS = 14
SLOT_LOOKBACK = 14  # a weekday seen in this many days counts as a habit's slot
RECENT_GAMMA = 0.8  # per occurrence, for the forgetting rate
MENTION_CAP = 3.0
PERSONAL = frozenset({"counter_logit", "counter_logit*hidden", "recent_logit", "recent_logit*hidden", "streak"})
REGIME_SIGNALS = ("late7", "mood3", "resid7")
DENSE_HIDDEN = ("late1", "late3", "late7", "late14", "mood1", "mood3", "mood7", "resid3", "resid7", "resid14",
                "streak", "late7*h", "mood3*h", "resid7*h")
DENSE_BASE = ("bias", "hidden", "plan", "prior_logit", "counter_logit", "counter_logit*hidden", "evidence",
              "recent_logit", "recent_logit*hidden", "horizon", "horizon*hidden")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def logit(p: float) -> float:
    p = min(max(p, P_MIN), P_MAX)
    return math.log(p / (1 - p))


# -- the population prior --
PriorKey = tuple[str | int, ...]


def prior_keys(family: str, kind: str, stem: str, horizon: int) -> tuple[PriorKey, ...]:
    """Coarse to fine: all, family, family+kind, +stem, +horizon. No weekday: a query is about a
    day the activity can happen, and which weekdays those are says nothing about the rate (on
    dev, a weekday level cost 0.003 nats)."""
    return ((), (family,), (family, kind), (family, kind, stem), (family, kind, stem, horizon))


@dataclass(frozen=True, eq=False)
class PriorTable:
    """P(happens | key) from train queries, each level shrunk to the one above (``smoothing``)."""
    cells: Mapping[PriorKey, tuple[float, float]]  # key -> (times happened, queries)
    smoothing: float = 10.0

    def p(self, family: str, kind: str, stem: str, horizon: int = 1, depth: int = 5) -> float:
        est = 0.5
        for key in prior_keys(family, kind, stem, horizon)[:depth]:
            k, n = self.cells.get(key, (0.0, 0.0))
            est = (k + self.smoothing * est) / (n + self.smoothing)
        return est


# -- the Counter's estimate --
@dataclass(frozen=True)
class CounterSpec:
    """Beta pseudo-counts: an activity's rate shrinks to the population (``strength``, or
    ``count_strength`` in the count family, whose people differ little), and a weekday's rate
    to the activity's (``weekday_strength``)."""
    strength: float = 6.0
    count_strength: float = 100.0
    weekday_strength: float = 100.0


def counter_estimate(spec: CounterSpec, prior: PriorTable | None, hist: PersonHistory, family: str, kind: str,
                     subject: str, as_of: int, target: int) -> tuple[float, int]:
    """(p, occurrences seen): did / (did + skipped + cancelled), activity then weekday."""
    p0 = prior.p(family, kind, hist.stem(subject), depth=4) if prior is not None else 0.5
    k_a, n_a = hist.rate(subject, as_of)
    k_w, n_w = hist.rate(subject, as_of, hist.weekday(target))
    s = spec.count_strength if family == "count" else spec.strength
    p_a = (k_a + s * p0) / (n_a + s)
    return (k_w + spec.weekday_strength * p_a) / (n_w + spec.weekday_strength), n_a


# -- the day's shared reads --
@dataclass(frozen=True)
class DayContext:
    late: Mapping[int, float]  # window -> late-night messages per day
    mood: Mapping[int, float]  # window -> mean mood minus the population's centre (0 if none)
    resid: Mapping[int, float]  # window -> sum of (y - own rate) over hidden occurrences / (n + 1)
    rates: Mapping[str, float]  # activity -> its own smoothed rate
    slots: Mapping[str, frozenset[int]]  # activity -> weekdays it occurred on lately
    mentions: Mapping[int, Mapping[str, int]]  # window -> subject -> count
    answers: tuple[tuple[int, str, tuple[str, ...]], ...]


def day_context(hist: PersonHistory, as_of: int, mood_center: float) -> DayContext:
    def make() -> DayContext:
        rates = {a: (hist.rate(a, as_of)[0] + 1.0) / (hist.rate(a, as_of)[1] + 2.0) for a in hist.activities}
        resid: dict[int, float] = {}
        for w in (3, 7, 14):
            total, n = 0.0, 0
            for a in hist.activities:
                if hist.families.get(a) == "hidden":
                    for d, y in hist.record(a, as_of):
                        if d > as_of - w:
                            total, n = total + y - rates[a], n + 1
            resid[w] = total / (n + 1)
        mood = {w: hist.mood_mean(as_of, w) for w in (1, 3, 7)}
        slots = {a: frozenset(hist.weekday(d) for d, _ in hist.record(a, as_of) if d > as_of - SLOT_LOOKBACK)
                 for a in hist.activities}
        return DayContext(
            late={w: hist.late_count(as_of, w) / w for w in (1, 3, 7, 14)},
            mood={w: 0.0 if m is None else m - mood_center for w, m in mood.items()},
            resid=resid, rates=rates, slots=slots,
            mentions={w: Tally(hist.mentioned(as_of, w)) for w in (2, 7)},
            answers=tuple(hist.recent_answers(as_of, ANSWER_DAYS)))
    return hist.cached(("day", as_of, mood_center), make)


# -- the feature spec --
@dataclass(frozen=True)
class FeatureSpec:
    vocab: tuple[str, ...]  # activity stems; any other shares the "<other>" slot
    keys: tuple[str, ...] = ()  # the sparse keys the fit kept
    mood_center: float = 3.0
    prior: PriorTable | None = None
    counter: CounterSpec = field(default_factory=CounterSpec)

    @property
    def acts(self) -> tuple[str, ...]:
        return (*self.vocab, OTHER)

    @cached_property
    def dense_names(self) -> tuple[str, ...]:
        return (*DENSE_BASE, *DENSE_HIDDEN, *(f"act={a}" for a in self.acts), *(f"act={a}|count" for a in self.acts),
                *(f"act={a}*{s}" for s in REGIME_SIGNALS for a in self.acts))

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.dense_names, *self.keys)

    @property
    def size(self) -> int:
        return len(self.dense_names) + len(self.keys)

    @cached_property
    def key_index(self) -> Mapping[str, int]:
        return {k: len(self.dense_names) + i for i, k in enumerate(self.keys)}

    @property
    def personal(self) -> np.ndarray:
        """Which weights may differ from person to person: the trust in their own counts against
        their recent occurrences, their streak, and each activity's response to the regime (one in
        four people has an activity that a busy spell helps)."""
        signals = tuple(f"*{s}" for s in REGIME_SIGNALS)
        mask: np.ndarray = np.array([n in PERSONAL or (n.startswith("act=") and n.endswith(signals))
                                     for n in self.names])
        return mask


@dataclass(frozen=True)
class RawFeatures:
    dense: np.ndarray
    keys: Mapping[str, float]
    counter_p: float  # the Counter's estimate, the offset that key selection tests against
    hidden: bool
    counter_n: float = 1.0  # the Counter's effective count: occurrences plus pseudo-counts


def raw_features(spec: FeatureSpec, hist: PersonHistory, subject: str, as_of: int, target: int, kind: str,
                 family: str) -> RawFeatures:
    d, h = as_of, target - as_of
    stem = hist.stem(subject)
    slot = stem if stem in spec.vocab else OTHER
    hidden = family == "hidden"
    g = float(hidden)
    p0 = spec.prior.p(family, kind, stem, h) if spec.prior is not None else 0.5
    pc, n = counter_estimate(spec.counter, spec.prior, hist, family, kind, subject, d, target)
    recent, lc = logit(hist.decayed_rate(subject, d, RECENT_GAMMA)), logit(pc)
    ctx = day_context(hist, d, spec.mood_center)
    hz = (h - 1) / 6.0
    regime = {"late7": ctx.late[7], "mood3": ctx.mood[3], "resid7": ctx.resid[7]}
    act = [float(a == slot) for a in spec.acts]
    dense = np.array([
        1.0, g, float(kind == "plan"), logit(p0), lc, g * lc, math.log1p(n), recent, g * recent,
        hz, g * hz,
        *(g * v for v in (ctx.late[1], ctx.late[3], ctx.late[7], ctx.late[14], ctx.mood[1], ctx.mood[3],
                          ctx.mood[7], ctx.resid[3], ctx.resid[7], ctx.resid[14],
                          float(hist.streak(subject, d)), hz * regime["late7"], hz * regime["mood3"],
                          hz * regime["resid7"])),
        *(g * v for v in act), *((1 - g) * v for v in act), *(g * v * regime[s] for s in REGIME_SIGNALS for v in act),
    ], dtype=float)
    keys = _keys(hist, ctx, subject, slot, d, target) if hidden else {}
    s = spec.counter.count_strength if family == "count" else spec.counter.strength
    return RawFeatures(dense, keys, pc, hidden, n + s)


def _keys(hist: PersonHistory, ctx: DayContext, subject: str, slot: str, as_of: int, target: int) -> dict[str, float]:
    keys: dict[str, float] = {}
    for w, counts in ctx.mentions.items():
        for m, c in counts.items():
            v = min(float(c), MENTION_CAP)
            keys[f"m{w}:{m}"] = v
            keys[f"{slot}|m{w}:{m}"] = v
    prev = target - 1
    for b in hist.activities:
        if b == subject:
            continue
        if prev <= as_of:
            v = float(hist.outcome(b, prev) == 1)
        elif hist.live_plan(b, prev, as_of) or hist.weekday(prev) in ctx.slots.get(b, frozenset()):
            v = ctx.rates.get(b, 0.5)
        else:
            v = 0.0
        if v:
            keys[f"{slot}|prev:{hist.stem(b)}"] = v
    for day, subj, toks in ctx.answers:
        name = f"ans:{_answer_name(subj, toks)}:{_answer_timing(toks, day, target)}"
        keys[name] = 1.0
        keys[f"{slot}|{name}"] = 1.0
    return keys


def _answer_name(subject: str, tokens: Sequence[str]) -> str:
    words = [t.lower() for t in tokens[1:] if "=" not in t and not t.lstrip("-").isdigit()]
    if not words or words[0] != subject.lower():
        words.insert(0, subject.lower())
    return "/".join(words)


def _answer_timing(tokens: Sequence[str], day: int, target: int) -> str:
    """When the answer's news falls, seen from the target day."""
    for tok in tokens:
        (count,) = token_ints([tok]) or (None,)
        if count is None:
            continue
        if tok.startswith("in="):
            return f"u{min(max(day + count - target, -1), 5)}"  # days until; -1 is past
        if tok.startswith("left="):
            return f"r{min(max(count - (target - day), 0), 2)}"  # days left; 0 is over
    age = target - day
    return "a0" if age <= 2 else "a1" if age <= 6 else "a2"


def assemble(spec: FeatureSpec, raw: RawFeatures) -> np.ndarray:
    x = np.zeros(spec.size)
    x[:len(raw.dense)] = raw.dense
    for k, v in raw.keys.items():
        i = spec.key_index.get(k)
        if i is not None:
            x[i] = v
    return x


def features(spec: FeatureSpec, hist: PersonHistory, subject: str, as_of: int, target: int, kind: str,
             family: str = "hidden") -> np.ndarray:
    """The full vector (``spec.names``) of "will ``subject`` happen on ``target``?" as of ``as_of``."""
    return assemble(spec, raw_features(spec, hist, subject, as_of, target, kind, family))


def select_keys(rows: Iterable[RawFeatures], ys: Iterable[float], groups: Iterable[int], *, z_min: float = 3.0,
                min_groups: int = 20, max_keys: int = 150) -> tuple[str, ...]:
    """Keys whose score test against the Counter's estimate as an offset clears ``z_min``.

    The statistic is cluster-robust, the clusters being lives (``groups``): with
    ``S_g = sum v (y - p)`` over a life's rows where the key is on, ``z = sum_g S_g / sqrt(sum_g S_g^2)``.
    A life's rows share its person, so counting them as independent would pass person markers (a
    favourite team, a show) as signals. A key must be on in ``min_groups`` lives.
    """
    per_group: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for raw, y, g in zip(rows, ys, groups, strict=True):
        r = y - raw.counter_p
        for k, v in raw.keys.items():
            per_group[k][g] += v * r
    z: dict[str, float] = {}
    for k, by_life in per_group.items():
        sq = sum(s * s for s in by_life.values())
        if len(by_life) >= min_groups and sq > 0:
            z[k] = abs(sum(by_life.values())) / math.sqrt(sq)
    kept = sorted((k for k in z if z[k] >= z_min), key=lambda k: (-z[k], k))[:max_keys]
    return tuple(sorted(kept))
