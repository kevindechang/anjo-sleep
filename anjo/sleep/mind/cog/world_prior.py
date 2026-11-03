"""The cog world's population priors and one person's parameters, drawn from them. Pure numpy.

A person's hidden state on a day is ``z = regime * N_FACT + fact``: a workload regime (calm, busy,
crunch; a sticky Markov chain) and a fact chain (none, or one of an exam countdown, a visitor's
arrival and stay, an injury), the two independent. Every outcome is a Bernoulli draw whose logit is

    base[a, day] + effect[a, z] + streak_w[a] * streak + trigger_w[a] * (trigger source done yesterday)

where ``base`` is the fixed weekday habit (with any regime shift applied from its day on),
``effect`` the regime and fact effects (zero for a "steady" activity), ``streak`` the run of
missed occurrences (capped at 2), and the trigger an edge from another activity. A trigger source
has no streak or trigger of its own, so the oracle's exact forward pass stays small.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from anjo.sleep.mind.cog import render

N_REGIME: Final = len(render.REGIMES)
PLAN_LEAD_MAX: Final = 4  # a plan is made 1-4 days ahead (fewer near day 0)
VISIT_STAYS: Final = (2, 3, 4)
INJURY_SPANS: Final = (5, 6, 7)
MAX_STREAK: Final = 2

# The fact chain's states, in index order; each is the answer to "anything big coming up?".
FACT_TOKENS: Final = render.answer_options("plans")
N_FACT: Final = len(FACT_TOKENS)
N_STATE: Final = N_REGIME * N_FACT
FACT_NONE: Final = 0
EXAM: Final = {c: FACT_TOKENS.index(("answer", "plans", "exam", f"in={c}")) for c in range(render.EXAM_DAYS + 1)}
PRE: Final = {c: FACT_TOKENS.index(("answer", "plans", "visitor", f"in={c}"))
              for c in range(1, render.VISITOR_LEAD + 1)}
VISIT: Final = {r: FACT_TOKENS.index(("answer", "plans", "visitor", "here", f"left={r}"))
                for r in range(1, render.VISITOR_STAY + 1)}
INJURY: Final = {r: FACT_TOKENS.index(("answer", "plans", "injury", f"left={r}"))
                 for r in range(1, render.INJURY_DAYS + 1)}

# Fact effects on a sensitive activity's logit, by the activity's groups.
EXAM_SOCIAL: Final = -2.5  # social activities in the 3 days before an exam
EXAM_STUDY: Final = 1.2  # the study group in the 4 days before
VISIT_SPORT: Final = -1.5
VISIT_ROUTINE: Final = -1.2
VISIT_DINNER: Final = 0.8
INJURY_HIT: Final = -4.0  # run, gym, climbing stop
INJURY_SPORT: Final = -1.2  # other sport slows
MOOD_FACT: Final = {"exam": -0.4, "injury": -0.6, "visit": 0.4}
HINT_ON: Final = {"reading": 0.35, "guests_pre": 0.45, "guests_visit": 0.3, "knee": 0.45}
HINT_FALSE: Final = 0.03
FADE: Final = 2.0  # a "fade" shift lowers the activity's logit this much
FADE_PLANS: Final = 0.4  # and keeps this share of its plans
SHIFT_COUNTS: Final = ((0, 1, 2), (0.25, 0.5, 0.25))
WEEKDAY_SD: Final = 0.25  # a habit's per-weekday deviation from the person's level


@dataclass(frozen=True)
class Spec:
    """An activity kind in a population: its typical weekdays and mean regime effects."""
    stem: str
    planned: bool
    days: tuple[int, ...]
    busy: float
    crunch: float
    steady: float  # the chance that a person's activity of this kind is steady (count family)
    groups: frozenset[str]


@dataclass(frozen=True)
class Population:
    name: str
    specs: tuple[Spec, ...]
    triggers: tuple[tuple[str, str, float], ...]  # (source, target, mean weight)
    idiosyncrasy: float  # the chance a person has one regime effect against the population sign
    stay: tuple[float, float]  # the regime's daily stay probability, drawn uniformly per person
    leave: tuple[tuple[float, float, float], ...]  # where a leaving regime goes
    late: tuple[float, float, float]  # P(each of two late-night message chances) per regime
    mood: tuple[float, float, float]  # mean mood level per regime
    fact_rates: tuple[float, float, float]  # daily chance an exam, a visitor, an injury begins
    streak: float  # mean logit change per missed occurrence
    base_mean: float = 1.6  # a sensitive habit's slot-day logit (calm, no streak)
    keep_mean: float = 1.8  # a sensitive plan's keep logit
    steady_mean: float = 2.1  # a steady commitment's logit: reliable, and alike across people
    level_sd: float = 0.5  # between people, sensitive activities
    steady_sd: float = 0.35  # between people, steady activities


def _spec(stem: str, planned: bool, days: tuple[int, ...], busy: float, crunch: float, steady: float,
          *groups: str) -> Spec:
    return Spec(stem, planned, days, busy, crunch, steady, frozenset(groups))


BASE: Final = Population(
    name="base",
    specs=(
        _spec("run", False, (1, 3, 5), -1.52, -3.12, 0.0, "sport", "injury"),
        _spec("gym", False, (0, 2, 4), -1.35, -2.86, 0.0, "sport", "injury"),
        _spec("yoga", False, (2, 6), -1.18, -2.47, 0.5, "sport"),
        _spec("class", False, (0, 2), -0.68, -1.82, 0.7, "routine"),
        _spec("study_group", True, (1, 3), -0.85, -2.08, 0.3, "routine", "study"),
        _spec("call_mom", False, (2, 6), -1.18, -2.34, 0.5),
        _spec("dinner", True, (4, 5), -1.69, -3.38, 0.2, "social", "dinner"),
        _spec("climbing", True, (5, 6), -1.52, -2.99, 0.0, "sport", "injury", "social"),
        _spec("choir", False, (1, 3), -1.01, -2.34, 0.6, "social", "routine"),
        _spec("volunteering", True, (5, 6), -1.52, -3.12, 0.4, "social", "routine"),
        _spec("swim", False, (1, 4), -1.18, -2.6, 0.5, "sport"),
        _spec("piano", False, (2, 5), -1.01, -2.08, 0.7, "routine"),
    ),
    triggers=(("dinner", "call_mom", 1.0), ("gym", "run", -0.8), ("climbing", "yoga", 0.8),
              ("study_group", "class", 0.9), ("choir", "piano", 0.8), ("volunteering", "call_mom", 0.7),
              ("swim", "gym", -0.7), ("run", "swim", -0.6)),
    idiosyncrasy=0.25, stay=(0.92, 0.96),
    leave=((0.0, 0.75, 0.25), (0.55, 0.0, 0.45), (0.3, 0.7, 0.0)),
    late=(0.05, 0.2, 0.5), mood=(3.9, 3.3, 2.5),
    fact_rates=(1 / 80, 1 / 90, 1 / 100), streak=-0.55,
)
OOD: Final = Population(
    name="ood",
    specs=(
        _spec("run", False, (0, 2, 4, 6), -0.85, -1.56, 0.0, "sport", "injury"),
        _spec("gym", False, (1, 3, 5), 0.68, 1.56, 0.0, "sport", "injury"),
        _spec("yoga", False, (0, 4), -1.52, -2.86, 0.5, "sport"),
        _spec("class", False, (1, 3), -0.34, -1.04, 0.7, "routine"),
        _spec("study_group", True, (0, 2), 0.51, 1.17, 0.3, "routine", "study"),
        _spec("call_mom", False, (3, 5), -1.69, -3.12, 0.5),
        _spec("dinner", True, (2, 3), -2.03, -3.9, 0.2, "social", "dinner"),
        _spec("climbing", True, (1, 2), -1.01, -2.34, 0.0, "sport", "injury", "social"),
        _spec("choir", False, (0, 4), -1.35, -2.6, 0.6, "social", "routine"),
        _spec("volunteering", True, (6, 0), -0.85, -1.95, 0.4, "social", "routine"),
        _spec("swim", False, (0, 3), -1.52, -2.86, 0.5, "sport"),
        _spec("piano", False, (5, 6), 0.51, 1.04, 0.7, "routine"),
    ),
    triggers=(("dinner", "call_mom", -0.8), ("gym", "run", 0.7), ("yoga", "climbing", 0.8),
              ("class", "study_group", 0.9), ("piano", "choir", 0.6), ("call_mom", "volunteering", 0.7),
              ("gym", "swim", -0.7), ("swim", "run", 0.6)),
    idiosyncrasy=0.5, stay=(0.86, 0.93),
    leave=((0.0, 0.6, 0.4), (0.5, 0.0, 0.5), (0.4, 0.6, 0.0)),
    late=(0.08, 0.25, 0.45), mood=(3.7, 3.3, 2.7),
    fact_rates=(1 / 70, 1 / 70, 1 / 80), streak=-0.7,
)
PRIORS: Final = {"base": BASE, "ood": OOD}


@dataclass(frozen=True)
class Person:
    """One person's true parameters. Arrays are indexed [activity, day] or [activity, state]."""
    start_weekday: int
    people: tuple[str, ...]
    topics: tuple[str, ...]
    activities: tuple[str, ...]
    planned: np.ndarray  # (A,) bool
    steady: np.ndarray  # (A,) bool: no regime, fact or recent-event effect (the count family)
    base: np.ndarray  # (A, T) logit on each day (slot day for a habit, keep for a plan)
    pattern: np.ndarray  # (A, T) bool: a habit's slot day, or a day a plan can be for
    plan_rate: np.ndarray  # (A, T) chance a plannable day gets a plan
    effect: np.ndarray  # (A, S) regime + fact effect per hidden state
    streak_w: np.ndarray  # (A,)
    trigger_src: np.ndarray  # (A,) int, -1 for none
    trigger_w: np.ndarray  # (A,)
    chain: np.ndarray  # (S, S) hidden-state transition
    init: np.ndarray  # (S,) day-0 state distribution
    late_p: np.ndarray  # (S,)
    mood_p: np.ndarray  # (S, 5)
    hint_p: np.ndarray  # (S, 3), in render.HINTS order
    report: dict[str, Any]  # the parameters for Truth.params, as plain Python

    def outcome_logits(self, day: int, z: np.ndarray, streak: np.ndarray, prev_done: np.ndarray) -> np.ndarray:
        """(n, A) logits for ``n`` paths in states ``z`` with these streaks and yesterday's outcomes."""
        src = np.maximum(self.trigger_src, 0)
        logits: np.ndarray = (self.base[:, day][None, :] + self.effect[:, z].T + self.streak_w[None, :] * streak
                              + self.trigger_w[None, :] * prev_done[:, src])
        return logits

    def weekday(self, day: int) -> int:
        return (self.start_weekday + day) % 7


# ── the hidden chains ─────────────────────────────────────────────────────────────────────────
def fact_chain(rates: tuple[float, float, float]) -> np.ndarray:
    q_exam, q_visit, q_injury = rates
    p = np.zeros((N_FACT, N_FACT))
    p[FACT_NONE, FACT_NONE] = 1.0 - q_exam - q_visit - q_injury
    p[FACT_NONE, EXAM[render.EXAM_DAYS]] = q_exam
    p[FACT_NONE, PRE[render.VISITOR_LEAD]] = q_visit
    for r in INJURY_SPANS:
        p[FACT_NONE, INJURY[r]] = q_injury / len(INJURY_SPANS)
    for countdown in (EXAM, PRE, VISIT, INJURY):
        for c, state in countdown.items():
            if c - 1 in countdown:
                p[state, countdown[c - 1]] = 1.0
    p[EXAM[0], FACT_NONE] = p[VISIT[1], FACT_NONE] = p[INJURY[1], FACT_NONE] = 1.0
    for n in VISIT_STAYS:
        p[PRE[1], VISIT[n]] = 1.0 / len(VISIT_STAYS)
    return p


def _regime_chain(stay: float, leave: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    chain: np.ndarray = stay * np.eye(N_REGIME) + (1.0 - stay) * np.array(leave)
    return chain


def _stationary(p: np.ndarray) -> np.ndarray:
    a = np.vstack([p.T - np.eye(len(p)), np.ones(len(p))])
    b = np.zeros(len(p) + 1)
    b[-1] = 1.0
    solution: np.ndarray = np.linalg.lstsq(a, b, rcond=None)[0]
    return solution


def _fact_kind(f: int) -> str:
    if f in EXAM.values():
        return "exam"
    if f in PRE.values():
        return "pre"
    if f in VISIT.values():
        return "visit"
    return "injury" if f in INJURY.values() else "none"


def _fact_effects(groups: frozenset[str]) -> np.ndarray:
    eff = np.zeros(N_FACT)
    if "social" in groups:
        eff[[EXAM[c] for c in (1, 2, 3)]] += EXAM_SOCIAL
    if "study" in groups:
        eff[[EXAM[c] for c in (1, 2, 3, 4)]] += EXAM_STUDY
    visiting = list(VISIT.values())
    for group, value in (("sport", VISIT_SPORT), ("routine", VISIT_ROUTINE), ("dinner", VISIT_DINNER)):
        if group in groups:
            eff[visiting] += value
    if "injury" in groups:
        eff[list(INJURY.values())] += INJURY_HIT
    elif "sport" in groups:
        eff[list(INJURY.values())] += INJURY_SPORT
    return eff


def _mood_table(means: np.ndarray, sd: float) -> np.ndarray:
    """(S, 5): P(level 1..5) of a normal mood rounded to the nearest level, ends open."""
    cuts = np.array([1.5, 2.5, 3.5, 4.5])
    z = (cuts[None, :] - means[:, None]) / (sd * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z))
    full = np.hstack([np.zeros((len(means), 1)), cdf, np.ones((len(means), 1))])
    table: np.ndarray = np.clip(np.diff(full, axis=1), 1e-9, None)
    return table


def _hint_table() -> np.ndarray:
    p = np.full((N_FACT, len(render.HINTS)), HINT_FALSE)
    for c in range(1, render.EXAM_DAYS + 1):
        p[EXAM[c], 0] = HINT_ON["reading"]
    p[list(PRE.values()), 1] = HINT_ON["guests_pre"]
    p[list(VISIT.values()), 1] = HINT_ON["guests_visit"]
    p[list(INJURY.values()), 2] = HINT_ON["knee"]
    return p


# ── sampling a person ─────────────────────────────────────────────────────────────────────────
def _choose_activities(rng: np.random.Generator, pop: Population) -> tuple[list[Spec], np.ndarray]:
    while True:
        specs = [pop.specs[i] for i in sorted(rng.choice(len(pop.specs), 6, replace=False))]
        if not any(s.planned for s in specs) or sum(not s.planned for s in specs) < 2:
            continue
        chance = np.array([s.steady for s in specs])
        for _ in range(20):
            steady = rng.random(len(specs)) < chance
            if 2 <= steady.sum() <= len(specs) - 2:
                return specs, steady


def _weekdays(rng: np.random.Generator, typical: tuple[int, ...]) -> tuple[int, ...]:
    while True:
        u = rng.random(7)
        days = tuple(d for d in range(7) if u[d] < (0.85 if d in typical else 0.08))
        if 1 <= len(days) <= 4:
            return days


def _scaled(rng: np.random.Generator, mean: float, sd: float) -> float:
    return float(mean * math.exp(rng.normal(0.0, sd)))


def sample_person(rng: np.random.Generator, pop: Population, n_days: int) -> Person:
    start_weekday = int(rng.integers(7))
    people = tuple(str(n) for n in rng.choice(render.PEOPLE, int(rng.integers(3, 6)), replace=False))
    topics = (str(rng.choice(list(render.SHOWS))), render.WEATHER, str(rng.choice(list(render.TEAMS))))
    specs, steady = _choose_activities(rng, pop)
    friend = people[int(rng.integers(len(people)))]
    names = tuple(render.dinner_id(friend) if s.stem == "dinner" else s.stem for s in specs)
    n_act = len(specs)
    weekdays = np.array([(start_weekday + d) % 7 for d in range(n_days)])

    days = [_weekdays(rng, s.days) for s in specs]
    level = np.array([rng.normal(pop.steady_mean, pop.steady_sd) if steady[a] else
                      rng.normal(pop.keep_mean if s.planned else pop.base_mean, pop.level_sd)
                      for a, s in enumerate(specs)])
    by_weekday = level[:, None] + rng.normal(0.0, WEEKDAY_SD, (n_act, 7))
    base = by_weekday[:, weekdays]
    pattern = np.array([np.isin(weekdays, d) for d in days])
    rates = np.array([rng.uniform(0.6, 0.9) if s.planned else 0.0 for s in specs])
    plan_rate = np.repeat(rates[:, None], n_days, axis=1)

    regime = np.zeros((n_act, N_REGIME))
    for a, s in enumerate(specs):
        if not steady[a]:
            regime[a, 1:] = _scaled(rng, s.busy, 0.3), _scaled(rng, s.crunch, 0.3)
    odd: tuple[tuple[str, str], ...] = ()
    if rng.random() < pop.idiosyncrasy:
        a = int(rng.choice(np.flatnonzero(~steady)))
        regime[a, 1:] *= -rng.uniform(0.5, 0.9)
        odd = ((names[a], "regime"),)
    facts = np.array([np.zeros(N_FACT) if steady[a] else _fact_effects(s.groups) for a, s in enumerate(specs)])
    effect = (regime[:, :, None] + facts[:, None, :]).reshape(n_act, N_STATE)

    trigger_src, trigger_w = _triggers(rng, pop, specs, steady)
    streak_w = np.array([_scaled(rng, pop.streak, 0.3) if not steady[a] and a not in trigger_src
                         and rng.random() < 0.6 else 0.0 for a in range(n_act)])

    stay = float(rng.uniform(*pop.stay))
    p_regime = _regime_chain(stay, pop.leave)
    init = np.kron(_stationary(p_regime), np.eye(N_FACT)[FACT_NONE])
    late = np.clip(np.array(pop.late) * np.exp(rng.normal(0.0, 0.2, N_REGIME)), 0.01, 0.9)
    regime_of: np.ndarray = np.repeat(np.arange(N_REGIME), N_FACT)
    fact_of = np.tile(np.arange(N_FACT), N_REGIME)
    mood_mean = (np.array(pop.mood)[regime_of] + rng.normal(0.0, 0.3)
                 + np.array([MOOD_FACT.get(_mood_kind(f), 0.0) for f in fact_of]))
    mood_p = _mood_table(mood_mean, float(rng.uniform(0.7, 1.0)))

    shifts = _shifts(rng, n_days, names, specs, steady, base, pattern, plan_rate, by_weekday, level, weekdays, days)
    report = {
        "activities": {names[a]: {
            "mode": "planned" if s.planned else "habitual", "steady": bool(steady[a]),
            "weekdays": days[a], "base_by_weekday": tuple(float(x) for x in by_weekday[a]),
            "plan_rate": float(rates[a]) if s.planned else None,
            "regime_effects": {"busy": float(regime[a, 1]), "crunch": float(regime[a, 2])},
            "population_mean": {"busy": s.busy, "crunch": s.crunch},
            "streak_weight": float(streak_w[a]),
            "trigger": (names[int(trigger_src[a])], float(trigger_w[a])) if trigger_src[a] >= 0 else None,
        } for a, s in enumerate(specs)},
        "idiosyncratic": odd, "shifts": shifts, "shift_subjects": tuple(_shift_subjects(s) for s in shifts),
        "regime_stay": stay,
    }
    return Person(
        start_weekday=start_weekday, people=people, topics=topics, activities=names,
        planned=np.array([s.planned for s in specs]), steady=steady, base=base, pattern=pattern,
        plan_rate=plan_rate, effect=effect, streak_w=streak_w, trigger_src=trigger_src, trigger_w=trigger_w,
        chain=np.kron(p_regime, fact_chain(pop.fact_rates)), init=init, late_p=late[regime_of],
        mood_p=mood_p, hint_p=_hint_table()[fact_of], report=report,
    )


def _shift_subjects(shift: dict[str, Any]) -> tuple[str, ...]:
    """The subjects a shift changed: its activity, and the friend who moved away."""
    activity = str(shift["activity"])
    if shift["kind"] == "friend_moves":
        return activity, activity[len(render.DINNER_PREFIX):].capitalize()
    return (activity,)


def _mood_kind(f: int) -> str:
    kind = _fact_kind(f)
    if kind == "exam":
        return "exam" if f in [EXAM[c] for c in (0, 1, 2, 3)] else "none"
    return kind


def _triggers(rng: np.random.Generator, pop: Population, specs: list[Spec],
              steady: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Up to two edges source -> target; a target is sensitive and never a source, a source has
    no trigger of its own (and, chosen later, no streak)."""
    stems = [s.stem for s in specs]
    src = np.full(len(specs), -1)
    weight = np.zeros(len(specs))
    for i in rng.permutation(len(pop.triggers)):
        source, target, mean = pop.triggers[i]
        if source not in stems or target not in stems:
            continue
        s, t = stems.index(source), stems.index(target)
        if steady[t] or src[t] >= 0 or t in src or src[s] >= 0 or (src >= 0).sum() >= 2:
            continue
        if rng.random() < 0.6:
            src[t], weight[t] = s, _scaled(rng, mean, 0.25)
    return src, weight


def _shifts(rng: np.random.Generator, n_days: int, names: tuple[str, ...], specs: list[Spec], steady: np.ndarray,
            base: np.ndarray, pattern: np.ndarray, plan_rate: np.ndarray, by_weekday: np.ndarray,
            level: np.ndarray, weekdays: np.ndarray, days: list[tuple[int, ...]]) -> tuple[dict[str, Any], ...]:
    """0-2 permanent changes to sensitive activities, applied in place to the day arrays from their
    day on. A steady activity never shifts: the count family is the fixed weekday habit."""
    lo = max(7, n_days // 4)
    count = int(rng.choice(SHIFT_COUNTS[0], p=SHIFT_COUNTS[1])) if n_days - lo > lo else 0
    when = sorted(int(d) for d in rng.choice(np.arange(lo, n_days - lo), count, replace=False))
    sensitive = [a for a in range(len(specs)) if not steady[a]]
    used: set[int] = set()
    out = []
    for day in when:
        habits = [a for a in sensitive if not specs[a].planned and a not in used]
        if habits and rng.random() < 0.5:
            a = int(rng.choice(habits))
            old = days[a]
            while True:
                new = tuple(sorted(int(d) for d in rng.choice(7, len(old), replace=False)))
                if new != old:
                    break
            fresh = level[a] + rng.normal(0.0, WEEKDAY_SD, 7)
            pattern[a, day:] = np.isin(weekdays[day:], new)
            base[a, day:] = fresh[weekdays[day:]]
            out.append({"day": day, "kind": "new_schedule", "activity": names[a], "param": "weekdays",
                        "before": old, "after": new})
        else:
            a = int(rng.choice([a for a in sensitive if a not in used]))
            before = float(np.mean(by_weekday[a, list(days[a])]))
            base[a, day:] -= FADE
            if specs[a].planned:
                plan_rate[a, day:] *= FADE_PLANS
            kind = "friend_moves" if specs[a].stem == "dinner" else "less_time"
            out.append({"day": day, "kind": kind, "activity": names[a], "param": "base",
                        "before": before, "after": before - FADE})
        used.add(a)
    return tuple(out)
