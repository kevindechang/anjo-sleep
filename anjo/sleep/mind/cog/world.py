"""The cog world: ``make_life(seed)`` is one simulated person's 84 days, for MIND's cognition sprint.

A person has 6 activities (habits on weekday slots, or plans made 1-4 days ahead and then kept or
cancelled), 3-5 people and 3 distractor topics. Outcomes depend on the fixed weekday habit and, for
"sensitive" activities, on a hidden workload regime (calm/busy/crunch, a sticky Markov chain), on
hidden facts (an exam, a visitor, an injury), and on recent events (a streak of misses, a trigger
activity done yesterday). The regime shows only through late-night messages, mood and outcomes; a
fact only through weak hints and outcomes; distractor mentions show nothing. ``world_prior`` holds
the population priors and ``world_filter`` the exact references.

Everything is drawn at construction from the seed, on fixed-shape draws from one stream per purpose,
so a life is deterministic, and changing one outcome (``interventions``, for causal tests) changes
nothing before its day. Answers are noiseless (``ANSWER_NOISE``): they state the true regime, fact
state or topic value of their day. No clock, file, environment or network.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from anjo.sleep.mind.cog import render
from anjo.sleep.mind.cog.types import Event, LifeMeta, Query, Question, Resolution, Truth
from anjo.sleep.mind.cog.world_filter import forward, log_emissions, occurrence_probs, predictive
from anjo.sleep.mind.cog.world_prior import (
    EXAM, FACT_NONE, FACT_TOKENS, INJURY, MAX_STREAK, N_FACT, PLAN_LEAD_MAX, PRE, PRIORS, VISIT, Person,
    sample_person,
)

POPULATIONS: Final = ("base", "ood")
HORIZON: Final = 7  # queries are about the next 1-7 days
HORIZON_WEIGHTS: Final = np.log(np.array([4.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]))  # weighted to the next 1-2
QUERIES_PER_DAY: Final = (6, 10)
COSTS: Final = (0.05, 0.2)
# A day keeps at most this many events: daytime messages are trimmed first, then distractor
# mentions; neither enters the filter, and no day in 2000 lives needed the second.
MAX_EVENTS: Final = 15
ANSWER_NOISE: Final = 0.0
_STREAMS: Final = ("person", "chain", "outcomes", "plans", "signals", "distractors", "queries", "questions",
                   "text")
_SLOTS: Final = 32  # pre-drawn per-day text choices; a day has at most MAX_EVENTS events
_CANDIDATES: Final = 64  # pre-drawn per-day query keys; a day has at most 6 * HORIZON candidates
_VISIT_LEFT: Final = {state: r for r, state in VISIT.items()}
_INJURY_LEFT: Final = {state: r for r, state in INJURY.items()}

Item = tuple[str, str, tuple[str, ...], float | None]  # (kind, subject, tokens, value)
Said = tuple[int, str, tuple[str, ...], int]  # an answer: (day, subject, tokens, phrasing)


def make_life(seed: int, *, population: str = "base", n_days: int = 84) -> SimLife:
    return SimLife(seed, population=population, n_days=n_days)


class SimLife:
    """A ``Life`` (types.py), all of it computed at construction; the methods look it up."""

    def __init__(self, seed: int, *, population: str = "base", n_days: int = 84,
                 interventions: frozenset[tuple[str, int]] = frozenset()) -> None:
        if population not in POPULATIONS:
            raise ValueError(f"population must be one of {POPULATIONS}")
        if seed < 0 or n_days < 2:
            raise ValueError("a life needs a seed >= 0 and at least 2 days")
        seq = np.random.SeedSequence([seed, POPULATIONS.index(population)])
        rng = {name: np.random.default_rng(s) for name, s in zip(_STREAMS, seq.spawn(len(_STREAMS)), strict=True)}
        person = sample_person(rng["person"], PRIORS[population], n_days)
        flips = _flips(person, interventions, n_days)
        z = _hidden_path(rng["chain"], person, n_days)
        made = _plans(rng["plans"], person, n_days)
        y, occ, streak_in, prev_in = _outcomes(rng["outcomes"], person, z, made, flips)
        signals = _signals(rng["signals"], person, z)
        distract = _distractors(rng["distractors"], n_days)
        alpha = forward(person, log_emissions(person, y, occ, streak_in, prev_in, signals["late"],
                                              signals["mood"], signals["hints"]))
        self.meta = LifeMeta(person_id=f"{population}-{seed}", n_days=n_days, start_weekday=person.start_weekday,
                             activities=person.activities, people=person.people, topics=person.topics)
        self._person, self._alpha, self._z, self._y, self._occ, self._plan_made = person, alpha, z, y, occ, made
        pred = _references(person, alpha, z, made, y, streak_in)
        self._events = _all_events(rng["text"], person, y, occ, made, signals, distract)
        self._queries, self._resolutions = _all_queries(rng["queries"], self.meta.person_id, person, made, y, pred)
        self._questions, self._answers = _all_questions(rng["questions"], self.meta.person_id, person, z, distract)
        self._truth = _truth(population, person, z)

    def events(self, day: int) -> tuple[Event, ...]:
        return self._events[self._day(day)]

    def queries(self, day: int) -> tuple[Query, ...]:
        return self._queries[self._day(day)]

    def questions(self, day: int) -> tuple[Question, ...]:
        return self._questions[self._day(day)]

    def answer(self, question_id: str) -> Event:
        day, subject, tokens, choice = self._answers[question_id]
        return Event(day=day, kind="answer", subject=subject, text=render.event_text(tokens, choice), tokens=tokens)

    def resolve(self, qid: str) -> Resolution:
        return self._resolutions[qid]

    def truth(self) -> Truth:
        return self._truth

    def _day(self, day: int) -> int:
        if not 0 <= day < self.meta.n_days:
            raise ValueError(f"day must be in 0..{self.meta.n_days - 1}")
        return day


# ── the dynamics ──────────────────────────────────────────────────────────────────────────────
def _flips(person: Person, interventions: frozenset[tuple[str, int]], n_days: int) -> np.ndarray:
    flips: np.ndarray = np.zeros((n_days, len(person.activities)), dtype=bool)
    for activity, day in interventions:
        if activity not in person.activities or not 0 <= day < n_days:
            raise ValueError("an intervention names one of the life's activities and days")
        flips[day, person.activities.index(activity)] = True
    return flips


def _hidden_path(rng: np.random.Generator, person: Person, n_days: int) -> np.ndarray:
    u = rng.random(n_days)
    last = len(person.init) - 1
    cum_chain = np.cumsum(person.chain, axis=1)
    z: np.ndarray = np.empty(n_days, dtype=int)
    z[0] = min(int(np.searchsorted(np.cumsum(person.init), u[0], side="right")), last)
    for d in range(1, n_days):
        z[d] = min(int(np.searchsorted(cum_chain[z[d - 1]], u[d], side="right")), last)
    return z


def _plans(rng: np.random.Generator, person: Person, n_days: int) -> np.ndarray:
    """(A, T): the day the plan for each target day was made, or -1 for no plan."""
    n_act = len(person.activities)
    u, lead_u = rng.random((n_act, n_days)), rng.random((n_act, n_days))
    days = np.arange(n_days)
    span = np.maximum(np.minimum(PLAN_LEAD_MAX, days), 1)
    lead = 1 + np.floor(lead_u * span[None, :]).astype(int)
    ok = person.planned[:, None] & person.pattern & (u < person.plan_rate) & (days >= 1)[None, :]
    made: np.ndarray = np.where(ok, days[None, :] - lead, -1)
    return made


def _outcomes(rng: np.random.Generator, person: Person, z: np.ndarray, made: np.ndarray,
              flips: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_days, n_act = flips.shape
    u = rng.random((n_days, n_act))
    occ = np.where(person.planned[None, :], (made >= 0).T, person.pattern.T)
    y = np.zeros((n_days, n_act), dtype=bool)
    streak_in = np.zeros((n_days, n_act), dtype=int)
    prev_in = np.zeros((n_days, n_act))
    streak, prev = np.zeros((1, n_act), dtype=int), np.zeros((1, n_act))
    for d in range(n_days):
        streak_in[d], prev_in[d] = streak[0], prev[0]
        p = 1.0 / (1.0 + np.exp(-person.outcome_logits(d, z[d:d + 1], streak, prev)[0]))
        done = occ[d] & ((u[d] < p) ^ flips[d])
        y[d] = done
        streak = np.where(done, 0, np.where(occ[d], np.minimum(streak + 1, MAX_STREAK), streak))
        prev = done[None, :].astype(float)
    return y, occ, streak_in, prev_in


def _signals(rng: np.random.Generator, person: Person, z: np.ndarray) -> dict[str, np.ndarray]:
    n_days = len(z)
    u_mood, u_late = rng.random(n_days), rng.random((n_days, 2))
    u_hint, u_day = rng.random((n_days, len(render.HINTS))), rng.random((n_days, 2))
    who = rng.integers(0, len(person.people), (n_days, 8))
    mood = np.clip(1 + (u_mood[:, None] > np.cumsum(person.mood_p[z], axis=1)).sum(axis=1), 1, 5)
    return {"mood": mood, "late": (u_late < person.late_p[z][:, None]).sum(axis=1),
            "hints": (u_hint < person.hint_p[z]).astype(float), "daytime": 2 + (u_day < 0.5).sum(axis=1),
            "who": who}


def _distractors(rng: np.random.Generator, n_days: int) -> dict[str, np.ndarray]:
    """Each topic's value each day (uniform, independent of everything) and which are mentioned."""
    return {"value": rng.random((n_days, 3)), "count": 2 + (rng.random(n_days) < 0.5),
            "order": np.argsort(rng.random((n_days, 3)), axis=1)}


def _topic_value(topic: str, u: float) -> str:
    values = render.TOPIC_VALUES[render.topic_kind(topic)]
    return values[int(u * len(values))]


def _references(person: Person, alpha: np.ndarray, z: np.ndarray, made: np.ndarray, y: np.ndarray,
                streak_in: np.ndarray) -> np.ndarray:
    """(A, 2 * n_ask, HORIZON): rows 0..n_ask-1 are oracle_p, the rest state_p, per asked day."""
    n_ask = len(z) - 1
    rows_t = np.concatenate([np.arange(n_ask), np.arange(n_ask)])
    starts = np.vstack([alpha[:n_ask], np.eye(alpha.shape[1])[z[:n_ask]]])
    occ = np.tile(occurrence_probs(person, made, n_ask, HORIZON), (1, 2, 1))
    streak0 = np.tile(streak_in[1:], (2, 1))
    prev0 = np.tile(y[:n_ask].astype(float), (2, 1))
    return predictive(person, starts, rows_t, occ, streak0, prev0, HORIZON)


# ── what the person says ──────────────────────────────────────────────────────────────────────
def _all_events(rng: np.random.Generator, person: Person, y: np.ndarray, occ: np.ndarray, made: np.ndarray,
                signals: dict[str, np.ndarray], distract: dict[str, np.ndarray]) -> tuple[tuple[Event, ...], ...]:
    n_days = len(y)
    choices = rng.integers(0, render.BANK_SPAN, (n_days, _SLOTS))
    order = rng.random((n_days, _SLOTS))
    plans_on: list[list[tuple[int, int]]] = [[] for _ in range(n_days)]
    for a, target in zip(*np.nonzero(made >= 0), strict=True):
        plans_on[made[a, target]].append((int(a), int(target)))
    out = []
    for d in range(n_days):
        items = _day_items(d, person, y, occ, plans_on[d], signals, distract)
        if len(items) > _SLOTS:
            raise RuntimeError("a day outgrew its pre-drawn text choices")
        rank = np.argsort(order[d, :len(items)])
        out.append(tuple(Event(day=d, kind=items[i][0], subject=items[i][1],
                               text=render.event_text(items[i][2], int(choices[d, k])), tokens=items[i][2],
                               value=items[i][3]) for k, i in enumerate(rank)))
    return tuple(out)


def _day_items(d: int, person: Person, y: np.ndarray, occ: np.ndarray, plans: list[tuple[int, int]],
               signals: dict[str, np.ndarray], distract: dict[str, np.ndarray]) -> list[Item]:
    level = int(signals["mood"][d])
    items: list[Item] = [("mood", "mood", ("mood", f"level={level}"), float(level))]
    dow = f"dow={person.weekday(d)}"
    for a, name in enumerate(person.activities):
        if occ[d, a]:
            kind = "did" if y[d, a] else "cancel" if person.planned[a] else "skipped"
            items.append((kind, name, (kind, name, dow), None))
    for a, target in plans:
        name = person.activities[a]
        items.append(("plan", name, ("plan", name, f"in={target - d}", f"dow={person.weekday(target)}"),
                      float(target)))
    who = iter(signals["who"][d])
    for _ in range(int(signals["late"][d])):
        name = person.people[next(who)]
        items.append(("message", name, ("message", name, "late"), None))
    for j, (subject, value) in enumerate(render.HINTS):
        if signals["hints"][d, j]:
            items.append(("mention", subject, ("mention", subject, value), None))
    room = max(0, MAX_EVENTS - len(items))  # on a day already this full, distractors give way
    for j in distract["order"][d, :min(int(distract["count"][d]), room)]:
        topic = person.topics[j]
        items.append(("mention", topic, ("mention", topic, _topic_value(topic, float(distract["value"][d, j]))),
                      None))
    for _ in range(min(int(signals["daytime"][d]), max(0, MAX_EVENTS - len(items)))):
        name = person.people[next(who)]
        items.append(("message", name, ("message", name, "day"), None))
    return items


# ── queries and questions ─────────────────────────────────────────────────────────────────────
def _all_queries(rng: np.random.Generator, pid: str, person: Person, made: np.ndarray, y: np.ndarray,
                 pred: np.ndarray) -> tuple[tuple[tuple[Query, ...], ...], dict[str, Resolution]]:
    n_days = len(y)
    n_ask = n_days - 1
    sizes = rng.integers(QUERIES_PER_DAY[0], QUERIES_PER_DAY[1] + 1, n_days)
    coin = rng.random(n_days) < 0.5
    gumbel = -np.log(-np.log(rng.random((n_days, _CANDIDATES))))
    choices = rng.integers(0, render.BANK_SPAN, (n_days, QUERIES_PER_DAY[1]))
    days, resolutions = [], {}
    for t in range(n_days):
        picked = _pick_queries(t, person, made, int(sizes[t]), bool(coin[t]), gumbel[t]) if t < n_ask else []
        queries = []
        for i, (h, a, kind) in enumerate(picked):
            target, name = t + h, person.activities[a]
            tokens = ("query", kind, name, f"in={h}", f"dow={person.weekday(target)}")
            q = Query(qid=f"{pid}/d{t}/q{i}", asked_day=t, target_day=target, subject=name,
                      text=render.query_text(tokens, int(choices[t, i])), tokens=tokens,
                      family="count" if person.steady[a] else "hidden")
            queries.append(q)
            resolutions[q.qid] = Resolution(qid=q.qid, outcome=bool(y[target, a]),
                                            oracle_p=float(pred[a, t, h - 1]), state_p=float(pred[a, n_ask + t, h - 1]))
        days.append(tuple(queries))
    return tuple(days), resolutions


def _pick_queries(t: int, person: Person, made: np.ndarray, size: int, coin: bool,
                  gumbel: np.ndarray) -> list[tuple[int, int, str]]:
    """Up to ``size`` queries, half per family where the candidates allow, nearer days likelier."""
    pools: dict[bool, list[tuple[float, tuple[int, int, str]]]] = {True: [], False: []}
    n_days = person.base.shape[1]
    for h in range(1, min(HORIZON, n_days - 1 - t) + 1):
        target = t + h
        for a in range(len(person.activities)):
            if person.planned[a]:
                ok, kind = 0 <= made[a, target] <= t, "kept"
            else:
                ok, kind = bool(person.pattern[a, target]), "happen"
            if ok:
                key = HORIZON_WEIGHTS[h - 1] + gumbel[len(pools[True]) + len(pools[False])]
                pools[bool(person.steady[a])].append((float(key), (h, a, kind)))
    for pool in pools.values():
        pool.sort(key=lambda item: -item[0])
    want_hidden = size // 2 + (int(coin) if size % 2 else 0)
    hidden = min(want_hidden, len(pools[False]))
    count = min(size - hidden, len(pools[True]))
    hidden = min(size - count, len(pools[False]))
    picked = [c for _, c in pools[False][:hidden]] + [c for _, c in pools[True][:count]]
    return sorted(picked)


def _all_questions(rng: np.random.Generator, pid: str, person: Person, z: np.ndarray,
                   distract: dict[str, np.ndarray]) -> tuple[tuple[tuple[Question, ...], ...], dict[str, Said]]:
    n_days = len(z)
    n_topics = rng.integers(1, len(person.topics) + 1, n_days)
    topic_order = np.argsort(rng.random((n_days, len(person.topics))), axis=1)
    costs = np.round(rng.uniform(COSTS[0], COSTS[1], (n_days, 5)), 2)
    asked, said = rng.integers(0, render.BANK_SPAN, (n_days, 5)), rng.integers(0, render.BANK_SPAN, (n_days, 5))
    order = np.argsort(rng.random((n_days, 5)), axis=1)
    options = {s: render.answer_options(s) for s in ("work", "plans", *person.topics)}
    days, answers = [], {}
    for d in range(n_days):
        subjects = ["work", "plans", *(person.topics[j] for j in sorted(topic_order[d, :n_topics[d]]))]
        truth = {"work": ("answer", "work", render.REGIMES[z[d] // N_FACT]), "plans": FACT_TOKENS[z[d] % N_FACT]}
        for j, topic in enumerate(person.topics):
            truth[topic] = ("answer", topic, _topic_value(topic, float(distract["value"][d, j])))
        questions = []
        for i, k in enumerate(o for o in order[d] if o < len(subjects)):
            subject = subjects[k]
            qid = f"{pid}/d{d}/x{i}"
            questions.append(Question(question_id=qid, day=d, subject=subject,
                                      text=render.question_text(("ask", subject), int(asked[d, i])),
                                      tokens=("ask", subject), cost=float(costs[d, i]), options=options[subject]))
            answers[qid] = (d, subject, truth[subject], int(said[d, i]))
        days.append(tuple(questions))
    return tuple(days), answers


# ── the truth ─────────────────────────────────────────────────────────────────────────────────
def _hidden_facts(facts: np.ndarray) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d, f in enumerate(int(f) for f in facts):
        if f == FACT_NONE or (d > 0 and facts[d - 1] != FACT_NONE):
            continue
        fid = f"fact{len(out) + 1}"
        if f == EXAM[render.EXAM_DAYS]:
            out[fid] = {"kind": "exam", "known_from": d, "day": d + render.EXAM_DAYS}
        elif f == PRE[render.VISITOR_LEAD]:
            arrive = d + render.VISITOR_LEAD
            stay = _VISIT_LEFT.get(int(facts[arrive])) if arrive < len(facts) else None
            out[fid] = {"kind": "visitor", "known_from": d, "arrive": arrive,
                        "leave": None if stay is None else arrive + stay - 1}
        else:
            out[fid] = {"kind": "injury", "start": d, "end": d + _INJURY_LEFT[f] - 1}
    return out


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _truth(population: str, person: Person, z: np.ndarray) -> Truth:
    params = {"population": population, **person.report,
              "regime_path": tuple(render.REGIMES[s // N_FACT] for s in z),
              "fact_path": tuple(FACT_TOKENS[s % N_FACT] for s in z)}
    return Truth(shift_days=tuple(s["day"] for s in person.report["shifts"]),
                 hidden_facts=_freeze(_hidden_facts(z % N_FACT)), params=_freeze(params))
