"""An independent world of ordinary lives, for testing predictors of behaviour.

``make_life(seed, n_days=84)`` simulates one person for 12 weeks and returns a ``Life``
(``anjo.sleep.mind.cog.types``). Everything is drawn from a seed with numpy; one random stream per
day and purpose, so nothing on day d depends on later days (a shorter run is a prefix of a longer).

What the people are like (the author's choices)
-----------------------------------------------
* Six archetypes (office worker, student, shift worker, parent, retiree, freelancer) set the pool
  of activities, sociability (about 0.5 to 6 plans a week), work or study pressure and regularity.
  One person in eight is chaotic (regularity 0.1-0.3); retirees run like clockwork (0.7-0.98).
* Weekly rhythm: 2-4 activities the person's life can push around (fixed classes, k-days-a-week
  habits, weekend outings, daily cooking), each with a per-weekday rate on a good day, blurred by
  regularity. Shift workers and a third of parents (custody weekends) run on a two-week rota.
* 1-3 robust habits (reading, Duolingo, a weekly family call...) follow fixed per-weekday rates and
  nothing else: no state, trip, illness or plan moves them. Queries about them are "count".
* Hidden state, one joint Markov chain over the factors this person has: health (well / run-down
  / sick; colds come more in winter and in crunch), work pressure (normal / crunch, mean 5-12
  days), a low-mood spell (55% of people), weather (fair / wet, seasonal, persistent), engagement
  in up to two fickle habits (lapsed / active / keen: honeymoons fade, habits lapse, some get
  taken up), a life change (45%: new job, move, relationship, therapy, semester, health kick;
  often announced 8-24 days ahead, sometimes silent), a key friend's availability (around / busy;
  partnered sports need them) and an injury (keen people overtrain).
* Exogenous: trips (booked 2-21 days ahead and 8% cancelled on the day, or spontaneous), public
  holidays (mentioned 7-10 days ahead), the season.
* Plans with people are made 0-7 days ahead at a rate the person's state moves (crunch, low mood
  and illness cut it). Whether a plan happens depends on that day's state, the friend's
  reliability (0.7-0.97), a hidden per-plan shakiness (hinted by hedges and "might have to push"
  messages), weather for outdoor plans and travel. Any plan that day crowds out evening
  activities. Cancellations are only ever reported on the day itself.
* Talk: mood reports (15-90% of days), work, health, weather and injury talk, enthusiasm or guilt
  about a habit, messages from the key friend and the boss, intentions ("gym tomorrow") that bind
  nothing, and noise about TV, food, pets, football... that predicts nothing.

Families: a query is "count" iff it is about a robust habit; activity and plan queries are
"hidden", since every one of them depends on the hidden chain (and plans, trips, holidays).

References: ``oracle_p`` is an exact forward filter over the joint chain, conditioned on every
event through the asked day (not on answers to questions, which depend on the inquirer), then an
exact forward pass that also marginalises plans not yet made and trips not yet booked.
``state_p`` runs the same pass from the true hidden state (and true plan shakiness). Both use the
same probability tables the simulation draws from, so the oracle is exact, not approximate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import numpy as np

from anjo.sleep.mind.cog import world_indep_text as tx
from anjo.sleep.mind.cog.types import Event, LifeMeta, Query, Question, Resolution, Truth

HORIZON = 7
MAX_LEAD = 7
P_CLIP = (0.004, 0.985)
P_CANCEL_TRIP = 0.08
MAX_STATES = 1600
LEVELS = np.arange(1.0, 6.0)
LEAD_W = np.array([0.22, 0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06])
DOW_PLAN_W = np.array([0.45, 0.5, 0.6, 0.8, 1.6, 1.85, 1.2])
B, BN, BP, F, A = range(5)  # life-change phase: before, announced today, pending, first day, after
LAPSED, ACTIVE, KEEN = range(3)

HEALTH_MULT = {"physical": (1, 0.65, 0.06), "class": (1, 0.75, 0.1), "social": (1, 0.8, 0.15),
               "chore": (1, 0.85, 0.4), "outing": (1, 0.75, 0.1)}
MOOD_MULT = {"physical": 0.55, "class": 0.6, "social": 0.5, "chore": 0.7, "outing": 0.6}
CRUNCH_WORKDAY = {"evening": 0.45, "morning": 0.8, "day": 0.7}
KAPPA = {"evening": 0.4, "day": 0.6, "morning": 0.95}
ENG_MULT = (0.07, 1.0, 1.3)
HEDGE = (0.1, 0.45)  # P(plan worded tentatively | shaky)
PLAN_MSG = np.array([[0.83, 0.02, 0.15], [0.70, 0.25, 0.05]])  # none, doubt, confirm | shaky

# Talk channels: P(content | factor value); column 0 is "said nothing".
CH_WORK = np.array([[0.82, 0.04, 0.06, 0.05, 0.03], [0.53, 0.25, 0.20, 0.02, 0.0]])
CH_BOSS = np.array([[0.92, 0.03, 0.05], [0.65, 0.30, 0.05]])


def _quieter(table: np.ndarray, factor: float) -> np.ndarray:
    """The same channel on a day off: everything but silence scaled down."""
    out = table.copy()
    out[:, 1:] *= factor
    out[:, 0] = 1 - out[:, 1:].sum(1)
    return out


CH_WORK_OFF, CH_BOSS_OFF = _quieter(CH_WORK, 0.35), _quieter(CH_BOSS, 0.25)
CH_HEALTH = np.array([[0.925, 0.005, 0.03, 0.04], [0.70, 0.04, 0.25, 0.01], [0.35, 0.55, 0.10, 0.0]])
CH_INJURY = np.array([[0.995, 0.005], [0.7, 0.3]])
CH_WEATHER = np.array([[0.87, 0.01, 0.12], [0.745, 0.25, 0.005]])
CH_KEY = np.array([[0.80, 0.02, 0.10, 0.08], [0.67, 0.25, 0.02, 0.06]])
CH_KEEN = np.array([[0.935, 0.005, 0.06], [0.95, 0.04, 0.01], [0.82, 0.18, 0.0]])
CH_CHANGE = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0.95, 0, 0.05, 0, 0],
                      [0.25, 0, 0, 0.75, 0], [0.96, 0, 0, 0, 0.04]], dtype=float)
WORK_CONTENT = ("", "stressed", "busy", "fine", "quiet")
HEALTH_CONTENT = ("", "sick", "tired", "great")

# Answers: P(option | true state), options as in tx.QUESTION_OPTIONS.
ANS_HEALTH = np.array([[0.82, 0.16, 0.02], [0.25, 0.65, 0.10], [0.05, 0.25, 0.70]])
ANS_MOOD = np.array([[0.55, 0.40, 0.05], [0.08, 0.42, 0.50]])
ANS_WORK = np.array([[0.6, 0.35, 0.05], [0.05, 0.35, 0.6]])
ANS_INJURY = np.array([[0.85, 0.13, 0.02], [0.05, 0.3, 0.65]])
ANS_CHANGE = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.12, 0.05, 0.83]])
ANS_HABIT = np.array([[0.02, 0.18, 0.8], [0.15, 0.75, 0.10], [0.75, 0.23, 0.02]])
ANS_FRIEND = np.array([[0.85, 0.15], [0.15, 0.85]])
ANS_PLAN = np.array([[0.8, 0.17, 0.03], [0.25, 0.4, 0.35]])


def _trip_cover() -> tuple[np.ndarray, np.ndarray]:
    """P(a trip decided j days before a day covers it): booked (lead 2-21, 2-8 days), spontaneous."""
    booked = np.zeros(HORIZON + 1)
    for j in range(HORIZON + 1):
        booked[j] = sum(np.mean(np.arange(2, 9) > j - lead) for lead in range(2, min(j, 21) + 1)) / 20
    spont = np.array([np.mean(np.arange(2, 5) > j) for j in range(HORIZON + 1)])
    return booked, spont


TRIP_BOOKED, TRIP_SPONT = _trip_cover()


def _cat(rng: np.random.Generator, p: np.ndarray) -> int:
    c: np.ndarray = np.cumsum(p)
    return min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(c) - 1)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---- the person --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Act:
    name: str
    spec: tx.Spec
    slot: str
    base: np.ndarray  # (2 phases, 14 day types): the rate on a good day
    fickle: int  # index of its engagement factor, or -1
    eng_init: tuple[float, ...]
    eng_t: tuple[float, ...]  # lapsed->keen, active->lapsed, active->keen, keen->active
    away: float
    kappa: float


@dataclass(frozen=True)
class _Count:
    name: str
    probs: np.ndarray  # (7,)


@dataclass(frozen=True)
class _Friend:
    name: str
    role: str
    w_before: float
    w_after: float
    reliability: float
    flaky: float
    chat: float


@dataclass(frozen=True)
class _Person:
    pid: str
    archetype: str
    start_weekday: int
    start_doy: int
    rota: bool
    regularity: float
    acts: tuple[_Act, ...]
    counts: tuple[_Count, ...]
    people: tuple[_Friend, ...]
    key: int
    boss: int
    work_topic: str
    topics: tuple[str, ...]
    noise: tuple[tuple[str, float], ...]
    has_m: bool
    has_q: bool
    has_i: bool
    injury_act: int
    change: str
    health: tuple[float, ...]  # well->rundown, well->sick, rundown->well, rundown->sick, sick->well, sick->rundown
    crunch_mult: float
    winter_amp: float
    crunch: tuple[float, ...]  # on, off, on after the change, off after
    lowmood: tuple[float, ...]
    weather: tuple[float, ...]  # base wet share, seasonal amplitude, persistence
    friend_t: tuple[float, ...]
    injury_t: tuple[float, ...]  # on, off, keen multiplier
    change_h: tuple[float, float]  # announced, silent hazards
    soc: tuple[float, float]
    plan_dow: np.ndarray  # (2 rota weeks, 7)
    trips: tuple[float, float]
    mood: tuple[float, ...]  # baseline, spread, shift after the change, report rate
    tau: float  # gives a reason when skipping
    iota: float  # states intentions
    holidays: tuple[tuple[int, int], ...]  # (day, days of notice)

    @property
    def has_w(self) -> bool:
        return bool(self.work_topic)

    @property
    def has_p(self) -> bool:
        return bool(self.change)

    @property
    def n_fickle(self) -> int:
        return sum(a.fickle >= 0 for a in self.acts)


def _profile(rng: np.random.Generator, sched: str, slot: str, reg: float) -> np.ndarray:
    p = np.empty(7)
    if sched == "fixed":
        n = 1 if rng.random() < 0.6 else 2
        days = rng.choice(5 if slot == "evening" else 7, size=n, replace=False)
        p[:] = rng.uniform(0.0, 0.04, 7)
        p[days] = rng.uniform(0.75, 0.95, n)
    elif sched == "freq":
        k = int(rng.integers(1, 6))
        days = rng.choice(7, size=k, replace=False)
        p[:] = rng.uniform(0.02, 0.15, 7)
        p[days] = rng.uniform(0.55, 0.9, k)
    elif sched == "weekend":
        p[:5] = rng.uniform(0.0, 0.08, 5)
        p[5:] = rng.uniform(0.2, 0.85, 2)
    elif sched == "sunday":
        p[:] = rng.uniform(0.0, 0.05, 7)
        p[5], p[6] = _u(rng, 0.0, 0.25), _u(rng, 0.7, 0.95)
    else:
        p[:5] = rng.uniform(0.45, 0.85, 5)
        p[5:] = rng.uniform(0.3, 0.8, 2)
    flat = p.mean() * rng.uniform(0.7, 1.3, 7)
    return np.clip(reg * p + (1 - reg) * flat, 0.0, 0.95)


def _rota(rng: np.random.Generator, week: np.ndarray, spec: tx.Spec, slot: str, arch: str) -> np.ndarray:
    b = week.copy()
    if arch == "shift":  # nights week: evenings gone, mornings slept through
        b[:5] *= _u(rng, 0.15, 0.5) if slot == "evening" else _u(rng, 0.4, 0.8)
        b[5:] *= _u(rng, 1.0, 1.3)
    elif spec.kind == "social" and spec.venue:  # kids' weekend: church and grandma, with them
        b[5:] *= 1.3
    else:
        b[5:] *= _u(rng, 0.2, 0.5)
    return np.clip(b, 0.0, 0.95)


def _after_change(rng: np.random.Generator, b0: np.ndarray, spec: tx.Spec, change: str,
                  boosted: bool) -> np.ndarray:
    b = b0.copy()
    if change == "new job":
        b = np.tile(b[:7], 2)  # a new schedule, and no rota
        if rng.random() < 0.5:
            b[:5] = np.roll(b[:5], int(rng.choice([-2, -1, 1, 2])))
            b[7:12] = b[:5]
        b *= _u(rng, 0.6, 1.1)
    elif change == "moving":
        b *= _u(rng, 0.15, 0.6) if spec.venue else (_u(rng, 1.0, 1.4) if spec.outdoor else 1.0)
    elif change == "relationship" and spec.kind in ("physical", "class", "outing"):
        b *= _u(rng, 0.65, 0.9)
    elif change == "therapy" and spec.kind == "physical":
        b *= _u(rng, 1.0, 1.2)
    elif change == "semester" and spec.kind == "social":
        b *= _u(rng, 0.7, 0.9)
    elif change == "health kick" and boosted:
        b = np.maximum(b * _u(rng, 1.8, 3.0), _u(rng, 0.15, 0.35))
    return np.clip(b, 0.0, 0.95)


def _away_mult(spec: tx.Spec) -> float:
    if spec.venue or spec.kind == "class":
        return 0.02
    return {"physical": 0.2, "chore": 0.1, "outing": 0.15, "social": 0.05}[spec.kind]


def _draw_acts(rng: np.random.Generator, arch: tx.Archetype, arch_name: str, reg: float,
               rota: bool, change: str) -> tuple[tuple[_Act, ...], tuple[_Count, ...]]:
    names = [str(x) for x in rng.choice(arch.pool, size=int(rng.integers(2, 5)), replace=False)]
    boost = int(rng.integers(len(names)))
    acts, n_fickle = [], 0
    for i, name in enumerate(names):
        spec = tx.STATEFUL[name]
        slot = spec.slot
        if spec.sched == "freq" and spec.kind == "physical":
            slot = "morning" if rng.random() < 0.5 else "evening"
        week = _profile(rng, spec.sched, slot, reg)
        b0 = np.concatenate([week, _rota(rng, week, spec, slot, arch_name) if rota else week])
        b1 = _after_change(rng, b0, spec, change, i == boost) if change else b0
        fickle = -1
        eng_init: tuple[float, ...] = (0.0, 1.0, 0.0)
        eng_t: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
        if n_fickle < 2 and rng.random() < 0.35:
            fickle, n_fickle = n_fickle, n_fickle + 1
            aspiring = rng.random() < 0.35
            eng_init = (0.88, 0.1, 0.02) if aspiring else (0.05, 0.7, 0.25)
            eng_t = (_u(rng, 0.01, 0.03) if aspiring else _u(rng, 0.004, 0.02), _u(rng, 0.004, 0.02),
                     _u(rng, 0.002, 0.008), _u(rng, 0.03, 0.1))
        acts.append(_Act(name, spec, slot, np.stack([b0, b1]), fickle, eng_init, eng_t,
                         _away_mult(spec), KAPPA[slot]))
    counts = []
    for name in rng.choice(list(tx.COUNT_SCHED), size=int(rng.integers(1, 4)), replace=False):
        sched = tx.COUNT_SCHED[str(name)]
        if sched == "daily":
            week = rng.uniform(0.4, 0.92, 7) * np.where(np.arange(7) >= 5, _u(rng, 0.7, 1.1), 1.0)
        else:
            week = _profile(rng, sched, "day", max(reg, 0.5))
        counts.append(_Count(str(name), np.clip(week, 0.01, 0.97)))
    return tuple(acts), tuple(counts)


def _draw_people(rng: np.random.Generator, arch_name: str, has_work: bool, change: str,
                 has_q: bool) -> tuple[tuple[_Friend, ...], int, int]:
    names = [str(x) for x in rng.permutation(tx.FIRST_NAMES)]
    out: list[_Friend] = []

    def add(name: str, role: str, wb: float, wa: float, rel: tuple[float, float],
            flaky: tuple[float, float], chat: tuple[float, float]) -> None:
        out.append(_Friend(name, role, wb, wa, _u(rng, *rel), _u(rng, *flaky), _u(rng, *chat)))

    family = [names.pop(), names.pop()] if arch_name == "retiree" else ["Mom", "Dad"][: 1 + int(rng.random() < 0.45)]
    if arch_name != "retiree" and rng.random() < 0.55:
        family.append(names.pop())
    for name in family:
        add(name, "family", 0.6, 0.6, (0.85, 0.97), (0.03, 0.15), (0.03, 0.15))
    for i in range(int(rng.integers(2, 4))):
        w = 1.4 if i == 0 and has_q else 1.0
        add(names.pop(), "friend", w, w, (0.82, 0.98), (0.03, 0.3), (0.02, 0.12))
    key = len(family) if has_q else -1
    if change == "relationship":
        add(names.pop(), "date", 0.3, 3.0, (0.75, 0.95), (0.1, 0.3), (0.03, 0.1))
    elif rng.random() < 0.45:
        add(names.pop(), "partner", 1.6, 1.6, (0.85, 0.97), (0.03, 0.15), (0.05, 0.2))
    boss = -1
    if has_work:
        boss = len(out)
        add(names.pop(), "boss", 0.25, 0.25, (0.8, 0.95), (0.05, 0.2), (0.0, 0.03))
    return tuple(out), key, boss


def _draw_person(seed: int) -> _Person:
    rng = np.random.default_rng([seed, 0])
    arch_names = list(tx.ARCHETYPES)
    weights = np.array([tx.ARCHETYPES[n].weight for n in arch_names])
    arch_name = arch_names[_cat(rng, weights / weights.sum())]
    arch = tx.ARCHETYPES[arch_name]
    reg = _u(rng, *arch.reg) if rng.random() > 0.12 else _u(rng, 0.1, 0.3)
    rota = arch_name == "shift" or (arch_name == "parent" and rng.random() < 0.35)
    change = str(rng.choice(arch.changes)) if rng.random() < 0.45 else ""
    acts, counts = _draw_acts(rng, arch, arch_name, reg, rota, change)
    has_q = any(a.spec.partner for a in acts) or rng.random() < 0.35
    injury_acts = [i for i, a in enumerate(acts) if a.spec.injury]
    has_i = bool(injury_acts) and rng.random() < 0.7
    injury_act = int(rng.choice(injury_acts)) if has_i else -1
    has_m = rng.random() < 0.55
    work_topic = arch.work
    people, key, boss = _draw_people(rng, arch_name, bool(work_topic), change, has_q)
    k = 3 * 2 * (2 if work_topic else 1) * (2 if has_m else 1) * 3 ** sum(a.fickle >= 0 for a in acts)
    k *= (5 if change else 1) * (2 if has_q else 1) * (2 if has_i else 1)
    if k > MAX_STATES:  # keep the exact filter cheap: one fickle habit at most
        acts = tuple(a if a.fickle < 1 else _Act(a.name, a.spec, a.slot, a.base, -1, (0.0, 1.0, 0.0),
                                                   (0.0,) * 4, a.away, a.kappa) for a in acts)
    c_on, c_off = _u(rng, *arch.crunch_on), _u(rng, *arch.crunch_off)
    c_on_after, c_off_after = c_on, c_off
    m_on, m_off = _u(rng, 0.004, 0.02), _u(rng, 0.05, 0.14)
    m_on_after, m_off_after = m_on, m_off
    soc = _u(rng, *arch.soc)
    soc_after, mu_after = soc, 0.0
    if change == "new job":
        c_on_after, soc_after = c_on * _u(rng, 0.7, 2.0), soc * _u(rng, 0.8, 1.1)
    elif change == "semester":
        c_on_after, c_off_after, soc_after, mu_after = c_on * _u(rng, 1.5, 2.5), c_off * 0.8, soc * 1.1, -0.1
    elif change == "relationship":
        m_on_after, soc_after, mu_after = m_on * 0.5, soc * _u(rng, 1.2, 1.6), 0.3
    elif change == "therapy":
        m_on_after, m_off_after, mu_after = m_on * 0.35, min(0.4, m_off * 2), 0.2
    elif change == "moving":
        m_on_after, soc_after, mu_after = m_on * 1.3, soc * _u(rng, 0.7, 1.0), -0.1
    elif change == "health kick":
        mu_after = 0.1
    plan_dow = np.tile(DOW_PLAN_W * rng.uniform(0.8, 1.2, 7), (2, 1))
    if rota:
        plan_dow[1, 5:] *= 0.4 if arch_name == "parent" else 1.0
        plan_dow[1, :5] *= 0.5 if arch_name == "shift" else 1.0
    hazard = _u(rng, 0.01, 0.035)
    announced = tx.CHANGE_ANNOUNCED.get(change, 0.0)
    noise_topics = rng.choice(tx.NOISE_TOPICS, size=int(rng.integers(3, 6)), replace=False)
    noise = tuple((str(t), _u(rng, 0.02, 0.15)) for t in noise_topics) + (
        ("family", _u(rng, 0.02, 0.08)), ("home", _u(rng, 0.02, 0.08)))
    core = ([work_topic] if work_topic else []) + ["health", "weather", "travel", "home", "family",
                                                    "relationships", "holidays"]
    topics = tuple(str(t) for t in rng.permutation(core + [str(t) for t in noise_topics]))
    start_weekday = int(rng.integers(7))
    cands = [d for d in range(10, 130) if (start_weekday + d) % 7 in (0, 4)]
    hol_days = sorted(int(d) for d in rng.choice(cands, size=int(rng.integers(0, 3)), replace=False))
    holidays = tuple((d, int(rng.integers(7, 11))) for d in hol_days)
    return _Person(
        pid=f"indep-{seed}", archetype=arch_name, start_weekday=start_weekday,
        start_doy=int(rng.integers(365)), rota=rota, regularity=reg, acts=acts, counts=counts,
        people=people, key=key, boss=boss, work_topic=work_topic, topics=topics, noise=noise,
        has_m=has_m, has_q=has_q, has_i=has_i, injury_act=injury_act, change=change,
        health=(_u(rng, 0.015, 0.05), _u(rng, 0.003, 0.012) * arch.sick, _u(rng, 0.2, 0.4),
                _u(rng, 0.03, 0.08), _u(rng, 0.15, 0.3), _u(rng, 0.08, 0.2)),
        crunch_mult=_u(rng, 1.5, 2.5), winter_amp=_u(rng, 0.2, 0.7),
        crunch=(c_on, c_off, c_on_after, c_off_after), lowmood=(m_on, m_off, m_on_after, m_off_after),
        weather=(_u(rng, 0.12, 0.42), _u(rng, 0.0, 0.12), _u(rng, 0.3, 0.65)),
        friend_t=(_u(rng, 0.02, 0.07), _u(rng, 0.08, 0.25)),
        injury_t=(_u(rng, 0.002, 0.008), _u(rng, 0.04, 0.12), _u(rng, 2.0, 3.5)),
        change_h=(hazard * announced, hazard * (1 - announced)), soc=(soc, soc_after),
        plan_dow=plan_dow, trips=(_u(rng, *arch.trips), _u(rng, 0.001, 0.01)),
        mood=(_u(rng, 2.7, 4.0), _u(rng, 0.5, 1.1), mu_after, _u(rng, 0.15, 0.9)),
        tau=_u(rng, 0.2, 0.8), iota=_u(rng, 0.03, 0.3), holidays=holidays,
    )


# ---- the model: the joint hidden chain and every probability drawn from it ---------------


class _Model:
    """Hidden factors as one tensor, parents on earlier axes than their children; a day's
    transition updates each factor from the old values of itself and its parent, children first."""

    def __init__(self, person: _Person, n_ext: int) -> None:
        self.p = p = person
        fac = ([("P", 5)] if p.has_p else []) + ([("W", 2)] if p.has_w else []) + [("H", 3)]
        fac += ([("M", 2)] if p.has_m else []) + [(f"E{i}", 3) for i in range(p.n_fickle)]
        fac += ([("I", 2)] if p.has_i else []) + [("R", 2)] + ([("Q", 2)] if p.has_q else [])
        self.names = tuple(n for n, _ in fac)
        self.sizes = tuple(s for _, s in fac)
        self.pos = {n: i for i, n in enumerate(self.names)}
        self.K = int(np.prod(self.sizes))
        grid = np.indices(self.sizes).reshape(len(self.sizes), -1)
        self.idx = {n: grid[i] for n, i in self.pos.items()}
        self._calendar(n_ext)
        self._chain(n_ext)
        self._statics()

    # -- calendar
    def _calendar(self, n_ext: int) -> None:
        p, m = self.p, np.arange(n_ext)
        self.dow: np.ndarray = (p.start_weekday + m) % 7
        self.parity: np.ndarray = ((p.start_weekday + m) // 7) % 2 if p.rota else np.zeros(n_ext, int)
        self.holiday = np.isin(m, [d for d, _ in p.holidays])
        self.weekend = self.dow >= 5
        self.workday = ~self.weekend & ~self.holiday
        winter = 0.5 * (1 + np.cos(2 * np.pi * (((p.start_doy + m) % 365) - 15) / 365))
        self.warmth = 1 - winter
        self.wet_pi = np.clip(p.weather[0] + p.weather[1] * (2 * winter - 1), 0.05, 0.75)
        self.sick_season = 1 + p.winter_amp * (2 * winter - 1)
        dt = self.parity * 7 + self.dow
        self.bday = []
        for act in p.acts:
            b, sat = act.base[:, dt].T, act.base[:, self.parity * 7 + 5].T
            shut = 0.4 if act.spec.venue or act.spec.kind == "class" else 1.0
            self.bday.append(np.where(self.holiday[:, None], (0.5 * b + 0.5 * sat) * shut, b))

    # -- chain
    def _chain(self, n_ext: int) -> None:
        p = self.p
        wr, ws, rw, rs, sw, sr = p.health
        th = []
        for m in range(n_ext):
            s = self.sick_season[m]
            rows = []
            for w in range(2 if p.has_w else 1):
                cm = p.crunch_mult if w else 1.0
                rows.append([[1 - wr * cm - ws * cm * s, wr * cm, ws * cm * s],
                             [rw, 1 - rw - rs * s, rs * s], [sw, sr, 1 - sw - sr]])
            th.append(np.array(rows) if p.has_w else np.array(rows[0]))
        self.TH = th
        rho = p.weather[2]
        self.TR = [np.array([[1 - pi * (1 - rho), pi * (1 - rho)], [(1 - pi) * (1 - rho), pi + rho * (1 - pi)]])
                   for pi in self.wet_pi]

        def two(on: float, off: float) -> np.ndarray:
            return np.array([[1 - on, on], [off, 1 - off]])

        def by_phase(on: float, off: float, on2: float, off2: float) -> np.ndarray:
            if not p.has_p:
                return two(on, off)
            return np.array([two(on2, off2) if ph in (F, A) else two(on, off) for ph in range(5)])

        static: dict[str, np.ndarray] = {}
        if p.has_w:
            static["W"] = by_phase(*p.crunch)
        if p.has_m:
            static["M"] = by_phase(*p.lowmood)
        for a in p.acts:
            if a.fickle >= 0:
                lk, al, ak, ka = a.eng_t
                static[f"E{a.fickle}"] = np.array([[1 - lk, 0, lk], [al, 1 - al - ak, ak], [0, ka, 1 - ka]])
        if p.has_q:
            static["Q"] = two(*p.friend_t)
        self.inj_parent = ""
        if p.has_i:
            on, off, keen = p.injury_t
            fk = p.acts[p.injury_act].fickle
            if fk >= 0:
                self.inj_parent = f"E{fk}"
                static["I"] = np.array([two(on * f, off) for f in (0.2, 1.0, keen)])
            else:
                static["I"] = two(on, off)
        ha, hu = p.change_h
        tp = np.zeros((5, 5))
        tp[B] = [1 - ha - hu, ha, 0, hu, 0]
        tp[BN, BP] = tp[BP, BP] = tp[F, A] = tp[A, A] = 1
        switch = tp.copy()
        switch[BP] = [0, 0, 0, 1, 0]
        self.TP, self.TP_switch, self.static = tp, switch, static
        parents = {"H": "W", "I": self.inj_parent, "W": "P", "M": "P"}
        order = ["H", "I", "W", "M"] + [f"E{i}" for i in range(p.n_fickle)] + ["P", "R", "Q"]
        self.order = []  # (factor, axis, parent axis or -1, shape that exposes them to matmul)
        for f in order:
            if f not in self.pos:
                continue
            i, par = self.pos[f], parents.get(f, "")
            j = self.pos.get(par, -1) if par else -1
            post = int(np.prod(self.sizes[i + 1:]))
            if j < 0:
                shape: tuple[int, ...] = (-1, self.sizes[i], post)
            else:
                shape = (-1, self.sizes[j], int(np.prod(self.sizes[j + 1:i])), self.sizes[i], post)
            self.order.append((f, i, j, shape))
        self.TT = {f: self._tt(T) for f, T in static.items()}
        self.TTH = [self._tt(T) for T in self.TH]
        self.TTR = [self._tt(T) for T in self.TR]
        self.TTP, self.TTP_switch = self._tt(tp), self._tt(switch)
        inits = {"H": np.array([0.86, 0.11, 0.03]), "R": np.array([1 - self.wet_pi[0], self.wet_pi[0]]),
                 "P": np.eye(5)[B], "I": np.array([0.96, 0.04])}
        if p.has_w:
            inits["W"] = np.array([p.crunch[1], p.crunch[0]]) / (p.crunch[0] + p.crunch[1])
        if p.has_m:
            inits["M"] = np.array([p.lowmood[1], p.lowmood[0]]) / (p.lowmood[0] + p.lowmood[1])
        if p.has_q:
            inits["Q"] = np.array([p.friend_t[1], p.friend_t[0]]) / sum(p.friend_t)
        for a in p.acts:
            if a.fickle >= 0:
                inits[f"E{a.fickle}"] = np.array(a.eng_init)
        self.inits = [inits[n] for n in self.names]
        self.init = reduce(np.multiply.outer, self.inits).ravel()

    @staticmethod
    def _tt(T: np.ndarray) -> np.ndarray:
        """T[(parent,) from, to] as the left matmul operand: [(parent, 1,) to, from]."""
        if T.ndim == 2:
            return np.ascontiguousarray(T.T)
        return np.ascontiguousarray(np.transpose(T, (0, 2, 1)))[:, None]

    def _T(self, f: str, m: int, change_day: int | None, tt: bool = False) -> np.ndarray:
        if f == "H":
            return (self.TTH if tt else self.TH)[m]
        if f == "R":
            return (self.TTR if tt else self.TR)[m]
        if f == "P":
            if change_day == m:
                return self.TTP_switch if tt else self.TP_switch
            return self.TTP if tt else self.TP
        return self.TT[f] if tt else self.static[f]

    def transition(self, X: np.ndarray, m: int, change_day: int | None) -> np.ndarray:
        """X: (batch, K), distributions on day m-1; returns them on day m."""
        n = X.shape[0]
        for f, _, _, shape in self.order:
            X = np.matmul(self._T(f, m, change_day, tt=True), X.reshape(shape))
        return X.reshape(n, self.K)

    def sample_init(self, rng: np.random.Generator) -> tuple[int, ...]:
        return tuple(_cat(rng, v) for v in self.inits)

    def sample_step(self, vals: tuple[int, ...], m: int, change_day: int | None,
                    rng: np.random.Generator) -> tuple[int, ...]:
        new = list(vals)
        for f, i, j, _ in self.order:
            T = self._T(f, m, change_day)
            new[i] = _cat(rng, T[vals[j], vals[i]] if j >= 0 else T[vals[i]])
        return tuple(new)

    def flat(self, vals: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(vals, self.sizes))

    # -- per-state vectors
    def v(self, name: str, table: Any, default: int = 0) -> np.ndarray:
        t = np.asarray(table, dtype=float)
        i = self.idx.get(name)
        return t[i] if i is not None else np.full(self.K, t[default])

    def _flag(self, name: str) -> np.ndarray:
        i = self.idx.get(name)
        return i == 1 if i is not None else np.zeros(self.K, bool)

    def _statics(self) -> None:
        p, v = self.p, self.v
        self.after = np.isin(self.idx["P"], (F, A)) if p.has_p else np.zeros(self.K, bool)
        self.wet, self.crunch, self.low = self._flag("R"), self._flag("W"), self._flag("M")
        self.busy, self.injured, self.hidx = self._flag("Q"), self._flag("I"), self.idx["H"]
        self.s_act, self.crunch_wd = [], []
        for a, act in enumerate(p.acts):
            k = act.spec.kind
            s = v("H", HEALTH_MULT[k]) * np.where(self.low, MOOD_MULT[k], 1.0)
            if act.fickle >= 0:
                s = s * v(f"E{act.fickle}", ENG_MULT)
            if act.spec.partner and p.has_q:
                s = s * np.where(self.busy, 0.15, 1.0)
            if a == p.injury_act:
                s = s * np.where(self.injured, 0.05, 1.0)
            elif k == "physical":
                s = s * np.where(self.injured, 0.6, 1.0)
            self.s_act.append(s)
            self.crunch_wd.append(np.where(self.crunch, CRUNCH_WORKDAY[act.slot], 1.0))
        self.crunch_we = np.where(self.crunch, 0.85, 1.0)
        mu0, _, mu_after, _ = p.mood
        self.mu_static = (mu0 + v("H", (0, -0.6, -1.4)) + np.where(self.low, -1.3, 0)
                          + np.where(self.injured, -0.4, 0) + np.where(self.after, mu_after, 0))
        self.s_plan = (np.where(self.crunch, 0.6, 1) * np.where(self.low, 0.45, 1) * v("H", (1, 0.75, 0.2))
                       * np.where(self.after, p.soc[1], p.soc[0]))
        wb = np.array([f.w_before for f in p.people])
        wa = np.array([f.w_after for f in p.people])
        w = np.where(self.after[None, :], wa[:, None], wb[:, None])
        if p.key >= 0:
            w[p.key] = w[p.key] * np.where(self.busy, 0.3, 1.0)
        self.who = w / w.sum(0)
        self.s_plan_out = v("H", (1, 0.85, 0.12)) * np.where(self.crunch, 0.8, 1) * np.where(self.low, 0.7, 1)

    def base(self, a: int, m: int) -> np.ndarray:
        return np.where(self.after, self.bday[a][m, 1], self.bday[a][m, 0])

    def act_p(self, a: int, m: int, plan_today: bool, away: bool) -> np.ndarray:
        act = self.p.acts[a]
        x = self.base(a, m) * self.s_act[a] * (self.crunch_wd[a] if self.workday[m] else self.crunch_we)
        if away:
            x = x * act.away
        elif act.spec.outdoor:
            x = x * np.where(self.wet, 0.35, 1.0) * (0.6 + 0.4 * self.warmth[m])
        elif act.spec.kind == "physical":
            x = x * np.where(self.wet, 1.05, 1.0)
        if plan_today:
            x = x * act.kappa
        return np.clip(x, *P_CLIP)

    def reason_given(self, a: int, m: int) -> np.ndarray:
        return self.p.tau * (self.base(a, m) >= 0.25)

    def skip_reasons(self, a: int, m: int, plan_today: bool, away: bool) -> np.ndarray:
        act, h = self.p.acts[a], self.hidx
        rows = [4.0 * (h == 2) + 0.3 * (h == 1), 1.2 * (h == 1) + 1.0 * self.low + 0.15,
                (2.0 * self.crunch + 0.1) * float(self.workday[m] and self.p.has_w),
                3.0 * self.wet * float(act.spec.outdoor and not away), 3.0 * float(plan_today), 6.0 * float(away),
                4.0 * self.injured * float(act.spec.kind == "physical"),
                3.0 * self.busy * float(act.spec.partner), 0.6 + 0.6 * self.low]
        w = np.array([np.broadcast_to(r, (self.K,)) for r in rows], dtype=float)
        return w / w.sum(0)

    def intent_p(self, a: int, m: int, away: bool) -> np.ndarray:
        act = self.p.acts[a]
        x = self.p.iota * self.base(a, m + 1) * self.v("H", (1, 0.6, 0.15)) * np.where(self.low, 0.5, 1.0)
        if act.fickle >= 0:
            x = x * self.v(f"E{act.fickle}", (0.15, 1.0, 1.6))
        return np.minimum(x * (0.3 if away else 1.0), 0.9)

    def mood_probs(self, m: int, away: bool) -> np.ndarray:
        mu = (self.mu_static + np.where(self.crunch, -0.5 if self.workday[m] else -0.25, 0.0)
              + (0.0 if away else np.where(self.wet, -0.15, 0.0))
              + 0.25 * self.weekend[m] + 0.4 * away + 0.3 * self.holiday[m])
        z = -((LEVELS[:, None] - mu[None, :]) ** 2) / (2 * self.p.mood[1] ** 2)
        e = np.exp(z - z.max(0))
        return e / e.sum(0)

    def plan_rate(self, lead: int, t: int) -> np.ndarray:
        w = LEAD_W[lead] * self.p.plan_dow[self.parity[t], 5 if self.holiday[t] else self.dow[t]]
        return np.minimum(0.6, w * self.s_plan)

    def plan_p(self, plan: _Plan, shaky: bool, away: bool) -> np.ndarray:
        f = self.p.people[plan.who]
        x = f.reliability * (1 - 0.01 * (plan.target - plan.made)) * self.s_plan_out
        if plan.who == self.p.key:
            x = x * np.where(self.busy, 0.4, 1.0)
        if shaky:
            x = x * 0.4
        if away:
            x = x * 0.06
        elif plan.kind in tx.OUTDOOR_KINDS:
            x = x * np.where(self.wet, 0.45, 1.0)
        return np.clip(x, 0.01, 0.99)

    def cancel_reasons(self, plan: _Plan, away: bool) -> np.ndarray:
        h = self.hidx
        rows = [4.0 * (h == 2) + 0.3 * (h == 1), 1.2 * (h == 1) + 1.0 * self.low + 0.15,
                (2.0 * self.crunch + 0.15) * float(self.p.has_w),
                3.0 * self.wet * float(plan.kind in tx.OUTDOOR_KINDS and not away), 6.0 * float(away),
                1.0 + 2.0 * self.busy * float(plan.who == self.p.key)]
        w = np.array([np.broadcast_to(r, (self.K,)) for r in rows], dtype=float)
        return w / w.sum(0)


# ---- the simulation ----------------------------------------------------------------------


@dataclass
class _Plan:
    made: int
    target: int
    who: int
    kind: str
    shaky: bool
    hedge: bool
    logodds: float  # what the messages so far say about shakiness
    happened: bool | None = None


@dataclass
class _Trip:
    decided: int
    start: int
    days: int
    booked: bool
    cancelled: bool | None  # known on the start day

    def covers(self, day: int) -> bool:
        return self.start <= day < self.start + self.days


@dataclass(frozen=True)
class _QSpec:
    kind: str  # act | count | plan
    ref: int
    target: int


class IndepLife:
    """One simulated person: everything precomputed, read-only."""

    def __init__(self, meta: LifeMeta, events: list[tuple[Event, ...]], queries: list[tuple[Query, ...]],
                 questions: list[tuple[Question, ...]], answers: dict[str, Event],
                 resolutions: dict[str, Resolution], truth: Truth) -> None:
        self.meta = meta
        self._events, self._queries, self._questions = events, queries, questions
        self._answers, self._resolutions, self._truth = answers, resolutions, truth

    def events(self, day: int) -> tuple[Event, ...]:
        return self._events[day] if 0 <= day < len(self._events) else ()

    def queries(self, day: int) -> tuple[Query, ...]:
        return self._queries[day] if 0 <= day < len(self._queries) else ()

    def questions(self, day: int) -> tuple[Question, ...]:
        return self._questions[day] if 0 <= day < len(self._questions) else ()

    def answer(self, question_id: str) -> Event:
        return self._answers[question_id]

    def resolve(self, qid: str) -> Resolution:
        return self._resolutions[qid]

    def truth(self) -> Truth:
        return self._truth


@dataclass
class _Sim:
    p: _Person
    md: _Model
    seed: int
    n: int
    plans: list[_Plan] = field(default_factory=list)
    trips: list[_Trip] = field(default_factory=list)
    traj: list[tuple[int, ...]] = field(default_factory=list)
    events: list[tuple[Event, ...]] = field(default_factory=list)
    queries: list[tuple[Query, ...]] = field(default_factory=list)
    questions: list[tuple[Question, ...]] = field(default_factory=list)
    answers: dict[str, Event] = field(default_factory=dict)
    pending: dict[str, tuple[_QSpec, float, float]] = field(default_factory=dict)
    change_day: int | None = None
    post: np.ndarray | None = None

    def run(self) -> IndepLife:
        self.did: np.ndarray = np.zeros((len(self.p.acts), self.n), bool)
        self.cdid: np.ndarray = np.zeros((len(self.p.counts), self.n), bool)
        for m in range(self.n):
            self._day(m)
        return IndepLife(self._meta(), self.events, self.queries, self.questions, self.answers,
                         self._resolutions(), self._truth())

    # -- one day
    def _day(self, m: int) -> None:
        md = self.md
        rng = np.random.default_rng([self.seed, 1, m])
        trng = np.random.default_rng([self.seed, 5, m])
        if self.post is None:
            vals, prior = md.sample_init(rng), md.init
        else:
            vals = md.sample_step(self.traj[-1], m, self.change_day, rng)
            prior = md.transition(self.post[None], m, self.change_day)[0]
        self.traj.append(vals)
        si = md.flat(vals)
        lik = np.ones(md.K)
        ev: list[Event] = []
        away = self._travel(m, rng, trng, ev)
        self._holiday_notice(m, ev)
        plan_today = self._make_plans(m, si, rng, trng, ev, lik)
        self._activities(m, si, plan_today, away, rng, trng, ev, lik)
        self._plan_outcomes(m, si, away, rng, trng, ev, lik)
        self._mood(m, si, away, rng, trng, ev, lik)
        self._talk(m, vals, away, rng, trng, ev, lik)
        self._intentions(m, si, away, rng, trng, ev, lik)
        self._plan_messages(m, rng, trng, ev)
        self._noise(m, away, rng, trng, ev)
        q = prior * lik
        total = q.sum()
        if not total > 0:
            raise RuntimeError(f"{self.p.pid} day {m}: the filter gave the truth zero probability")
        self.post = q / total
        self.events.append(tuple(ev[i] for i in trng.permutation(len(ev))))
        self._make_queries(m, si)
        self._make_questions(m, vals)

    def _ev(self, ev: list[Event], m: int, kind: str, subject: str, text: str, *tokens: str,
            value: float | None = None) -> None:
        ev.append(Event(m, kind, subject, text, (kind, *tokens), value))

    def _travel(self, m: int, rng: np.random.Generator, trng: np.random.Generator, ev: list[Event]) -> bool:
        booked, spont = self.p.trips
        if rng.random() < booked:
            lead, days = int(rng.integers(2, 22)), int(rng.integers(2, 9))
            self.trips.append(_Trip(m, m + lead, days, True, None))
            self._ev(ev, m, "plan", "travel", tx.trip_plan_text(trng, lead, days), "trip",
                     f"starts_in={lead}", f"days={days}", value=float(m + lead))
        started = False
        if rng.random() < spont:
            days = int(rng.integers(2, 5))
            self.trips.append(_Trip(m, m, days, False, False))
            self._ev(ev, m, "mention", "travel", tx.spontaneous_trip_text(trng, days), "trip", "spontaneous",
                     f"days={days}")
            started = True
        for tr in self.trips:
            if tr.booked and tr.start == m:
                tr.cancelled = bool(rng.random() < P_CANCEL_TRIP)
                started = True
                if tr.cancelled:
                    self._ev(ev, m, "cancel", "travel", tx.trip_cancel_text(trng), "trip")
                else:
                    self._ev(ev, m, "mention", "travel", tx.trip_start_text(trng), "trip", "started")
        away = any(tr.cancelled is False and tr.covers(m) for tr in self.trips)
        if away and not started and rng.random() < 0.3:
            self._ev(ev, m, "mention", "travel", tx.travel_noise_text(trng), "trip", "chat")
        return away

    def _holiday_notice(self, m: int, ev: list[Event]) -> None:
        for day, notice in self.p.holidays:
            if day - notice == m:
                dow = (self.p.start_weekday + day) % 7
                self._ev(ev, m, "mention", "holidays", tx.holiday_text(notice, dow), "holiday",
                         f"holiday_in={notice}", f"dow={dow}", value=float(day))

    def _make_plans(self, m: int, si: int, rng: np.random.Generator, trng: np.random.Generator,
                    ev: list[Event], lik: np.ndarray) -> bool:
        md = self.md
        for lead in range(MAX_LEAD + 1):
            t = m + lead
            rate = md.plan_rate(lead, t)
            if rng.random() >= rate[si]:
                lik *= 1 - rate
                continue
            j = _cat(rng, md.who[:, si])
            lik *= rate * md.who[j]
            f = self.p.people[j]
            taken = {pl.kind for pl in self.plans if pl.who == j and pl.target == t}
            kinds = [k for k in tx.PLAN_KINDS[f.role] if k not in taken]
            kind = tx.pick(rng, kinds) if kinds else f"{tx.pick(rng, tx.PLAN_KINDS[f.role])} #{len(taken) + 1}"
            shaky = bool(rng.random() < f.flaky)
            hedge = bool(rng.random() < HEDGE[shaky])
            lo = math.log(f.flaky / (1 - f.flaky))
            lo += math.log(HEDGE[1] / HEDGE[0]) if hedge else math.log((1 - HEDGE[1]) / (1 - HEDGE[0]))
            self.plans.append(_Plan(m, t, j, kind, shaky, hedge, lo))
            dow = int(md.dow[t])
            toks = (kind, f"with={f.name}", f"for=dow{dow}", f"in={lead}") + (("tentative",) if hedge else ())
            self._ev(ev, m, "plan", f.name, tx.plan_text(trng, kind, f.name, lead, dow, hedge), *toks,
                     value=float(t))
        return any(pl.target == m for pl in self.plans)

    def _activities(self, m: int, si: int, plan_today: bool, away: bool, rng: np.random.Generator,
                    trng: np.random.Generator, ev: list[Event], lik: np.ndarray) -> None:
        md, dow = self.md, f"dow={int(self.md.dow[m])}"
        key = self.p.people[self.p.key].name if self.p.key >= 0 else "my friend"
        for a, act in enumerate(self.p.acts):
            pv = md.act_p(a, m, plan_today, away)
            if rng.random() < pv[si]:
                self.did[a, m] = True
                lik *= pv
                self._ev(ev, m, "did", act.name, tx.did_text(trng, act.name), act.name, dow)
                continue
            lik *= 1 - pv
            given, reason = md.reason_given(a, m), None
            if rng.random() < given[si]:
                w = md.skip_reasons(a, m, plan_today, away)
                r = _cat(rng, w[:, si])
                lik *= given * w[r]
                reason = tx.SKIP_REASONS[r]
            else:
                lik *= 1 - given
            extra = (f"reason={reason}",) if reason else ()
            self._ev(ev, m, "skipped", act.name, tx.skipped_text(trng, act.name, reason, key), act.name, dow, *extra)
        for c, cnt in enumerate(self.p.counts):
            if rng.random() < cnt.probs[md.dow[m]]:
                self.cdid[c, m] = True
                self._ev(ev, m, "did", cnt.name, tx.did_text(trng, cnt.name), cnt.name, dow)
            else:
                self._ev(ev, m, "skipped", cnt.name, tx.skipped_text(trng, cnt.name, None, key), cnt.name, dow)

    def _plan_outcomes(self, m: int, si: int, away: bool, rng: np.random.Generator,
                       trng: np.random.Generator, ev: list[Event], lik: np.ndarray) -> None:
        for pl in self.plans:
            if pl.target != m:
                continue
            name = self.p.people[pl.who].name
            p0, p1 = self.md.plan_p(pl, False, away), self.md.plan_p(pl, True, away)
            w1 = _sigmoid(pl.logodds)
            pl.happened = bool(rng.random() < (p1 if pl.shaky else p0)[si])
            toks = (pl.kind, f"with={name}", f"dow={int(self.md.dow[m])}")
            if pl.happened:
                lik *= (1 - w1) * p0 + w1 * p1
                self._ev(ev, m, "did", name, tx.plan_done_text(trng, pl.kind, name), *toks)
                continue
            miss = (1 - w1) * (1 - p0) + w1 * (1 - p1)
            if rng.random() < 0.7:
                w = self.md.cancel_reasons(pl, away)
                r = _cat(rng, w[:, si])
                lik *= miss * w[r]
                reason = tx.CANCEL_REASONS[r]
                self._ev(ev, m, "cancel", name, tx.cancel_text(trng, pl.kind, name, reason), pl.kind,
                         f"with={name}", f"reason={reason}")
            else:
                lik *= miss
            self._ev(ev, m, "skipped", name, tx.plan_missed_text(trng, pl.kind, name), *toks)

    def _mood(self, m: int, si: int, away: bool, rng: np.random.Generator, trng: np.random.Generator,
              ev: list[Event], lik: np.ndarray) -> None:
        if rng.random() >= self.p.mood[3]:
            return
        probs = self.md.mood_probs(m, away)
        lvl = _cat(rng, probs[:, si])
        lik *= probs[lvl]
        self._ev(ev, m, "mood", "mood", tx.mood_text(trng, lvl + 1), f"level={lvl + 1}", value=float(lvl + 1))

    def _channel(self, name: str, table: np.ndarray, vals: tuple[int, ...], rng: np.random.Generator,
                 lik: np.ndarray) -> int:
        c = _cat(rng, table[vals[self.md.pos[name]]])
        lik *= table[self.md.idx[name], c]
        return c

    def _talk(self, m: int, vals: tuple[int, ...], away: bool, rng: np.random.Generator,
              trng: np.random.Generator, ev: list[Event], lik: np.ndarray) -> None:
        p = self.p
        if p.has_w:
            workday = bool(self.md.workday[m])
            c = self._channel("W", CH_WORK if workday else CH_WORK_OFF, vals, rng, lik)
            if c:
                self._ev(ev, m, "mention", p.work_topic, tx.talk_text(trng, p.work_topic, WORK_CONTENT[c]),
                         p.work_topic, WORK_CONTENT[c])
            c = self._channel("W", CH_BOSS if workday else CH_BOSS_OFF, vals, rng, lik)
            if c:
                boss = p.people[p.boss].name
                content = ("", "deadline", "fyi")[c]
                self._ev(ev, m, "message", boss, tx.boss_message_text(trng, boss, content), f"from={boss}", content)
        c = self._channel("H", CH_HEALTH, vals, rng, lik)
        if c:
            self._ev(ev, m, "mention", "health", tx.talk_text(trng, "health", HEALTH_CONTENT[c]), "health",
                     HEALTH_CONTENT[c])
        if p.has_i and self._channel("I", CH_INJURY, vals, rng, lik):
            self._ev(ev, m, "mention", "health", tx.talk_text(trng, "health", "sore"), "health", "sore")
        if not away:
            c = self._channel("R", CH_WEATHER, vals, rng, lik)
            if c:
                content = ("", "rain", "sunny")[c]
                self._ev(ev, m, "mention", "weather", tx.talk_text(trng, "weather", content), "weather", content)
        if p.has_q:
            c = self._channel("Q", CH_KEY, vals, rng, lik)
            if c:
                key = p.people[p.key].name
                content = ("", "swamped", "invite", "chat")[c]
                shared = next((a.name for a in p.acts if a.spec.partner), "hanging out")
                self._ev(ev, m, "message", key, tx.key_message_text(trng, key, content, shared), f"from={key}", content)
        for act in p.acts:
            if act.fickle >= 0:
                c = self._channel(f"E{act.fickle}", CH_KEEN, vals, rng, lik)
                if c:
                    content = ("", "loving", "guilt")[c]
                    self._ev(ev, m, "mention", act.name, tx.keen_text(trng, act.name, content), act.name, content)
        if p.has_p:
            c = self._channel("P", CH_CHANGE, vals, rng, lik)
            topic = tx.CHANGE_TOPIC[p.change]
            date = next((f.name for f in p.people if f.role == "date"), "")
            if c == 1:
                lead = int(rng.integers(8, 25))
                self.change_day = m + lead
                self._ev(ev, m, "plan", topic, tx.change_text(p.change, "announce", lead, date), "change", topic,
                         f"starts_in={lead}", value=float(m + lead))
            elif c:
                content = tx.CHANGE_CONTENTS[c - 1]
                self._ev(ev, m, "mention", topic, tx.change_text(p.change, content, 0, date), topic, f"change_{content}")

    def _intentions(self, m: int, si: int, away: bool, rng: np.random.Generator, trng: np.random.Generator,
                    ev: list[Event], lik: np.ndarray) -> None:
        for a, act in enumerate(self.p.acts):
            ip = self.md.intent_p(a, m, away)
            if rng.random() < ip[si]:
                lik *= ip
                self._ev(ev, m, "plan", act.name, tx.intention_text(trng, act.name), act.name,
                         f"for=dow{int(self.md.dow[m + 1])}", "in=1", value=float(m + 1))
            else:
                lik *= 1 - ip

    def _plan_messages(self, m: int, rng: np.random.Generator, trng: np.random.Generator, ev: list[Event]) -> None:
        for pl in self.plans:
            if not pl.made < m < pl.target:
                continue
            c = _cat(rng, PLAN_MSG[int(pl.shaky)])
            pl.logodds += math.log(PLAN_MSG[1, c] / PLAN_MSG[0, c])
            if c:
                name, content = self.p.people[pl.who].name, ("", "doubt", "confirm")[c]
                lead, dow = pl.target - m, int(self.md.dow[pl.target])
                self._ev(ev, m, "message", name, tx.plan_message_text(trng, name, content, pl.kind, lead, dow),
                         f"from={name}", content, pl.kind, f"for=dow{dow}", f"in={lead}")

    def _noise(self, m: int, away: bool, rng: np.random.Generator, trng: np.random.Generator,
               ev: list[Event]) -> None:
        for topic, rate in self.p.noise:
            if rng.random() < rate:
                content = tx.NOISE_CONTENTS[int(rng.integers(3))]
                self._ev(ev, m, "mention", topic, tx.noise_text(topic, content), topic, content)
        for f in self.p.people:
            if rng.random() < f.chat:
                self._ev(ev, m, "message", f.name, tx.chat_text(trng, f.name), f"from={f.name}", "chat")

    # -- queries and the references
    def _make_queries(self, m: int, si: int) -> None:
        last = min(m + HORIZON, self.n - 1)
        if last <= m:
            self.queries.append(())
            return
        rng = np.random.default_rng([self.seed, 2, m])
        targets = np.arange(m + 1, last + 1)
        n_q = int(rng.integers(6, 11))
        specs: list[_QSpec] = []
        pend = [j for j, pl in enumerate(self.plans) if pl.made <= m < pl.target <= last]
        for j in rng.permutation(len(pend)):
            if len(specs) < 3 and rng.random() < 0.85:
                specs.append(_QSpec("plan", pend[j], self.plans[pend[j]].target))
        n_count = max(2, int(round(n_q * rng.uniform(0.25, 0.42))))
        for _ in range(60):
            if sum(s.kind == "count" for s in specs) >= n_count:
                break
            c = int(rng.integers(len(self.p.counts)))
            pr = self.p.counts[c].probs[self.md.dow[targets]]
            w = 0.3 + np.minimum(pr, 1 - pr)
            spec = _QSpec("count", c, int(targets[_cat(rng, w / w.sum())]))
            if spec not in specs:
                specs.append(spec)
        for _ in range(100):
            if len(specs) >= n_q:
                break
            a = int(rng.integers(len(self.p.acts)))
            w = 0.25 + self.md.bday[a][targets, 0]
            spec = _QSpec("act", a, int(targets[_cat(rng, w / w.sum())]))
            if spec not in specs:
                specs.append(spec)
        probs = self._predict(m, si, specs)
        out = []
        for i, (spec, (po, ps)) in enumerate(zip(specs, probs)):
            qid = f"{self.p.pid}:{m}:{i}"
            lead, dow = spec.target - m, int(self.md.dow[spec.target])
            if spec.kind == "plan":
                pl = self.plans[spec.ref]
                name = self.p.people[pl.who].name
                out.append(Query(qid, m, spec.target, name, tx.query_plan_text(pl.kind, name, lead, dow),
                                 ("plan_happens", pl.kind, f"with={name}", f"dow={dow}", f"in={lead}"), "hidden"))
            else:
                name = (self.p.acts[spec.ref] if spec.kind == "act" else self.p.counts[spec.ref]).name
                out.append(Query(qid, m, spec.target, name, tx.query_act_text(name, lead, dow),
                                 ("will_do", name, f"dow={dow}", f"in={lead}"),
                                 "hidden" if spec.kind == "act" else "count"))
            self.pending[qid] = (spec, po, ps)
        self.queries.append(tuple(out))

    def _p_away(self, t: int, d: int) -> float:
        home = 1.0
        for tr in self.trips:
            if tr.covers(t):
                home *= (1.0 if tr.cancelled else 0.0) if tr.start <= d else P_CANCEL_TRIP
        booked, spont = self.p.trips
        for day in range(d + 1, t + 1):
            j = t - day
            home *= (1 - booked * (1 - P_CANCEL_TRIP) * TRIP_BOOKED[j]) * (1 - spont * TRIP_SPONT[j])
        return 1.0 - home

    def _forward(self, m: int, si: int, n_h: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """For each day t in (m, m + n_h]: the state distribution on t (``total``) and its part with
        no plan made for t yet (``free``), from the filter's posterior (row 0) and the truth (row 1).
        Plans not yet made are drawn day by day from the states they would be made in."""
        md, K = self.md, self.md.K
        planned = {pl.target for pl in self.plans}
        start = np.zeros((2, K))
        start[0], start[1, si] = self.post, 1.0
        total = start
        free = np.zeros((2, n_h, K))
        for h in range(n_h):
            if m + 1 + h not in planned:
                free[:, h] = start
        snaps = {}
        for h in range(n_h):
            day, rest = m + 1 + h, n_h - h
            Z = np.concatenate([total[:, None], free[:, h:]], axis=1).reshape(-1, K)
            Z = md.transition(Z, day, self.change_day).reshape(2, 1 + rest, K)
            for h2 in range(rest):
                Z[:, 1 + h2] *= 1 - md.plan_rate(h2, day + h2)
            total, free[:, h:] = Z[:, 0], Z[:, 1:]
            snaps[day] = (total, Z[:, 1])
        return snaps

    def _predict(self, m: int, si: int, specs: list[_QSpec]) -> list[tuple[float, float]]:
        md = self.md
        need = [s.target for s in specs if s.kind != "count"]
        snaps = self._forward(m, si, max(need) - m) if need else {}
        out = []
        for s in specs:
            if s.kind == "count":
                pr = float(self.p.counts[s.ref].probs[md.dow[s.target]])
                out.append((pr, pr))
                continue
            (total, free), pa = snaps[s.target], self._p_away(s.target, m)
            norm = total.sum(1)
            if s.kind == "act":
                v0, v1 = ((1 - pa) * md.act_p(s.ref, s.target, c, False) + pa * md.act_p(s.ref, s.target, c, True)
                          for c in (False, True))
                pr2 = (total @ v1 + free @ (v0 - v1)) / norm
            else:
                pl = self.plans[s.ref]
                pp = [(1 - pa) * md.plan_p(pl, sh, False) + pa * md.plan_p(pl, sh, True) for sh in (False, True)]
                w1 = _sigmoid(pl.logodds)
                pr2 = np.array([total[0] @ ((1 - w1) * pp[0] + w1 * pp[1]), total[1] @ pp[int(pl.shaky)]]) / norm
            out.append((float(pr2[0]), float(pr2[1])))
        return out

    # -- questions
    def _answer_row(self, kind: str, ref: int, vals: tuple[int, ...]) -> np.ndarray:
        p, pos = self.p, self.md.pos
        if kind == "health":
            return ANS_HEALTH[vals[pos["H"]]]
        if kind == "mood":
            return ANS_MOOD[vals[pos["M"]] if p.has_m else 0]
        if kind == "work":
            return ANS_WORK[vals[pos["W"]]] if p.has_w else np.array([0.9, 0.1, 0.0])
        if kind == "injury":
            return ANS_INJURY[vals[pos["I"]]] if p.has_i else np.array([0.85, 0.15, 0.0])
        if kind == "change":
            return ANS_CHANGE[(0, 1, 1, 2, 2)[vals[pos["P"]]]] if p.has_p else ANS_CHANGE[0]
        if kind == "habit":
            fk = p.acts[ref].fickle
            return ANS_HABIT[vals[pos[f"E{fk}"]] if fk >= 0 else ACTIVE]
        if kind == "friend":
            return ANS_FRIEND[vals[pos["Q"]]] if ref == p.key else np.array([0.7, 0.3])
        return ANS_PLAN[int(self.plans[ref].shaky)]

    def _make_questions(self, m: int, vals: tuple[int, ...]) -> None:
        p = self.p
        rng = np.random.default_rng([self.seed, 3, m])
        pool: list[tuple[str, str, int]] = [("health", "health", -1), ("mood", "mood", -1),
                                            ("injury", "health", -1), ("change", "life", -1)]
        pool += [("work", p.work_topic, -1)] if p.has_w else []
        pool += [("habit", a.name, i) for i, a in enumerate(p.acts)]
        pool += [("friend", f.name, j) for j, f in enumerate(p.people) if f.role in ("friend", "partner", "date")]
        pool += [("plan", p.people[pl.who].name, j) for j, pl in enumerate(self.plans) if pl.made <= m < pl.target]
        w = np.array([2.0 if k == "plan" else 1.0 for k, _, _ in pool])
        n = min(int(rng.integers(3, 6)), len(pool))
        out = []
        for i, ci in enumerate(sorted(rng.choice(len(pool), size=n, replace=False, p=w / w.sum()))):
            kind, subject, ref = pool[ci]
            extra: tuple[str, ...] = ()
            desc = ""
            if kind == "plan":
                pl = self.plans[ref]
                dow = int(self.md.dow[pl.target])
                extra = (pl.kind, f"dow={dow}")
                desc = f"{pl.kind} with {subject} {tx.day_phrase(pl.target - m, dow)}"
            opts = tx.QUESTION_OPTIONS[kind]
            arng = np.random.default_rng([self.seed, 4, m, i])
            ans = opts[_cat(arng, self._answer_row(kind, ref, vals))]
            qid = f"{p.pid}:{m}:ask{i}:{kind}"
            options = tuple(("answer", kind, subject, *extra, o) for o in opts)
            out.append(Question(qid, m, subject, tx.question_text(kind, subject, desc), ("ask", kind, subject, *extra),
                                tx.QUESTION_COST[kind], options))
            self.answers[qid] = Event(m, "answer", subject, tx.answer_text(kind, ans), ("answer", kind, subject, *extra, ans))
        self.questions.append(tuple(out))

    # -- the end
    def _resolutions(self) -> dict[str, Resolution]:
        out = {}
        for qid, (s, po, ps) in self.pending.items():
            if s.kind == "act":
                y = bool(self.did[s.ref, s.target])
            elif s.kind == "count":
                y = bool(self.cdid[s.ref, s.target])
            else:
                y = bool(self.plans[s.ref].happened)
            out[qid] = Resolution(qid, y, po, ps)
        return out

    def _meta(self) -> LifeMeta:
        p = self.p
        names = [a.name for a in p.acts] + [c.name for c in p.counts]
        order = np.random.default_rng([self.seed, 6]).permutation(len(names))
        return LifeMeta(p.pid, self.n, p.start_weekday, tuple(names[i] for i in order),
                        tuple(f.name for f in p.people), p.topics)

    def _truth(self) -> Truth:
        p, md = self.p, self.md
        traj = np.array(self.traj)
        shifts = set()
        for name in md.names:
            col = traj[:, md.pos[name]]
            for m in range(1, self.n):
                if name == "P" and col[m] == F and col[m - 1] != F:
                    shifts.add(m)
                if name.startswith("E") and (col[m] == LAPSED) != (col[m - 1] == LAPSED):
                    shifts.add(m)
        facts = {
            "archetype": p.archetype, "change": p.change,
            "states": {name: [int(x) for x in traj[:, md.pos[name]]] for name in md.names},
            "plans": [{"made": pl.made, "target": pl.target, "with": p.people[pl.who].name, "kind": pl.kind,
                       "shaky": pl.shaky, "happened": pl.happened} for pl in self.plans],
            "trips": [{"decided": t.decided, "start": t.start, "days": t.days, "booked": t.booked,
                       "cancelled": t.cancelled} for t in self.trips],
            "count_activities": [c.name for c in p.counts],
        }
        return Truth(tuple(sorted(shifts)), facts, _params(p))


def _params(p: _Person) -> dict[str, object]:
    acts = [{"name": a.name, "kind": a.spec.kind, "slot": a.slot, "outdoor": a.spec.outdoor,
             "partner": a.spec.partner, "fickle": a.fickle >= 0, "eng_t": list(a.eng_t),
             "base_before": [round(float(x), 4) for x in a.base[0]],
             "base_after": [round(float(x), 4) for x in a.base[1]]} for a in p.acts]
    return {
        "archetype": p.archetype, "regularity": p.regularity, "rota": p.rota, "start_doy": p.start_doy,
        "activities": acts, "counts": {c.name: [round(float(x), 4) for x in c.probs] for c in p.counts},
        "people": [{"name": f.name, "role": f.role, "reliability": f.reliability, "flaky": f.flaky}
                   for f in p.people],
        "key_friend": p.people[p.key].name if p.key >= 0 else None, "work": p.work_topic,
        "factors": {"low_mood": p.has_m, "friend": p.has_q, "injury": p.has_i, "change": p.change},
        "health": list(p.health), "crunch": list(p.crunch), "crunch_mult": p.crunch_mult,
        "lowmood": list(p.lowmood), "weather": list(p.weather), "friend_t": list(p.friend_t),
        "injury_t": list(p.injury_t), "change_hazards": list(p.change_h), "sociability": list(p.soc),
        "trips": list(p.trips), "mood": list(p.mood), "tau": p.tau, "iota": p.iota,
        "holidays": [list(h) for h in p.holidays],
    }


def make_life(seed: int, n_days: int = 84) -> IndepLife:
    """One person's ``n_days`` from ``seed``: pure, deterministic, no I/O."""
    if n_days < 1:
        raise ValueError("n_days must be positive")
    seed_u = int(seed) % (2**63)
    person = _draw_person(seed_u)
    model = _Model(person, n_days + HORIZON + 2)
    return _Sim(person, model, seed_u, n_days).run()
