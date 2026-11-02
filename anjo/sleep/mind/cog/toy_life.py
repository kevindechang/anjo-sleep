"""A small fake ``Life`` for the cog core's tests and its training until ``world.py`` lands.

One person, two activities and a hidden two-state work regime (calm/busy, a sticky Markov chain):

- **run** has a weekday habit (three days drawn per person) and is regime-sensitive: on a habit day
  it happens with 0.85 when calm and 0.2 when busy (``reverse`` flips that: busy runs more, the
  idiosyncratic person). After ``shift_day`` (if any) the habit moves to three other weekdays for good.
- **gym** has a weekday habit (two days), ignores the regime, and a plan made two days ahead
  (``plan`` event, ``value`` = the planned day) makes it near certain.
- **Signals** of the regime: late work messages and tired moods, each likelier when busy.
- **Distractor**: weather mentions with three random words; nothing depends on them.
- **Questions** each day: work ("work calm" / "work busy", the true regime) and weather (a random
  word, the distractor). Queries: three a day about run or gym, one to three days ahead.

Outcomes show as events: ``did`` when it happened, ``skipped`` when it was a habit or planned day
and didn't; nothing otherwise (absent means it didn't happen). ``oracle_p`` is the forward filter
over the regime with the true parameters on the passive events up to the asked day; ``state_p``
also knows the regime on the target day. Pure: everything is drawn at construction from the seed.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from anjo.sleep.mind.cog.types import Event, Life, LifeMeta, Query, Question, Resolution, Truth

WEEKDAYS: Final = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
STAY: Final = 0.92  # the regime's daily probability of staying
RUN_HABIT: Final = (0.85, 0.2)  # run on a habit day, calm vs busy
RUN_OFF: Final = 0.08
GYM_HABIT: Final = 0.8
GYM_OFF: Final = 0.1
GYM_PLANNED: Final = 0.95
PLAN_RATE: Final = 0.25  # a gym plan for two days ahead, per day
PLAN_LEAD: Final = 2
LATE: Final = (0.25, 0.6)  # a late work message, calm vs busy
TIRED: Final = (0.3, 0.5)
WEATHER_WORDS: Final = ("sunny", "rainy", "windy")
QUERIES_PER_DAY: Final = 3
MAX_HORIZON: Final = 3
QUESTION_COST: Final = 0.05


@dataclass(frozen=True)
class ToyParams:
    run_days: tuple[int, ...]
    gym_days: tuple[int, ...]
    run_days_after: tuple[int, ...]
    reverse: bool
    shift_day: int | None


def _p_run(params: ToyParams, day: int, weekday: int, regime: int) -> float:
    days = params.run_days if params.shift_day is None or day < params.shift_day else params.run_days_after
    if weekday not in days:
        return RUN_OFF
    return RUN_HABIT[1 - regime] if params.reverse else RUN_HABIT[regime]


def _p_gym(params: ToyParams, weekday: int, planned: bool) -> float:
    if planned:
        return GYM_PLANNED
    return GYM_HABIT if weekday in params.gym_days else GYM_OFF


class ToyLife:
    """A ``Life`` (types.Life), drawn whole from ``seed``."""

    def __init__(
        self, seed: int, *, n_days: int = 56, reverse: bool | None = None, shift_day: int | None | bool = None,
        p_reverse: float = 0.25, p_shift: float = 0.3,
    ) -> None:
        rng = np.random.default_rng(seed)
        start_weekday = int(rng.integers(7))
        run_days = tuple(sorted(int(x) for x in rng.choice(7, 3, replace=False)))
        others = [d for d in range(7) if d not in run_days]
        run_after = tuple(sorted(int(x) for x in rng.choice(others, 3, replace=False)))
        gym_days = tuple(sorted(int(x) for x in rng.choice(7, 2, replace=False)))
        rev = bool(rng.random() < p_reverse) if reverse is None else reverse
        if shift_day is None:
            shift: int | None = int(rng.integers(n_days // 4, 3 * n_days // 4)) if rng.random() < p_shift else None
        elif shift_day is False:
            shift = None
        else:
            shift = int(shift_day)
        self.params = ToyParams(run_days, gym_days, run_after, rev, shift)
        self.meta = LifeMeta(f"toy-{seed}", n_days, start_weekday, ("run", "gym"), (), ("work", "weather"))
        self._regime = self._draw_regime(rng, n_days)
        self._planned = rng.random(n_days) < PLAN_RATE  # a plan made on day s for day s + PLAN_LEAD
        self._draw_days(rng)
        self._draw_queries(rng)

    # ── generation ───────────────────────────────────────────────────────────

    def weekday(self, day: int) -> int:
        return (self.meta.start_weekday + day) % 7

    @staticmethod
    def _draw_regime(rng: np.random.Generator, n_days: int) -> np.ndarray:
        regime = np.zeros(n_days, dtype=np.int64)
        regime[0] = int(rng.random() < 0.5)
        for d in range(1, n_days):
            regime[d] = regime[d - 1] if rng.random() < STAY else 1 - regime[d - 1]
        return regime

    def planned_for(self, day: int) -> bool:
        return day >= PLAN_LEAD and bool(self._planned[day - PLAN_LEAD])

    def _draw_days(self, rng: np.random.Generator) -> None:
        n = self.meta.n_days
        self._events: list[tuple[Event, ...]] = []
        self._did: dict[tuple[str, int], bool] = {}
        self._late = np.zeros(n, dtype=bool)
        self._tired = np.zeros(n, dtype=bool)
        for d in range(n):
            r, wd = int(self._regime[d]), self.weekday(d)
            day: list[Event] = []
            self._late[d] = rng.random() < LATE[r]
            self._tired[d] = rng.random() < TIRED[r]
            if self._late[d]:
                day.append(Event(d, "message", "work", "a late message about work", ("message", "late", "work")))
            mood = "tired" if self._tired[d] else "fine"
            day.append(Event(d, "mood", "self", f"feeling {mood}", ("mood", mood)))
            if rng.random() < 0.7:
                word = WEATHER_WORDS[int(rng.integers(3))]
                day.append(Event(d, "mention", "weather", f"it is {word}", ("mention", "weather", word)))
            if self._planned[d] and d + PLAN_LEAD < n:
                target = d + PLAN_LEAD
                name = WEEKDAYS[self.weekday(target)]
                day.append(Event(d, "plan", "gym", f"gym on {name}", ("plan", "gym", name), float(target)))
            for act, p, expected in (
                ("run", _p_run(self.params, d, wd, r), self._run_expected(d, wd)),
                ("gym", _p_gym(self.params, wd, self.planned_for(d)), wd in self.params.gym_days or self.planned_for(d)),
            ):
                did = bool(rng.random() < p)
                self._did[(act, d)] = did
                if did:
                    day.append(Event(d, "did", act, f"went to {act}", ("did", act)))
                elif expected:
                    day.append(Event(d, "skipped", act, f"skipped {act}", ("skipped", act)))
            self._events.append(tuple(day))
        self._weather = tuple(WEATHER_WORDS[int(rng.integers(3))] for _ in range(n))

    def _run_expected(self, day: int, weekday: int) -> bool:
        p = self.params
        days = p.run_days if p.shift_day is None or day < p.shift_day else p.run_days_after
        return weekday in days

    def _draw_queries(self, rng: np.random.Generator) -> None:
        n = self.meta.n_days
        self._queries: list[tuple[Query, ...]] = []
        self._by_qid: dict[str, Query] = {}
        for d in range(n):
            day: list[Query] = []
            for i in range(QUERIES_PER_DAY):
                h = int(rng.integers(1, MAX_HORIZON + 1))
                if d + h >= n:
                    continue
                act = ("run", "gym")[int(rng.integers(2))]
                name = WEEKDAYS[self.weekday(d + h)]
                q = Query(f"{self.meta.person_id}-q{d}-{i}", d, d + h, act, f"Will they {act} on {name}?",
                          ("will", act, name), "hidden" if act == "run" else "count")
                day.append(q)
                self._by_qid[q.qid] = q
            self._queries.append(tuple(day))

    # ── the Life protocol ────────────────────────────────────────────────────

    def events(self, day: int) -> tuple[Event, ...]:
        return self._events[day]

    def queries(self, day: int) -> tuple[Query, ...]:
        return self._queries[day]

    def questions(self, day: int) -> tuple[Question, ...]:
        pid = self.meta.person_id
        return (
            Question(f"{pid}-work-{day}", day, "work", "How is work going?", ("how", "work"), QUESTION_COST,
                     (("work", "calm"), ("work", "busy"))),
            Question(f"{pid}-weather-{day}", day, "weather", "How is the weather?", ("how", "weather"),
                     QUESTION_COST, tuple(("weather", w) for w in WEATHER_WORDS)),
        )

    def answer(self, question_id: str) -> Event:
        kind, day = question_id.rsplit("-", 2)[-2:]
        d = int(day)
        if kind == "work":
            word = ("calm", "busy")[int(self._regime[d])]
            return Event(d, "answer", "work", f"work is {word}", ("work", word))
        word = self._weather[d]
        return Event(d, "answer", "weather", f"it is {word}", ("weather", word))

    def resolve(self, qid: str) -> Resolution:
        q = self._by_qid[qid]
        outcome = self._did[(q.subject, q.target_day)]
        return Resolution(qid, outcome, self._oracle(q), self._state_p(q))

    def truth(self) -> Truth:
        p = self.params
        facts: Mapping[str, object] = {}
        return Truth(() if p.shift_day is None else (p.shift_day,), facts,
                     {"reverse": p.reverse, "run_days": p.run_days, "gym_days": p.gym_days})

    # ── references ───────────────────────────────────────────────────────────

    def regime(self, day: int) -> int:
        return int(self._regime[day])

    def _p_outcome(self, q: Query, regime: int, plan_known: bool) -> float:
        wd = self.weekday(q.target_day)
        if q.subject == "run":
            return _p_run(self.params, q.target_day, wd, regime)
        plan_made_by = q.target_day - PLAN_LEAD
        if plan_made_by <= q.asked_day or plan_known:
            return _p_gym(self.params, wd, self.planned_for(q.target_day))
        return PLAN_RATE * GYM_PLANNED + (1 - PLAN_RATE) * _p_gym(self.params, wd, False)

    def _state_p(self, q: Query) -> float:
        return self._p_outcome(q, self.regime(q.target_day), plan_known=True)

    def _oracle(self, q: Query) -> float:
        belief = self.filtered(q.asked_day)
        trans = np.array([[STAY, 1 - STAY], [1 - STAY, STAY]])
        for _ in range(q.target_day - q.asked_day):
            belief = belief @ trans
        return float(sum(belief[r] * self._p_outcome(q, r, plan_known=False) for r in (0, 1)))

    def filtered(self, through: int) -> np.ndarray:
        """P(regime on day ``through`` | the passive events of days 0..through), the true parameters."""
        if not hasattr(self, "_filtered"):
            self._filtered = self._forward()
        return self._filtered[through].copy()

    def _forward(self) -> np.ndarray:
        trans = np.array([[STAY, 1 - STAY], [1 - STAY, STAY]])
        out = np.zeros((self.meta.n_days, 2))
        belief = np.array([0.5, 0.5])
        for d in range(self.meta.n_days):
            if d > 0:
                belief = belief @ trans
            like = np.ones(2)
            for r in (0, 1):
                like[r] *= LATE[r] if self._late[d] else 1 - LATE[r]
                like[r] *= TIRED[r] if self._tired[d] else 1 - TIRED[r]
                p = _p_run(self.params, d, self.weekday(d), r)
                like[r] *= p if self._did[("run", d)] else 1 - p
            belief = belief * like / (belief * like).sum()
            out[d] = belief
        return out


def make_toy_life(seed: int, population: str = "base", n_days: int = 56, **kw: object) -> Life:
    """``world.make_life``'s signature, on the toy world. ``population`` is accepted and ignored."""
    del population
    return ToyLife(seed, n_days=n_days, **kw)  # type: ignore[arg-type]
