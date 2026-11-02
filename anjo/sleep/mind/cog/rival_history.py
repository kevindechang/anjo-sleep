"""What the rivals keep of one person: their events, read from fields and tokens, never text.

The world's semantics (world.py's docstring) are what it is built on:

- an activity has an *occurrence* on a day when the person says ``did``, ``skipped`` (a habit's
  slot day) or ``cancel`` (a plan's day); no event means no occurrence, never a miss. An outcome
  is 1 for ``did`` and 0 for ``skipped`` or ``cancel``, so cancels count in every denominator;
- a ``plan`` names its day in ``value`` (else the first later day a token names, else tomorrow);
  a plan is kept when its day has a ``did``;
- ``mood`` carries its level in ``value`` (else a mood word, ``MOOD_WORDS``); a ``message`` is
  late-night when a token says ``late``/``night``/``midnight`` or an hour 22:00-04:59;
- ``mention`` subjects are kept by day (a fact's hints and distractor topics alike: which matter
  is for a fit to learn); answers are kept with their day, so a count in them (``in=3``,
  ``left=2``) can be moved to any later day;
- a subject's family (count or hidden) is learned from the queries about it (``note_queries``).

Activity names that embed one of the person's people (``dinner_sam``) share a stem across people
(``dinner_<friend>``), so a population fit pools them.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar, cast

from anjo.sleep.mind.cog.types import Event, LifeMeta, Query

LATE_WORDS = ("late", "night", "midnight")
PLAN_WORDS = ("plan", "kept")
LATE_HOURS = frozenset({22, 23, 0, 1, 2, 3, 4})
MOOD_WORDS: Mapping[str, float] = {
    "awful": 1.0, "terrible": 1.0, "bad": 2.0, "low": 2.0, "sad": 2.0, "stressed": 2.0,
    "tired": 2.5, "meh": 3.0, "ok": 3.0, "okay": 3.0, "fine": 3.0, "good": 4.0, "happy": 4.5,
    "great": 5.0,
}
OUTCOME_KINDS = ("did", "skipped", "cancel")
FRIEND = "<friend>"
T = TypeVar("T")


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


def stem(activity: str, people: Sequence[str]) -> str:
    """``dinner_sam`` -> ``dinner_<friend>`` when Sam is one of the person's people."""
    for person in people:
        suffix = "_" + person.lower()
        if activity.lower().endswith(suffix) and len(activity) > len(suffix):
            return activity[: -len(suffix)] + "_" + FRIEND
    return activity


class PersonHistory:
    """One person's events so far; every read is "as of" a day and sees nothing after it."""

    def __init__(self, meta: LifeMeta) -> None:
        self.meta = meta
        self.activities: list[str] = list(meta.activities)
        self.occ: dict[str, dict[int, int]] = defaultdict(dict)  # activity -> day -> 1 did / 0 missed
        self.plans: dict[tuple[str, int], int] = {}  # (activity, day it is for) -> day it was made
        self.moods: dict[int, list[float]] = defaultdict(list)
        self.late: dict[int, int] = defaultdict(int)
        self.mentions: dict[int, list[str]] = defaultdict(list)
        self.answers: list[tuple[int, str, tuple[str, ...]]] = []  # (day, subject, tokens)
        self.families: dict[str, str] = {}
        self._cache: dict[tuple[object, ...], object] = {}

    def weekday(self, day: int) -> int:
        return (self.meta.start_weekday + day) % 7

    def stem(self, activity: str) -> str:
        return stem(activity, self.meta.people)

    def add(self, events: Iterable[Event]) -> None:
        for e in events:
            self._add(e)
        self._cache.clear()

    def cached(self, key: tuple[object, ...], make: Callable[[], T]) -> T:
        """``make()`` memoized until the next event: for reads several queries share."""
        if key not in self._cache:
            self._cache[key] = make()
        return cast(T, self._cache[key])

    def note_queries(self, queries: Iterable[Query]) -> None:
        """Learn each subject's family from the queries about it (the same every time it is asked)."""
        for q in queries:
            if self.families.get(q.subject) != q.family:
                self.families[q.subject] = q.family
                self._cache.clear()

    def _activity(self, name: str) -> None:
        if name not in self.activities:
            self.activities.append(name)

    def _add(self, e: Event) -> None:
        d = e.day
        if e.kind in OUTCOME_KINDS:
            self._activity(e.subject)
            # a cancel names its day (said ahead, it is read only from that day on)
            day = d if e.kind != "cancel" or e.value is None else int(e.value)
            self.occ[e.subject][day] = int(e.kind == "did")
        elif e.kind == "plan":
            self._activity(e.subject)
            later = [n for n in token_ints(e.tokens) if n > d]
            target = int(e.value) if e.value is not None else (later[0] if later else d + 1)
            self.plans[(e.subject, target)] = d
        elif e.kind == "mood":
            level = e.value if e.value is not None else next(
                (MOOD_WORDS[t.lower()] for t in e.tokens if t.lower() in MOOD_WORDS), None)
            if level is not None:
                self.moods[d].append(float(level))
        elif e.kind == "message":
            self.late[d] += int(is_late(e.tokens))
        elif e.kind == "mention":
            self.mentions[d].append(e.subject)
        elif e.kind == "answer":
            self.answers.append((d, e.subject, tuple(e.tokens)))

    # -- an activity's record --
    def outcome(self, activity: str, day: int) -> int | None:
        return self.occ.get(activity, {}).get(day)

    def record(self, activity: str, as_of: int, weekday: int | None = None) -> list[tuple[int, int]]:
        """(day, outcome) of the activity's occurrences up to ``as_of``, oldest first."""
        days = self.occ.get(activity, {})
        return sorted((d, y) for d, y in days.items()
                      if d <= as_of and (weekday is None or self.weekday(d) == weekday))

    def rate(self, activity: str, as_of: int, weekday: int | None = None) -> tuple[int, int]:
        rec = self.record(activity, as_of, weekday)
        return sum(y for _, y in rec), len(rec)

    def decayed_rate(self, activity: str, as_of: int, gamma: float) -> float:
        """The rate with each earlier occurrence weighted ``gamma`` times the next, a prior of one
        success in two: it forgets an old habit, which the plain counts never do."""
        k, n, weight = 1.0, 2.0, 1.0
        for _, y in reversed(self.record(activity, as_of)):
            k, n, weight = k + weight * y, n + weight, weight * gamma
        return k / n

    def streak(self, activity: str, as_of: int, cap: int = 2) -> int:
        """Misses in a row at the end of the activity's record."""
        out = 0
        for _, y in reversed(self.record(activity, as_of)):
            if y or out >= cap:
                break
            out += 1
        return out

    def counts(self, activity: str, as_of: int, days: int) -> tuple[int, int]:
        """(done, missed) in the last ``days`` days."""
        rec = [y for d, y in self.record(activity, as_of) if d > as_of - days]
        return sum(rec), len(rec) - sum(rec)

    def miss_rate(self, as_of: int, days: int, family: str | None = None) -> tuple[int, int]:
        """(missed, occurrences) over every activity (of ``family``, when given) in the window."""
        missed = total = 0
        for a, days_of in self.occ.items():
            if family is not None and self.families.get(a) != family:
                continue
            for d, y in days_of.items():
                if as_of - days < d <= as_of:
                    missed, total = missed + (1 - y), total + 1
        return missed, total

    def done_on(self, day: int) -> list[str]:
        return [a for a, days in self.occ.items() if days.get(day) == 1]

    # -- plans --
    def live_plan(self, activity: str, target: int, as_of: int) -> bool:
        """A plan made by ``as_of`` whose day has not yet been decided by then."""
        made = self.plans.get((activity, target))
        return made is not None and made <= as_of and (target > as_of or self.outcome(activity, target) is None)

    def kept(self, as_of: int, activity: str | None = None) -> tuple[int, int]:
        """(plans kept, plans that came due) up to ``as_of``, for one activity or all."""
        k = n = 0
        for (a, t), made in self.plans.items():
            if made <= as_of and t <= as_of and (activity is None or a == activity):
                k, n = k + int(self.outcome(a, t) == 1), n + 1
        return k, n

    # -- signals --
    def mood_mean(self, as_of: int, days: int) -> float | None:
        vals = [v for j in range(as_of - days + 1, as_of + 1) for v in self.moods.get(j, ())]
        return sum(vals) / len(vals) if vals else None

    def late_count(self, as_of: int, days: int) -> int:
        return sum(self.late.get(j, 0) for j in range(as_of - days + 1, as_of + 1))

    def mentioned(self, as_of: int, days: int) -> list[str]:
        return [s for j in range(as_of - days + 1, as_of + 1) for s in self.mentions.get(j, ())]

    def recent_answers(self, as_of: int, days: int) -> list[tuple[int, str, tuple[str, ...]]]:
        """The latest answer per subject from the last ``days`` days, up to ``as_of``."""
        latest: dict[str, tuple[int, str, tuple[str, ...]]] = {}
        for d, subject, toks in self.answers:
            if as_of - days < d <= as_of:
                latest[subject] = (d, subject, toks)
        return list(latest.values())

    def last_outcomes(self, activity: str, as_of: int, n: int) -> list[int]:
        return [y for _, y in self.record(activity, as_of)[-n:]]


def query_kind(query: Query, hist: PersonHistory, as_of: int) -> str:
    plan = is_plan_query(query) or hist.live_plan(query.subject, query.target_day, as_of)
    return "plan" if plan else "habit"
