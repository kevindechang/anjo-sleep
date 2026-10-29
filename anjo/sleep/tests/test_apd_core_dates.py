"""The days MIND core's template facts name (MIND's data): a fact is said on the day its schedule
puts it, and a schedule can put it on any of several days, so what it names must hold whichever.

- **A day named relative to the writing** (a singular weekday such as "Monday's rehearsal",
  "next week", "yesterday") resolves from the day it is said. Two facts said on different days
  that both name such a day can name two different dates for one event (2026-10-29: "Sunday's
  takings" said on Friday and "on Sunday" said on the next Tuesday named two Sundays), so a
  template keeps them to one fact; the others pin the date ("Sunday the 6th").
- **A pinned date** lies outside the life's days, so it is on the same side of every day a fact
  naming it can be said: its tense never turns ("is booked for the 12th", said on the 13th) and is
  never wrong in every draw ("Opening night is on the 14th", always said on the 15th or 16th). A
  week ("the week of the 8th") keeps one side in every draw. A weekday pinned to a date is its own.
- **A recurring day** ("on Tuesdays", "Friday mornings", "Monday to Thursday") holds any day.

Every scheduled draw is checked: each core split's every spec and attempt (``generate.plan``),
and the pairs' unrelated items before the fork (``pairs.prefix_plan``). Pure: no model is called.
"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import date, timedelta

from anjo.sleep.mind.apd import core_families, core_splits, generate, pairs
from anjo.sleep.mind.apd.families import Template
from anjo.sleep.mind.apd.generate import LIFE_START, Dice, ItemPlan
from anjo.sleep.mind.apd.types import CORE_SPLITS, LIFE_DAYS

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_ORDINAL = r"the (\d{1,2})(?:st|nd|rd|th)\b"
PINNED = re.compile(rf"\b({'|'.join(DAYS)}) {_ORDINAL}")  # "Sunday the 6th"
WEEK_OF = re.compile(rf"\bthe week of {_ORDINAL}")  # "the week of the 8th": that Monday to Sunday
DATED = re.compile(rf"\b{_ORDINAL}")  # "the 26th"
WEEKDAY = re.compile(rf"\b(?:{'|'.join(DAYS)})\b")  # singular: "Tuesdays" is not matched
RELATIVE = re.compile(r"\b(?:(?:this|last|next) week|the week before|yesterday|tomorrow)\b", re.IGNORECASE)
# Days that recur, true whichever day says them; each is found in a drawn template (no stale entry).
RECURRING = ("on a Saturday", "Tuesday and Wednesday evenings", "Monday to Thursday", "Wednesday to Friday",
             "a Wednesday matinee", "the Saturday class", "the Saturday farmers' market")
_PARTS = re.compile(rf"\b(?:{'|'.join(DAYS)}) (?:mornings|afternoons|evenings|nights)\b")  # "Friday mornings"
MONTHS = ((2026, 8), (2026, 9), (2026, 10))  # a life's days are in September 2026; a pinned date is near them
LAST_DAY = LIFE_START + timedelta(days=LIFE_DAYS - 1)


def _drawn() -> dict[str, Template]:
    """Every template a core split's life or a pair's prefix can draw, by id."""
    return {t.id: t for split in CORE_SPLITS for name in core_families.fields(split)
            for t in core_families.templates_for(split, name)}


def _facts(t: Template) -> list[str]:
    """A template's facts in schedule order: its origins, its disambiguating fact, its surface."""
    return [*t.origins, t.disambiguating, t.surface]


def _days(plan: ItemPlan) -> list[int]:
    return [*plan.origin_days, plan.disambiguating_day, plan.surface_day]


def _plans() -> Iterator[ItemPlan]:
    for split in CORE_SPLITS:
        for spec in core_splits.core_specs(split):
            for attempt in range(generate.MAX_ATTEMPTS):
                yield from generate.plan(spec.split, spec.field, spec.families, spec.prevalence,
                                         generate.attempt_seed(spec.seed, attempt))
    for pair in pairs.pair_specs():
        unrelated = core_families.template(pair.unrelated)
        assert unrelated is not None
        for attempt in range(generate.MAX_ATTEMPTS):
            yield pairs.prefix_plan(unrelated, Dice(f"{pairs.PAIR_LABEL}:{generate.attempt_seed(pair.seed, attempt)}"))


def _weekday_date(weekday: str, day: int) -> date:
    """The one date near a life whose day of the month is ``day`` and weekday is ``weekday``."""
    found = [d for year, month in MONTHS if (d := _on(year, month, day)) is not None and DAYS[d.weekday()] == weekday]
    assert len(found) == 1, (weekday, day, found)  # the weekday names a date the life is near
    return found[0]


def _on(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _pinned(fact: str) -> list[tuple[date, date]]:
    """The dates a fact pins, each as the span it names (a day, or the week of a day)."""
    found: list[tuple[date, date]] = []
    for weekday, day in PINNED.findall(fact):
        found.append((_weekday_date(weekday, int(day)),) * 2)
    rest = PINNED.sub("", fact)
    for day in WEEK_OF.findall(rest):
        start = date(2026, 9, int(day)) - timedelta(days=date(2026, 9, int(day)).weekday())
        found.append((start, start + timedelta(days=6)))
    for day in DATED.findall(WEEK_OF.sub("", rest)):  # a date with no weekday: its side is what's checked
        found.append((date(2026, 9, int(day)),) * 2)
    return found


def _relative(fact: str) -> list[str]:
    """What a fact names relative to the day it is said: singular weekdays, unpinned and not
    recurring, and the days and weeks around it."""
    rest = PINNED.sub("", fact)
    for phrase in RECURRING:
        rest = rest.replace(phrase, "")
    rest = _PARTS.sub("", rest)
    return WEEKDAY.findall(rest) + RELATIVE.findall(rest)


def _side(said: date, span: tuple[date, date]) -> str:
    return "after" if span[0] > said else "before" if span[1] < said else "during"


def test_every_recurring_phrase_is_still_drawn() -> None:
    texts = " ".join(fact for t in _drawn().values() for fact in _facts(t))
    assert all(phrase in texts for phrase in RECURRING), [p for p in RECURRING if p not in texts]


def test_a_day_named_relative_to_its_writing_is_named_in_one_fact_at_most() -> None:
    named = {t.id: [i for i, fact in enumerate(_facts(t)) if _relative(fact)] for t in _drawn().values()}
    assert {tid: facts for tid, facts in named.items() if len(facts) > 1} == {}


def test_a_pinned_weekday_is_its_dates_own() -> None:
    for t in _drawn().values():
        for fact in (*_facts(t), t.resolution):
            for weekday, day in PINNED.findall(fact):
                assert DAYS[_weekday_date(weekday, int(day)).weekday()] == weekday, (t.id, fact)


def test_every_scheduled_draw_keeps_each_pinned_date_on_one_side_of_the_day_it_is_said() -> None:
    # Arrange
    drawn, sides = _drawn(), defaultdict(set)

    # Act: every draw's every fact, said on its scheduled day, against each date it pins
    for plan in _plans():
        assert plan.template.id in drawn
        for index, (fact, day) in enumerate(zip(_facts(plan.template), _days(plan), strict=True)):
            said = LIFE_START + timedelta(days=day)
            assert LIFE_START <= said <= LAST_DAY
            for span in _pinned(fact):
                sides[(plan.template.id, index, span)].add(_side(said, span))

    # Assert: a pinned day lies outside the life's days; a week keeps one side in every draw
    inside = {(tid, index, span[0]) for tid, index, span in sides if span[0] == span[1]
              and LIFE_START <= span[0] <= LAST_DAY}
    assert inside == set()
    assert {key: found for key, found in sides.items() if len(found) > 1} == {}


def test_the_sunday_takings_name_the_6th_in_every_fact() -> None:
    takings = core_families.template("truck-sunday-takings-v2")
    assert takings is not None and core_families.template("truck-sunday-takings") is None
    sixth = date(2026, 9, 6)
    assert DAYS[sixth.weekday()] == "Sunday" and sixth < LIFE_START
    for fact in (*_facts(takings), takings.resolution):
        assert "the 6th" in fact and not _relative(fact), fact
