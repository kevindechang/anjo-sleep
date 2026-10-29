"""MIND core's reversal plant lines (MIND's data): a reversal's disambiguating fact states the raw
fact that dissolves the apparent problem, with the numbers, names and dates its planted-fact check
needs, and never what it means; joining it to the earlier fact is the reader's work at the surface.
A line that also said the problem was gone (``dry since``, ``loved it``) gave the whole item away on
its own day, so the leak gate (``leak_v1.md``) failed every life that drew it. APD's own templates
stay frozen: the one the core drew that states its outcome is kept out of the core's draws instead.
A plant line also never names the apparent problem beside what dissolves it ("the line on the desk
top sits in the wax coat" was both halves on one day), and neither does a decoy's: "she could come
at 7:40 pm" named the arrival it explained, and the leak gate flagged it, alone or with the next
day, in all three of one life's attempts (batch 6). Pure: no model is called."""
from __future__ import annotations

import re
from datetime import timedelta

from anjo.sleep.mind.apd import core_families, core_splits, families, generate
from anjo.sleep.mind.apd.generate import LIFE_START, _key_terms
from anjo.sleep.mind.apd.types import CORE_SPLITS, FAMILIES, MAX_PREVALENCE, CoreSplit

# The lines as they were (core-v1), by the id they had, and the id each one has now: a new id where
# no pinned set names the old one, the same id where the pinned pair specs do (core-test). A weekday
# a line named may give way to the date it pins (``test_apd_core_dates``).
BEFORE = {
    "truck-cold-reviews": "All three reviews were delivery app orders from Monday, when the app's couriers ran an "
                          "hour late, and walk-up customers have said nothing of the kind.",
    "furniture-desk-line": "Wiping the line with mineral spirits took it away: it was in the wax layer, and the wood "
                           "beneath is whole.",
    "podcast-download-dip": "The hosting service says its dashboard stopped counting one big podcast app for a week, "
                            "and its corrected numbers, posted on Friday, show downloads level.",
    "pottery-celadon-lines": "The celadon is a crackle glaze sold for its web of fine lines, and the bowls hold water "
                             "without leaking.",
    "bikes-brake-squeal": "The brake maker's guide says new pads squeal until they bed in over about 20 stops, and "
                          "riders who came back after a week say the squeal has gone.",
    "wedding-salty-lamb": "The tasting lamb was the only dish a trainee cooked, and the couple retasted the lamb on "
                          "Tuesday and loved it.",
    "theatre-hoarse-lead": "The lead had shouted at a football match on Sunday, rested his voice on Tuesday, and sang "
                           "all of Thursday's run clearly.",
    "photo-green-cast": "I'd viewed the test prints under the studio's new fluorescent tubes, and under the gallery's "
                        "daylight lamps the same prints look neutral.",
}
NOW = {
    "truck-cold-reviews": "truck-cold-reviews-v3", "furniture-desk-line": "furniture-desk-line-v3",
    "podcast-download-dip": "podcast-download-dip-v4", "pottery-celadon-lines": "pottery-celadon-lines-v3",
    "bikes-brake-squeal": "bikes-brake-squeal-v2", "wedding-salty-lamb": "wedding-salty-lamb",
    "theatre-hoarse-lead": "theatre-hoarse-lead", "photo-green-cast": "photo-green-cast",
}
GONE = ("truck-cold-reviews-v2", "furniture-desk-line-v2", "podcast-download-dip-v2",  # ids a later rewrite took over
        "podcast-download-dip-v3", "pottery-celadon-lines-v2")
KEPT_OUT = (
    "reno-damp-patch",  # APD's, frozen: it states its outcome ("the patch has been dry since")
    # a day only the day it's said can place ("next Monday", "Friday's order"), or a date inside the life
    "reno-tile-membrane", "reno-floor-cabinets", "film-rehearsal-late", "bake-night-oven", "bake-online-orders",
    # a plant line that says what its fact means (``APD_SWEPT``)
    "film-score-room", "reno-skip-share", "bake-rye-sales", "reno-wiring-quote",
)
# Words that tell a reader the apparent problem is gone; a reversal's plant line never needs them.
OUTCOME_WORDS = ("since", "whole", "nothing of the kind", "has gone", "loved", "clearly", "neutral", "level",
                 "without leaking", "is fine", "are fine", "no longer", "as it should", "took it away", "still damp")
WEEKDAY = re.compile(r"\b(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b")
WEEKDAYS = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"})
DAMP_HOURS = 72  # ``reno-tile-shade``: how long new tiles stay damp, by its plant line


def _drawn(split: CoreSplit) -> tuple[families.Template, ...]:
    return tuple(t for name in core_families.fields(split) for t in core_families.templates_for(split, name))


def test_the_template_sets_load_and_every_core_field_still_fills_a_life() -> None:
    for split in CORE_SPLITS:
        for name in core_families.fields(split):
            found = core_families.templates_for(split, name)
            assert len(found) == MAX_PREVALENCE and {t.family for t in found} == set(FAMILIES), (split, name)
    assert all(core_families.template(new) is not None for new in NOW.values())


def test_each_rewritten_reversal_line_keeps_its_evidence_tokens() -> None:
    for old, line in BEFORE.items():
        rewritten = core_families.template(NOW[old])
        assert rewritten is not None and rewritten.family == "reversal", old
        assert rewritten.disambiguating != line, old
        kept = _key_terms(line) - WEEKDAYS  # every number, name and date kept; a weekday may be pinned instead
        assert kept <= _key_terms(rewritten.disambiguating), old
    renamed = [old for old, new in NOW.items() if new != old]
    assert all(core_families.template(old) is None for old in (*renamed, *GONE))  # no id keeps another's text


def test_no_reversal_line_the_core_draws_states_its_outcome() -> None:
    for split in CORE_SPLITS:
        for t in (t for t in _drawn(split) if t.family == "reversal"):
            assert not any(word in t.disambiguating.lower() for word in OUTCOME_WORDS), (split, t.id)


def test_the_desk_line_states_the_wax_and_never_the_line_it_explains() -> None:
    desk = core_families.template("furniture-desk-line-v3")
    assert desk is not None and desk.family == "reversal"
    line = desk.disambiguating.lower()
    # the raw facts the reader joins to the crack: when the wax went on, and what the tin warns
    assert {"6", "10"} <= _key_terms(desk.disambiguating) and "crazes" in line
    # never the apparent problem beside what dissolves it (``leak_v1.md``: both halves on one day)
    assert not any(word in line for word in ("crack", "hairline", "the line", "wax coat", "the wood"))
    assert not WEEKDAY.search(desk.disambiguating)  # nothing a schedule could put before the crack


def test_the_core_draws_keep_out_the_apd_templates_it_replaces() -> None:
    drawn = {t.id for split in CORE_SPLITS for t in _drawn(split)}
    apd = {t.id for t in families.TEMPLATES["development"]}
    for kept_out in KEPT_OUT:
        assert kept_out not in drawn and kept_out in apd
        assert core_families.template(kept_out) is not None  # still APD's, and findable by its id
        field = families.TEMPLATES["development"][[t.id for t in families.TEMPLATES["development"]].index(kept_out)].field
        assert kept_out in {t.id for t in families.templates_for("development", field)}  # APD's draws unchanged
    assert apd - set(KEPT_OUT) <= drawn  # every other APD development template still drawn by core-train


def test_reno_tile_shade_holds_on_every_day_its_draws_put_it() -> None:
    # Arrange: every core-train life's every attempt that draws it, as scheduled
    tiles = core_families.template("reno-tile-shade")
    assert tiles is not None
    drawn = [(attempt, plan) for spec in core_splits.core_specs("core-train") if spec.field == tiles.field
             for attempt in range(generate.MAX_ATTEMPTS)
             for plan in generate.plan(spec.split, spec.field, spec.families, spec.prevalence,
                                       generate.attempt_seed(spec.seed, attempt))
             if plan.template.id == tiles.id]

    # Assert: 13 lives draw it on their first attempt; on each draw the tiles are compared the morning
    # they went up (no weekday a schedule could put more than 72 hours back), and the question, days
    # later, is never told they are still damp
    assert sum(1 for attempt, _ in drawn if attempt == 0) == 13
    (origin,) = tiles.origins
    assert "this morning" in origin and str(DAMP_HOURS) in tiles.disambiguating
    for _, plan in drawn:
        compared = LIFE_START + timedelta(days=plan.origin_days[0])
        went_up = compared  # "this morning": the day of the comparison
        assert (compared - went_up) <= timedelta(hours=DAMP_HOURS)
        assert plan.origin_days[0] < plan.disambiguating_day < plan.surface_day
        for fact in (*tiles.origins, tiles.disambiguating, tiles.resolution):
            assert not WEEKDAY.search(fact) and "still damp" not in fact, fact


def test_the_rehearsal_line_states_the_class_and_never_the_arrival_it_explains() -> None:
    late = core_families.template("film-rehearsal-late-core-v2")
    assert late is not None and late.family == "decoy" and core_families.template("film-rehearsal-late-core") is None
    origins = set().union(*(_key_terms(origin) for origin in late.origins))
    line = late.disambiguating.lower()
    # the raw fact the reader joins to the 7:40 arrival: a class this month that ends at 7:15 pm, agreed
    assert "7:15" in _key_terms(late.disambiguating) and "class" in line and "agreed" in line
    # never the arrival it explains (``leak_v1.md``: both halves on one day)
    assert not _key_terms(late.disambiguating) & origins
    assert not any(word in line for word in ("7:40", "late", "arriv", "call sheet", "7:00"))
    assert "7:40" in late.surface and "7:15" in late.resolution


# Every plant line that stated its own meaning or outcome beside its fact (leak_v1.md:19-20, 25-27),
# as it was, by the id it had, and the id it has now: a new id in core-train and core-val, the same
# id in core-test, whose ids the pinned pair specs name. Swept 2026-10-29, after batch 6.
SWEPT = {
    "film-rehearsal-late-core": ("film-rehearsal-late-core-v2", "I texted the lead actor that she could come to "
                                 "rehearsals at 7:40 pm this month because of her evening class."),
    "podcast-download-dip-v3": ("podcast-download-dip-v4", "The hosting service says its dashboard stopped counting "
                                "one big podcast app for a week, and its corrected numbers, posted on Friday, put "
                                "episode 9 at 5,200 downloads and episode 8 at 5,100."),
    "pottery-celadon-lines-v2": ("pottery-celadon-lines-v3", "The celadon is a crackle glaze, sold for its web of "
                                 "fine lines."),
    "pottery-seconds-plates": ("pottery-seconds-plates-v2", "The restaurant's chef said small flaws are fine and "
                               "wants 30 plates by next month."),
    "bikes-august-sales": ("bikes-august-sales-v2", "Every year since the shop opened, August sales have been about "
                           "half of July's while the town is away on its summer holidays."),
    "bikes-courier-fleet": ("bikes-courier-fleet-v2", "The courier company's manager said they'd buy the bikes and "
                            "pay us to service them every month."),
    "wedding-rsvp-count": ("wedding-rsvp-count", "The couple told me the other 60 are family who confirmed by phone "
                           "and will be on the final list."),
    "theatre-school-matinee": ("theatre-school-matinee", "The school's head of English said they'd book a full house "
                               "of 200 students for a Wednesday matinee at 8 each."),
    "photo-framing-quote": ("photo-framing-quote", "This show's prints are twice the size and use museum glass, which "
                            "the last show's didn't."),
}
# APD's own plant lines that say what their fact means, drawn by core-train until 2026-10-29: APD's stay
# frozen, so each is kept out of the core's draws (``CORE_KEPT_OUT``) and drawn in a ``-core`` version.
APD_SWEPT = {
    "film-score-room": ("film-score-room-core", "The church hall caretaker said any recording is fine as long as it "
                        "ends by ten."),
    "reno-skip-share": ("reno-skip-share-core", "The neighbour said they'd happily share the bin; it's a large "
                        "six-metre one, half empty."),
    "bake-rye-sales": ("bake-rye-sales-core", "The mill's rye deliveries stopped for two weeks because of road works, "
                       "and the rye loaf was off the shelf for those two weeks; it was back on the shelf last Monday."),
    "reno-wiring-quote": ("reno-wiring-quote-core", "The quote includes replacing the old fuse box and adding a garden "
                          "socket circuit, which my friend's job did not."),
}
# Phrases that state what a fact means or how it came out, built from every case so far: crude, but a
# plant line that brings one back fails here before a live run pays for it.
STATED = ("is fine", "are fine", "was fine", "fine after all", "not a problem", "no problem", "nothing of the kind",
          "has gone", "no longer", "dry since", "loved it", "clearly", "neutral", "level", "without leaking",
          "took it away", "is whole", "still damp", "as it should", "sold for", "corrected", "about half",
          "the other", "will be on the", "full house", "could come", "pay us", "happily", "back on the", "did not",
          "didn't")


def _stated(line: str) -> list[str]:
    return [phrase for phrase in STATED if re.search(rf"\b{re.escape(phrase)}\b", line.lower())]


def test_the_stated_outcome_phrases_catch_every_case_so_far() -> None:
    cases = [*BEFORE.values(), *(line for _, line in (*SWEPT.values(), *APD_SWEPT.values())),
             "The patch has been dry since the gutter was cleared."]  # reno-damp-patch's kind, APD's own
    assert [line for line in cases if not _stated(line)] == []


def test_no_plant_line_drawn_into_a_core_split_states_its_meaning_or_outcome() -> None:
    # every plant line any core split draws, of every family, APD's own included (kept out when one fails)
    stated = {t.id: _stated(t.disambiguating) for split in CORE_SPLITS for t in _drawn(split)}
    assert {tid: found for tid, found in stated.items() if found} == {}


def test_each_apd_line_that_says_its_meaning_is_drawn_in_a_core_version_and_kept_for_apd() -> None:
    drawn = {t.id: t for t in _drawn("core-train")}
    apd = {t.id: t for t in families.TEMPLATES["development"]}
    for old, (new, line) in APD_SWEPT.items():
        assert old in core_families.CORE_KEPT_OUT and old not in drawn and apd[old].disambiguating == line, old
        mine = drawn[new]
        assert (mine.family, mine.field, mine.origins, mine.surface) == (apd[old].family, apd[old].field,
                                                                         apd[old].origins, apd[old].surface), old
        assert mine.disambiguating != line and not _stated(mine.disambiguating), old


def test_each_swept_line_is_rewritten_under_its_splits_id_rule() -> None:
    test_ids = {t.id for name in core_families.fields("core-test") for t in core_families.templates_for("core-test", name)}
    for old, (new, line) in SWEPT.items():
        now = core_families.template(new)
        assert now is not None and now.disambiguating != line, old
        assert (new == old) == (old in test_ids), old  # core-test keeps its ids; the others take new ones
        assert new == old or core_families.template(old) is None, old
