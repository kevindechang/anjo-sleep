"""MIND core's reversal plant lines (MIND's data): a reversal's disambiguating fact states the raw
fact that dissolves the apparent problem, with the numbers, names and dates its planted-fact check
needs, and never what it means; joining it to the earlier fact is the reader's work at the surface.
A line that also said the problem was gone (``dry since``, ``loved it``) gave the whole item away on
its own day, so the leak gate (``leak_v1.md``) failed every life that drew it. APD's own templates
stay frozen: the one the core drew that states its outcome is kept out of the core's draws instead.
Pure: no model is called."""
from __future__ import annotations

from anjo.sleep.mind.apd import core_families, families
from anjo.sleep.mind.apd.generate import _key_terms
from anjo.sleep.mind.apd.types import CORE_SPLITS, FAMILIES, MAX_PREVALENCE, CoreSplit

# The lines as they were (core-v1), by the id they had, and the id each one has now: a new id where
# no pinned set names the old one, the same id where the pinned pair specs do (core-test).
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
    "truck-cold-reviews": "truck-cold-reviews-v2", "furniture-desk-line": "furniture-desk-line-v2",
    "podcast-download-dip": "podcast-download-dip-v2", "pottery-celadon-lines": "pottery-celadon-lines-v2",
    "bikes-brake-squeal": "bikes-brake-squeal-v2", "wedding-salty-lamb": "wedding-salty-lamb",
    "theatre-hoarse-lead": "theatre-hoarse-lead", "photo-green-cast": "photo-green-cast",
}
KEPT_OUT = ("reno-damp-patch",)  # APD's, frozen: it states its outcome ("the patch has been dry since")
# Words that tell a reader the apparent problem is gone; a reversal's plant line never needs them.
OUTCOME_WORDS = ("since", "whole", "nothing of the kind", "has gone", "loved", "clearly", "neutral", "level",
                 "without leaking", "is fine", "are fine", "no longer", "as it should")


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
        assert _key_terms(line) <= _key_terms(rewritten.disambiguating), old  # every number, name and date kept
    renamed = [old for old, new in NOW.items() if new != old]
    assert all(core_families.template(old) is None for old in renamed)  # an old line keeps no id it never had


def test_no_reversal_line_the_core_draws_states_its_outcome() -> None:
    for split in CORE_SPLITS:
        for t in (t for t in _drawn(split) if t.family == "reversal"):
            assert not any(word in t.disambiguating.lower() for word in OUTCOME_WORDS), (split, t.id)


def test_the_core_draws_keep_out_apds_reversals_that_state_their_outcome() -> None:
    drawn = {t.id for split in CORE_SPLITS for t in _drawn(split)}
    apd = {t.id for t in families.TEMPLATES["development"]}
    for kept_out in KEPT_OUT:
        assert kept_out not in drawn and kept_out in apd
        assert core_families.template(kept_out) is not None  # still APD's, and findable by its id
        field = families.TEMPLATES["development"][[t.id for t in families.TEMPLATES["development"]].index(kept_out)].field
        assert kept_out in {t.id for t in families.templates_for("development", field)}  # APD's draws unchanged
    assert apd - set(KEPT_OUT) <= drawn  # every other APD development template still drawn by core-train
