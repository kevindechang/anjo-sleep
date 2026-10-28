# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
import json
from dataclasses import replace
from pathlib import Path

import pytest

from anjo.sleep.personas import model as m

DIR = Path(m.__file__).parent
PAIRS = (("film_student", "film_student_donor"), ("film_founder", "film_founder_donor"))
NAMES = tuple(n for pair in PAIRS for n in pair)


def load(name):
    return m.load_person_file(DIR / f"{name}.json")


@pytest.mark.parametrize("name", NAMES)
def test_persona_file_validates_and_meets_counts(name):
    pf = load(name)
    v, h = pf.visible, pf.held_out
    total = len(v.preferences) + len(h.preferences)
    assert total >= 75
    assert len(h.preferences) >= 15 and len(h.preferences) * 5 >= total
    assert len(h.new_work) >= 25 and len(v.plans) >= 3 and len(v.tone_requests) >= 3
    assert len(v.project.constraints) >= 3
    days = [w.day for w in h.new_work]
    assert min(days) == 2 and max(days) <= 8 + 7
    assert {w.parent_project for w in h.new_work} == {v.project.title}
    assert len(h.taste_ab) >= len(h.preferences)


@pytest.mark.parametrize("name", NAMES)
def test_held_out_text_absent_from_visible_part(name):
    pf = load(name)
    visible = json.dumps(m.to_visible_dict(pf.visible)).lower()
    for s in pf.held_out.preferences:
        assert s.text.lower() not in visible
    for w in pf.held_out.new_work:
        assert w.text.lower() not in visible


@pytest.mark.parametrize("name_a,name_b", PAIRS)
def test_pair_shares_taste_items_with_opposite_labels(name_a, name_b):
    a, b = load(name_a), load(name_b)
    assert a.visible.field == b.visible.field
    ta = {t.id: t for t in a.held_out.taste_ab}
    tb = {t.id: t for t in b.held_out.taste_ab}
    assert ta.keys() == tb.keys() and len(ta) >= 15
    for k in ta:
        assert ta[k].options == tb[k].options and ta[k].prompt == tb[k].prompt
        assert ta[k].pick != tb[k].pick
    assert {s.text for s in a.visible.preferences}.isdisjoint(s.text for s in b.visible.preferences)


@pytest.mark.parametrize("name", NAMES)
def test_no_medical_or_romance_content(name):
    banned = ("therap", "anxiety", "depress", "diagnos", "medic", "romance", "boyfriend",
              "girlfriend", "dating", "kiss", "crush")
    blob = (DIR / f"{name}.json").read_text().lower()
    assert not [w for w in banned if w in blob]


def test_loader_rejects_too_few_held_out_preferences():
    pf = load(NAMES[0])
    short = replace(pf, held_out=replace(pf.held_out, preferences=pf.held_out.preferences[:10]))
    with pytest.raises(m.PersonaError):
        m.validate_person_file(short)


def test_loader_rejects_new_work_outside_arrival_window():
    pf = load(NAMES[0])
    late = replace(pf.held_out.new_work[0], day=99)
    bad = replace(pf, held_out=replace(pf.held_out, new_work=(late, *pf.held_out.new_work[1:])))
    with pytest.raises(m.PersonaError):
        m.validate_person_file(bad)


def test_loader_rejects_held_out_preference_leaked_into_visible():
    pf = load(NAMES[0])
    leaked = replace(pf.held_out.preferences[0], text=pf.visible.preferences[0].text)
    bad = replace(
        pf, held_out=replace(pf.held_out, preferences=(leaked, *pf.held_out.preferences[1:]))
    )
    with pytest.raises(m.PersonaError):
        m.validate_person_file(bad)


def test_person_file_round_trips_through_dicts():
    pf = load(NAMES[1])
    data = {"visible": m.to_visible_dict(pf.visible), "held_out": m.to_held_out_dict(pf.held_out)}
    assert m.person_file_from_dict(data) == pf


def test_field_domains_required_lowercase_and_non_empty():
    pf = load(NAMES[0])
    for bad in ((), ("Film",), ("",)):
        with pytest.raises(m.PersonaError):
            m.validate_person_file(replace(pf, visible=replace(pf.visible, field_domains=bad)))
    assert "movies" in pf.visible.field_domains
