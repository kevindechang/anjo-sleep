# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
import json
import re
from pathlib import Path

import pytest

from anjo.sleep.personas import model as pm
from anjo.sleep.prereg import ledger
from anjo.sleep.prereg import register_personas as rp
from anjo.sleep.sets.loader import collapse_for, load_collapse

DIR = Path(pm.__file__).parent
PAIRS = (("founder", "founder_donor"), ("researcher", "researcher_donor"))
NAMES = [n for p in PAIRS for n in p]


def load(n):
    return pm.load_person_file(DIR / f"{n}.json")


@pytest.mark.parametrize("name", NAMES)
def test_new_persona_validates_and_meets_counts(name):
    pf = load(name)
    v, h = pf.visible, pf.held_out
    total = len(v.preferences) + len(h.preferences)
    assert total >= 75 and len(h.preferences) >= 15 and len(h.preferences) * 5 >= total
    assert len(h.new_work) >= 25 and len(v.plans) >= 3 and len(v.tone_requests) >= 3
    assert {w.parent_project for w in h.new_work} == {v.project.title}
    assert min(w.day for w in h.new_work) == 2 and max(w.day for w in h.new_work) <= 15
    assert v.age >= 18 and "hand_written" in json.loads((DIR / f"{name}.json").read_text())


@pytest.mark.parametrize("a,b", PAIRS)
def test_pairs_share_taste_items_with_opposite_picks(a, b):
    pa, pb = load(a), load(b)
    assert pa.visible.field == pb.visible.field
    ta = {t.id: t for t in pa.held_out.taste_ab}
    tb = {t.id: t for t in pb.held_out.taste_ab}
    assert ta.keys() == tb.keys() and len(ta) == 16
    for k in ta:
        assert ta[k].options == tb[k].options and ta[k].pick != tb[k].pick
    assert {s.text for s in pa.visible.preferences}.isdisjoint(s.text for s in pb.visible.preferences)


@pytest.mark.parametrize("name", NAMES)
def test_collapse_keeps_at_least_fifty(name):
    kept = collapse_for(load(name).visible.field_domains, load_collapse())
    assert len(kept) >= 50


@pytest.mark.parametrize("name", NAMES)
def test_domains_use_collapse_spellings_where_they_overlap(name):
    domains = {c.domain for c in load_collapse()}
    for d in load(name).visible.field_domains:
        assert d == d.lower()
    assert "recycling" in load("founder").visible.field_domains
    assert domains  # spellings come from the file itself


@pytest.mark.parametrize("name", NAMES)
def test_excluded_topics_absent(name):
    blob = (DIR / f"{name}.json").read_text().lower()
    banned = r"\b(therapy|therapist|anxiety|depress\w*|diagnos\w*|medical|romance|romantic|boyfriend|girlfriend|dating|dietitian|nutrition|tutoring|coaching|election|politic\w*)\b"
    assert not re.findall(banned, blob)
    assert "companion" not in blob


@pytest.mark.parametrize("name", NAMES)
def test_held_out_never_in_visible(name):
    pf = load(name)
    visible = json.dumps(pm.to_visible_dict(pf.visible)).lower()
    assert not [s for s in pf.held_out.preferences if s.text.lower() in visible]
    assert not [w for w in pf.held_out.new_work if w.text.lower() in visible]


def test_pairing_is_symmetric_and_names_only_registered_ids():
    ids = {p.stem for p in DIR.glob("*.json")}
    for a, b in rp.DEV_PAIRS:
        m = rp.pairing(a, b)
        assert m[m[a]] == a and m[m[b]] == b and set(m) == {a, b}
        assert set(m) <= ids
    assert rp.DEV_PAIRS == PAIRS


def files(*names):
    return tuple(DIR / f"{n}.json" for n in names)


def test_registration_hashes_only_what_it_is_given():
    items = rp.compute_items(files("founder", "founder_donor"))
    labels = {(i.kind, i.label) for i in items}
    assert labels == {
        ("persona_visible", "founder"), ("persona_held_out", "founder"),
        ("persona_visible", "founder_donor"), ("persona_held_out", "founder_donor"),
        ("donor_pairs", "founder+founder_donor")}
    assert not any("researcher" in i.label or "film" in i.label for i in items)


def test_donor_pair_hash_is_the_canonical_sorted_json():
    items = {i.kind: i.sha256 for i in rp.compute_items(files("researcher", "researcher_donor"))}
    body = b'{"researcher":"researcher_donor","researcher_donor":"researcher"}'
    assert items["donor_pairs"] == ledger.sha256_bytes(body)


def test_half_a_pair_or_a_film_persona_is_refused():
    with pytest.raises(rp.RegisterPersonasError):
        rp.compute_items(files("founder"))
    with pytest.raises(rp.RegisterPersonasError):
        rp.compute_items(files("film_student"))
    with pytest.raises(rp.RegisterPersonasError):
        rp.compute_items()


def test_retire_entry_is_the_hash_of_person_and_reason():
    items = rp.compute_items(retire="film_student", reason="replaced")
    assert [(i.kind, i.label) for i in items] == [("persona_retired", "film_student")]
    body = b'{"person_id":"film_student","reason":"replaced"}'
    assert items[0].sha256 == ledger.sha256_bytes(body)
    with pytest.raises(rp.RegisterPersonasError):
        rp.compute_items(retire="film_student")


def test_register_only_appends_and_skips_repeats(tmp_path, monkeypatch):
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    old = ledger.register("prompt_probe", "a" * 64, "old entry (before night 1)")
    items = rp.compute_items(files("founder", "founder_donor"), retire="film_student", reason="r")
    from anjo.sleep.prereg.register_night1 import register_items

    first = register_items(items, suffix=rp.SUFFIX)
    assert len(first) == len(items) == 6
    entries = ledger.read_entries()
    assert entries[0] == old and ledger.verify() == 7
    assert register_items(items, suffix=rp.SUFFIX) == []
    assert {e.kind for e in entries[1:]} == {
        "persona_visible", "persona_held_out", "donor_pairs", "persona_retired"}


def test_script_never_pushes_or_reads_a_token():
    src = Path(rp.__file__).read_text()
    assert "push_prereg" not in src.replace("run `push_prereg`", "") and "TOKEN" not in src


def _words(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3}


def _jaccard(a, b):
    x, y = _words(a), _words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


@pytest.mark.parametrize("name", NAMES)
def test_held_out_preferences_state_new_things(name):
    pf = load(name)
    visible = [s.text for s in pf.visible.preferences]
    worst = max(_jaccard(h.text, v) for h in pf.held_out.preferences for v in visible)
    assert worst < 0.30
    assert {s.domain for s in pf.held_out.preferences}.isdisjoint(
        {s.domain for s in pf.visible.preferences}
    )
    assert all(s.domain for s in pf.held_out.preferences)


@pytest.mark.parametrize("name", NAMES)
def test_new_work_text_is_not_in_the_visible_part(name):
    pf = load(name)
    blob = json.dumps(pm.to_visible_dict(pf.visible)).lower()
    plans = {p.text.lower() for p in pf.visible.plans}
    for w in pf.held_out.new_work:
        assert w.text.lower() not in blob and w.text.lower() not in plans
        assert _jaccard(w.text, pf.visible.project.logline) < 0.5
