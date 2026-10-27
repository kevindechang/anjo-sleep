# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
import json
from dataclasses import replace

import pytest

from anjo.sleep.lives import draw as d


@pytest.fixture(scope="module")
def pairs():
    return d.load_pairs()


def test_lives_json_has_twenty_same_field_pairs(pairs):
    assert len(pairs) == 20
    assert len({p.field for p in pairs}) == 20
    assert all(len(p.lives) == 2 for p in pairs)
    assert all(len(lf.plans) >= 3 for p in pairs for lf in p.lives)


def test_lives_are_same_field_pairs_with_opposite_taste_labels(pairs):
    for p in pairs:
        assert p.taste_ab
        a, b = p.lives
        for t in p.taste_ab:
            assert t.pick[a.id] != t.pick[b.id]
            assert set(t.pick) == {a.id, b.id}


def test_lives_exclude_anti_project_domains(pairs):
    for p in pairs:
        for lf in p.lives:
            assert not d._BANNED.search(p.field) and not d._BANNED.search(lf.line)


def test_banned_domain_is_rejected(tmp_path):
    data = json.loads(d.LIVES_PATH.read_text())
    data["pairs"][0]["field"] = "sports coaching"
    f = tmp_path / "l.json"
    f.write_text(json.dumps(data))
    with pytest.raises(d.LivesError):
        d.load_pairs(f)


def test_same_label_pick_is_rejected(tmp_path):
    data = json.loads(d.LIVES_PATH.read_text())
    t = data["pairs"][0]["taste_ab"][0]
    t["pick"] = {k: 0 for k in t["pick"]}
    f = tmp_path / "l.json"
    f.write_text(json.dumps(data))
    with pytest.raises(d.LivesError):
        d.load_pairs(f)


def test_draw_is_reproducible_from_seed():
    assert d.draw_lives(12345) == d.draw_lives(12345)
    assert len({d.draw_lives(s).pair[0].id for s in range(40)}) > 5


def test_single_life_drawn_from_another_pair():
    for seed in range(200):
        dr = d.draw_lives(seed)
        assert dr.single.pair_id != dr.pair[0].pair_id
        assert dr.single.field != dr.pair[0].field


def test_donor_map_fixed_by_registered_rule():
    for seed in (0, 1, 99):
        dr = d.draw_lives(seed)
        a, b = dr.pair
        assert dr.donors == {a.id: b.id, b.id: a.id, dr.single.id: a.id}
        assert d.donor_map(dr.pair, dr.single) == dr.donors
    assert "first life of the drawn pair" in d.DONOR_RULE


def test_expected_pair_count_is_enforced(tmp_path):
    data = json.loads(d.LIVES_PATH.read_text())
    data["pairs"] = data["pairs"][:5]
    f = tmp_path / "l.json"
    f.write_text(json.dumps(data))
    with pytest.raises(d.LivesError):
        d.load_pairs(f)
    assert len(d.load_pairs(f, expected_pairs=5)) == 5


def test_life_dataclass_is_frozen(pairs):
    with pytest.raises(AttributeError):
        replace(pairs[0].lives[0]).id = "x"


def test_draw_enforces_twenty_pairs_and_registered_hash(tmp_path):
    import hashlib

    good = hashlib.sha256(d.LIVES_PATH.read_bytes()).hexdigest()
    assert d.draw_lives(3, expected_sha256=good) == d.draw_lives(3)
    with pytest.raises(d.LivesError):
        d.draw_lives(3, expected_sha256="0" * 64)
    data = json.loads(d.LIVES_PATH.read_text())
    data["pairs"] = data["pairs"][:5]
    f = tmp_path / "l.json"
    f.write_text(json.dumps(data))
    with pytest.raises(d.LivesError):
        d.draw_lives(3, f)
