# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
import json

import pytest

from anjo.sleep.sets import loader as s


def test_general_set_loads_with_provenance():
    items = s.load_general()
    assert len(items) == 200
    assert {i.source for i in items} == s.GENERAL_SOURCES
    assert all(0 <= i.answer < len(i.choices) for i in items)
    assert (s.SETS_DIR / "PROVENANCE.md").exists()


def test_anchors_are_disjoint_from_every_gate_item():
    anchors = s.load_anchors()
    assert len(anchors) == s.ANCHOR_ITEMS and {a.source for a in anchors} == s.ANCHOR_SOURCES

    def norm(text):
        return " ".join(text.lower().split())

    general = s.load_general()
    gate_text = {norm(g.question) for g in general} | {norm(c.prompt) for c in s.load_collapse()}
    gate_items = {(norm(g.question), tuple(norm(c) for c in g.choices)) for g in general}
    assert not {norm(a.question) for a in anchors} & gate_text
    assert not {(norm(a.question), tuple(norm(c) for c in a.choices)) for a in anchors} & gate_items
    assert not {a.id for a in anchors} & {g.id for g in general}


def test_anchors_refuse_a_gate_source(tmp_path):
    row = {"id": "a1", "source": "mmlu", "subject": "s", "question": "q", "choices": ["a", "b", "c"], "answer": 0}
    f = tmp_path / "a.jsonl"
    f.write_text(json.dumps(row) + "\n")
    with pytest.raises(s.SetError):
        s.load_anchors(f, expected=None)


def test_collapse_set_loads():
    assert len(s.load_collapse()) == 60


def test_collapse_excludes_prompts_in_the_persons_field():
    prompts = s.load_collapse()
    kept = s.collapse_for(("cooking",), prompts)
    assert len(kept) < len(prompts) and all(c.domain != "cooking" for c in kept)


def test_film_student_collapse_set_drops_film_prompts_and_keeps_fifty():
    from pathlib import Path

    from anjo.sleep.personas import model as m

    pf = m.load_person_file(Path(m.__file__).parent / "film_student.json")
    prompts = s.load_collapse()
    kept = s.collapse_for(pf.visible.field_domains, prompts)
    assert not [c for c in kept if c.domain in ("movies", "film", "tv", "video")]
    assert any(c.domain == "movies" for c in prompts)
    assert len(kept) >= 50


def test_general_rejects_bad_answer(tmp_path):
    row = {"id": "x", "source": "mmlu", "subject": "s", "question": "q",
           "choices": ["a", "b", "c", "d"], "answer": 7}
    f = tmp_path / "g.jsonl"
    f.write_text(json.dumps(row) + "\n")
    with pytest.raises(s.SetError):
        s.load_general(f, expected=None)


def test_duplicate_ids_rejected(tmp_path):
    row = {"id": "c1", "domain": "d", "prompt": "p"}
    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(s.SetError):
        s.load_collapse(f, expected=None)
