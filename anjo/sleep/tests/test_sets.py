import json

import pytest

from anjo.sleep.sets import loader as s


def test_general_set_loads_with_provenance():
    items = s.load_general()
    assert len(items) == 200
    assert {i.source for i in items} == s.GENERAL_SOURCES
    assert all(0 <= i.answer < len(i.choices) for i in items)
    assert (s.SETS_DIR / "PROVENANCE.md").exists()


def test_collapse_set_loads():
    assert len(s.load_collapse()) == 60


def test_collapse_excludes_prompts_in_the_persons_field():
    prompts = s.load_collapse()
    kept = s.collapse_for("cooking", prompts)
    assert len(kept) < len(prompts) and all(c.domain != "cooking" for c in kept)


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
