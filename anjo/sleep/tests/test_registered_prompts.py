# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
from pathlib import Path

from anjo.sleep.prereg import register_night1 as r

SLEEP = Path(r.__file__).parents[1]


def read(rel):
    return (SLEEP / rel).read_text()


def test_judge_prompt_warns_about_dates_and_names_and_has_slots():
    t = read("probes/prompts/judge_v1.md")
    assert "missing a date or a name" in t
    for slot in ("{{BRIEF}}", "{{PROMPT}}", "{{REPLY_A}}", "{{REPLY_B}}"):
        assert slot in t
    assert "tie" in t


def test_neutral_prompt_limits_length():
    t = read("probes/prompts/neutral_v1.md")
    assert "under 150 words" in t and "{{REPLY}}" in t and "Keep every point" in t


def test_brief_prompt_has_the_five_sections_in_order():
    t = read("context/prompts/brief_v1.md")
    heads = [
        "## The project and its constraints", "## Stated preferences", "## Corrections",
        "## What homework found, with sources", "## What practice showed",
    ]
    pos = [t.index(h) for h in heads]
    assert pos == sorted(pos)


def test_probe_prompt_is_cue_free_and_held_out_only():
    t = read("probes/prompts/probe_v1.md")
    for kind in ("tone_hold", "taste_ab", "vocab", "pushback", "new_work"):
        assert kind in t
    assert "No cues" in t and "SOURCES" in t and "held-out" in t


def test_sim_prompt_has_no_held_out_slot():
    t = read("sim/prompts/sim_v1.md")
    assert "held" not in t.lower() and "{{PREFERENCES}}" in t


def test_registration_list_is_stable_and_complete():
    items = r.compute_items()
    kinds = [i.kind for i in items]
    for k in ("prompt_probe", "prompt_judge", "prompt_neutral", "prompt_brief", "prompt_sim", "thresholds",
              "set_general", "set_collapse", "lives", "donor_rule"):
        assert k in kinds
    assert kinds.count("persona_visible") == 2 and kinds.count("persona_held_out") == 2
    assert items == r.compute_items()


def test_visible_and_held_out_hashes_are_separate():
    items = {(i.kind, i.label): i.sha256 for i in r.compute_items()}
    assert items[("persona_visible", "film_student")] != items[("persona_held_out", "film_student")]


def test_register_writes_ledger_and_skips_repeats(tmp_path, monkeypatch):
    from anjo.sleep.prereg import ledger

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    items = [i for i in r.compute_items() if not i.missing]
    first = r.register_items(items)
    assert len(first) == len(items) and ledger.verify() == len(items)
    assert r.register_items(items) == []


def test_register_refuses_when_an_input_is_missing(tmp_path, monkeypatch):
    import pytest

    from anjo.sleep.prereg import ledger

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    items = r.compute_items(root=SLEEP)
    items = [*items[:1], r.Item("x", "gone", None)]
    with pytest.raises(ledger.LedgerError):
        r.register_items(items)
    assert not (tmp_path / "sleep" / "ledger.jsonl").exists()
