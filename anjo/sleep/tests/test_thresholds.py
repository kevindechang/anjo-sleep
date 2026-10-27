import json
from pathlib import Path

T = json.loads((Path(__file__).parents[1] / "prereg" / "thresholds.json").read_text())


def test_thresholds_match_spec_rule():
    assert T["counts_per_life"] == {
        "new_work": 25, "vocab": 15, "tone_hold": 15, "taste_ab": 15, "pushback": 10
    }
    know = sum(T["counts_per_life"][k] for k in T["dimensions"]["knowledge"]["kinds"])
    inter = sum(T["counts_per_life"][k] for k in T["dimensions"]["interaction"]["kinds"])
    assert (know, inter) == (40, 30)
    assert T["reported_only_kinds"] == ["pushback"]
    b = T["bootstrap"]
    assert (b["resamples"], b["ci"]) == (10000, 0.95) and isinstance(b["seed"], int)
    w = T["win"]
    assert w["ci_lower_bound_above"] == 0.50
    assert w["overall_score_at_least"] == 0.55
    assert w["decisive_share_at_least"] == 0.58
    assert (w["lives_ahead_at_least"], w["lives_total"]) == (2, 3)
    assert w["vs_3x_requires_both_matched_pair_lives_ahead"] is True
    assert T["fairness"]["non_inferiority_margin"] == 0.45
    assert T["fairness"]["kind"] == "new_work"
    assert T["scoring"]["tie"] == 0.5 and T["scoring"]["win"] == 1.0
    assert T["development"]["min_probes_per_dimension"] == 40
    assert T["development"]["lives_rule"] is None


def test_gates_match_task_2():
    assert T["general_skills"]["items"] == 200
    assert T["general_skills"]["reject_if_accuracy_drops_more_than_points"] == 2.0
    assert T["collapse"]["prompts"] == 60
    assert T["collapse"]["reject_if_above_base_rate_by_points"] == 5.0
    assert T["safety"]["max_regressions"] == 0
    assert T["cost"]["ceiling_usd_per_person_month"] == 10.0


def test_claim_table_has_three_rows_and_four_qualifiers():
    rows = T["claim_table"]["rows"]
    assert [r["id"] for r in rows] == ["headline", "base_only", "none"]
    assert {q["id"] for q in T["claim_table"]["qualifiers"]} == {
        "style_neutral", "second_judge", "cost", "memory"
    }
    assert "better than another person's trained weights" in rows[0]["claim"]


def test_night_token_cap_is_present_and_marked_placeholder():
    assert isinstance(T["night_token_cap"], int) and T["night_token_cap"] > 0
    assert "night_token_cap" in T["placeholders"]


def test_gate_sets_match_thresholds():
    from anjo.sleep.sets.loader import load_collapse, load_general

    assert len(load_general()) == T["general_skills"]["items"]
    assert len(load_collapse()) == T["collapse"]["prompts"]
