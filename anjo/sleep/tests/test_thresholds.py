# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
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


def test_replays_token_penalty_is_registered_here_as_a_dated_placeholder():
    from anjo.sleep.mind.replay import settings

    assert settings.token_penalty() == T["replay_token_penalty"] == 0.01
    note = T["placeholders"]["replay_token_penalty"]
    assert note.startswith("2026-10-28:") and "a priori on fakes" in note and "not set on development" in note
    assert "before the Nov 1 12:00 ET registration" in note


def test_gate_sets_match_thresholds():
    from anjo.sleep.sets.loader import load_collapse, load_general

    assert len(load_general()) == T["general_skills"]["items"]
    assert len(load_collapse()) == T["collapse"]["prompts"]


def test_thresholds_pin_the_review_items():
    # 1. the win rule is applied per dimension, and each dimension is claimed separately
    assert T["win"]["applies_per"] == "dimension"
    assert T["win"]["applies_to"] == ["knowledge", "interaction"]
    assert T["win"]["comparisons"] == ["3 vs 1", "3 vs 3x"]
    # 2. checkpoint rule
    c = T["checkpoint_rule"]
    assert "last promoted" in c["scored"] and "base model" in c["scored"]
    assert c["never"] == "the best night" and c["scoring_runs"] == 1
    # 3. placeholders
    ph = T["placeholders"]
    assert {"bootstrap.seed", "night_token_cap", "nights_per_life_final"} <= set(ph)
    assert ph["nights_per_life_final"] == "set at the Nov 1 freeze, before the reveal (null until then)"
    assert T["nights_per_life_final"] is None
    # 4. fairness: two separate comparisons, no donor conditions
    f = T["fairness"]
    assert [x["id"] for x in f["comparisons"]] == ["4_vs_2C", "4_vs_2F"]
    assert all(x["kind"] == "new_work" for x in f["comparisons"])
    assert "above 45%" in f["each_comparison_rule"]["matches"]
    assert "3x" not in f["each_comparison_rule"]["win"]
    assert f["reported"] == "once"
    # 5. collapse scope
    assert "field_domains" in T["collapse"]["scope"]
    # 6. donor rule in full
    d = T["donor_rule"]
    assert "one pair plus one single life" in d["draw"]
    assert "each other's 3x" in d["pair_members"]
    assert "first member" in d["single_life_donor"]
    assert "fixed at the draw and logged" in d["fixed"]
    # 7. bootstrap blocks and scope
    b = T["bootstrap"]
    assert "every probe derived from it" in b["blocks"]["taste_ab"]
    assert "not a population estimate" in b["interval_scope"]
    # 8. human check, judges, working product, fallback
    h = T["human_check"]
    assert (h["final_pairs"], h["strata"]) == (30, ["3 vs 1", "3 vs 3x"])
    assert "sample_seed" in h and "human_check.sample_seed" in ph
    assert T["judging"]["style_neutral"]["kinds"] == ["new_work", "vocab"]
    assert "interaction" not in T["judging"]["style_neutral"]["kinds"]
    assert "per life and per dimension" in T["judging"]["second_judge"]["report"]
    assert "not yet at a viable cost" in T["working_product"]["over_ceiling_wording"]
    assert "MIND" in T["no_win_fallback"]["claim"]
    # 9. hand-written note
    assert "final lives" in T["hand_written"]["founder_and_researcher_pairs_taste_ab"]
    assert "retired before any" in T["hand_written"]["film_pair"]


def test_only_the_development_keys_and_replays_placeholder_differ_from_the_registered_version():
    import copy

    frozen = json.loads((Path(__file__).parent / "frozen" / "thresholds_registered.json").read_text())
    now = copy.deepcopy(T)
    for obj in (frozen, now):
        obj["development"].pop("pooled_lives")
        obj["donor_rule"].pop("dev_lives")
        obj.pop("hand_written")
    now.pop("replay_token_penalty")  # Task 17b's λ, added after registration as a dated placeholder
    now["placeholders"].pop("replay_token_penalty")
    assert now == frozen  # the final-test rule cannot drift
    assert T["development"]["pooled_lives"] == [
        "founder", "founder_donor", "researcher", "researcher_donor"]
    assert "(personas-v2)" in T["donor_rule"]["dev_lives"]
