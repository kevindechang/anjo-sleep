"""The registered bars (prereg §4) applied to the held-out score files. usage: endpoints.py SCORE_DIR"""
import json
import sys
from pathlib import Path

O = Path(sys.argv[1])
load = lambda name: json.loads((O / name).read_text())  # noqa: E731


def comp(report: dict, a: str, b: str) -> dict:
    hits = [c for c in report["comparisons"] if c["a"] == a and c["b"] == b]
    if len(hits) != 1:
        raise SystemExit(f"no single comparison {a} vs {b}")
    return hits[0]


def t3_comp(report: dict, a: str, b: str) -> dict:
    hits = [c for c in report["t3_comparisons"] if c["a"] == a and c["b"] == b]
    if len(hits) != 1:
        raise SystemExit(f"no single T3 comparison {a} vs {b}")
    return hits[0]


p1 = comp(load("P1_main.json"), "m_mind_nonight", "m_online_lr")["log_loss_diff"]
p2 = comp(load("P2_qe7.json"), "q_mind_nonight", "q_gemini_stats_platt")["log_loss_diff"]
p3 = load("P3.json")
t3 = load("P4_t3.json")
voi, maxu = t3["runs"]["mind_nonight+mind_voi"]["t3"], t3["runs"]["mind_nonight+max_uncertainty"]["t3"]
voi_vs_maxu = t3_comp(t3, "mind_nonight+mind_voi", "mind_nonight+max_uncertainty")["utility_diff"]
p5 = comp(load("P5_night.json"), "m_mind", "m_mind_nonight")
nofast = comp(load("P6_lesions.json"), "m_mind_nofast", "m_mind")
k1 = load("P6_k1.json")
p4 = {"a_utility_per_life": {"mean": voi["utility_per_life"], "ci": voi["utility_ci"], "pass": voi["utility_ci"][0] > 0},
      "b_voi_minus_max_uncertainty": {**voi_vs_maxu, "pass": voi_vs_maxu["ci"][0] > 0},
      "c_distractor_share": {"value": voi["distractor_share"], "bar": 0.10, "pass": voi["distractor_share"] <= 0.10}}
p4["pass"] = all(v["pass"] for v in p4.values())
p6 = {"nofast_T1": {**nofast["log_loss_diff"], "pass": nofast["log_loss_diff"]["ci"][0] > 0},
      "nofast_T4_recovery_days": {**nofast["t4"]["recovery_days_diff"], "pass": nofast["t4"]["recovery_days_diff"]["ci"][0] > 0},
      "k1_T2_aurc": {**k1, "pass": k1["ci"][0] > 0},
      "max_uncertainty_distractor_share": {"max_uncertainty": maxu["distractor_share"], "mind_voi": voi["distractor_share"],
                                           "pass": maxu["distractor_share"] > voi["distractor_share"]}}
gem_usage = {name: run.get("usage") for name, run in load("P2_qe7.json")["runs"].items() if "gemini" in name}
gem_usage.update({name: run.get("usage") for name, run in load("S_indep_qe7.json")["runs"].items() if "gemini" in name})
out = {
    "P1": {**p1, "pass": p1["ci"][1] < 0},
    "P2": {**p2, "pass": p2["ci"][1] < 0},
    "P3": {"aurc": p3["aurc"], "spearman_u_excess": p3["spearman_u_excess"],
           "pass": p3["aurc"]["ci"][1] < 0 and p3["spearman_u_excess"]["ci"][0] > 0},
    "P4": p4,
    "P5_no_bar": {"log_loss_diff": p5["log_loss_diff"], "t4": p5.get("t4")},
    "P6": {**p6, "pass": all(v["pass"] for v in p6.values())},
    "secondary": {
        "indep_zeroshot_vs_online_lr": comp(load("S_indep.json"), "i_zeroshot", "i_online_lr")["log_loss_diff"],
        "indep_protocol2_vs_online_lr": comp(load("S_indep.json"), "i_p2", "i_online_lr")["log_loss_diff"],
        "indep_protocol2_P3": load("S_indep_P3.json"),
        "indep_qe7_protocol2_vs_online_lr": comp(load("S_indep_qe7.json"), "iq_p2", "iq_online_lr")["log_loss_diff"],
        "main_qe7_gemini_minus_mind": {a: comp(load("S_qe7_vs_mind.json"), a, "q_mind_nonight")["log_loss_diff"]
                                       for a in ("q_gemini", "q_gemini_stats", "q_gemini_platt", "q_gemini_stats_platt")},
        "indep_qe7_gemini_minus_protocol2": {a: comp(load("S_indep_qe7_vs_p2.json"), a, "iq_p2")["log_loss_diff"]
                                             for a in ("iq_gemini", "iq_gemini_stats", "iq_gemini_platt", "iq_gemini_stats_platt")},
        "gemini_usage": gem_usage,
    },
}
print(json.dumps(out, indent=1, sort_keys=True))
