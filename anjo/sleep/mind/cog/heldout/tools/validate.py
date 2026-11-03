"""Validate held-out records before a run counts as done and before scoring (prereg §3-4).
usage: validate.py SEEDS.json H run LABEL   one run: its files exist, carry the expected run names,
                                            cover exactly the registered seeds, once each, 84 days
       validate.py SEEDS.json H all         every run, plus: identical queries and outcomes across the
                                            runs of each pairing group, and each Platt file is the
                                            frozen map applied to its raw run"""
import json
import math
import sys
from pathlib import Path

from anjo.sleep.mind.cog.gemini import apply_platt

N_DAYS = 84
PLATT = {"q_gemini_platt": ("q_gemini", 0.577884, 0.375832), "q_gemini_stats_platt": ("q_gemini_stats", 0.598332, 0.328086),
         "iq_gemini_platt": ("iq_gemini", 0.694137, -0.072288), "iq_gemini_stats_platt": ("iq_gemini_stats", 0.661167, -0.134386)}
GROUPS = {  # group -> (seeds key, {label: [(relative file, run name)]})
    "m": ("main", {**{f"m_{p}": [(f"m_{p}/m_{p}.jsonl", f"m_{p}")] for p in
                      ("online_lr", "population", "counter", "surprise_lr", "mind_nonight", "mind", "mind_nofast", "mind_k1")},
                   **{f"t3_{i}": [(f"t3_{i}/mind_nonight.jsonl", "mind_nonight"), (f"t3_{i}/mind_nonight+{i}.jsonl", f"mind_nonight+{i}")]
                      for i in ("mind_voi", "max_uncertainty", "random")}}),
    "q": ("main_gemini", {**{f"q_{p}": [(f"q_{p}/q_{p}.jsonl", f"q_{p}")] for p in ("mind_nonight", "online_lr", "gemini", "gemini_stats")},
                          "q_gemini_platt": [("q_gemini/q_gemini_platt.jsonl", "q_gemini_platt")],
                          "q_gemini_stats_platt": [("q_gemini_stats/q_gemini_stats_platt.jsonl", "q_gemini_stats_platt")]}),
    "i": ("indep", {f"i_{p}": [(f"i_{p}/i_{p}.jsonl", f"i_{p}")] for p in
                    ("online_lr", "population", "counter", "surprise_lr", "zeroshot", "p2")}),
    "iq": ("indep_gemini", {**{f"iq_{p}": [(f"iq_{p}/iq_{p}.jsonl", f"iq_{p}")] for p in ("p2", "online_lr", "gemini", "gemini_stats")},
                            "iq_gemini_platt": [("iq_gemini/iq_gemini_platt.jsonl", "iq_gemini_platt")],
                            "iq_gemini_stats_platt": [("iq_gemini_stats/iq_gemini_stats_platt.jsonl", "iq_gemini_stats_platt")]}),
}


def fail(msg: str) -> None:
    raise SystemExit(f"INVALID: {msg}")


def read(path: Path, run: str, seeds: list[int]) -> dict[int, dict]:
    if not path.is_file():
        fail(f"{path} missing")
    out: dict[int, dict] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        r = json.loads(line)
        if r["run"] != run:
            fail(f"{path}:{n} run {r['run']!r}, expected {run!r}")
        if r["seed"] in out:
            fail(f"{path}:{n} seed {r['seed']} twice")
        if r["n_days"] != N_DAYS or not r["queries"]:
            fail(f"{path}:{n} n_days {r['n_days']} or no queries")
        out[r["seed"]] = r
    if set(out) != set(seeds):
        fail(f"{path}: seeds differ from the registered set ({len(set(out) ^ set(seeds))} differ)")
    return out


def key(r: dict) -> tuple:
    return tuple((q["qid"], q["asked_day"], q["target_day"], q["outcome"], q["oracle_p"]) for q in r["queries"])


def main() -> None:
    seeds, h, mode = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3]
    where = {label: (g, files) for g, (_, runs) in GROUPS.items() for label, files in runs.items()}
    labels = [sys.argv[4]] if mode == "run" else list(where)
    if mode not in ("run", "all") or any(label not in where for label in labels):
        fail(f"bad arguments {sys.argv[3:]}")
    loaded: dict[str, dict[int, dict]] = {}
    for label in labels:
        g, files = where[label]
        for rel, run in files:
            loaded[rel] = read(h / rel, run, seeds[GROUPS[g][0]])
    if mode == "all":
        for g, (skey, runs) in GROUPS.items():
            rels = [rel for files in runs.values() for rel, _ in files]
            ref = loaded[rels[0]]
            for rel in rels[1:]:
                for s in seeds[skey]:
                    if key(loaded[rel][s]) != key(ref[s]):
                        fail(f"group {g}: {rel} seed {s} has other queries or outcomes than {rels[0]}")
        for label, (raw, a, b) in PLATT.items():
            cal, src = loaded[where[label][1][0][0]], loaded[where[raw][1][0][0]]
            for s, r in cal.items():
                for qc, qr in zip(r["queries"], src[s]["queries"], strict=True):
                    if not math.isclose(qc["p"], apply_platt(qr["p"], (a, b)), rel_tol=0, abs_tol=1e-12):
                        fail(f"{label} seed {s} {qc['qid']}: p is not the frozen map of {raw}'s p")
    print(f"VALID {mode} {' '.join(labels) if mode == 'run' else f'{len(loaded)} files'}")


if __name__ == "__main__":
    main()
