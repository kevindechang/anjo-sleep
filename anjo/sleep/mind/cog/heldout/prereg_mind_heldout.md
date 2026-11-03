# MIND cognition sprint: held-out pre-registration

Written 2026-11-02 Taipei, before any held-out seed exists. This file's sha256 goes into the prereg
ledger (kind `mind_prereg`) together with the tool manifest, and both are pushed before the draw.
After that nothing here changes. Each held-out run happens once and is scored once. The dev numbers
quoted are in `frozen_summary.md`, beside the run records.

## 1. What is frozen
- **Code:** branch `cog/core` at `3451998c4b8f56c9f0eb53a1ad827f1073d7fee1`, local tag `mind-freeze`.
  Every held-out command runs from a clean worktree at that literal sha; the scripts refuse any other
  checkout or a dirty tree.
  - `3451998c` is `6d6d277a`, the code candidate B was calibrated and scored with on dev, plus
    `d6d02d1d` from `hackathon/mind-cog`. That commit changes only the "optimal" reference that the
    AURC report prints, not AURC or Spearman.
  - The main and independent world generators and the Gemini predictor are byte-identical to the
    code that made the dev Gemini records.
- **Tools:** the scripts in `heldout_tools/` are listed with their sha256 in `MANIFEST.sha256`, whose
  sha256 is registered (kind `mind_tools`). The run and score scripts refuse to start if any tool
  differs from the manifest or the manifest is not the registered one.
- **MIND (candidate B):** `cog_core_final.pt`, sha256
  `4009ebe1db7e8088c5e7cc67c98809900b648968777b4ee10481b1f327566a00`.
  - Platt a=1.041637, b=-0.077124.
  - Fallback prior: count 0.8827, hidden 0.5111.
- **Frozen MIND is `mind_nonight`:** night off, no asks for T1, T2 and T4. The gated night was
  harmful on dev (P5).
- **Protocol-2 checkpoint** (MIND trained on the independent world): `cog_core_platt.pt`, sha256
  `d06ff4ff2b95c83b7bc665d00b105b7f5a09b02d782aa2963627228b0cc2963f`. It is secondary only.
- **Gemini Platt maps,** fitted once on all dev records (`fit_platt`, l2 1e-4) and frozen here:

  | World | Predictor | a | b | Dev records |
  |---|---|---|---|---|
  | main | gemini | 0.577884 | 0.375832 | 60 lives, 5,070 queries |
  | main | gemini_stats | 0.598332 | 0.328086 | 60 lives, 5,070 queries |
  | indep | gemini | 0.694137 | -0.072288 | 60 lives, 5,265 queries |
  | indep | gemini_stats | 0.661167 | -0.134386 | 60 lives, 5,265 queries |

  The dev comparison used cross-validated maps. Held-out records get these full-dev maps, applied
  to the raw run's p after the run (no new calls).
- **The best Gemini on dev** is gemini_stats with Platt (0.5277; gemini with Platt was 0.5280).
  P2 compares MIND against it.
- **Settings:**
  - n_days 84; `--max-asks 1`.
  - T3 window 7. mind_voi uses lam 1.0 and week 7; both were fixed before any T3 dev result.
  - Report defaults: margin 0.05, eps 1e-4, 15 ECE bins, bootstrap seed 0.
- **Not run:** `mind_uniform`. It has no endpoint and no dev result for candidate B.

## 2. The draw
- **Inputs.**
  - `C` is the prereg-repo commit sha that the push of this file's entry returns.
  - `D` is that commit's committer date, from the GitHub API.
  - The beacon is drand's default chain (hash `8990e7a9…b2ce`, 30-s period). `R` is its first
    round whose time is at least 10 minutes after `D`.
  - Round R's randomness does not exist when the entry is pushed. Neither the push time nor any
    later choice can steer it, and anyone can fetch and verify it from any drand relay. If no
    relay serves round R, the draw waits; R does not change.
- **Seeds.** For each world `w` in (`main`, `indep`), seed `i` is
  `100000 + int(sha256(f"{randomness_R}|{C}|mind-cog-heldout|{w}|{i}").hexdigest(), 16) % 900000`
  for i = 0, 1, 2, …, keeping the first 200 distinct values in draw order (`mind_draw.py`). No
  held-out seed can overlap a seed used before: training 0–9999, dev 10000–10199, OOD 20000–20199.
- **Subsets.** The Gemini subset of each world is its first 60 seeds in draw order.
- **Registration.** The seeds file records C, D, R, the randomness and the beacon's signature with
  the lists. Its sha256 is registered (kind `mind_seeds`) and pushed before any held-out run starts.
  The run and score scripts refuse a seeds file that is not the registered one.

## 3. Runs (one each; `--workers` may vary, results do not depend on it)
- **Main world, 200 lives, every day, no asks:**
  - online_lr, population, counter, surprise_lr
  - mind_nonight (= MIND), mind, mind_nofast, mind_k1, all with checkpoint B
- **Main world, 200 lives, T3:** mind_nonight with inquirer mind_voi (lam 1.0), max_uncertainty and
  random, all with checkpoint B.
- **Main world, Gemini subset (60), `--query-every 7`:** mind_nonight (B), online_lr, gemini,
  gemini_stats. The Gemini runs are paid (vertex_mind_usd), so they start only after Kevin approves
  the spend.
- **Independent world, 200 lives, every day:** online_lr, population, counter, surprise_lr, MIND
  zero-shot (mind_nonight, checkpoint B), and protocol 2 (mind_nonight, the protocol-2 checkpoint).
- **Independent world, Gemini subset (60), `--query-every 7`:** protocol 2, online_lr, gemini,
  gemini_stats (paid).
- **Done.** A run is done when its eval exits 0 and `validate.py` accepts its records:
  - the expected run names;
  - exactly the registered seeds, once each, 84 days.
  A done run is never rerun.
- **Failure rule.** Each run gets at most two attempts, and the run script counts them.
  - The second attempt is the one resume allowed. It is for an infrastructure failure only
    (process killed, machine slept, provider outage).
  - It needs a written reason, which the script registers in the ledger (kind
    `mind_heldout_resume`) before starting. It uses the same command and seeds; the eval
    rewrites the run's records from scratch.
  - A run that fails twice, or fails for any other reason, is reported as failed.
  - A Gemini query whose reply is missing or unparsable gets the tool's 0.5 fallback, as on dev.
    The count is reported.
- **The score.** The score is `SCORES.json`, written last and atomically by
  `mind_heldout_score.sh`, which refuses to run again once it exists.
  - Before scoring, `validate.py` checks every file. The runs of each pairing group must have
    identical queries and outcomes, and each Platt file must be exactly the frozen map of its raw
    run.
  - The per-run `report.json` that each eval writes is a byproduct. It is not opened before
    scoring, and no decision uses it.
  - A scoring attempt that fails before writing SCORES.json may be repeated.

## 4. Primary endpoints
All endpoints are paired over lives, with a 95% percentile bootstrap CI over lives: 10,000
resamples, except P3 and the k1 leg of P6, which use 1,000 (`t2boot.py`). `t2boot.py` uses the
report's own AURC and Spearman with the same tie handling, the report's loss (clip 1e-4) and
unrounded CI bounds; the dev P3 and k1 figures below were recomputed with it and agree with the
earlier dev tool to 4 decimals. Bars compare unrounded bounds with 0. Each endpoint passes or
fails on its own bar. There is no multiplicity correction. P1 and P2 carry the headline claim. A
failed bar is reported as a failure.

| # | Test | Comparison | Bar (passes if) | Dev |
|---|---|---|---|---|
| P1 | T1 | MIND − online_lr log loss, main, 200 lives | upper CI < 0 | −0.0035 [−0.0049, −0.0020] |
| P2 | T1 vs the LLM | MIND − gemini_stats + frozen Platt log loss, main Gemini subset, qe7 | upper CI < 0 | −0.0520 [−0.0634, −0.0412] (CV map) |
| P3 | T2 | MIND − online_lr AURC(u, loss) **and** Spearman(u, excess over the oracle) | AURC upper CI < 0 **and** Spearman lower CI > 0 | −0.0765 [−0.0878, −0.0645]; +0.0585 [+0.0399, +0.0785] |
| P4 | T3 | (a) mind_voi utility per life; (b) mind_voi − max_uncertainty utility; (c) mind_voi's pooled distractor share | (a) lower CI > 0; (b) lower CI > 0; (c) ≤ 0.10. P4 passes only if all three do | (a) +0.36 [−0.24, +0.96]; (b) +11.44 [+10.74, +12.12]; (c) 0.040 |
| P5 | T4, the night | mind − mind_nonight: recovery days, log loss, gap14 | No bar. Registered as "no benefit"; the numbers are reported | +0.33 d [+0.02, +0.71]; +0.0038; +0.0025 |
| P6 | T5, lesions (vs `mind`, night on, as measured on dev) | nofast: T1 log loss and T4 recovery days. k1: T2 AURC. max_uncertainty: distractor share vs mind_voi's | nofast: both lower CIs > 0. k1: lower CI > 0. max_uncertainty: its pooled share > mind_voi's (point estimates; the tool gives no CI for shares) | nofast +0.0050 [+0.0037, +0.0064], +0.74 d [+0.28, +1.25]; k1 +0.1071 [+0.0965, +0.1166]; 1.000 vs 0.040 |

- **k1's T2 convention.** k1 gives no u, so its order is random and its AURC is its mean loss.
  For each bootstrap resample, k1's AURC is the pooled mean loss of the resampled lives, and
  mind's is `t2boot.py`'s AURC on the same lives.
- **Expectations.** The leg I expect to fail is P4(a). P1's dev effect is inflated by selection
  (caveats 1–2 below); my point expectation is about −0.0025, and P1 still passes at that size if
  the dev CI width holds.

## 5. Secondary (reported whatever they show; no bars)
- Every run's log loss, Brier, ECE and per-family numbers; the gap to the oracle; the learning
  curve by week.
- MIND zero-shot on the independent world against online_lr. Dev was a negative: +0.2068, worse
  than a constant 0.5.
- Protocol 2 against online_lr, every day and qe7, and against both Gemini predictors with their
  frozen maps, on the independent Gemini subset.
- P2 against each other Gemini variant (raw, and gemini with Platt).
- T4 for MIND against online_lr: recovery days and gap14.

## 6. Selection caveats (dev is not a clean estimate; repeated in the write-up)
1. B's epoch (8 of 12) was picked by no-ask dev log loss on the same 200 dev lives.
2. B replaced candidate A on a paired dev win of −0.0010 [−0.0019, −0.0001].
3. B's Platt map is in-sample: the core trained on seeds 0–9999, and the map was fitted on
   7000–7199, which lie inside that range.
4. Dev seeds 10000–10199 were reused across many experiments: A vs B, the night variants, the
   hazard gate and the calibration checks.
5. Several choices were made on dev:
   - Night off was decided on dev and on train seeds 7000–7039.
   - The best Gemini variant was picked on 60 dev lives.
   - The frozen Gemini maps were fitted on those same dev records.
6. The P6 lesions are measured with the night on (the registered predictors), while frozen MIND
   has it off.
7. For T3, the first 100 dev lives were seen with candidate A before the 200-life run.

## 7. Order and cost
1. Register this file (kind `mind_prereg`) and `MANIFEST.sha256` (kind `mind_tools`), then push.
   The push returns C.
2. Wait for round R. Draw, register the seeds file (kind `mind_seeds`), push.
3. Run the local runs.
4. Ask Kevin, then run the four Gemini runs. Each is capped at $6 on vertex_mind_usd; dev cost
   $3.1–3.7 each, about $14 in all.
5. Score once and register the score file's sha256.
