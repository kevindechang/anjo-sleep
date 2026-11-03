# MIND cog core: frozen dev numbers for the held-out preregistration (candidate B)

Written 2026-11-02 13:40 Taipei by the cog/core worker. All paths are under
`R = /private/tmp/claude-501/-Users-k/f8dbbf52-df21-4aab-acb6-f342415fbeb3/scratchpad/mind_research/runs`.
How to read the numbers:
- Each difference is paired over lives, with a 95% percentile bootstrap CI over lives. There are 10,000 resamples except P3 (1,000, see below).
- Dev is main world 10000-10199 (200 lives, 123,579 queries) unless a row says otherwise.
- Negative log-loss differences mean MIND is better.
Every command is in `scratchpad/cog/real_eval.sh`.

## What is frozen
- Code: branch cog/core at 6d6d277a. Candidate B was trained at e59fa635 and calibrated at 6d6d277a.
- Checkpoint: `R/snap_tally10k_e8/cog_core_final.pt`, sha256 `4009ebe1db7e8088c5e7cc67c98809900b648968777b4ee10481b1f327566a00`. It is also copied to `R/cog_core/cog_core.pt`, the CLI default.
  - Weights: 10k-life run, epoch 8.
  - Platt a=1.041637, b=-0.077124.
  - Fallback prior: count 0.8827, hidden 0.5111.
- MIND as frozen is **`mind_nonight`**: night off, no asks for T1/T2/T4. The gated night costs log loss on dev (P5).
- T3 uses `mind_nonight` + `mind_voi`, lam 1.0, week 7.
  - Both were fixed before any T3 dev result.
  - week 7 was chosen on train seeds 7000-7049.

## P1 (T1): mind - online_lr, main world
- **-0.0035 [-0.0049, -0.0020]**: 0.4916 vs 0.4950. This closes 11% of online_lr's 0.031 gap to the oracle (0.4640).
- Brier: -0.0014 [-0.0019, -0.0007].
- By family: count -0.0006 [-0.0012, -0.0000]; hidden -0.0061 [-0.0087, -0.0035].
- ECE 0.0050 vs 0.0068. ECE by family: count 0.0021, hidden 0.0098.
- Week 0 is a tie (+0.0003). The gain comes from week 1 on (-0.005).
- Records:
  - MIND: `R/eval_10k_e8p_nonight/tally10k_e8p.jsonl`
  - online_lr: `R/eval_online_lr/online_lr.jsonl`
  - Paired: `R/frozen_P1_main.json`
- With the gated night (`mind`): +0.0003 [-0.0014, +0.0021]. Records: `R/evalB_mind/B_mind.jsonl`.

## P2 (T1 vs the LLM): main dev 10000-10059, --query-every 7 (5,070 queries)
Log loss and ECE per run:

| Run | Log loss | ECE |
|---|---|---|
| MIND (B) | 0.4757 | 0.0166 |
| online_lr | 0.4832 | 0.0151 |
| gemini raw | 0.5628 | 0.0686 |
| gemini + Platt-CV | 0.5280 | 0.0415 |
| gemini_stats raw | 0.5568 | 0.0592 |
| **gemini_stats + Platt-CV (best on dev)** | **0.5277** | 0.0373 |

- Gemini+Platt-CV and gemini_stats+Platt-CV tie (0.5280 vs 0.5277).
- **mind - best Gemini (gemini_stats + Platt-CV): -0.0520 [-0.0634, -0.0412]**. count -0.0387, hidden -0.0651.
- mind - online_lr on this subset: -0.0075 [-0.0117, -0.0032].
- Records:
  - MIND: `R/qe7_candB_main/candB_qe7.jsonl`
  - online_lr: `R/qe7_online_lr_main/online_lr_qe7.jsonl`
  - Gemini: `R/gem_dev60/gem_dev60{,_plattcv}.jsonl` and `R/gemstats_dev60/gemstats_dev60{,_plattcv}.jsonl`
  - Paired: `R/frozen_P2_main_qe7.json`
- MIND's p on weekly days equals its every-day p (max |dp| 0, checked on indep), because its state reads events only.

## P3 (T2): mind vs online_lr
- AURC: 0.3932 vs 0.4696, **diff -0.0765 [-0.0878, -0.0645]**.
- Spearman(u, excess): 0.1677 vs 0.1092, **diff +0.0585 [+0.0399, +0.0785]**.
- Point estimates come from `score.aurc` and `score.spearman`. The CIs come from 1,000 resamples over lives that rank without tie averaging (`scratchpad/cog/p3boot.py`). The report command gives no paired CI for these.
- Output: `R/p3_B_vs_online_lr.json`.

## P4 (T3): mind_nonight + inquirer, 200 dev lives, window 7 (`R/frozen_P4_t3.json`)

| Inquirer | Asks/life | Utility/life [CI] | Loss reduction | Cost | Distractor share |
|---|---|---|---|---|---|
| mind_voi | 29.1 | **+0.36 [-0.24, +0.96]** (null) | 3.30 | 2.94 | **0.040** |
| max_uncertainty | 84.0 | -11.08 [-11.53, -10.63] | -0.53 | 10.55 | 1.000 |
| random | 24.6 | -2.36 [-2.70, -2.01] | 0.73 | 3.09 | 0.474 |

- **voi - max_uncertainty: +11.44 [+10.74, +12.12]**. voi - random: +2.72 [+2.20, +3.21].
- Suggested BAR_DISTRACTOR: 0.10. That is 2.5x mind_voi's dev share and well under random's 0.47.
- Records: `R/t3B_{mind_voi,max_uncertainty,random}/*.jsonl`.
- Expect the "utility > 0" leg of P4 to fail on held-out at this power. Candidate A, at 200 lives, was also null: +0.31 [-0.16, +0.78].

## P5 (T4, the night) and T4
- mind (gated surprise night) - mind_nonight:
  - log loss +0.0038 [+0.0029, +0.0047]
  - **recovery days +0.33 [+0.02, +0.71]** (3.19 vs 2.93)
  - gap14 +0.0025 [+0.0002, +0.0048]
- Dev shows the night is **harmful**, not merely null. Recommendation: register P5 as "night reported as no benefit", not as a held-out claim.
- Paired file: `R/frozen_P5_night.json`.
- T4 for frozen MIND vs online_lr:
  - recovery 2.93 vs 2.95 days, **-0.18 [-0.64, +0.28]** (tie)
  - recovered share 0.983 vs 0.966
  - gap14 -0.0050 [-0.0089, -0.0011]
  - 140 lives have shifts.
- The transfer night (`mind_transfer`) is a reported negative: +0.145 nats on train seeds 7000-7039 with candidate A. The fast memory relearns what was moved.

## P6 (lesions, registered predictors, so the gated night is on), vs `mind`
- **nofast**:
  - T1 +0.0050 [+0.0037, +0.0064]
  - T4 recovery +0.74 days [+0.28, +1.25]
  - gap14 +0.0078 [+0.0044, +0.0114]
  - Both legs pass on dev.
- **k1**:
  - T1 +0.0111 [+0.0088, +0.0135], ECE 0.050.
  - T2 is undefined: k1 has u = None, so AURC and Spearman can't be computed.
  - The prereg must fix a convention. Proposed: no u means random selection, so AURC = mean loss: 0.5065 vs 0.3994 (+0.107, no CI).
  - T4 +0.05 [-0.72, +0.86].
- max_uncertainty -> distractor share 1.000 vs mind_voi's 0.040 (P4 table).
- Records: `R/evalB_{mind,mind_nofast,mind_k1}/*.jsonl`. Paired: `R/frozen_P6_lesions.json`.

## Secondary (no bars)
- **Zero-shot indep**, B with the fallback, scored once, not tuned:
  - 0.7208 vs online_lr 0.5140: +0.2068 [+0.1927, +0.2213], ECE 0.223.
  - It is worse than a constant 0.5, because the main world's base-rate prior is wrong for another world.
  - Records: `R/zeroshot_indep_candB/zeroshot_candB.jsonl`.
- **Protocol 2** (MIND trained on indep train seeds, report only): see the last section.
- Other main-world numbers:
  - Candidate A, OOD population (20000-20199, no Platt): mind_nonight +0.0064 [+0.0042, +0.0086] vs online_lr.
  - Hazard gate (candidate A's recipe + `--hazard`): +0.0008 [+0.0002, +0.0012] vs A, T4 tie. Training drove its weights to about 0.

## Selection caveats (dev is not a clean estimate)
1. B's epoch (8 of 12) was picked by no-ask dev log loss on these same 200 lives.
2. B replaced A on a paired dev win of only -0.0010 [-0.0019, -0.0001]. The run's final epoch 11 then tied B (-0.0002 [-0.0009, +0.0006]) and was not taken. Expect held-out P1 nearer -0.0025 than -0.0035.
3. B's Platt map is in-sample: the 10k core trained on all of seeds 0-9999, including the 7000-7199 used for the fit. A's map (0.978, 0.106) was out-of-sample.
4. Dev seeds 10000-10199 were reused across many experiments: A vs B, the night variants, the hazard gate and the calibration checks. That is a garden of forking paths for T1 and T4.
5. Night off was decided on dev (and on train seeds 7000-7039). The best Gemini variant was picked on dev (60 lives); its Platt is cross-validated on the dev records.
6. P6 lesions were measured with the night on (the registered predictors), while frozen MIND is night off.
7. T3: lam and week were fixed before any T3 dev result. The first 100 dev lives were seen (on candidate A) before the 200-life run.

## Protocol 2, final (report only): MIND trained on the independent world
- Checkpoint: `R/snap_indep_final/cog_core_platt.pt`, sha256 `d06ff4ff2b95c83b7bc665d00b105b7f5a09b02d782aa2963627228b0cc2963f`.
  - Weights: epoch 9 of 12, best indep no-ask dev 0.48164. It trained on indep seeds 0-9999 with early stopping on indep dev 10000-10199.
  - Platt a=1.0298, b=-0.0276, fitted on indep 7000-7199, which is **in-sample**.
- Indep dev 10000-10199, 132,233 queries, vs indep online_lr:
  - **-0.0327 [-0.0363, -0.0291]**: 0.4813 vs 0.5140. The oracle is 0.4483, so this closes 50% of the 0.066 gap.
  - count -0.0195 [-0.0236, -0.0157]; hidden -0.0393 [-0.0442, -0.0347].
  - ECE 0.0034 vs 0.0118.
  - AURC 0.3045 vs 0.5476: -0.243 [-0.256, -0.230]. Spearman(u, excess) 0.180 vs -0.052: +0.233 [+0.210, +0.255].
  - T4 recovery 2.11 vs 5.16 days: -3.42 [-4.65, -2.34].
  - Records: `R/indep_final_nonight/indep_final.jsonl`. Paired: `R/frozen_protocol2_indep.json` and `R/p3_indep_final_vs_online_lr.json`.
- Indep --query-every 7, all 200 lives (17,593 queries):
  - vs the coordinator's `R/rivals_indep_q7/online_lr_indep_q7.jsonl` (0.5118): MIND 0.4810, **-0.0308 [-0.0358, -0.0259]**.
  - Records: `R/qe7_indep_final/indep_final_qe7.jsonl`. Paired: `R/frozen_protocol2_indep_qe7_200.json`.
- The same, on lives 10000-10059 (5,265 queries):

| Run | Log loss |
|---|---|
| MIND | 0.4803 |
| online_lr | 0.5189 |
| gemini raw | 0.5327 |
| gemini + Platt-CV | 0.5180 |
| gemini_stats raw | 0.5367 |
| gemini_stats + Platt-CV (best) | 0.5171 |

  - **MIND - best Gemini: -0.0368 [-0.0438, -0.0298]**.
  - Records: `R/qe7_indep_final/indep_final_qe7_60.jsonl`, the 60-life subset of the 200-life run with the run renamed. Paired: `R/frozen_protocol2_indep_qe7_60.json`.
- The provisional numbers it replaces: e1 -0.0272, e4 -0.0306.
