# Results: every run, October 27 to November 3, 2026

<!-- COPY-PENDING(kevin): first draft. -->

This page lists every experiment we ran during the hackathon: the two registered tests, the
development runs that chose the recipe, and the runs that produced no result, each with its cause.
Every person in these runs is simulated. No real person's data is in any number here.

- **Registered** means the probes, prompts, thresholds, judge and scoring code were hashed into the
  pre-registration ledger (the `prereg` branch of this repo, `ledger.jsonl`, entries chained by
  hash) before the run, and the run was scored once. "Seq" numbers below point into that ledger.
- **Development** means a run on development lives that we used to choose or tune the recipe.
  These are not claims.
- **Score** is the share of blind pairwise comparisons won, ties counted as half. The judge sees
  both answers in both orders. 0.5 means no difference.

Arms, as in the night log: **1** base Muse Glimmer, no notes. **2C** base Glimmer with the night's
concise notes. **2F** base Glimmer with the full evidence. **3** the person's own trained adapter,
no notes. **3x** another person's adapter, no notes. **4** own adapter plus concise notes.

## 1. Registered: the Sleep final (seq 54 to 94)

**Setup.** The code froze at `1b27bce0` (seq 83). Three people were drawn from a list of 40 by a
seed committed before the freeze and revealed after it (seq 74, 75): P03a, P03b and P07a. Each ran
two accelerated nights, and one full morning of probes scored each. The judge was DeepSeek V4.1
Flash on Fireworks. Scored once (seq 91 to 94).

**Registered claim rows.** A comparison wins when every condition holds: the 95% interval's lower
bound is above 0.5, the score is at least 0.55, at least 58% of the decisive comparisons are wins,
and at least 2 of the 3 people are ahead.

| Dimension | Comparison | Score | 95% CI | People ahead | Registered outcome |
|---|---|---|---|---|---|
| Knowledge | 3 v 1 (own adapter vs base) | 0.571 | 0.500 to 0.637 | 2 of 3 | inconclusive |
| Knowledge | 3 v 3x (own adapter vs another's) | 0.779 | 0.713 to 0.842 | 3 of 3 | **win** |
| Interaction | 3 v 1 | 0.517 | 0.450 to 0.578 | 1 of 3 | inconclusive |
| Interaction | 3 v 3x | 0.600 | 0.533 to 0.661 | 3 of 3 | **win** |

Knowledge 3 v 1 met three of the four conditions. Its interval's lower bound was exactly 0.500, not
above it.

Per person (knowledge / interaction):

| Person | 3 v 1 | 3 v 3x | Adapter scored |
|---|---|---|---|
| P07a | 0.738 / 0.617 | 0.662 / 0.567 | night 2 |
| P03a | 0.537 / 0.483 | 0.775 / 0.567 | night 2 |
| P03b | 0.438 / 0.450 | 0.900 / 0.667 | night 1: the gate didn't promote its night-2 adapter |

**Measured in the same mornings, not claim rows.**

| Comparison | Knowledge | Interaction | Note |
|---|---|---|---|
| 2C v 1 (notes vs base) | 0.762 | 0.661 | mean of the three people, all three ahead on knowledge |
| 2F v 1 (full evidence vs base) | 0.771 | 0.656 | mean of the three people |
| 4 v 2C (adapter + notes vs notes), new_work | 0.320 (0.233 to 0.413) | | 75 probes: adding the adapter made noted answers worse |
| 4 v 2F, new_work | 0.333 (0.247 to 0.420) | | 75 probes |

**Cost.** Thirty times each person's mean logged night is $41 to $60 a person a month, training
every night. Our ceiling was $10, so this configuration isn't viable yet.

**Deviations, each approved before it ran and registered.**
- Seq 89: the frozen budget guard reserved about $1.26 per judge call (real cost about $0.005), so
  the first morning stopped with BudgetStop after 736 calls. A patch changed only that reservation
  and the arms hold. The scoring code didn't change, and the run's stamp still reads the freeze sha.
  The diff and its sha256 are in the ledger entry.
- Seq 90: two mornings failed with ProxyFailed (Modal's proxy answered 408 before vLLM saw the
  request) and were rerun once; the rerun was registered first. No score had been written for either.

**Other incidents in the final's run.**
- Another session's command (`pkill -f` with BSD argument parsing) killed the run during night 1,
  and the resume retrained night 1 for all three people.
- Colab failed to upload or install for every night-2 training (4 attempts). Each fell back to
  Modal, as the runner is built to do.
- P03a's night 2 failed once with OSError because this Mac's disk was full, and was retrained.

The claim text in `final.json` says "a week of nightly training". Each person ran two nights.

## 2. Registered: MIND held-out (seq 84 to 88)

MIND is a small model we trained from scratch to give odds on what a person will do. It's research:
the app's predictions come from the night agent, not from MIND. Everything for this test is in
[`mind/cog/heldout/`](mind/cog/heldout/): the pre-registration, the tool manifest, the seeds, every
run's report and the scores. Their sha256 values match ledger seq 84, 85, 86 and 88.

Held-out: 60 simulated people the model never saw, 12 weeks each, drawn after the freeze from a
public randomness round. Log loss, lower is better; "excess" is the loss above the world's own
exact odds.

| Test | Result | 95% CI | Bar |
|---|---|---|---|
| P1: MIND vs online logistic regression, log loss | −0.0028 | −0.0040 to −0.0016 | pass |
| P2: MIND vs the best Gemini setup (stats prompt + Platt), log loss | −0.0526 | −0.0605 to −0.0449 | pass |
| P3: uncertainty ranks its own errors (AURC, MIND vs online LR) | −0.071 | −0.081 to −0.060 | pass |
| P4: asking questions pays off, utility per life | +0.40 | −0.23 to +1.01 | **fail** |
| P5 (no bar): the night step's effect on log loss | +0.0043 | +0.0034 to +0.0053 | harmful |

In excess log loss, MIND's 0.029 against the best Gemini setup's 0.082 is a third of the
avoidable error. On a world written independently of MIND's training world, untrained MIND scored
0.721, worse than a constant 50/50 (0.693). Trained on that world (protocol 2), it beat online
logistic regression by 0.040 (0.036 to 0.045) and beat every Gemini setup we ran there.

## 3. Development runs: Sleep

Every scored night is in the appendix, generated from `results/*/scores/`.

| Dates | Accounts | What it tested | Result | What we did |
|---|---|---|---|---|
| Oct 28 | `film-founder` (Teo) | first full loop: day, night, morning | night 1 promoted; $1.19 for night, homework and morning | Teo's week became the demo |
| Oct 29 | `obj-today`, `obj-fold`, `obj-protect`, `obj-protect-random`, `obj-seed2`, `obj-seed-cap` | training-objective arms and a random-protection control, 3 nights each | no objective differed significantly on 69 knowledge items | found the knowledge scorer moved up to 14.5 points between identical runs (FP8 scaling under 16-way concurrency); scored one request at a time after that |
| Oct 29 | critic A/B, 160 scenes | DeepSeek V4.1 Flash as the practice critic, vs Muse Spark | non-inferior: 0.5375, lower bound 0.484 vs a bar of 0.40; unreadable replies 0 vs 29 | switched the critic to DeepSeek |
| Oct 30 | `obj-founder`, `obj-founder-donor`, `obj-researcher`, `obj-researcher-donor` | development read: 4 people, 2 nights, full mornings, judged by Muse Spark | knowledge 3 v 1 **0.578** (0.509 to 0.647, a pass); interaction 3 v 1 0.521; 3 v 3x 0.559 / 0.525; 4 v 2C knowledge 0.405 | kept the recipe; a DeepSeek rescore was started Nov 1 and not finished |
| Oct 30 to 31 | `obj-mind-film-founder`, `obj-mind-week2-founder` | MIND's notes and inquiry over long accelerated lives | 21 and 6 scored nights (daily probes, appendix) | used to build MIND's inquiry; no claim |
| Nov 1 | `obj-dd-founder-donor` | the final recipe: seeded SFT+DPO objective, extra capacity, day dreams | 3 v 1 knowledge 0.575, interaction 0.633; 2C v 1 0.750 / 0.733; 4 v 2C 0.350 / 0.567 | met the bar we wrote before it ran (knowledge at least 0.55, interaction not below the development read's 0.52), so this recipe was frozen |

## 4. Development runs: MIND

The cognition sprint's development numbers are in
[`mind/cog/heldout/frozen_summary.md`](mind/cog/heldout/frozen_summary.md). Two other tests had
their bars written in our run log before the data existed. They aren't in the hashed ledger.


- **CMS memory levels** (from the HOPE paper) on Glimmer's real latents, 34 lives, 5 seeds: false
  calls per 100 days fell from 1.33 to 0.00 (−1.33, −4.03 to +1.36), and recall fell from 0.320 to
  0.163. Neither change was significant, so CMS stayed off. On toy lives it had cut false calls
  (−0.44, −0.69 to −0.19).
- **MIND's fast memory on vs off**, same latents: recall 0.432 to 0.320 (−0.113, −0.772 to +0.547).
  Not significant.

One more development result:

- **Transfer night** (moving fast memory into the adapter) made log loss worse by 0.082, 0.145 and
  0.180 nats at move strengths 0.2, 0.5 and 1.0. Frozen MIND runs with the night off.

## 5. Runs that produced no result

| Dates | Run | What happened | Cause |
|---|---|---|---|
| Oct 28 | night 1 rerun | trained, then failed at the gate | a serving request failed mid-gate |
| Oct 29 | three `obj-seed` mornings | lost | Modal answers requests past 150 s with a 303 redirect, which our client didn't follow. Fixed. |
| Oct 30 | `obj-mind-film-student` | 19 of 21 nights lost | the serving container shut down and its replacement sat pending for 3 hours |
| Oct 30 to 31 | lever runs 2, 3 and 4 (practice scenes 16 to 20, seq 52) | no read | Fireworks timeouts and 503s, a Colab CLI crash, cold serving. Lever 3 was confounded: one person had 1 night, the other 2. |
| Oct 31 | week-alone continuation | 9 of 13 nights, then 3 nights failed at the brief step | transport errors (ReadError) |
| Oct 31 | `obj-loop-founder` (MIND steers practice) | ran 5 nights; the loop never had a focus | MIND's inquiry ran only on nights the person was away, and this life had none |
| Oct 31 | adapter-scale, two people | both died before a verdict | this Mac's idle sleep dropped connections (found Nov 1) |
| Nov 1 to 2 | SDFT (self-distillation), `obj-sd`, `obj-sd2`, `obj-sd3` | no morning reached | Colab upload failure, a Fireworks 429, a stale Modal training app, a timer alarm. Its registered rule couldn't be met, so SDFT is off in the frozen recipe. |

## 6. What is and isn't in this repo

- In the repo: every score file, chart, night summary and log under `results/` (simulated people
  only; the exporter refuses anything else), the MIND held-out package, the ledger on `prereg`, and
  the code at every commit.
- Not in the repo:
  - MIND's raw per-life prediction records, about 30 MB a run;
  - the trained checkpoints (`.pt`, weights are refused by the publish scan; their sha256 values
    are in the pre-registration);
  - the Gemini response cache.

  We kept them and can share them on request.

## Appendix: every scored night

Generated from `results/*/scores/night-*.json`. Columns are arm vs base (1) unless named, as
knowledge / interaction. "–" means the arm wasn't run that morning. Daily-v1 mornings use 8
knowledge probes and are noisy; full mornings use 40.

| Account | id | Night | Probe set | Knowledge n | Adapter | 2C v 1 | 3 v 1 | 4 v 1 | 3 v 3x | 4 v 2C |
|---|---|---|---|---|---|---|---|---|---|---|
| film-founder | 9a9e5e9b | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.50 / 0.67 | – / – | – / – | – / – |
| obj-today | cad8f654 | 2026-10-24 | daily-v1 | 8 | promoted | – / – | 0.31 / 0.50 | – / – | – / – | – / – |
| obj-today | cad8f654 | 2026-10-25 | daily-v1 | 8 | promoted | – / – | 0.56 / 0.75 | – / – | – / – | – / – |
| obj-today | cad8f654 | 2026-10-26 | daily-v1 | 8 | promoted | – / – | 0.88 / 0.58 | – / – | – / – | – / – |
| obj-fold | a3c7fdd0 | 2026-10-24 | daily-v1 | 8 | promoted | – / – | 0.50 / 0.75 | – / – | – / – | – / – |
| obj-fold | a3c7fdd0 | 2026-10-25 | daily-v1 | 8 | promoted | – / – | 0.44 / 0.75 | – / – | – / – | – / – |
| obj-fold | a3c7fdd0 | 2026-10-26 | daily-v1 | 8 | promoted | – / – | 0.56 / 0.75 | – / – | – / – | – / – |
| obj-protect | 521bc730 | 2026-10-24 | daily-v1 | 8 | promoted | – / – | 0.38 / 0.58 | – / – | – / – | – / – |
| obj-protect | 521bc730 | 2026-10-25 | daily-v1 | 8 | promoted | – / – | 0.75 / 0.75 | – / – | – / – | – / – |
| obj-protect | 521bc730 | 2026-10-26 | daily-v1 | 8 | promoted | – / – | 0.44 / 0.67 | – / – | – / – | – / – |
| obj-protect-random | 048a59b6 | 2026-10-24 | daily-v1 | 8 | promoted | – / – | 0.62 / 0.58 | – / – | – / – | – / – |
| obj-protect-random | 048a59b6 | 2026-10-25 | daily-v1 | 8 | promoted | – / – | 0.56 / 0.58 | – / – | – / – | – / – |
| obj-protect-random | 048a59b6 | 2026-10-26 | daily-v1 | 8 | promoted | – / – | 0.69 / 0.75 | – / – | – / – | – / – |
| obj-seed2 | 69b587a5 | 2026-10-24 | daily-v1 | 8 | promoted | – / – | 0.38 / 0.75 | – / – | – / – | – / – |
| obj-seed2 | 69b587a5 | 2026-10-25 | daily-v1 | 8 | promoted | – / – | 0.75 / 0.75 | – / – | – / – | – / – |
| obj-seed2 | 69b587a5 | 2026-10-26 | daily-v1 | 8 | promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-seed-cap | c777cb5b | 2026-10-24 | daily-v1 | 8 | promoted | – / – | 0.56 / 0.58 | – / – | – / – | – / – |
| obj-seed-cap | c777cb5b | 2026-10-25 | daily-v1 | 8 | promoted | – / – | 0.69 / 0.58 | – / – | – / – | – / – |
| obj-seed-cap | c777cb5b | 2026-10-26 | daily-v1 | 8 | promoted | – / – | 0.62 / 0.58 | – / – | – / – | – / – |
| obj-ff-donor | da1b273d | 2026-10-24 | full | 40 | promoted | 0.70 / 0.60 | 0.47 / 0.45 | 0.61 / 0.53 | – / – | 0.45 / 0.50 |
| obj-founder-donor | a7982a09 | 2026-10-27 | daily-v1 | 8 | promoted | – / – | 0.56 / 0.42 | – / – | – / – | – / – |
| obj-founder-donor | a7982a09 | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.81 / 0.42 | – / – | – / – | – / – |
| obj-researcher-donor | b5c7000b | 2026-10-27 | daily-v1 | 8 | promoted | – / – | 0.38 / 0.58 | – / – | – / – | – / – |
| obj-researcher-donor | b5c7000b | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.50 / 0.75 | – / – | – / – | – / – |
| obj-founder | 5c25b2d8 | 2026-10-27 | daily-v1 | 8 | promoted | – / – | 0.56 / 0.50 | – / – | – / – | – / – |
| obj-founder | 5c25b2d8 | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.50 / 0.42 | – / – | – / – | – / – |
| obj-researcher | a245a5e9 | 2026-10-27 | daily-v1 | 8 | promoted | – / – | 0.38 / 0.25 | – / – | – / – | – / – |
| obj-researcher | a245a5e9 | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.44 / 0.25 | – / – | – / – | – / – |
| obj-founder | 5c25b2d8 | 2026-10-28 | full | 40 | promoted | 0.79 / 0.70 | 0.57 / 0.53 | 0.81 / 0.68 | 0.50 / 0.57 | 0.49 / 0.53 |
| obj-founder-donor | a7982a09 | 2026-10-28 | full | 40 | promoted | 0.75 / 0.67 | 0.61 / 0.48 | 0.74 / 0.70 | 0.53 / 0.38 | 0.38 / 0.47 |
| obj-researcher | a245a5e9 | 2026-10-28 | full | 40 | promoted | 0.65 / 0.47 | 0.50 / 0.57 | 0.65 / 0.50 | 0.60 / 0.62 | 0.49 / 0.40 |
| obj-researcher-donor | b5c7000b | 2026-10-28 | full | 40 | promoted | 0.72 / 0.63 | 0.62 / 0.50 | 0.68 / 0.72 | 0.61 / 0.53 | 0.42 / 0.55 |
| obj-mind-film-founder | bcbe8960 | 2026-10-07 | daily-v1 | 8 | promoted | – / – | 0.44 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-08 | daily-v1 | 8 | promoted | – / – | 0.50 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-09 | daily-v1 | 8 | promoted | – / – | 0.38 / 0.42 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-10 | daily-v1 | 8 | promoted | – / – | 0.31 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-11 | daily-v1 | 8 | promoted | – / – | 0.69 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-12 | daily-v1 | 8 | promoted | – / – | 0.62 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-13 | daily-v1 | 8 | promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-14 | daily-v1 | 8 | promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-15 | daily-v1 | 8 | not_promoted | – / – | 0.62 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-16 | daily-v1 | 8 | not_promoted | – / – | 0.94 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-17 | daily-v1 | 8 | not_promoted | – / – | 0.69 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-18 | daily-v1 | 8 | not_promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-19 | daily-v1 | 8 | not_promoted | – / – | 0.81 / 0.67 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-20 | daily-v1 | 8 | not_promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-21 | daily-v1 | 8 | not_promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-22 | daily-v1 | 8 | not_promoted | – / – | 0.62 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-23 | daily-v1 | 8 | not_promoted | – / – | 0.94 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-24 | daily-v1 | 8 | not_promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-25 | daily-v1 | 8 | not_promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-26 | daily-v1 | 8 | not_promoted | – / – | 0.81 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-founder | bcbe8960 | 2026-10-27 | daily-v1 | 8 | not_promoted | – / – | 0.88 / 0.75 | – / – | – / – | – / – |
| obj-mind-film-student | 7286c841 | 2026-10-07 | daily-v1 | 8 | promoted | – / – | 0.38 / 0.33 | – / – | – / – | – / – |
| obj-lever3-founder-donor | 5974ad07 | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.62 / 0.33 | – / – | – / – | – / – |
| obj-lever2-researcher-do | bb5716dd | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.50 / 0.58 | – / – | – / – | – / – |
| obj-lever2-founder | aeb62d9c | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.69 / 0.58 | – / – | – / – | – / – |
| obj-lever3-founder-donor | 5974ad07 | 2026-10-28 | full | 40 | promoted | 0.64 / 0.63 | 0.44 / 0.50 | 0.66 / 0.70 | 0.62 / 0.52 | 0.50 / 0.53 |
| obj-mind-week2-founder | f71d8354 | 2026-10-16 | daily-v1 | 8 | promoted | – / – | 0.19 / 0.58 | – / – | – / – | – / – |
| obj-mind-week2-founder | f71d8354 | 2026-10-17 | daily-v1 | 8 | promoted | – / – | 0.31 / 0.58 | – / – | – / – | – / – |
| obj-mind-week2-founder | f71d8354 | 2026-10-19 | daily-v1 | 8 | promoted | – / – | 0.62 / 0.58 | – / – | – / – | – / – |
| obj-mind-week2-founder | f71d8354 | 2026-10-21 | daily-v1 | 8 | not_promoted | – / – | 0.75 / 0.58 | – / – | – / – | – / – |
| obj-mind-week2-founder | f71d8354 | 2026-10-22 | daily-v1 | 8 | not_promoted | – / – | 0.62 / 0.75 | – / – | – / – | – / – |
| obj-mind-week2-founder | f71d8354 | 2026-10-23 | daily-v1 | 8 | not_promoted | – / – | 0.50 / 0.58 | – / – | – / – | – / – |
| obj-loop-founder | d2972e79 | 2026-10-17 | daily-v1 | 8 | promoted | – / – | 0.75 / 0.58 | – / – | – / – | – / – |
| obj-loop-founder | d2972e79 | 2026-10-18 | daily-v1 | 8 | promoted | – / – | 0.69 / 0.75 | – / – | – / – | – / – |
| obj-loop-founder | d2972e79 | 2026-10-20 | daily-v1 | 8 | promoted | – / – | 0.75 / 0.75 | – / – | – / – | – / – |
| obj-dd-founder-donor | ea730ced | 2026-10-27 | daily-v1 | 8 | promoted | – / – | 0.62 / 0.25 | – / – | – / – | – / – |
| obj-dd-founder-donor | ea730ced | 2026-10-28 | daily-v1 | 8 | promoted | – / – | 0.38 / 0.50 | – / – | – / – | – / – |
| obj-dd-founder-donor | ea730ced | 2026-10-28 | full | 40 | promoted | 0.75 / 0.73 | 0.57 / 0.63 | 0.56 / 0.70 | – / – | 0.35 / 0.57 |
| obj-fin-a | a3aec189 | 2026-10-28 | full | 40 | promoted | 0.81 / 0.65 | 0.54 / 0.48 | 0.62 / 0.55 | 0.78 / 0.57 | 0.25 / 0.37 |
| obj-fin-b | 4fc642c6 | 2026-10-28 | full | 40 | not_promoted | 0.69 / 0.58 | 0.44 / 0.45 | 0.40 / 0.47 | 0.90 / 0.67 | 0.19 / 0.38 |
| obj-fin-s | 13f17861 | 2026-10-28 | full | 40 | promoted | 0.79 / 0.75 | 0.74 / 0.62 | 0.70 / 0.72 | 0.66 / 0.57 | 0.42 / 0.38 |
