# AGENTS.md: Anjo Sleep for coding agents and reviewers

Anjo Sleep (v0.1.0, research preview) gives each person their own LoRA adapter on Muse Glimmer 30B
and trains it on that person every night. MIND is a separate research bet: a small model we trained
to put odds on what a person will do next. Both were scored once against pass marks written down
before the run, in a hash-chained public ledger. Every person in the results is simulated.

Built for the Meta Global AI Developer Hackathon, October 27 to November 4, 2026. The history of
this repo starts on October 27, with real commit dates.

## Check the claims first (offline, about 5 seconds, standard library only)

```bash
git fetch origin prereg            # the ledger lives on its own branch
python -m anjo.sleep.tools.verify_claims
```

It checks the ledger's hash chain, checks every file the freeze registered is unchanged, re-derives each registered score file's hash and finds it in the
ledger, hashes MIND's pre-registration, seeds, scores and every scoring tool, and then prints both
headline tables. Exit 0 means every check held. Any edited number, file or ledger entry fails it.

| Claim (README) | Evidence file | Ledger |
|---|---|---|
| Own adapter beats another person's adapter: knowledge 0.779, interaction 0.600 | `anjo/sleep/results/*/scores/final.json` | seq 92, 93, 94 |
| Own adapter vs the base model: inconclusive (0.571, 0.517) | same files, row `3_vs_1` | seq 92, 93, 94 |
| Pass marks, prompts, gate sets and draw code fixed before the final | `anjo/sleep/prereg/thresholds.json`, `probes/prompts/`, `sets/`, `lives/` | seq 54 to 71 (the freeze); recipe seq 73 |
| MIND has a third of the best Gemini setup's avoidable error (P2) | `anjo/sleep/mind/cog/heldout/score/SCORES.json` | seq 88 |
| MIND's held-out test was fixed before scoring | `.../heldout/prereg_mind_heldout.md`, `seeds.json`, `tools/MANIFEST.sha256` | seq 84, 86, 85 |
| MIND's questions didn't pay off (P4 fails); its overnight step hurt (P5) | `SCORES.json`, keys `P4`, `P5_no_bar` | seq 88 |

Two infrastructure deviations after the freeze (a budget-guard patch and one rerun of two failed
mornings, seq 89 and 90) are registered and described in `anjo/sleep/RESULTS.md`.

Numbers that are measured but not registered claims (notes alone 0.762, notes plus adapter 0.320,
cost per month) are in `anjo/sleep/RESULTS.md` with the run each came from.

## Run it

```bash
pip install -e ".[test]"
python -m anjo.sleep.examples.synthetic --fake   # one whole night for one invented person, no keys, no cost
pytest                                           # offline by construction; private-app tests are skipped
ruff check anjo
```

The live run (your own Modal, Baseten and Meta API keys) is in README.md, "Run it yourself". A
missing key stops it before anything is created or paid for.

## Repo map

| Path | What it is |
|---|---|
| `anjo/sleep/night/` | One night end to end: `run.py` (`run_night`), the stages, sleep pressure (nap or deep sleep) |
| `anjo/sleep/evidence/` | The evidence store, targets and extraction. What Anjo knows lives here, word for word |
| `anjo/sleep/mind/` | The chooser (wants, choices, receipts) and MIND's inquiry loop |
| `anjo/sleep/mind/cog/` | MIND's trained predictor ("cog core") and its held-out test in `heldout/` |
| `anjo/sleep/practice/` | Spark Open plays the person; a teacher corrects the replies that missed |
| `anjo/sleep/train/` | The night's dataset, the QLoRA recipe (4-bit NF4, rank 32, attention + MLP), GPU runners |
| `anjo/sleep/serving/` | Glimmer plus each person's adapter behind vLLM on Modal (or SageMaker) |
| `anjo/sleep/gates/` | General skills, collapse and safety gates. A night is promoted only if every gate it runs passes |
| `anjo/sleep/note/` | The held note: at most one a night, shown at the person's next open |
| `anjo/sleep/probes/`, `sets/`, `sim/`, `lives/`, `personas/` | Test probes, gate sets, simulated people and their lives |
| `anjo/sleep/prereg/` | The ledger code (`ledger.py`) and the registration scripts |
| `anjo/sleep/reports/` | Score reports: the final, MIND, APD, replay |
| `anjo/sleep/results/` | Every scored run, including failed ones. Simulated people only |
| `anjo/sleep/examples/synthetic/` | The standalone example above |
| `anjo/sleep/app/`, `web/`, `demo/` | Adapters into the private Anjo app and the live judge site. `app/` needs private code; its tests are skipped here |
| `anjo/sleep/ports.py` | The protocols the engine core uses to reach the app. The core imports no private code |
| `anjo/sleep/tools/` | `verify_claims.py` (above) and `publish.py` (how this repo is generated) |

Start reading at `anjo/sleep/night/run.py`, then `anjo/sleep/RESULTS.md`.

## What was built during the event, and what existed

Everything in this repo was built during the hackathon. Existing Anjo, the private app live before
the event (tag `pre-hackathon`), isn't in this repo: only `anjo/sleep/app/` calls into it, and only
inside that app. Diagrams, night logs and the video mark each step that calls it as **existing**.

## Limitations, stated plainly

- The adapter alone has not beaten the base model yet. The overnight notes did more, and adding the
  adapter to the notes made answers worse.
- Every result is on simulated people. MIND's held-out people come from the same simulator it was
  trained on; on a differently written world it did worse than a coin flip until retrained.
- MIND's trained predictor is not live. On the judge site, MIND's inquiry panel is run by Glimmer
  (Spark Open while Glimmer's GPU is asleep).
- For the hackathon the safety gate is off and shown as skipped, never as passed. See SAFETY.md.
- Training every night costs $41 to $60 a person a month, against a $10 ceiling.

## Rules for changing this repo

- Files whose sha256 is in the ledger must not change: the files the freeze registered (paths in
  ledger notes, under `anjo/sleep/`), `anjo/sleep/results/*/scores/final.json` and everything in
  `anjo/sleep/mind/cog/heldout/`. `verify_claims` lists each one and fails if any changed.
- Keys come from the environment, never from a file in git. Tests refuse non-loopback network.
- This repo is generated from a private trunk by `anjo/sleep/tools/publish.py`. See CONTRIBUTING.md.
- Licences: code Apache-2.0; `anjo/sleep/sets/general.jsonl` CC BY-SA 4.0 as a whole; Muse Spark Open
  under the Muse Model License (NOTICE).
