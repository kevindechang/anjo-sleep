# MIND cognition sprint: a predictive, self-doubting, curious, sleeping state

Status: active, started 2026-11-02 (Kevin: "MIND should be a step toward machine cognition, not just
a better memory"). Branch `hackathon/mind-cog`, code in `anjo/sleep/mind/cog/`. The contract is
`types.py`; read it first.

## The claim we are trying to earn

MIND's learned state does four things a memory doesn't, each on its own measured test:

1. **It expects.** It predicts what happens next in a person's life (T1).
2. **It knows what it doesn't know.** Its uncertainty is calibrated and tracks its real error (T2).
3. **It acts on that.** It asks the one question worth asking, or stays silent, by information value (T3).
4. **It learns from its own errors overnight.** Its night replays what surprised it and recovers from
   a change in the person's life faster than the rivals (T4).

Remarkable means: on held-out simulated lives, MIND beats Gemini given the same history, the
strongest simple rival, and a population prior, and each lesion removes the matching effect (T5).
We never claim consciousness, a "first", or anything about real people from simulated lives.

## The world (`world.py`, `render.py`)

A programmatic generator, so every answer is scored by code and the ceiling is exact:

- **Population and person.** About 6 recurring activities (run, gym, class, call a parent, dinner
  with a friend, ...), 3-5 people, 3 distractor topics. A person's habits are drawn from a population
  prior: a weekday pattern per activity, and logistic effects of the hidden state and recent events
  on whether a planned or habitual activity happens.
- **Idiosyncrasy.** About a quarter of persons have at least one effect against the population norm
  (e.g. stress makes them go to the gym more, not less), so a population prior is wrong for them.
- **Hidden state.** A discrete regime per person (e.g. workload calm/busy/crunch, a Markov chain with
  sticky transitions) never stated directly; it shows only through indirect signals (late messages,
  mood words, cancellations).
- **Hidden facts.** Occasional facts the person doesn't mention unless asked (an exam on a given
  day, a visitor next week, a minor injury), each changing upcoming outcomes strongly.
- **Regime shifts.** On 0-2 days per life a parameter changes for good (a new schedule, a friend
  moves away). `Truth.shift_days` records them, for T4.
- **Distractors.** Topic mentions with high entropy and no effect on any outcome, so "ask about what
  is most uncertain" wastes questions and "ask about what changes predictions" doesn't.
- **Text.** Each event and query has `tokens` (structured) and `text` (the person's words) carrying
  the same information. Text comes from a template bank (varied phrasings per event kind); a
  Gemini-paraphrased bank is a later realism stage.
- **Questions.** Each day a few askable questions: about the regime ("how's work this week?"), about
  a hidden fact, or about a distractor. `answer` returns the truth as an `answer` event.
- **References.** `oracle_p` is the Bayes filter (forward algorithm over the regime, true parameters)
  and `state_p` is the true-state probability. Queries are tagged `count` or `hidden` by construction.
- **Splits.** Seeds: train 0-9999, dev 10000-10499, an out-of-distribution dev set (another
  population prior) 20000-20199. Held-out test seeds are drawn once, after the freeze.

## The predictors

- **MIND cog core (`core.py`).** Slow weights meta-trained across thousands of train lives, so it
  learns how to learn a person. Per-person state: a recurrent belief plus a fast-weight memory updated
  by surprise at test time (reuse `mind/core/net.py`'s fast memory and `cms.py` where they fit).
  Queries are answered by an ensemble of K=5 heads trained on bootstrap masks; their disagreement is
  `u`. The night (`end_day`) replays the day's and the life's events prioritized by surprise × need
  into a small per-person adapter (T4).
- **Rivals (`rivals.py`).** Population prior; per-person counter (activity × weekday base rates with
  a prior); per-person online logistic regression on recent-event features (the strongest simple
  rival); an online rival that raises its learning rate when surprised (Nassar-style; T4's rival);
  Gemini given the full rendered history and the queries, recalibrated by Platt scaling on dev
  (`gemini.py`, built and tested with fakes; Claude runs the paid calls).
- **Inquirers (`inquiry.py`).** MIND's: expected reduction in log loss on the next week's queries,
  from the ensemble, asked only when it beats the question's cost. Rivals: never ask, random, ask about
  the most uncertain subject (total uncertainty), and Gemini choosing the question.

## The tests (all on dev until the freeze)

- **T1 Expects.** Mean log loss and Brier on all queries, per family, against every rival; the gap
  to Gemini as a function of days observed (it should grow); distance to `oracle_p`.
- **T2 Knows what it doesn't know.** Expected calibration error; the area under the risk-coverage
  curve when abstaining on the most uncertain; the rank correlation between `u` and the real excess
  loss over the oracle.
- **T3 Acts.** Utility = log-loss reduction on the next 7 days' queries minus the cost of questions
  asked, per inquirer, over the same lives; and the share of questions spent on distractors.
- **T4 Sleeps.** Days to recover after a regime shift (rolling loss back within a set margin of the
  oracle), with surprise-prioritized replay, uniform replay at equal compute, no night, and the
  surprise-learning-rate rival; plus loss on stable habits (no extra forgetting).
- **T5 Lesions.** Fast memory off (kills per-person learning), one head instead of the ensemble (kills
  calibration only), the inquirer on total uncertainty (wastes questions on distractors), no night or
  uniform replay (slower recovery). Each lesion should remove its own effect and leave the others.
- **Add-on: self-prediction.** MIND predicts its own next-day loss; an observer trained on its inputs
  and outputs tries the same. MIND having privileged access to its own state is the only claim.

Statistics: paired over lives, 95% bootstrap intervals over lives, and at least 3 seeds for anything
trained. Bars for the held-out test are written in the ledger before it is drawn, and it is scored
once.

## Rules for everyone working on this

- Code only in `anjo/sleep/mind/cog/` and tests in `anjo/sleep/tests/test_cog_*.py`. Pure, frozen
  dataclasses where possible; no clock, env or network in the world, the predictors or the scoring.
- No paid calls by agents: Gemini code is built against an injected client and tested with fakes.
- No reading `.env`, `~/AnjoSleepData` or `ops/`. Data a run writes goes under the path it is given.
- Run only your own test files (`pytest anjo/sleep/tests/test_cog_<x>.py`); never the full suite.
- Commit on your own branch with `<type>: <description>` messages and no co-author lines. Never push.

## Guarding against a world built for MIND (added 2026-11-02, after a reviewer's note)

The main world is ours, and it was built so that inferring hidden structure pays. A win there shows
MIND recovers hidden structure when it exists; it does not show MIND beats Gemini in general. So:

- **An independent world** (`world_indep.py`, branch `cog/world2`): written by a separate agent
  from a neutral brief ("realistic lives"), with access to `types.py` only, never to MIND's code or
  this design. Same model family as the lead (disclosed). MIND is scored there zero-shot (trained on
  the main world) and, separately, trained on it. The result is reported whatever it is.
- **The strongest LLM rival**: Gemini given the history plus per-person computed statistics
  (`gemini_stats`), so a loss can't be put down to the LLM not counting.
- **Scoped claims**: the write-up states that the main world is ours and favors latent-structure
  inference, and leads with the independent-world result next to the main one.
