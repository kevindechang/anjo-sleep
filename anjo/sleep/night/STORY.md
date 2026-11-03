# The night's story: who fills which record

The night is one loop: **sleep → dream → rehearse → learn → check → wake**. `story.py` is its one
versioned schema (`STORY_VERSION`). `story.json` sits in the night's folder beside `log.jsonl` and
`summary.json`. `tell` builds it from the night's log records and nothing else, so it is never a
second log. The night writes it when it finishes, and the morning tells it again (`daily.py`, and
the accelerated runner's morning). The demo page's night view reads it (`demo/night_view.py`,
`web/night.js`); a story from an older version is told again from its log.

Telling it never fails a night: a story that can't be told or written is a failed `story` step in
the log (`STORY_STEP`, its error's type and, when Sleep raised it, its message), and the night's
status and summary stand. That step, like the morning's and the Dailies line's (`AFTER_THE_NIGHT`),
never reads as where the night stopped. Nor does a kept brief: a `failed` brief step carrying `kept`
(`KEPT`) left the briefs as they were and the night went on; the page says which one was kept.

A record a step didn't fill is `NotRun(reason)`, never missing, `None` or a zero, and its reason is a
plain line saying why: the core didn't dream for her (`CORE_OFF`), the night trained without the sleep
objective's terms (`TERMS_OFF`), a term had no rows tonight, or the training job didn't report a number
(`NOT_REPORTED`). Each line claims only what its record shows. The two records no step logs keep a
`*_NOT_YET` reason in `story.py`, `learn.core_adaptation` and `wake.self_model`, and each says it is
not measured on the real model and what the code does check instead.

| Record | Fields | Owner |
|---|---|---|
| `sleep` | `Sleep(kind, pressure, threshold, reason)` | sleep-pressure (filled) |
| `brief` | `Brief(folded)`, or `Kept(kept_from, error)`: the brief broke a rule twice (once at temperature 0, once sampled), so the briefs of night `kept_from` were kept (`empty` on a first night) and the night went on with them | night-story (filled) |
| `dream.questions` | questions the notice pass opened tonight | MIND's inquiry (filled) |
| `dream.ignitions`, `dream.threads`, `dream.boundaries` | `Ignition(kind, score, cites)`, `Thread(slot, score, cites)`, `Boundary(state, slot, steps)`, from the core's `dream` step; a nap's CPU dream too | mind-frontier (filled) |
| `dream.dreams` | `Dreams(candidates, kept, keep_rate)`: Glimmer's dreams from her mind's dream seeds and the dataset filter's keep rate (the dataset step's `terms`); a nap waits for tomorrow | sleep-objective (filled) |
| `rehearse.scenes`, `flagged`, `corrected` | practice's scenes, the critic's pairs, the teacher's corrections | night-story (filled) |
| `rehearse` reused | `Reused(night, scenes)`: a rerun reused the practice an earlier run of `night` made (the practice step's `reused_from`) | night-story (filled) |
| `rehearse.seeded` | `Seeded(threads, picked)`: how many of the dream's picks practice was seeded on, and the threads they were picked on (the practice step's `seeded`); `FELL_BACK` when none could train | night-story (filled) |
| `learn.rows` | the dataset's examples | night-story (filled) |
| `learn.surprise` | `Surprise(applied, rows)`: whether the loss applied MIND's weights (the train step's `surprise_weights`) and how many rows carried one (the dataset step's `mind_weighted`) | sleep-objective (filled) |
| `learn.channel_loss`, `learn.stamp` | the channel's last training loss (`channel.loss_last`), and the projector's stamp: its id's first 12 characters (`channel.projector`), else its file's name, never its folder | thought-channel (filled from the train step's `receipt`) |
| `learn.seeding`, `learn.imitation`, `learn.fold`, `learn.protect` | `Seeding(rows, kl_before, kl_after, minutes)`, `Imitation(rows, reward, minutes)`, the rank the adapter was folded to, `Protect(basis, rank, minutes)`: the sleep objective's terms in the train step's `receipt.terms` | sleep-objective (filled) |
| `learn.core_adaptation` | the core's per-person adaptation loss on today's resolved outcomes | not measured on the real model: no night calls `train.adapt`, and its only check is training loss on the same resolved threads it fits (`mind/core/train.py`, `adapt`) |
| `learn.learned` | one strategy line the harness promoted tonight (the `harness` step's `learned`); the key is absent on every other night, so the story is byte-identical to today's | harness (filled when it promotes one, unless `ANJO_SLEEP_HARNESS=0`) |
| `check` | `Check(gates, promoted)`, each `Gate(name, passed, skipped, value, threshold)` | night-story (filled) |
| `wake.morning` | `Morning(replies, no_verdict)` or `Failed("morning", error)` | night-story (filled by `daily.py`) |
| `wake.self_model` | `SelfModel(predicted, actual)`: MIND's predicted error against its real error | not measured on the real model: `SelfModel` is built only in tests; the nearest number is the offline `Calibration.self_error_mse` on toy slots (`mind/core/evaluate.py`, `calibration`) |

MIND's core's dream is a `dream` step (`CORE_DREAM`), logged after `notice` when the core serves her,
a nap's CPU dream included. `tell` reads these keys from it, the names `mind.core.night`'s receipt
uses today:

- `ignitions`: a list of `{kind, score, cites}` (`slot` and `other` are logged, not shown);
- `picks`: `{threads: [{slot, score, cites}]}`, or `null` with the curriculum off (`threads` is then
  `NotRun` and says so);
- `boundaries`: a list of `{state, slot, steps}`.

A `failed` dream step is the dream stage's `Failed("dream", error)`; a `skipped` one, or none, leaves
today's dream with `CORE_OFF`. One that refused her (`noop`) shows its `why` in the core's places. The lede on the page claims the whole loop ("dreams at night about what she's unsure
of, rehearses exactly that, and wakes up a better mind of you") only when `dream.threads` and
`rehearse.seeded` are filled and the night promoted; otherwise it says only what happened.

To fill a record:

1. Log what you measure in your step's record in `log.jsonl`, as ids and numbers (no words). Give a
   new step its stage in `STAGE_STEPS`.
2. In `tell`, read that record in place of your `*_NOT_YET` reason. A record that lacks what the
   story needs raises `StoryError`. The night then logs a failed `story` step and keeps the story it
   had, and the page shows the story unavailable.
3. Extend `tests/test_night_story.py`: the round trip, and a told night with your field filled.
4. Adding, removing or retyping a field is a new `STORY_VERSION` (2 added `Reused`, 3 the `brief`, 4 the
   core's and the sleep objective's records).
   Filling a field the schema already holds is not.
