# The night's story: who fills which record

The night is one loop: **sleep → dream → rehearse → learn → check → wake**. `story.py` is its one
versioned schema (`STORY_VERSION`). `story.json` sits in the night's folder beside `log.jsonl` and
`summary.json`. `tell` builds it from the night's log records and nothing else, so it is never a
second log. The night writes it when it finishes, and `daily.py` tells it again after the morning.
The demo page's night view reads it (`demo/night_view.py`, `web/night.js`).

A record a step can't fill yet is `NotRun(reason)`, never missing, `None` or a zero. Each one still
waiting for its branch names that branch in a `*_NOT_YET` reason in `story.py`.

| Record | Fields | Owner |
|---|---|---|
| `sleep` | `Sleep(kind, pressure, threshold, reason)` | sleep-pressure (filled) |
| `dream.questions` | questions the notice pass opened tonight | MIND's inquiry (filled) |
| `dream.ignitions`, `dream.threads`, `dream.boundaries` | `Ignition(kind, score, cites)`, `Thread(slot, score, cites)`, `Boundary(state, slot, steps)` | **mind-frontier** |
| `rehearse.scenes`, `flagged`, `corrected` | practice's scenes, the critic's pairs, the teacher's corrections | night-story (filled) |
| `rehearse.seeded` | `Seed(slot, scenes)` per picked thread (the dream curriculum) | **mind-frontier** |
| `learn.rows` | the dataset's examples | night-story (filled) |
| `learn.surprise` | `Surprise(applied, rows, mean)`: whether the loss applied MIND's weights | **sleep-objective** (the weights come from mind-frontier's dream) |
| `learn.channel_loss`, `learn.stamp` | the projector and reader's loss, and the projector version | **thought-channel** |
| `learn.seeding_kl`, `learn.fold` | the seeding term's KL, and the rank the adapter was folded to | **sleep-objective** |
| `learn.core_adaptation` | the core's per-person adaptation loss on today's resolved outcomes | **mind-frontier** |
| `check` | `Check(gates, promoted)`, each `Gate(name, passed, skipped, value, threshold)` | night-story (filled) |
| `wake.morning` | `Morning(replies, no_verdict)` or `Failed("morning", error)` | night-story (filled by `daily.py`) |
| `wake.self_model` | `SelfModel(predicted, actual)`: MIND's predicted error against its real error | **mind-frontier** |

To fill a record:

1. Log what you measure in your step's record in `log.jsonl`, as ids and numbers (no words). Give a
   new step its stage in `STAGE_STEPS`.
2. In `tell`, read that record in place of your `*_NOT_YET` reason. A record that lacks what the
   story needs raises `StoryError`. The night then logs a warning and keeps no story, and the page
   shows the story unavailable.
3. Extend `tests/test_night_story.py`: the round trip, and a told night with your field filled.
4. Adding, removing or retyping a field is a new `STORY_VERSION`. Filling a field the schema already
   holds is not.
