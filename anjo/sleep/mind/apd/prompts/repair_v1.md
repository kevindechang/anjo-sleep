# repair_v1

This day was written before and failed a check of the whole life. Write it again, as the same day.

- `before` is the day as it was written, and `reason` says what the check found.
- `before` is rejected text, not an answer to copy. Read `reason` against those messages and
  remove or rewrite the wording that causes the failure. At least one message must change in
  substance; changing only case or spacing does not repair the day.
- `rewrite_ask` counts the asks for this repair. On ask 2, copying has already failed: choose a
  different wording that removes the cause in `reason`, while preserving the required facts.
- `later` holds the days written after it. Keep the day consistent with them, and change only what
  the reason needs.
- Keep every beat: each `plant` fact said in passing, in a message of its own, each `ask` raised,
  and `placed` saying where.
- Say each fact plainly and leave it there. Do not tie it to another fact, day, plan, date or
  number. Do not call anything a problem, a conflict, a mistake, a gap, a risk, a saving or an
  opening, or say that something is fine after all.
- Add nothing that could be a problem, a conflict, a missing step or an opening.

Return a NEW JSON object using the same schema: `messages` and `placed`. Return the complete
revised day, including messages you kept, and update `placed` if message positions changed.
Do not return the old message contents unchanged.
