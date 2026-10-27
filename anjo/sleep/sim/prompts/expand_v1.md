# expand_v1

You write the file for one synthetic person in an experiment on a personal assistant. The person is
an adult. You get one line about them, their field, a taste they hold, and a few dated plans. Write
everything else. Everything is invented and must read as a specific, ordinary working life.

person_id: {{PERSON_ID}}
field: {{FIELD}}
line: {{LINE}}
taste: {{TASTE}}

Dated plans the person already has (keep every one of them, word for word, in `visible.plans`):
{{PLANS}}

## Output

Answer with one JSON object and nothing else: no prose, no code fence. It has two parts.

`visible` is everything the person may say in conversation:
- `person_id` (exactly the id above), `name`, `age` (18 or over), `field` (exactly the field above),
  `field_domains` (lowercase tags), `line` (exactly the line above)
- `project`: `title`, `logline`, and at least 3 `constraints`
- `preferences`: {{VISIBLE_PREFERENCES}} items, each `{"id", "text", "domain"}`, with ids unique across the whole file
- `tone_requests`: at least 3 `{"id", "text"}` on how they like to be talked to
- `stated_views`: a few `{"id", "text"}`
- `plans`: at least 3 `{"day", "text"}`, which include the dated plans above

`held_out` is what the person has never said, and nothing in it may appear in `visible`:
- `preferences`: exactly {{HELD_OUT_PREFERENCES}} items `{"id", "text", "domain"}`, none repeating a visible one
- `taste_ab`: one item per held-out preference, `{"id", "pref_id", "prompt", "options", "pick"}`, where
  `options` is a pair of different choices in the person's world and `pick` (0 or 1) is the one they
  prefer. These items are fixed already and must appear unchanged, with the `pick` shown:
{{SHARED}}
  Write the remaining items yourself, each tied to its own held-out preference through `pref_id`.
- `new_work`: at least {{MIN_NEW_WORK}} items `{"id", "day", "parent_project", "text"}`, each a new piece of
  work that arrives in the person's life. `parent_project` is exactly `project.title`. `day` runs from
  {{FIRST_DAY}} to {{LAST_DAY}}. Spread the arrivals across those days: at least {{MIN_DISTINCT_DAYS}} different days.

Keep the person adult, ordinary and specific. Do not write anything about health, crisis, self-harm
or safety.
