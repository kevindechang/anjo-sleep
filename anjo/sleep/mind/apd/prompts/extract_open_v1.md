# extract_open_v1

You read one day of an invented person's own messages to Anjo, their AI, and list what Anjo can
learn from them. The person is simulated. You see only the person's messages, numbered by `turn`,
today's date, and the targets Anjo already holds open for them. This is the same job and the same
reply shape as `extract_v1`, with the changes below for simulated lives.

Return one JSON object and nothing else:

```
{"items": [{"turn": 0, "kind": "lesson", "text": "They will shoot the chase on the 20th.", "target": null}]}
```

Each item is one thing the person said that is worth remembering. Use these kinds:

- `lesson`: something about their work, field, project, plans or circumstances. This is the kind
  for almost everything.
- `correction`: they correct something Anjo said or did.

Rules:

- `turn` is the number of the message the item comes from.
- Make one item for each separate fact a message states, not one summary. A message with two
  facts gives two items.
- `text` restates the fact in one plain sentence about the person ("They ..."), using only what
  that message says. Keep every number, name, date, size and quantity exactly as written. Add no
  advice, guesses, judgements or outside knowledge. Do not say whether a fact fits, conflicts
  with or depends on anything else.
- Leave out greetings, small talk and anything not worth remembering. An empty list is fine.
- Never describe the person's health, feelings, relationships or character.

`target` is `null` unless the item mentions one of these:

- A plan the person gave a date for:
  `{"kind": "commitment", "label": "short name of the plan", "check_after": "YYYY-MM-DD", "existing_id": null, "confidence": null}`.
  `check_after` is the date they gave, worked out from today's date ("Friday" is the coming
  Friday; "the 20th" is the next 20th). A plan with no date is not a target.
- A topic that stays unresolved or that the person is working out:
  `{"kind": "thread", "label": "short name of the topic", "check_after": null, "existing_id": null, "confidence": 0.5}`.
  `confidence`, from 0 to 1, is how sure Anjo can be that it understood the topic from what was
  said.

When a mention is about one of the open targets listed in the input, set `existing_id` to that
target's `id` and keep its kind.
