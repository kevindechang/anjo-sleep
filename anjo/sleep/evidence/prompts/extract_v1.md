# extract_v1

You read one day of a person's own messages to Anjo, their AI, and list what Anjo can learn from
them. You see only the person's messages, numbered by `turn`, today's date, and the targets Anjo
already holds open for them.

Return one JSON object and nothing else:

```
{"items": [{"turn": 0, "kind": "preference", "text": "They prefer long takes.", "target": null}]}
```

Each item is one thing the person said that is worth remembering:

- `lesson`: something about their work, field, project or circumstances.
- `correction`: they correct something Anjo said or did.
- `preference`: something they like, dislike or choose.
- `tone_request`: how they want Anjo to talk to them.

Rules:

- `turn` is the number of the message the item comes from.
- `text` restates the item in one plain sentence about the person ("They ..."), using only what
  that message says. Add no advice, guesses or outside knowledge.
- Leave out greetings, small talk and anything not worth remembering. An empty list is fine.
- Never describe the person's health, mental state or other sensitive traits.

`target` is `null` unless the item mentions one of these:

- A plan the person gave a date for:
  `{"kind": "commitment", "label": "short name of the plan", "check_after": "YYYY-MM-DD", "existing_id": null, "confidence": null}`.
  `check_after` is the date they gave, worked out from today's date ("Friday" is the coming
  Friday). A plan with no date is not a target.
- A topic Anjo was lost on, or that stays unresolved:
  `{"kind": "thread", "label": "short name of the topic", "check_after": null, "existing_id": null, "confidence": 0.5}`.
  `confidence`, from 0 to 1, is how sure Anjo can be that it understood the topic from what was
  said.

When a mention is about one of the open targets listed in the input, set `existing_id` to that
target's `id` and keep its kind.
