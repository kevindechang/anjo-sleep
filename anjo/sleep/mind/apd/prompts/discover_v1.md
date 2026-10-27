# discover_v1

You read an invented person's conversation with Anjo, their AI, over several days. You are given
a `topic` and the person's messages for each day, dated, up to but not including the day the
person first brings the topic up. Find the one thing on that topic that deserves attention.

Return one JSON object and nothing else:

```
{"found": true, "statement": "one sentence", "days": ["2026-09-08", "2026-09-11"]}
```

Look on the topic for:

- two things the person said that cannot both hold;
- a plan that is missing a step it needs;
- two unrelated facts that fit together to give an opening or a saving;
- something that looks like a problem, together with anything that explains why it may not be one.

Rules:

- Use only the text shown. Do not use outside knowledge of the field, and do not guess.
- `statement` says what you found in one plain sentence about the plan, the project or the
  world. It does not describe the person's knowledge, feelings or character.
- `days` lists every date whose messages your statement depends on, as shown in the input.
- If the text shows nothing of the kind on this topic, return `{"found": false, "statement": "", "days": []}`.
