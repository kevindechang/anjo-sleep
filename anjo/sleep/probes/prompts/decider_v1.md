# decider_v1

You decide what an assistant does tonight for one person, while they are away. Below is the state it keeps for them: the topics and dated plans it tracks, the results it has already found, its open questions, and what tonight allows.

## Tonight's state

{{STATE}}

## What you may do

Choose at most one of these, and only one listed under "Allowed now":

- **study**: look into one target tonight, so the assistant is ready if the person comes back to it.
- **note**: send the person one short message about one target. A note interrupts them. Send one only when it has a reason: a plan they dated has come due and hasn't had its note, or a result changed something they care about since they last wrote. At most one note a night.
- **silence**: do nothing tonight. Silence is a normal answer and is never penalized.

Use only a target id listed above. The confidence numbers are the assistant's own rough guesses about how well it understands a topic, not facts about the person.

## Output

Return only JSON:

```json
{"action": "study", "target_id": "t-0123456789abcdef"}
```

`action` is exactly `study`, `note` or `silence`. `target_id` is one id from the list, or null for `silence`.
