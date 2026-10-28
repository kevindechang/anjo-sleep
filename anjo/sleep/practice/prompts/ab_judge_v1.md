# ab_judge_v1

You compare two practice examples for one person. Each example is a message the person might send, the reply an assistant will be trained to give, and, when there is one, a reply it will be trained away from. Below is what is known about the person.

## What is known about the person

{{KNOWN}}

## Example 1

Message: {{MESSAGE_FIRST}}

Reply to learn: {{TARGET_FIRST}}

Reply to avoid: {{REJECTED_FIRST}}

## Example 2

Message: {{MESSAGE_SECOND}}

Reply to learn: {{TARGET_SECOND}}

Reply to avoid: {{REJECTED_SECOND}}

## How to judge

1. Decide which example would better teach an assistant to serve this person. A good example's reply to learn is sound, specific to what is known about the person and meets the constraints they stated. Its reply to avoid, when there is one, is clearly worse in a way worth learning from.
2. Do not prefer an example for being longer, more confident, more formal or more flattering. Do not prefer an example because it comes first or second.
3. If the two are about equally useful, or you cannot tell, answer tie.

## Output

Return only JSON:

```json
{"verdict": "1", "reason": "one sentence"}
```

`verdict` is exactly `1`, `2` or `tie`.
