# relevance_v1

You label whether one topic is relevant to one piece of work a person brought up. You see the topic and the work, nothing else.

## The topic

{{TARGET}}

## The work

Project: {{PROJECT}}

{{ITEM}}

## How to label

1. The topic is relevant when knowing about it would directly help with this piece of work: the work is about the same thing, needs it as a step, or is a part of it.
2. Sharing only the broad field, the same project or a loose theme is not enough.
3. Judge only these two texts. If you cannot tell, the answer is `false`.

## Output

Return only JSON:

```json
{"relevant": true}
```

`relevant` is exactly `true` or `false`.
