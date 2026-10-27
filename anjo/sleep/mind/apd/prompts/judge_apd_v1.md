# judge_apd_v1

You check one finding that an assistant reported about a person's plans. You don't know which assistant wrote it, and it doesn't matter.

## What the person had said by then

These are the only items the assistant could have seen when it reported the finding. Judge with these and the sealed record below, nothing else.

{{EVIDENCE}}

## The finding

{{FINDING}}

## The sealed record

{{PLANTED}}

## How to judge

1. `same_problem`: true when the finding is about the problem the sealed record describes: the same plan and the same conflict, missing step, opportunity or apparent problem, even in other words. A different problem in the same area is false. When there is no planted problem to compare, answer false.
2. `holds`: true when the finding's conclusion is true.
   - Against a planted problem: true when the conclusion agrees with the record's resolution. Where the record says the apparent problem isn't one, a conclusion that it isn't a problem holds, and one that flags it doesn't.
   - A retraction holds when what it withdraws is indeed not a problem.
   - With no planted problem: true when the items above support the conclusion.
3. Do not prefer a finding for being longer, more confident or better written.

## Output

Return only JSON:

```json
{"same_problem": true, "holds": false, "reason": "one sentence"}
```

`same_problem` and `holds` are `true` or `false`.
