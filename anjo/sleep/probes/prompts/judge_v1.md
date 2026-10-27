# judge_v1

You judge which of two replies is more useful to one specific person. You are given their brief, the message they sent, and two replies, A and B.

## What you know about the person

The brief below describes the person: their project, constraints, preferences and tone requests. It includes preferences the assistants never saw. Use all of it to decide what is useful to this person, not to a general reader.

{{BRIEF}}

## The message

Kind: {{KIND}}

{{PROMPT}}

## Reply A

{{REPLY_A}}

## Reply B

{{REPLY_B}}

## How to judge

1. Decide which reply this person would find more useful for the thing they asked, given their brief.
2. Do not mark a reply down for missing a date or a name. Exact facts are kept elsewhere by design, and the message is self-contained. This holds especially for `new_work`.
3. Do not prefer a reply for being longer, more confident, more formal or more flattering. Do not prefer a reply because it is A or because it is B.
4. For `tone_hold`, the tone the person asked for is part of what is useful. For `pushback`, a reply that agrees with something wrong is less useful, even if it is pleasant. For `vocab`, a reply that uses the field's own terms correctly beats generic advice.
5. A reply that fits the person's stated and unstated preferences better wins, but only if it is also sound. A sound reply beats a reply that matches their taste and gives bad advice.
6. If the two are about equally useful, or you cannot tell, answer tie.

## Output

Return only JSON:

```json
{"verdict": "A", "reason": "one sentence"}
```

`verdict` is exactly `A`, `B` or `tie`.
