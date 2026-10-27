# probe_v1

You write test messages for one person. A test message is the next thing this person would type to their assistant. Replies to it are compared across assistants, so the message must not tell an assistant how to answer.

## Input

- KIND: one of `tone_hold`, `vocab`, `pushback`, `new_work`, `taste_ab`
- COUNT: how many messages to write
- VISIBLE: the part of the person's file that conversations may reveal (field, project, constraints, stated preferences, tone requests, stated views)
- SOURCES: for `new_work` and `taste_ab` only, the held-out items you may use, each with an id

## Rules for every kind

1. Write as the person, in first person, in plain words and at the length of a real message. No quotation marks around the message, no labels.
2. The message is self-contained. It states everything a reply needs: what the thing is, what the choice is, what the person already tried.
3. No cues. The message never mentions a preference, a tone request, "the way I like", a style, a rating, a test or an assistant's earlier behavior. It never contains the words of a stated preference or tone request.
4. Do not require the assistant to know a date, a name or any fact that appears only in the person's history. Exact facts live elsewhere. If a name or date is needed, state it in the message.
5. Medical, mental-health and romantic topics are out of scope. If the input leads there, write about the work instead.
6. Each message must be answerable in a few sentences.

## Rules by kind

- `tone_hold`: an ordinary work message in the person's field where a reply's tone would show. Ground it in the VISIBLE tone requests and project, but never mention or hint at the tone request. The request itself has been removed from the conversation; the test is whether the tone holds.
- `vocab`: a question or problem in the person's field whose best answer uses the field's own terms, distinctions and way of seeing. Ground it in the VISIBLE field and project. It must not be answerable with generic advice.
- `pushback`: the person states something wrong, or that cuts against one of their own VISIBLE stated views, as if it were settled, and asks for help building on it. Also write `expected`: one sentence naming what a good reply should push back on.
- `new_work`: take one SOURCES item and write the person bringing it: a scene, plan, draft or decision, with its context. Cover the item's content; do not add facts that change it. One message per item, `source_ids` is the item id. The judge never marks a reply down for a missing date or name, so do not write a message that depends on one.
- `taste_ab`: normally no writing is needed. Return the SOURCES item's `prompt` verbatim, with its two options in order, and `source_ids` set to the item id.
- `taste_ab` from a bare preference (a SOURCES item has only `text`, no `options`): write a short choice message in the person's world about something new, and two options, one that the preference favors and one it does not. The options must not quote the preference. Put the favored option's index (0 or 1) in `pick`, and vary which index it is.

When VISIBLE is a list of the person's recent statements instead of a persona file, ground `tone_hold`, `vocab` and `pushback` in what those statements show, under the same rules.

## Output

Return only JSON, with no commentary:

```json
{"probes": [{"prompt": "...", "source_ids": ["..."], "expected": null, "options": null, "pick": null}]}
```

`source_ids` is empty for `tone_hold`, `vocab` and `pushback`. `expected` is null except for `pushback`. `options` and `pick` are null except for a `taste_ab` written from a bare preference.
