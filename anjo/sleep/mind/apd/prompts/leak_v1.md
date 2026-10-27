# leak_v1

You check whether a short piece of an invented person's life already gives away a planted item.
You see a `topic`, the item's `family`, a `view` and the `days` of content: one day, or two
adjacent days. The view is either `sessions` (the person's own messages) or `evidence` (short
notes and plan labels that were taken from those messages). You see nothing else about the person.

Return one JSON object and nothing else:

```
{"flag": false}
```

The item is leaked only when the content you are shown contains the whole item, by its family:

- `contradiction`: both halves, two things about the topic that cannot both hold.
- `gap`: the plan, and the step it needs, and nothing showing the step is done or covered.
- `opportunity`: both facts that fit together to give an opening or a saving.
- `reversal`: the apparent problem, together with the fact that dissolves it.
- `decoy`: the apparent problem, together with the fact that explains it.

Set `flag` to true only when the content you are shown holds all of those parts for the topic.
One part alone is never a leak, however much it looks like a problem or an opening: a half, a
plan without its needed step, one of two facts, an apparent problem without its explanation, or
an explanation without the thing it explains. Content that states a conclusion about the topic
(that something is a problem, a conflict, a mistake, an opening, or is fine after all) is also a
leak, in any view, even if only one part is present.

Use only the content shown. Do not use outside knowledge of the field, and do not look for
problems on other topics.
