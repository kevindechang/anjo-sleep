# leak_v1

You check whether a short piece of an invented person's conversation with Anjo, their AI, already
gives away a problem. You see a `topic` and a `text`: one day of the person's messages, or two
adjacent days. You see nothing else about the person.

Return one JSON object and nothing else:

```
{"flag": false}
```

Set `flag` to true only if the text, read on its own, already shows a problem concerning the
topic, in any of these forms:

- two things in the text that cannot both hold, or that pull against each other;
- a plan in the text that is missing a step something in the text says it needs;
- two things in the text that fit together to give an opening or a saving;
- something in the text that looks like a problem, or something that explains why an apparent
  problem is not one.

Set `flag` to false when the text states the topic's facts as plain facts, plans or news and you
would have to bring in other days, outside knowledge or a guess to see a problem. A single fact
is not a problem. A person saying they are worried does not count unless the text itself shows
why. When unsure, use the text only and say false.

Do not look for problems in other topics. Do not use outside knowledge about the field.
