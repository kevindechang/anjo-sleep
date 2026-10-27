# identify_v1

A reader looked at part of a made-up person's life and wrote a `statement` about what it found.
You see that statement and the sealed `item` it was meant to find: the item's family, topic,
scheduled facts and resolution.

Return one JSON object and nothing else: `{"match": true}` or `{"match": false}`.

Set `match` to true only if the statement names the same problem, opening or apparent problem as
the item, built from the item's own facts. A statement that is vague, names a different issue,
names only part of the item, or only restates the topic is false. For a reversal or a decoy, the
statement must identify the apparent problem; it need not state the resolution.
