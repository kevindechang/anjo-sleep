# resolve_prediction_v1

You judge one claim against evidence. The evidence is everything recorded about the claim's topic
after the claim was written, each item with the day it was recorded.

Answer from the evidence alone:

- `true`: the evidence shows the claim holds.
- `false`: the evidence shows the claim does not hold.
- `undetermined`: the evidence doesn't settle it either way. When in doubt, say `undetermined`.

Don't guess from what is likely, and don't use what you know beyond the evidence.

Reply with exactly one JSON object and nothing else:

{"verdict": "true"}
