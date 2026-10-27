# Shared gate sets

- `general.jsonl`: 200 multiple-choice items scored by log-likelihood (general-skills gate). Built by
  `build_general.py`; see `PROVENANCE.md` for sources, revisions and sampling.
- `collapse.jsonl`: 60 prompts outside any one field, judged for being dragged into the person's field.
  The gate drops prompts whose `domain` matches the person's field.

## Attribution and licences

- 80 items from MMLU (`cais/mmlu`, MIT licence).
- 60 items from HellaSwag (`Rowan/hellaswag`, MIT licence).
- 60 items from the AI2 Reasoning Challenge, ARC-Challenge (`allenai/ai2_arc`), by the Allen Institute for
  AI, licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). The items are included
  unmodified apart from whitespace trimming; this set is shared under the same licence for those items.
