# Shared gate sets

- `general.jsonl`: 200 multiple-choice items scored by log-likelihood (general-skills gate). Built by
  `build_general.py`; see `PROVENANCE.md` for sources, revisions and sampling.
- `collapse.jsonl`: 60 prompts outside any one field, judged for being dragged into the person's field.
  The gate drops prompts whose `domain` matches the person's field.
- `anchors.jsonl`: 32 hand-written multiple-choice items (written for this repo, no outside source). The
  training job scores them before and after the dream block. No gate scores them, and `tests/test_sets.py`
  holds them disjoint from every gate item, so training never sees one.
- `knowledge_film_founder.jsonl`: 73 forced-choice items built from the film_founder persona's visible
  statements only (`knowledge.py`: values, contrasts and quotes swapped, no model call), frozen by its sha-256.
  `tools/knowledge_score.py` scores it after the runs on the base model and each arm's nightly adapters; no
  gate scores it and training never reads it.

## Attribution and licences

- 80 items from MMLU (`cais/mmlu`, MIT licence).
- 60 items from HellaSwag (`Rowan/hellaswag`, MIT licence).
- 60 items from the AI2 Reasoning Challenge, ARC-Challenge (`allenai/ai2_arc`), by the Allen Institute for
  AI, licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). The items are included
  unmodified apart from whitespace trimming; this set is shared under the same licence for those items.
