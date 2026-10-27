# PROVENANCE: general.jsonl

200 four-way multiple-choice items for log-likelihood scoring (general knowledge and commonsense check).
Built by `build_general.py` (stdlib only) from public Hugging Face datasets via the datasets-server rows API.

## Sources

| Source tag | Dataset | Config | Split | Hub revision at build | Pool | Sampled |
|---|---|---|---|---|---|---|
| `mmlu` | `cais/mmlu` | `all` | `test` | `c30699e8356da336a370243923dbaf21066bb9fe` | full split (14,042 rows) | 80 |
| `arc_challenge` | `allenai/ai2_arc` | `ARC-Challenge` | `test` | `210d026faf9955653af8916fad021475a3f00453` | full split (1,172 rows) | 60 |
| `hellaswag` | `Rowan/hellaswag` | `default` | `validation` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | first 1,000 rows of validation (has labels; test does not) | 60 |

Revisions are the Hub commit SHAs read on 2026-10-27. The rows API does not take a revision, so a rerun
uses whatever the API serves. If a dataset changes upstream, the set changes too.

## Sampling

- Seed: `random.Random(20261027)`, one RNG for all three sources in the order MMLU, ARC, HellaSwag.
- `rng.sample` picks each source's items from its pool. Then `rng.shuffle` interleaves the 200 items.
- Ids are assigned `g001`..`g200` after the shuffle.
- The MMLU pool is the full split, not the first 1,000 rows. The first 1,000 rows cover only about 9 subjects,
  so they cannot meet the "spread across subjects" requirement. The 80 sampled items cover 37 subjects.
- The ARC `subject` field is the constant `science`, because ARC has no subject column.
- HellaSwag `subject` is the `activity_label` (the category).

## Processing

- All text is stripped of leading and trailing whitespace.
- HellaSwag: the question is the `ctx` field. The light cleanup follows the lm-evaluation-harness HellaSwag
  task. It replaces ` [title]` with `. `, removes `[...]` spans, and collapses double spaces. This applies to
  the context and the endings.
- ARC: answer keys `1`..`4` (and `5`) are mapped to `A`..`E`. The answer index is the position of the key in the
  choice labels, so ARC choice order is kept as published.
- HellaSwag labels are strings `"0"`..`"3"`, converted to int.
- MMLU answers are already int 0..3.

## Licences (checked on each dataset card, 2026-10-27)

| Dataset | Licence | Where it is stated | Note |
|---|---|---|---|
| `cais/mmlu` | MIT | YAML `license: mit`; body links the MIT licence in `hendrycks/test` | No difference. |
| `allenai/ai2_arc` | CC BY-SA 4.0 | YAML `license: cc-by-sa-4.0` | **ShareAlike.** See the action item below. |
| `Rowan/hellaswag` | MIT | Body text ("MIT", linking `rowanz/hellaswag/LICENSE`); the YAML front matter has **no `license` field** | Difference: the licence is only in the body text. |

### Action item: ARC is CC BY-SA 4.0, not Apache-2.0

The 60 `arc_challenge` items are adapted from a CC BY-SA 4.0 dataset. ShareAlike requires that the adapted
material keep CC BY-SA 4.0. Apache-2.0 is not one of the licences CC lists as compatible with BY-SA 4.0. So
publishing these items inside an Apache-2.0 repo, under Apache-2.0, is not clean. Options, for the repo owner
to decide:

1. Keep the ARC items in a separate file under CC BY-SA 4.0, with its own licence notice. Or
2. Drop `arc_challenge` and sample 60 more items from MMLU or HellaSwag. This changes the set.

Until that is decided, do not treat `general.jsonl` as Apache-2.0 in full.

## Attribution text (for the repo NOTICE or README)

> The file `general.jsonl` contains items derived from the following datasets:
>
> - MMLU (`cais/mmlu`), MIT License. Hendrycks et al., "Measuring Massive Multitask Language Understanding",
>   ICLR 2021. https://github.com/hendrycks/test
> - HellaSwag (`Rowan/hellaswag`), MIT License. Zellers et al., "HellaSwag: Can a Machine Really Finish Your
>   Sentence?", ACL 2019. https://github.com/rowanz/hellaswag
> - AI2 ARC (`allenai/ai2_arc`), CC BY-SA 4.0. Clark et al., "Think you have Solved Question Answering? Try ARC,
>   the AI2 Reasoning Challenge", 2018. https://allenai.org/data/arc
>   The ARC-derived items are adapted material under CC BY-SA 4.0 (see the licence note above).
>
> Items were fetched from the Hugging Face Hub and processed as described in PROVENANCE.md.
