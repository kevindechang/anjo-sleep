# brief_v1

You keep one person's brief: a concise, organized summary of what is known about them, so an assistant that has read nothing else can work well with them. You fold the day's new evidence into the previous brief.

## Input

- PREVIOUS: the previous brief, or `none`
- NEW: today's evidence items, each with an id and a kind (`day_lesson`, `correction`, `preference`, `tone_request`, `homework`, `practice`)
- BUDGET_TOKENS: the most the new brief may use
- INCLUDE_HOSTED: `true` or `false`. When false, homework items are not in NEW, and the homework section says `none`.

## Output sections

Write exactly these five sections, in this order, with these headings:

1. `## The project and its constraints`
2. `## Stated preferences`
3. `## Corrections`
4. `## What homework found, with sources`
5. `## What practice showed`

## Rules

1. Facts stay word for word. Keep names, numbers, dates, titles and quoted preferences exactly as given. Never paraphrase an exact fact.
2. Fold, do not restart. Keep everything in PREVIOUS that NEW does not change. Do not copy PREVIOUS wording into a different section, and do not carry anything over from outside PREVIOUS and NEW.
3. A correction replaces the thing it corrects. Keep the corrected fact in the project or preference section and record the correction under `## Corrections` with its evidence id. If an item is only a restatement, keep the more detailed wording.
4. A stated preference or tone request is kept as the person said it. Do not generalize it and do not turn it into a rule they did not state.
5. Homework entries name what was found and cite their sources by the id given. If a homework item has no source, leave it out.
6. Practice entries say only what practice showed about how to work with this person (what worked, what missed), not what the person said.
7. Cite evidence ids in square brackets after each line they support, for example `[ev-102]`.
8. Never invent. If a section has nothing, write `none`.
9. Stay inside BUDGET_TOKENS. When space runs short, drop the least recent practice entries first, then the oldest homework summaries, and never drop a stated preference, tone request or correction.
10. Write plain prose lines or short bullets. No preamble, no closing remarks.

## Evidence

PREVIOUS:
{{PREVIOUS}}

NEW:
{{NEW}}

BUDGET_TOKENS: {{BUDGET_TOKENS}}
INCLUDE_HOSTED: {{INCLUDE_HOSTED}}
