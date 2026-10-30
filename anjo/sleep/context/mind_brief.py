"""Bounded, extractive served-brief refresh when a model fold strands MIND homework.

Only complete homework items are folded. A backlog of other evidence stays pending for the
normal compiler; it cannot crowd out the new finding. No model calls or ancestry changes here.
"""
from __future__ import annotations

from collections.abc import Sequence

from anjo.sleep.context.brief import EMPTY_BRIEF, SECTIONS, BriefError, _check, _line
from anjo.sleep.context.recall_common import CHARS_PER_TOKEN
from anjo.sleep.types import Evidence


def refresh(
    prev: str | None, homework: Sequence[Evidence], budget_tokens: int,
) -> tuple[str, tuple[Evidence, ...]]:
    """Keep complete new homework with ids and sources, then fit the previous sections."""
    # EMPTY_BRIEF includes all headings, separators and a 'none' placeholder per section.
    room = budget_tokens * CHARS_PER_TOKEN - len(EMPTY_BRIEF)
    included: list[Evidence] = []
    lines: list[str] = []
    for item in homework:
        line = _line(item)
        cost = len(line) + 1  # its newline, including between homework and old text
        if cost <= room:
            included.append(item)
            lines.append(line)
            room -= cost
    if not included:
        raise BriefError("new MIND homework cannot fit in the brief budget")
    sections: list[str] = []
    for i, heading in enumerate(SECTIONS):
        old = ""
        if prev:
            end = prev.find(SECTIONS[i + 1]) if i + 1 < len(SECTIONS) else len(prev)
            old = prev[prev.index(heading) + len(heading):end].strip()
            limit = max(0, room // len(SECTIONS))
            if len(old) > limit:
                old = old[:max(0, limit - 1)] + "…" if limit else ""
        body = [*lines, old] if i == 3 else [old]
        sections.append(heading + "\n" + ("\n".join(s for s in body if s) or "none"))
    return _check("\n\n".join(sections), budget_tokens), tuple(included)
