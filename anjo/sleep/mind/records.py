"""MIND's records as the lines arm 4M is served: what she found and what she noticed, ids kept.

Pure: a folded ``Workspace`` in, lines out. The workspace stays the one copy of her records; this
only selects their text for the probe arm's context, so a person whose MIND left none contributes
nothing and 4M's context is arm 4's, byte for byte (``probes.arms``). Nothing here writes, calls a
model or reads a clock: it serves records already stored, never a new finding.
"""
from __future__ import annotations

from typing import Final

from anjo.sleep.mind.inquiry.types import Workspace

NOTES_MAX: Final = 24  # lines one arm's context may carry: a night's records, not a transcript


def notes(workspace: Workspace) -> tuple[str, ...]:
    """MIND's newest findings, then the questions it noticed that have no finding yet, one line
    each: a finding with its kind, a noticed question as such. A finding updated in place moves to
    the end of ``workspace.findings`` (``kernel._with_finding``), so the last ``NOTES_MAX`` are the
    newest; keeping the front would drop them once a workspace outgrew one night's notes. Questions
    fill only the room the findings leave, in the order the workspace holds them."""
    reported = frozenset(finding.question_id for finding in workspace.findings)
    findings = workspace.findings[-NOTES_MAX:]
    lines = [f"- [{finding.id}] ({finding.kind}) {finding.statement}" for finding in findings]
    noticed = [
        f"- [{question.id}] (noticed) {question.subject}"
        for question in workspace.questions
        if question.id not in reported
    ]
    lines += noticed[:NOTES_MAX - len(lines)]
    return tuple(lines)
