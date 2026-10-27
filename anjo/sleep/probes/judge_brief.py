"""The brief judges read (Task 2, spec 5.1): the person as they really are.

It includes the held-out preferences, which no arm ever sees, so only ``score`` (through its
judges) and the live page's brief panel may call it (RF5). The arms never import this module.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from anjo.sleep.personas.model import PersonFile, Statement, load_person_file
from anjo.sleep.types import Evidence

PERSONAS_DIR: Final = Path(__file__).parents[1] / "personas"
_ID: Final = re.compile(r"[A-Za-z0-9_-]+")
DATE_RULE: Final = (
    "Never mark a `new_work` reply down for a missing date or name: exact facts are kept "
    "elsewhere by design."
)


def _bullets(statements: Sequence[Statement]) -> str:
    return "\n".join(f"- {s.text}" for s in statements) or "- none"


def _from_file(pf: PersonFile) -> str:
    v, h = pf.visible, pf.held_out
    constraints = "\n".join(f"- {c}" for c in v.project.constraints) or "- none"
    return "\n\n".join((
        f"{v.name}, {v.age}. Field: {v.field}. {v.line}",
        f"## Their project\n{v.project.title}: {v.project.logline}\nConstraints:\n{constraints}",
        f"## Tone they asked for\n{_bullets(v.tone_requests)}",
        f"## Stated views\n{_bullets(v.stated_views)}",
        f"## Preferences the assistants saw\n{_bullets(v.preferences)}",
        f"## Preferences no assistant saw\n{_bullets(h.preferences)}",
        DATE_RULE,
    ))


def _from_evidence(items: Sequence[Evidence]) -> str:
    lines = "\n".join(f"- ({item.kind}) {item.text}" for item in items) or "- none"
    return f"## Everything they have told Anjo\n{lines}\n\n{DATE_RULE}"


def judge_brief(
    person_id: str,
    files: Mapping[str, PersonFile] | None = None,
    *,
    evidence: Callable[[str], Sequence[Evidence]] | None = None,
) -> str:
    """The judge's brief for ``person_id``: from their persona or life file, else (Kevin) from
    all their evidence, held-out items included. ``files`` replaces the registered persona files."""
    if files is not None and person_id in files:
        return _from_file(files[person_id])
    if files is None and _ID.fullmatch(person_id) and (PERSONAS_DIR / f"{person_id}.json").is_file():
        return _from_file(load_person_file(PERSONAS_DIR / f"{person_id}.json"))
    if evidence is not None:
        return _from_evidence(evidence(person_id))
    raise LookupError(f"no persona file or evidence source for {person_id}")
