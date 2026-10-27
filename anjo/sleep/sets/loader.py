"""Loaders and validation for the two shared gate sets.

general.jsonl  200 multiple-choice (4-way, one 3-way) items with provenance (log-likelihood scored)
collapse.jsonl 60 prompts outside the person's field (judged for field drag)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SETS_DIR = Path(__file__).parent
GENERAL_ITEMS = 200
COLLAPSE_PROMPTS = 60
GENERAL_SOURCES = frozenset({"mmlu", "arc_challenge", "hellaswag"})
MIN_CHOICES = 3  # one ARC-Challenge item (g129) genuinely has three
MAX_CHOICES = 4


class SetError(ValueError):
    """A shared set broke a rule."""


@dataclass(frozen=True)
class GeneralItem:
    id: str
    source: str
    subject: str
    question: str
    choices: tuple[str, ...]
    answer: int


@dataclass(frozen=True)
class CollapsePrompt:
    id: str
    domain: str
    prompt: str


def _rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise SetError(f"{path.name} line {n}: not JSON") from exc
    return rows


def _nonempty(row: dict[str, object], key: str, where: str) -> str:
    v = row.get(key)
    if not isinstance(v, str) or not v.strip():
        raise SetError(f"{where}: '{key}' must be non-empty text")
    return v


def load_general(
    path: str | Path | None = None, *, expected: int | None = GENERAL_ITEMS
) -> tuple[GeneralItem, ...]:
    p = Path(path or SETS_DIR / "general.jsonl")
    items = []
    for row in _rows(p):
        where = str(row.get("id"))
        choices = row.get("choices")
        answer = row.get("answer")
        if not isinstance(choices, list) or not MIN_CHOICES <= len(choices) <= MAX_CHOICES:
            raise SetError(f"{where}: needs {MIN_CHOICES} to {MAX_CHOICES} choices")
        if not all(isinstance(c, str) and c.strip() for c in choices):
            raise SetError(f"{where}: empty choice")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices):
            raise SetError(f"{where}: answer must index a choice")
        source = _nonempty(row, "source", where)
        if source not in GENERAL_SOURCES:
            raise SetError(f"{where}: unknown source '{source}'")
        items.append(
            GeneralItem(
                _nonempty(row, "id", where),
                source,
                _nonempty(row, "subject", where),
                _nonempty(row, "question", where),
                tuple(choices),
                answer,
            )
        )
    _check_ids([i.id for i in items], expected, "general")
    return tuple(items)


def load_collapse(
    path: str | Path | None = None, *, expected: int | None = COLLAPSE_PROMPTS
) -> tuple[CollapsePrompt, ...]:
    p = Path(path or SETS_DIR / "collapse.jsonl")
    prompts = [
        CollapsePrompt(
            _nonempty(r, "id", "collapse"),
            _nonempty(r, "domain", str(r.get("id"))),
            _nonempty(r, "prompt", str(r.get("id"))),
        )
        for r in _rows(p)
    ]
    _check_ids([c.id for c in prompts], expected, "collapse")
    return tuple(prompts)


def collapse_for(
    field_domains: tuple[str, ...], prompts: tuple[CollapsePrompt, ...]
) -> tuple[CollapsePrompt, ...]:
    """The gate excludes prompts whose domain is one of the person's `field_domains`."""
    tags = {d.strip().lower() for d in field_domains}
    return tuple(c for c in prompts if c.domain.strip().lower() not in tags)


def _check_ids(ids: list[str], expected: int | None, name: str) -> None:
    if len(set(ids)) != len(ids):
        raise SetError(f"{name}: duplicate ids")
    if expected is not None and len(ids) != expected:
        raise SetError(f"{name}: expected {expected} items, got {len(ids)}")
