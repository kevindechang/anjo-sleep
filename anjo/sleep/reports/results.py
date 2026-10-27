"""The results table (Task 12a): every arm per night, general skills and safety.

It reads a person's results tree, ``<root>/<person>/``:

- ``nights/<date>/summary.json``: the night (its gates, status, cost), stamped by the accelerated
  runner with its label (``development`` or ``final``).
- ``scores/night-NN.json``: ``{"night": "<date>", "arms": {"<arm>": {"<dimension>": score}}}``, the
  six arms of §5.1 scored on the night's checkpoint.

Development and final results are printed in separate sections and never mixed. A night whose
summary carries no label is refused: an unlabeled result could be read as the other kind. The root
defaults to ``$ANJO_DATA_DIR/sleep/results``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

LABELS: Final = ("development", "final")
ARMS: Final = ("1", "2C", "2F", "3", "3x", "4")
DIMENSIONS: Final = ("knowledge", "interaction")  # shown first, in this order; others follow alphabetically
DATA_DIR_ENV: Final = "ANJO_DATA_DIR"
MISSING: Final = "-"


class ResultsError(ValueError):
    """A results file is missing a field the report needs."""


@dataclass(frozen=True)
class Night:
    person: str
    night: str
    label: str
    status: str
    cost_usd: float
    gates: Mapping[str, Mapping[str, Any]]
    arms: Mapping[str, Mapping[str, float]]


def results_root(root: Path | None = None) -> Path:
    if root is not None:
        return root
    data = os.environ.get(DATA_DIR_ENV, "").strip()
    if not data:
        raise ResultsError(f"{DATA_DIR_ENV} names the data dir the results live in")
    return Path(data) / "sleep" / "results"


def _json(path: Path) -> dict[str, Any]:
    found = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(found, dict):
        raise ResultsError(f"{path.name} is not a record")
    return found


def _scores(folder: Path) -> dict[str, Mapping[str, Mapping[str, float]]]:
    found: dict[str, Mapping[str, Mapping[str, float]]] = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("night-*.json")):
        record = _json(path)
        if "night" not in record or not isinstance(record.get("arms"), dict):
            raise ResultsError(f"{path.name} lacks a night and its arms")
        found[str(record["night"])] = {
            str(arm): {str(d): float(v) for d, v in dims.items()} for arm, dims in record["arms"].items()
        }
    return found


def load(person: str, root: Path | None = None) -> list[Night]:
    """Every finished night of ``person``, in date order, with its scores when it has them."""
    base = results_root(root) / person
    scores = _scores(base / "scores")
    nights = []
    for summary in sorted((base / "nights").glob("*/summary.json")):
        record = _json(summary)
        label = record.get("label")
        if label not in LABELS:
            raise ResultsError(f"{person} {summary.parent.name}: the night has no development or final label")
        night = str(record.get("night", summary.parent.name))
        nights.append(Night(
            person=person, night=night, label=str(label), status=str(record.get("status", "?")),
            cost_usd=float(record.get("cost_usd", 0.0)), gates=record.get("gates") or {},
            arms=scores.get(night, {}),
        ))
    return sorted(nights, key=lambda n: n.night)


def dimensions(nights: Sequence[Night]) -> tuple[str, ...]:
    seen = {d for n in nights for dims in n.arms.values() for d in dims}
    return (*(d for d in DIMENSIONS if d in seen), *sorted(seen - set(DIMENSIONS)))


def cell(night: Night, arm: str, dims: Sequence[str]) -> str:
    scored = night.arms.get(arm)
    if not scored:
        return MISSING
    return "/".join(f"{scored[d]:.2f}" if d in scored else MISSING for d in dims) or MISSING


def gate_cell(night: Night, gate: str) -> str:
    found = night.gates.get(gate)
    if not found:
        return MISSING
    return f"{'ok' if found.get('passed') else 'FAIL'} {float(found.get('value', 0.0)):g}"


def _section(label: str, people: Sequence[str], loaded: Mapping[str, list[Night]]) -> list[str]:
    lines = [f"== {label.upper()} =="]
    for person in people:
        nights = [n for n in loaded[person] if n.label == label]
        if not nights:
            continue
        dims = dimensions(nights)
        head = ["night", "status", *(f"arm {a}" for a in ARMS), "general", "safety", "cost"]
        lines += ["", f"{person} (arm cells: {'/'.join(dims) or 'no scores yet'})", " | ".join(head)]
        for n in nights:
            row = [n.night, n.status, *(cell(n, a, dims) for a in ARMS), gate_cell(n, "general"),
                   gate_cell(n, "safety"), f"${n.cost_usd:.2f}"]
            lines.append(" | ".join(row))
    return lines


def table(people: Sequence[str], *, root: Path | None = None) -> str:
    """Every arm per night, general skills and safety, development apart from final."""
    loaded = {person: load(person, root) for person in people}
    empty = [p for p in people if not loaded[p]]
    out: list[str] = []
    for label in LABELS:
        if any(n.label == label for p in people for n in loaded[p]):
            out += [*_section(label, people, loaded), ""]
    out += [f"{p}: no nights" for p in empty]
    return "\n".join(out).rstrip() + "\n"
