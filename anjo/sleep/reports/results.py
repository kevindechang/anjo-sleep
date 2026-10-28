"""The results table (Task 12a): every arm per night, general skills and safety.

It reads a person's results tree, ``<root>/<person>/``:

- ``nights/<date>/summary.json``: the night (its gates, status, cost), stamped by the accelerated
  runner with its label (``development`` or ``final``).
- ``scores/night-NN.json`` (``probes.morning``): ``{"night": "<date>", "arms": {"<arm>": {"<dimension>":
  stat}}}``, the six arms of §5.1 scored on the night's checkpoint. A ``stat`` is a plain score, or the
  full record the claim rules need: ``n``, ``wins``, ``ties``, ``losses``, ``score`` (tie-inclusive),
  ``decisive_share`` and the block-bootstrap ``ci`` ``[low, high]``. The table prints the CI beside it.
  The same file's ``pairs`` (``3_vs_3x``, ``4_vs_2C``, ...) are the headline comparisons, in the same
  shape; the table lists each under its night with its score, CI and decisive share. A file run on a
  subset of the probes names it (``subset``, ``probes.morning.DAILY``); each night carries its probe
  set (``probes``: the subset's name, or ``full``), and the table and the chart keep the sets apart,
  never one line across two sets.

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
OVERALL: Final = "overall"  # the all-dimensions stat in a score file; the table shows the dimensions
FULL_SET: Final = "full"  # a night scored on every sealed probe


class ResultsError(ValueError):
    """A results file is missing a field the report needs."""


@dataclass(frozen=True)
class Stat:
    """One arm's result in one dimension. Only ``score`` is always known."""

    score: float
    n: int | None = None
    wins: int | None = None
    ties: int | None = None
    losses: int | None = None
    decisive_share: float | None = None
    ci: tuple[float, float] | None = None


def _stat(raw: Any) -> Stat:
    if not isinstance(raw, dict):
        return Stat(float(raw))
    ci = raw.get("ci")
    return Stat(
        score=float(raw["score"]), n=raw.get("n"), wins=raw.get("wins"), ties=raw.get("ties"),
        losses=raw.get("losses"), decisive_share=raw.get("decisive_share"),
        ci=None if ci is None else (float(ci[0]), float(ci[1])),
    )


@dataclass(frozen=True)
class Night:
    person: str
    night: str
    label: str
    status: str
    cost_usd: float
    gates: Mapping[str, Mapping[str, Any]]
    arms: Mapping[str, Mapping[str, Stat]]
    pairs: Mapping[str, Mapping[str, Stat]]
    probes: str = FULL_SET  # the probe set its scores come from: ``full`` or a subset's name


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


def _stats(raw: Any) -> dict[str, dict[str, Stat]]:
    return {str(name): {str(d): _stat(v) for d, v in dims.items()} for name, dims in raw.items()}


Scored = tuple[dict[str, dict[str, Stat]], dict[str, dict[str, Stat]], str]


def _scores(folder: Path) -> dict[str, Scored]:
    """Each scored night's arms, headline pairs and probe set."""
    found: dict[str, Scored] = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("night-*.json")):
        record = _json(path)
        if "night" not in record or not isinstance(record.get("arms"), dict):
            raise ResultsError(f"{path.name} lacks a night and its arms")
        found[str(record["night"])] = (_stats(record["arms"]), _stats(record.get("pairs") or {}),
                                       str(record.get("subset") or FULL_SET))
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
        arms, pairs, probes = scores.get(night, ({}, {}, FULL_SET))
        nights.append(Night(
            person=person, night=night, label=str(label), status=str(record.get("status", "?")),
            cost_usd=float(record.get("cost_usd", 0.0)), gates=record.get("gates") or {},
            arms=arms, pairs=pairs, probes=probes,
        ))
    return sorted(nights, key=lambda n: n.night)


def dimensions(nights: Sequence[Night]) -> tuple[str, ...]:
    seen = {d for n in nights for dims in n.arms.values() for d in dims} - {OVERALL}
    return (*(d for d in DIMENSIONS if d in seen), *sorted(seen - set(DIMENSIONS)))


def cell(night: Night, arm: str, dims: Sequence[str]) -> str:
    scored = night.arms.get(arm)
    if not scored:
        return MISSING
    return "/".join(_shown(scored[d]) if d in scored else MISSING for d in dims) or MISSING


def pair_cell(stat: Stat) -> str:
    share = "n/a" if stat.decisive_share is None else f"{stat.decisive_share:.2f}"
    return f"{_shown(stat)} share {share}"


def _shown(stat: Stat) -> str:
    if stat.ci is None:
        return f"{stat.score:.2f}"
    return f"{stat.score:.2f} [{stat.ci[0]:.2f},{stat.ci[1]:.2f}]"


def gate_cell(night: Night, gate: str) -> str:
    found = night.gates.get(gate)
    if not found:
        return MISSING
    if found.get("skipped"):
        return "skipped"  # switched off: judged nothing, never passed
    return f"{'ok' if found.get('passed') else 'FAIL'} {float(found.get('value', 0.0)):g}"


def _section(label: str, people: Sequence[str], loaded: Mapping[str, list[Night]]) -> list[str]:
    lines = [f"== {label.upper()} =="]
    for person in people:
        nights = [n for n in loaded[person] if n.label == label]
        if not nights:
            continue
        dims = dimensions(nights)
        head = ["night", "status", "probes", *(f"arm {a}" for a in ARMS), "general", "safety", "cost"]
        lines += ["", f"{person} (arm cells: {'/'.join(dims) or 'no scores yet'})", " | ".join(head)]
        for n in nights:
            row = [n.night, n.status, n.probes, *(cell(n, a, dims) for a in ARMS), gate_cell(n, "general"),
                   gate_cell(n, "safety"), f"${n.cost_usd:.2f}"]
            lines.append(" | ".join(row))
        for n in nights:
            for name, stats in sorted(n.pairs.items()):
                shown = "; ".join(f"{d}: {pair_cell(stats[d])}" for d in (*dims, OVERALL) if d in stats)
                lines.append(f"{n.night} | pair {name} | {shown}")
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
