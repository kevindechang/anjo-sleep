"""What changed (Task 9): plain lines from a person's night logs and probe score deltas, and the
night views the nights routes return.

A person's results folder holds ``nights/<date>[.n]/{log.jsonl, summary.json}`` (``night/nightlog.py``)
and ``scores/night-NN.json`` (``probes/morning.py``). Every line and view here is built from ids,
codes, counts and gate names. None carries a person's words, a target's label or a gate's failure
reason, which can name a safety case.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

SUMMARY: Final = "summary.json"
LOG: Final = "log.jsonl"
PERSONAL_ARM: Final = "3"  # her own weights, scored against the starting model (arm 1)
OVERALL: Final = "overall"
MIN_MOVE: Final = 0.01  # a score move smaller than this is not reported
Kind = Literal["promoted", "rejected", "failed", "nothing_new", "studied", "note", "score"]


class ChangesError(ValueError):
    """A night record that can't be read."""


@dataclass(frozen=True)
class Change:
    night: date
    kind: Kind
    line: str


@dataclass(frozen=True)
class StepView:
    step: str
    status: str
    uses: str
    system: str | None


@dataclass(frozen=True)
class NightView:
    night: date
    status: str
    failed_at: str | None
    checkpoint: str | None
    gates: tuple[tuple[str, bool], ...]
    steps: tuple[StepView, ...]
    action: str | None  # the chooser's action code, or None
    note: str | None  # the note step's outcome code, or None
    skipped: tuple[str, ...] = ()  # gates switched off that night: judged nothing, never passed


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ChangesError(f"{path.name} can't be read") from None


def _steps(folder: Path) -> list[Mapping[str, Any]]:
    path = folder / LOG
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError):
        raise ChangesError(f"{LOG} can't be read") from None


def _view(folder: Path) -> NightView | None:
    summary = folder / SUMMARY
    if not summary.is_file():
        return None  # a night still running, or cut short before it finished
    record = _json(summary)
    try:
        checkpoint = record.get("checkpoint")
        choices = record.get("choices") or []
        steps = _steps(folder)
        note = next((s.get("outcome") for s in steps if s.get("step") == "note"), None)
        action = next((s.get("action") for s in steps if s.get("step") == "choose"), None)
        if action is None and choices:
            action = choices[0].get("action")
        return NightView(
            night=date.fromisoformat(record["night"]), status=str(record["status"]),
            failed_at=record.get("failed_at"), checkpoint=None if checkpoint is None else str(checkpoint["id"]),
            gates=tuple((str(name), bool(gate["passed"]) and not gate.get("skipped"))
                        for name, gate in (record.get("gates") or {}).items()),
            steps=tuple(StepView(str(s["step"]), str(s["status"]), str(s["uses"]), s.get("system")) for s in steps),
            action=action, note=None if note is None else str(note),
            skipped=tuple(str(name) for name, gate in (record.get("gates") or {}).items() if gate.get("skipped")),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ChangesError(f"{folder.name}: not a night record") from None


def night_views(person_dir: Path) -> list[NightView]:
    """Every finished night under ``person_dir``, oldest first; a night run again is listed again."""
    root = person_dir / "nights"
    found = [_view(folder) for folder in sorted(root.iterdir())] if root.is_dir() else []
    return sorted((view for view in found if view is not None), key=lambda view: view.night)


def _night_lines(view: NightView) -> list[Change]:
    when = view.night.isoformat()
    lines: list[Change] = []
    if view.status == "promoted":
        lines.append(Change(view.night, "promoted", f"{when}: she learned from your day, and this new version is the one you talk to."))
    elif view.status == "rejected":
        failed = ", ".join(name for name, passed in view.gates if not passed and name not in view.skipped)
        lines.append(Change(view.night, "rejected", f"{when}: she trained overnight but held the update back ({failed or 'a check'} did not pass), so the earlier version still serves."))
    elif view.status == "nothing_new":
        lines.append(Change(view.night, "nothing_new", f"{when}: there was nothing new to learn from, so she changed nothing."))
    else:
        where = f" at {view.failed_at}" if view.failed_at else ""
        lines.append(Change(view.night, "failed", f"{when}: the night stopped{where}, so the earlier version still serves."))
    if view.action in ("prepare_commitment", "revisit_thread"):
        lines.append(Change(view.night, "studied", f"{when}: she chose to study one of your open threads."))
    if view.note == "held":
        lines.append(Change(view.night, "note", f"{when}: she left you a note."))
    elif view.note == "note_declined":
        lines.append(Change(view.night, "note", f"{when}: she considered a note and decided there was nothing worth saying."))
    return lines


def _score(raw: Any) -> tuple[float, int | None]:
    """A stat's score, and how many probes it counts when the file says."""
    if not isinstance(raw, dict):
        return float(raw), None
    n = raw.get("n")
    return float(raw["score"]), n if isinstance(n, int) and not isinstance(n, bool) else None


def _scores(person_dir: Path) -> list[tuple[date, str | None, Mapping[str, tuple[float, int | None]]]]:
    """Each scored night's probe subset (None: the full set) and personal-arm score and count per
    dimension, in night order."""
    folder = person_dir / "scores"
    found = []
    for path in sorted(folder.glob("night-*.json")) if folder.is_dir() else []:
        record = _json(path)
        try:
            arm = record["arms"].get(PERSONAL_ARM) or {}
            found.append((date.fromisoformat(record["night"]), record.get("subset"),
                          {name: _score(stat) for name, stat in arm.items()}))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ChangesError(f"{path.name}: not a score record") from None
    return sorted(found, key=lambda item: item[0])


def _score_lines(person_dir: Path, since: date) -> list[Change]:
    lines: list[Change] = []
    scored = _scores(person_dir)
    for (_, was, before), (night, now, after) in zip(scored, scored[1:], strict=False):
        if night < since or was != now:  # a score moves only against a night scored on the same probes
            continue
        for name in sorted(set(before) & set(after)):
            (was_score, _), (score, n) = before[name], after[name]
            move = score - was_score
            if abs(move) < MIN_MOVE or (now is not None and n is None):  # a subset's move always says its n
                continue
            label = "overall" if name == OVERALL else name
            way = "up" if move > 0 else "down"
            asked = "your own test questions" if n is None else f"{n} of your own test questions"
            lines.append(Change(night, "score", f"{night.isoformat()}: on {asked}, her {label} score against the starting model went {way} from {was_score:.2f} to {score:.2f}."))
    return lines


def what_changed(user_id: str, since: date, *, results: Callable[[str], Path]) -> list[Change]:
    """Plain lines for every night from ``since`` on, then the probe score moves, oldest first."""
    person_dir = results(user_id)
    lines = [line for view in night_views(person_dir) if view.night >= since for line in _night_lines(view)]
    return sorted([*lines, *_score_lines(person_dir, since)], key=lambda change: change.night)
