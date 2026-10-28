"""The night log (Task 5, step 11): one folder per person and night, holding ids, counts, times and
verdicts, never a person's words.

- ``log.jsonl``: one record per step, written before the next step starts, so a night cut short
  still shows how far it got. Every record says whether the step is new Sleep code or an existing
  Anjo system (``uses``), names that system, and says when its port is a no-op.
- ``summary.json``: the ``NightResult``, serialized field by field (its mappings are read-only, so
  ``dataclasses.asdict`` can't copy it), with the night's receipts. Written once, atomically.
- ``mind_points.jsonl``: a synthetic life's two decision points (Task 16, ``mind.points``), ids,
  numbers and times only.
- ``story.json``: the night as one loop (``night/story.py``), told from ``log.jsonl`` when the night
  finishes and told again after the morning; rewritten whole, atomically, each time.

A night run again on the same date gets its own folder (``<date>.2``, ``<date>.3``), so no log is
ever overwritten. A log whose folder was removed (the person deleted their data, Task 9) writes
nothing and creates nothing again. For a synthetic person the folders live under the results tree that is
published; for a real person, outside it.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal

from anjo.sleep.checkpoints import ref_record
from anjo.sleep.night.story import StoryError, tell
from anjo.sleep.types import Choice, GateResult, NightResult, NightStatus

LOG: Final = "log.jsonl"
SUMMARY: Final = "summary.json"
POINTS: Final = "mind_points.jsonl"
STORY: Final = "story.json"
Uses = Literal["new", "existing"]
StepStatus = Literal["ok", "skipped", "failed"]
logger = logging.getLogger(__name__)


def night_folder(root: Path, night: date) -> Path:
    """A fresh folder for this night's log: ``<date>``, or ``<date>.<n>`` on a rerun."""
    base = night.isoformat()
    folder, attempt = root / base, 1
    while folder.exists():
        attempt += 1
        folder = root / f"{base}.{attempt}"
    folder.mkdir(parents=True)
    return folder


def gate_record(result: GateResult) -> dict[str, Any]:
    """A gate's record; a switched-off gate's says ``skipped`` (a gate that ran writes no such key)."""
    return {
        "gate": result.gate, "passed": result.passed, "value": result.value,
        "threshold": result.threshold, "reason": result.reason, **({"skipped": True} if result.skipped else {}),
    }


def choice_record(choice: Choice) -> dict[str, Any]:
    return {
        "opportunity": choice.opportunity, "action": choice.action, "target_id": choice.target_id,
        "question_id": choice.question_id,
        "considered": [
            {"want_id": c.want_id, "kind": c.kind, "target_id": c.target_id, "action": c.action,
             "strength": c.strength, "weight": c.weight, "status": c.status, "reason": c.reason}
            for c in choice.considered
        ],
    }


def result_record(result: NightResult) -> dict[str, Any]:
    """``NightResult`` field by field."""
    return {
        "status": result.status, "failed_at": result.failed_at,
        "checkpoint": None if result.checkpoint is None else ref_record(result.checkpoint),
        "gates": {name: gate_record(gate) for name, gate in result.gates.items()},
        "choices": [choice_record(choice) for choice in result.choices],
        "cost_usd": result.cost_usd, "cost_parts": dict(result.cost_parts),
        "log_path": str(result.log_path),
    }


def _write_line(path: Path, record: Mapping[str, Any]) -> None:
    if not path.parent.is_dir():
        return  # the folder was removed by a delete (RF2): nothing is written after it
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class NightLog:
    """One night's log folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    @property
    def path(self) -> Path:
        return self.folder / LOG

    def step(
        self, step: str, status: StepStatus, *, uses: Uses, system: str | None, noop: bool = False,
        at: datetime | None = None, **receipt: Any,
    ) -> None:
        if (uses == "existing") != (system is not None):
            raise ValueError("an existing step names the Anjo system it calls, and only then")
        record = {"step": step, "status": status, "uses": uses, "system": system, "noop": noop, **receipt}
        if at is not None:
            record["at"] = at.isoformat()
        _write_line(self.path, record)

    def steps(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def point(self, record: Mapping[str, Any]) -> None:
        _write_line(self.folder / POINTS, record)

    def points(self) -> list[dict[str, Any]]:
        path = self.folder / POINTS
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def finish(self, result: NightResult, receipts: Mapping[str, Any], *, night: date | None = None) -> None:
        """The summary; given the ``night``, its story too, told from this log as it stands."""
        self._replace(SUMMARY, json.dumps({**result_record(result), **receipts}, sort_keys=True, indent=1, default=str))
        if night is not None:
            self.tell(night, result.status)

    def tell(self, night: date, status: NightStatus) -> None:
        """The night as one loop (``story.tell``) from this log's records, replacing the one told before.
        A log the story can't be told from is logged and leaves the story as it was: the night's own
        record never fails it, and the page says its story is unavailable."""
        try:
            record = tell(night, status, self.steps()).record()
        except StoryError as error:
            logger.warning("the story of night %s can't be told from its log: %s", night.isoformat(), error)
            return
        self._replace(STORY, json.dumps(record, indent=1))

    def _replace(self, name: str, text: str) -> None:
        if not self.folder.is_dir():
            return  # removed by a delete (RF2)
        pending = self.folder / f".{name}.tmp"
        pending.write_text(text, encoding="utf-8")
        os.replace(pending, self.folder / name)


def summaries(root: Path) -> list[dict[str, Any]]:
    """Every finished night's summary under ``root``, in folder order."""
    if not root.is_dir():
        return []
    found = []
    for folder in sorted(root.iterdir()):
        summary = folder / SUMMARY
        if summary.is_file():
            found.append(json.loads(summary.read_text(encoding="utf-8")))
    return found
