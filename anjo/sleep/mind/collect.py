"""Collect MIND life records (Task 16): replay five deciders on recorded development nights.

Reads ``mind_points.jsonl`` from each labeled night folder, replays D0–D4 through
``deciders.replay_point`` / ``night_picks``, attaches held-out ``new_work`` arrivals and frozen
relevance labels (never labeling during collect), and returns ``reports.mind.LifeRecord`` tuples.
Offline only: D1 goes through an injected ``ChatFn`` (``None`` leaves D1 unobserved).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Literal

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.evidence.targets import EventLogTargetStore
from anjo.sleep.mind import expect, points
from anjo.sleep.mind.deciders import night_picks, replay_point
from anjo.sleep.night.nightlog import NightLog, SUMMARY
from anjo.sleep.personas.model import PersonFile, load_person_file, to_visible_dict
from anjo.sleep.probes import relevance
from anjo.sleep.probes.relevance import FrozenLabels, PairKey
from anjo.sleep.reports import mind
from anjo.sleep.types import Expectation

Split = Literal["development", "held_out"]
LABELS_FILE: Final = "relevance.json"
TARGETS_FILE: Final = "targets.jsonl"
EVIDENCE_FILE: Final = "evidence.jsonl"
PERSONAS_DIR: Final = Path(__file__).parents[1] / "personas"
DEVELOPMENT_PERSONAS: Final = frozenset({"film_founder", "founder", "researcher", "film_student"})
DEVELOPMENT_LIVES: Final = (
    ("film_founder", "film_founder"),
    ("founder", "founder"),
    ("researcher", "researcher"),
    ("film_student", "film_student"),
)
_REAL_USER: Final = re.compile(r"^u-[0-9a-f]{4,}$", re.IGNORECASE)


class CollectRefused(PermissionError):
    """A life or account id this collector never reads."""


def collect_life(
    log_dir: Path,
    persona_file: str | Path,
    *,
    split: Split = "development",
    chat: ChatFn | None = None,
) -> mind.LifeRecord:
    """One life's replays from its night folders and persona file."""
    nights_dir, life_root, user_id = _layout(log_dir)
    persona = load_person_file(persona_file)
    life_id = persona.visible.person_id
    _refuse_ids(life_id, user_id)
    if split == "development" and life_id not in DEVELOPMENT_PERSONAS:
        raise CollectRefused(f"development collect uses registered personas, not {life_id!r}")
    labels = load_frozen_labels(life_root / LABELS_FILE)
    target_labels = _target_labels(life_root / TARGETS_FILE, user_id)
    expectations = _expectations(life_root / TARGETS_FILE, user_id)
    returns = _returns(life_root / EVIDENCE_FILE, user_id)
    folders = _collectible_folders(nights_dir, split)
    arrivals = _arrivals(persona, _life_start(folders))
    picks = _night_picks(folders, target_labels, chat)
    last_day = picks[-1].night if picks else _life_start(folders)
    return mind.LifeRecord(
        life_id=life_id,
        split=split,
        nights=tuple(picks),
        arrivals=arrivals,
        labels=labels,
        last_day=last_day,
        expectations=expectations,
        returns=returns,
    )


def collect_development_lives(
    results_root: Path,
    personas_dir: Path = PERSONAS_DIR,
    *,
    split: Split = "development",
    chat: ChatFn | None = None,
) -> tuple[mind.LifeRecord, ...]:
    """The four registered development personas under ``results_root/<user_id>/nights``."""
    return tuple(
        collect_life(
            results_root / user_id / "nights", personas_dir / f"{stem}.json", split=split, chat=chat
        )
        for stem, user_id in DEVELOPMENT_LIVES
    )


def load_frozen_labels(path: Path) -> FrozenLabels:
    """Read labels frozen before scoring; refuse a file that doesn't verify."""
    data = json.loads(path.read_text(encoding="utf-8"))
    prompt_sha = data["prompt_sha256"]
    digest = data["digest"]
    labels: dict[PairKey, bool] = {}
    for row in data["labels"]:
        target, item, value = row[0], row[1], row[2]
        labels[(target, item)] = bool(value)
    unreadable = frozenset((row[0], row[1]) for row in data.get("unreadable", ()))
    frozen = relevance.freeze(prompt_sha, labels, unreadable)
    if frozen.digest != digest:
        raise relevance.LabelError("the labels don't match their hash")
    return frozen


def _layout(log_dir: Path) -> tuple[Path, Path, str]:
    root = log_dir.resolve()
    if root.name == "nights":
        nights_dir = root
    elif (root / "nights").is_dir():
        nights_dir = root / "nights"
    elif (root / SUMMARY).is_file() and root.parent.name == "nights":
        nights_dir = root.parent
    else:
        nights_dir = root
    if not any(f.is_dir() and (f / SUMMARY).is_file() for f in nights_dir.iterdir()):
        raise CollectRefused(f"no night folders under {nights_dir}")
    return nights_dir, nights_dir.parent, nights_dir.parent.name


def _refuse_ids(life_id: str, user_id: str) -> None:
    for name in (life_id, user_id):
        if "kevin" in name.casefold():
            raise CollectRefused(f"refusing {name!r}")
    if _REAL_USER.match(user_id):
        raise CollectRefused(f"refusing non-synthetic account id {user_id!r}")


def _summary_label(split: Split) -> str:
    """Accelerated summaries use ``final`` for held-out nights; ``LifeRecord.split`` stays ``held_out``."""
    return "final" if split == "held_out" else "development"


def _collectible_folders(nights_dir: Path, split: Split) -> tuple[Path, ...]:
    """One folder per virtual night: the latest attempt that matches the split and has decision points."""
    want = _summary_label(split)
    by_day: dict[str, Path] = {}
    for folder in _night_folders(nights_dir):
        summary = json.loads((folder / SUMMARY).read_text(encoding="utf-8"))
        if summary.get("label") != want:
            continue
        if not NightLog(folder).points():
            continue
        day, _, _ = folder.name.partition(".")
        prev = by_day.get(day)
        if prev is None or _folder_key(folder) > _folder_key(prev):
            by_day[day] = folder
    if not by_day:
        raise CollectRefused(f"no {split} nights with decision points under {nights_dir}")
    return tuple(sorted(by_day.values(), key=_folder_key))


def _life_start(folders: Sequence[Path]) -> date:
    dated = [
        date.fromisoformat(str(json.loads((folder / SUMMARY).read_text(encoding="utf-8"))["night"]))
        for folder in folders
    ]
    return min(dated)


def _night_folders(nights_dir: Path) -> Sequence[Path]:
    if not nights_dir.is_dir():
        return ()

    return tuple(
        sorted((f for f in nights_dir.iterdir() if (f / SUMMARY).is_file()), key=_folder_key)
    )


def _folder_key(folder: Path) -> tuple[str, int]:
    day, _, attempt = folder.name.partition(".")
    return day, int(attempt) if attempt.isdigit() else 1


def _target_labels(path: Path, user_id: str) -> dict[str, str]:
    if not path.is_file():
        return {}
    store = EventLogTargetStore(path, user_id, codec=PLAIN, guard=NO_GUARD)
    return {target.id: target.label for target in store.targets()}


def _expectations(path: Path, user_id: str) -> tuple[Expectation, ...]:
    if not path.is_file():
        return ()
    store = EventLogTargetStore(path, user_id, codec=PLAIN, guard=NO_GUARD)
    return tuple(store.expectations())


def _returns(path: Path, user_id: str) -> dict[str, tuple[date, ...]]:
    """When the person's own evidence came back to each target (start of that virtual day)."""
    if not path.is_file():
        return {}
    store = JsonlEvidenceStore(path, user_id, codec=PLAIN, guard=NO_GUARD)
    by_target: dict[str, list[date]] = {}
    for item in store.stored(date.max):
        if item.target_id is not None and expect.returning([item], item.target_id):
            by_target.setdefault(item.target_id, []).append(item.day)
    return {target: tuple(sorted(days)) for target, days in by_target.items()}


def _arrivals(persona: PersonFile, life_start: date) -> tuple[mind.Arrival, ...]:
    visible = json.dumps(to_visible_dict(persona.visible), sort_keys=True).casefold()
    out: list[mind.Arrival] = []
    for item in persona.held_out.new_work:
        if item.text.strip().casefold() in visible:
            raise CollectRefused(f"held-out item {item.id} appears in the visible persona")
        out.append(mind.Arrival(item.id, life_start + timedelta(days=item.day)))
    return tuple(out)


def _night_picks(
    folders: Sequence[Path],
    labels: Mapping[str, str],
    chat: ChatFn | None,
) -> list[mind.NightPicks]:
    out: list[mind.NightPicks] = []
    for folder in folders:
        records = NightLog(folder).points()
        parsed = [points.point_from_record(record) for record in records]
        pre = next((p for p in parsed if p.phase == "pre_study"), None)
        if pre is None:
            raise ValueError(f"{folder.name}: a recorded night needs a pre-study point")
        post = next((p for p in parsed if p.phase == "post_result"), None)
        merged = dict(labels)
        missing = {t.id for p in parsed for t in p.raw.targets if t.id not in merged}
        if missing:
            raise CollectRefused(f"no label for targets {sorted(missing)}")
        out.append(
            night_picks(
                replay_point(pre, merged, chat=chat),
                None if post is None else replay_point(post, merged, chat=chat),
            )
        )
    return out
