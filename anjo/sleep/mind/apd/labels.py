"""MIND core's event streams and training labels, exported from the data dir (MIND's data).

::

    <out>/manifest.json          the export: its schema, split, subset, whether its labels are
                                 sealed, and each life's id, field, days and event count
    <out>/events/<life>.jsonl    the life's events in order, one per line
    <out>/labels/<life>.json     its planted items as labels: labelled splits only

**An event** is one evidence item an arm may see (a held-out item never), in the order the life
said them: by day, then in the order extraction gave (the life's own order). It keeps every
field ``Evidence`` has, so ``evidence_of`` rebuilds exactly what the judge is shown for a finding
reported on any night (``arms.seen_by``), as for B's; MIND's latents key on ``event_id``,
``life_or_person``, ``t`` and ``kind``. A core finding cites these ids.

**Labels** (``item_labels``): each planted item with its family, template, topic and sealed
resolution, the events it arises from (``origin_ids``) and the one that settles it
(``disambiguating_ids``), the event and day from which it is knowable (its last origin event), the
one from which it is resolvable (its last origin or disambiguating event), its surface day and the
judge's cutoff (the day before). ``actionable`` is a problem family's (``PROBLEM_FAMILIES``).

**Pairs** (``export_pairs``): ``pairs_manifest.json`` (``PAIRS_MANIFEST``, so a split's export in
the same folder keeps its own), each counterfactual pair's two histories as event streams
(``events/<pair>.failed.jsonl`` and ``.kept.jsonl``) and ``pairs/<pair>.json``: the fork days, the
slot's events, each history's outcome events, the shared last day's events, the decision each
history should take about the slot, and the events of the item that should not move. No topic or
resolution: pairs are test material, never trained on.

**Sealed.** Only ``LABELLED_SPLITS`` export labels (``DataDir.labelled_problems``). ``core-val``
and ``core-test`` export their event streams only and never read a problem file: their items
reach the judge through ``DataDir.sealed``, the path B's runs are judged on. Pure apart from
``export``, which reads the data dir and writes ``out``.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd.core_splits import in_subset
from anjo.sleep.mind.apd.store import DataDir, write_text
from anjo.sleep.mind.apd.types import (
    CORE_SPLITS,
    LABELLED_SPLITS,
    PAIR_DECISIONS,
    PROBLEM_FAMILIES,
    CounterfactualPair,
    Life,
    LifeSplit,
    PlantedItem,
    ProblemFile,
)
from anjo.sleep.types import Evidence

SCHEMA: Final = "mind-core-events-v1"
EXPORT_SPLITS: Final[tuple[LifeSplit, ...]] = ("development", *CORE_SPLITS)
PAIRS_MANIFEST: Final = "pairs_manifest.json"  # beside a split's manifest.json, never over it


def events(life: Life) -> tuple[dict[str, Any], ...]:
    """The life's events in order (``seq`` from 0), each with every field of its evidence."""
    seen = sorted(enumerate(item for item in life.evidence if not item.held_out), key=lambda pair: (pair[1].day, pair[0]))
    return tuple({
        "event_id": item.id, "life_or_person": item.person_id, "seq": seq, "t": item.created_at.isoformat(),
        "day": item.day.isoformat(), "kind": item.kind, "text": item.text, "source": item.source,
        "target_id": item.target_id, "model": item.model,
        "ancestry": None if item.ancestry is None else sorted(item.ancestry),
    } for seq, (_, item) in enumerate(seen))


def evidence_of(records: Sequence[Mapping[str, Any]]) -> tuple[Evidence, ...]:
    """The evidence ``events`` exported, rebuilt: what the judge is shown, for any finding."""
    return tuple(Evidence(
        id=str(r["event_id"]), person_id=str(r["life_or_person"]), day=date.fromisoformat(str(r["day"])),
        kind=r["kind"], text=str(r["text"]), source=str(r["source"]), held_out=False,
        created_at=datetime.fromisoformat(str(r["t"])), model=r["model"],
        ancestry=None if r["ancestry"] is None else frozenset(r["ancestry"]), target_id=r["target_id"],
    ) for r in records)


def item_labels(life: Life, problems: ProblemFile) -> tuple[dict[str, Any], ...]:
    """Each planted item as a label, its events located in the life's stream."""
    if problems.life_id != life.id:
        raise ValueError("a problem file is its own life's")
    stream = {record["event_id"]: record for record in events(life)}

    def last(ids: Sequence[str]) -> Mapping[str, Any]:
        missing = [i for i in ids if i not in stream]
        if missing:
            raise ValueError(f"{life.id}: a planted item cites evidence outside its stream: {missing}")
        return max((stream[i] for i in ids), key=lambda record: record["seq"])

    labels = []
    for item in problems.items:
        knowable, resolvable = last(item.origin_ids), last((*item.origin_ids, *item.disambiguating_ids))
        labels.append({
            "item_id": item.id, "family": item.family, "actionable": item.family in PROBLEM_FAMILIES,
            "template": item.template, "topic": item.topic, "resolution": item.resolution,
            "origin_ids": list(item.origin_ids), "disambiguating_ids": list(item.disambiguating_ids),
            "knowable_event_id": knowable["event_id"], "knowable_day": knowable["day"],
            "resolvable_event_id": resolvable["event_id"], "resolvable_day": resolvable["day"],
            "surface_day": item.surface_day.isoformat(),
            "cutoff_day": (item.surface_day - timedelta(days=1)).isoformat(),
        })
    return tuple(labels)


@dataclass(frozen=True)
class Exported:
    """What ``export`` wrote: the lives, and whether their labels were left sealed."""

    lives: tuple[str, ...]
    events: int
    sealed: bool


def _lines(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def export(data: DataDir, split: LifeSplit, out: Path, *, subset: str | None = None) -> Exported:
    """Every life of ``split`` on disk (or of ``subset``), its events, and its labels when the split
    is labelled, written under ``out``."""
    if split not in EXPORT_SPLITS:
        raise ValueError(f"the core's exports are of {EXPORT_SPLITS}, not {split}")
    lives = data.lives(split)
    if subset is not None:
        lives = in_subset(lives, subset)
    sealed = split not in LABELLED_SPLITS
    manifest: list[dict[str, Any]] = []
    total = 0
    for life in lives:
        stream = events(life)
        write_text(out / "events" / f"{life.id}.jsonl", _lines(stream))
        if not sealed:
            labels = {"life_id": life.id, "split": life.split, "field": life.field,
                      "items": list(item_labels(life, data.labelled_problems(life.id)))}
            write_text(out / "labels" / f"{life.id}.json", json.dumps(labels, ensure_ascii=False, sort_keys=True))
        manifest.append({"life_id": life.id, "field": life.field, "first_day": life.days[0].isoformat(),
                         "last_day": life.days[-1].isoformat(), "events": len(stream)})
        total += len(stream)
    write_text(out / "manifest.json", json.dumps({
        "schema": SCHEMA, "split": split, "subset": subset, "sealed": sealed, "lives": manifest,
    }, ensure_ascii=False, sort_keys=True, indent=1))
    return Exported(tuple(life.id for life in lives), total, sealed)


def _item_ids(item: PlantedItem) -> dict[str, Any]:
    return {"item_id": item.id, "origin_ids": list(item.origin_ids), "disambiguating_ids": list(item.disambiguating_ids),
            "surface_day": item.surface_day.isoformat()}


def pair_record(pair: CounterfactualPair) -> dict[str, Any]:
    """What a transplant test needs of a pair: what should differ, and what should not."""
    slot = next(item for item in pair.failed_problems.items if item.id != pair.unrelated_item)
    unrelated = next(item for item in pair.kept_problems.items if item.id == pair.unrelated_item)
    return {
        "pair_id": pair.id, "template": pair.template, "field": pair.failed.field,
        "fork_days": [day.isoformat() for day in pair.fork_days], "last_day": pair.failed.days[-1].isoformat(),
        "slot_ids": list(pair.slot_ids), "failed_ids": list(pair.failed_ids), "kept_ids": list(pair.kept_ids),
        "final_ids": list(pair.final_ids), "decisions": dict(PAIR_DECISIONS),
        "slot_item": _item_ids(slot), "unrelated_item": _item_ids(unrelated),
    }


def export_pairs(data: DataDir, out: Path) -> tuple[str, ...]:
    """Every pair on disk, written under ``out``."""
    found = data.pairs()
    for pair in found:
        for name, life in (("failed", pair.failed), ("kept", pair.kept)):
            write_text(out / "events" / f"{pair.id}.{name}.jsonl", _lines(events(life)))
        write_text(out / "pairs" / f"{pair.id}.json", json.dumps(pair_record(pair), ensure_ascii=False, sort_keys=True))
    write_text(out / PAIRS_MANIFEST, json.dumps({
        "schema": SCHEMA, "pairs": [pair.id for pair in found], "sealed": False,
        "note": "counterfactual pairs in core-test's fields: test material, never trained on",
    }, ensure_ascii=False, sort_keys=True, indent=1))
    return tuple(pair.id for pair in found)


def read_events(path: Path) -> tuple[dict[str, Any], ...]:
    """One life's exported events."""
    return tuple(read_records(path, PLAIN))
