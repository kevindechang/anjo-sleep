"""MIND core's data: the latents folder and the data export, read into labelled streams. Torch-free.

**Latents** (the latents worker's ``anjo-latents/1`` folder): ``index.jsonl`` is a header line
(``d``, the layers, the model revision, the pooling, ``count``), then one line per matrix row
(``row``, ``event_id``, ``life_or_person``, ``t``, ``kind``); ``latents.npy`` is the ``(count, d)``
float32 matrix. ``read_latents`` checks the two against each other and never unpickles.

**The export** (the data worker's ``mind-core-events-v1``): ``manifest.json`` lists each life with
its first and last day; ``events/<life>.jsonl`` holds its events in order; ``labels/<life>.json``
its planted items, for labelled splits only. A sealed split (``core-val``, ``core-test``) has no
label files, and the core reads none: its lives come back with no labels.

**A planted item becomes a ``ThreadLabel``:** its events are its origin and disambiguating ids,
its origin ids are what a finding must cite (APD's match rule), it is knowable from its knowable
event and raised at the start of its surface day. Problem families (``actionable``) fail; decoys
and reversals don't. A gap is a missing prerequisite.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from anjo.sleep.mind.core.types import EventLatent, LabelledStream, ThreadLabel

LATENTS_FORMAT: Final = "anjo-latents/1"
INDEX_FILE: Final = "index.jsonl"
MATRIX_FILE: Final = "latents.npy"
EXPORT_SCHEMA: Final = "mind-core-events-v1"


class DataError(ValueError):
    """A latents folder or an export the core can't read."""


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataError(f"{path.name} can't be read: {type(error).__name__}") from None


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise DataError("an event's t is an ISO time")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise DataError(f"an event's t can't be read: {value!r}") from None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def read_latents(folder: Path) -> tuple[dict[str, Any], dict[str, EventLatent]]:
    """The folder's header and each event's latent, by event id."""
    try:
        lines = (folder / INDEX_FILE).read_text(encoding="utf-8").splitlines()
        header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:] if line.strip()]
        matrix = np.load(folder / MATRIX_FILE, allow_pickle=False)
    except (OSError, ValueError, IndexError) as error:
        raise DataError(f"not a latents folder: {type(error).__name__}") from None
    if header.get("format") != LATENTS_FORMAT:
        raise DataError(f"the index isn't {LATENTS_FORMAT}")
    if header.get("count") != len(rows) or matrix.shape[0] != len(rows):
        raise DataError("the index's count, its rows and the matrix disagree")
    d = int(header["d"])
    if matrix.dtype != np.float32 or matrix.shape != (len(rows), d):
        raise DataError(f"the matrix isn't float32 with D = {d}")
    latents: dict[str, EventLatent] = {}
    for n, row in enumerate(rows):
        at = int(row.get("row", n))
        latents[str(row["event_id"])] = EventLatent(
            str(row["event_id"]), str(row["life_or_person"]), _time(row["t"]), str(row["kind"]), matrix[at])
    return header, latents


def label_of(record: Mapping[str, Any], times: Mapping[str, datetime]) -> ThreadLabel:
    """One exported planted item as a thread label; ``times`` maps the life's event ids to their times."""
    origin = tuple(str(i) for i in record["origin_ids"])
    settles = tuple(str(i) for i in record["disambiguating_ids"])
    surface = date.fromisoformat(str(record["surface_day"]))
    family = str(record["family"])
    return ThreadLabel(
        id=str(record["item_id"]), family=family, event_ids=tuple(dict.fromkeys((*origin, *settles))),
        origin_ids=origin, knowable_at=times[str(record["knowable_event_id"])],
        outcome_at=datetime.combine(surface, time(0), tzinfo=timezone.utc), failed=bool(record["actionable"]),
        missing=family == "gap", template=str(record["template"]) if record.get("template") else None,
    )


def stream_of(
    entry: Mapping[str, Any], events: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]],
    latents: Mapping[str, EventLatent],
) -> LabelledStream:
    """One life from its manifest entry, exported events and labels, with each event's latent."""
    life_id = str(entry["life_id"])
    missing = [str(e["event_id"]) for e in events if str(e["event_id"]) not in latents]
    if missing:
        raise DataError(f"life {life_id}: no latent for {missing[:5]}")
    ordered = sorted(events, key=lambda e: (_time(e["t"]), int(e.get("seq", 0))))
    stream = tuple(latents[str(e["event_id"])] for e in ordered)
    first, last = date.fromisoformat(str(entry["first_day"])), date.fromisoformat(str(entry["last_day"]))
    days = tuple(first + timedelta(days=n) for n in range((last - first).days + 1))
    times = {e.event_id: e.t for e in stream}
    return LabelledStream(life_id, days, stream, tuple(label_of(record, times) for record in labels))


def load_export(export: Path, latents_folder: Path) -> tuple[LabelledStream, ...]:
    """Every life of an export, in manifest order, with its latents (and its labels, unless sealed)."""
    manifest = _json(export / "manifest.json")
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise DataError(f"the export isn't {EXPORT_SCHEMA}")
    _, latents = read_latents(latents_folder)
    streams = []
    for entry in manifest["lives"]:
        life_id = str(entry["life_id"])
        lines = (export / "events" / f"{life_id}.jsonl").read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
        label_file = export / "labels" / f"{life_id}.json"
        labels = [] if manifest.get("sealed") or not label_file.exists() else _json(label_file)["items"]
        streams.append(stream_of(entry, events, labels, latents))
    return tuple(streams)
