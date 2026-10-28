"""MIND core's data side, torch-free: reading the latents folder and the data export into labelled
streams, turning the core's calls into APD finding records, and the deterministic pre-score."""
from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anjo.sleep.mind.core import data, findings

D = 6
DAY0 = date(2026, 10, 1)


def _t(day: int, hour: int) -> str:
    return datetime(2026, 10, 1 + day, hour, tzinfo=timezone.utc).isoformat()


EVENTS: list[dict[str, Any]] = [  # the export's event rows (only the fields the core reads, plus text)
    {"event_id": "e-d0-1", "life_or_person": "life-1", "seq": 0, "t": _t(0, 9), "day": "2026-10-01",
     "kind": "commitment", "text": "Booked the rental house for Thursday."},
    {"event_id": "e-d1-1", "life_or_person": "life-1", "seq": 1, "t": _t(1, 10), "day": "2026-10-02",
     "kind": "turn", "text": "The rental house cancelled on us last month."},
    {"event_id": "e-d2-1", "life_or_person": "life-1", "seq": 2, "t": _t(2, 8), "day": "2026-10-03",
     "kind": "turn", "text": "Lunch was good."},
]
LABEL: dict[str, Any] = {  # one planted item, as the data export writes it
    "item_id": "life-1-gap-0", "family": "gap", "actionable": True, "template": "t1", "topic": "the shoot",
    "resolution": "sealed", "origin_ids": ["e-d0-1", "e-d1-1"], "disambiguating_ids": ["e-d2-1"],
    "knowable_event_id": "e-d1-1", "knowable_day": "2026-10-02", "resolvable_event_id": "e-d2-1",
    "resolvable_day": "2026-10-03", "surface_day": "2026-10-04", "cutoff_day": "2026-10-03",
}


def _latents(folder: Path, rows: list[dict[str, Any]], *, d: int = D, count: int | None = None) -> np.ndarray:
    folder.mkdir(parents=True, exist_ok=True)
    matrix: np.ndarray = np.arange(len(rows) * d, dtype=np.float32).reshape(len(rows), d)
    np.save(folder / "latents.npy", matrix, allow_pickle=False)
    header = {"format": "anjo-latents/1", "d": d, "count": len(rows) if count is None else count, "layers": [35],
              "model": "glimmer", "revision": "r1", "pooling": "mean"}
    lines = [json.dumps(header)] + [json.dumps({"row": n, "event_id": r["event_id"], "life_or_person": r["life_or_person"],
                                                "t": r["t"], "kind": r["kind"], "key": f"k{n}"}) for n, r in enumerate(rows)]
    (folder / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return matrix


def _export(folder: Path, *, sealed: bool = False) -> None:
    (folder / "events").mkdir(parents=True)
    (folder / "events" / "life-1.jsonl").write_text("".join(json.dumps(e) + "\n" for e in EVENTS), encoding="utf-8")
    if not sealed:
        (folder / "labels").mkdir()
        (folder / "labels" / "life-1.json").write_text(json.dumps({"life_id": "life-1", "items": [LABEL]}), encoding="utf-8")
    manifest = {"schema": "mind-core-events-v1", "split": "core-train", "sealed": sealed, "lives": [
        {"life_id": "life-1", "field": "film", "first_day": "2026-10-01", "last_day": "2026-10-05", "events": 3}]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# ── Latents and the export ────────────────────────────────────────────────────


def test_latents_folder_reads_back_and_refuses_a_bad_one(tmp_path: Path) -> None:
    matrix = _latents(tmp_path / "ok", EVENTS)
    header, latents = data.read_latents(tmp_path / "ok")
    assert header["d"] == D and set(latents) == {e["event_id"] for e in EVENTS}
    assert np.array_equal(latents["e-d1-1"].vec, matrix[1])
    assert latents["e-d1-1"].t == datetime(2026, 10, 2, 10, tzinfo=timezone.utc)

    _latents(tmp_path / "count", EVENTS, count=2)
    with pytest.raises(data.DataError, match="count"):
        data.read_latents(tmp_path / "count")
    _latents(tmp_path / "dim", EVENTS)
    np.save(tmp_path / "dim" / "latents.npy", np.zeros((3, D + 1), np.float32), allow_pickle=False)
    with pytest.raises(data.DataError, match="D"):
        data.read_latents(tmp_path / "dim")


def test_a_planted_item_becomes_a_thread_label() -> None:
    times: dict[str, datetime] = {str(e["event_id"]): datetime.fromisoformat(str(e["t"])) for e in EVENTS}
    label = data.label_of(LABEL, times)
    assert label.id == "life-1-gap-0" and label.failed and label.missing
    assert label.origin_ids == ("e-d0-1", "e-d1-1") and label.event_ids == ("e-d0-1", "e-d1-1", "e-d2-1")
    assert label.knowable_at == times["e-d1-1"]
    assert label.outcome_at == datetime(2026, 10, 4, tzinfo=timezone.utc)  # raised on its surface day
    decoy = data.label_of({**LABEL, "family": "decoy", "actionable": False}, times)
    assert not decoy.failed and not decoy.missing


def test_the_export_and_latents_become_labelled_streams(tmp_path: Path) -> None:
    _latents(tmp_path / "lat", EVENTS)
    _export(tmp_path / "train")
    (stream,) = data.load_export(tmp_path / "train", tmp_path / "lat")
    assert stream.life_id == "life-1" and len(stream.days) == 5 and stream.days[0] == DAY0
    assert [e.event_id for e in stream.events] == ["e-d0-1", "e-d1-1", "e-d2-1"]
    assert [label.id for label in stream.labels] == ["life-1-gap-0"]

    _export(tmp_path / "test", sealed=True)
    (sealed,) = data.load_export(tmp_path / "test", tmp_path / "lat")
    assert sealed.labels == ()  # a sealed split exports no labels, and the core reads none

    _latents(tmp_path / "short", EVENTS[:2])
    with pytest.raises(data.DataError, match="e-d2-1"):
        data.load_export(tmp_path / "train", tmp_path / "short")


# ── Finding records and the pre-score ─────────────────────────────────────────


def _day(n: int) -> date:
    return date(2026, 10, 1 + n)


def test_core_calls_become_finding_records() -> None:
    calls = [
        findings.Raised(_day(1), 0.40, ("e-d0-1",), "speak"),  # below the threshold: not reported
        findings.Raised(_day(1), 0.70, ("e-d0-1", "e-d1-1"), "speak"),
        findings.Raised(_day(2), 0.80, ("e-d1-1",), "speak"),  # the same thread: updates its finding
        findings.Raised(_day(2), 0.90, ("e-d2-1",), "risk"),  # an ignition, a new thread
        findings.Raised(_day(2), 0.70, ("e-d3-1",), "missing"),  # an ignition under the dream's own line
    ]
    out = findings.reports(calls, threshold=0.5, ignite=0.85, topic=lambda cites: f"about {len(cites)} events")
    assert [(r.finding_id, r.day, r.seq, r.p, r.cites) for r in out] == [
        ("core-1", _day(1), 0, 0.70, ("e-d0-1", "e-d1-1")),
        ("core-1", _day(2), 1, 0.80, ("e-d1-1",)),
        ("core-2", _day(2), 2, 0.90, ("e-d2-1",)),
    ]
    assert all(r.kind == "suspicion" and r.problem.startswith("about") for r in out)
    assert [r.cites for r in findings.reports(calls, threshold=0.5)] == [("e-d0-1", "e-d1-1"), ("e-d1-1",)]  # no dream line


def test_renderer_keeps_each_b_finding_kind() -> None:
    def quote(cites: tuple[str, ...]) -> str:
        return "quoted " + " / ".join(cites)

    base = findings.reports([findings.Raised(_day(1), 0.7, ("e-d0-1", "e-d1-1"), "speak")], threshold=0.5)[0]
    seen = {}
    for kind in ("suspicion", "retraction", "resolution"):
        theirs = dataclasses.replace(base, finding_id="b-3", problem="B's own prose", conclusion="The decoy is fine.",
                                     kind=kind)
        mine = findings.in_core_form(theirs, quote)
        assert mine.problem == "quoted e-d0-1 / e-d1-1"
        assert (mine.finding_id, mine.day, mine.p, mine.cites, mine.kind) == ("b-3", _day(1), 0.7, theirs.cites, kind)
        seen[kind] = mine.conclusion
    assert seen["suspicion"] == findings.CONCLUSIONS["speak"]  # only a suspicion reads as the core's flag
    assert seen["retraction"] == findings.RETRACTED  # a fixed sentence of its own kind, never a flag
    assert seen["resolution"] == "The decoy is fine."  # it scores on whether its own conclusion holds


def test_prescore_counts_cited_overlap_inside_each_items_window() -> None:
    item = findings.item_of(LABEL)
    decoy = findings.item_of({**LABEL, "item_id": "life-1-decoy-0", "family": "decoy", "actionable": False,
                              "origin_ids": ["e-d4-1"], "disambiguating_ids": ["e-d4-2"], "knowable_day": "2026-10-05",
                              "surface_day": "2026-10-07", "cutoff_day": "2026-10-06"})
    report = findings.reports([findings.Raised(_day(2), 0.9, ("e-d2-1",), "speak")], threshold=0.5)
    caught = findings.prescore(report, (item, decoy))
    assert (caught.problems, caught.caught, caught.decoys, caught.flagged, caught.unmatched) == (1, 1, 1, 0, 0)

    late = findings.reports([findings.Raised(_day(3), 0.9, ("e-d1-1",), "speak")], threshold=0.5)  # the day it surfaced
    assert findings.prescore(late, (item,)).caught == 0
    early = findings.reports([findings.Raised(_day(0), 0.9, ("e-d0-1",), "speak")], threshold=0.5)  # before knowable
    assert findings.prescore(early, (item,)).caught == 0
    alarm = findings.reports([findings.Raised(_day(5), 0.9, ("e-d4-1",), "speak"),
                              findings.Raised(_day(5), 0.9, ("e-x",), "risk")], threshold=0.5, ignite=0.5)
    scored = findings.prescore(alarm, (item, decoy))
    assert (scored.flagged, scored.unmatched, scored.findings) == (1, 1, 2)


def test_the_core_reads_the_latents_workers_event_record() -> None:
    from anjo.sleep.mind.core import latents, types

    assert types.EventLatent is latents.EventLatent  # one record, no duplicate
    at = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    good = latents.EventLatent("e1", "life-1", at, "turn", np.ones(4, dtype=np.float32))
    types.LabelledStream("life-1", (at.date(),), (good,), ())
    for bad in (latents.EventLatent("e1", "life-1", at, "turn", np.array([1.0, np.nan], dtype=np.float32)),
                latents.EventLatent("e1", "life-1", at, "turn", np.ones((2, 2), dtype=np.float32)),
                latents.EventLatent("e1", "life-1", at.replace(tzinfo=None), "turn", np.ones(4, dtype=np.float32))):
        with pytest.raises(ValueError):
            types.LabelledStream("life-1", (at.date(),), (bad,), ())
