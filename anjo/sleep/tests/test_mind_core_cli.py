# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND core's CLI end to end on a tiny export of toy lives: train, write the model card, run the
core night by night into finding records, and pre-score them (CPU, offline)."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from anjo.sleep.mind.core import cli, toy  # noqa: E402
from anjo.sleep.mind.core.types import LabelledStream  # noqa: E402


def _export(out: Path, streams: tuple[LabelledStream, ...]) -> None:
    """Toy lives in the data export's and the latents worker's formats."""
    (out / "events").mkdir(parents=True)
    (out / "labels").mkdir()
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    lives: list[dict[str, object]] = []
    for s in streams:
        events = [{"event_id": e.event_id, "life_or_person": s.life_id, "seq": n, "t": e.t.isoformat(),
                   "day": e.t.date().isoformat(), "kind": e.kind, "text": f"said {e.event_id}"} for n, e in enumerate(s.events)]
        (out / "events" / f"{s.life_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
        at = {e.t: e.event_id for e in s.events}
        items = [{"item_id": lab.id, "family": lab.family, "actionable": lab.failed, "template": "toy", "topic": "toy",
                  "resolution": "sealed", "origin_ids": list(lab.origin_ids), "disambiguating_ids": list(lab.event_ids),
                  "knowable_event_id": at[lab.knowable_at], "knowable_day": lab.knowable_at.date().isoformat(),
                  "surface_day": lab.outcome_at.date().isoformat(),
                  "cutoff_day": (lab.outcome_at.date() - timedelta(days=1)).isoformat()} for lab in s.labels]
        (out / "labels" / f"{s.life_id}.json").write_text(json.dumps({"life_id": s.life_id, "items": items}))
        lives.append({"life_id": s.life_id, "field": "toy", "first_day": s.days[0].isoformat(),
                      "last_day": s.days[-1].isoformat(), "events": len(events)})
        rows += [{"row": len(rows) + n, "event_id": e.event_id, "life_or_person": s.life_id, "t": e.t.isoformat(),
                  "kind": e.kind, "key": e.event_id} for n, e in enumerate(s.events)]
        vectors += [e.vec for e in s.events]
    (out / "manifest.json").write_text(json.dumps({"schema": "mind-core-events-v1", "split": "core-train",
                                                    "sealed": False, "lives": lives}))
    latents = out / "latents"
    latents.mkdir()
    np.save(latents / "latents.npy", np.stack(vectors).astype(np.float32), allow_pickle=False)
    header = {"format": "anjo-latents/1", "d": toy.DIM, "count": len(rows), "layers": [35], "model": "toy",
              "revision": "r0", "pooling": "mean"}
    (latents / "index.jsonl").write_text("\n".join(json.dumps(r) for r in [header, *rows]) + "\n")


def test_train_findings_and_prescore_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _export(tmp_path / "export", toy.lives(8, seed=21))
    export, latents, model = tmp_path / "export", tmp_path / "export" / "latents", tmp_path / "model"
    assert cli.main(["train", "--export", str(export), "--latents", str(latents), "--out", str(model),
                     "--epochs", "2", "--width", "16", "--slots", "4"]) == 0
    capsys.readouterr()
    card = json.loads((model / cli.CARD_FILE).read_text())
    assert card["params"] > 0 and card["data"]["fit"] + card["data"]["held_back"] == 8
    assert card["data"]["latents"]["d"] == toy.DIM and card["timing"]["observe_ms_per_event"] > 0

    found = tmp_path / "findings.jsonl"
    assert cli.main(["findings", "--model", str(model / cli.MODEL_FILE), "--export", str(export),
                     "--latents", str(latents), "--out", str(found), "--dream", "2", "--threshold", "0"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["lives"] == 8 and summary["events"] > 0 and summary["findings"] > 0 and summary["arm"] == "core-N"
    rows = [json.loads(line) for line in found.read_text().splitlines()]
    assert len(rows) == summary["findings"]
    assert all(row["kind"] == "suspicion" and row["cites"] and row["problem"].startswith("said ") for row in rows)
    assert {row["arm"] for row in rows} == {"core-N"}

    assert cli.main(["prescore", "--findings", str(found), "--labels", str(export / "labels")]) == 0
    scored = json.loads(capsys.readouterr().out)
    assert scored["problems"] > 0 and 0 <= scored["caught"] <= scored["problems"]
