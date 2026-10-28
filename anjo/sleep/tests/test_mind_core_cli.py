# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND core's CLI end to end on a tiny export of toy lives: train, write the model card, run the
core night by night into finding records, and pre-score them (CPU, offline)."""
from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from anjo.sleep.mind.apd.store import dumps  # noqa: E402
from anjo.sleep.mind.apd.types import ArmRun, FindingReport  # noqa: E402
from anjo.sleep.mind.core import cli, evaluate, findings, toy  # noqa: E402
from anjo.sleep.mind.core.types import LabelledStream  # noqa: E402


def _export(out: Path, streams: tuple[LabelledStream, ...]) -> None:
    """Toy lives in the data export's and the latents worker's formats."""
    (out / "events").mkdir(parents=True)
    (out / "labels").mkdir()
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    lives: list[dict[str, object]] = []
    for index, s in enumerate(streams):
        events = [{"event_id": e.event_id, "life_or_person": s.life_id, "seq": n, "t": e.t.isoformat(),
                   "day": e.t.date().isoformat(), "kind": e.kind, "text": f"said {e.event_id}"} for n, e in enumerate(s.events)]
        (out / "events" / f"{s.life_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
        at = {e.t: e.event_id for e in s.events}
        items = [{"item_id": lab.id, "family": lab.family, "actionable": lab.failed, "template": f"{lab.family}-{index % 3}", "topic": "toy",
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
                     "--latents", str(latents), "--out", str(found), "--dream", "--threshold", "0"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["lives"] == 8 and summary["events"] > 0 and summary["findings"] > 0
    assert summary["arm"] == "core-N-dream"  # ignitions stay out of the headline arm
    assert summary["threshold_by_hand"] is True and summary["threshold"] == 0.0
    assert summary["dream_steps"] == evaluate.DREAM_STEPS and summary["ignite_threshold"] == card["thresholds"]["ignite"]
    rows = [json.loads(line) for line in found.read_text().splitlines()]
    assert len(rows) == summary["findings"]
    assert all(row["kind"] == "suspicion" and row["cites"] and row["problem"].startswith("said ") for row in rows)
    assert {row["arm"] for row in rows} == {"core-N-dream"} and all(row["threshold_by_hand"] for row in rows)
    dreamt = [row for row in rows if row["conclusion"] != findings.CONCLUSIONS["speak"]]
    assert all(row["p"] >= card["thresholds"]["ignite"] for row in dreamt)  # ignitions at their own locked line

    locked = tmp_path / "locked.jsonl"
    assert cli.main(["findings", "--model", str(model / cli.MODEL_FILE), "--export", str(export),
                     "--latents", str(latents), "--out", str(locked)]) == 0
    plain = json.loads(capsys.readouterr().out)
    assert plain["threshold_by_hand"] is False and plain["arm"] == "core-N"
    assert cli.main(["findings", "--model", str(model / cli.MODEL_FILE), "--export", str(export),
                     "--latents", str(latents), "--out", str(locked), "--dream", "--arm", "core-N"]) == 0
    assert json.loads(capsys.readouterr().out)["arm"] == "core-N-dream"  # an override can't put ignitions in the headline

    assert cli.main(["prescore", "--findings", str(found), "--labels", str(export / "labels")]) == 0
    scored = json.loads(capsys.readouterr().out)
    assert scored["problems"] > 0 and 0 <= scored["caught"] <= scored["problems"]


def _templated(streams: tuple[LabelledStream, ...]) -> list[LabelledStream]:
    return [dataclasses.replace(s, labels=tuple(dataclasses.replace(lab, template=f"{lab.family}-{n % 4}")
                                                for lab in s.labels)) for n, s in enumerate(streams)]


def test_lives_are_held_back_by_template() -> None:
    streams = _templated(toy.lives(40, seed=8))
    fit, held = cli.split_lives(streams, 0.2, seed=1)
    assert fit and held and len(fit) + len(held) == len(streams)
    held_templates = {lab.template for s in held for lab in s.labels} - {lab.template for s in fit for lab in s.labels}
    assert held_templates  # templates no fitted life uses
    for s in fit:
        assert not {lab.template for lab in s.labels} & held_templates
    assert any(not s.labels for s in held) or not any(not s.labels for s in streams)  # problem-free lives too
    again = cli.split_lives(streams, 0.2, seed=1)
    assert [[s.life_id for s in side] for side in again] == [[s.life_id for s in side] for side in (fit, held)]
    fitted = {s.life_id for s in fit}
    assert [s.life_id for s in fit] == [s.life_id for s in streams if s.life_id in fitted]  # export order kept


def test_another_arms_findings_render_in_the_cores_form(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    streams = toy.lives(2, seed=21)
    _export(tmp_path / "export", streams)
    life = streams[0]
    first, second, third = (e.event_id for e in life.events[:3])
    report = FindingReport(finding_id="b-1", day=life.days[1], seq=0, problem="My own long prose about it",
                           conclusion="A conclusion in B's words", p=0.7, cites=(first, second, third), kind="suspicion")
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / f"{life.life_id}.B.json").write_text(dumps(ArmRun(arm="B", life_id=life.life_id, reports=(report,),
                                                             tokens=(10,), contexts={})))
    out = tmp_path / "b.jsonl"
    assert cli.main(["render", "--runs", str(runs), "--arm", "B", "--export", str(tmp_path / "export"),
                     "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["findings"] == 1
    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert row["problem"] == f"said {first} / said {second}" and row["conclusion"] == findings.CONCLUSIONS["speak"]
    assert (row["arm"], row["life_id"], row["finding_id"], row["p"], row["cites"]) == (
        "B", life.life_id, "b-1", 0.7, [first, second, third])
    assert cli.main(["prescore", "--findings", str(out), "--labels", str(tmp_path / "export" / "labels")]) == 0


def test_toy_training_takes_the_revision_pairs_unless_told_not_to(tmp_path: Path) -> None:
    assert not cli.parser().parse_args(["toy", "--out", str(tmp_path)]).no_revisions
    assert cli.parser().parse_args(["toy", "--out", str(tmp_path), "--no-revisions"]).no_revisions
