# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND core's event streams and labels (MIND's data): every event in the life's order with its
id, time, kind and text; core-train's planted items mapped to those ids with the day each is
knowable and resolvable; core-val and core-test exported as streams only, their problem files
never read; and the stream rebuilds exactly what the judge is shown for a finding, as for B's.
Every model is a fake."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import core_splits, gen_batch, labels, score
from anjo.sleep.mind.apd.arms import night_as_of, seen_by
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import MAX_PREVALENCE, PROBLEM_FAMILIES, CoreSplit, FindingReport, Life, ProblemFile
from anjo.sleep.mind.spark_provider import FIREWORKS


def _saved(root: Path, split: CoreSplit, prevalence: int) -> tuple[DataDir, Life, ProblemFile]:
    spec = next(s for s in core_splits.core_specs(split) if s.prevalence == prevalence)
    made = gen_batch.generate_life(spec, backends.fake_backend(FIREWORKS))
    data = DataDir(root / "data")
    data.save_life(made.life, made.problems)
    return data, made.life, made.problems


def test_labels_map_planted_items_to_event_ids(tmp_path: Path) -> None:
    # Arrange
    data, life, problems = _saved(tmp_path, "core-train", MAX_PREVALENCE)
    out = tmp_path / "out"

    # Act
    done = labels.export(data, "core-train", out)

    # Assert: the stream is every visible evidence item, in the life's order, numbered from 0
    records = labels.read_events(out / "events" / f"{life.id}.jsonl")
    visible = [e for e in life.evidence if not e.held_out]
    assert [r["event_id"] for r in records] == [e.id for e in visible] and done.events == len(visible)
    assert [r["seq"] for r in records] == list(range(len(records)))
    assert {r["life_or_person"] for r in records} == {life.id} and all(r["t"] and r["kind"] for r in records)
    by_id = {r["event_id"]: r for r in records}
    # ... and each planted item is located in it
    found = json.loads((out / "labels" / f"{life.id}.json").read_text(encoding="utf-8"))
    assert [label["item_id"] for label in found["items"]] == [item.id for item in problems.items]
    for label, item in zip(found["items"], problems.items, strict=True):
        assert label["origin_ids"] == list(item.origin_ids)
        assert label["disambiguating_ids"] == list(item.disambiguating_ids)
        knowable = max(item.origin_ids, key=lambda i: by_id[i]["seq"])
        resolvable = max((*item.origin_ids, *item.disambiguating_ids), key=lambda i: by_id[i]["seq"])
        assert (label["knowable_event_id"], label["knowable_day"]) == (knowable, by_id[knowable]["day"])
        assert (label["resolvable_event_id"], label["resolvable_day"]) == (resolvable, by_id[resolvable]["day"])
        assert label["knowable_day"] < label["resolvable_day"] < label["surface_day"]
        assert label["cutoff_day"] == (item.surface_day - timedelta(days=1)).isoformat()
        assert label["actionable"] == (item.family in PROBLEM_FAMILIES)
        assert (label["family"], label["template"], label["resolution"]) == (item.family, item.template, item.resolution)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sealed"] is False and [entry["life_id"] for entry in manifest["lives"]] == [life.id]


def test_a_problem_free_life_exports_no_items(tmp_path: Path) -> None:
    data, life, _ = _saved(tmp_path, "core-train", 0)
    labels.export(data, "core-train", tmp_path / "out")
    assert json.loads((tmp_path / "out" / "labels" / f"{life.id}.json").read_text(encoding="utf-8"))["items"] == []


@pytest.mark.parametrize("split", ["core-val", "core-test"])
def test_val_and_test_labels_stay_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: CoreSplit) -> None:
    # Arrange: a sealed problem file that may not be read
    data, life, problems = _saved(tmp_path, split, 4)
    out = tmp_path / "out"

    def refuse(self: DataDir, life_id: str) -> ProblemFile:
        raise AssertionError("a sealed problem file was read")

    with monkeypatch.context() as patched:
        patched.setattr(DataDir, "sealed", refuse)

        # Act
        done = labels.export(data, split, out)

    # Assert: the stream only, and nothing of the items in it
    assert done.sealed and done.lives == (life.id,)
    assert sorted(path.name for path in out.iterdir()) == ["events", "manifest.json"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sealed"] is True and set(manifest["lives"][0]) == {"life_id", "field", "first_day", "last_day",
                                                                        "events"}
    written = "".join(path.read_text(encoding="utf-8") for path in out.rglob("*") if path.is_file())
    for item in problems.items:
        assert item.resolution not in written and item.template not in written and item.id not in written
    # ... and the training path refuses it
    with pytest.raises(StoreError):
        data.labelled_problems(life.id)
    with pytest.raises(ValueError):
        labels.export(data, "held_out", tmp_path / "apd")


def test_the_stream_rebuilds_what_the_judge_is_shown(tmp_path: Path) -> None:
    # Arrange: a core-test life, exported
    data, life, problems = _saved(tmp_path, "core-test", 4)
    labels.export(data, "core-test", tmp_path / "out")

    # Act
    rebuilt = labels.evidence_of(labels.read_events(tmp_path / "out" / "events" / f"{life.id}.jsonl"))

    # Assert: a finding citing the stream's ids is judged on the same prompt as B's, on any night
    assert rebuilt == tuple(e for e in life.evidence if not e.held_out)
    for day in life.days:
        report = FindingReport("f1", day, 0, "the plan", "it can't hold", 0.6, (rebuilt[0].id,), "suspicion")
        for item in (*problems.items, None):
            ours = score.judge_prompt(report, seen_by(rebuilt, night_as_of(day)), item)
            assert ours == score.judge_prompt(report, seen_by(life.evidence, night_as_of(day)), item)


def test_the_labels_command_exports_the_b12_subset(tmp_path: Path) -> None:
    data = tmp_path / "pilot"
    lines: list[str] = []
    assert cli.main(["--fake", "--data", str(data), "generate", "--split", "core-test"], out=lines.append) == 0
    assert cli.main(["--data", str(data), "labels", "--split", "core-test", "--subset", "b12", "--out",
                     str(tmp_path / "out")], out=lines.append) == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    b12 = core_splits.in_subset(DataDir(data).lives("core-test"), "b12")
    assert [entry["life_id"] for entry in manifest["lives"]] == [life.id for life in b12] and len(b12) == 12
    assert manifest["subset"] == "b12" and manifest["sealed"] is True
    assert "exported 12 lives of core-test (b12)" in lines[-1]
