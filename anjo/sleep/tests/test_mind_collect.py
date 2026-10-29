# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Collect MIND life records (Task 16): replay deciders on recorded points only, offline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind import points
from anjo.sleep.mind.collect import CollectRefused, collect_life
from anjo.sleep.mind.deciders import DECIDERS
from anjo.sleep.mind.expect import Coefficients
from anjo.sleep.night.nightlog import POINTS, SUMMARY
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.personas.model import load_person_file, to_visible_dict
from anjo.sleep.probes import relevance
from anjo.sleep.reports import mind
from anjo.sleep.sim.accelerated import PERSONAS
from anjo.sleep.tests.test_mind_deciders import FakeChat, _points, _world

PERSONA = PERSONAS / "film_founder.json"
COEFFICIENTS = Coefficients(
    intercept=-0.5, understand=0.4, unacted=0.3, days=-0.05, base_rate=0.3, n=12
)
NIGHT = date(2026, 11, 3)


def _write_summary(folder: Path, night: date, *, label: str = "development") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / SUMMARY).write_text(
        json.dumps({"night": night.isoformat(), "label": label, "status": "promoted"})
    )


def _write_labels(
    path: Path,
    relevant: set[tuple[str, str]],
    targets: tuple[str, ...],
    items: tuple[str, ...],
) -> relevance.FrozenLabels:
    pairs = {(t, i): (t, i) in relevant for t in targets for i in items}
    frozen = relevance.freeze("p" * 64, pairs, frozenset())
    path.write_text(
        json.dumps(
            {
                "prompt_sha256": frozen.prompt_sha256,
                "digest": frozen.digest,
                "labels": [[t, i, v] for (t, i), v in sorted(pairs.items())],
                "unreadable": [],
            },
            sort_keys=True,
        )
    )
    return frozen


def _life_root(tmp_path: Path, user_id: str = "p1") -> Path:
    return tmp_path / user_id


def test_collect_replays_all_deciders_on_recorded_points_only(tmp_path: Path) -> None:
    world, homework = _world(tmp_path / "run")
    life = _life_root(tmp_path)
    nights_dir = life / "nights"
    for offset in (0, 1):
        night = NIGHT + timedelta(days=offset)
        result = run_night_batch_with(
            world.deps(homework=homework, estimator=COEFFICIENTS, is_synthetic=lambda uid: True),
            {"p1": night},
        )["p1"]
        folder = nights_dir / night.isoformat()
        folder.mkdir(parents=True)
        (folder / SUMMARY).write_text(
            json.dumps({"night": night.isoformat(), "label": "development", "status": "promoted"})
        )
        (folder / POINTS).write_text((result.log_path.parent / POINTS).read_text(encoding="utf-8"))
    _write_summary(nights_dir / "2099-01-01", date(2099, 1, 1), label="development")
    _write_summary(nights_dir / "2099-01-02", date(2099, 1, 2), label="final")
    records = [
        json.loads(line)
        for line in (nights_dir / NIGHT.isoformat() / POINTS).read_text().splitlines()
        if line
    ]
    pre = points.point_from_record(records[0])
    targets = tuple(t.id for t in pre.raw.targets)
    pf = load_person_file(PERSONA)
    items = tuple(w.id for w in pf.held_out.new_work)
    _write_labels(life / "relevance.json", set(), targets, items)
    for target in pre.raw.targets:
        world.targets("p1").upsert(replace(target, person_id="p1"))
    (life / "targets.jsonl").write_text(
        (tmp_path / "run" / "p1" / "targets.jsonl").read_text(encoding="utf-8")
    )
    record = collect_life(
        nights_dir, PERSONA, chat=FakeChat({"action": "silence", "target_id": None})
    )
    assert len(record.nights) == 2
    for picks in record.nights:
        assert set(picks.study) >= set(DECIDERS)
    assert record.nights[0].study["D0"] == points.study_target(pre.choice, pre.raw)


def test_collect_uses_frozen_relevance_labels_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, homework = _world(tmp_path / "run")
    life = _life_root(tmp_path)
    result = run_night_batch_with(
        world.deps(homework=homework, estimator=COEFFICIENTS, is_synthetic=lambda uid: True),
        {"p1": NIGHT},
    )["p1"]
    folder = life / "nights" / NIGHT.isoformat()
    folder.mkdir(parents=True)
    (folder / SUMMARY).write_text(
        json.dumps({"night": NIGHT.isoformat(), "label": "development", "status": "promoted"})
    )
    (folder / POINTS).write_text((result.log_path.parent / POINTS).read_text(encoding="utf-8"))
    pre, _ = _points(result.log_path)
    targets = tuple(t.id for t in pre.raw.targets)
    pf = load_person_file(PERSONA)
    items = tuple(w.id for w in pf.held_out.new_work)
    frozen = _write_labels(life / "relevance.json", {(targets[0], items[0])}, targets, items)
    for target in pre.raw.targets:
        world.targets("p1").upsert(replace(target, person_id="p1"))
    (life / "targets.jsonl").write_text(
        (tmp_path / "run" / "p1" / "targets.jsonl").read_text(encoding="utf-8")
    )

    def forbid_label(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("collect must not label pairs")

    monkeypatch.setattr(relevance, "label_pairs", forbid_label)
    record = collect_life(
        life / "nights", PERSONA, chat=FakeChat({"action": "silence", "target_id": None})
    )
    assert record.labels.digest == frozen.digest
    report = mind.mind_report(
        [record],
        registered={record.life_id: frozen.digest},
        coefficients=COEFFICIENTS,
        resamples=20,
    )
    assert report.lives[0].life_id == "film_founder"


def test_collect_refuses_kevin_or_non_synthetic_ids(tmp_path: Path) -> None:
    life = _life_root(tmp_path, "kevin-demo")
    nights = life / "nights" / NIGHT.isoformat()
    _write_summary(nights, NIGHT)
    _write_labels(life / "relevance.json", set(), (), ())
    with pytest.raises(CollectRefused, match="kevin"):
        collect_life(nights, PERSONA)
    real = tmp_path / "u-4f2a1b2c"
    real.mkdir()
    rn = real / "nights" / NIGHT.isoformat()
    _write_summary(rn, NIGHT)
    _write_labels(real / "relevance.json", set(), (), ())
    with pytest.raises(CollectRefused, match="non-synthetic"):
        collect_life(rn, PERSONA)


def test_collect_arrivals_from_persona_held_out_not_visible(tmp_path: Path) -> None:
    world, homework = _world(tmp_path / "run")
    life = _life_root(tmp_path)
    result = run_night_batch_with(
        world.deps(homework=homework, estimator=COEFFICIENTS, is_synthetic=lambda uid: True),
        {"p1": NIGHT},
    )["p1"]
    folder = life / "nights" / NIGHT.isoformat()
    folder.mkdir(parents=True)
    (folder / SUMMARY).write_text(
        json.dumps({"night": NIGHT.isoformat(), "label": "development", "status": "promoted"})
    )
    (folder / POINTS).write_text((result.log_path.parent / POINTS).read_text(encoding="utf-8"))
    pre, _ = _points(result.log_path)
    pf = load_person_file(PERSONA)
    held_ids = {w.id for w in pf.held_out.new_work}
    _write_labels(
        life / "relevance.json",
        set(),
        tuple(t.id for t in pre.raw.targets),
        tuple(w.id for w in pf.held_out.new_work),
    )
    for target in pre.raw.targets:
        world.targets("p1").upsert(replace(target, person_id="p1"))
    (life / "targets.jsonl").write_text(
        (tmp_path / "run" / "p1" / "targets.jsonl").read_text(encoding="utf-8")
    )
    record = collect_life(
        life / "nights", PERSONA, chat=FakeChat({"action": "silence", "target_id": None})
    )
    assert {a.item_id for a in record.arrivals} <= held_ids
    assert len(record.arrivals) == len(pf.held_out.new_work)
    by_id = {w.id: w for w in pf.held_out.new_work}
    for arrival in record.arrivals:
        assert arrival.day == NIGHT + timedelta(days=by_id[arrival.item_id].day)
    visible = json.dumps(to_visible_dict(pf.visible))
    for item in pf.held_out.new_work:
        assert item.text not in visible
