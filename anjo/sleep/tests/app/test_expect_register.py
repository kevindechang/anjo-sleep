# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Task 16: register and load the ``p_needed`` estimator. Offline only."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind import expect
from anjo.sleep.mind.expect_register import (
    ESTIMATOR,
    EstimatorError,
    fit_from_logs,
    load_estimator,
    save_estimator,
)
from anjo.sleep.night.nightlog import POINTS, SUMMARY
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.sim.accelerated import PERSONAS
from anjo.sleep.tests.night_fakes import NIGHT as FAKE_NIGHT, World, own, step, thread, window
from anjo.sleep.tests.test_mind_deciders import StudyingHomework, _points, _world

FOUNDER_PERSONA = PERSONAS / "founder.json"

COEFFICIENTS = expect.Coefficients(
    intercept=-0.5, understand=0.4, unacted=0.3, days=-0.05, base_rate=0.3, n=12
)


def _write_summary(folder: Path, night: date, *, label: str = "development") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / SUMMARY).write_text(
        json.dumps({"night": night.isoformat(), "label": label, "status": "promoted"})
    )


def _write_labels(path: Path, targets: tuple[str, ...], items: tuple[str, ...]) -> None:
    from anjo.sleep.probes import relevance

    pairs = {(t, i): False for t in targets for i in items}
    frozen = relevance.freeze("prompt", pairs, frozenset())
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
        + "\n",
        encoding="utf-8",
    )


def _founder_world(tmp_path: Path) -> tuple[Any, Any]:

    world = World(tmp_path / "run")
    world.planned[("founder", FAKE_NIGHT)] = [own("founder", FAKE_NIGHT, i) for i in range(3)]
    world.targets("founder").upsert(
        replace(thread("founder", "sorting black film"), id="founder-t1", person_id="founder")
    )
    world.targets("founder").upsert(
        replace(thread("founder", "pricing the pilot"), id="founder-t2", person_id="founder")
    )
    return world, StudyingHomework(world)


def _founder_life_with_dual_attempts(tmp_path: Path) -> Path:
    """Development and final attempts on the same night; only the development row may enter the fit."""
    world, homework = _founder_world(tmp_path)
    life = tmp_path / "founder"
    result = run_night_batch_with(
        world.deps(homework=homework, estimator=COEFFICIENTS, is_synthetic=lambda uid: True),
        {"founder": FAKE_NIGHT},
    )["founder"]
    pre, _ = _points(result.log_path)
    pf = load_person_file(FOUNDER_PERSONA)
    items = tuple(w.id for w in pf.held_out.new_work)
    target_ids = tuple(t.id for t in pre.raw.targets)
    life.mkdir(parents=True, exist_ok=True)
    _write_labels(life / "relevance.json", target_ids, items)
    for target in pre.raw.targets:
        world.targets("founder").upsert(replace(target, person_id="founder"))
    target_lines = []
    for line in (
        (tmp_path / "run" / "founder" / "targets.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if not line:
            continue
        record = json.loads(line)
        if record.get("op") == "record_expectation":
            record["expectation"]["outcome"] = "needed"
        target_lines.append(json.dumps(record))
    (life / "targets.jsonl").write_text("\n".join(target_lines) + "\n", encoding="utf-8")
    dev = life / "nights" / FAKE_NIGHT.isoformat()
    _write_summary(dev, FAKE_NIGHT, label="development")
    (dev / POINTS).write_text((result.log_path.parent / POINTS).read_text(encoding="utf-8"))
    final = life / "nights" / f"{FAKE_NIGHT.isoformat()}.2"
    _write_summary(final, FAKE_NIGHT, label="final")
    final_points = [
        json.loads(line) for line in (dev / POINTS).read_text(encoding="utf-8").splitlines() if line
    ]
    study_target = pre.choice.target_id if pre.choice else target_ids[0]
    for record in final_points:
        if record.get("phase") != "pre_study" or not record.get("choice"):
            continue
        record["choice"] = {
            **record["choice"],
            "target_id": study_target,
            "want": {**record["choice"]["want"], "kind": "follow_up", "target_id": study_target},
        }
        for target in record["targets"]:
            if target["id"] == study_target:
                target["revision"] = 6
                target["acted_revision"] = 1
    (final / POINTS).write_text(
        "\n".join(json.dumps(record) for record in final_points) + "\n", encoding="utf-8"
    )
    return life


def test_fit_refuses_held_out_rows(tmp_path: Path) -> None:
    from anjo.sleep.mind import expect_register

    life = _founder_life_with_dual_attempts(tmp_path)
    rows = expect_register._fit_rows_from_root(
        life, personas_dir=PERSONAS, split="development", is_synthetic=lambda user_id: user_id == "founder"
    )
    assert len(rows) == 1
    dev_folder = life / "nights" / FAKE_NIGHT.isoformat()
    final_folder = life / "nights" / f"{FAKE_NIGHT.isoformat()}.2"
    pre_dev = _points(dev_folder / SUMMARY)[0]
    pre_final = _points(final_folder / SUMMARY)[0]
    study_target = pre_dev.choice.target_id if pre_dev.choice else pre_dev.raw.targets[0].id
    dev_target = next(t for t in pre_dev.raw.targets if t.id == study_target)
    final_target = next(t for t in pre_final.raw.targets if t.id == study_target)
    dev_kind = pre_dev.choice.want.kind if pre_dev.choice and pre_dev.choice.want else "understand"
    dev_row = expect.fit_row(dev_target, dev_kind, pre_dev.raw.as_of, "needed", "development")
    final_row = expect.fit_row(
        final_target, "follow_up", pre_final.raw.as_of, "needed", "development"
    )
    assert dev_row is not None and final_row is not None
    assert rows[0].features == dev_row.features
    assert rows[0].features.unacted != final_row.features.unacted
    with pytest.raises(ValueError, match="development"):
        expect.fit([dev_row, expect.FitRow(final_row.features, final_row.needed, "held_out")])


def test_fit_refuses_non_synthetic_development_logs(tmp_path: Path) -> None:
    life = _founder_life_with_dual_attempts(tmp_path)
    with pytest.raises(EstimatorError, match="non-synthetic"):
        fit_from_logs([life], is_synthetic=lambda user_id: False, personas_dir=PERSONAS)


def _mapped_founder_life(tmp_path: Path, user_id: str) -> Path:
    original = _founder_life_with_dual_attempts(tmp_path)
    mapped = original.rename(tmp_path / user_id)
    # Match the fixture's ownership fields to the account that actually owns the logs.
    targets = mapped / "targets.jsonl"
    targets.write_text(targets.read_text(encoding="utf-8").replace('"founder"', json.dumps(user_id)))
    return mapped


def test_fit_mapped_uuid_uses_registered_persona_and_account_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.sleep.mind import expect_register

    user_id = "6f8063cb-7e48-4dc8-8d4e-26665047418e"
    life = _mapped_founder_life(tmp_path, user_id)
    seen: list[str] = []
    fitted_rows: list[list[object]] = []

    def synthetic(uid: str) -> bool:
        seen.append(uid)
        return uid == user_id

    # One fixture life gives one row, too few for the solver: check the rows the fit receives.
    monkeypatch.setattr(expect_register.expect, "fit", lambda rows: fitted_rows.append(list(rows)))
    fit_from_logs([life], is_synthetic=synthetic, personas_dir=PERSONAS, lives=[("founder", user_id)])
    assert [len(rows) for rows in fitted_rows] == [1]
    assert seen == [user_id]
    with pytest.raises(EstimatorError, match="refuses account"):
        fit_from_logs([life], is_synthetic=synthetic, personas_dir=PERSONAS)
    with pytest.raises(EstimatorError, match="non-synthetic"):
        fit_from_logs(
            [life], is_synthetic=lambda uid: False, personas_dir=PERSONAS,
            lives=[("founder", user_id)],
        )
    with pytest.raises(EstimatorError, match="refuses account"):
        fit_from_logs(
            [life], is_synthetic=synthetic, personas_dir=PERSONAS,
            lives=[("researcher", "8a701d9d-0730-4df8-914f-4b8f627a6888")],
        )


@pytest.mark.parametrize("user_id", ["u-abcd", "kevin-test"])
def test_mapped_fit_keeps_collect_refusals(tmp_path: Path, user_id: str) -> None:
    from anjo.sleep.mind.collect import CollectRefused

    life = _mapped_founder_life(tmp_path, user_id)
    with pytest.raises(CollectRefused, match="refusing"):
        fit_from_logs(
            [life], is_synthetic=lambda uid: True, personas_dir=PERSONAS,
            lives=[("founder", user_id)],
        )


@pytest.mark.parametrize("lives", [
    [("stranger", "account")],
    [("founder", "one"), ("founder", "two")],
    [("founder", "one"), ("researcher", "one")],
])
def test_fit_mapping_cannot_widen_registered_lives(lives) -> None:
    with pytest.raises(EstimatorError, match="unregistered|duplicate"):
        fit_from_logs([], is_synthetic=lambda uid: True, lives=lives)


def test_expect_register_round_trip_json_hash(tmp_path: Path) -> None:
    path = tmp_path / "estimator.json"
    fitted = expect.Coefficients(0.1, -0.2, 0.05, -0.01, base_rate=0.4, n=9, undated_days=7.0)
    save_estimator(fitted, path)
    loaded = load_estimator(path)
    assert loaded == fitted
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["sha256"] == raw["sha256"].lower() and len(raw["sha256"]) == 64
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["intercept"] = 9.0
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n")
    with pytest.raises(EstimatorError, match="sha256"):
        load_estimator(path)
    broken = json.loads(path.read_text(encoding="utf-8"))
    broken["intercept"] = 5.0
    broken["n"] = None
    path.write_text(json.dumps(broken, sort_keys=True) + "\n")
    with pytest.raises(EstimatorError, match="missing required coefficients"):
        load_estimator(path)
    save_estimator(fitted, path)
    almost = json.loads(path.read_text(encoding="utf-8"))
    almost["n"] = 4.9
    payload = {
        key: almost[key]
        for key in (
            "version",
            "intercept",
            "understand",
            "unacted",
            "days",
            "undated_days",
            "base_rate",
            "n",
        )
    }
    almost["sha256"] = (
        __import__("hashlib")
        .sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        )
        .hexdigest()
    )
    path.write_text(json.dumps(almost, sort_keys=True) + "\n")
    with pytest.raises(EstimatorError, match="n must be an integer"):
        load_estimator(path)


def test_registered_estimator_loads_in_build_for_synthetic_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from anjo.sleep.app import night
    from anjo.sleep.app.cleared_sessions import register_synthetic
    from anjo.sleep.mind import expect_register
    from anjo.sleep.night.deps import NightOptions
    from anjo.sleep.tests.app.test_night_wiring import NoTrainer, _night_env

    path = tmp_path / "estimator.json"
    save_estimator(COEFFICIENTS, path)
    monkeypatch.setattr(expect_register, "ESTIMATOR", path)
    monkeypatch.setenv("ANJO_SLEEP_HOST", "1")
    _night_env(monkeypatch, tmp_path)
    register_synthetic(
        "synthetic-founder",
        at=__import__("datetime").datetime(
            2026, 10, 28, 9, 0, tzinfo=__import__("datetime").timezone.utc
        ),
    )
    deps = night.build(
        NoTrainer(),
        NightOptions(night=date(2026, 10, 28), personas={"synthetic-founder": "founder"}),
    )
    assert deps.estimator == COEFFICIENTS


def test_night_records_expectation_when_estimator_registered(tmp_path: Path) -> None:
    world, homework = _world(tmp_path / "w")
    result = run_night_batch_with(
        world.deps(homework=homework, estimator=COEFFICIENTS, is_synthetic=lambda uid: True),
        {"p1": FAKE_NIGHT},
    )["p1"]
    [made] = world.targets("p1").expectations()
    end = window(FAKE_NIGHT)[1]
    assert made.target_id == "p1-t1" and made.made_at == end
    assert step(result.log_path, "choose")["expectation"] == made.id


def test_night_skips_expectation_for_real_person_with_estimator(tmp_path: Path) -> None:
    world, homework = _world(tmp_path / "w")
    run_night_batch_with(
        world.deps(homework=homework, estimator=COEFFICIENTS, is_synthetic=lambda uid: False),
        {"p1": FAKE_NIGHT},
    )
    assert world.targets("p1").expectations() == []


def test_committed_estimator_placeholder_loads_as_none() -> None:
    assert load_estimator(ESTIMATOR) is None
