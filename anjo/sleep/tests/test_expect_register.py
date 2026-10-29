# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Task 16: register and load the ``p_needed`` estimator. Offline only."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from anjo.sleep.mind import expect
from anjo.sleep.mind.expect_register import (
    ESTIMATOR,
    EstimatorError,
    fit_from_logs,
    load_estimator,
    save_estimator,
)
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.tests.night_fakes import NIGHT as FAKE_NIGHT, step, window
from anjo.sleep.tests.test_mind_deciders import _world

COEFFICIENTS = expect.Coefficients(
    intercept=-0.5, understand=0.4, unacted=0.3, days=-0.05, base_rate=0.3, n=12
)


def test_fit_refuses_held_out_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development"):
        fit_from_logs([tmp_path], split="held_out")
    rows = [
        expect.FitRow(expect.Features(True, 1, 7.0), True, "development"),
        expect.FitRow(expect.Features(False, 0, 3.0), False, "held_out"),
    ]
    with pytest.raises(ValueError, match="development"):
        expect.fit(rows)


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


def test_committed_estimator_placeholder_loads_as_none() -> None:
    assert load_estimator(ESTIMATOR) is None
