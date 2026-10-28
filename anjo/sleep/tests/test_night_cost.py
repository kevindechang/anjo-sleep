# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's cost model (``python -m anjo.sleep.night.cost --from <night dir>``): one night's
cost by step from its summary and receipts, ranked, and a person-month projected at the current
settings beside the settings of the first live nights. Numbers only; no file but the night's."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.night import cost
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.morning import JUDGE_USD_PER_CALL

PRICES = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.8)
PARTS = {"extract": 0.02, "brief": 0.05, "practice": 0.40, "training": 0.12, "gates": 0.60, "mind": 0.01}


def _night(folder: Path, *, parts: dict[str, float] = PARTS, wonder: float = 0.25) -> Path:
    folder.mkdir(parents=True)
    steps: list[dict[str, Any]] = [
        {"step": "extract", "status": "ok", "items": 10},
        {"step": "brief", "status": "ok", "folded": 16},
        {"step": "practice", "status": "ok", "scenes": 16, "seconds": 400.0,
         "spark": {"person": {"calls": 1, "in": 3000, "out": 6000, "cut": 0}}},
        {"step": "train", "status": "ok", "receipt": {"gpu_minutes": 2.88, "gpu": "A100"}},
        {"step": "gates", "status": "ok", "seconds": 1200.0, "gate_seconds": {"general": 60.0, "collapse": 140.0, "safety": 1000.0}},
        {"step": "finish", "status": "ok", "night_status": "promoted"},
        {"step": "morning", "status": "ok", "seconds": 90.0, "replies": 28, "judge_calls": 22},
    ]
    (folder / "log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps), encoding="utf-8")
    summary = {"night": "2026-10-29", "status": "promoted", "failed_at": None, "trained": True,
               "cost_parts": parts, "cost_usd": math.fsum(parts.values()), "wonder_usd": wonder}
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder


def test_cost_model_sums_a_night_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = _night(tmp_path / "2026-10-29")
    summary, steps = cost.read_night(folder)
    lines = cost.night_lines(summary, steps, PRICES)
    by_step = {line.step: line for line in lines}

    # The summary's parts are all there, practice split into its Spark Open calls and the GPU's wait.
    gpu = 400.0 / 3600 * PRICES.serve_gpu_per_hour
    assert by_step["practice_gpu"].usd == pytest.approx(gpu) and by_step["practice_spark"].usd == pytest.approx(0.40 - gpu)
    assert math.fsum(line.usd for line in lines if line.step in cost.SUMMARY_LINES) == pytest.approx(summary["cost_usd"])
    assert by_step["training"].gpu_minutes == 2.88 and by_step["gates"].gpu_minutes == pytest.approx(20.0)
    # What the summary never holds: Wonder's homework and the morning run, outside the ceiling.
    assert by_step["homework"].usd == 0.25 and not by_step["homework"].in_ceiling
    assert by_step["morning_judge"].usd == pytest.approx(22 * JUDGE_USD_PER_CALL)
    assert by_step["morning_gpu"].usd == pytest.approx(90.0 / 3600 * PRICES.serve_gpu_per_hour)
    assert not by_step["mind"].in_ceiling  # §5.5: every part but MIND
    assert cost.ceiling_usd(lines) == pytest.approx(0.02 + 0.05 + 0.40 + 0.12 + 0.60)
    assert [line.usd for line in lines] == sorted((line.usd for line in lines), reverse=True)  # ranked, the biggest first

    # A month at the spec's formula, from a trained night and a quiet one.
    assert cost.month(1.0, 0.1, trained_share=0.5) == pytest.approx(30 * (0.5 * 1.0 + 0.5 * 0.1))
    measured = cost.measured(summary, steps, PRICES)
    before = cost.project(cost.BEFORE, cost.UNITS, PRICES, measured)
    current = cost.project(cost.CURRENT, cost.UNITS, PRICES, measured)
    assert before.trained["training"] == current.trained["training"] == pytest.approx(2.88 / 60 * 2.5)  # measured
    assert current.trained["gates"] < before.trained["gates"]  # the base's side is read back, not recomputed
    assert current.morning < before.morning / 10  # the light daily morning
    assert set(current.quiet) == {"extract", "brief"}

    assert cost.main(["--from", str(folder), "--prices", "1.25,4.25,2.5,1.8", "--trained-share", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "practice_spark" in out and "projected month" in out and "worst case" in out
    assert "placeholder" in out  # the judge's price per call is not measured yet, and says so
    assert "$0.6000" in out.split("projected")[0]  # the logged gates line, as measured


def test_a_night_without_cost_parts_is_refused(tmp_path: Path) -> None:
    folder = _night(tmp_path / "old")
    summary = json.loads((folder / "summary.json").read_text())
    del summary["cost_parts"]
    (folder / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit):
        cost.main(["--from", str(folder)])
