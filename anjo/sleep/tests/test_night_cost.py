# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night's cost model (``python -m anjo.sleep.night.cost --from <night dir>``): one night's
cost by step from its summary and receipts, ranked, and a person-month projected at the current
settings beside the settings of the first live nights. Numbers only; no file but the night's."""
from __future__ import annotations

import json
import math
from dataclasses import replace
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
    assert set(current.quiet) == {"extract", "brief", "mind", "latent_encode"} and current.quiet["latent_encode"] == 0.0

    assert cost.main(["--from", str(folder), "--prices", "1.25,4.25,2.5,1.8", "--trained-share", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "practice_spark" in out and "projected month" in out and "worst case" in out
    assert "placeholder" in out  # the judge's price per call is not measured yet, and says so
    assert "$0.6000" in out.split("projected")[0]  # the logged gates line, as measured


def test_cost_model_projects_a_batched_night(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The whole loop at each proposal: one wake for five people shares its cold start and idle window,
    the base side stays cached, the safety gate is off by default and what it adds is shown, and MIND's
    latent encode and joint training are lines of the night, the encode cheaper in the training container."""
    summary, steps = cost.read_night(_night(tmp_path / "2026-10-29"))
    night = cost.measured(summary, steps, PRICES)
    by_name = {s.name: cost.project(s, cost.UNITS, PRICES, night) for s in cost.SCENARIOS}
    one, batch = by_name["one system"], by_name["batch of 5"]
    serve = PRICES.serve_gpu_per_hour / 3600
    units = cost.UNITS
    assert one.trained["serve_overhead"] == pytest.approx((units.wake_s + 600 + units.train_idle_s) * serve)
    assert batch.trained["serve_overhead"] == pytest.approx(((units.wake_s + 600) / 5 + units.train_idle_s) * serve)
    assert batch.trained["gates"] == one.trained["gates"]  # each person's checkpoint side; the base side cached

    # The anchor night (both sides, safety on) is the 77 measured minutes; safety off is its own step, and
    # cuts most of it; the cache then halves what is left.
    names = ("10-28.4 (safety on)", "+ safety off", "+ base cache")
    assert [s.name for s in cost.SCENARIOS[:3]] == list(names)
    anchor, off, cached = (by_name[n].trained["gates"] for n in names)
    assert anchor == pytest.approx(units.gates_anchor_s * serve) and cached == pytest.approx(off / 2)
    adds = cost.safety_adds(cost.ONE_SYSTEM, units, PRICES, night)  # one side: the base side is cached
    assert off < anchor / 4 and off + 2 * adds == pytest.approx(anchor)

    # MIND's lines: the encode on its own A100 start, or in the training container; training grows by the joint share.
    a100 = PRICES.train_gpu_per_hour / 3600
    assert one.trained["latent_encode"] == pytest.approx((units.encode_s + units.encode_load_s) * a100)
    assert by_name["encode in train"].trained["latent_encode"] == pytest.approx(units.encode_s * a100)
    assert one.quiet["latent_encode"] == one.trained["latent_encode"]  # a nap has no training container
    assert one.trained["training"] == pytest.approx(2.88 / 60 * 2.5 * (1 + units.joint_training_share))

    # Practice by role: a third-party critic is priced apart; Glimmer as the person moves its call to the GPU.
    deepseek, glimmer = by_name["critic=deepseek"], by_name["person=glimmer"]
    assert deepseek.trained["practice_critic"] > 0
    assert deepseek.trained["practice_spark"] < one.trained["practice_spark"]
    assert glimmer.trained["practice_gpu"] > one.trained["practice_gpu"]
    total = {name: math.fsum(p.trained.values()) for name, p in by_name.items()}
    assert total["all"] < min(t for name, t in total.items() if name != "all")

    assert cost.main(["--from", str(tmp_path / "2026-10-29"), "--prices", "1.25,4.25,2.5,1.8", "--scenarios"]) == 0
    out = capsys.readouterr().out
    assert "batch of 5" in out and "safety on adds" in out and "latent_encode" in out
    assert "10-28.4 (safety on)" in out and "H100 rows: $3.95/h is Modal's list price, unverified" in out


def test_cost_model_projects_a_month_from_a_pressure_trace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every night naps (CPU, plus the day's latent encode, its one GPU cost); a deep sleep comes when
    pressure crosses its threshold, at least every 7th night. A month is a trace of the two."""
    folder = _night(tmp_path / "2026-10-29")
    summary, steps = cost.read_night(folder)
    night = cost.measured(summary, steps, PRICES)
    one = cost.project(cost.ONE_SYSTEM, cost.UNITS, PRICES, night)
    deep, nap = math.fsum(one.trained.values()) + one.morning, math.fsum(one.quiet.values())
    assert set(one.quiet) == {"extract", "brief", "mind", "latent_encode"}  # no practice, training or gates on a nap
    assert cost.ACTIVE.deep_share == pytest.approx(2.5 / 7) and cost.QUIET.deep_share == pytest.approx(1 / 7)
    assert cost.trace_month(one, cost.QUIET) == pytest.approx(30 * (deep / 7 + nap * 6 / 7))
    assert cost.trace_month(one, cost.QUIET) < cost.trace_month(one, cost.ACTIVE) < 30 * deep
    for wrong in ("", "DNNNNNNN", "DXN"):  # seven naps running break the 7-day cap
        with pytest.raises(ValueError):
            cost.Trace("bad", wrong)

    # The nap's encode: one A100 start a person, shared by five, restored from a snapshot, or deferred.
    a100, units = PRICES.train_gpu_per_hour / 3600, cost.UNITS

    def nap_encode(**kw: Any) -> float:
        return cost.project(replace(cost.ONE_SYSTEM, **kw), units, PRICES, night).quiet["latent_encode"]

    assert nap_encode() == pytest.approx((units.encode_load_s + units.encode_s) * a100)
    assert nap_encode(encode_people=5) == pytest.approx((units.encode_load_s / 5 + units.encode_s) * a100)
    assert nap_encode(encode_snapshot=True) == pytest.approx((units.snapshot_load_s + units.encode_s) * a100)
    assert nap_encode(encode_deferred=True) == pytest.approx(units.encode_s * a100)  # paid at the next deep wake

    assert cost.main(["--from", str(folder), "--scenarios", "--trace", "DNNDNNN"]) == 0
    out = capsys.readouterr().out
    assert "active 2.5/wk" in out and "quiet 1.0/wk" in out and "trace 2.0/wk" in out and "encode deferred" in out
    with pytest.raises(SystemExit):
        cost.main(["--from", str(folder), "--scenarios", "--trace", "NNNNNNN"])


def test_a_night_without_cost_parts_is_refused(tmp_path: Path) -> None:
    folder = _night(tmp_path / "old")
    summary = json.loads((folder / "summary.json").read_text())
    del summary["cost_parts"]
    (folder / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit):
        cost.main(["--from", str(folder)])
