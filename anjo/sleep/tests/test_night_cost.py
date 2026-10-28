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
from anjo.sleep.night.deps import NightSettings, Prices
from anjo.sleep.night.sleep_pressure import SleepPolicy
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
    # A log from before the judge was metered: its calls at the hold's per-call bound, marked estimated.
    assert by_step["morning_judge"].usd == pytest.approx(22 * JUDGE_USD_PER_CALL) and by_step["morning_judge"].estimated
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
    assert set(current.quiet) == {"extract", "brief"} and current.nap_encode == 0.0
    # The morning: today's measured arms (165 s for the daily subset) scaled to the subset, its judge calls
    # estimated at the hold's per-call bound (no night's tokens yet), and the dailies.
    assert current.morning == pytest.approx(22 * JUDGE_USD_PER_CALL + 165.0 / 3600 * PRICES.serve_gpu_per_hour + 0.005)

    assert cost.main(["--from", str(folder), "--prices", "1.25,4.25,2.5,1.8"]) == 0
    out = capsys.readouterr().out
    assert "practice_spark" in out and "projected month" in out and "worst case" in out
    assert "month, active" in out and "month, quiet" in out  # one projection, from the pressure traces
    night_part = out.split("projected")[0]
    assert "morning_judge" in night_part and night_part.count("estimated") >= 2  # the line and its note say so
    assert "estimated at $0.02 a call" in out.split("projected")[1]  # the projection's judge, too
    assert cost.WONDER_MONTH_USD == NightSettings.__dataclass_fields__["wonder_usd_month"].default
    assert "$0.6000" in out.split("projected")[0]  # the logged gates line, as measured


def test_a_metered_morning_judge_is_its_settled_tokens(tmp_path: Path) -> None:
    """A night logged after the judge was metered: the line is what the ledger settled, estimated only
    when some call reported no tokens."""
    summary, steps = cost.read_night(_night(tmp_path / "night"))
    morning = {"step": "morning", "status": "ok", "seconds": 90.0, "judge_calls": 22, "judge_usd": 0.15}
    tokens = {line.step: line for line in cost.night_lines(summary, [*steps[:-1], {**morning, "judge_estimated": 0}],
                                                              PRICES)}["morning_judge"]
    assert tokens.usd == pytest.approx(0.15) and not tokens.estimated
    some = {line.step: line for line in cost.night_lines(summary, [*steps[:-1], {**morning, "judge_estimated": 2}],
                                                            PRICES)}["morning_judge"]
    assert some.usd == pytest.approx(0.15) and some.estimated


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

    # Today's night (2026-10-29: safety off, the base cache missed, so both sides) is its measured gates;
    # the cache halves them; turning the safety gate on would add more than they cost.
    names = ("today (10-29)", "+ base cache")
    assert [s.name for s in cost.SCENARIOS[:2]] == list(names) and not cost.ANCHOR.safety_gate
    anchor, cached = (by_name[n].trained["gates"] for n in names)
    assert anchor == pytest.approx(units.gates_anchor_s * serve) and cached == pytest.approx(anchor / 2)
    assert units.gates_anchor_s * 1.96 / 3600 == pytest.approx(0.214, abs=0.001)  # $0.214 at $1.96/h
    adds = cost.safety_adds(cost.ONE_SYSTEM, units, PRICES, night)  # one side: the base side is cached
    assert adds > anchor

    # MIND's lines: the encode on its own A100 start, or in the training container; training grows by the joint share.
    a100 = PRICES.train_gpu_per_hour / 3600
    assert one.trained["latent_encode"] == pytest.approx((units.encode_s + units.encode_load_s) * a100)
    assert by_name["encode in train"].trained["latent_encode"] == pytest.approx(units.encode_s * a100)
    assert one.nap_encode == one.trained["latent_encode"]  # a nap has no training container
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
    assert "today (10-29)" in out and "H100 rows: $3.95/h is Modal's list price, unverified" in out
    assert "active month" in out and "quiet month" in out and " deep)" in out


def test_cost_model_projects_a_month_from_a_pressure_trace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """One month projection: the headline's days go through the nights' own nap-or-deep decision. A nap
    wakes no Glimmer: extract, the brief, MIND on the CPU and, with MIND's core on, a share of one batched
    latent encode. A deep sleep comes when pressure crosses its threshold, at least every 7th night."""
    folder = _night(tmp_path / "2026-10-29")
    summary, steps = cost.read_night(folder)
    night = cost.measured(summary, steps, PRICES)
    current, one = (cost.project(s, cost.UNITS, PRICES, night) for s in (cost.CURRENT, cost.ONE_SYSTEM))
    assert set(one.quiet) == {"extract", "brief"}  # what a nap's ledger counts: no GPU step, MIND waits for deep sleep
    assert current.nap_encode == 0.0 < one.nap_encode  # the encode only with the core on: planned, not live
    policy = SleepPolicy()
    active, quiet = (cost.month_from_trace(u.days, policy, current) for u in (cost.ACTIVE, cost.QUIET))
    assert len(active.nights) == len(quiet.nights) == 30
    assert [n.reason for n in quiet.nights if n.kind == "deep"] == ["max_days"] * quiet.deep  # too little for pressure
    # her pressure, not the seventh day, wakes the active one; the cap, with Wonder's room held back, gives
    # neither more than it holds
    assert any(n.reason == "pressure" for n in active.nights) and active.deep >= quiet.deep
    assert quiet.usd <= active.usd
    deep, nap = math.fsum(current.trained.values()) + current.morning, math.fsum(current.quiet.values())
    assert quiet.usd == pytest.approx(quiet.deep * deep + quiet.naps * nap + cost.WONDER_MONTH_USD)  # Wonder spent
    assert active.usd <= policy.month_usd and quiet.usd <= policy.month_usd  # the cap holds the whole month

    # The nap's encode: one A100 start a person, shared by five, restored from a snapshot, or deferred.
    a100, units = PRICES.train_gpu_per_hour / 3600, cost.UNITS

    def nap_encode(**kw: Any) -> float:
        return cost.project(replace(cost.ONE_SYSTEM, **kw), units, PRICES, night).nap_encode

    assert nap_encode() == pytest.approx((units.encode_load_s + units.encode_s) * a100)
    assert nap_encode(encode_people=5) == pytest.approx((units.encode_load_s / 5 + units.encode_s) * a100)
    assert nap_encode(encode_snapshot=True) == pytest.approx((units.snapshot_load_s + units.encode_s) * a100)
    assert nap_encode(encode_deferred=True) == pytest.approx(units.encode_s * a100)  # paid at the next deep wake

    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps([{"turns": 60, "evidence": 12}] * 30))
    assert cost.main(["--from", str(folder), "--scenarios", "--pressure-trace", str(trace)]) == 0
    out = capsys.readouterr().out
    assert "active month" in out and "quiet month" in out and "trace month" in out and "encode deferred" in out
    assert "--trace" not in out and "month_cap" in out  # one projection: the pattern traces are gone
    with pytest.raises(SystemExit):
        cost.main(["--from", str(folder), "--trace", "DNNNNNN"])


def test_a_night_without_cost_parts_is_refused(tmp_path: Path) -> None:
    folder = _night(tmp_path / "old")
    summary = json.loads((folder / "summary.json").read_text())
    del summary["cost_parts"]
    (folder / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit):
        cost.main(["--from", str(folder)])


def test_a_month_from_a_pressure_trace_prints_each_nights_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = _night(tmp_path / "night")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps([{"turns": 60, "evidence": 12}, {"turns": 2, "evidence": 1}] * 15))
    assert cost.main(["--from", str(folder), "--pressure-trace", str(trace), "--month-usd", "10"]) == 0
    out = capsys.readouterr().out
    assert "a month from the pressure trace" in out and "of the month's $10 cap" in out
    assert "deep  pressure" in out and "nap   below_threshold" in out
    trace.write_text("[1, 2]")
    with pytest.raises(SystemExit, match="pressure trace"):
        cost.main(["--from", str(folder), "--pressure-trace", str(trace)])
