"""The cost report on fake results (hackathon): per-step nights, the person-month, the drivers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from anjo.sleep.night.cost import DEFAULT_PRICES
from anjo.sleep.reports import cost

PROMOTED_PARTS = {"extract": 0.01, "brief": 0.005, "practice": 0.2, "training": 0.107, "gates": 0.12, "mind": 0.0}
QUIET_PARTS = {"extract": 0.01, "brief": 0.005, "practice": 0.0, "training": 0.0, "gates": 0.0, "mind": 0.0}
MORNING_GPU = 600 * DEFAULT_PRICES.serve_gpu_per_hour / 3600.0  # the fake morning's 600 seconds


def _night(root: Path, person: str, folder: str, *, status: str = "promoted",
           parts: dict[str, float] | None = PROMOTED_PARTS, wonder: float = 0.0,
           morning: bool = True, judge_usd: float | None = 0.44, judge_calls: int | None = 22) -> None:
    base = root / person / "nights" / folder
    base.mkdir(parents=True)
    summary: dict[str, Any] = {"night": folder.split(".")[0], "status": status, "wonder_usd": wonder}
    if parts is None:
        summary |= {"cost_usd": 0.5}
    else:
        summary |= {"cost_parts": parts, "cost_usd": math.fsum(parts.values())}
    (base / "summary.json").write_text(json.dumps(summary))
    if not morning:
        return
    record: dict[str, Any] = {"step": "morning", "status": "ok", "seconds": 600}
    if judge_usd is None:
        record["judge_calls"] = judge_calls
    else:
        record |= {"judge_usd": judge_usd, "judge_calls": judge_calls, "judge_estimated": 0}
    (base / "log.jsonl").write_text(json.dumps(record) + "\n")


def test_a_night_is_broken_down_by_step_including_the_morning(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", wonder=0.227)
    found = cost.user_cost("p1", tmp_path)
    entry = found.nights[0]
    assert entry.promoted and entry.costed
    assert {line.step for line in entry.steps} == set(cost.STEPS)
    morning = next(line for line in entry.steps if line.step == "morning")
    assert math.isclose(morning.usd, 0.44 + MORNING_GPU, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(entry.ceiling_usd, 0.442, rel_tol=1e-9)
    text = cost.render(found)
    for label in ("practice", "training GPU", "gates", "extraction", "brief", "MIND", "morning", "wonder"):
        assert label in text
    assert "$0.7667" in text and "$0.2270" in text
    assert "ceiling parts $0.4420" in text


def test_the_month_at_the_observed_rate_and_training_every_night(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01")
    _night(tmp_path, "p1", "2026-11-02", status="nothing_new", parts=QUIET_PARTS, morning=False)
    _night(tmp_path, "p1", "2026-11-03")
    _night(tmp_path, "p1", "2026-11-04", status="nothing_new", parts=QUIET_PARTS, morning=False)
    found = cost.user_cost("p1", tmp_path)
    assert found.month is not None
    assert found.month.nights == 4 and found.month.promoted == 2
    assert math.isclose(found.month.promote_share, 0.5)
    assert found.month.at_pace is not None and math.isclose(found.month.at_pace, 30 * (0.5 * 0.442 + 0.5 * 0.015))
    assert found.month.every_night is not None and math.isclose(found.month.every_night, 30 * 0.442)
    text = cost.render(found)
    assert "$6.8550" in text and "$13.2600" in text
    assert "over the $10 ceiling by $3.2600" in text


def test_the_top_three_drivers_and_what_cutting_each_saves(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01")
    _night(tmp_path, "p1", "2026-11-02", status="nothing_new", parts=QUIET_PARTS, morning=False)
    _night(tmp_path, "p1", "2026-11-03")
    _night(tmp_path, "p1", "2026-11-04", status="nothing_new", parts=QUIET_PARTS, morning=False)
    found = cost.user_cost("p1", tmp_path)
    assert [driver.step for driver in found.drivers] == ["morning", "practice", "gates"]
    assert math.isclose(found.drivers[0].mean_usd, 2 * (0.44 + MORNING_GPU) / 4)
    assert math.isclose(found.drivers[0].month_usd, 30 * found.drivers[0].mean_usd)
    assert math.isclose(found.drivers[1].month_usd, 30 * 0.1)  # practice: $0.20 on half the nights
    text = cost.render(found)
    assert "each cut to zero" in text
    assert "1. morning" in text and "saves $11.5000 a person-month" in text


def test_a_summary_without_cost_parts_is_shown_but_not_priced(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", parts=None, morning=False)
    found = cost.user_cost("p1", tmp_path)
    entry = found.nights[0]
    assert not entry.costed and entry.steps == ()
    assert found.month is None
    text = cost.render(found)
    assert "$0.5000" in text and "no cost parts" in text


def test_a_night_without_cost_parts_is_left_out_of_a_partly_costed_month(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01")
    _night(tmp_path, "p1", "2026-11-02", parts=None, morning=False)
    found = cost.user_cost("p1", tmp_path)
    assert found.month is not None and found.month.nights == 1 and found.month.promoted == 1


def test_a_judge_logged_only_by_calls_is_estimated_at_the_hold_bound(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", judge_usd=None, judge_calls=22)
    entry = cost.user_cost("p1", tmp_path).nights[0]
    morning = next(line for line in entry.steps if line.step == "morning")
    assert morning.estimated
    assert math.isclose(morning.usd, 22 * cost.JUDGE_USD_PER_CALL + MORNING_GPU, rel_tol=1e-9)
    assert "estimated" in cost.render(cost.user_cost("p1", tmp_path))


def test_no_promoted_night_leaves_training_every_night_unpriced(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", status="nothing_new", parts=QUIET_PARTS, morning=False)
    found = cost.user_cost("p1", tmp_path)
    assert found.month is not None and found.month.promote_share == 0.0
    assert found.month.trained == 0
    assert found.month.every_night is None
    assert "training every night  -  (cannot be priced)" in cost.render(found)


def test_a_rejected_night_trained_and_prices_training_every_night(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", status="rejected", morning=False)
    found = cost.user_cost("p1", tmp_path)
    assert found.nights[0].trained and not found.nights[0].promoted
    assert found.month is not None
    assert found.month.promoted == 0 and found.month.trained == 1
    assert found.month.promote_share == 0.0  # the rate stays promoted/known
    assert found.month.every_night is not None
    assert math.isclose(found.month.every_night, 30 * 0.442)
    assert "training every night  $13.2600" in cost.render(found)


def test_a_rerun_folder_is_its_own_night(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01")
    _night(tmp_path, "p1", "2026-11-01.2")
    found = cost.user_cost("p1", tmp_path)
    assert len(found.nights) == 2
    assert "rerun" in cost.render(found)


def test_cli_reports_each_user_and_says_when_one_has_no_nights(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01")
    lines: list[str] = []
    assert cost.main(["--root", str(tmp_path), "--user", "p1", "--user", "nobody"], out=lines.append) == 0
    text = "\n".join(lines)
    assert "cost: p1" in text and "cost: nobody" in text and "no nights" in text


def test_an_accelerated_life_labels_its_month_forced_deep_and_shows_the_modeled_schedule(tmp_path: Path) -> None:
    for folder in ("2026-11-01", "2026-11-02"):
        _night(tmp_path, "p1", folder)
        summary_path = tmp_path / "p1" / "nights" / folder / "summary.json"
        summary_path.write_text(json.dumps({**json.loads(summary_path.read_text()), "accelerated": True}))
    found = cost.user_cost("p1", tmp_path)
    text = cost.render(found)
    assert "every night deep (accelerated runs force deep)  $13.2600" in text
    assert "at the observed rate" not in text
    assert [m.usage for m in found.modeled] == ["active", "quiet"]
    for entry in found.modeled:
        assert entry.deep + entry.naps == 30 and entry.cap_usd > 0
        assert f"modeled nap/deep schedule, {entry.usage} month (night.cost, ${entry.cap_usd:g} cap," in text
        assert f"({entry.deep} deep, {entry.naps} naps)" in text


def test_a_daily_life_keeps_the_observed_rate_line_unchanged(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01")
    found = cost.user_cost("p1", tmp_path)
    text = cost.render(found)
    assert "  at the observed rate  $13.2600" in text and "modeled" not in text and found.modeled == ()
