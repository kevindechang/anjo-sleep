"""The learning-curve report on fake results (hackathon): scores, intervals, status and the trend."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anjo.sleep.probes.score import BootstrapConfig
from anjo.sleep.reports import curve

BOOTSTRAP = BootstrapConfig(resamples=400, seed=7, ci=0.95)


def _stat(score: float, low: float | None = None, high: float | None = None) -> dict[str, Any]:
    ci = None if low is None or high is None else [low, high]
    return {"n": 8, "wins": 4, "ties": 1, "losses": 3, "score": score, "decisive_share": 0.57, "ci": ci}


def _night(root: Path, person: str, night: str, number: int, score: float, *, folder: str | None = None,
           status: str = "promoted", subset: str | None = None, arm3: bool = True,
           label: str = "development") -> None:
    base = root / person / "nights" / (folder or night)
    base.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps(
        {"night": night, "label": label, "status": status, "cost_usd": 0.2}))
    if not arm3:
        return
    arms = {"1": {"knowledge": _stat(0.5), "overall": _stat(0.5)}}
    arms["3"] = {
        "knowledge": _stat(score, round(score - 0.05, 3), round(score + 0.05, 3)),
        "interaction": _stat(score, round(score - 0.06, 3), round(score + 0.06, 3)),
        "overall": _stat(score, round(score - 0.04, 3), round(score + 0.04, 3)),
    }
    scores = root / person / "scores"
    scores.mkdir(exist_ok=True)
    record: dict[str, Any] = {"night": night, "label": label, "arms": arms}
    if subset is not None:
        record["subset"] = subset
    (scores / f"night-{number:02d}.json").write_text(json.dumps(record))


def _trend(found: curve.Curve, *, label: str = "development", probes: str = "full",
           series: str = "overall") -> curve.Trend:
    return next(entry for entry in found.trends
                if entry.label == label and entry.probes == probes and entry.series == series)


def test_each_night_shows_its_score_interval_and_status(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", 1, 0.50, status="promoted")
    _night(tmp_path, "p1", "2026-11-02", 2, 0.55, status="nothing_new")
    text = curve.render(curve.curve("p1", tmp_path, config=BOOTSTRAP))
    assert "2026-11-01" in text and "2026-11-02" in text
    assert "promoted" in text and "nothing_new" in text
    assert "0.55 [0.50,0.60]" in text  # knowledge: the score beside its 95% interval
    assert "0.55 [0.51,0.59]" in text  # overall: the headline column
    assert "knowledge" in text and "interaction" in text and "overall" in text


def test_trend_has_a_positive_slope_and_the_two_ends(tmp_path: Path) -> None:
    for number, score in enumerate((0.40, 0.45, 0.50, 0.55, 0.60), start=1):
        _night(tmp_path, "p1", f"2026-11-0{number}", number, score)
    found = curve.curve("p1", tmp_path, config=BOOTSTRAP)
    overall = _trend(found)
    assert overall.nights == 5
    assert overall.slope is not None and overall.slope > 0.0
    assert overall.ci is not None and overall.ci[0] > 0.0 and overall.ci[0] <= overall.ci[1]
    assert overall.first is not None and abs(overall.first - 0.45) < 1e-9
    assert overall.last is not None and abs(overall.last - 0.55) < 1e-9
    assert "slope/day" in curve.render(found)  # the x axis is the calendar, not the night count
    assert "slope/night" not in curve.render(found)


def test_fewer_than_three_scored_nights_is_insufficient(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", 1, 0.50)
    _night(tmp_path, "p1", "2026-11-02", 2, 0.52)
    text = curve.render(curve.curve("p1", tmp_path, config=BOOTSTRAP))
    assert "insufficient: 2 scored night(s), three are needed for a trend" in text
    assert "slope/day" not in text
    assert "2026-11-02" in text  # the nights we have are still reported


def test_a_night_without_an_arm_three_score_is_not_counted(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", 1, 0.50)
    _night(tmp_path, "p1", "2026-11-02", 2, 0.52, arm3=False)
    found = curve.curve("p1", tmp_path, config=BOOTSTRAP)
    assert [point.night for point in found.points] == ["2026-11-01"]
    assert "insufficient" in curve.render(found)


def test_a_daily_subset_and_a_full_morning_are_fitted_apart(tmp_path: Path) -> None:
    for number, score in enumerate((0.40, 0.45, 0.50, 0.55), start=1):
        _night(tmp_path, "p1", f"2026-11-0{number}", number, score, subset="daily-v1")
    _night(tmp_path, "p1", "2026-11-05", 5, 0.70)
    found = curve.curve("p1", tmp_path, config=BOOTSTRAP)
    daily = _trend(found, probes="daily-v1")
    full = _trend(found, probes="full")
    assert daily.nights == 4 and daily.slope is not None
    assert full.nights == 1 and full.slope is None
    assert "full" in curve.render(found)


def test_three_attempts_of_one_day_are_one_point_not_a_crash(tmp_path: Path) -> None:
    for attempt in ("", ".2", ".3"):
        _night(tmp_path, "p1", "2026-11-01", 1, 0.50, folder=f"2026-11-01{attempt}")
    found = curve.curve("p1", tmp_path, config=BOOTSTRAP)
    assert [point.night for point in found.points] == ["2026-11-01"]
    assert all(entry.slope is None for entry in found.trends)  # one day has no trend, never a raise
    assert "insufficient: 1 scored night(s)" in curve.render(found)


def test_a_rerun_folder_does_not_add_a_second_point_for_a_night(tmp_path: Path) -> None:
    for number, score in enumerate((0.40, 0.45, 0.50), start=1):
        day = f"2026-11-0{number}"
        for attempt in ("", ".2"):
            _night(tmp_path, "p1", day, number, score, folder=day + attempt)
    found = curve.curve("p1", tmp_path, config=BOOTSTRAP)
    assert [point.night for point in found.points] == ["2026-11-01", "2026-11-02", "2026-11-03"]
    overall = _trend(found)
    assert overall.nights == 3 and overall.slope is not None and overall.slope > 0.0
    assert "insufficient" not in curve.render(found)


def test_a_night_scored_on_both_sets_keeps_a_point_in_each(tmp_path: Path) -> None:
    for number, score in enumerate((0.40, 0.45, 0.50), start=1):
        day = f"2026-11-0{number}"
        _night(tmp_path, "p1", day, number, score, subset="daily-v1")
        _night(tmp_path, "p1", day, number + 10, round(score + 0.10, 3))
    found = curve.curve("p1", tmp_path, config=BOOTSTRAP)
    daily = _trend(found, probes="daily-v1")
    full = _trend(found, probes="full")
    assert len(found.points) == 6  # the full set does not hide the daily one, nor the other way
    assert daily.nights == 3 and daily.slope is not None
    assert full.nights == 3 and full.slope is not None


def test_development_and_final_nights_are_fitted_apart(tmp_path: Path) -> None:
    for number, score in enumerate((0.40, 0.45, 0.50, 0.55), start=1):
        _night(tmp_path, "p1", f"2026-11-0{number}", number, score)
    for number, score in enumerate((0.60, 0.62, 0.64), start=5):
        _night(tmp_path, "p1", f"2026-11-0{number}", number, score, label="final")
    found = curve.curve("p1", tmp_path, config=BOOTSTRAP)
    development = _trend(found, label="development")
    final = _trend(found, label="final")
    assert development.nights == 4 and final.nights == 3
    assert development.slope is not None and final.slope is not None
    assert development.slope != final.slope  # two answers, never one line across both
    text = curve.render(found)
    assert "development" in text and "final" in text


def test_cli_prints_the_report(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-01", 1, 0.50)
    lines: list[str] = []
    assert curve.main(["--root", str(tmp_path), "--user", "p1"], out=lines.append, config=BOOTSTRAP) == 0
    assert "learning curve: p1" in "\n".join(lines)
