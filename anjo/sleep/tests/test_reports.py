# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The results table and chart (Task 12a): every arm per night, general skills and safety, with
development and final results labeled apart."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.reports import chart, results
from anjo.sleep.reports.results import ResultsError

ARMS = ("1", "2C", "2F", "3", "3x", "4")


def _night(root: Path, person: str, date: str, number: int, *, label: str | None = "development",
           safety: bool = True, scored: bool = True) -> None:
    folder = root / person / "nights" / date
    folder.mkdir(parents=True)
    summary: dict[str, Any] = {
        "status": "promoted", "night": date, "cost_usd": 0.31,
        "gates": {"general": {"passed": True, "value": 0.5, "threshold": 2.0},
                  "collapse": {"passed": True, "value": 0.0, "threshold": 1.0},
                  "safety": {"passed": safety, "value": 0.0 if safety else 1.0, "threshold": 0.0}},
    }
    if label is not None:
        summary |= {"accelerated": True, "git_sha": "ab" * 20, "label": label, "day_index": number - 1}
    (folder / "summary.json").write_text(json.dumps(summary))
    if scored:
        scores = root / person / "scores"
        scores.mkdir(exist_ok=True)
        arms = {arm: {"knowledge": 0.4 + 0.02 * number + 0.01 * i, "style": 0.5} for i, arm in enumerate(ARMS)}
        (scores / f"night-{number:02d}.json").write_text(json.dumps({"night": date, "label": label, "arms": arms}))


def test_table_shows_every_arm_per_night_with_general_and_safety(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-03", 1)
    _night(tmp_path, "p1", "2026-11-04", 2, safety=False)
    text = results.table(["p1"], root=tmp_path)
    for arm in ARMS:
        assert f"arm {arm}" in text
    assert "general" in text and "safety" in text
    lines = {line.split("|")[0].strip(): line for line in text.splitlines() if "|" in line}
    assert "ok" in lines["2026-11-03"] and "FAIL" in lines["2026-11-04"]
    assert "0.42/0.50" in lines["2026-11-03"]  # arm 1, knowledge/style


def test_development_and_final_are_labeled_apart(tmp_path: Path) -> None:
    _night(tmp_path, "dev", "2026-11-03", 1, label="development")
    _night(tmp_path, "fin", "2026-11-05", 1, label="final")
    text = results.table(["dev", "fin"], root=tmp_path)
    assert text.index("DEVELOPMENT") < text.index("dev") < text.index("FINAL") < text.index("fin")
    development = text[text.index("DEVELOPMENT"):text.index("FINAL")]
    assert "2026-11-05" not in development and "2026-11-03" in development


def test_a_night_without_a_label_is_refused(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-03", 1, label=None)
    with pytest.raises(ResultsError, match="label"):
        results.table(["p1"], root=tmp_path)


def test_a_night_without_scores_shows_dashes_and_no_nights_says_so(tmp_path: Path) -> None:
    _night(tmp_path, "p1", "2026-11-03", 1, scored=False)
    assert " - " in results.table(["p1"], root=tmp_path)
    assert "no nights" in results.table(["nobody"], root=tmp_path)


def test_chart_draws_every_arm_in_labeled_panels(tmp_path: Path) -> None:
    for number, date in enumerate(("2026-11-03", "2026-11-04", "2026-11-05"), start=1):
        _night(tmp_path, "dev", date, number, label="development")
    _night(tmp_path, "fin", "2026-11-05", 1, label="final")
    out = chart.render(["dev", "fin"], root=tmp_path, out=tmp_path / "charts" / "nights.svg")
    svg = out.read_text(encoding="utf-8")
    assert out == tmp_path / "charts" / "nights.svg" and svg.startswith("<svg")
    assert "DEVELOPMENT" in svg and "FINAL" in svg
    for arm in ARMS:
        assert f"arm {arm}" in svg
    assert "general skills" in svg and "safety" in svg
    assert svg == chart.render(["dev", "fin"], root=tmp_path, out=tmp_path / "again.svg").read_text(encoding="utf-8")


def test_chart_escapes_person_ids(tmp_path: Path) -> None:
    _night(tmp_path, "a<b&c", "2026-11-03", 1)
    svg = chart.render(["a<b&c"], root=tmp_path, out=tmp_path / "c.svg").read_text(encoding="utf-8")
    assert "a<b&c" not in svg and "a&lt;b&amp;c" in svg
