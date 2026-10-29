"""The demo baseline picker (``tools.demo_baseline``): pooled development 4 vs 2C and 4 vs 2F on new_work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anjo.sleep.tools import demo_baseline as pick


def _write(root: Path, person: str, night: str, *, outcomes: list[dict[str, Any]]) -> None:
    record = {
        "night": night,
        "label": "development",
        "probes": "full",
        "arms": {},
        "pairs": {},
        "outcomes": outcomes,
    }
    folder = root / person / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "night-01.json").write_text(json.dumps(record), encoding="utf-8")


def _nw(pair: str, outcome: str, probe: str = "nw") -> dict[str, Any]:
    return {"probe": probe, "pair": pair, "kind": "new_work", "outcome": outcome, "flip": False}


def test_demo_baseline_picks_2C_when_4_beats_2C_more_than_2F(tmp_path: Path) -> None:
    """Arm 4 wins more against 2F than against 2C, so 2C is the stronger notes-only baseline."""
    for person in pick.pooled_lives():
        _write(
            tmp_path,
            person,
            "2026-10-28",
            outcomes=[
                _nw(pick.PAIR_2C, "loss", "a"),
                _nw(pick.PAIR_2F, "win", "b"),
            ],
        )
    got = pick.recommend(tmp_path)
    assert got["baseline"] == "2C"
    assert "2C stronger" in got["reason"]


def test_demo_baseline_picks_2F_when_2F_stronger_on_new_work(tmp_path: Path) -> None:
    for person in pick.pooled_lives():
        _write(
            tmp_path,
            person,
            "2026-10-28",
            outcomes=[
                _nw(pick.PAIR_2C, "win", "a"),
                _nw(pick.PAIR_2F, "loss", "b"),
            ],
        )
    got = pick.recommend(tmp_path)
    assert got["baseline"] == "2F"
    assert "2F stronger" in got["reason"]


def test_demo_baseline_inconclusive_defaults_to_2C(tmp_path: Path) -> None:
    for person in pick.pooled_lives():
        _write(
            tmp_path,
            person,
            "2026-10-28",
            outcomes=[
                _nw(pick.PAIR_2C, "win", "a"),
                _nw(pick.PAIR_2F, "win", "b"),
            ],
        )
    tied = pick.recommend(tmp_path)
    assert tied["baseline"] == "2C" and "inconclusive" in tied["reason"]

    empty = pick.recommend(tmp_path / "missing")
    assert empty["baseline"] == "2C" and "missing" in empty["reason"].lower()
