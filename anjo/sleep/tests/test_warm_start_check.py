# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
"""Warm-start vs clean-slate development comparison (Task 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anjo.sleep.tests.test_thresholds import T
from anjo.sleep.train import warm_start_check as wsc

COMP_3V1 = wsc.COMP_3V1
COMP_3V3X = wsc.COMP_3V3X


def _stats(
    *,
    score: float,
    ci_low: float,
    ci_high: float,
    decisive: float,
    per_life: dict[str, bool] | None = None,
    n: int = 40,
) -> dict[str, Any]:
    wins = int(round(score * n))
    ties = n - wins
    lives = per_life or {
        "founder": True,
        "founder_donor": False,
        "researcher": False,
        "researcher_donor": False,
    }
    return {
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": 0,
        "score": score,
        "decisive_share": decisive,
        "ci": [ci_low, ci_high],
        "per_life": {
            life: {
                "n": 10,
                "wins": 10 if ahead else 0,
                "ties": 0 if ahead else 10,
                "losses": 0,
                "score": 1.0 if ahead else 0.5,
                "ahead": ahead,
            }
            for life, ahead in lives.items()
        },
    }


def _record(life: str, night: str, policy: str, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "life": life,
        "night": night,
        "label": "development",
        "metadata": {"training_policy": policy},
        "outcomes": outcomes,
    }


def _outcome_rows(
    comparison: str, dimension: str, wins: int, losses: int = 0
) -> list[dict[str, Any]]:
    kinds = ["new_work", "vocab"] if dimension == "knowledge" else ["tone_hold", "taste_ab"]
    rows: list[dict[str, Any]] = []
    for i in range(wins):
        rows.append(
            {
                "probe": f"w{i}-{comparison}-{dimension}",
                "pair": comparison,
                "kind": kinds[i % 2],
                "outcome": "win",
                "flip": False,
            }
        )
    for i in range(losses):
        rows.append(
            {
                "probe": f"l{i}-{comparison}-{dimension}",
                "pair": comparison,
                "kind": kinds[i % 2],
                "outcome": "loss",
                "flip": False,
            }
        )
    return rows


def _scoped(life: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "probe": f"{life}-{row['probe']}"} for row in rows]


def _paired_nights(
    warm_outcomes: list[dict[str, Any]],
    clean_outcomes: list[dict[str, Any]],
    *,
    night: str = "2026-10-28",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lives = T["development"]["pooled_lives"]
    warm = [_record(life, night, wsc.WARM_LABEL, _scoped(life, warm_outcomes)) for life in lives]
    clean = [_record(life, night, wsc.CLEAN_LABEL, _scoped(life, clean_outcomes)) for life in lives]
    return warm, clean


def test_warm_start_check_picks_higher_scoring_policy_per_dimension() -> None:
    warm_stats = _stats(score=0.62, ci_low=0.56, ci_high=0.68, decisive=0.60)
    clean_stats = _stats(score=0.52, ci_low=0.48, ci_high=0.56, decisive=0.52)
    assert wsc.pick_policy(warm_stats, clean_stats, T) == "warm_start"
    assert wsc.pick_policy(clean_stats, warm_stats, T) == "clean_slate"

    warm, clean = _paired_nights(
        _outcome_rows(COMP_3V1, "knowledge", 30, 10)
        + _outcome_rows(COMP_3V3X, "knowledge", 28, 12),
        _outcome_rows(COMP_3V1, "knowledge", 22, 18)
        + _outcome_rows(COMP_3V3X, "knowledge", 20, 20),
    )
    extra = _outcome_rows(COMP_3V1, "interaction", 8, 2) + _outcome_rows(
        COMP_3V3X, "interaction", 8, 2
    )
    extra_clean = _outcome_rows(COMP_3V1, "interaction", 4, 6) + _outcome_rows(
        COMP_3V3X, "interaction", 4, 6
    )
    for life in T["development"]["pooled_lives"]:
        warm.append(_record(life, "2026-10-29", wsc.WARM_LABEL, _scoped(life, extra)))
        clean.append(_record(life, "2026-10-29", wsc.CLEAN_LABEL, _scoped(life, extra_clean)))
    result = wsc.compare_policies(warm, clean, T)
    know = result["dimensions"]["knowledge"]
    assert know["comparisons"][COMP_3V1]["winner"] == "warm_start"
    assert know["comparisons"][COMP_3V3X]["winner"] == "warm_start"
    assert know["winner"] == "warm_start"
    inter = result["dimensions"]["interaction"]
    assert inter["comparisons"][COMP_3V1]["winner"] == "warm_start"


def test_warm_start_check_tie_is_inconclusive() -> None:
    tied = _stats(score=0.55, ci_low=0.51, ci_high=0.59, decisive=0.55)
    assert wsc.pick_policy(tied, dict(tied), T) == "inconclusive"
    warm, clean = _paired_nights(
        _outcome_rows(COMP_3V1, "knowledge", 20, 20)
        + _outcome_rows(COMP_3V3X, "knowledge", 20, 20),
        _outcome_rows(COMP_3V1, "knowledge", 20, 20)
        + _outcome_rows(COMP_3V3X, "knowledge", 20, 20),
    )
    result = wsc.compare_policies(warm, clean, T)
    assert result["recommendation"] == "inconclusive"
    assert result["dimensions"]["knowledge"]["winner"] == "inconclusive"


def test_warm_start_check_uses_development_pooled_rule_not_final_lives_rule() -> None:
    """Passes CI, score and decisive share with only one life ahead: development win, final inconclusive."""
    stats = _stats(
        score=0.58,
        ci_low=0.56,
        ci_high=0.62,
        decisive=0.60,
        per_life={
            "founder": True,
            "founder_donor": False,
            "researcher": False,
            "researcher_donor": False,
        },
    )
    assert wsc.development_outcome(stats, T) == "win"
    assert wsc.final_outcome(stats, T) == "inconclusive"
    weaker = _stats(
        score=0.52,
        ci_low=0.48,
        ci_high=0.54,
        decisive=0.52,
        per_life={
            "founder": False,
            "founder_donor": False,
            "researcher": False,
            "researcher_donor": False,
        },
    )
    assert wsc.pick_policy(stats, weaker, T) == "warm_start"


def test_warm_start_cli_prints_json_lines(tmp_path: Path) -> None:
    warm, clean = _paired_nights(
        _outcome_rows(COMP_3V1, "knowledge", 30, 10),
        _outcome_rows(COMP_3V1, "knowledge", 18, 22),
    )
    warm_path = tmp_path / "warm.jsonl"
    clean_path = tmp_path / "clean.jsonl"
    warm_path.write_text("\n".join(json.dumps(r) for r in warm) + "\n", encoding="utf-8")
    clean_path.write_text("\n".join(json.dumps(r) for r in clean) + "\n", encoding="utf-8")
    lines: list[str] = []
    assert wsc.main(["--warm", str(warm_path), "--clean", str(clean_path)], out=lines.append) == 0
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["line"] == "recommendation"
    assert {row["line"] for row in parsed} == {"recommendation", "dimension"}
    assert len([row for row in parsed if row["line"] == "dimension"]) == 2
