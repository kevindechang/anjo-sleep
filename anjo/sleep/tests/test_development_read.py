# mypy: disallow-untyped-decorators=False
"""Development go/no-go read (``reports.development``): pooled full mornings, per-dimension labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.probes.score import BootstrapConfig
from anjo.sleep.reports import development as dev

CFG = BootstrapConfig(resamples=400, seed=7, ci=0.95)
POOLED = {
    "founder": "acct-founder",
    "founder_donor": "acct-founder-donor",
    "researcher": "acct-researcher",
    "researcher_donor": "acct-researcher-donor",
}


def _write_score(
    root: Path,
    account: str,
    number: int,
    night: str,
    outcomes: list[dict[str, Any]],
    *,
    subset: str | None = None,
) -> None:
    folder = root / account / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "night": night,
        "checkpoint": f"ckpt-{account}-{number}",
        "checkpoint_basis": "promoted",
        "probes": len({row["probe"] for row in outcomes}),
        "arms": {},
        "pairs": {},
        "outcomes": outcomes,
    }
    if subset:
        record["subset"] = subset
    (folder / f"night-{number:02d}.json").write_text(json.dumps(record), encoding="utf-8")


def _row(
    probe: str, pair: str, kind: str, outcome: str, *, source: str | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "probe": probe,
        "pair": pair,
        "kind": kind,
        "outcome": outcome,
        "flip": False,
    }
    if source is not None:
        row["source"] = source
    return row


def _knowledge_wins(account: str, pair: str, count: int, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(count):
        rows.append(_row(f"{prefix}-nw{i}", pair, "new_work", "win", source=f"{prefix}-src-{i}"))
        rows.append(_row(f"{prefix}-vo{i}", pair, "vocab", "win"))
    return rows


def _interaction_losses(account: str, pair: str, count: int, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(count):
        rows.append(_row(f"{prefix}-th{i}", pair, "tone_hold", "loss"))
        rows.append(_row(f"{prefix}-ta{i}", pair, "taste_ab", "loss", source=f"{prefix}-taste-{i}"))
    return rows


def _fill_accounts(root: Path, pair: str, builder) -> None:
    for persona, account in POOLED.items():
        rows = builder(persona, account, pair)
        _write_score(root, account, 1, "2026-10-31", rows)


def _comparison_lines(
    root: Path, *, bootstrap: BootstrapConfig | None = CFG
) -> list[dict[str, Any]]:
    lines = dev.read(root, POOLED, bootstrap=bootstrap)
    return [line for line in lines if line["line"] == "comparison"]


def test_development_pools_four_lives_outcomes_for_each_headline_pair(tmp_path: Path) -> None:
    for persona, account in POOLED.items():
        outcomes = []
        for pair in dev.HEADLINE:
            outcomes.extend(
                [
                    _row(
                        f"{persona}-{pair}-k1",
                        pair,
                        "new_work",
                        "win",
                        source=f"{persona}-{pair}-s1",
                    ),
                    _row(f"{persona}-{pair}-k2", pair, "vocab", "tie"),
                ]
            )
        _write_score(tmp_path, account, 1, "2026-10-31", outcomes)

    for pair in dev.HEADLINE:
        by_pair = [line for line in _comparison_lines(tmp_path) if line["pair"] == pair]
        knowledge = next(line for line in by_pair if line["dimension"] == "knowledge")
        assert knowledge["n"] == 8
        assert knowledge["wins"] == 4 and knowledge["ties"] == 4


def test_development_win_needs_ci_over_half_and_score_and_decisive_share() -> None:
    rule = dev.WinRule(0.5, 0.55, 0.58)
    strong = (0.61, 0.72)
    assert dev.label_headline(ci=strong, score=0.6, decisive=0.65, rule=rule) == "win"
    assert dev.label_headline(ci=strong, score=0.54, decisive=0.65, rule=rule) == "inconclusive"
    assert dev.label_headline(ci=strong, score=0.6, decisive=0.57, rule=rule) == "inconclusive"
    assert dev.label_headline(ci=(0.49, 0.7), score=0.6, decisive=0.65, rule=rule) == "inconclusive"


def test_development_dimensions_scored_separately(tmp_path: Path) -> None:
    per_life = 10

    def builder(persona: str, account: str, pair: str) -> list[dict[str, Any]]:
        return [
            *_knowledge_wins(account, pair, per_life, persona),
            *_interaction_losses(account, pair, per_life, persona),
        ]

    _fill_accounts(tmp_path, "3_vs_1", builder)
    lines = [line for line in _comparison_lines(tmp_path) if line["pair"] == "3_vs_1"]
    know = next(line for line in lines if line["dimension"] == "knowledge")
    talk = next(line for line in lines if line["dimension"] == "interaction")
    assert know["n"] == 4 * per_life * 2 and know["label"] == "win"
    assert talk["n"] == 4 * per_life * 2 and talk["label"] == "loss"
    assert know["score"] > 0.9 and talk["score"] < 0.1


def test_development_fairness_4_vs_2C_match_needs_margin_not_win(tmp_path: Path) -> None:
    def builder(persona: str, account: str, pair: str) -> list[dict[str, Any]]:
        rows = []
        for i in range(12):
            rows.append(
                _row(f"{persona}-nw-w{i}", pair, "new_work", "win", source=f"{persona}-src-{i}")
            )
        for i in range(8):
            rows.append(
                _row(f"{persona}-nw-l{i}", pair, "new_work", "loss", source=f"{persona}-src-l-{i}")
            )
        return rows

    _fill_accounts(tmp_path, "4_vs_2C", builder)
    line = next(
        line
        for line in _comparison_lines(tmp_path)
        if line["pair"] == "4_vs_2C" and line["dimension"] == "knowledge"
    )
    assert line["score"] == 0.6 and line["decisive_share"] == 0.6
    assert line["ci"][0] > dev.fairness_margin()
    assert line["label"] == "match"


def test_inconclusive_never_labeled_match() -> None:
    rule = dev.WinRule(0.5, 0.55, 0.58)
    low = (0.44, 0.56)
    assert dev.label_fairness(ci=low, score=0.5, decisive=0.5, rule=rule) == "inconclusive"
    assert dev.label_fairness(ci=(0.45, 0.55), score=0.5, decisive=0.5, rule=rule) == "inconclusive"
    assert dev.label_fairness(ci=low, score=0.5, decisive=0.5, rule=rule) != "match"


def test_development_refuses_daily_subset_scores(tmp_path: Path) -> None:
    for account in POOLED.values():
        _write_score(
            tmp_path,
            account,
            1,
            "2026-10-31",
            [_row("p1", "3_vs_1", "new_work", "win")],
            subset="daily-v1",
        )
    with pytest.raises(dev.DevelopmentError, match="no full morning"):
        dev.read(tmp_path, POOLED, bootstrap=CFG)
