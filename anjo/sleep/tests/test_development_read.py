# mypy: disallow-untyped-decorators=False
"""Development go/no-go read (``reports.development``): pooled full mornings, per-dimension labels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.probes.score import BootstrapConfig
from anjo.sleep.reports import development as dev
from anjo.sleep.tests.sleep_fakes import probe as make_probe
from anjo.sleep.types import Probe

CFG = BootstrapConfig(resamples=400, seed=7, ci=0.95)
SEALED = Path("sealed")
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
    persona: str | None = None,
    sealed_dir: Path | None = None,
    checkpoint_basis: str = "promoted",
    label: str = "development",
) -> None:
    folder = root / account / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "night": night,
        "label": label,
        "checkpoint": f"ckpt-{account}-{number}",
        "checkpoint_basis": checkpoint_basis,
        "probes": len({row["probe"] for row in outcomes}),
        "arms": {},
        "pairs": {},
        "outcomes": outcomes,
    }
    if subset:
        record["subset"] = subset
    (folder / f"night-{number:02d}.json").write_text(json.dumps(record), encoding="utf-8")
    if persona is not None and sealed_dir is not None:
        _write_sealed(sealed_dir, persona, outcomes)


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


def _probe_from_row(persona: str, row: dict[str, Any]) -> Probe:
    kind = row["kind"]
    if kind == "new_work":
        source_ids = (str(row.get("source") or f"{row['probe']}-src"),)
    elif kind == "taste_ab":
        source_ids = (str(row.get("source") or row["probe"]),)
    else:
        source_ids = ()
    return make_probe(str(row["probe"]), kind, person=persona, source_ids=source_ids)


def _write_sealed(sealed_dir: Path, persona: str, outcomes: list[dict[str, Any]]) -> None:
    sealed_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Probe] = {}
    for row in outcomes:
        seen.setdefault(str(row["probe"]), _probe_from_row(persona, row))
    lines = []
    for item in seen.values():
        d = {
            "id": item.id,
            "person_id": persona,
            "kind": item.kind,
            "prompt": item.prompt,
            "source_ids": list(item.source_ids),
            "expected": item.expected,
            "options": list(item.options) if item.options else None,
        }
        lines.append(json.dumps(d))
    (sealed_dir / f"{persona}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _fill_accounts(
    root: Path,
    pair: str,
    builder,
    *,
    sealed_dir: Path,
    subset: str | None = None,
) -> None:
    for persona, account in POOLED.items():
        rows = builder(persona, account, pair)
        _write_score(
            root,
            account,
            1,
            "2026-10-31",
            rows,
            subset=subset,
            persona=persona,
            sealed_dir=sealed_dir,
        )


def _comparison_lines(
    root: Path,
    *,
    bootstrap: BootstrapConfig | None = CFG,
    sealed_dir: Path,
) -> list[dict[str, Any]]:
    lines = dev.read(root, POOLED, bootstrap=bootstrap, probes_root=sealed_dir)
    return [line for line in lines if line["line"] == "comparison"]


def test_development_pools_four_lives_outcomes_for_each_headline_pair(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED
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
        _write_score(
            tmp_path, account, 1, "2026-10-31", outcomes, persona=persona, sealed_dir=sealed
        )

    for pair in dev.HEADLINE:
        by_pair = [
            line for line in _comparison_lines(tmp_path, sealed_dir=sealed) if line["pair"] == pair
        ]
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
    loss_ci = (0.28, 0.49)
    assert dev.label_headline(ci=loss_ci, score=0.45, decisive=0.42, rule=rule) == "loss"


def test_development_dimensions_scored_separately(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED
    per_life = 10

    def builder(persona: str, account: str, pair: str) -> list[dict[str, Any]]:
        return [
            *_knowledge_wins(account, pair, per_life, persona),
            *_interaction_losses(account, pair, per_life, persona),
        ]

    _fill_accounts(tmp_path, "3_vs_1", builder, sealed_dir=sealed)
    lines = [
        line for line in _comparison_lines(tmp_path, sealed_dir=sealed) if line["pair"] == "3_vs_1"
    ]
    know = next(line for line in lines if line["dimension"] == "knowledge")
    talk = next(line for line in lines if line["dimension"] == "interaction")
    assert know["n"] == 4 * per_life * 2 and know["label"] == "win"
    assert talk["n"] == 4 * per_life * 2 and talk["label"] == "loss"
    assert know["score"] > 0.9 and talk["score"] < 0.1


def test_development_fairness_4_vs_2C_match_needs_margin_not_win(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED

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

    _fill_accounts(tmp_path, "4_vs_2C", builder, sealed_dir=sealed)
    line = next(
        line
        for line in _comparison_lines(tmp_path, sealed_dir=sealed)
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
    assert dev.label_fairness(ci=(0.0, 0.0), score=0.5, decisive=None, rule=rule) == "inconclusive"


def test_development_one_life_cannot_win_without_full_pool(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED
    rows = [
        _row(f"founder-nw{i}", "3_vs_1", "new_work", "win", source=f"src-{i}") for i in range(50)
    ]
    for persona, account in POOLED.items():
        life_rows = rows if persona == "founder" else []
        placeholder = [
            _row(f"{persona}-pad", "4_vs_2F", "new_work", "tie", source=f"{persona}-pad")
        ]
        _write_sealed(sealed, persona, rows if persona == "founder" else placeholder)
        _write_score(
            tmp_path, account, 1, "2026-10-31", life_rows, persona=persona, sealed_dir=sealed
        )
    line = next(
        line
        for line in _comparison_lines(tmp_path, sealed_dir=sealed)
        if line["pair"] == "3_vs_1" and line["dimension"] == "knowledge"
    )
    assert line["n"] == 50 and line["label"] == "inconclusive"


def test_development_refuses_probe_missing_from_sealed_set(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED
    rows = [_row("only-probe", "3_vs_1", "new_work", "win", source="src-0")]
    for persona, account in POOLED.items():
        other = [_row(f"{persona}-registered", "4_vs_2F", "new_work", "tie", source=f"{persona}-s")]
        _write_sealed(sealed, persona, rows if persona == "founder" else other)
        _write_score(tmp_path, account, 1, "2026-10-31", rows)
    with pytest.raises(dev.DevelopmentError, match="not in the sealed set"):
        dev.read(tmp_path, POOLED, bootstrap=CFG, probes_root=sealed)


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


def test_development_3_vs_3x_pool_ready_ignores_unpromoted_donor(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED
    per_life = 15

    def wins_3x(persona: str, account: str, pair: str) -> list[dict[str, Any]]:
        return _knowledge_wins(account, pair, per_life, persona)

    for persona, account in POOLED.items():
        rows = wins_3x(persona, account, "3_vs_3x")
        basis = "not_promoted" if persona == "founder_donor" else "promoted"
        if persona == "founder_donor":
            rows = [_row(f"{persona}-pad", "4_vs_2C", "new_work", "tie", source=f"{persona}-pad")]
        _write_score(
            tmp_path,
            account,
            1,
            "2026-10-31",
            rows,
            persona=persona,
            sealed_dir=sealed,
            checkpoint_basis=basis,
        )

    line = next(
        line
        for line in _comparison_lines(tmp_path, sealed_dir=sealed)
        if line["pair"] == "3_vs_3x" and line["dimension"] == "knowledge"
    )
    assert line["n"] == 3 * per_life * 2 and line["label"] == "win"


def test_development_reports_each_life_score(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED
    _fill_accounts(
        tmp_path,
        "3_vs_1",
        lambda persona, account, pair: [
            _row(f"{persona}-nw", pair, "new_work", "win", source=f"{persona}-src"),
            _row(f"{persona}-vo", pair, "vocab", "loss"),
        ],
        sealed_dir=sealed,
    )
    lines = dev.read(tmp_path, POOLED, bootstrap=CFG, probes_root=sealed)
    founder_know = next(
        line
        for line in lines
        if line.get("line") == "life"
        and line["persona"] == "founder"
        and line["pair"] == "3_vs_1"
        and line["dimension"] == "knowledge"
    )
    assert founder_know["n"] == 2 and founder_know["wins"] == 1 and founder_know["losses"] == 1
    assert founder_know["score"] == 0.5
    assert any(line["line"] == "comparison" for line in lines)


def test_development_reads_a_scores_only_root_as_the_full_tree(tmp_path: Path) -> None:
    """A rerun's folder: ``<account>/scores`` alone, no ``nights/``, each score file carrying its label."""
    sealed = tmp_path / SEALED
    full, scores_only = tmp_path / "results", tmp_path / "rescore"
    for persona, account in POOLED.items():
        outcomes: list[dict[str, Any]] = []
        for pair in dev.HEADLINE:
            outcomes += [
                _row(f"{persona}-{pair}-nw", pair, "new_work", "win", source=f"{persona}-{pair}-src"),
                _row(f"{persona}-{pair}-vo", pair, "vocab", "tie"),
                _row(f"{persona}-{pair}-th", pair, "tone_hold", "loss"),
            ]
        _write_score(full, account, 1, "2026-10-31", outcomes, persona=persona, sealed_dir=sealed)
        night = full / account / "nights" / "2026-10-31"
        night.mkdir(parents=True)
        (night / "summary.json").write_text(
            json.dumps({"night": "2026-10-31", "label": "development", "status": "promoted"}), encoding="utf-8"
        )
        shutil.copytree(full / account / "scores", scores_only / account / "scores")  # the scores, never the nights
    assert not any((scores_only / account / "nights").exists() for account in POOLED.values())
    assert dev.read(scores_only, POOLED, bootstrap=CFG, probes_root=sealed) == dev.read(
        full, POOLED, bootstrap=CFG, probes_root=sealed
    )


def test_reported_only_harness_pair_is_carried_when_present_and_absent(tmp_path: Path) -> None:
    sealed = tmp_path / SEALED
    for persona, account in POOLED.items():
        outcomes = [
            _row(f"{persona}-2h-nw", "2H_vs_2C", "new_work", "win", source=f"{persona}-2h-src"),
            _row(f"{persona}-2h-vo", "2H_vs_2C", "vocab", "tie"),
            _row(f"{persona}-2h-th", "2H_vs_2C", "tone_hold", "loss"),
            _row(f"{persona}-2h-ta", "2H_vs_2C", "taste_ab", "win", source=f"{persona}-2h-taste"),
        ]
        _write_score(tmp_path, account, 1, "2026-10-31", outcomes, persona=persona, sealed_dir=sealed)
    lines = [line for line in dev.read(tmp_path, POOLED, bootstrap=CFG, probes_root=sealed)
             if line["line"] == "comparison" and line["pair"] == "2H_vs_2C"]
    assert {line["dimension"] for line in lines} == {"knowledge", "interaction"}
    assert all(line["label"] == "reported" for line in lines)
    assert all(line["pair"] not in dev.HEADLINE + dev.FAIRNESS for line in lines)

    absent = tmp_path / "absent"
    absent_sealed = tmp_path / "absent-sealed"
    _fill_accounts(
        absent,
        "3_vs_1",
        lambda persona, account, pair: [
            _row(f"{persona}-3v1-nw", pair, "new_work", "win", source=f"{persona}-3v1-src"),
        ],
        sealed_dir=absent_sealed,
    )
    assert not [line for line in dev.read(absent, POOLED, bootstrap=CFG, probes_root=absent_sealed)
                if line.get("pair") == "2H_vs_2C"]
