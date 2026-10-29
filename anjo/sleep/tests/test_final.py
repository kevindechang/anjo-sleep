# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False
"""Task 12b final report (offline): morning score files and registered rules only."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from anjo.sleep.lives import draw as lives_draw
from anjo.sleep.reports import final
from anjo.sleep.tests.sleep_fakes import probe, taste
from anjo.sleep.types import Probe

FREEZE_SHA = "a" * 40
TAG = "sleep-freeze"
PAIR = ("life-a", "life-b")
SINGLE = "life-c"
LIVES = (*PAIR, SINGLE)
DONORS = lives_draw.donor_map(
    (
        lives_draw.Life("life-a", "p1", "film", "line", "t", ()),
        lives_draw.Life("life-b", "p1", "film", "line", "t", ()),
    ),
    lives_draw.Life("life-c", "p2", "music", "line", "t", ()),
)


def _sealed_catalog() -> dict[str, list[Probe]]:
    """Registered probes for test lives: ``new_work`` blocks follow held-out source ids, not probe ids."""
    found: dict[str, list[Probe]] = {}
    for life in LIVES:
        rows: list[Probe] = []
        for i in range(40):
            for kind in ("new_work", "vocab"):
                pid = f"{life}-{kind}-{i}"
                src = (f"{life}-nw-src-{i // 2}",) if kind == "new_work" else ()
                rows.append(probe(pid, kind, person=life, source_ids=src))
        for i in range(40):
            rows.append(probe(f"{life}-th-{i}", "tone_hold", person=life))
            rows.append(taste(f"{life}-ta-{i}", person=life, source=f"shared-taste-{i}"))
        for i in range(1, 30):
            rows.append(
                probe(
                    f"{life}-new_work-{i:02d}",
                    "new_work",
                    person=life,
                    source_ids=(f"{life}-item-{(i - 1) // 5}",),
                )
            )
        rows.append(
            probe(f"{life}-mark", "new_work", person=life, source_ids=(f"{life}-mark-src",))
        )
        found[life] = rows
    return found


_SEALED = _sealed_catalog()


def _sealed(user_id: str) -> Sequence[Probe]:
    return _SEALED[user_id]


def _write_morning(
    root: Path,
    life_id: str,
    *,
    night: int,
    outcomes: list[dict[str, Any]],
    unavailable: dict[str, str] | None = None,
    basis: str = "promoted",
) -> None:
    folder = root / life_id / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "night": f"2026-11-{night:02d}",
        "label": "final",
        "checkpoint": f"{life_id}-ckpt",
        "checkpoint_basis": basis,
        "probes": len(outcomes),
        "arms": {},
        "pairs": {},
        "unavailable": unavailable or {},
        "outcomes": outcomes,
    }
    (folder / f"night-{night:02d}.json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )


def _wins(
    life: str, pair: str, *, n: int = 12, kinds: tuple[str, ...] = ("new_work", "vocab")
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        for kind in kinds:
            rows.append(
                {
                    "probe": f"{life}-{kind}-{i}",
                    "pair": pair,
                    "kind": kind,
                    "outcome": "win",
                    "flip": False,
                },
            )
    return rows


def _interaction_wins(life: str, pair: str, *, n: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "probe": f"{life}-th-{i}",
                "pair": pair,
                "kind": "tone_hold",
                "outcome": "win",
                "flip": False,
            }
        )
        rows.append(
            {
                "probe": f"{life}-ta-{i}",
                "pair": pair,
                "kind": "taste_ab",
                "outcome": "win",
                "flip": False,
            }
        )
    return rows


def _rig_all_wins(root: Path, *, pair_3x_weak: str | None = None) -> None:
    for life in LIVES:
        rows = _wins(life, final.COMP_3V1) + _wins(life, final.COMP_3V3X)
        rows += _interaction_wins(life, final.COMP_3V1) + _interaction_wins(life, final.COMP_3V3X)
        rows += _wins(life, final.COMP_4V2C, kinds=("new_work",)) + _wins(
            life, final.COMP_4V2F, kinds=("new_work",)
        )
        if pair_3x_weak == life:
            rows = [r if r["pair"] != final.COMP_3V3X else {**r, "outcome": "loss"} for r in rows]
        _write_morning(root, life, night=2, outcomes=rows)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "results"


def _run(root: Path, **kwargs: Any) -> dict[str, Any]:
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA):
        return final.run(
            TAG,
            root,
            LIVES,
            donors=DONORS,
            pair_lives=PAIR,
            single_life=SINGLE,
            git=FREEZE_SHA,
            sealed=_sealed,
            **kwargs,
        )


def test_run_records_freeze_sha(root: Path) -> None:
    _rig_all_wins(root)
    report = _run(root)
    assert report["freeze_sha"] == report["git_sha"] == FREEZE_SHA
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA):
        with pytest.raises(final.FinalError, match="not the freeze tag"):
            final.run(
                TAG, root, LIVES, donors=DONORS, pair_lives=PAIR, single_life=SINGLE, git="b" * 40
            )


def test_final_scores_serving_checkpoint_not_best(root: Path) -> None:
    life = PAIR[0]
    _write_morning(root, life, night=1, outcomes=_wins(life, final.COMP_3V1, n=40))
    serving = (
        _wins(life, final.COMP_3V1, n=3)
        + _wins(life, final.COMP_3V3X, n=3)
        + _interaction_wins(life, final.COMP_3V1, n=2)
        + _interaction_wins(life, final.COMP_3V3X, n=2)
        + [
            {
                "probe": f"{life}-mark",
                "pair": final.COMP_3V1,
                "kind": "new_work",
                "outcome": "loss",
                "flip": False,
            }
        ]
    )
    _write_morning(root, life, night=2, outcomes=serving, basis="not_promoted")
    for other in LIVES[1:]:
        rows = _wins(other, final.COMP_3V1) + _wins(other, final.COMP_3V3X)
        rows += _interaction_wins(other, final.COMP_3V1) + _interaction_wins(other, final.COMP_3V3X)
        rows += _wins(other, final.COMP_4V2C, kinds=("new_work",)) + _wins(
            other, final.COMP_4V2F, kinds=("new_work",)
        )
        _write_morning(root, other, night=2, outcomes=rows)
    path = final.serving_morning_path(root, life)
    assert path.name == "night-02.json"
    report = _run(root)
    per = report["dimensions"]["knowledge"][final.COMP_3V1]["stats"]["per_life"][life]
    assert per["n"] < 20  # night-01 alone carries 80 knowledge rows for 3 vs 1
    assert per["score"] < 1.0  # not the all-win best night
    assert json.loads(path.read_text(encoding="utf-8"))["checkpoint_basis"] == "not_promoted"


def test_final_refuses_rerun_once_a_score_exists(root: Path) -> None:
    _rig_all_wins(root)
    _run(root)
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA):
        with pytest.raises(final.FinalError, match="already has final.json"):
            final.run(
                TAG, root, LIVES, donors=DONORS, pair_lives=PAIR, single_life=SINGLE, git=FREEZE_SHA
            )


def test_claim_row_follows_table_order() -> None:
    assert final._claim_row("win", "inconclusive", False) == "base_only"
    assert final._claim_row("win", "loss", True) == "base_only"
    assert final._claim_row("inconclusive", "inconclusive", False) == "none"
    assert final._claim_row("win", "win", True) == "headline"


def test_inconclusive_never_reported_as_match() -> None:
    stats = {
        "n": 10,
        "wins": 5,
        "ties": 0,
        "losses": 5,
        "score": 0.5,
        "decisive_share": 0.5,
        "ci": [0.40, 0.60],
        "per_life": {},
    }
    assert final._fairness_outcome(stats) == "inconclusive"
    assert final._fairness_outcome({**stats, "ci": [0.44, 0.55]}) == "inconclusive"


def test_dimensions_scored_and_claimed_separately(root: Path) -> None:
    for life in LIVES:
        know = _wins(life, final.COMP_3V1) + _wins(life, final.COMP_3V3X)
        weak = _interaction_wins(life, final.COMP_3V1, n=1) + _interaction_wins(
            life, final.COMP_3V3X, n=1
        )
        for row in weak:
            row["outcome"] = "loss"
        _write_morning(root, life, night=2, outcomes=know + weak)
    report = _run(root)
    assert report["dimensions"]["knowledge"]["claim_row"] == "headline"
    assert report["dimensions"]["interaction"]["claim_row"] == "none"
    assert (
        report["dimensions"]["knowledge"]["claim_row"]
        != report["dimensions"]["interaction"]["claim_row"]
    )


def test_donor_never_changes_after_draw(root: Path) -> None:
    _rig_all_wins(root)
    report = _run(root)
    assert report["donors"] == DONORS
    alt = {**DONORS, SINGLE: "someone-else"}
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA):
        with pytest.raises(final.FinalError, match="already has final.json"):
            final.run(
                TAG,
                root,
                LIVES,
                donors=alt,
                pair_lives=PAIR,
                single_life=SINGLE,
                git=FREEZE_SHA,
                rerun_reason="infra",
            )
    stored = json.loads(final.score_path(root, PAIR[0]).read_text(encoding="utf-8"))
    assert stored["donors"] == DONORS


def test_rerun_reason_does_not_overwrite_existing_score(root: Path) -> None:
    _rig_all_wins(root)
    _run(root)
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA):
        with pytest.raises(final.FinalError, match="already has final.json"):
            final.run(
                TAG,
                root,
                LIVES,
                donors=DONORS,
                pair_lives=PAIR,
                single_life=SINGLE,
                git=FREEZE_SHA,
                rerun_reason="infra",
            )


def test_loss_uses_ci_upper_bound() -> None:
    stats = {
        "n": 100,
        "score": 0.4,
        "decisive_share": 0.4,
        "ci": [0.1, 0.8],
        "per_life": {
            "a": {"ahead": False, "behind": True},
            "b": {"ahead": False, "behind": True},
            "c": {"ahead": False, "behind": True},
        },
    }
    assert final._headline_outcome(stats) == "inconclusive"


def test_loss_needs_two_lives_strictly_behind() -> None:
    stats = {
        "n": 100,
        "score": 0.4,
        "decisive_share": 0.4,
        "ci": [0.1, 0.3],
        "per_life": {
            "a": {"ahead": False, "behind": True},
            "b": {"ahead": False, "behind": False},
            "c": {"ahead": False, "behind": False},
        },
    }
    assert final._headline_outcome(stats) == "inconclusive"


def test_fairness_new_work_does_not_merge_life_into_one_block() -> None:
    records = {
        life: {
            "outcomes": [
                {
                    "probe": f"{life}-new_work-{i:02d}",
                    "kind": "new_work",
                    "pair": final.COMP_4V2C,
                    "outcome": "win" if i <= 15 else "loss",
                }
                for i in range(1, 26)
            ]
        }
        for life in ("life-a", "life-b", "life-c")
    }
    registries = {
        life: final.registered_probes(life, records[life], _sealed, results_root=Path("/unused"))
        for life in records
    }
    stats = final._fairness_new_work(records, registries, final.COMP_4V2C)
    assert stats is not None
    assert final._fairness_outcome(stats) != "win"


def test_pooled_scoring_refuses_probe_ids_missing_from_registered_set() -> None:
    records = {
        "life-a": {
            "outcomes": [
                {
                    "probe": "not-in-set",
                    "kind": "new_work",
                    "pair": final.COMP_3V1,
                    "outcome": "win",
                },
            ]
        }
    }
    registries = {
        "life-a": final.registered_probes(
            "life-a", records["life-a"], _sealed, results_root=Path("/unused")
        )
    }
    with pytest.raises(final.FinalError, match="not in the registered set"):
        final._pool_comparison(records, registries, final.COMP_3V1, "knowledge")


def test_unpromoted_donor_marks_specificity_inconclusive(root: Path) -> None:
    _rig_all_wins(root)
    _write_morning(
        root,
        SINGLE,
        night=2,
        outcomes=_wins(SINGLE, final.COMP_3V1) + _wins(SINGLE, final.COMP_3V3X),
        unavailable={"3x": final.DONOR_UNAVAILABLE},
    )
    report = _run(root)
    know = report["dimensions"]["knowledge"]
    assert know[final.COMP_3V1]["outcome"] == "win"
    assert know[final.COMP_3V3X]["outcome"] == "inconclusive"
    assert know["claim_row"] == "base_only"


def test_matches_needs_registered_margin() -> None:
    base = {
        "n": 20,
        "wins": 10,
        "ties": 0,
        "losses": 10,
        "score": 0.5,
        "decisive_share": 0.5,
        "per_life": {"l1": {"ahead": False}, "l2": {"ahead": False}, "l3": {"ahead": False}},
    }
    assert final._fairness_outcome({**base, "ci": [0.46, 0.80]}) == "matches"
    assert final._fairness_outcome({**base, "ci": [0.45, 0.80]}) == "inconclusive"


def test_learned_her_row_needs_both_pair_lives_ahead(root: Path) -> None:
    _rig_all_wins(root, pair_3x_weak=PAIR[1])
    report = _run(root)
    know = report["dimensions"]["knowledge"]
    assert know[final.COMP_3V1]["outcome"] == "win"
    assert know[final.COMP_3V3X]["outcome"] == "win"
    assert know["pair_lives_ahead_on_3x"] is False
    assert know["claim_row"] != "headline"
    assert know["claim_row"] == "none"
