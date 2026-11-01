# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False
"""Task 12b final report (offline): morning score files and registered rules only."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.lives import draw as lives_draw
from anjo.sleep.prereg import freeze_manifest, ledger
from anjo.sleep.probes.generate import _line as sealed_line
from anjo.sleep.probes.morning import MORNING_PROBES
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


KNOWLEDGE_ROWS = 20  # new_work + vocab each → 40 knowledge outcomes per comparison per life
INTERACTION_ROWS = 15  # tone_hold + taste_ab each → 30 interaction outcomes
NEW_WORK_ROWS = 25


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
    judge_model: str = fireworks_api.DEEPSEEK_V4P1_FLASH,
) -> None:
    folder = root / life_id / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "night": f"2026-11-{night:02d}",
        "label": "final",
        "judge_model": judge_model,
        "synthetic": True,
        "accelerated": True,
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
    life: str, pair: str, *, n: int = KNOWLEDGE_ROWS, kinds: tuple[str, ...] = ("new_work", "vocab")
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


def _interaction_wins(life: str, pair: str, *, n: int = INTERACTION_ROWS) -> list[dict[str, Any]]:
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
        rows += _wins(life, final.COMP_4V2C, kinds=("new_work",), n=NEW_WORK_ROWS) + _wins(
            life, final.COMP_4V2F, kinds=("new_work",), n=NEW_WORK_ROWS
        )
        if pair_3x_weak == life:
            rows = [r if r["pair"] != final.COMP_3V3X else {**r, "outcome": "loss"} for r in rows]
        _write_morning(root, life, night=2, outcomes=rows)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "results"


def _sealed_files(base: Path) -> dict[str, Path]:
    """Each test life's sealed probe file, in ``seal_probes``' line format (the catalog repeats ids)."""
    found: dict[str, Path] = {}
    (base / "sealed").mkdir(parents=True, exist_ok=True)
    for life in LIVES:
        found[life] = base / "sealed" / f"{life}.jsonl"
        if not found[life].exists():
            found[life].write_text("".join(sealed_line(p) + "\n" for p in _SEALED[life]), encoding="utf-8")
    return found


def _manifest(base: Path, sealed: dict[str, Path]) -> Path:
    """A freeze manifest of this checkout's frozen files and the test's sealed files at the freeze sha,
    in tmp, registered in the tmp ledger (``_ledger``)."""
    out = base / "freeze_manifest.json"
    if not out.exists():
        freeze_manifest.write(freeze_manifest.build(list(sealed.values()), commit=FREEZE_SHA), out,
                              ledger_file=_ledger(base))
    return out


def _ledger(base: Path) -> Path:
    return base / "sleep" / "ledger.jsonl"  # where the prereg ledger is under ANJO_DATA_DIR=base


def _run(root: Path, **kwargs: Any) -> dict[str, Any]:
    sealed = _sealed_files(root.parent)
    kwargs = {"sealed_files": sealed, "manifest": _manifest(root.parent, sealed), "donors": DONORS,
              "ledger_file": _ledger(root.parent), **kwargs}
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA):
        return final.run(
            TAG,
            root,
            LIVES,
            pair_lives=PAIR,
            single_life=SINGLE,
            git=FREEZE_SHA,
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
        outcomes=_wins(SINGLE, final.COMP_3V1)
        + _wins(SINGLE, final.COMP_3V3X)
        + _interaction_wins(SINGLE, final.COMP_3V1)
        + _interaction_wins(SINGLE, final.COMP_3V3X),
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
        "per_life": {
            "l1": {"n": NEW_WORK_ROWS, "ahead": False},
            "l2": {"n": NEW_WORK_ROWS, "ahead": False},
            "l3": {"n": NEW_WORK_ROWS, "ahead": False},
        },
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


def test_headline_win_needs_registered_probe_counts_per_life() -> None:
    stats = {
        "n": 3,
        "wins": 3,
        "ties": 0,
        "losses": 0,
        "score": 1.0,
        "decisive_share": 1.0,
        "ci": [0.99, 1.0],
        "per_life": {
            "life-a": {"n": 1, "ahead": True, "behind": False, "score": 1.0},
            "life-b": {"n": 1, "ahead": True, "behind": False, "score": 1.0},
            "life-c": {"n": 1, "ahead": True, "behind": False, "score": 1.0},
        },
    }
    assert final._headline_outcome(stats, min_per_life=40) == "inconclusive"


def test_fairness_matches_needs_registered_new_work_per_life() -> None:
    stats = {
        "n": 1,
        "wins": 1,
        "ties": 0,
        "losses": 0,
        "score": 1.0,
        "decisive_share": 1.0,
        "ci": [0.46, 0.99],
        "per_life": {"life-a": {"n": 1, "ahead": True, "behind": False, "score": 1.0}},
    }
    assert final._fairness_outcome(stats) == "inconclusive"


def test_loss_at_complement_score_is_inclusive() -> None:
    stats = {
        "n": 75,
        "wins": 25,
        "ties": 25,
        "losses": 25,
        "score": 0.45,
        "decisive_share": 0.42,
        "ci": [0.05, 0.40],
        "per_life": {
            "a": {"n": 25, "ahead": False, "behind": True, "score": 0.45},
            "b": {"n": 25, "ahead": False, "behind": True, "score": 0.45},
            "c": {"n": 25, "ahead": False, "behind": True, "score": 0.45},
        },
    }
    assert final._headline_outcome(stats, reverse=True, min_per_life=25) == "win"


def test_final_morning_without_subset_uses_full_set_not_night_summary(tmp_path: Path) -> None:
    root = tmp_path / "results"
    life = "life-a"
    night = "2026-10-24"
    (root / life / "nights" / night).mkdir(parents=True)
    (root / life / "nights" / night / "summary.json").write_text(
        json.dumps({"night": night, MORNING_PROBES: "daily-v1"}),
        encoding="utf-8",
    )
    record = {
        "night": night,
        "label": "final",
        "outcomes": [
            {
                "probe": f"{life}-new_work-01",
                "kind": "new_work",
                "pair": final.COMP_3V1,
                "outcome": "win",
            }
        ],
    }
    reg = final.registered_probes(life, record, _sealed, results_root=root)
    assert f"{life}-new_work-01" in reg
    assert f"{life}-new_work-02" in reg


# ── the registered judge, the freeze manifest and provenance (audit fixes) ─────


def test_final_refuses_a_score_file_from_an_unregistered_judge(root: Path) -> None:
    _rig_all_wins(root)
    rows = _wins(SINGLE, final.COMP_3V1)
    _write_morning(root, SINGLE, night=2, outcomes=rows, judge_model="muse-spark-1.3")
    with pytest.raises(final.FinalError, match=r"night-02\.json was judged by 'muse-spark-1\.3', not the registered judge"):
        _run(root)
    assert not final.score_path(root, PAIR[0]).exists()


def test_final_refuses_a_score_file_with_no_judge_model(root: Path) -> None:
    _rig_all_wins(root)
    path = root / SINGLE / "scores" / "night-02.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("judge_model")
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(final.FinalError, match="judged by None"):
        _run(root)


def test_registered_judges_are_read_from_thresholds() -> None:
    assert final.registered_judge() == fireworks_api.DEEPSEEK_V4P1_FLASH
    assert final.registered_second_judge() == "gemini-3.8-flash"
    with pytest.raises(final.FinalError, match="names no judge"):
        final.registered_model("Muse Spark with judge_v1.md", final.JUDGE_MODELS, "judging.judge")


def test_final_checks_the_second_judges_model(root: Path) -> None:
    _rig_all_wins(root)
    with pytest.raises(final.FinalError, match="not the registered second judge 'gemini-3.8-flash'"):
        _run(root, second_judge={"model": "muse-spark-1.3", "knowledge": {"agreement": 0.9}})
    with pytest.raises(final.FinalError, match="from None"):
        _run(root, second_judge={"knowledge": {"agreement": 0.9}})
    report = _run(root, second_judge={"model": "gemini-3.8-flash", "knowledge": {"agreement": 0.9}})
    assert report["dimensions"]["knowledge"]["second_judge"] == {"agreement": 0.9}


def test_final_refuses_without_a_freeze_manifest(root: Path) -> None:
    _rig_all_wins(root)
    with pytest.raises(final.FinalError, match="no freeze manifest"):
        _run(root, manifest=root.parent / "nowhere.json")


def test_final_refuses_a_sealed_file_that_differs_from_the_manifest(root: Path) -> None:
    _rig_all_wins(root)
    sealed = _sealed_files(root.parent)
    manifest = _manifest(root.parent, sealed)
    sealed[SINGLE].write_text(sealed[SINGLE].read_text(encoding="utf-8").replace("prompt", "prompt "), encoding="utf-8")
    with pytest.raises(final.FinalError, match=f"sealed probe file {SINGLE}.jsonl differs from the manifest"):
        _run(root, sealed_files=sealed, manifest=manifest)


def test_final_refuses_a_frozen_file_that_differs_from_the_manifest(root: Path) -> None:
    _rig_all_wins(root)
    sealed = _sealed_files(root.parent)
    built = freeze_manifest.build(list(sealed.values()), commit=FREEZE_SHA)
    for rel in freeze_manifest.FROZEN_FILES:
        tampered = {**built, "files": {**built["files"], rel: "0" * 64}}
        base = root.parent / rel.replace("/", "-")
        out = base / "manifest.json"
        freeze_manifest.write(tampered, out, ledger_file=_ledger(base))
        with pytest.raises(final.FinalError, match=f"{rel} differs from the manifest"):
            _run(root, sealed_files=sealed, manifest=out, ledger_file=_ledger(base))


def test_final_refuses_thresholds_that_drift_from_the_registered_copy(root: Path, tmp_path: Path) -> None:
    _rig_all_wins(root)
    drifted = json.loads(final.THRESHOLDS.read_text(encoding="utf-8"))
    drifted["win"]["overall_score_at_least"] = 0.5
    edited = tmp_path / "thresholds.json"
    edited.write_text(json.dumps(drifted), encoding="utf-8")
    with patch.object(final, "THRESHOLDS", edited):
        with pytest.raises(final.FinalError, match="differs from thresholds_registered.json beyond the allowed keys: win"):
            _run(root)
    allowed = json.loads(final.THRESHOLDS.read_text(encoding="utf-8"))
    allowed["hand_written"] = {"note": "allowed to differ"}
    edited.write_text(json.dumps(allowed), encoding="utf-8")
    assert freeze_manifest.thresholds_drift(edited) is None


def test_manifest_is_written_once_and_names_every_file(tmp_path: Path) -> None:
    sealed = _sealed_files(tmp_path)
    out = _manifest(tmp_path, sealed)
    record = json.loads(out.read_text(encoding="utf-8"))
    assert set(record["files"]) == set(freeze_manifest.FROZEN_FILES)
    assert set(record["sealed"]) == {f"{life}.jsonl" for life in LIVES}
    assert record["git_commit"] == FREEZE_SHA
    with pytest.raises(freeze_manifest.ManifestError, match="already exists"):
        freeze_manifest.write(record, out, ledger_file=_ledger(tmp_path))
    with pytest.raises(freeze_manifest.ManifestError, match="already registers a freeze manifest"):
        freeze_manifest.write(record, tmp_path / "second.json", ledger_file=_ledger(tmp_path))
    entries = ledger.read_entries(_ledger(tmp_path))
    assert [(e.kind, e.sha256) for e in entries] == [("freeze_manifest", ledger.sha256_file(out))]
    with pytest.raises(freeze_manifest.ManifestError, match="pass --sealed"):
        freeze_manifest.build([], commit="c" * 40)


def test_final_json_carries_provenance(root: Path) -> None:
    _rig_all_wins(root)
    report = _run(root)
    assert report["synthetic"] is True and report["accelerated"] is True
    assert report["provenance"][SINGLE] == {"synthetic": True, "accelerated": True}
    assert report["judge_model"] == fireworks_api.DEEPSEEK_V4P1_FLASH
    assert report["manifest_commit"] == FREEZE_SHA
    assert report["attempt"] == 1 and report["rerun_reason"] is None
    stored = json.loads(final.score_path(root, SINGLE).read_text(encoding="utf-8"))
    assert stored["synthetic"] is True and stored["accelerated"] is True


# ── the command line ──────────────────────────────────────────────────────────

ACCOUNTS = {"life-a": "acct-1", "life-b": "acct-2", "life-c": "acct-3"}


def _cli_tree(root: Path) -> tuple[Path, Path]:
    """The lives' morning files under their account ids, the sealed dir and a manifest in tmp."""
    _rig_all_wins(root)
    for life, account in ACCOUNTS.items():
        (root / life).rename(root / account)
    sealed = _sealed_files(root.parent)
    return root.parent / "sealed", _manifest(root.parent, sealed)


def _cli(root: Path, sealed_dir: Path, manifest: Path, *extra: str, draw: list[str] | None = None) -> tuple[int, list[str]]:
    lines: list[str] = []
    argv = [*(f"--account={life}={account}" for life, account in ACCOUNTS.items()),
            *(draw if draw is not None else ["--pair", *PAIR, "--single", SINGLE,
                                             *(f"--donor={k}={v}" for k, v in DONORS.items())]),
            "--root", str(root), "--sealed-dir", str(sealed_dir), "--manifest", str(manifest),
            "--require-sha", FREEZE_SHA, *extra]
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA), \
            patch.object(final, "git_sha", return_value=FREEZE_SHA), \
            patch.dict(os.environ, {ledger.DATA_DIR_ENV: str(root.parent)}):
        code = final.main(argv, out=lines.append)
    return code, lines


def test_cli_scores_the_accounts_and_prints_the_claim_rows(root: Path) -> None:
    sealed_dir, manifest = _cli_tree(root)
    code, lines = _cli(root, sealed_dir, manifest)
    assert code == 0, lines
    assert lines[0].startswith("final: life-a=acct-1, life-b=acct-2, life-c=acct-3 at aaaaaaaaaaaa")
    assert "(synthetic people), accelerated nights" in lines[0]
    assert lines[1].startswith("knowledge: headline: ") and lines[2].startswith("interaction: headline: ")
    stored = json.loads(final.score_path(root, "acct-3").read_text(encoding="utf-8"))
    assert stored["single_life"] == "acct-3" and stored["pair_lives"] == ["acct-1", "acct-2"]
    assert stored["donors"] == {"acct-1": "acct-2", "acct-2": "acct-1", "acct-3": "acct-1"}
    code, lines = _cli(root, sealed_dir, manifest)  # a second run without a reason is refused
    assert code == 1 and "already has final.json" in lines[0]
    first = final.score_path(root, "acct-1").read_bytes()
    code, lines = _cli(root, sealed_dir, manifest, "--rerun-reason", "infra")  # absolute once scored
    assert code == 1 and "refusal is absolute" in lines[0]
    assert final.score_path(root, "acct-1").read_bytes() == first
    assert sorted(p.name for p in (root / "acct-1" / "scores").iterdir()) == ["final.json", "night-02.json"]


def test_cli_reads_a_draw_file(root: Path, tmp_path: Path) -> None:
    sealed_dir, manifest = _cli_tree(root)
    draw = tmp_path / "draw.json"
    draw.write_text(json.dumps({"pair": list(PAIR), "single": SINGLE, "donors": DONORS}), encoding="utf-8")
    code, lines = _cli(root, sealed_dir, manifest, draw=["--draw", str(draw)])
    assert code == 0, lines


def test_cli_refuses_before_scoring(root: Path, tmp_path: Path) -> None:
    sealed_dir, manifest = _cli_tree(root)
    wrong_donors = ["--pair", *PAIR, "--single", SINGLE, *(f"--donor={k}={v}" for k, v in {**DONORS, SINGLE: PAIR[1]}.items())]
    assert "registered donor rule" in _cli(root, sealed_dir, manifest, draw=wrong_donors)[1][0]
    lines: list[str] = []
    argv = [*(f"--account={k}={v}" for k, v in ACCOUNTS.items()), "--pair", *PAIR, "--single", SINGLE,
            *(f"--donor={k}={v}" for k, v in DONORS.items()), "--root", str(root), "--sealed-dir", str(sealed_dir),
            "--manifest", str(manifest), "--require-sha", "b" * 40]
    with patch.object(final, "freeze_sha", return_value=FREEZE_SHA):
        assert final.main(argv, out=lines.append) == 1
    assert "--require-sha is not the commit" in lines[0]
    assert "no freeze manifest" in _cli(root, sealed_dir, tmp_path / "missing.json")[1][0]
    path = root / "acct-2" / "scores" / "night-02.json"
    path.write_text(path.read_text(encoding="utf-8").replace(fireworks_api.DEEPSEEK_V4P1_FLASH, "muse-spark-1.3"),
                    encoding="utf-8")
    code, lines = _cli(root, sealed_dir, manifest)
    assert code == 1 and "not the registered judge" in lines[0]
    assert not any(final.score_path(root, account).exists() for account in ACCOUNTS.values())


# ── the registered rerun rule and the freeze commit ───────────────────────────


def _attempt_logged(root: Path) -> None:
    """An earlier attempt the ledger logged that wrote no score (an infrastructure failure)."""
    ledger.register(final.RUN_KIND, "e" * 64, f"lives={','.join(sorted(LIVES))} freeze={FREEZE_SHA} attempt=1",
                    path=_ledger(root.parent))


def test_every_attempt_and_score_is_logged_to_the_ledger(root: Path) -> None:
    _rig_all_wins(root)
    _run(root)
    kinds = [e.kind for e in ledger.read_entries(_ledger(root.parent))]
    assert kinds == ["freeze_manifest", final.RUN_KIND, *[final.SCORE_KIND] * 3]
    run_note = ledger.read_entries(_ledger(root.parent))[1].note
    assert run_note == f"lives={','.join(sorted(LIVES))} freeze={FREEZE_SHA} attempt=1"
    assert final.logged_attempts(LIVES, _ledger(root.parent)) == 1


def test_a_rerun_reason_without_a_logged_attempt_is_refused(root: Path) -> None:
    _rig_all_wins(root)
    with pytest.raises(final.FinalError, match="only for a logged attempt"):
        _run(root, rerun_reason="infra")
    assert not final.score_path(root, SINGLE).exists()


def test_a_logged_attempt_with_no_score_needs_its_infrastructure_reason(root: Path) -> None:
    _rig_all_wins(root)
    _sealed_files(root.parent)
    _manifest(root.parent, _sealed_files(root.parent))
    _attempt_logged(root)
    with pytest.raises(final.FinalError, match="earlier final attempt"):
        _run(root)
    with pytest.raises(final.FinalError, match="says why"):
        _run(root, rerun_reason="  ")
    report = _run(root, rerun_reason="the VM was preempted mid-write")
    assert report["attempt"] == 2 and report["rerun_reason"] == "the VM was preempted mid-write"
    stored = json.loads(final.score_path(root, SINGLE).read_text(encoding="utf-8"))
    assert stored["rerun_reason"] == "the VM was preempted mid-write"
    notes = [e.note for e in ledger.read_entries(_ledger(root.parent)) if e.kind == final.RUN_KIND]
    assert notes[-1].endswith("attempt=2 rerun: the VM was preempted mid-write")
    with pytest.raises(final.FinalError, match="refusal is absolute"):
        _run(root, rerun_reason="again")


def test_the_manifest_commit_must_be_the_freeze_sha_and_head(root: Path, tmp_path: Path) -> None:
    _rig_all_wins(root)
    sealed = _sealed_files(root.parent)
    base = tmp_path / "other"
    out = base / "manifest.json"
    freeze_manifest.write(freeze_manifest.build(list(sealed.values()), commit="c" * 40), out, ledger_file=_ledger(base))
    with pytest.raises(final.FinalError, match="not the freeze commit"):
        _run(root, sealed_files=sealed, manifest=out, ledger_file=_ledger(base))


def test_the_manifest_must_be_the_one_the_ledger_registers(root: Path, tmp_path: Path) -> None:
    _rig_all_wins(root)
    sealed = _sealed_files(root.parent)
    manifest = _manifest(root.parent, sealed)
    with pytest.raises(final.FinalError, match="registers 0 freeze manifests"):
        _run(root, ledger_file=tmp_path / "empty.jsonl")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")  # same content, other bytes
    with pytest.raises(final.FinalError, match="not the freeze manifest the ledger registers"):
        _run(root)


def test_the_manifest_lives_in_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ledger.DATA_DIR_ENV, str(tmp_path))
    assert freeze_manifest.manifest_path() == tmp_path / "sleep" / "freeze_manifest.json"
    monkeypatch.delenv(ledger.DATA_DIR_ENV)
    with pytest.raises(freeze_manifest.ManifestError, match="ANJO_DATA_DIR is not set"):
        freeze_manifest.manifest_path()


def test_the_manifest_is_written_only_from_the_freeze_checkout() -> None:
    answers = {("rev-parse", "HEAD"): "a" * 40 + "\n", ("rev-parse", "sleep-freeze^{commit}"): "b" * 40 + "\n"}

    def git(repo: Path, *args: str) -> str:
        return answers.get(args[:2], "")

    with patch.object(freeze_manifest, "_git", side_effect=git):
        with pytest.raises(freeze_manifest.ManifestError, match="HEAD is not the commit 'sleep-freeze' names"):
            freeze_manifest.head_commit()
    answers[("rev-parse", "sleep-freeze^{commit}")] = "a" * 40 + "\n"
    with patch.object(freeze_manifest, "_git", side_effect=git):
        assert freeze_manifest.head_commit() == "a" * 40
