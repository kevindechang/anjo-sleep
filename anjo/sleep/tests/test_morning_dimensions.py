# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The comparison's mornings by dimension (``tools.morning_dimensions``): each arm's stored daily-v1
scores, arm 3 against base, split into knowledge of her world and taste and style, per night and per
kind, with the counts behind each number. Score files written here; nothing is called."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anjo.sleep.tools import morning_dimensions as dims

KINDS = {"nw1": "new_work", "nw2": "new_work", "v1": "vocab", "th1": "tone_hold", "t1": "taste_ab", "pb1": "pushback"}


def _stat(wins: int, ties: int, losses: int, ci: list[float] | None) -> dict[str, Any]:
    n = wins + ties + losses
    return {"n": n, "wins": wins, "ties": ties, "losses": losses, "score": (wins + ties / 2) / n,
            "decisive_share": None, "ci": ci}


def _score(root: Path, account: str, number: int, night: str, outcomes: dict[str, tuple[str, bool]],
           subset: str | None = "daily-v1") -> None:
    rows = [{"probe": probe, "pair": "3_vs_1", "kind": KINDS[probe], "outcome": outcome, "flip": flip}
            for probe, (outcome, flip) in outcomes.items()]
    rows.append({"probe": "nw1", "pair": "4_vs_2C", "kind": "new_work", "outcome": "loss", "flip": False})  # another pair
    record = {"night": night, "checkpoint": f"p-abc-n{number}", "checkpoint_basis": "promoted", "probes": 5,
              "arms": {"3": {"knowledge": _stat(2, 1, 0, [0.5, 1.0]), "interaction": _stat(1, 0, 1, [0.0, 1.0]),
                             "overall": _stat(3, 1, 1, [0.4, 0.9])}},
              "pairs": {}, "outcomes": rows, **({"subset": subset} if subset else {})}
    folder = root / account / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"night-{number:02d}.json").write_text(json.dumps(record), encoding="utf-8")


def _lines(root: Path, *argv: str) -> list[dict[str, Any]]:
    said: list[str] = []
    assert dims.main(list(argv), root=root, out=said.append) == 0
    return [json.loads(line) for line in said]


def test_each_night_splits_arm_3_against_base_by_dimension_and_kind_with_its_counts(tmp_path: Path) -> None:
    _score(tmp_path, "acct-fold", 1, "2026-11-01", {"nw1": ("win", False), "nw2": ("tie", True), "v1": ("win", False),
                                                   "th1": ("loss", False), "t1": ("no_verdict", False),
                                                   "pb1": ("win", False)})
    _score(tmp_path, "acct-fold", 2, "2026-11-01", {"nw1": ("loss", False)}, subset=None)  # the full set: not daily-v1
    [night] = [line for line in _lines(tmp_path, "--account", "fold=acct-fold") if line["line"] == "night"]
    assert (night["arm"], night["night"], night["checkpoint"]) == ("fold", "2026-11-01", "p-abc-n1")
    assert set(night["dimensions"]) == {"knowledge", "interaction"}  # pushback is reported, in no dimension
    assert night["kinds"]["pushback"]["wins"] == 1
    knowledge = night["dimensions"]["knowledge"]
    assert {k: knowledge[k] for k in ("probes", "orders", "wins", "ties", "flips", "losses", "unjudged")} == {
        "probes": 3, "orders": 6, "wins": 2, "ties": 1, "flips": 1, "losses": 0, "unjudged": 0}
    assert knowledge["score"] == 0.8333 and knowledge["ci"] == [0.5, 1.0]  # the scorer's own interval, as stored
    taste = night["dimensions"]["interaction"]
    assert (taste["probes"], taste["losses"], taste["unjudged"], taste["score"]) == (2, 1, 1, 0.0)
    assert night["kinds"]["new_work"]["probes"] == 2 and night["kinds"]["taste_ab"]["unjudged"] == 1
    assert "ci" not in night["kinds"]["new_work"]  # the scorer has no interval per kind


def test_an_arm_pools_its_mornings_without_an_interval_and_an_account_without_scores_says_so(tmp_path: Path) -> None:
    for number, night in ((1, "2026-11-01"), (2, "2026-11-02")):
        _score(tmp_path, "acct-seed", number, night, {"nw1": ("win", False), "th1": ("tie", True)})
    lines = _lines(tmp_path, "--account", "seed=acct-seed", "--account", "seed_cap=acct-cap")
    [pooled] = [line for line in lines if line["line"] == "arm"]
    assert pooled["mornings"] == 2 and pooled["nights"] == ["2026-11-01", "2026-11-02"]
    assert pooled["dimensions"]["knowledge"]["wins"] == 2 and pooled["dimensions"]["interaction"]["flips"] == 2
    assert pooled["dimensions"]["knowledge"]["ci"] is None and "same probes" in pooled["why_no_ci"]
    assert {"line": "missing", "arm": "seed_cap", "account": "acct-cap", "why": "no daily-v1 morning scores"} in lines


def test_the_probe_line_says_how_little_one_morning_can_tell() -> None:
    line = dims.probe_line()
    assert line["per_kind"] == {"new_work": 5, "vocab": 3, "tone_hold": 3, "taste_ab": 3}
    assert line["dimensions"] == {"knowledge": 8, "interaction": 6}
    assert line["one_probe_moves"] == {"knowledge": 0.125, "interaction": 0.1667}
