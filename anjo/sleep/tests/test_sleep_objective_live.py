# mypy: disallow-untyped-decorators=False
"""The sleep objective's live comparison script, offline: plan first, fresh accounts only, the
arms in order with their switches, the control matched to the seed arm's minutes, and the report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.tools import sleep_objective_live as live
from anjo.sleep.train import objective

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
ACCOUNTS = ["--account", "seed=syn-a", "--account", "fold=syn-b", "--account", "today=syn-c"]


class _Ledger:
    def __init__(self, left: float) -> None:
        self.left = left

    def remaining(self, pool: str) -> float:
        assert pool == "modal_usd"
        return self.left


def _night(root: Path, account: str, day: str, *, minutes: float, terms: dict[str, Any] | None) -> None:
    folder = root / account / "nights" / day
    folder.mkdir(parents=True)
    train = {"step": "train", "train_minutes": minutes, **({"terms": terms} if terms else {})}
    gates = {"step": "gates", "gates": {"general": {"value": 0.5}}}
    (folder / "log.jsonl").write_text("\n".join(json.dumps(r) for r in (train, gates)) + "\n")
    (folder / "summary.json").write_text(json.dumps({"cost_usd": 1.25, "promoted": {"id": "x"}}))
    scores = root / account / "scores"
    scores.mkdir(exist_ok=True)
    n = len(list(scores.glob("night-*.json"))) + 1
    (scores / f"night-{n:02d}.json").write_text(json.dumps({"arms": {"3": {"overall": {"score": 0.6}},
                                                                     "4": {"overall": {"score": 0.7}}}}))


def _run(root: Path, calls: list[tuple[list[str], dict[str, str]]], fail: str | None = None) -> Any:
    def run(cmd: list[str], *, env: dict[str, str], check: bool) -> Any:
        calls.append((cmd, env))
        account = cmd[cmd.index("--user") + 1]
        _night(root, account, "2026-10-24", minutes=6.0 if account == "syn-a" else 3.0,
               terms={"minutes": {"seed": 1.0}, "rows": {"seed": 4}, "dreams": None} if account != "syn-c" else None)
        return SimpleNamespace(returncode=1 if account == fail else 0)
    return run


def test_the_plan_runs_nothing_and_prices_every_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_DATA_DIR", raising=False)  # a plan needs no data dir
    lines: list[str] = []
    calls: list[Any] = []
    assert live.main([*ACCOUNTS, "--days", "3", "--night-usd", "2.0"], run=lambda *a, **k: calls.append(a),
                     out=lines.append, now=lambda: NOW) == 0
    assert not calls and "plan only" in lines[0] and "--start 2026-10-25" in "\n".join(lines)
    assert live.plan_usd(["seed", "today"], 3, 2.0) == {"seed": round(3 * (2.0 + 3.0 / 60 * 2.5), 2),
                                                        "today": round(3 * (2.0 + 3.0 / 60 * 2.5), 2)}
    for bad in (["--account", "fold=syn-b"], ["--account", "seed=a", "--account", "fold=a"],
                ["--account", "seed=a", "--account", "nap=b"]):
        with pytest.raises(SystemExit):
            live.main(bad, out=lines.append)


def test_yes_runs_the_arms_in_order_on_new_accounts_with_their_switches(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(objective.DREAMS_ENV, "1")  # a stray switch in the shell never reaches an arm
    monkeypatch.setattr(live, "default_ledger", lambda: _Ledger(100.0))
    root = tmp_path / "sleep" / "results"
    calls: list[tuple[list[str], dict[str, str]]] = []
    lines: list[str] = []
    assert live.main([*ACCOUNTS, "--night-usd", "2.0", "--yes"], run=_run(root, calls), out=lines.append,
                     now=lambda: NOW) == 0
    users = [cmd[cmd.index("--user") + 1] for cmd, _ in calls]
    assert users == ["syn-a", "syn-b", "syn-c"]  # the matched control last
    seed_env, fold_env, today_env = (env for _, env in calls)
    assert seed_env[objective.OBJECTIVE_ENV] == "sft_dpo_seed" and objective.DREAMS_ENV not in seed_env
    assert fold_env[objective.FOLD_ENV] == "1" and today_env[objective.OBJECTIVE_ENV] == "sft_dpo"
    assert today_env[objective.MATCH_ENV] == "6.00"  # the seed arm's mean training minutes
    assert any(line.startswith("seed: ") for line in lines)
    # The same accounts again: they have results now, so nothing runs.
    with pytest.raises(SystemExit):
        live.main([*ACCOUNTS, "--night-usd", "2.0", "--yes"], run=_run(root, calls), out=lines.append, now=lambda: NOW)
    assert len(calls) == 3


def test_yes_stops_before_any_run_without_room_or_a_price(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live, "default_ledger", lambda: _Ledger(1.0))
    calls: list[Any] = []
    for argv in ([*ACCOUNTS, "--night-usd", "2.0", "--yes"], [*ACCOUNTS, "--yes"]):
        with pytest.raises(SystemExit):
            live.main(argv, run=_run(tmp_path, calls), out=lambda line: None, now=lambda: NOW)
    assert not calls


def test_a_failed_arm_stops_the_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live, "default_ledger", lambda: _Ledger(100.0))
    calls: list[Any] = []
    root = tmp_path / "sleep" / "results"
    assert live.main([*ACCOUNTS, "--night-usd", "2.0", "--yes"], run=_run(root, calls, fail="syn-b"),
                     out=lambda line: None, now=lambda: NOW) == 1
    assert len(calls) == 2


def test_the_report_reads_numbers_per_arm(tmp_path: Path) -> None:
    dreams = {"by_seed": {"boundary": {"keep_rate": 0.5}, "random": {"keep_rate": 0.25}}, "block_kept": True}
    _night(tmp_path, "syn-a", "2026-10-24", minutes=6.0, terms={"minutes": {"seed": 1.0}, "dreams": dreams})
    _night(tmp_path, "syn-a", "2026-10-25", minutes=4.0, terms={"minutes": {"seed": 1.0}, "dreams": None})
    table = live.report(tmp_path, {"seed": "syn-a"}, lambda line: None)
    seed = table["seed"]
    assert seed["nights"] == 2 and seed["promoted"] == 2 and seed["train_minutes_mean"] == 5.0
    assert seed["arm3_by_night"] == [0.6, 0.6] and seed["arm4_mean"] == 0.7 and seed["general_lost_by_night"] == [0.5, 0.5]
    assert seed["dreams_by_seed"] == [dreams["by_seed"]] and seed["dream_blocks_kept"] == 1 and seed["cost_usd"] == 2.5
    assert live.matched_minutes(tmp_path, "syn-a") == 5.0
    with pytest.raises(ValueError, match="no night"):
        live.matched_minutes(tmp_path, "syn-z")
