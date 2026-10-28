# mypy: disallow-untyped-decorators=False
"""The sleep objective's live comparison script, offline: plan first, fresh accounts only, every claim
with its matched control, the controls matched night by night through a match file, one hold for the
whole worst case shrinking arm by arm, sleep pressure off, and the report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.budget import POOLS, Ledger
from anjo.sleep.tools import sleep_objective_live as live
from anjo.sleep.train import objective
from anjo.sleep.train.terms import DREAMS_NOT_WIRED

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
ACCOUNTS = ["--account", "seed=syn-a", "--account", "fold=syn-b", "--account", "today=syn-c"]
DAY = "2026-10-24"
SIM_TOKEN = "ANJO_SLEEP_SIM_TOKEN"  # what app.sim_transport reads: the account each line is posted as


def _tokens(tmp_path: Path, accounts: list[str]) -> list[str]:
    """A token file per arm of ``accounts`` (``--account`` pairs), each holding that arm's own token."""
    folder = tmp_path / "tokens"
    folder.mkdir(parents=True, exist_ok=True)
    argv: list[str] = []
    for arm in (pair.partition("=")[0] for pair in accounts[1::2]):
        (folder / arm).write_text(f"tok-{arm}\n")
        argv += ["--token-file", f"{arm}={folder / arm}"]
    return argv


def _ledger(tmp_path: Path, cap: float) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", caps={**dict.fromkeys(POOLS, 0.0), "modal_usd": cap})


def _night(root: Path, account: str, day: str, *, minutes: dict[str, float] | None, dream: Any = None,
           match: dict[str, float] | None = None) -> None:
    folder = root / account / "nights" / day
    folder.mkdir(parents=True)
    train = {"step": "train", "train_minutes": 5.0, **({"terms": {"minutes": minutes}} if minutes else {}),
             **({"match": match} if match else {})}
    dataset = {"step": "dataset", **({"terms": {"dreams": dream, "dream_seconds": 30.0}} if dream is not None else {})}
    gates = {"step": "gates", "gates": {"general": {"value": 0.5}}}
    (folder / "log.jsonl").write_text("\n".join(json.dumps(r) for r in (train, dataset, gates)) + "\n")
    (folder / "summary.json").write_text(json.dumps({"cost_usd": 1.25, "promoted": {"id": "x"}}))
    scores = root / account / "scores"
    scores.mkdir(exist_ok=True)
    n = len(list(scores.glob("night-*.json"))) + 1
    (scores / f"night-{n:02d}.json").write_text(json.dumps({"arms": {"3": {"overall": {"score": 0.6}},
                                                                     "4": {"overall": {"score": 0.7}}}}))


class _Runs:
    """The accelerated runs, faked: each writes one night; it keeps the command, the env, the match
    file's contents and the ledger's open holds at launch."""

    def __init__(self, root: Path, ledger: Ledger, fail: str | None = None, dreams: Any = 3) -> None:
        self.root, self.ledger, self.fail, self.dreams = root, ledger, fail, dreams
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cmd: list[str], *, env: dict[str, str], check: bool) -> Any:
        account = cmd[cmd.index("--user") + 1]
        match = json.loads(Path(cmd[cmd.index("--match-file") + 1]).read_text()) if "--match-file" in cmd else None
        self.calls.append({"account": account, "env": env, "match": match,
                           "held": [r.bound.amount for r in self.ledger.open_holds()]})
        seeded = env[objective.OBJECTIVE_ENV] == "sft_dpo_seed"
        _night(self.root, account, DAY, minutes={"sft": 1.0, "seed": 2.0, "dpo": 0.5} if seeded else None,
               dream=self.dreams if env.get(objective.DREAMS_ENV) == "1" else None,
               match={"budget_minutes": match[DAY], "minutes": match[DAY] * 1.01} if match else None)
        return SimpleNamespace(returncode=1 if account == self.fail else 0)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: float = 100.0) -> tuple[Path, Ledger]:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    ledger = _ledger(tmp_path, cap)
    monkeypatch.setattr(live, "default_ledger", lambda: ledger)
    return tmp_path / "sleep" / "results", ledger


def test_the_plan_runs_nothing_and_prices_every_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANJO_DATA_DIR", raising=False)  # a plan needs no data dir
    monkeypatch.delenv("ANJO_SLEEP_CHANNEL", raising=False)
    lines: list[str] = []
    calls: list[Any] = []
    assert live.main([*ACCOUNTS, "--days", "3", "--night-usd", "2.0"], run=lambda *a, **k: calls.append(a),
                     out=lines.append, now=lambda: NOW) == 0
    text = "\n".join(lines)
    assert not calls and "plan only" in lines[0] and "3 arms" in lines[0] and "--start 2026-10-25" in text
    assert "ANJO_SLEEP_PRESSURE=0" in text and "--match-file <seed arm's minutes per night>" in text
    assert "thought channel: off" in text  # the shell's, the same in every arm
    assert live.arm_usd("seed", 3, 2.0) == round(3 * (2.0 + 3.0 / 60 * 2.5), 2)
    assert live.arm_usd("today", 3, 2.0) == round(3 * (2.0 + (2.83 + 3.0) / 60 * 2.5), 2)  # its whole budget
    assert live.arm_usd("today", 3, 2.0, {"a": 6.0, "b": 6.0, "c": 6.0}) == round(3 * 2.0 + 18.0 / 60 * 2.5, 2)
    for bad in (["--account", "fold=syn-b"], ["--account", "seed=a", "--account", "today=a"],
                ["--account", "seed=a", "--account", "nap=b"], ["--account", "seed=a"],
                ["--account", "seed=a", "--account", "today=b", "--account", "dreams=c"],
                ["--account", "seed=a", "--account", "today=b", "--account", "seed_matched=c"]):
        with pytest.raises(SystemExit):  # unknown arms, shared accounts, and a claim without its control
            live.main(bad, out=lines.append)


def test_yes_runs_each_control_after_its_arm_matched_night_by_night(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv(objective.DREAMS_ENV, "1")  # a stray switch in the shell never reaches an arm
    monkeypatch.setenv(live.PRESSURE_ENV, "1")
    runs, lines = _Runs(root, ledger), list[str]()
    assert live.main([*ACCOUNTS, *_tokens(tmp_path, ACCOUNTS), "--days", "1", "--night-usd", "2.0", "--yes"],
                     run=runs, out=lines.append, now=lambda: NOW) == 0
    assert [c["account"] for c in runs.calls] == ["syn-a", "syn-c", "syn-b"]  # seed, its control, the fold
    seed, today, fold = runs.calls
    assert seed["env"][objective.OBJECTIVE_ENV] == "sft_dpo_seed" and objective.DREAMS_ENV not in seed["env"]
    assert all(c["env"][live.PRESSURE_ENV] == "0" for c in runs.calls)  # sleep pressure off in every arm
    assert today["env"][objective.OBJECTIVE_ENV] == "sft_dpo" and fold["env"][objective.FOLD_ENV] == "1"
    assert seed["match"] is None and today["match"] == {DAY: 3.0}  # the seed night's minutes, its DPO left out
    # The hold covers the arms still to come at each launch, and none is left at the end.
    priced = {arm: live.arm_usd(arm, 1, 2.0) for arm in ("seed", "today", "fold")}
    assert seed["held"] == [pytest.approx(priced["today"] + priced["fold"])]
    assert today["held"] == [pytest.approx(priced["fold"])] and fold["held"] == [] and not ledger.open_holds()
    assert any(line.startswith("today: ") for line in lines) and not any("isn't matched" in line for line in lines)
    with pytest.raises(SystemExit):  # the same accounts again: they have results now, so nothing runs
        live.main([*ACCOUNTS, *_tokens(tmp_path, ACCOUNTS), "--night-usd", "2.0", "--yes"], run=runs,
                  out=lines.append, now=lambda: NOW)
    assert len(runs.calls) == 3


def test_each_arm_gets_its_own_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch)
    runs, lines = _Runs(root, ledger), list[str]()
    assert live.main([*ACCOUNTS, *_tokens(tmp_path, ACCOUNTS), "--days", "1", "--night-usd", "2.0", "--yes"],
                     run=runs, out=lines.append, now=lambda: NOW) == 0
    assert {c["account"]: c["env"][SIM_TOKEN] for c in runs.calls} == {"syn-a": "tok-seed", "syn-c": "tok-today",
                                                                        "syn-b": "tok-fold"}
    assert [line for line in lines if "token file" in line] == [f"    {arm}: token file ok" for arm in ("seed", "today", "fold")]
    assert not any("tok-" in line or "tokens" in line for line in lines)  # neither a token nor its file is printed


def test_shell_token_never_reaches_an_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv(SIM_TOKEN, "tok-shell")
    runs = _Runs(root, ledger)
    assert live.main([*ACCOUNTS, *_tokens(tmp_path, ACCOUNTS), "--days", "1", "--night-usd", "2.0", "--yes"],
                     run=runs, out=lambda line: None, now=lambda: NOW) == 0
    assert len(runs.calls) == 3 and not any("tok-shell" in c["env"].values() for c in runs.calls)
    assert live.arm_env("fold", {SIM_TOKEN: "tok-shell", "PATH": "/bin"}, "tok-fold") == {
        "PATH": "/bin", SIM_TOKEN: "tok-fold", **live.ENVS["fold"], live.PRESSURE_ENV: "0"}


def test_yes_refuses_a_missing_token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch)
    runs = _Runs(root, ledger)
    ok = _tokens(tmp_path, ACCOUNTS)  # seed, fold, today
    both = ok[:4]
    for argv in (both, [*both, "--token-file", f"today={tmp_path / 'gone'}"],  # not given, not there
                 [*ok, "--token-file", f"dreams={tmp_path / 'tokens' / 'seed'}"]):  # an arm that isn't run
        with pytest.raises(SystemExit):
            live.main([*ACCOUNTS, *argv, "--days", "1", "--night-usd", "2.0", "--yes"], run=runs,
                      out=lambda line: None, now=lambda: NOW)
    (tmp_path / "tokens" / "today").write_text(" \n")  # there, but empty
    with pytest.raises(SystemExit):
        live.main([*ACCOUNTS, *ok, "--days", "1", "--night-usd", "2.0", "--yes"], run=runs, out=lambda line: None,
                  now=lambda: NOW)
    assert not runs.calls and not ledger.open_holds()
    lines: list[str] = []  # the plan says which, and runs nothing
    assert live.main([*ACCOUNTS, *both, "--days", "1"], out=lines.append, now=lambda: NOW) == 0
    assert "    seed: token file ok" in lines and "    today: no token file" in lines


PROTECT = ["--account", "seed=a", "--account", "today=b", "--account", "fold=c", "--account", "protect=d",
           "--account", "protect_random=e"]


def test_protect_runs_after_the_fold_with_its_random_twin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv(objective.PROTECT_ENV, "anchors")  # a stray switch in the shell never reaches an arm
    runs, lines = _Runs(root, ledger), list[str]()
    assert live.main([*PROTECT, *_tokens(tmp_path, PROTECT), "--days", "1", "--night-usd", "2.0", "--yes"], run=runs,
                     out=lines.append, now=lambda: NOW) == 0
    assert [c["account"] for c in runs.calls] == ["a", "b", "c", "d", "e"]
    seed, today, fold, protect, twin = runs.calls
    assert protect["env"][objective.PROTECT_ENV] == "anchors" and twin["env"][objective.PROTECT_ENV] == "random"
    assert protect["env"][objective.FOLD_ENV] == twin["env"][objective.FOLD_ENV] == "1"  # the slot the fold keeps apart
    assert twin["match"] is None and not any(objective.PROTECT_ENV in c["env"] for c in (seed, today, fold))
    assert not any("part from" in line for line in lines)  # the twin's minutes matched protect's
    assert live.arm_usd("protect_random", 3, 2.0) == live.arm_usd("protect", 3, 2.0) == round(3 * 2.0 + 3 * 4.6 / 60 * 2.5, 2)
    for bad in (PROTECT[:8], [*PROTECT[:4], *PROTECT[6:]], [*PROTECT[:6], *PROTECT[8:]]):  # no twin, no fold, no protect
        with pytest.raises(SystemExit):
            live.main(bad, out=lambda line: None)
    _night(root, "e", "2026-10-25", minutes={"sft": 9.0, "seed": 9.0})  # the twin trains far longer one night
    flagged: list[str] = []
    live.report(root, dict(pair.split("=") for pair in PROTECT[1::2]), flagged.append)
    assert any(line.startswith("  protect_random: its training minutes part from protect's") for line in flagged)


def test_unwired_dreams_skip_their_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch)
    runs, lines = _Runs(root, ledger, dreams=DREAMS_NOT_WIRED), list[str]()
    argv = ["--account", "seed=a", "--account", "today=b", "--account", "dreams=c", "--account", "seed_matched=d"]
    argv += _tokens(tmp_path, argv)
    assert live.main([*argv, "--days", "1", "--night-usd", "2.0", "--yes"], run=runs, out=lines.append, now=lambda: NOW) == 0
    assert [c["account"] for c in runs.calls] == ["a", "b", "c"] and any("doesn't run" in line for line in lines)
    assert not ledger.open_holds()
    root, ledger = _setup(tmp_path / "wired", monkeypatch)
    runs = _Runs(root, ledger)
    assert live.main([*argv, "--days", "1", "--night-usd", "2.0", "--yes"], run=runs, out=lines.append, now=lambda: NOW) == 0
    assert runs.calls[-1]["account"] == "d" and runs.calls[-1]["match"] == {DAY: 3.5}  # serving minutes counted
    assert runs.calls[-1]["env"][objective.OBJECTIVE_ENV] == "sft_dpo_seed" and objective.DREAMS_ENV not in runs.calls[-1]["env"]


def test_yes_stops_before_any_run_without_room_or_a_price(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch, cap=1.0)
    runs = _Runs(root, ledger)
    tokens = _tokens(tmp_path, ACCOUNTS)
    for argv in ([*ACCOUNTS, *tokens, "--night-usd", "2.0", "--yes"], [*ACCOUNTS, *tokens, "--yes"]):
        with pytest.raises(SystemExit):
            live.main(argv, run=runs, out=lambda line: None, now=lambda: NOW)
    assert not runs.calls and not ledger.open_holds()


def test_a_failed_arm_stops_the_rest_and_frees_the_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ledger = _setup(tmp_path, monkeypatch)
    runs = _Runs(root, ledger, fail="syn-c")
    assert live.main([*ACCOUNTS, *_tokens(tmp_path, ACCOUNTS), "--days", "1", "--night-usd", "2.0", "--yes"], run=runs,
                     out=lambda line: None, now=lambda: NOW) == 1
    assert len(runs.calls) == 2 and not ledger.open_holds()


def test_the_report_reads_numbers_per_arm_and_flags_an_unmatched_night(tmp_path: Path) -> None:
    dreams = {"by_seed": {"boundary": {"keep_rate": 0.5}, "random": {"keep_rate": 0.25}}, "block_kept": True}
    _night(tmp_path, "syn-a", "2026-10-24", minutes={"seed": 1.0, "dpo": 0.5})
    _night(tmp_path, "syn-a", "2026-10-25", minutes={"seed": 3.0})
    (tmp_path / "syn-a" / "nights" / "2026-10-24" / "log.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"step": "train", "train_minutes": 6.0, "terms": {"minutes": {"seed": 1.0, "dpo": 0.5}, "dreams": dreams}},
        {"step": "gates", "gates": {"general": {"value": 0.5}}})) + "\n")
    _night(tmp_path, "syn-c", "2026-10-24", minutes=None, match={"budget_minutes": 1.0, "minutes": 1.2})
    lines: list[str] = []
    table = live.report(tmp_path, {"seed": "syn-a", "today": "syn-c"}, lines.append)
    seed = table["seed"]
    assert seed["nights"] == 2 and seed["promoted"] == 2 and seed["train_minutes_mean"] == 5.5
    assert seed["arm3_by_night"] == [0.6, 0.6] and seed["arm4_mean"] == 0.7 and seed["general_lost_by_night"] == [0.5, 0.5]
    assert seed["dreams_by_seed"] == [dreams["by_seed"]] and seed["dream_blocks_kept"] == 1 and seed["cost_usd"] == 2.5
    assert table["today"]["match"] == [{"night": "2026-10-24", "budget": 1.0, "used": 1.2, "mismatch": True}]
    assert any("isn't matched" in line for line in lines)
    assert live.budgets(tmp_path, "syn-a") == {"2026-10-24": 1.0, "2026-10-25": 3.0}
    with pytest.raises(ValueError, match="no night"):
        live.budgets(tmp_path, "syn-z")
    with pytest.raises(ValueError, match="minutes by term"):
        live.budgets(tmp_path, "syn-c")
