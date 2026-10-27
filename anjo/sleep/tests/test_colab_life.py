# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""One Colab VM per accelerated life (Task 12a): the base weights download once, every Colab rule
still holds (timeouts, synthetic people only, only the dataset and the previous adapter go up,
results come off before the stop, the VM stops on failure, holds are reserved and settled), and the
CLI is a fake: no VM, no network."""
from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.serving import config
from anjo.sleep.stops import BudgetStop
from anjo.sleep.tests.test_colab_runner import FakeColab
from anjo.sleep.train.lora import DATASET_FILE, JOB, adapter_name
from anjo.sleep.train.runners.colab import (
    A100_CU_PER_HOUR,
    ColabCli,
    ColabError,
    ColabRunner,
    RealPersonRefused,
)

NIGHT_S = 7200.0
LIFE_S = 6 * 3600.0


class Minute:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 60.0
        return self.now


def _runner(fake: FakeColab, ledger: Ledger, *, allow_real: bool = False) -> ColabRunner:
    return ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake), ledger=ledger,
                       is_synthetic=lambda uid: uid.startswith("synthetic"), allow_real=allow_real, clock=Minute())


def _night(tmp_path: Path, person: str, number: int, *, warm: bool) -> dict[str, Any]:
    work = tmp_path / person / f"work{number}"
    work.mkdir(parents=True, exist_ok=True)
    (work / DATASET_FILE).write_text('{"id": "x"}\n')
    prev = None
    if warm:
        prev = tmp_path / person / f"prev{number}"
        prev.mkdir(exist_ok=True)
        for name in config.ADAPTER_FILES:
            (prev / name).write_text("{}")
    return {"person_id": person, "adapter_name": adapter_name(person, number, 0), "night": f"2026-11-0{number}",
            "dataset_path": str(work / DATASET_FILE), "workdir": str(work), "config": {"r": 16},
            "prev_adapter_dir": None if prev is None else str(prev)}


def _events(ledger: Ledger) -> list[dict[str, Any]]:
    return read_records(ledger.path, PLAIN)


class FailNth(FakeColab):
    """The nth call of one subcommand fails."""

    def __init__(self, command: str, nth: int) -> None:
        super().__init__()
        self.command, self.nth, self.seen = command, nth, 0

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        if list(args)[3] == self.command:
            self.seen += 1
            if self.seen == self.nth:
                self.calls.append((list(args), timeout_s))
                return subprocess.CompletedProcess(list(args), 1, "", "boom")
        return super().__call__(args, timeout_s)


def test_one_vm_per_accelerated_life(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
        for number in (1, 2, 3):
            out = lives.run(JOB, _night(tmp_path, "synthetic-1", number, warm=number > 1), timeout_s=NIGHT_S)
            assert out["steps"] == 9
        assert fake.commands().count("stop") == 0  # the VM waits for the next night
    commands = fake.commands()
    assert commands.count("new") == 1 and commands.count("install") == 1  # base weights come down once
    assert commands.count("exec") == 3 and commands[-1] == "stop" and commands.count("stop") == 1
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle"]  # one hold for the life
    assert len({args[args.index("-s") + 1] for args, _ in fake.calls}) == 1  # one session throughout


def test_two_lives_get_two_vms_and_both_stop(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
        for number in (1, 2):
            for person in ("synthetic-1", "synthetic-2"):
                lives.run(JOB, _night(tmp_path, person, number, warm=number > 1), timeout_s=NIGHT_S)
    assert fake.commands().count("new") == 2 and fake.commands().count("stop") == 2
    sessions = {args[args.index("-s") + 1] for args, _ in fake.calls}
    assert len(sessions) == 2 and all(s.startswith("anjo-sleep-") for s in sessions)  # the sweep can find them
    assert not any("synthetic" in s for s in sessions)  # and a session name never carries an id


def test_only_each_nights_dataset_and_the_previous_adapter_go_up(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
        for number in (1, 2):
            lives.run(JOB, _night(tmp_path, "synthetic-1", number, warm=number > 1), timeout_s=NIGHT_S)
    uploaded = [Path(args[-2]) for args, _ in fake.calls if args[3] == "upload"]
    assert uploaded == [tmp_path / "synthetic-1" / "work1" / DATASET_FILE,
                        tmp_path / "synthetic-1" / "work2" / DATASET_FILE,
                        *(tmp_path / "synthetic-1" / "prev2" / f for f in config.ADAPTER_FILES)]
    assert not any(".env" in part for args, _ in fake.calls for part in args)
    downloads = [args for args, _ in fake.calls if args[3] == "download"]
    assert len(downloads) == 2 * len(config.ADAPTER_FILES + ("receipt.json",))  # results off before the stop
    assert fake.commands()[-1] == "stop" and fake.commands().index("stop") > max(
        i for i, c in enumerate(fake.commands()) if c == "download")


def test_every_call_of_a_life_has_a_finite_timeout(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
        for number in (1, 2):
            lives.run(JOB, _night(tmp_path, "synthetic-1", number, warm=number > 1), timeout_s=NIGHT_S)
    assert all(0 < timeout < math.inf for _, timeout in fake.calls)


@pytest.mark.parametrize("kind", ["night", "life"])
def test_colab_runner_refuses_non_synthetic_person(tmp_path: Path, kind: str) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    runner = _runner(fake, ledger)
    inputs = _night(tmp_path, "kevin", 1, warm=False)
    with pytest.raises(RealPersonRefused):
        if kind == "night":
            runner.run(JOB, inputs, timeout_s=NIGHT_S)
        else:
            with runner.lives(life_timeout_s=LIFE_S) as lives:
                lives.run(JOB, inputs, timeout_s=NIGHT_S)
    assert fake.calls == [] and _events(ledger) == []  # nothing started, nothing held


@pytest.mark.parametrize("kind", ["night", "life"])
def test_colab_runner_stops_vm_on_failure(tmp_path: Path, kind: str) -> None:
    fake, ledger = FailNth("exec", 1), Ledger(tmp_path / "budget.jsonl")
    runner = _runner(fake, ledger)
    with pytest.raises(ColabError):
        if kind == "night":
            runner.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
        else:
            with runner.lives(life_timeout_s=LIFE_S) as lives:
                lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
    assert fake.commands()[-1] == "stop" and fake.commands().count("stop") == 1
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle"]


def test_a_failed_night_stops_that_lifes_vm_and_the_next_night_starts_a_fresh_one(tmp_path: Path) -> None:
    fake, ledger = FailNth("exec", 2), Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
        lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
        with pytest.raises(ColabError):
            lives.run(JOB, _night(tmp_path, "synthetic-1", 2, warm=True), timeout_s=NIGHT_S)
        assert fake.commands()[-1] == "stop"  # stopped at the failure, not at the end of the batch
        lives.run(JOB, _night(tmp_path, "synthetic-1", 3, warm=True), timeout_s=NIGHT_S)
    assert fake.commands().count("new") == 2 and fake.commands().count("stop") == 2
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle", "reserve", "settle"]


def test_the_batch_ending_in_an_error_still_stops_every_vm(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(RuntimeError, match="batch broke"):
        with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
            lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
            lives.run(JOB, _night(tmp_path, "synthetic-2", 1, warm=False), timeout_s=NIGHT_S)
            raise RuntimeError("batch broke")
    assert fake.commands().count("stop") == 2
    assert not ledger.open_holds()


def test_a_stop_that_fails_at_the_end_says_so_and_still_stops_the_others(tmp_path: Path) -> None:
    fake, ledger = FailNth("stop", 1), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabError, match="by hand"):
        with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
            lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
            lives.run(JOB, _night(tmp_path, "synthetic-2", 1, warm=False), timeout_s=NIGHT_S)
    assert fake.commands().count("stop") == 2
    assert not ledger.open_holds()  # the failed one is settled at its whole hold


def test_a_life_holds_its_worst_case_before_any_vm_and_settles_what_ran(tmp_path: Path) -> None:
    fake = FakeColab()
    tight = Ledger(tmp_path / "tight.jsonl", caps={**CAPS, "colab_cu": 1.0})
    with pytest.raises(BudgetStop):
        with _runner(fake, tight).lives(life_timeout_s=LIFE_S) as lives:
            lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
    assert fake.calls == []
    ledger = Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives:
        lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
        reserve = _events(ledger)[0]
        assert reserve["pools"] == ["colab_cu"] and reserve["timeout_s"] == LIFE_S
        assert reserve["amount"] == pytest.approx(LIFE_S / 3600 * A100_CU_PER_HOUR)
    settle = _events(ledger)[1]
    assert 0 < settle["amount"] < reserve["amount"]


def test_a_night_that_would_outlast_the_lifes_hold_is_refused_before_it_starts(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=NIGHT_S + 60.0) as lives:
        lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
        calls = len(fake.calls)
        with pytest.raises(ColabError, match="hold"):
            lives.run(JOB, _night(tmp_path, "synthetic-1", 2, warm=True), timeout_s=NIGHT_S)
        assert len(fake.calls) == calls  # no call was made for the refused night
    assert fake.commands().count("stop") == 1


def test_only_the_training_job_runs_on_a_life(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _runner(fake, ledger).lives(life_timeout_s=LIFE_S) as lives, pytest.raises(ColabError):
        lives.run("serve", _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
    assert fake.calls == []
