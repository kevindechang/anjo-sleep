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
    command_label,
    A100_CU_PER_HOUR,
    MAX_IDLE_BETWEEN_NIGHTS_S,
    ColabCli,
    ColabLives,
    ColabError,
    ColabRunner,
    RealPersonRefused,
)

NIGHT_S = 7200.0
LIFE_S = 6 * 3600.0
SHORT_DAY_S = 60.0


class Minute:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 60.0
        return self.now


def _runner(fake: FakeColab, ledger: Ledger, *, allow_real: bool = False) -> ColabRunner:
    return ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake), ledger=ledger,
                       is_synthetic=lambda uid: uid.startswith("synthetic"), allow_real=allow_real, clock=Minute())


def _lives(runner: ColabRunner, life_s: float = LIFE_S, day_s: float | None = SHORT_DAY_S) -> ColabLives:
    """A life runner told that each life's simulated days are short (so the VM may be kept)."""
    lives = runner.lives(life_timeout_s=life_s)
    if day_s is not None:
        for person in ("synthetic-1", "synthetic-2"):
            lives.note_day(person, day_s)
    return lives


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
        if command_label(list(args)[3:]) == self.command:
            self.seen += 1
            if self.seen == self.nth:
                self.calls.append((list(args), timeout_s))
                return subprocess.CompletedProcess(list(args), 1, "", "boom")
        return super().__call__(args, timeout_s)


def test_one_vm_per_accelerated_life(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        for number in (1, 2, 3):
            out = lives.run(JOB, _night(tmp_path, "synthetic-1", number, warm=number > 1), timeout_s=NIGHT_S)
            assert out["steps"] == 9
        assert fake.commands().count("stop") == 1  # the VM of nights 2 and 3 waits for the next night
    commands = fake.commands()
    # night 1 has nothing measured, so its VM stops; nights 2 and 3 share one VM (base weights once)
    assert commands.count("new") == 2 and commands.count("install") == 2
    assert commands.count("mkdir") == commands.count("new")  # the remote folders once a VM, never on a warm night
    assert commands.count("exec") == 3 and commands[-1] == "stop" and commands.count("stop") == 2
    assert commands.index("stop") < commands.index("exec", commands.index("exec") + 1)
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle", "reserve", "settle"]


def test_two_lives_get_two_vms_and_both_stop(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        for number in (1, 2):
            for person in ("synthetic-1", "synthetic-2"):
                lives.run(JOB, _night(tmp_path, person, number, warm=number > 1), timeout_s=NIGHT_S)
    assert fake.commands().count("new") == 4 and fake.commands().count("stop") == 4
    sessions = {args[args.index("-s") + 1] for args, _ in fake.calls}
    assert len(sessions) == 2 and all(s.startswith("anjo-sleep-") for s in sessions)  # the sweep can find them
    assert not any("synthetic" in s for s in sessions)  # and a session name never carries an id


def test_only_each_nights_dataset_and_the_previous_adapter_go_up(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        for number in (1, 2):
            lives.run(JOB, _night(tmp_path, "synthetic-1", number, warm=number > 1), timeout_s=NIGHT_S)
    uploaded = [Path(args[-2]) for args, _ in fake.calls if args[3] == "upload"]
    assert uploaded == [tmp_path / "synthetic-1" / "work1" / DATASET_FILE,
                        tmp_path / "synthetic-1" / "work2" / DATASET_FILE,
                        *(tmp_path / "synthetic-1" / "prev2" / f for f in config.ADAPTER_FILES)]
    assert not any(".env" in part for args, _ in fake.calls for part in args)
    downloads = [args for args, _ in fake.calls if args[3] == "download"]
    assert len(downloads) == 2 * len(config.ADAPTER_FILES + ("receipt.json",))  # results off before the stop
    commands = fake.commands()
    assert commands[-1] == "stop" and len(commands) - 1 > max(i for i, c in enumerate(commands) if c == "download")


def test_every_call_of_a_life_has_a_finite_timeout(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
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
            with _lives(runner) as lives:
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
            with _lives(runner) as lives:
                lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
    assert fake.commands()[-1] == "stop" and fake.commands().count("stop") == 1
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle"]


def test_a_failed_night_stops_that_lifes_vm_and_the_next_night_starts_a_fresh_one(tmp_path: Path) -> None:
    fake, ledger = FailNth("exec", 2), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        _run(lives, tmp_path, "synthetic-1", 1)
        with pytest.raises(ColabError):
            _run(lives, tmp_path, "synthetic-1", 2)
        assert fake.commands()[-1] == "stop"  # stopped at the failure, not at the end of the batch
        _run(lives, tmp_path, "synthetic-1", 3)
    assert fake.commands().count("new") == 3 and fake.commands().count("stop") == 3
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle"] * 3


def test_the_batch_ending_in_an_error_still_stops_every_vm(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(RuntimeError, match="batch broke"):
        with _lives(_runner(fake, ledger)) as lives:
            for person in ("synthetic-1", "synthetic-2"):
                _run(lives, tmp_path, person, 1)
                _run(lives, tmp_path, person, 2)  # kept: a short day follows
            raise RuntimeError("batch broke")
    assert fake.commands().count("stop") == 4
    assert not ledger.open_holds()


def test_a_stop_that_fails_at_the_end_says_so_and_still_stops_the_others(tmp_path: Path) -> None:
    fake, ledger = FailNth("stop", 3), Ledger(tmp_path / "budget.jsonl")  # the two first-night stops pass
    with pytest.raises(ColabError, match="by hand"):
        with _lives(_runner(fake, ledger)) as lives:
            for person in ("synthetic-1", "synthetic-2"):
                _run(lives, tmp_path, person, 1)
                _run(lives, tmp_path, person, 2)
    assert fake.commands().count("stop") == 4
    assert not ledger.open_holds()  # the failed one is settled at its whole hold


def test_a_life_holds_its_worst_case_before_any_vm_and_settles_what_ran(tmp_path: Path) -> None:
    fake = FakeColab()
    tight = Ledger(tmp_path / "tight.jsonl", caps={**CAPS, "colab_cu": 1.0})
    with pytest.raises(BudgetStop):
        with _lives(_runner(fake, tight)) as lives:
            lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
    assert fake.calls == []
    ledger = Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
        reserve = _events(ledger)[0]
        assert reserve["pools"] == ["colab_cu"] and reserve["timeout_s"] == LIFE_S
        assert reserve["amount"] == pytest.approx(LIFE_S / 3600 * A100_CU_PER_HOUR)
    settle = _events(ledger)[1]
    assert 0 < settle["amount"] < reserve["amount"]


def test_only_the_training_job_runs_on_a_life(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives, pytest.raises(ColabError):
        lives.run("serve", _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=NIGHT_S)
    assert fake.calls == []


def _run(lives: ColabLives, tmp_path: Path, person: str, number: int, timeout_s: float = NIGHT_S) -> None:
    lives.run(JOB, _night(tmp_path, person, number, warm=number > 1), timeout_s=timeout_s)


def test_a_refused_night_stops_the_vm_it_leaves_up(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger), NIGHT_S + 60.0) as lives:
        _run(lives, tmp_path, "synthetic-1", 1)  # the first night stops its VM
        _run(lives, tmp_path, "synthetic-1", 2)  # a short day: this VM is kept
        assert fake.commands().count("stop") == 1
        with pytest.raises(ColabError, match="hold"):
            _run(lives, tmp_path, "synthetic-1", 3)
        assert fake.commands().count("stop") == 2 and fake.commands()[-1] == "stop"  # no VM idles through the days
    assert fake.commands().count("stop") == 2 and not ledger.open_holds()


def test_a_night_with_too_little_time_stops_the_vm_it_leaves_up(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        _run(lives, tmp_path, "synthetic-1", 1)
        _run(lives, tmp_path, "synthetic-1", 2)
        with pytest.raises(ColabError, match="too little time"):
            _run(lives, tmp_path, "synthetic-1", 3, timeout_s=1000.0)
        assert fake.commands()[-1] == "stop" and fake.commands().count("stop") == 2
    assert not ledger.open_holds()


def test_the_first_night_stops_the_vm_because_nothing_is_measured(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        _run(lives, tmp_path, "synthetic-1", 1)
        assert fake.commands()[-1] == "stop"
    assert lives.records("synthetic-1") == [{"night": "2026-11-01", "vm_kept": False, "idle_s_before": None, "day_s": 60.0,
                                                "scoring_s": None}]


def test_a_short_simulated_day_keeps_the_vm_and_a_long_one_stops_it(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        _run(lives, tmp_path, "synthetic-1", 1)
        _run(lives, tmp_path, "synthetic-1", 2)  # the day before was 60 s: kept
        assert fake.commands().count("stop") == 1
        lives.note_day("synthetic-1", MAX_IDLE_BETWEEN_NIGHTS_S)  # exactly the limit still keeps it
        _run(lives, tmp_path, "synthetic-1", 3)
        assert fake.commands().count("stop") == 1
        lives.note_day("synthetic-1", MAX_IDLE_BETWEEN_NIGHTS_S + 1.0)  # a long day: stopped
        _run(lives, tmp_path, "synthetic-1", 4)
        assert fake.commands()[-1] == "stop" and fake.commands().count("stop") == 2
    kept = [r["vm_kept"] for r in lives.records("synthetic-1")]
    idle = [r["idle_s_before"] for r in lives.records("synthetic-1")]
    assert kept == [False, True, True, False]
    assert idle[0] is None and idle[1] is None and isinstance(idle[2], float) and idle[2] > 0 and idle[3] > 0  # type: ignore[operator]
    assert fake.commands().count("new") == 2  # nights 2 to 4 shared one VM; night 1 had its own


def test_a_life_with_no_measured_day_is_stopped_after_every_night(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger), day_s=None) as lives:
        for number in (1, 2, 3):
            _run(lives, tmp_path, "synthetic-1", number)
    assert fake.commands().count("stop") == 3 and fake.commands().count("new") == 3


def test_a_failing_finish_on_one_life_still_stops_the_others(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    runner = _runner(fake, ledger)
    real = runner._finish
    calls: list[str] = []

    def flaky(vm: Any, session: str, reservation: Any, started: float) -> float:
        calls.append(session)
        if len(calls) == 1:
            raise RuntimeError("the ledger broke")
        return real(vm, session, reservation, started)

    with pytest.raises(RuntimeError, match="the ledger broke"):
        with _lives(runner) as lives:
            for person in ("synthetic-1", "synthetic-2"):
                _run(lives, tmp_path, person, 1)
                _run(lives, tmp_path, person, 2)
            runner._finish = flaky  # type: ignore[method-assign]
    assert len(calls) == 2  # both lives were tried
    assert fake.commands().count("stop") == 3  # the two first-night stops, and the second life's


def test_the_morning_scoring_counts_toward_the_idle_gap(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with _lives(_runner(fake, ledger)) as lives:
        _run(lives, tmp_path, "synthetic-1", 1)
        lives.note_day("synthetic-1", 400.0)
        lives.note_scoring("synthetic-1", 150.0)  # 550 s in all: kept
        _run(lives, tmp_path, "synthetic-1", 2)
        assert fake.commands().count("stop") == 1
        lives.note_day("synthetic-1", 400.0)
        lives.note_scoring("synthetic-1", 300.0)  # each is under the limit, the sum is not: stopped
        _run(lives, tmp_path, "synthetic-1", 3)
        assert fake.commands()[-1] == "stop" and fake.commands().count("stop") == 2
    records = lives.records("synthetic-1")
    assert [(r["day_s"], r["scoring_s"], r["vm_kept"]) for r in records[1:]] == [(400.0, 150.0, True), (400.0, 300.0, False)]
