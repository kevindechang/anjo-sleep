# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The Colab runner (Task 5): the VM is always stopped, the worst case is held before it starts,
only the dataset goes up, every CLI call has a timeout, and real people are refused. The CLI is a
fake: no VM, no network."""
from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Bound, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.serving import config
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train.lora import DATASET_FILE, JOB, adapter_name
from anjo.sleep.train.runners import colab
from anjo.sleep.train.runners.colab import (
    A100_CU_PER_HOUR,
    ColabCli,
    ColabError,
    ColabRunner,
    RealPersonRefused,
)

TIMEOUT_S = 7200.0
NAME = adapter_name("p1", 1, 0)


class FakeColab:
    """Records every call; ``fail`` makes one subcommand exit 1 or time out."""

    def __init__(self, fail: str | None = None, *, timeout: bool = False) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.fail = fail
        self.timeout = timeout

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        args = list(args)
        self.calls.append((args, timeout_s))
        command = args[3]
        if command == self.fail:
            if self.timeout:
                raise subprocess.TimeoutExpired(args, timeout_s)
            return subprocess.CompletedProcess(args, 1, "", "boom")
        if command == "download":
            local = Path(args[-1])
            if local.name == "receipt.json":
                local.write_text(json.dumps({"steps": 9, "epochs": 3, "train_minutes": 4.0, "examples": 3,
                                             "dropped_truncated": 0, "gpu": "A100-SXM4-80GB"}))
            else:
                local.write_bytes(b"{}")
        return subprocess.CompletedProcess(args, 0, "", "")

    def commands(self) -> list[str]:
        return [args[3] for args, _ in self.calls]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 900.0  # every reading is 15 minutes on
        return self.now


def _inputs(tmp_path: Path, person: str = "synthetic-1", prev: Path | None = None) -> dict[str, Any]:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / DATASET_FILE).write_text('{"id": "x"}\n')
    return {"person_id": person, "adapter_name": NAME, "dataset_path": str(work / DATASET_FILE),
            "workdir": str(work), "config": {"r": 16}, "prev_adapter_dir": None if prev is None else str(prev)}


def _runner(fake: FakeColab, ledger: Ledger, *, allow_real: bool = False) -> ColabRunner:
    return ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake), ledger=ledger,
                       is_synthetic=lambda uid: uid.startswith("synthetic"), allow_real=allow_real, clock=Clock())


def _events(ledger: Ledger) -> list[dict[str, Any]]:
    return read_records(ledger.path, PLAIN)


@pytest.mark.parametrize(("fail", "timeout"), [(None, False), ("exec", False), ("exec", True),
                                               ("new", True), ("upload", False), ("download", False)])
def test_colab_runner_always_stops_vm(tmp_path: Path, fail: str | None, timeout: bool) -> None:
    fake, ledger = FakeColab(fail, timeout=timeout), Ledger(tmp_path / "budget.jsonl")
    runner = _runner(fake, ledger)
    if fail is None:
        outputs = runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
        assert outputs["steps"] == 9 and outputs["gpu_minutes"] == pytest.approx(15.0)
    else:
        with pytest.raises((ColabError, subprocess.TimeoutExpired)):
            runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.commands()[-1] == "stop"
    session = fake.calls[-1][0][-1]
    assert session == f"anjo-sleep-{NAME}"
    assert [e["op"] for e in _events(ledger)] == ["reserve", "settle"]  # settled, never left open


def test_a_failed_stop_settles_the_whole_hold_and_says_so(tmp_path: Path) -> None:
    fake, ledger = FakeColab("stop"), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabError, match="stop it by hand"):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert ledger.used("colab_cu") == pytest.approx(Bound(A100_CU_PER_HOUR, TIMEOUT_S).amount)


def test_colab_runner_reserves_worst_case_bound(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    reserve, settle = _events(ledger)
    assert reserve["pools"] == ["colab_cu"]
    assert reserve["rate_per_hour"] == A100_CU_PER_HOUR == 5.3
    assert reserve["timeout_s"] == TIMEOUT_S
    assert reserve["amount"] == pytest.approx(TIMEOUT_S / 3600 * 5.3)
    assert settle["amount"] == pytest.approx(0.25 * 5.3)  # what the VM ran for
    # Every call's timeout together fits inside the job's own, so the hold is the worst case.
    assert sum(timeout for _, timeout in fake.calls) <= TIMEOUT_S


def test_no_room_in_the_pool_stops_before_any_vm(tmp_path: Path) -> None:
    fake = FakeColab()
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "colab_cu": 1.0})
    with pytest.raises(BudgetStop):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.calls == []


def test_colab_runner_refuses_real_person_without_allow(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(RealPersonRefused):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path, person="kevin"), timeout_s=TIMEOUT_S)
    assert fake.calls == [] and _events(ledger) == []
    _runner(fake, ledger, allow_real=True).run(JOB, _inputs(tmp_path, person="kevin"), timeout_s=TIMEOUT_S)
    assert fake.commands()[-1] == "stop"


def test_only_the_dataset_and_a_warm_starts_adapter_go_up(tmp_path: Path) -> None:
    prev = tmp_path / "prev"
    prev.mkdir()
    for file in config.ADAPTER_FILES:
        (prev / file).write_text("{}")
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    _runner(fake, ledger).run(JOB, _inputs(tmp_path, prev=prev), timeout_s=TIMEOUT_S)
    uploaded = [Path(args[-2]) for args, _ in fake.calls if args[3] == "upload"]
    assert uploaded == [tmp_path / "work" / DATASET_FILE, *(prev / f for f in config.ADAPTER_FILES)]
    executed = next(args for args, _ in fake.calls if args[3] == "exec")
    assert Path(executed[executed.index("-f") + 1]) == colab.SCRIPT
    assert not any(".env" in part for args, _ in fake.calls for part in args)
    with pytest.raises(ColabError):
        bad = _inputs(tmp_path)
        bad["dataset_path"] = str(tmp_path / ".env")
        _runner(FakeColab(), Ledger(tmp_path / "other.jsonl")).run(JOB, bad, timeout_s=TIMEOUT_S)


def test_cli_call_has_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen.append({"args": args, **kw})
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ColabCli(Path("/fake/colab")).call("stop", "-s", "x", timeout_s=30.0)
    assert seen[0]["timeout"] == 30.0 and "shell" not in seen[0]
    assert seen[0]["args"][1:4] == ["--auth", "oauth2", "stop"]
    for bad in (0.0, math.inf, -1.0):
        with pytest.raises(ValueError):
            ColabCli(Path("/fake/colab")).call("stop", timeout_s=bad)


def test_every_runner_call_passes_a_finite_timeout(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert all(0 < timeout < math.inf for _, timeout in fake.calls)
    assert fake.commands() == ["new", "install", "upload", "exec", "download", "download", "download", "stop"]
