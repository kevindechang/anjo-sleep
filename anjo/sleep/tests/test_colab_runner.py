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
    hold_base,
    stop_leftover_sessions,
    stop_person_sessions,
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
    return {"person_id": person, "adapter_name": NAME, "night": "2026-10-28", "dataset_path": str(work / DATASET_FILE),
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
    assert reserve["rate_per_hour"] == A100_CU_PER_HOUR == 7.5  # spec §11.5, MIND's compute line
    assert reserve["timeout_s"] == TIMEOUT_S
    assert reserve["amount"] == pytest.approx(TIMEOUT_S / 3600 * 7.5)
    assert settle["amount"] == pytest.approx(0.25 * 7.5)  # what the VM ran for
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


class Both(FakeColab):
    """``new`` and then ``stop`` both fail: ``new`` exits 1, or times out."""

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        command = list(args)[3]
        if command == "stop":
            self.calls.append((list(args), timeout_s))
            return subprocess.CompletedProcess(list(args), 1, "", "no such session")
        return super().__call__(args, timeout_s)


def test_a_failed_new_settles_what_ran_and_raises_its_own_error(tmp_path: Path) -> None:
    fake, ledger = Both("new"), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabError) as raised:
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert "stop it by hand" not in str(raised.value) and "new" in str(raised.value)
    assert fake.commands() == ["new", "stop"]  # the stop is still tried
    assert ledger.used("colab_cu") == pytest.approx(0.25 * A100_CU_PER_HOUR)  # 15 minutes, not the hold


def test_a_new_that_timed_out_tries_the_stop_and_settles_the_whole_hold(tmp_path: Path) -> None:
    fake, ledger = FakeColab("new", timeout=True), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(subprocess.TimeoutExpired):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.commands() == ["new", "stop"]
    assert ledger.used("colab_cu") == pytest.approx(Bound(A100_CU_PER_HOUR, TIMEOUT_S).amount)  # a VM may have run


SESSIONS = (
    "[anjo-sleep-p-aaaaaaaaaaaa-n3] gpu-a100-s-1 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
    "[?] gpu-t4-s-2 | Hardware: T4 | Shape: Standard | Variant: GPU\n"
    "[voice-test] gpu-a100-s-3 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
)


class Listing(FakeColab):
    """``colab sessions`` prints ``listed``; ``stop`` may fail."""

    def __init__(self, listed: str, fail: str | None = None) -> None:
        super().__init__(fail)
        self.listed = listed

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        done = super().__call__(args, timeout_s)
        if list(args)[3] == "sessions" and done.returncode == 0:
            return subprocess.CompletedProcess(list(args), 0, self.listed, "")
        return done


def test_leftover_sessions_the_runner_made_are_stopped_and_no_other() -> None:
    fake = Listing(SESSIONS)
    assert stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=fake)) == (1, 2)
    assert [args[3:] for args, _ in fake.calls] == [["sessions"], ["stop", "-s", "anjo-sleep-p-aaaaaaaaaaaa-n3"]]
    assert all(0 < timeout < math.inf for _, timeout in fake.calls)
    none = Listing("[colab] No active sessions found on server.\n")
    assert stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=none)) == (0, 0)


def test_a_leftover_that_wont_stop_says_so() -> None:
    with pytest.raises(ColabError, match="stop it by hand"):
        stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=Listing(SESSIONS, fail="stop")))
    with pytest.raises(ColabError):  # an unlisted VM is unknown: the night doesn't start
        stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=Listing("", fail="sessions")))


def test_the_colab_hold_is_keyed_by_night_and_person_with_an_attempt(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    for _ in range(2):
        _runner(FakeColab(), ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    jobs = [e["job"] for e in _events(ledger) if e["op"] == "reserve"]
    assert jobs == [f"{hold_base('2026-10-28', 'synthetic-1')}:a1", f"{hold_base('2026-10-28', 'synthetic-1')}:a2"]
    assert hold_base("2026-10-28", "synthetic-1").startswith("colab_train:2026-10-28:")


@pytest.mark.parametrize("listed", [
    "NAME  ENDPOINT  HARDWARE\nanjo-sleep-p-aaaaaaaaaaaa-n3  gpu-a100-s-1  A100\n",  # a new table format
    '[{"name": "anjo-sleep-p-aaaaaaaaaaaa-n3"}]\n',  # a new JSON format
    "",  # nothing at all
])
def test_a_session_list_the_sweep_cant_read_fails_loud(listed: str) -> None:
    fake = Listing(listed)
    with pytest.raises(ColabError, match="by hand"):
        stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=fake))
    assert [args[3] for args, _ in fake.calls] == ["sessions"]


def test_a_cli_notice_beside_the_list_is_read_past() -> None:
    notice = "[colab] Run 'uv tool upgrade google-colab-cli' to update.\n"
    assert stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=Listing(
        notice + "[colab] No active sessions found on server.\n"))) == (0, 0)
    assert stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=Listing(notice + SESSIONS))) == (1, 2)


def test_a_delete_stops_only_that_persons_sessions() -> None:
    from anjo.sleep.train.runners.colab import _life_session

    listing = (
        "[anjo-sleep-p-aaaaaaaaaaaa-n3] gpu-a100-s-1 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
        "[anjo-sleep-p-bbbbbbbbbbbb-n1] gpu-a100-s-2 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
        f"[{_life_session('p1')}] gpu-a100-s-3 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
        "[voice-test] gpu-a100-s-4 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
    )
    fake = Listing(listing)
    stopped = stop_person_sessions(ColabCli(Path("/fake/colab"), run=fake), "p1", {"p-aaaaaaaaaaaa-n3"})
    assert stopped == 2
    assert [args[3:] for args, _ in fake.calls][1:] == [
        ["stop", "-s", "anjo-sleep-p-aaaaaaaaaaaa-n3"], ["stop", "-s", _life_session("p1")]]
    none = Listing("[colab] No active sessions found on server.\n")
    assert stop_person_sessions(ColabCli(Path("/fake/colab"), run=none), "p1", set()) == 0


MIXED = (
    "[anjo-sleep-p-aaaaaaaaaaaa-n3] gpu-a100-s-1 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
    "[anjo-mind-apd-1a2b3c4d] gpu-a100-s-2 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
    "[anjo-mind-p-bbbbbbbbbbbb-n0] gpu-a100-s-3 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
    "[voice-test] gpu-a100-s-4 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
)


def test_sleeps_sweep_and_minds_sweep_leave_each_others_sessions_alone() -> None:
    sleep = Listing(MIXED)
    assert stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=sleep)) == (1, 3)
    assert [args[3:] for args, _ in sleep.calls][1:] == [["stop", "-s", "anjo-sleep-p-aaaaaaaaaaaa-n3"]]
    mind = Listing(MIXED)
    assert stop_leftover_sessions(ColabCli(Path("/fake/colab"), run=mind), prefix=colab.MIND_SESSION_PREFIX) == (2, 2)
    assert [args[3:] for args, _ in mind.calls][1:] == [
        ["stop", "-s", "anjo-mind-apd-1a2b3c4d"], ["stop", "-s", "anjo-mind-p-bbbbbbbbbbbb-n0"]]
    assert not colab.MIND_SESSION_PREFIX.startswith(colab.SESSION_PREFIX)
    assert not colab.SESSION_PREFIX.startswith(colab.MIND_SESSION_PREFIX)


def test_a_runner_names_its_sessions_with_its_own_prefix(tmp_path: Path) -> None:
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    runner = ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake), ledger=ledger,
                         is_synthetic=lambda uid: uid.startswith("synthetic"), clock=Clock(),
                         session_prefix=colab.MIND_SESSION_PREFIX)
    runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    sessions = {args[args.index("-s") + 1] for args, _ in fake.calls if "-s" in args}
    assert sessions == {f"anjo-mind-{NAME}"}
    with runner.lives(life_timeout_s=3 * TIMEOUT_S) as lives:
        lives.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert all(args[args.index("-s") + 1].startswith("anjo-mind-") for args, _ in fake.calls if "-s" in args)
    with pytest.raises(ValueError, match="prefix"):
        ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake), ledger=ledger, is_synthetic=bool,
                    session_prefix="voice-")
