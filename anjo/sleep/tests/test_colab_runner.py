# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The Colab runner (Task 5): the VM is always stopped, the worst case is held before it starts,
only the dataset goes up, every CLI call has a timeout, and real people are refused. The CLI is a
fake: no VM, no network."""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import math
import subprocess
import sys
import tarfile
import threading
import time
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Bound, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.serving import config
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train import modal_job
from anjo.sleep.train.lora import DATASET_FILE, JOB, adapter_name
from anjo.sleep.train.runners import colab
from anjo.sleep.train.runners.colab import (
    command_label,
    A100_CU_PER_HOUR,
    SEED_TIMEOUT_S,
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
        self.executed = ""

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        args = list(args)
        args[3] = command_label(args[3:])  # the remote folders' exec counts as mkdir
        self.calls.append((args, timeout_s))
        command = args[3]
        if command == "exec":
            self.executed = Path(args[args.index("-f") + 1]).read_text(encoding="utf-8")
        if command == self.fail:
            if self.timeout:
                raise subprocess.TimeoutExpired(args, timeout_s)
            return subprocess.CompletedProcess(args, 1, "", "boom")
        if command == "download":
            local = Path(args[-1])
            if local.name == "receipt.json":
                local.write_text(json.dumps({"steps": 9, "epochs": 3, "train_minutes": 4.0, "examples": 3,
                                             "dropped_truncated": 0, "gpu": "A100-SXM4-80GB",
                                             "gpu_memory_mb": 81920}))
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
    assert reserve["rate_per_hour"] == A100_CU_PER_HOUR == 8.88  # `colab usage`, 2026-11-02
    assert reserve["timeout_s"] == TIMEOUT_S
    assert reserve["amount"] == pytest.approx(TIMEOUT_S / 3600 * 8.88)
    assert settle["amount"] == pytest.approx(0.25 * 8.88)  # what the VM ran for
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
    assert fake.commands() == ["new", "install", "mkdir", "upload", "exec", "download", "download", "download", "stop"]


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


def _seeded_inputs(tmp_path: Path, *, prev: Path | None = None, objective: str = "sft_dpo_seed") -> dict[str, Any]:
    inputs = _inputs(tmp_path, prev=prev)
    inputs["config"] = {"r": 16, "objective": objective}
    return inputs


def _env(executed: list[str]) -> dict[str, str]:
    pairs = [executed[i + 1] for i, part in enumerate(executed) if part == "--env"]
    return {key: value for key, value in (pair.split("=", 1) for pair in pairs)}


@pytest.mark.parametrize("warm", [False, True])
def test_a_seeded_night_ships_the_shared_files_and_runs_modal_entry(tmp_path: Path, warm: bool) -> None:
    prev = None
    if warm:
        prev = tmp_path / "prev"
        prev.mkdir()
        for file in config.ADAPTER_FILES:
            (prev / file).write_text("{}")
    inputs = _seeded_inputs(tmp_path, prev=prev)
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    outputs = _runner(fake, ledger).run(JOB, inputs, timeout_s=TIMEOUT_S)

    archive = tmp_path / "work" / colab.SEED_ARCHIVE
    uploads = [(Path(args[-2]), args[-1]) for args, _ in fake.calls if args[3] == "upload"]
    assert uploads == [
        (tmp_path / "work" / DATASET_FILE, f"{colab.REMOTE_DIR}/{DATASET_FILE}"),
        (archive, f"{colab.REMOTE_DIR}/{colab.SEED_ARCHIVE}"),
        *([] if prev is None else [(prev / file, f"{colab.REMOTE_DIR}/prev/{file}") for file in config.ADAPTER_FILES]),
    ]
    with tarfile.open(archive) as tar:
        members = tar.getnames()
    assert sorted(members) == sorted(remote.lstrip("/") for _, remote in modal_job.TRAIN_FILES)
    assert fake.executed == colab.SEED_ENTRY and modal_job.REMOTE_SCRIPT in fake.executed
    assert colab.SEED_ARCHIVE_ENTRY in fake.executed  # unpacked before the entry runs
    executed = next(args for args, _ in fake.calls if args[3] == "exec")
    env = _env(executed)
    assert env == {
        "ANJO_TRAIN_CONFIG": json.dumps(inputs["config"], sort_keys=True),
        "ANJO_TRAIN_DATASET": f"{colab.REMOTE_DIR}/{DATASET_FILE}",
        "ANJO_TRAIN_OUT": f"{colab.REMOTE_DIR}/out",
        **({"ANJO_TRAIN_PREV": f"{colab.REMOTE_DIR}/prev"} if warm else {}),
    }
    reserve, _ = _events(ledger)
    assert reserve["timeout_s"] == SEED_TIMEOUT_S
    assert sum(timeout for _, timeout in fake.calls) <= SEED_TIMEOUT_S
    assert outputs["gpu"] == "A100-SXM4-80GB" and outputs["gpu_memory_mb"] == 81920


def test_a_warm_seed_night_leaves_at_least_100_minutes_to_train(tmp_path: Path) -> None:
    prev = tmp_path / "prev"
    prev.mkdir()
    for file in config.ADAPTER_FILES:
        (prev / file).write_text("{}")
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    _runner(fake, ledger).run(JOB, _seeded_inputs(tmp_path, prev=prev), timeout_s=TIMEOUT_S)
    executed = next(args for args, _ in fake.calls if args[3] == "exec")
    exec_s = float(executed[executed.index("--timeout") + 1])
    assert exec_s >= 6000.0  # the base weights' download and 4-bit load, then a 27-minute night
    # 120: mkdir; 30: a keep-alive ping still running when the exec ends
    assert exec_s == SEED_TIMEOUT_S - 600.0 - 900.0 - 120.0 - 4 * 300.0 - 3 * 300.0 - 180.0 - 60.0 - 30.0


def test_the_seed_archive_refuses_a_file_outside_the_shared_list(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("raise SystemExit\n")
    with pytest.raises(ColabError, match="shared training files"):
        colab._seed_archive(work, [*modal_job.TRAIN_FILES, (outside, f"{colab.REMOTE_DIR}/outside.py")])
    assert not (work / colab.SEED_ARCHIVE).exists()


@pytest.mark.parametrize("objective", ["sft", "sft_dpo"])
def test_the_other_objectives_upload_and_exec_exactly_as_before(tmp_path: Path, objective: str) -> None:
    inputs = _seeded_inputs(tmp_path, objective=objective)
    fake, ledger = FakeColab(), Ledger(tmp_path / "budget.jsonl")
    _runner(fake, ledger).run(JOB, inputs, timeout_s=TIMEOUT_S)
    uploads = [Path(args[-2]) for args, _ in fake.calls if args[3] == "upload"]
    assert uploads == [tmp_path / "work" / DATASET_FILE]
    assert fake.executed == colab.SCRIPT.read_text(encoding="utf-8")
    reserve, _ = _events(ledger)
    assert reserve["timeout_s"] == TIMEOUT_S  # two hours, as today


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
    holds = [str(e["job"]) for e in _events(ledger) if e["op"] == "reserve"]
    assert [job.split(":")[0] for job in holds] == ["mind_train", "mind_life"]  # MIND's holds, named apart from Sleep's
    with pytest.raises(ValueError, match="prefix"):
        ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake), ledger=ledger, is_synthetic=bool,
                    session_prefix="voice-")


def test_a_new_vm_makes_the_remote_folders_before_any_upload(tmp_path: Path) -> None:
    # 2026-11-02: the first real Colab night failed at its first upload: a new VM has no /content/anjo,
    # and an upload into a missing folder fails (HTTP 500 from the VM's contents API).
    seen: list[str] = []

    class Reads(FakeColab):
        def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
            if command_label(list(args)[3:]) == colab.MKDIR:
                seen.append(Path(args[args.index("-f") + 1]).read_text(encoding="utf-8"))
            return super().__call__(args, timeout_s)

    fake, ledger = Reads(), Ledger(tmp_path / "budget.jsonl")
    _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    commands = fake.commands()
    assert commands.index("install") < commands.index("mkdir") < commands.index("upload")
    assert seen == [colab.MKDIR_SCRIPT]
    local = tmp_path / "vm" / "content" / "anjo"  # the same script, its folders moved under tmp_path
    exec(compile(colab.MKDIR_SCRIPT.replace(colab.REMOTE_DIR, str(local)), "mkdir", "exec"), {})
    assert local.is_dir() and (local / "prev").is_dir()
    assert all(remote.startswith(f"{colab.REMOTE_DIR}/") for remote in
               (args[-1] for args, _ in fake.calls if args[3] == "upload"))
    assert not (tmp_path / colab.MKDIR_FILE).exists()  # the script never stays in the night's folder


# ── The keep-alive (2026-11-02: an A100 vanished about 35 minutes into a training exec) ──────────────

PING = ("/cli/bin/python", "-I", str(colab.PING_SCRIPT))
PERIOD_S = 0.02


class SlowExec(FakeColab):
    """An ``exec`` that takes ``exec_s`` of real time (and exits ``exec_code``); a ping (the CLI's python
    running the ping script) exits ``ping_code``. Pings are recorded apart, with whether the exec ran."""

    def __init__(self, *, exec_s: float = 0.3, exec_code: int = 0, ping_code: int = 0) -> None:
        super().__init__()
        self.exec_s, self.exec_code, self.ping_code = exec_s, exec_code, ping_code
        self.pings: list[tuple[list[str], float, bool]] = []
        self.in_exec = False

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        args = list(args)
        if args[1] == "-I":
            self.pings.append((args, timeout_s, self.in_exec))
            return subprocess.CompletedProcess(args, self.ping_code, "", "")
        if command_label(args[3:]) != "exec":
            return super().__call__(args, timeout_s)
        self.in_exec = True
        try:
            time.sleep(self.exec_s)
        finally:
            self.in_exec = False
        if self.exec_code:
            self.calls.append((args, timeout_s))
            return subprocess.CompletedProcess(args, self.exec_code, "", "boom")
        return super().__call__(args, timeout_s)


def _pinging(fake: FakeColab, ledger: Ledger, period_s: float = PERIOD_S) -> ColabRunner:
    return ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake, keepalive=PING), ledger=ledger,
                       is_synthetic=lambda uid: uid.startswith("synthetic"), clock=Clock(), keepalive_period_s=period_s)


def _keepalive_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == colab.KEEPALIVE_THREAD]


def test_a_training_exec_pings_its_sessions_keepalive_with_the_clis_python(tmp_path: Path) -> None:
    fake, ledger = SlowExec(), Ledger(tmp_path / "budget.jsonl")
    out = _pinging(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert len(fake.pings) >= 3  # 0.3 s of exec at a 0.02 s period
    assert all(args == [*PING, f"anjo-sleep-{NAME}"] for args, _, _ in fake.pings)
    assert all(timeout == colab.KEEPALIVE_TIMEOUT_S for _, timeout, _ in fake.pings)
    assert all(during for _, _, during in fake.pings)  # only while the training exec runs
    assert (out["keepalive_pings"], out["keepalive_failed"]) == (len(fake.pings), 0)
    assert fake.commands() == ["new", "install", "mkdir", "upload", "exec", "download", "download", "download", "stop"]
    pinged = len(fake.pings)
    time.sleep(10 * PERIOD_S)
    assert len(fake.pings) == pinged and _keepalive_threads() == []  # stopped when the exec returned


def test_the_keepalive_stops_when_the_exec_fails(tmp_path: Path) -> None:
    fake, ledger = SlowExec(exec_code=1), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabError, match="colab exec failed"):
        _pinging(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.pings and all(during for _, _, during in fake.pings)
    pinged = len(fake.pings)
    time.sleep(10 * PERIOD_S)
    assert len(fake.pings) == pinged and _keepalive_threads() == []
    assert fake.commands()[-1] == "stop" and [e["op"] for e in _events(ledger)] == ["reserve", "settle"]


def test_a_failed_ping_is_counted_and_never_fails_the_night(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fake, ledger = SlowExec(ping_code=1), Ledger(tmp_path / "budget.jsonl")
    with caplog.at_level(logging.WARNING):
        out = _pinging(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert out["steps"] == 9 and out["keepalive_pings"] == out["keepalive_failed"] == len(fake.pings) > 0
    assert "keep-alive" in caplog.text


def test_no_ping_runs_before_one_period_into_the_exec(tmp_path: Path) -> None:
    fake, ledger = SlowExec(exec_s=0.05), Ledger(tmp_path / "budget.jsonl")
    out = _pinging(fake, ledger, period_s=colab.KEEPALIVE_PERIOD_S).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.pings == [] and (out["keepalive_pings"], out["keepalive_failed"]) == (0, 0)
    assert colab.KEEPALIVE_PERIOD_S == 60.0 and colab.KEEPALIVE_TIMEOUT_S < colab.KEEPALIVE_PERIOD_S


def _entry(tmp_path: Path, shebang: str, *, python: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if python:
        (tmp_path / "bin").mkdir(exist_ok=True)
        (tmp_path / "bin" / "python").write_text("")
    entry = tmp_path / "colab"
    entry.write_text(f"{shebang}\n# -*- coding: utf-8 -*-\nimport sys\n")
    return entry


def test_the_ping_runs_on_the_python_named_by_the_clis_shebang(tmp_path: Path) -> None:
    python = tmp_path / "bin" / "python"
    assert colab.keepalive_command(_entry(tmp_path, f"#!{python}")) == (str(python), "-I", str(colab.PING_SCRIPT))
    for shebang in ("#!/bin/sh", "#!/usr/bin/env python3", f"#!{python} -E", "import sys"):
        assert colab.keepalive_command(_entry(tmp_path, shebang)) is None
    assert colab.keepalive_command(_entry(tmp_path / "gone", f"#!{tmp_path}/gone/bin/python", python=False)) is None
    assert colab.keepalive_command(tmp_path / "missing") is None
    assert colab.PING_SCRIPT.is_file() and colab.PING_SCRIPT.parent == Path(colab.__file__).parent


def test_without_a_python_shebang_the_night_trains_without_pings_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    fake, ledger = SlowExec(), Ledger(tmp_path / "budget.jsonl")
    runner = ColabRunner(cli=ColabCli(_entry(tmp_path, "#!/bin/sh"), run=fake), ledger=ledger,
                         is_synthetic=lambda uid: uid.startswith("synthetic"), clock=Clock(),
                         keepalive_period_s=PERIOD_S)
    with caplog.at_level(logging.WARNING):
        out = runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert out["steps"] == 9 and fake.pings == []
    assert (out["keepalive_pings"], out["keepalive_failed"]) == (1, 1)  # one try, then no more
    assert "without keep-alive" in caplog.text


def test_the_ping_script_imports_only_colab_cli_and_the_standard_library() -> None:
    tree = ast.parse(colab.PING_SCRIPT.read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "colab_cli" in imported and imported - {"colab_cli"} <= set(sys.stdlib_module_names)


class _ReadTimeout(Exception):
    pass


def _ping_module(monkeypatch: pytest.MonkeyPatch, sessions: dict[str, str], issue: Any) -> types.ModuleType:
    """The ping script, loaded against a fake ``colab_cli``: no OAuth, no network."""
    client = types.ModuleType("colab_cli.client")
    client.TUN_ENDPOINT = "/tun/m"  # type: ignore[attr-defined]
    client.requests = types.SimpleNamespace(exceptions=types.SimpleNamespace(ReadTimeout=_ReadTimeout))  # type: ignore[attr-defined]

    class State:
        def __init__(self) -> None:
            self.store = types.SimpleNamespace(
                get=lambda name: None if name not in sessions else types.SimpleNamespace(endpoint=sessions[name]))
            self.client = types.SimpleNamespace(colab_domain="https://colab.example", _issue_request=issue)

    common = types.ModuleType("colab_cli.common")
    common.State = State  # type: ignore[attr-defined]
    package = types.ModuleType("colab_cli")
    for name, module in (("colab_cli", package), ("colab_cli.client", client), ("colab_cli.common", common)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("colab_ping_under_test", colab.PING_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ping_script_gets_the_sessions_keepalive_and_a_read_timeout_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def issue(url: str, **kw: Any) -> None:
        seen.append((url, kw))

    ping = _ping_module(monkeypatch, {"anjo-sleep-x": "gpu-a100-s-1"}, issue)
    assert ping.main(["anjo-sleep-x"]) == 0
    url, kw = seen[0]
    assert url == "https://colab.example/tun/m/gpu-a100-s-1/keep-alive/"
    assert kw["headers"] == {"X-Colab-Tunnel": "Google"} and 0 < kw["timeout"] < colab.KEEPALIVE_TIMEOUT_S

    def times_out(url: str, **kw: Any) -> None:
        raise _ReadTimeout("read timed out")

    assert _ping_module(monkeypatch, {"anjo-sleep-x": "e"}, times_out).main(["anjo-sleep-x"]) == 0

    def refused(url: str, **kw: Any) -> None:
        raise ConnectionError("refused")

    with pytest.raises(ConnectionError):  # any other failure: uncaught, so the process exits 1
        _ping_module(monkeypatch, {"anjo-sleep-x": "e"}, refused).main(["anjo-sleep-x"])
    unknown = _ping_module(monkeypatch, {}, issue)
    assert unknown.main(["anjo-sleep-y"]) not in (0, 1) and unknown.main([]) not in (0, 1)
