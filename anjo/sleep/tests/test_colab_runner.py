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
from anjo.sleep.train.runners import colab_detached as detached
from anjo.sleep.train.runners.colab import (
    command_label,
    A100_CU_PER_HOUR,
    SEED_TIMEOUT_S,
    ColabCli,
    ColabError,
    ColabRunner,
    ColabTrainingTimeout,
    ColabVmLost,
    RealPersonRefused,
    hold_base,
    stop_leftover_sessions,
    stop_person_sessions,
)

TIMEOUT_S = 7200.0
NAME = adapter_name("p1", 1, 0)


ANSWERS = {colab.LAUNCH: f"{detached.MARK} launched 4242\n", colab.KILL: f"{detached.MARK} killed\n",
           colab.LOG_TAIL: f"{detached.MARK} log tail 2\nloss nan\nValueError: the loss is nan\n"}


def _literal(script: str, name: str) -> Any:
    """The value a generated script assigns to ``name``."""
    for node in ast.parse(script).body:
        if isinstance(node, ast.Assign) and [getattr(target, "id", None) for target in node.targets] == [name]:
            return ast.literal_eval(node.value)
    raise AssertionError(f"the script assigns no {name}")


class FakeColab:
    """Records every call; ``fail`` makes one subcommand exit 1 or time out. The detached training's
    polls read ``readings`` in turn, the last one repeating: ``running``, ``done <code>``, ``missing``,
    ``fail`` (the poll exits 1, as for a session the CLI no longer has) or ``hang`` (it times out). A
    keep-alive ping (the CLI's python, ``-I``) is recorded apart, in ``pings``."""

    def __init__(self, fail: str | None = None, *, timeout: bool = False, readings: Sequence[str] = ("done 0",),
                 answers: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.fail = fail
        self.timeout = timeout
        self.readings = list(readings)
        self.answers = {**ANSWERS, **(answers or {})}
        self.executed = ""  # the training entry the launcher starts
        self.keys: tuple[str, ...] = ()  # the environment names it passes on to the training
        self.scripts: dict[str, str] = {}  # each exec's script, by its command
        self.pings: list[list[str]] = []
        self.ping_code = 0

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        args = list(args)
        if args[1] == "-I":
            self.pings.append(args)
            return subprocess.CompletedProcess(args, self.ping_code, "", "")
        args[3] = command_label(args[3:])  # each exec of the runner's own scripts counts as its own
        self.calls.append((args, timeout_s))
        command = args[3]
        if "-f" in args:
            self.scripts[command] = Path(args[args.index("-f") + 1]).read_text(encoding="utf-8")
        if command == colab.LAUNCH:
            self.executed = _literal(self.scripts[command], "ENTRY_SOURCE")
            self.keys = _literal(self.scripts[command], "KEYS")
        if command == self.fail:
            if self.timeout:
                raise subprocess.TimeoutExpired(args, timeout_s)
            return subprocess.CompletedProcess(args, 1, "", "boom")
        if command == colab.POLL:
            return self._poll(args, timeout_s)
        if command in self.answers:
            return subprocess.CompletedProcess(args, 0, self.answers[command], "")
        if command == "download":
            local = Path(args[-1])
            if local.name == "receipt.json":
                local.write_text(json.dumps({"steps": 9, "epochs": 3, "train_minutes": 4.0, "examples": 3,
                                             "dropped_truncated": 0, "gpu": "A100-SXM4-80GB",
                                             "gpu_memory_mb": 81920}))
            else:
                local.write_bytes(b"{}")
        return subprocess.CompletedProcess(args, 0, "", "")

    def _poll(self, args: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        reading = self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]
        if reading == "hang":
            raise subprocess.TimeoutExpired(args, timeout_s)
        if reading == "fail":
            return subprocess.CompletedProcess(args, 1, "", "[colab] Session not found.\n")
        return subprocess.CompletedProcess(args, 0, f"{detached.MARK} {reading}\n", "")

    def commands(self) -> list[str]:
        return [args[3] for args, _ in self.calls]


class PollClock:
    """The poll loop's own clock: only its sleeps move it on."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


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


def _runner(fake: FakeColab, ledger: Ledger, *, allow_real: bool = False, polls: PollClock | None = None,
            keepalive: Sequence[str] | None = None) -> ColabRunner:
    polls = polls or PollClock()
    return ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake, keepalive=keepalive), ledger=ledger,
                       is_synthetic=lambda uid: uid.startswith("synthetic"), allow_real=allow_real, clock=Clock(),
                       poll_clock=polls, sleep=polls.sleep)


def _events(ledger: Ledger) -> list[dict[str, Any]]:
    return read_records(ledger.path, PLAIN)


@pytest.mark.parametrize(("fail", "timeout"), [(None, False), ("launch", False), ("launch", True), ("poll", False),
                                               ("poll", True), ("new", True), ("upload", False), ("download", False)])
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
    assert fake.executed == colab.SCRIPT.read_text(encoding="utf-8")
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
    assert fake.commands() == ["new", "install", "mkdir", "upload", "launch", "poll", "download", "download", "download",
                               "stop"]


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
    executed = next(args for args, _ in fake.calls if args[3] == colab.LAUNCH)
    env = _env(executed)
    assert fake.keys == tuple(env)  # the training gets this night's environment, and no other's
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
    fake, ledger, polls = FakeColab(readings=["running"]), Ledger(tmp_path / "budget.jsonl"), PollClock()
    with pytest.raises(ColabTrainingTimeout):  # run to its own deadline, which the poll clock then shows
        _runner(fake, ledger, polls=polls).run(JOB, _seeded_inputs(tmp_path, prev=prev), timeout_s=TIMEOUT_S)
    assert polls.now >= 6000.0  # the base weights' download and 4-bit load, then a 27-minute night
    assert polls.now == _train_s(SEED_TIMEOUT_S, uploads=4)


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


# ── A detached night (2026-11-02: one long exec dies when the CLI's proxy token expires, ~1 h in) ────

PING = ("/cli/bin/python", "-I", str(colab.PING_SCRIPT))
DONE = ["running", "running", "done 0"]


def _train_s(timeout_s: float, *, uploads: int) -> float:
    """The training's own time: the job's timeout less every call's, at the constants' values (new,
    install, mkdir, the uploads, the launch, a poll under way, its ping and the last poll, the
    downloads, the stop)."""
    return timeout_s - 600.0 - 900.0 - 120.0 - uploads * 300.0 - 180.0 - 2 * 90.0 - 30.0 - 3 * 300.0 - 180.0


def test_a_night_trains_detached_and_is_polled_until_it_is_done(tmp_path: Path) -> None:
    fake, ledger, polls = FakeColab(readings=DONE), Ledger(tmp_path / "budget.jsonl"), PollClock()
    out = _runner(fake, ledger, polls=polls, keepalive=PING).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert out["steps"] == 9
    assert fake.commands() == ["new", "install", "mkdir", "upload", "launch", "poll", "poll", "poll",
                               "download", "download", "download", "stop"]
    assert polls.slept == [colab.POLL_PERIOD_S] * 2 and colab.POLL_PERIOD_S == 90.0
    assert fake.pings == [[*PING, f"anjo-sleep-{NAME}"]] * 2  # one after each poll that found it running
    assert (out["polls"], out["polls_failed"], out["keepalive_pings"], out["keepalive_failed"]) == (3, 0, 2, 0)
    launch = next((args, timeout) for args, timeout in fake.calls if args[3] == colab.LAUNCH)
    poll = next((args, timeout) for args, timeout in fake.calls if args[3] == colab.POLL)
    assert launch[0][launch[0].index("--timeout") + 1] == str(colab.LAUNCH_EXEC_S) == "120.0"
    assert launch[1] == colab.LAUNCH_EXEC_S + colab.EXEC_MARGIN_S
    assert poll[0][poll[0].index("--timeout") + 1] == "30.0" and poll[1] == 90.0
    assert fake.executed == colab.SCRIPT.read_text(encoding="utf-8")
    assert fake.keys == tuple(_env(launch[0])) == ("ANJO_TRAIN_CONFIG", "ANJO_TRAIN_DATASET", "ANJO_TRAIN_OUT")
    assert fake.scripts[colab.LAUNCH] == detached.launch_script(colab.REMOTE_DIR, fake.executed, fake.keys)
    assert fake.scripts[colab.POLL] == detached.poll_script(colab.REMOTE_DIR)
    assert not list((tmp_path / "work").glob(".anjo-*"))  # no script stays in the night's folder
    assert ledger.open_holds() == ()


def test_three_polls_in_a_row_that_find_no_training_lose_the_vm(tmp_path: Path) -> None:
    fake, ledger = FakeColab(readings=["running", "fail", "hang", "missing"]), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabVmLost, match="3 polls in a row"):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    commands = fake.commands()
    assert commands.count("poll") == 4 and colab.LOST_POLLS == 3
    assert commands[-2:] == ["sessions", "stop"] and "download" not in commands  # the CLI drops a dead session
    assert ledger.open_holds() == () and 0 < ledger.used("colab_cu") < Bound(A100_CU_PER_HOUR, TIMEOUT_S).amount


@pytest.mark.parametrize("readings", [
    ["fail", "running", "done 0"],
    ["running", "hang", "fail", "running", "done 0"],
    ["missing", "fail", "done 0"],
    ["running", "fail", "fail", "running", "missing", "missing", "done 0"],
])
def test_fewer_than_three_misses_in_a_row_never_stop_the_training(tmp_path: Path, readings: list[str]) -> None:
    fake, ledger = FakeColab(readings=readings), Ledger(tmp_path / "budget.jsonl")
    out = _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert out["steps"] == 9 and out["polls"] == len(readings)
    assert out["polls_failed"] == sum(reading in ("fail", "hang", "missing") for reading in readings)
    assert fake.commands()[-4:] == ["download", "download", "download", "stop"]


def test_a_poll_that_prints_no_reading_is_a_miss(tmp_path: Path) -> None:
    fake, ledger = FakeColab(readings=["sleeping"]), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabVmLost):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.commands().count("poll") == colab.LOST_POLLS


def test_past_its_own_time_the_training_is_killed_and_the_night_times_out(tmp_path: Path) -> None:
    fake, ledger, polls = FakeColab(readings=["running"]), Ledger(tmp_path / "budget.jsonl"), PollClock()
    with pytest.raises(ColabTrainingTimeout, match="ran past its"):
        _runner(fake, ledger, polls=polls).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    commands = fake.commands()
    assert commands[-2:] == ["kill", "stop"] and "download" not in commands
    assert polls.now == _train_s(TIMEOUT_S, uploads=1) == 3810.0  # its last poll at the deadline, then the kill
    assert commands.count("poll") == math.ceil(polls.now / colab.POLL_PERIOD_S) + 1
    assert fake.scripts[colab.KILL] == detached.kill_script(colab.REMOTE_DIR)
    assert ledger.open_holds() == ()


def test_a_kill_that_fails_still_times_the_night_out(tmp_path: Path) -> None:
    fake, ledger = FakeColab("kill", readings=["running"]), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabTrainingTimeout):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.commands()[-2:] == ["kill", "stop"]


def test_a_training_that_exits_non_zero_fails_the_night_and_keeps_its_logs_last_lines(tmp_path: Path) -> None:
    fake, ledger = FakeColab(readings=["running", "done 1"]), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabError, match="the training on the VM exited 1") as raised:
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert not isinstance(raised.value, (ColabVmLost, ColabTrainingTimeout))
    commands = fake.commands()
    assert commands[-2:] == ["logtail", "stop"] and "download" not in commands
    tail = tmp_path / "work" / colab.LOG_TAIL_NAME
    assert "the loss is nan" in tail.read_text() and "the loss is nan" not in str(raised.value)
    assert str(tail) in str(raised.value) and ledger.open_holds() == ()


def test_a_log_tail_that_fails_leaves_the_trainings_own_failure(tmp_path: Path) -> None:
    fake, ledger = FakeColab("logtail", readings=["done 2"]), Ledger(tmp_path / "budget.jsonl")
    with pytest.raises(ColabError, match="exited 2; its log's last lines could not be brought back"):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert not (tmp_path / "work" / colab.LOG_TAIL_NAME).exists()


def test_a_launch_that_starts_no_training_fails_the_night_before_any_poll(tmp_path: Path) -> None:
    fake = FakeColab(answers={colab.LAUNCH: "Traceback (most recent call last):\n"})
    with pytest.raises(ColabError, match="started no training"):
        _runner(fake, Ledger(tmp_path / "budget.jsonl")).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert "poll" not in fake.commands() and fake.commands()[-1] == "stop"


FORGOTTEN = f"[colab] Session 'anjo-sleep-{NAME}' not found.\n"  # what ``colab stop`` says, exit 0, for a pruned one
UNNAMED = "[?] gpu-a100-s-9 | Hardware: A100 | GPU | HIGHMEM\n"  # an assignment this machine has no name for


@pytest.mark.parametrize(("listed", "whole"), [
    ("[colab] No active sessions found on server.\n", False),  # the VM is gone: what it ran
    (UNNAMED, True),  # an assignment the CLI can't name may be the VM, still running: the whole hold
    (None, True),  # the list failed, so it can't say: the whole hold
])
def test_a_stop_that_finds_no_record_of_the_session_looks_for_the_vm_the_cli_forgot(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, listed: str | None, whole: bool,
) -> None:
    """2026-11-02 (review): the CLI prunes a session on any 404 or 401 in an exec's error, and then
    ``colab stop`` exits 0 without unassigning a VM that may still run."""
    fake = FakeColab("sessions" if listed is None else None, readings=["running", "fail", "fail", "fail"],
                     answers={"stop": FORGOTTEN, **({} if listed is None else {"sessions": listed})})
    ledger = Ledger(tmp_path / "budget.jsonl")
    with caplog.at_level(logging.WARNING), pytest.raises(ColabVmLost):
        _runner(fake, ledger).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.commands()[-3:] == ["sessions", "stop", "sessions"] and ledger.open_holds() == ()
    hold = Bound(A100_CU_PER_HOUR, TIMEOUT_S).amount
    assert (ledger.used("colab_cu") == pytest.approx(hold)) is whole
    assert ("stop it by hand" in caplog.text) is whole


def test_a_stop_that_ends_the_session_lists_nothing_more(tmp_path: Path) -> None:
    fake = FakeColab(answers={"stop": "[colab] Stopping session...\n[colab] Session terminated.\n"})
    _runner(fake, Ledger(tmp_path / "budget.jsonl")).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert fake.commands()[-1] == "stop" and "sessions" not in fake.commands()


class Slowest(FakeColab):
    """Every call takes its whole hard timeout on one clock, which the polls' sleeps move on too; the
    training is done ``train_for`` seconds after its launch returns."""

    def __init__(self, train_for: float) -> None:
        super().__init__()
        self.now, self.train_for, self.launched = 0.0, train_for, math.inf

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        self.now += timeout_s
        if command_label(list(args)[3:]) == colab.LAUNCH:
            self.launched = self.now
        self.readings = ["done 0" if self.now - self.launched >= self.train_for else "running"]
        return super().__call__(args, timeout_s)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize("uploads", [1, 3])
def test_with_every_call_at_its_hard_timeout_the_night_still_ends_inside_its_hold(
    tmp_path: Path, uploads: int,
) -> None:
    """Done just before its deadline, just after it (read by the last poll), or never (killed): each
    call taking its whole hard timeout, the stop included, the night ends inside its hold."""
    warm = None
    if uploads == 3:
        warm = tmp_path / "prev"
        warm.mkdir()
        for file in config.ADAPTER_FILES:
            (warm / file).write_text("{}")
    train_s, outcomes = _train_s(TIMEOUT_S, uploads=uploads), {}
    for train_for in [*range(int(train_s) - 300, int(train_s) + 300, 7), math.inf]:
        fake, ledger = Slowest(train_for), Ledger(tmp_path / f"budget-{train_for}.jsonl")
        runner = ColabRunner(cli=ColabCli(Path("/fake/colab"), run=fake, keepalive=PING), ledger=ledger,
                             is_synthetic=lambda uid: uid.startswith("synthetic"), clock=fake.clock,
                             poll_clock=fake.clock, sleep=fake.sleep)
        try:
            runner.run(JOB, _inputs(tmp_path, prev=warm), timeout_s=TIMEOUT_S)
            outcomes[train_for] = "done"
        except ColabTrainingTimeout:
            outcomes[train_for] = "killed"
        assert fake.now <= TIMEOUT_S, train_for
        assert ledger.open_holds() == ()
    assert all(outcome == "done" for train_for, outcome in outcomes.items() if train_for <= train_s)
    assert outcomes[math.inf] == "killed" and "done" in {o for t, o in outcomes.items() if t > train_s}


def test_a_failed_ping_is_counted_and_never_fails_the_night(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fake, ledger = FakeColab(readings=DONE), Ledger(tmp_path / "budget.jsonl")
    fake.ping_code = 1
    with caplog.at_level(logging.WARNING):
        out = _runner(fake, ledger, keepalive=PING).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert out["steps"] == 9 and out["keepalive_pings"] == out["keepalive_failed"] == len(fake.pings) == 2
    assert "keep-alive" in caplog.text


def _entry(tmp_path: Path, shebang: str, *, python: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if python:
        (tmp_path / "bin").mkdir(exist_ok=True)
        (tmp_path / "bin" / "python").write_text("")
    entry = tmp_path / "colab"
    entry.write_text(f"{shebang}\n# -*- coding: utf-8 -*-\nimport sys\n")
    return entry


def _tool_python(tmp_path: Path) -> Path:
    tool = tmp_path / "tool" / "bin" / "python"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("")
    return tool


def test_the_ping_runs_on_the_python_named_by_the_clis_shebang(tmp_path: Path) -> None:
    python, none = tmp_path / "bin" / "python", tmp_path / "no-tool" / "python"
    assert colab.keepalive_command(_entry(tmp_path, f"#!{python}"), none) == (str(python), "-I", str(colab.PING_SCRIPT))
    for shebang in ("#!/bin/sh", "#!/usr/bin/env python3", f"#!{python} -E", "import sys"):
        assert colab.keepalive_command(_entry(tmp_path, shebang), none) is None
    assert colab.keepalive_command(_entry(tmp_path / "gone", f"#!{tmp_path}/gone/bin/python", python=False), none) is None
    assert colab.keepalive_command(tmp_path / "missing", none) is None
    assert colab.PING_SCRIPT.is_file() and colab.PING_SCRIPT.parent == Path(colab.__file__).parent


def test_a_launcher_in_front_of_the_cli_pings_on_the_cli_tools_own_python(tmp_path: Path) -> None:
    """2026-11-02: an operator put a shell wrapper at ``~/.local/bin/colab``; the watch would have been
    off for every night. The python of the CLI's own uv tool environment runs it then."""
    tool, python = _tool_python(tmp_path), tmp_path / "bin" / "python"
    command = (str(tool), "-I", str(colab.PING_SCRIPT))
    for shebang in ("#!/bin/sh", "#!/usr/bin/env python3"):
        assert colab.keepalive_command(_entry(tmp_path, shebang), tool) == command
    assert colab.keepalive_command(tmp_path / "missing", tool) == command
    first = colab.keepalive_command(_entry(tmp_path, f"#!{python}"), tool)
    assert first is not None and first[0] == str(python)  # the shebang's python comes first
    assert colab.COLAB_TOOL_PYTHON == Path.home() / ".local/share/uv/tools/google-colab-cli/bin/python"
    fake, ledger, polls = FakeColab(readings=DONE), Ledger(tmp_path / "budget.jsonl"), PollClock()
    runner = ColabRunner(cli=ColabCli(_entry(tmp_path, "#!/bin/sh"), run=fake, tool_python=tool), ledger=ledger,
                         is_synthetic=lambda uid: uid.startswith("synthetic"), clock=Clock(), poll_clock=polls,
                         sleep=polls.sleep)
    out = runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert len(fake.pings) == 2 and all(args[:3] == list(command) for args in fake.pings)
    assert out["keepalive_pings"] == len(fake.pings) and out["keepalive_failed"] == 0


def test_without_a_python_shebang_the_night_trains_without_pings_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    fake, ledger, polls = FakeColab(readings=DONE), Ledger(tmp_path / "budget.jsonl"), PollClock()
    runner = ColabRunner(cli=ColabCli(_entry(tmp_path, "#!/bin/sh"), run=fake, tool_python=tmp_path / "no-tool"),
                         ledger=ledger, is_synthetic=lambda uid: uid.startswith("synthetic"), clock=Clock(),
                         poll_clock=polls, sleep=polls.sleep)
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
