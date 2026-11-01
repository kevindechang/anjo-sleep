# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""``ANJO_SLEEP_TRAIN_RUNNER=colab_modal``: Colab trains, and Modal trains the same job when Colab's
infrastructure fails before the training script starts (no A100, install, upload) or times out. A
training-script failure, a refusal, a budget stop, a VM that won't stop and ``Cancelled`` fail the
night as ``colab`` does. Both pools' holds are settled, the receipt names the runner that trained,
and the ``modal`` and ``colab`` settings are as they were. The CLI and Modal are fakes: no VM, no
container, no network."""
from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Bound, Ledger
from anjo.sleep.night.sweep import Stopped
from anjo.sleep.serving import config, framing
from anjo.sleep.stops import BudgetStop, Cancelled
from anjo.sleep.tests.test_colab_life import _night
from anjo.sleep.tests.test_colab_runner import FakeColab
from anjo.sleep.tests.test_lora import REVISION, _examples
from anjo.sleep.tests.test_modal_runner import NAME, RENDERING, Clock, FakeCalls, _inputs
from anjo.sleep.tests.test_train_runner_choice import NoColab, _EmptyColab, _FlakyColab
from anjo.sleep.train.lora import JOB, TrainReceipt, adapter_name
from anjo.sleep.train.runners import choice
from anjo.sleep.train.runners.colab import (
    A100_CU_PER_HOUR,
    ColabCli,
    ColabError,
    ColabRunner,
    RealPersonRefused,
    command_label,
)
from anjo.sleep.train.runners.fallback import ColabModalLives, ColabModalRunner
from anjo.sleep.train.runners.modal import A100_USD_PER_HOUR, CALLS_DIR, CallRecords, ModalRunner
from anjo.sleep.train.trainer import OPTIONAL_KEYS, LoraTrainer

TIMEOUT_S = 7200.0
NIGHT = date(2026, 10, 28)
MINUTE_S = 60.0
# The kernel's own --timeout for a cold night with only the dataset to upload (colab._setup_s(1)).
KERNEL_S = TIMEOUT_S - (600.0 + 900.0 + 60.0 + 60.0 + 300.0 + 3 * 300.0 + 180.0 + 60.0)  # 60 + 60: mkdir


def _synthetic(uid: str) -> bool:
    return uid.startswith("synthetic")


class TimedColab(FakeColab):
    """``FakeColab`` on a clock that moves only as its calls take it: an ``exec`` ``exec_s``, a call that
    hangs its whole timeout, any other a minute. ``error`` is what ``fail`` raises instead of exiting 1,
    ``stops`` False makes every stop exit 1, and the adapter it downloads records the night's rendering."""

    def __init__(self, fail: str | None = None, *, timeout: bool = False, exec_s: float = 600.0,
                 error: BaseException | None = None, stops: bool = True) -> None:
        super().__init__(fail, timeout=timeout)
        self.clock = Clock()
        self.exec_s, self.error, self.stops = exec_s, error, stops

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        args = list(args)
        args[3] = command_label(args[3:])  # the remote folders' exec counts as mkdir
        command = args[3]
        hangs = command == self.fail and self.timeout
        self.clock.now += timeout_s if hangs else self.exec_s if command == "exec" else MINUTE_S
        if command == self.fail and self.error is not None:
            self.calls.append((args, timeout_s))
            raise self.error
        if command == "stop" and not self.stops:
            self.calls.append((args, timeout_s))
            return subprocess.CompletedProcess(args, 1, "", "boom")
        done = super().__call__(args, timeout_s)
        if command == "download" and done.returncode == 0 and Path(args[-1]).name == "adapter_config.json":
            Path(args[-1]).write_text(json.dumps({framing.RENDERING_KEY: RENDERING}))
        return done


def _modal(tmp_path: Path, calls: FakeCalls, ledger: Ledger) -> ModalRunner:
    return ModalRunner(calls=calls, ledger=ledger, records=CallRecords(tmp_path / CALLS_DIR), is_synthetic=_synthetic,
                       clock=calls.clock)


def _rig(tmp_path: Path, colab: TimedColab, *, ledger: Ledger | None = None,
         allow_real: bool = False) -> tuple[ColabModalRunner, FakeCalls, Ledger]:
    calls, ledger = FakeCalls(), ledger or Ledger(tmp_path / "budget.jsonl")
    runner = ColabModalRunner(cli=ColabCli(Path("/fake/colab"), run=colab), modal=_modal(tmp_path, calls, ledger),
                              ledger=ledger, is_synthetic=_synthetic, allow_real=allow_real, clock=colab.clock)
    return runner, calls, ledger


def _colab_alone(tmp_path: Path, colab: TimedColab) -> ColabRunner:
    """The ``colab`` setting's runner, on its own ledger."""
    return ColabRunner(cli=ColabCli(Path("/fake/colab"), run=colab), ledger=Ledger(tmp_path / "alone.jsonl"),
                       is_synthetic=_synthetic, clock=colab.clock)


def _warm(tmp_path: Path) -> Path:
    prev = tmp_path / "prev"
    prev.mkdir(exist_ok=True)
    for file in config.ADAPTER_FILES:
        (prev / file).write_bytes(f"prev {file}".encode())
    return prev


def _cu(seconds: float) -> float:
    return seconds / 3600.0 * A100_CU_PER_HOUR


# ── Colab trains ─────────────────────────────────────────────────────────────


def test_a_night_colab_trains_never_reaches_modal_and_makes_the_colab_settings_calls(tmp_path: Path) -> None:
    colab = TimedColab()
    runner, calls, ledger = _rig(tmp_path, colab)
    out = runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert out["runner"] == "colab" and "fallback_reason" not in out and out["steps"] == 9
    assert calls.spawned == [] and not runner.landed(NAME)
    assert ledger.open_holds() == () and ledger.used("modal_usd") == 0.0
    assert ledger.used("colab_cu") == pytest.approx(_cu(8 * MINUTE_S + 600.0))  # what the VM ran for (mkdir: a minute)
    # The same calls, in the same order with the same timeouts, and the same outputs as on ``colab``.
    alone = TimedColab()
    assert dict(_colab_alone(tmp_path / "alone", alone).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)) == {
        key: value for key, value in out.items() if key != "runner"}
    assert alone.calls == colab.calls


# ── Colab's infrastructure fails: Modal trains the same job ─────────────────


@pytest.mark.parametrize(("fail", "timeout", "reason", "colab_s"), [
    ("new", False, "colab new failed with exit 1", 2 * MINUTE_S),  # no A100 to assign: no VM, the time spent
    ("new", True, "colab new timed out", TIMEOUT_S),  # it may have made one: stopped, the whole hold settled
    ("install", False, "colab install failed with exit 1", 3 * MINUTE_S),
    ("mkdir", False, "colab mkdir failed with exit 1", 4 * MINUTE_S),
    ("upload", True, "colab upload timed out", 4 * MINUTE_S + 300.0),  # new, install, mkdir, stop
])
def test_a_colab_failure_before_training_trains_the_same_job_on_modal(
    tmp_path: Path, fail: str, timeout: bool, reason: str, colab_s: float,
) -> None:
    colab = TimedColab(fail, timeout=timeout)
    runner, calls, ledger = _rig(tmp_path, colab)
    out = runner.run(JOB, _inputs(tmp_path, prev=_warm(tmp_path)), timeout_s=TIMEOUT_S)
    assert out["runner"] == "modal" and out["fallback_reason"] == reason and runner.landed(NAME)
    assert colab.commands()[-1] == "stop" and "exec" not in colab.commands()  # Colab's own cleanup ran first
    # The job Modal runs on the ``modal`` setting: the dataset, the recipe, the warm start, the name, the time.
    alone = FakeCalls()
    _modal(tmp_path / "alone", alone, Ledger(tmp_path / "alone.jsonl")).run(
        JOB, _inputs(tmp_path, prev=_warm(tmp_path)), timeout_s=TIMEOUT_S)
    assert calls.spawned == alone.spawned and calls.spawned[0].prev == {
        file: f"prev {file}".encode() for file in config.ADAPTER_FILES}
    # Both pools are settled: Colab's for what its VM ran (all of it when it may have made one), Modal's own.
    assert ledger.open_holds() == ()
    assert ledger.used("colab_cu") == pytest.approx(min(_cu(colab_s), Bound(A100_CU_PER_HOUR, TIMEOUT_S).amount))
    assert ledger.used("modal_usd") == pytest.approx(900.0 / 3600.0 * A100_USD_PER_HOUR)
    assert CallRecords(tmp_path / CALLS_DIR).items() == []  # the call finished: nothing left to sweep


@pytest.mark.parametrize(("fail", "exec_s", "timeout", "reason"), [
    ("exec", 600.0, True, "colab exec timed out"),  # the CLI hung past the exec's hard timeout
    ("exec", KERNEL_S + 30.0, False, "colab exec ran past its timeout"),  # the kernel's own timeout ended it
    ("download", 600.0, True, "colab download timed out"),
])
def test_a_colab_timeout_trains_the_night_on_modal(
    tmp_path: Path, fail: str, exec_s: float, timeout: bool, reason: str,
) -> None:
    colab = TimedColab(fail, timeout=timeout, exec_s=exec_s)
    runner, calls, ledger = _rig(tmp_path, colab)
    out = runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert out["runner"] == "modal" and out["fallback_reason"] == reason and len(calls.spawned) == 1
    assert colab.commands()[-1] == "stop" and ledger.open_holds() == ()
    assert 0 < ledger.used("colab_cu") <= Bound(A100_CU_PER_HOUR, TIMEOUT_S).amount and ledger.used("modal_usd") > 0


# ── Everything else fails the night as on ``colab`` ─────────────────────────


@pytest.mark.parametrize("fail", ["exec", "download"])
def test_a_training_failure_on_colab_fails_the_night_without_modal(tmp_path: Path, fail: str) -> None:
    """An exec that exits non-zero before its timeout: the CLI gives only the exit code, so a Python
    error or a NaN the script reports can't be told from a lost VM, and would fail on Modal too."""
    colab = TimedColab(fail)
    runner, calls, ledger = _rig(tmp_path, colab)
    with pytest.raises(ColabError, match=f"colab {fail} failed with exit 1"):
        runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert calls.spawned == [] and not runner.landed(NAME) and ledger.open_holds() == ()
    alone = TimedColab(fail)
    with pytest.raises(ColabError, match=f"colab {fail} failed with exit 1"):
        _colab_alone(tmp_path / "alone", alone).run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert alone.calls == colab.calls  # exactly as the colab setting fails


@pytest.mark.parametrize("error", [Cancelled("the night was cancelled"), KeyboardInterrupt()])
@pytest.mark.parametrize("fail", ["new", "mkdir", "upload"])
def test_a_cancelled_night_never_falls_back(tmp_path: Path, fail: str, error: BaseException) -> None:
    colab = TimedColab(fail, error=error)
    runner, calls, ledger = _rig(tmp_path, colab)
    with pytest.raises(type(error)):
        runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert calls.spawned == [] and colab.commands()[-1] == "stop" and ledger.open_holds() == ()


def test_a_vm_that_wont_stop_never_falls_back(tmp_path: Path) -> None:
    colab = TimedColab("install", stops=False)  # the VM is up, the install failed, and the stop fails too
    runner, calls, ledger = _rig(tmp_path, colab)
    with pytest.raises(ColabError, match="stop it by hand"):
        runner.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert calls.spawned == [] and ledger.used("colab_cu") == pytest.approx(Bound(A100_CU_PER_HOUR, TIMEOUT_S).amount)


def test_refusals_and_budget_stops_never_fall_back(tmp_path: Path) -> None:
    runner, calls, _ = _rig(tmp_path, TimedColab())
    with pytest.raises(RealPersonRefused):
        runner.run(JOB, _inputs(tmp_path, person="real-1"), timeout_s=TIMEOUT_S)
    channel = {**_inputs(tmp_path), "config": {"r": 16, "channel": {"k": 4}}}
    with pytest.raises(ColabError, match="thought channel"):
        runner.run(JOB, channel, timeout_s=TIMEOUT_S)
    with pytest.raises(ColabError, match="too little time"):
        runner.run(JOB, _inputs(tmp_path), timeout_s=3000.0)
    capped = Ledger(tmp_path / "capped.jsonl", caps={**CAPS, "colab_cu": 1.0})
    stopped, stopped_calls, _ = _rig(tmp_path / "capped", TimedColab(), ledger=capped)
    with pytest.raises(BudgetStop):
        stopped.run(JOB, _inputs(tmp_path), timeout_s=TIMEOUT_S)
    assert calls.spawned == stopped_calls.spawned == []


# ── One Colab VM per accelerated life ───────────────────────────────────────


def test_a_life_falls_back_for_one_night_and_trains_the_next_on_a_fresh_colab_vm(tmp_path: Path) -> None:
    colab = TimedColab("new")  # the life's first night gets no A100
    runner, calls, ledger = _rig(tmp_path, colab)
    with runner.lives(life_timeout_s=6 * 3600.0) as lives:
        assert isinstance(lives, ColabModalLives)
        lives.note_day("synthetic-1", 60.0)
        first = lives.run(JOB, _night(tmp_path, "synthetic-1", 1, warm=False), timeout_s=TIMEOUT_S)
        colab.fail = None  # an A100 again
        second = lives.run(JOB, _night(tmp_path, "synthetic-1", 2, warm=True), timeout_s=TIMEOUT_S)
        assert [r["night"] for r in lives.records("synthetic-1")] == ["2026-11-02"]  # Colab's nights only
    assert (first["runner"], first["fallback_reason"], second["runner"]) == ("modal", "colab new failed with exit 1",
                                                                           "colab")
    assert [job.name for job in calls.spawned] == [adapter_name("synthetic-1", 1, 0)]
    assert runner.landed(adapter_name("synthetic-1", 1, 0)) and not runner.landed(adapter_name("synthetic-1", 2, 0))
    assert colab.commands().count("new") == 2 and colab.commands()[-1] == "stop" and ledger.open_holds() == ()


# ── The receipt and the publisher ───────────────────────────────────────────


def test_the_receipt_names_the_runner_that_trained_and_only_colabs_adapter_is_uploaded(tmp_path: Path) -> None:
    uploaded: list[str] = []
    for fail, trained in ((None, "colab"), ("new", "modal")):
        runner, _, _ = _rig(tmp_path / trained, TimedColab(fail))
        publish = choice.publisher(choice.COLAB_MODAL, lambda folder, name: uploaded.append(name), serving="modal",
                                   landed=runner.landed)
        trainer = LoraTrainer(runner, root=tmp_path / trained / "root", publish=publish, revision=REVISION)
        name = adapter_name("synthetic-1", 1 if trained == "colab" else 2, 0)
        _, logged = trainer.train("synthetic-1", NIGHT, None, _examples(2), name)
        assert logged["runner"] == trained and json.dumps(logged)
        assert logged.get("fallback_reason") == (None if fail is None else "colab new failed with exit 1")
        assert logged["gpu"] and logged["gpu_minutes"] > 0
    assert uploaded == [adapter_name("synthetic-1", 1, 0)]  # Modal's landed on the volume already


def test_the_modal_and_colab_settings_log_todays_receipt(tmp_path: Path) -> None:
    today = {field.name for field in dataclasses.fields(TrainReceipt)} - set(OPTIONAL_KEYS)
    calls = FakeCalls()
    for name, runner in (("modal", _modal(tmp_path, calls, Ledger(tmp_path / "m.jsonl"))),
                         ("colab", _colab_alone(tmp_path / "c", TimedColab()))):
        trainer = LoraTrainer(runner, root=tmp_path / name, publish=lambda folder, n: None, revision=REVISION)
        _, logged = trainer.train("synthetic-1", NIGHT, None, _examples(2), adapter_name("synthetic-1", 1, 0))
        assert set(logged) == today and "runner" not in logged and "fallback_reason" not in logged


def test_colab_modal_uploads_only_what_colab_trained_and_the_others_publish_as_before() -> None:
    uploads: list[str] = []

    def upload(folder: Path, name: str) -> None:
        uploads.append(name)

    landed = {"p-bbbbbbbbbbbb-n2"}.__contains__
    publish = choice.publisher(choice.COLAB_MODAL, upload, serving="modal", landed=landed)
    publish(Path("a"), "p-aaaaaaaaaaaa-n1")
    publish(Path("b"), "p-bbbbbbbbbbbb-n2")  # Modal trained it: on the volume already
    choice.publisher(choice.COLAB_MODAL, upload, serving="sagemaker", landed=landed)(Path("c"), "p-bbbbbbbbbbbb-n2")
    assert uploads == ["p-aaaaaaaaaaaa-n1", "p-bbbbbbbbbbbb-n2"]  # the bucket needs every one
    with pytest.raises(ValueError, match="colab_modal"):
        choice.publisher(choice.COLAB_MODAL, upload, serving="modal")
    assert choice.publisher("colab", upload, serving="modal", landed=lambda name: True) is upload
    assert choice.publisher("modal", upload, serving="sagemaker", landed=lambda name: False) is upload


# ── The setting ─────────────────────────────────────────────────────────────


def test_the_setting_names_colab_modal_and_keeps_modal_and_colab_as_they_were(tmp_path: Path) -> None:
    assert choice.runner_name({choice.RUNNER_ENV: "colab_modal"}) == choice.COLAB_MODAL == "colab_modal"
    assert choice.RUNNERS == ("modal", "colab") and choice.SETTINGS == ("modal", "colab", "colab_modal")
    for bad in ("colab-modal", "modal_colab", "Colab_modal"):
        with pytest.raises(choice.RunnerChoiceError, match="colab_modal"):
            choice.runner_name({choice.RUNNER_ENV: bad})
    built = {name: choice.build(name, ledger=Ledger(tmp_path / "budget.jsonl"), calls_dir=tmp_path / CALLS_DIR,
                                is_synthetic=_synthetic, colab_cli=NoColab, modal_calls=FakeCalls)
             for name in choice.SETTINGS}
    assert type(built["modal"].runner) is ModalRunner and type(built["colab"].runner) is ColabRunner
    assert built["modal"].landed is None and built["colab"].landed is None
    both = built["colab_modal"].runner
    assert isinstance(both, ColabModalRunner) and built["colab_modal"].landed == both.landed


def test_a_failed_colab_listing_is_retried_once_on_colab_modal(tmp_path: Path) -> None:
    """Colab trains these nights, so its VMs may be up: the sweep is never skipped, as on ``colab``."""
    cli = _FlakyColab()
    chosen = choice.build(choice.COLAB_MODAL, ledger=Ledger(tmp_path / "budget.jsonl"), calls_dir=tmp_path / CALLS_DIR,
                          is_synthetic=_synthetic, colab_cli=lambda: cli, modal_calls=FakeCalls)
    with pytest.raises(ColabError, match="exit 1"):
        chosen.sweep()
    assert cli.calls == 2
    clean = choice.build(choice.COLAB_MODAL, ledger=Ledger(tmp_path / "b.jsonl"), calls_dir=tmp_path / CALLS_DIR,
                         is_synthetic=_synthetic, colab_cli=lambda: _EmptyColab(), modal_calls=FakeCalls)
    assert clean.sweep() == Stopped(0, 0)


def test_the_night_and_the_accelerated_run_train_on_colab_modal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.night import __main__ as night_command
    from anjo.sleep.sim import accelerated
    from anjo.sleep.sim.day import DayRun

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(choice.RUNNER_ENV, choice.COLAB_MODAL)
    monkeypatch.setattr(night_command, "_colab_cli", lambda: _EmptyColab())
    monkeypatch.setattr(night_command, "_modal_calls", FakeCalls)
    seen: list[Any] = []

    def wiring(trainer: Any, options: object) -> object:
        seen.append(trainer._runner)
        raise RuntimeError("stop here")

    monkeypatch.setattr(night_command, "_wiring", lambda path: wiring)
    with pytest.raises(RuntimeError, match="stop here"):
        night_command.execute(wiring="w:x", night=datetime(2026, 10, 28).date(), users=["u1"], personas={"u1": "founder"})
    assert isinstance(seen[-1], ColabModalRunner)
    start = (datetime.now(timezone.utc) - timedelta(days=3)).date()
    wall = (datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc), datetime(2026, 10, 28, 9, 5, tzinfo=timezone.utc))
    named = {"w": wiring, "s": lambda *args: DayRun(wall, 3, 0, ()), "m": lambda *args: None}
    monkeypatch.setattr(accelerated, "_named", lambda path: named[path])
    with pytest.raises(RuntimeError, match="stop here"):
        accelerated.main(["--wiring", "w", "--simulate", "s", "--morning", "m", "--user", "u1", "--persona", "founder",
                          "--days", "1", "--start", start.isoformat()])
    assert isinstance(seen[-1], ColabModalLives)  # one Colab VM per life, as on colab
