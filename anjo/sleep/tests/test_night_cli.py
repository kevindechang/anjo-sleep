# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""``python -m anjo.sleep.night`` (Task 5): a real person runs only with ``--allow-real``, the wiring
is named, never imported by the core, and the training runner's leftovers are swept first."""
from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.night.__main__ import main
from anjo.sleep.tests.test_modal_runner import FakeCalls

BASE = ["--wiring", "anjo.sleep.tests.test_night_cli:never", "--night", "2026-10-28"]


def never(*args: object) -> object:
    raise AssertionError("the wiring is not reached")


class Cli:
    """The night command's Colab CLI: no VM, no network. ``colab sessions`` lists ``listed``."""

    def __init__(self, listed: str = "[colab] No active sessions found on server.\n") -> None:
        self.listed = listed
        self.calls: list[tuple[str, ...]] = []

    def call(self, *args: str, timeout_s: float) -> str:
        self.calls.append(args)
        return self.listed if args[0] == "sessions" else ""


@pytest.fixture(autouse=True)
def no_colab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Cli:
    from anjo.sleep.train.runners import colab

    cli = Cli()
    monkeypatch.setattr("anjo.sleep.night.__main__._colab_cli", lambda: cli)
    installed = tmp_path / "colab-cli"
    installed.write_text("")
    monkeypatch.setattr(colab, "COLAB_BIN", installed)  # the Colab CLI is installed (this fake answers)
    return cli


@pytest.fixture(autouse=True)
def no_modal(monkeypatch: pytest.MonkeyPatch) -> FakeCalls:
    calls = FakeCalls()
    monkeypatch.setattr("anjo.sleep.night.__main__._modal_calls", lambda: calls)
    return calls


def test_a_user_without_a_persona_needs_allow_real(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as stopped:
        main([*BASE, "--user", "u1"])
    assert stopped.value.code == 2


def test_more_personas_than_users_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        main([*BASE, "--user", "u1", "--persona", "founder", "--persona", "researcher"])


def test_the_wiring_gets_the_trainer_and_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.night.deps import NightOptions
    from anjo.sleep.train.trainer import LoraTrainer

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    seen: list[tuple[object, NightOptions]] = []

    def wiring(trainer: object, options: NightOptions) -> object:
        seen.append((trainer, options))
        raise RuntimeError("stop here")

    monkeypatch.setattr("anjo.sleep.tests.test_night_cli.never", wiring)
    with pytest.raises(RuntimeError, match="stop here"):
        main([*BASE, "--user", "u1", "--persona", "founder"])
    [(trainer, options)] = seen
    assert isinstance(trainer, LoraTrainer) and options.personas == {"u1": "founder"} and not options.allow_real
    assert options.force_deep is False
    with pytest.raises(RuntimeError, match="stop here"):
        main([*BASE, "--user", "u1", "--persona", "founder", "--force-deep"])
    assert seen[-1][1].force_deep is True  # tonight is a deep sleep whatever its pressure


def test_the_trainer_deletes_the_plain_text_dataset_even_on_failure(tmp_path: Path) -> None:
    from datetime import date

    from anjo.sleep.tests.test_lora import _examples
    from anjo.sleep.train.trainer import LoraTrainer

    class Failing:
        def run(self, job: str, inputs: object, *, timeout_s: float) -> object:
            raise RuntimeError("the VM went away")

    dataset = _examples(3)
    trainer = LoraTrainer(Failing(), root=tmp_path, publish=lambda folder, name: None, revision="a" * 40)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="went away"):
        trainer.train("u1", date(2026, 10, 28), None, dataset, "p-aaaaaaaaaaaa-n1")
    assert not list(tmp_path.rglob("dataset.jsonl"))


def test_a_failed_note_still_exits_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.tests.night_fakes import NIGHT, World, thread
    from anjo.sleep.tests.test_night_run import AlwaysNote, Raising

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    world = World(tmp_path)
    world.plan("u1", NIGHT)
    world.targets("u1").upsert(thread("u1"))
    monkeypatch.setattr("anjo.sleep.tests.test_night_cli.never",
                        lambda trainer, options: world.deps(chooser=AlwaysNote(), note_writer=Raising()))
    assert main([*BASE, "--user", "u1", "--persona", "founder"]) == 0


@pytest.mark.parametrize("runner", ["colab", "modal"])
def test_the_night_command_sweeps_a_killed_run_before_the_wiring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_colab: Cli, no_modal: FakeCalls, runner: str,
) -> None:
    from anjo.sleep.budget import Bound, default_ledger
    from anjo.sleep.tests.test_night_sweep import _killed
    from anjo.sleep.train.runners.modal import CALLS_DIR, CallRecords

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANJO_SLEEP_TRAIN_RUNNER", runner)
    no_colab.listed = "[anjo-sleep-p-aaaaaaaaaaaa-n3] gpu-a100-s-1 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
    records = CallRecords(tmp_path / "sleep" / CALLS_DIR)
    records.add("p-aaaaaaaaaaaa-n4", "fc-00000009")  # the Modal call a killed run left
    _killed(default_ledger())
    assert default_ledger().reserve("modal_train:2026-10-28:u1:a1", ("modal_usd",), Bound(2.5, 7200.0)) is not None
    seen: list[int] = []

    def wiring(trainer: object, options: object) -> object:
        seen.append(len(default_ledger().open_holds()))
        raise RuntimeError("stop here")

    monkeypatch.setattr("anjo.sleep.tests.test_night_cli.never", wiring)
    with pytest.raises(RuntimeError, match="stop here"):
        main([*BASE, "--user", "u1", "--persona", "founder"])
    assert seen == [0]  # every stale hold settled before the wiring ran, Modal's training holds too
    # Whichever runner trains, both runners' leftovers are stopped before their holds are settled.
    assert no_colab.calls == [("sessions",), ("stop", "-s", "anjo-sleep-p-aaaaaaaaaaaa-n3")]
    assert no_modal.cancelled == ["fc-00000009"] and records.items() == []


def test_a_modal_night_leaves_colab_alone_where_its_cli_isnt_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_colab: Cli,
) -> None:
    from anjo.sleep.train.runners import colab

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANJO_SLEEP_TRAIN_RUNNER", raising=False)
    monkeypatch.setattr(colab, "COLAB_BIN", tmp_path / "no-colab")
    monkeypatch.setattr("anjo.sleep.tests.test_night_cli.never", lambda *args: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        main([*BASE, "--user", "u1", "--persona", "founder"])
    assert no_colab.calls == []  # without the CLI, nothing here can have started a VM


def test_a_simulated_day_runs_its_night_on_the_virtual_clock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dataclasses import dataclass
    from datetime import date, datetime, timezone
    from typing import Any

    from anjo.sleep.night import __main__ as command
    from anjo.sleep.night.dreams import Terms
    from anjo.sleep.night.run import day_window
    from anjo.sleep.sim.day import DayRun

    @dataclass(frozen=True)
    class Deps:
        """The night's wiring stand-in: the command replaces its terms with the default recipe's."""

        terms: Terms = Terms()

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    wall = (datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc), datetime(2026, 10, 28, 9, 5, tzinfo=timezone.utc))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(command, "_wiring", lambda path: lambda trainer, options: Deps())
    monkeypatch.setattr(command, "_named", lambda path: lambda *args: DayRun(wall, 3, 0, ()))
    monkeypatch.setattr(command, "run_night_batch_with", lambda deps, nights, **kw: seen.update(kw) or {})
    night = date(2026, 11, 20)
    command.execute(wiring="w:x", night=night, users=["u1"], personas={"u1": "founder"}, simulate="s:x", day=3)
    assert seen["days"]["u1"].window == wall and seen["as_of"] == {"u1": day_window(night)[1]}
