"""``python -m anjo.sleep.night`` (Task 5): a real person runs only with ``--allow-real``, and the
wiring is named, never imported by the core."""
from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.night.__main__ import main

BASE = ["--wiring", "anjo.sleep.tests.test_night_cli:never", "--night", "2026-10-28"]


def never(*args: object) -> object:
    raise AssertionError("the wiring is not reached")


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
