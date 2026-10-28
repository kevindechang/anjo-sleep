# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Which runner trains (Task 5): ``ANJO_SLEEP_TRAIN_RUNNER`` is ``modal`` by default or ``colab``, and
the night, the accelerated runs and the example all pick through it, while MIND's training stays on
Colab whatever it says. The train smoke's plan on Modal prints the hold and spends nothing. Modal and
Colab are fakes: no container, no VM, no network."""
from __future__ import annotations

import ast
import importlib
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import anjo.sleep
from anjo.sleep.budget import Ledger
from anjo.sleep.serving import config
from anjo.sleep.tests.test_modal_runner import FakeCalls
from anjo.sleep.train import smoke, thinker
from anjo.sleep.train.lora import TrainError
from anjo.sleep.train.runners import choice
from anjo.sleep.train.runners.colab import MIND_SESSION_PREFIX, ColabCli, ColabRunner
from anjo.sleep.train.runners.modal import CALLS_DIR, A100_USD_PER_HOUR, CallRecords, ModalRunner

SLEEP = Path(anjo.sleep.__file__).parent


class NoColab(ColabCli):
    """A Colab CLI that must not be called."""

    def call(self, *args: str, timeout_s: float) -> str:
        raise AssertionError(f"colab {args[0]} was called")


def _build(tmp_path: Path, name: str) -> choice.Chosen:
    return choice.build(name, ledger=Ledger(tmp_path / "budget.jsonl"), calls_dir=tmp_path / CALLS_DIR,
                        is_synthetic=lambda uid: True, colab_cli=NoColab, modal_calls=FakeCalls)


def test_runner_setting_picks_modal_by_default_and_colab_on_request(tmp_path: Path) -> None:
    assert choice.DEFAULT_RUNNER == "modal" and choice.RUNNERS == ("modal", "colab")
    assert choice.runner_name({}) == "modal" and choice.runner_name({choice.RUNNER_ENV: " "}) == "modal"
    assert choice.runner_name({choice.RUNNER_ENV: "modal"}) == "modal"
    assert choice.runner_name({choice.RUNNER_ENV: "colab"}) == "colab"
    for bad in ("Modal", "sagemaker", "colab,modal"):
        with pytest.raises(choice.RunnerChoiceError):
            choice.runner_name({choice.RUNNER_ENV: bad})
    modal_run, colab_run = _build(tmp_path, "modal"), _build(tmp_path, "colab")
    assert isinstance(modal_run.runner, ModalRunner) and modal_run.name == "modal"
    assert isinstance(colab_run.runner, ColabRunner) and colab_run.name == "colab"
    assert modal_run.sweep() == (0, 0)  # no call left behind, and Colab is never asked


def test_the_night_trains_on_the_chosen_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.night import __main__ as night_command

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(night_command, "_colab_cli", NoColab)
    monkeypatch.setattr(night_command, "_modal_calls", FakeCalls)
    seen: list[Any] = []

    def wiring(trainer: Any, options: object) -> object:
        seen.append(trainer._runner)
        raise RuntimeError("stop here")

    monkeypatch.setattr(night_command, "_wiring", lambda path: wiring)
    for setting, kind in ((None, ModalRunner), ("modal", ModalRunner), ("colab", ColabRunner)):
        if setting is None:
            monkeypatch.delenv(choice.RUNNER_ENV, raising=False)
        else:
            monkeypatch.setenv(choice.RUNNER_ENV, setting)
        if kind is ColabRunner:
            monkeypatch.setattr(night_command, "_colab_cli", lambda: _EmptyColab())
        with pytest.raises(RuntimeError, match="stop here"):
            night_command.execute(wiring="w:x", night=datetime(2026, 10, 28).date(), users=["u1"],
                                  personas={"u1": "founder"})
        assert isinstance(seen[-1], kind)


class _EmptyColab:
    def call(self, *args: str, timeout_s: float) -> str:
        return "[colab] No active sessions found on server.\n" if args[0] == "sessions" else ""


def test_the_night_lets_modal_train_only_the_real_people_it_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.night import __main__ as night_command

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(choice.RUNNER_ENV, raising=False)
    monkeypatch.setattr(night_command, "_modal_calls", FakeCalls)
    seen: list[Any] = []

    def wiring(trainer: Any, options: object) -> object:
        seen.append(trainer._runner)
        raise RuntimeError("stop here")

    monkeypatch.setattr(night_command, "_wiring", lambda path: wiring)
    with pytest.raises(RuntimeError):
        night_command.execute(wiring="w:x", night=datetime(2026, 10, 28).date(), users=["kevin-dev", "u1"],
                              personas={"u1": "founder"}, allow_real=True)
    assert seen[-1].allow_real == frozenset({"kevin-dev"})
    with pytest.raises(RuntimeError):
        night_command.execute(wiring="w:x", night=datetime(2026, 10, 28).date(), users=["u1"], personas={"u1": "founder"})
    assert seen[-1].allow_real == frozenset()


def test_the_accelerated_run_trains_on_the_chosen_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.night import __main__ as night_command
    from anjo.sleep.sim import accelerated
    from anjo.sleep.sim.day import DayRun

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(night_command, "_modal_calls", FakeCalls)
    seen: list[Any] = []
    start = (datetime.now(timezone.utc) - timedelta(days=3)).date()
    wall = (datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc), datetime(2026, 10, 28, 9, 5, tzinfo=timezone.utc))

    def wiring(trainer: Any, options: object) -> object:
        seen.append(trainer._runner)
        raise RuntimeError("stop here")

    named = {"w": wiring, "s": lambda *args: DayRun(wall, 3, 0, ()), "m": lambda *args: None}
    monkeypatch.setattr(accelerated, "_named", lambda path: named[path])
    monkeypatch.delenv(choice.RUNNER_ENV, raising=False)
    with pytest.raises(RuntimeError, match="stop here"):
        accelerated.main(["--wiring", "w", "--simulate", "s", "--morning", "m", "--user", "u1", "--persona", "founder",
                          "--days", "1", "--start", start.isoformat()])
    assert isinstance(seen[-1], ModalRunner)  # one call a night: no VM is kept between nights
    monkeypatch.setenv(choice.RUNNER_ENV, "colab")
    monkeypatch.setattr(night_command, "_colab_cli", lambda: _EmptyColab())
    with pytest.raises(RuntimeError, match="stop here"):
        accelerated.main(["--wiring", "w", "--simulate", "s", "--morning", "m", "--user", "u1", "--persona", "founder",
                          "--days", "1", "--start", start.isoformat()])
    assert type(seen[-1]).__name__ == "ColabLives"  # Colab keeps one VM per life


def test_the_example_trains_on_the_chosen_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep import baseten_api, meta_api
    from anjo.sleep.examples.synthetic import live
    from anjo.sleep.serving.backend import MODAL_TOKEN_ENV, MODAL_URL_ENV

    env = {meta_api.KEY_ENV: "k", "BASETEN_API_KEY": "k", baseten_api.KEY_ENV: "k", MODAL_URL_ENV: "https://x.invalid",
           MODAL_TOKEN_ENV: "t", config.REVISION_ENV: "c" * 40, "MODAL_TOKEN_ID": "i", "MODAL_TOKEN_SECRET": "s"}
    real_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "modal" else real_find(name, *a))
    services = live.build_services(env)
    assert isinstance(services.gpu, ModalRunner)
    assert services.publish is not None and services.publish(Path("/nowhere"), "p-aaaaaaaaaaaa-n1") is None  # landed
    services = live.build_services({**env, choice.RUNNER_ENV: "colab"})
    assert isinstance(services.gpu, ColabRunner) and services.publish is None  # uploaded as before


def test_a_modal_trained_adapter_is_uploaded_only_where_it_didnt_land() -> None:
    uploads: list[tuple[Path, str]] = []

    def upload(folder: Path, name: str) -> None:
        uploads.append((folder, name))

    choice.publisher("modal", upload, serving="modal")(Path("a"), "p-aaaaaaaaaaaa-n1")  # already on the volume
    choice.publisher("modal", upload, serving="sagemaker")(Path("b"), "p-aaaaaaaaaaaa-n2")  # the bucket needs it
    choice.publisher("colab", upload, serving="modal")(Path("c"), "p-aaaaaaaaaaaa-n3")
    assert uploads == [(Path("b"), "p-aaaaaaaaaaaa-n2"), (Path("c"), "p-aaaaaaaaaaaa-n3")]


def test_mind_training_stays_on_colab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.tests.test_replay_train import _replay

    for setting in (None, "modal", "colab"):
        if setting is None:
            monkeypatch.delenv(choice.RUNNER_ENV, raising=False)
        else:
            monkeypatch.setenv(choice.RUNNER_ENV, setting)
        runner = thinker.colab_runner(ColabCli(Path("/fake/colab")), Ledger(tmp_path / "b.jsonl"))
        assert isinstance(runner, ColabRunner) and runner.session_prefix == MIND_SESSION_PREFIX
    modal = _build(tmp_path, "modal").runner
    with pytest.raises(TrainError, match="Colab"):  # replay refuses a Modal runner outright
        thinker.train_thinkers(_replay(), modal, Ledger(tmp_path / "b.jsonl"), workdir=tmp_path / "w",  # type: ignore[arg-type]
                               adapter_cache=tmp_path / "c", publish=lambda f, n: None, revision="a" * 40,
                               dry_run=False, out=lambda _: None)
    assert FakeCalls().spawned == []
    # MIND (the APD runner, replay's thinker and the night's inquiry) never reads the setting or names Modal's runner.
    mind_files = [*sorted((SLEEP / "mind").rglob("*.py")), SLEEP / "train" / "thinker.py"]
    for path in mind_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert not {"anjo.sleep.train.runners.choice", "anjo.sleep.train.runners.modal",
                    "anjo.sleep.train.modal_app"} & imported, path
        assert choice.RUNNER_ENV not in path.read_text(encoding="utf-8"), path


def test_smoke_plan_on_modal_spends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    calls = FakeCalls(on_spawn=lambda job: pytest.fail("the plan spawned a call"))
    assert smoke.gpu_main("modal", run=False, modal_calls=lambda: calls, colab_cli=NoColab) == 0
    printed = capsys.readouterr().out
    held = smoke.SMOKE_TIMEOUT_S / 3600.0 * A100_USD_PER_HOUR
    assert f"hold: ${held:.2f} on modal_usd" in printed and "nothing was held, started or spent" in printed
    assert "500.00 left" in printed  # read from the ledger, which the plan never writes
    assert calls.spawned == [] and not any(tmp_path.rglob("*.jsonl")) and not CallRecords(tmp_path / "sleep" / CALLS_DIR).used()
    assert smoke.gpu_main("colab", run=False, modal_calls=lambda: calls, colab_cli=NoColab) == 0
    assert "on colab_cu" in capsys.readouterr().out


def test_the_smoke_runs_on_a_gpu_runner_with_the_nights_own_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    from anjo.sleep.serving import framing
    from anjo.sleep.tests.test_modal_runner import finished

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(config.REVISION_ENV, "c" * 40)
    calls = FakeCalls()

    def receipt(call_id: str, timeout_s: float) -> dict[str, Any]:
        job = calls.spawned[-1]
        done = finished(job)
        done["receipt"] = {"steps": smoke.MIN_STEPS, "epochs": 1, "train_minutes": 1.0, "examples": 4,
                           "dropped_truncated": 1, "pairs": 2, "pairs_dropped_truncated": 1, "objective": "sft_dpo",
                           "rows_checked": 6, "dpo_reference": "adapter_off", "dpo_steps": 1,
                           "rendering": framing.rendering(config.reasoning_strength()), "gpu": "NVIDIA A100"}
        return done

    calls.wait = receipt  # type: ignore[method-assign]
    assert smoke.gpu_main("modal", run=True, modal_calls=lambda: calls, colab_cli=NoColab) == 0
    [job] = calls.spawned
    recipe = __import__("json").loads(job.config_json)
    assert recipe["objective"] == "sft_dpo" and recipe["revision"] == "c" * 40 and recipe["warm_start"] is False
    assert recipe["template_sha256"] == framing.TEMPLATE_SHA256  # Glimmer's own template, not the tiny model's
    printed = capsys.readouterr().out
    assert "smoke passed" in printed and f"modal volume rm {config.ADAPTERS_VOLUME} /{job.name}" in printed
    assert Ledger(tmp_path / "sleep" / "budget.jsonl").open_holds() == ()
