# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Nights train ``sft_dpo`` by default (Kevin, 2026-10-28): ``ANJO_SLEEP_OBJECTIVE`` is ``sft_dpo`` or ``sft``,
refused otherwise at start; a night with no usable pair trains ``sft`` and logs the counts; MIND's thinker
keeps its own recipe. Offline: the runners are fakes."""
from __future__ import annotations

import dataclasses
import inspect
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.serving import config
from anjo.sleep.tests.test_lora import REVISION, _examples, _Runner, _with_pairs
from anjo.sleep.train import colab_train, objective, thinker
from anjo.sleep.train.lora import LoraCfg, adapter_name, plan_steps, train_lora_with_receipt
from anjo.sleep.train.trainer import LoraTrainer
from anjo.sleep.types import CheckpointRef

NIGHT = date(2026, 10, 28)


class Seen(_Runner):
    """A fake GPU runner that keeps what it was sent: the recipe and the dataset's rows."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    def run(self, job: str, inputs: Any, *, timeout_s: float) -> Any:
        text = Path(str(inputs["dataset_path"])).read_text(encoding="utf-8")
        self.rows = [json.loads(line) for line in text.splitlines()]
        return super().run(job, inputs, timeout_s=timeout_s)


def _train(tmp_path: Path, data: Any, cfg: LoraCfg, prev: CheckpointRef | None = None) -> tuple[Seen, Any]:
    runner = Seen()
    _, receipt = train_lora_with_receipt(
        prev, data, cfg, runner, person_id="p1", night=NIGHT, name=adapter_name("p1", 2, 0),
        workdir=tmp_path / "work", adapter_cache=tmp_path / "adapters", publish=lambda folder, n: None,
        revision=REVISION)
    return runner, receipt


def test_nights_train_sft_dpo_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert objective.DEFAULT_OBJECTIVE == "sft_dpo" and objective.OBJECTIVES == ("sft_dpo", "sft")
    assert objective.night_cfg({}).objective == "sft_dpo"
    assert objective.night_cfg({objective.OBJECTIVE_ENV: " "}).objective == "sft_dpo"
    # The night command builds its trainer with it, and the pairs reach the job.
    from anjo.sleep.night import __main__ as command

    built: list[LoraCfg | None] = []

    class Capture(LoraTrainer):
        def __init__(self, *args: Any, cfg: LoraCfg | None = None, **kw: Any) -> None:
            built.append(cfg)
            super().__init__(*args, cfg=cfg, **kw)

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(objective.OBJECTIVE_ENV, raising=False)
    monkeypatch.setattr(command, "LoraTrainer", Capture)
    monkeypatch.setattr(command, "_wiring", lambda path: lambda trainer, options: "deps")
    monkeypatch.setattr(command, "run_night_batch_with", lambda deps, nights, **kw: {})
    command.execute(wiring="w:x", night=NIGHT, users=["u1"], personas={"u1": "founder"})
    assert [cfg.objective if cfg else None for cfg in built] == ["sft_dpo"]
    runner, receipt = _train(tmp_path, _with_pairs(3), objective.night_cfg({}))
    assert runner.calls[0][1]["config"]["objective"] == "sft_dpo"  # type: ignore[index]
    assert sum("chosen" in row for row in runner.rows) == 3 and receipt.objective == "sft_dpo"


def test_the_accelerated_run_and_the_example_train_with_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    from anjo.sleep.examples.synthetic import stages
    from anjo.sleep.sim import accelerated

    monkeypatch.delenv(objective.OBJECTIVE_ENV, raising=False)
    assert accelerated.main(["--fake", "--days", "3"]) == 0
    nights = json.loads(capsys.readouterr().out)["nights"]
    assert [n["objective"] for n in nights] == ["sft_dpo"] * 3 and all(n["pairs"] >= 1 for n in nights)
    sent: list[LoraCfg] = []

    def fake_train(prev: Any, data: Any, cfg: LoraCfg, *args: Any, **kw: Any) -> Any:
        sent.append(cfg)
        return SimpleNamespace(id="x"), SimpleNamespace(steps=1, gpu_minutes=0.0)

    monkeypatch.setattr(stages, "train_lora_with_receipt", fake_train)
    night = SimpleNamespace(person=SimpleNamespace(person_id="p1", day=NIGHT), run_id="r", workdir=tmp_path,
                            say=lambda text: None, services=SimpleNamespace(gpu=None, publish=print, revision=REVISION))
    stages.train(night, _examples(1))  # type: ignore[arg-type]
    assert [cfg.objective for cfg in sent] == ["sft_dpo"]
    monkeypatch.setenv(objective.OBJECTIVE_ENV, "sft")
    stages.train(night, _examples(1))  # type: ignore[arg-type]
    assert [cfg.objective for cfg in sent] == ["sft_dpo", "sft"]


def test_objective_setting_picks_sft_on_request_and_refuses_others(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    assert objective.OBJECTIVE_ENV == "ANJO_SLEEP_OBJECTIVE"
    assert objective.objective_name({objective.OBJECTIVE_ENV: "sft"}) == "sft"
    assert objective.night_cfg({objective.OBJECTIVE_ENV: "sft"}).objective == "sft"
    assert objective.objective_name({objective.OBJECTIVE_ENV: "sft_dpo"}) == "sft_dpo"
    for bad in ("dpo", "SFT", "sft,dpo", "none"):
        with pytest.raises(objective.ObjectiveError, match="ANJO_SLEEP_OBJECTIVE is sft_dpo or sft"):
            objective.objective_name({objective.OBJECTIVE_ENV: bad})
    # Refused at start: the night command stops before it builds a runner or a trainer.
    from anjo.sleep.night import __main__ as command

    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(objective.OBJECTIVE_ENV, "dpo")
    monkeypatch.setattr(command, "chosen_runner", lambda *a, **k: pytest.fail("a runner was built"))
    with pytest.raises(SystemExit, match="ANJO_SLEEP_OBJECTIVE"):
        command.execute(wiring="w:x", night=NIGHT, users=["u1"], personas={"u1": "founder"})


def test_night_without_pairs_trains_sft_and_logs_it(tmp_path: Path) -> None:
    cfg = LoraCfg(objective="sft_dpo")
    runner, receipt = _train(tmp_path, _examples(3), cfg)  # practice kept no (best, worst) pair
    assert runner.calls[0][1]["config"]["objective"] == "sft"  # type: ignore[index]
    assert not any("chosen" in row for row in runner.rows)
    assert (receipt.objective, receipt.pairs) == ("sft", 0)
    runner, receipt = _train(tmp_path / "b", _with_pairs(3), cfg)
    assert (receipt.objective, receipt.pairs) == ("sft_dpo", 3)
    # The trainer's receipt is what the night log records: counts and the objective's name, no words.
    trainer = LoraTrainer(Seen(), root=tmp_path / "t", publish=lambda f, n: None, revision=REVISION, cfg=cfg)
    _, logged = trainer.train("p1", NIGHT, None, _examples(2), adapter_name("p1", 3, 0))
    assert (logged["objective"], logged["pairs"]) == ("sft", 0)
    assert all(isinstance(v, int | float | str | bool) for v in logged.values())
    assert "answer" not in json.dumps(logged) and dataclasses.asdict(receipt)["pairs"] == 3


def test_warm_start_sft_dpo_job_is_valid(tmp_path: Path) -> None:
    prev = CheckpointRef("p-aaaaaaaaaaaa-n1", "p1", date(2026, 10, 27), "modal-volume://x/p-aaaaaaaaaaaa-n1", None)
    folder = tmp_path / "adapters" / prev.id
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16}))
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    runner, receipt = _train(tmp_path, _with_pairs(4), LoraCfg(objective="sft_dpo"), prev)
    _, inputs, _ = runner.calls[0]
    sent: Any = inputs["config"]
    assert sent["warm_start"] is True and inputs["prev_adapter_dir"] == str(folder)
    assert (sent["objective"], sent["learning_rate"], sent["dpo_learning_rate"]) == ("sft_dpo", 5e-5, 5e-6)
    loaded = colab_train.load_config(json.dumps(sent))  # the VM's own validation takes the warm job
    settings = colab_train.build_settings(loaded)
    dpo = settings["dpo"]  # what the script hands TRL's DPOConfig for this warm job
    assert (dpo["learning_rate"], dpo["beta"]) == (5e-6, 0.1) and settings["sft"]["learning_rate"] == 5e-5
    assert dpo["learning_rate"] <= colab_train.MAX_DPO_LR and dpo["num_train_epochs"] == 1
    assert dpo["gradient_accumulation_steps"] == loaded["accumulation"] and dpo["precompute_ref_log_probs"] is False
    assert colab_train.dpo_plan(4, loaded) == (plan_steps(4, LoraCfg(objective="sft_dpo")).accumulation,
                                               plan_steps(4, LoraCfg(objective="sft_dpo")).epochs)
    assert (receipt.objective, receipt.pairs, receipt.warm_start) == ("sft_dpo", 4, True)


class Counting(Seen):
    """A runner whose job reports the pairs it trained and dropped, as ``colab_train``'s receipt does."""

    def __init__(self, trained: int, dropped: int) -> None:
        super().__init__()
        self._extra = {"pairs": trained, "pairs_dropped_truncated": dropped}

    def run(self, job: str, inputs: Any, *, timeout_s: float) -> Any:
        return {**super().run(job, inputs, timeout_s=timeout_s), **self._extra}


def _receipt(tmp_path: Path, data: Any, runner: Seen) -> Any:
    _, receipt = train_lora_with_receipt(
        None, data, LoraCfg(objective="sft_dpo"), runner, person_id="p1", night=NIGHT,
        name=adapter_name("p1", 2, 0), workdir=tmp_path / "w", adapter_cache=tmp_path / "a",
        publish=lambda folder, n: None, revision=REVISION)
    return receipt


def test_receipt_counts_pairs_trained_not_sent(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, _with_pairs(5), Counting(trained=3, dropped=2))
    assert (receipt.objective, receipt.pairs) == ("sft_dpo", 3)


def test_all_pairs_dropped_records_sft(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, _with_pairs(2), Counting(trained=0, dropped=2))
    assert (receipt.objective, receipt.pairs) == ("sft", 0)


def test_thinker_objective_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(objective.OBJECTIVE_ENV, "sft_dpo")
    for fn in (thinker.plan, thinker.train_thinkers):
        assert inspect.signature(fn).parameters["cfg"].default.objective == "sft"
    source = Path(thinker.__file__).read_text(encoding="utf-8")
    assert "OBJECTIVE_ENV" not in source and "night_cfg" not in source
    assert LoraCfg().objective == "sft"  # the recipe's own default is untouched; only nights ask for the setting
