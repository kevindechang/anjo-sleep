# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The LoRA recipe and ``train_lora`` (Task 5, Kevin's 2026-10-27 recipe): rank capped at the
server's 32, MLP included, enough optimizer steps on a tiny night, names never reused, and the
training script builds its settings offline without the GPU stack."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pytest

from anjo.sleep.serving import config
from anjo.sleep.train import colab_train
from anjo.sleep.train.dataset import Dataset, Example
from anjo.sleep.train.lora import (
    DATASET_FILE,
    DEFAULT_TARGET_MODULES,
    LoraCfg,
    LoraCfgError,
    adapter_name,
    fresh_adapter_name,
    TrainError,
    plan_steps,
    train_lora_with_receipt,
)
from anjo.sleep.types import CheckpointRef

NIGHT = date(2026, 10, 28)


def _examples(n: int) -> Dataset:
    examples = tuple(
        Example(id=f"x{i}", prompt=(("user", f"question {i}"),), completion=f"answer {i}",
                context_dropped=True, tokens=10, new=True)
        for i in range(n)
    )
    return Dataset(examples=examples, tokens=10 * n, dropped={})


def test_rank_above_32_refused() -> None:
    assert LoraCfg(r=32).alpha == 64
    with pytest.raises(LoraCfgError):
        LoraCfg(r=33)
    with pytest.raises(LoraCfgError):
        LoraCfg(r=0)


def test_recipe_defaults_follow_the_ledgered_recipe() -> None:
    cfg = LoraCfg()
    assert (cfg.r, cfg.alpha, cfg.dropout) == (16, 32, 0.05)
    assert set(DEFAULT_TARGET_MODULES) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert cfg.target_modules == DEFAULT_TARGET_MODULES
    assert (cfg.lr, cfg.warm_lr) == (2e-4, 5e-5)
    assert cfg.max_seq_length >= 8192 and cfg.packing is False
    assert cfg.min_optimizer_steps == 8
    with pytest.raises(LoraCfgError):
        LoraCfg(max_seq_length=4096)
    with pytest.raises(LoraCfgError):
        LoraCfg(packing=True)
    assert config.MAX_LORA_RANK == 32


def test_min_optimizer_steps_reached_on_tiny_dataset() -> None:
    cfg = LoraCfg()
    for n in (3, 5, 8, 12, 40, 200):
        plan = plan_steps(n, cfg)
        assert 1 <= plan.accumulation <= 16
        assert plan.epochs <= 4
        assert plan.steps >= cfg.min_optimizer_steps, n
    # A tiny night: batch 1 x accumulation 16 would give one update; the plan gives eight or more.
    tiny = plan_steps(12, cfg)
    assert tiny.accumulation == 1 and tiny.steps >= 8
    # Two examples can't reach eight steps inside four epochs; the cap holds and says so.
    two = plan_steps(2, cfg)
    assert (two.accumulation, two.epochs, two.steps) == (1, 4, 8)
    one = plan_steps(1, cfg)
    assert (one.epochs, one.steps) == (4, 4)
    with pytest.raises(LoraCfgError):
        plan_steps(0, cfg)


def test_adapter_name_never_reused() -> None:
    name = adapter_name("p1", 3, 0)
    config.check_adapter_name(name)
    assert name.endswith("-n3")
    used = frozenset({name})
    again = fresh_adapter_name("p1", 3, used)
    assert again != name and again.endswith("-n3")
    config.check_adapter_name(again)
    assert fresh_adapter_name("p1", 3, frozenset()) == name
    assert adapter_name("p2", 3, 0) != name


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, object], float]] = []

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        self.calls.append((job, dict(inputs), timeout_s))
        out = Path(str(inputs["workdir"])) / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16}))
        (out / "adapter_model.safetensors").write_bytes(b"weights")
        return {"adapter_dir": str(out), "steps": 9, "epochs": 3, "train_minutes": 4.0,
                "gpu_minutes": 11.0, "compute_units": 0.97, "examples": 3, "dropped_truncated": 0,
                "gpu": "A100-80GB"}


REVISION = "a" * 40


def test_train_lora_runs_the_job_publishes_the_adapter_and_records_steps(tmp_path: Path) -> None:
    runner, published = _Runner(), []
    prev = CheckpointRef("p-aaaaaaaaaaaa-n1", "p1", date(2026, 10, 27), "modal-volume://x/p-aaaaaaaaaaaa-n1", None)
    cache = tmp_path / "adapters"
    (cache / prev.id).mkdir(parents=True)
    (cache / prev.id / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16}))
    (cache / prev.id / "adapter_model.safetensors").write_bytes(b"weights")
    name = adapter_name("p1", 2, 0)
    ckpt, receipt = train_lora_with_receipt(
        prev, _examples(3), LoraCfg(), runner, person_id="p1", night=NIGHT, name=name,
        workdir=tmp_path / "work", adapter_cache=cache,
        publish=lambda folder, n: published.append((folder.name, n)), revision=REVISION,
    )
    job, inputs, timeout = runner.calls[0]
    assert job == "train_lora" and timeout > 0
    settings = inputs["config"]
    assert isinstance(settings, dict)
    assert settings["learning_rate"] == 5e-5 and settings["warm_start"] is True
    assert inputs["prev_adapter_dir"] == str(cache / prev.id)
    assert published == [(name, name)]
    assert ckpt.id == name and ckpt.parent_id == prev.id and ckpt.person_id == "p1"
    assert (receipt.steps, receipt.epochs, receipt.gpu_minutes) == (9, 3, 11.0)
    assert (cache / name / "adapter_model.safetensors").exists()


def test_a_clean_slate_trains_at_the_full_rate_from_no_parent(tmp_path: Path) -> None:
    runner = _Runner()
    ckpt, _ = train_lora_with_receipt(
        None, _examples(3), LoraCfg(), runner, person_id="p1", night=NIGHT,
        name=adapter_name("p1", 1, 0), workdir=tmp_path / "work", adapter_cache=tmp_path / "a",
        publish=lambda folder, n: None, revision=REVISION,
    )
    settings = runner.calls[0][1]["config"]
    assert isinstance(settings, dict)
    assert settings["learning_rate"] == 2e-4 and settings["warm_start"] is False
    assert runner.calls[0][1]["prev_adapter_dir"] is None
    assert ckpt.parent_id is None


def test_training_script_parses_and_builds_its_config_offline() -> None:
    source = Path(colab_train.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])}
    assert not top & {"torch", "transformers", "peft", "trl", "bitsandbytes", "anjo"}
    assert colab_train.MODEL == config.MODEL
    plan = plan_steps(12, LoraCfg())
    raw = json.dumps({
        "r": 16, "alpha": 32, "dropout": 0.05, "target_modules": list(DEFAULT_TARGET_MODULES),
        "learning_rate": 2e-4, "seed": 7, "max_seq_length": 8192, "packing": False,
        "accumulation": plan.accumulation, "epochs": plan.epochs, "batch_size": 1,
        "warm_start": False, "revision": "a" * 40, "objective": "sft",
    })
    settings = colab_train.build_settings(colab_train.load_config(raw))
    assert settings["quantization"] == {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_storage": "bfloat16"}
    assert settings["lora"]["r"] == 16 and settings["lora"]["lora_alpha"] == 32
    assert settings["lora"]["target_modules"] == list(DEFAULT_TARGET_MODULES)
    sft = settings["sft"]
    assert sft["max_length"] == 8192 and sft["packing"] is False
    assert sft["completion_only_loss"] is True and sft["bf16"] is True
    assert sft["gradient_accumulation_steps"] == plan.accumulation
    assert sft["num_train_epochs"] == plan.epochs and sft["seed"] == 7
    with pytest.raises(ValueError):
        colab_train.load_config(json.dumps({"r": 64}))


def test_training_script_imports_without_the_gpu_stack() -> None:
    script = (
        "import json, sys\nimport anjo.sleep.train.colab_train, anjo.sleep.train.lora\n"
        "print(json.dumps(sorted(m for m in ('torch', 'transformers', 'peft', 'trl') if m in sys.modules)))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []


@pytest.mark.parametrize("revision", [None, "", "  "])
def test_a_missing_revision_is_refused_before_the_runner_starts(tmp_path: Path, revision: str | None) -> None:
    runner = _Runner()
    with pytest.raises(TrainError, match="revision"):
        train_lora_with_receipt(None, _examples(3), LoraCfg(), runner, person_id="p1", night=NIGHT,
                                name=adapter_name("p1", 1, 0), workdir=tmp_path / "work",
                                adapter_cache=tmp_path / "a", publish=lambda folder, n: None, revision=revision)
    assert runner.calls == [] and not (tmp_path / "work" / DATASET_FILE).exists()


def test_the_plain_text_dataset_is_deleted_whatever_the_run_did(tmp_path: Path) -> None:
    class Failing(_Runner):
        def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
            assert Path(str(inputs["dataset_path"])).exists()  # it is there while the job runs
            raise RuntimeError("the VM went away")

    for runner in (_Runner(), Failing()):
        try:
            train_lora_with_receipt(None, _examples(3), LoraCfg(), runner, person_id="p1", night=NIGHT,
                                    name=adapter_name("p1", 1, len(runner.calls)), workdir=tmp_path / "work",
                                    adapter_cache=tmp_path / "a", publish=lambda folder, n: None, revision=REVISION)
        except RuntimeError:
            pass
        assert not (tmp_path / "work" / DATASET_FILE).exists()
