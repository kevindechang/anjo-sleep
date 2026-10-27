# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The training stack for Glimmer (``muse_glimmer``): one set of pins every runner installs, a LoRA
that targets only the text decoder, the library settings of the pinned versions, and the smoke run's
offline checks. Nothing here imports torch, transformers, peft or trl."""
from __future__ import annotations

import ast
import json
import re
import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.train import colab_train, smoke
from anjo.sleep.train.lora import DATASET_FILE, DEFAULT_TARGET_MODULES, JOB, LoraCfg, adapter_name
from anjo.sleep.train.runners import colab

# Glimmer's linear modules, one of each kind (MuseGlimmerForConditionalGeneration, transformers 5.21)
TEXT_DECODER = [f"model.language_model.layers.{layer}.{module}" for layer in (0, 51) for module in (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.gate_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
ELSEWHERE = [
    "model.vision_tower.layers.0.attn.q_proj", "model.vision_tower.layers.49.attn.k_proj",
    "model.vision_tower.layers.0.attn.v_proj", "model.vision_tower.layers.0.attn.proj",
    "model.vision_tower.layers.0.mlp.fc1", "model.vision_tower.patch_embedder.patch_embedding",
    "model.vision_adapter.fc1", "model.vision_projection", "lm_head",
]


def test_the_pins_live_in_the_training_script_and_name_glimmers_transformers() -> None:
    assert colab.TRAIN_PACKAGES is colab_train.TRAIN_PACKAGES
    pins = dict(pin.split("==") for pin in colab_train.TRAIN_PACKAGES)
    assert set(pins) == {"transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"}
    major, minor, _ = (int(part) for part in pins["transformers"].split("."))
    assert (major, minor) >= (5, 15)  # the first release with muse_glimmer
    assert re.fullmatch(r"torch==2\.\d+\.\d+", colab_train.TORCH)  # in the set's range, >=2.6,<3
    assert smoke.pins() == {**pins, "torch": colab_train.TORCH.split("==")[1]}


def test_the_colab_runner_installs_the_pins_and_never_names_a_smoke_model(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[3] == "download" and Path(args[-1]).name == colab_train.RECEIPT:
            Path(args[-1]).write_text(json.dumps({"steps": 1}))
        elif args[3] == "download":
            Path(args[-1]).write_bytes(b"{}")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    work = tmp_path / "work"
    work.mkdir()
    (work / DATASET_FILE).write_text("{}\n")
    runner = colab.ColabRunner(cli=colab.ColabCli(Path("/fake/colab"), run=fake), ledger=Ledger(tmp_path / "b.jsonl"),
                               is_synthetic=lambda uid: True)
    runner.run(JOB, {"person_id": "synthetic-1", "adapter_name": adapter_name("p1", 1, 0), "night": "2026-10-28",
                     "dataset_path": str(work / DATASET_FILE), "workdir": str(work), "config": {"r": 16},
                     "prev_adapter_dir": None}, timeout_s=7200.0)
    install = next(args for args in calls if args[3] == "install")
    assert install[-len(colab_train.TRAIN_PACKAGES):] == list(colab_train.TRAIN_PACKAGES)
    run = next(args for args in calls if args[3] == "exec")
    assert not any(colab_train.SMOKE_ENV in part for part in run)


def test_the_lora_targets_the_text_decoder_and_never_the_vision_tower() -> None:
    pattern = colab_train.lora_targets(DEFAULT_TARGET_MODULES)
    assert all(re.fullmatch(pattern, name) for name in TEXT_DECODER)
    assert not any(re.fullmatch(pattern, name) for name in ELSEWHERE)
    # PEFT reads a list as suffixes, which would catch the vision tower's q_proj, k_proj and v_proj
    caught = [name for name in ELSEWHERE if any(name.endswith(f".{target}") for target in DEFAULT_TARGET_MODULES)]
    assert len(caught) == 3
    for bad in ([], ["q_proj|.*"], ["Q_proj"], ["q_proj", ""]):
        with pytest.raises(ValueError):
            colab_train.lora_targets(bad)


def _settings(objective: str) -> dict[str, dict[str, Any]]:
    raw = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": list(DEFAULT_TARGET_MODULES),
           "learning_rate": 2e-4, "seed": 7, "max_seq_length": 8192, "packing": False, "accumulation": 2,
           "epochs": 1, "batch_size": 1, "warm_start": False, "revision": "a" * 40, "objective": objective,
           "dpo_beta": 0.1, "dpo_learning_rate": 5e-7, "min_optimizer_steps": 8}
    return colab_train.build_settings(colab_train.load_config(json.dumps(raw)))


def test_settings_use_the_pinned_libraries_arguments_and_the_smoke_run_only_moves_to_the_cpu() -> None:
    settings = _settings("sft_dpo")
    for trainer in ("sft", "dpo"):
        assert settings[trainer]["warmup_steps"] == 0.03 and "warmup_ratio" not in settings[trainer]
        assert settings[trainer]["bf16"] is True and "use_cpu" not in settings[trainer]
    assert settings["lora"]["task_type"] == "CAUSAL_LM"
    cpu = colab_train.smoke_settings(settings)
    assert settings["sft"]["bf16"] is True  # the night's settings are untouched
    for trainer in ("sft", "dpo"):
        assert cpu[trainer] == {**settings[trainer], "bf16": False, "use_cpu": True}
    assert cpu["lora"] == settings["lora"] and cpu["quantization"] == settings["quantization"]
    assert "dpo" not in colab_train.smoke_settings(_settings("sft"))


def test_the_script_loads_glimmer_as_image_text_to_text() -> None:
    tree = ast.parse(Path(colab_train.__file__).read_text(encoding="utf-8"))
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "AutoModelForImageTextToText" in names and "AutoModelForCausalLM" not in names


class _Tokenizer:
    """transformers 5's default: tokenizing returns a dict unless ``return_dict=False``."""

    def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> object:
        ids = [0] * sum(len(message["content"].split()) for message in messages)
        return ids if kw.get("return_dict") is False else {"input_ids": ids, "attention_mask": ids}


def test_rows_are_measured_in_tokens_not_in_dict_keys() -> None:
    prompt = [{"role": "user", "content": "q"}]
    long = {"prompt": prompt, "completion": [{"role": "assistant", "content": "word " * 9000}]}
    short = {"prompt": prompt, "completion": [{"role": "assistant", "content": "a b"}]}
    pair = {"prompt": prompt, "chosen": short["completion"], "rejected": long["completion"]}
    fits = [colab_train._fits(_Tokenizer(), row, 8192) for row in (long, short, pair)]
    assert fits == [False, True, False]


def test_a_smoke_model_that_is_not_a_folder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(colab_train.SMOKE_ENV, raising=False)
    assert colab_train._smoke_model() is None
    monkeypatch.setenv(colab_train.SMOKE_ENV, str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        colab_train._smoke_model()
    monkeypatch.setenv(colab_train.SMOKE_ENV, str(tmp_path))
    assert colab_train._smoke_model() == str(tmp_path)


def _adapter(folder: Path, keys: list[str], target_modules: object) -> Path:
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 16, "target_modules": target_modules}))
    header = json.dumps({key: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
                         for i, key in enumerate(keys)}).encode()
    (folder / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(4 * len(keys)))
    return folder


def _keys(layers: int, extra: Sequence[str] = ()) -> list[str]:
    modules = [f"model.language_model.layers.{layer}.{module}" for layer in range(layers) for module in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.gate_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    return [f"base_model.model.{module}.lora_{side}.weight" for module in (*modules, *extra) for side in "AB"]


def test_the_smoke_runs_adapter_check_refuses_anything_outside_the_text_decoder(tmp_path: Path) -> None:
    target = colab_train.lora_targets(DEFAULT_TARGET_MODULES)
    assert smoke.check_adapter(_adapter(tmp_path / "ok", _keys(4), target), layers=4, r=16) == 32
    bad = [
        _adapter(tmp_path / "vision", _keys(4, ["model.vision_tower.layers.0.attn.q_proj"]), target),
        _adapter(tmp_path / "list", _keys(4), list(DEFAULT_TARGET_MODULES)),
        _adapter(tmp_path / "short", _keys(3), target),
    ]
    for folder in bad:
        with pytest.raises(smoke.SmokeError):
            smoke.check_adapter(folder, layers=4, r=16)


def test_the_smoke_run_checks_the_pins_and_trains_on_rows_that_must_be_dropped() -> None:
    exact = smoke.pins()
    assert smoke.pin_mismatches(exact) == []
    assert smoke.pin_mismatches({**exact, "torch": f"{exact['torch']}+cpu"}) == []
    assert smoke.pin_mismatches({**exact, "trl": "0.24.0", "peft": None}) == [
        f"peft=={exact['peft']} (found None)", f"trl=={exact['trl']} (found 0.24.0)"]
    data = smoke.smoke_dataset()
    rows = [json.loads(line) for line in data.to_jsonl(pairs=True).splitlines()]
    sft, pairs = colab_train.split_rows(rows)
    assert (len(sft), len(pairs)) == (5, 3)
    long = [row for row in rows if len(json.dumps(row).split()) > smoke.LONG_WORDS]
    assert len(long) == 2 and LoraCfg().max_seq_length < smoke.LONG_WORDS
