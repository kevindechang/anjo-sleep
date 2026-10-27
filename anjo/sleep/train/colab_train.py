"""The training script the GPU runner sends to its VM (Task 5): TRL SFT with a PEFT QLoRA adapter.

It is standalone on purpose. The Colab runner executes this one file on the VM (``colab exec -f``),
so it imports nothing from Anjo, and torch, transformers, peft, trl and datasets load only inside
``main``. Importing it on the night's host loads none of them, and ``build_settings`` turns the
night's recipe into the library settings offline, which the tests check.

The recipe (``anjo.sleep.train.lora``): Glimmer in 4-bit NF4 with double quantization and bf16
compute, a LoRA of rank r (at most 32) and alpha 2r on the attention and MLP projections, the loss
on the completion (her reply) only, no packing, and every row longer than ``max_seq_length``
dropped and counted rather than cut. A warm start loads the previous adapter as trainable.

The runner sets four environment variables: ``ANJO_TRAIN_CONFIG`` (the recipe as JSON),
``ANJO_TRAIN_DATASET`` (the dataset file), ``ANJO_TRAIN_OUT`` (where the adapter and
``receipt.json`` go) and, for a warm start, ``ANJO_TRAIN_PREV`` (the previous adapter's folder).
The receipt holds counts and timings only.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

MODEL = "meta-models/Muse-Glimmer-30B"
MAX_RANK = 32
MIN_SEQ_LENGTH = 8192
RECEIPT = "receipt.json"
_KEYS = (
    "r", "alpha", "dropout", "target_modules", "learning_rate", "seed", "max_seq_length", "packing",
    "accumulation", "epochs", "batch_size", "warm_start", "revision", "objective",
)


def _int(cfg: dict[str, Any], key: str, low: int, high: int) -> int:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return value


def load_config(raw: str) -> dict[str, Any]:
    """The recipe the runner sent, checked: a rank over the server's cap is refused first."""
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise ValueError("the training config is a JSON object")
    r = _int(cfg, "r", 1, MAX_RANK)
    missing = [key for key in _KEYS if key not in cfg]
    if missing:
        raise ValueError(f"the training config lacks {', '.join(missing)}")
    if cfg["alpha"] != 2 * r:
        raise ValueError("alpha is 2r")
    _int(cfg, "max_seq_length", MIN_SEQ_LENGTH, 131_072)
    _int(cfg, "accumulation", 1, 16)
    _int(cfg, "epochs", 1, 4)
    _int(cfg, "batch_size", 1, 1)
    if cfg["packing"] is not False or cfg["objective"] != "sft":
        raise ValueError("packing stays off and the objective is sft")
    if not isinstance(cfg["revision"], str) or not cfg["revision"]:
        raise ValueError("the base model's revision is pinned")
    return cfg


def build_settings(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The library settings for ``cfg``: bitsandbytes, PEFT's LoraConfig and TRL's SFTConfig."""
    return {
        "quantization": {
            "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_storage": "bfloat16",
        },
        "lora": {
            "r": cfg["r"], "lora_alpha": cfg["alpha"], "lora_dropout": cfg["dropout"],
            "target_modules": list(cfg["target_modules"]), "bias": "none", "task_type": "CAUSAL_LM",
        },
        "sft": {
            "max_length": cfg["max_seq_length"], "packing": False, "completion_only_loss": True,
            "bf16": True, "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["accumulation"], "num_train_epochs": cfg["epochs"],
            "learning_rate": cfg["learning_rate"], "seed": cfg["seed"], "data_seed": cfg["seed"],
            "gradient_checkpointing": True, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
            "logging_steps": 1, "save_strategy": "no", "report_to": "none",
        },
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fits(tokenizer: Any, row: dict[str, Any], max_length: int) -> bool:
    ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
    return len(ids) <= max_length


def main() -> int:  # pragma: no cover - runs only on the GPU VM
    import numpy  # type: ignore[import-not-found, unused-ignore]
    import torch  # type: ignore[import-not-found, unused-ignore]
    from datasets import Dataset  # type: ignore[import-not-found, unused-ignore]
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed  # type: ignore[import-not-found, unused-ignore]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found, unused-ignore]

    cfg = load_config(os.environ["ANJO_TRAIN_CONFIG"])
    settings = build_settings(cfg)
    out = Path(os.environ["ANJO_TRAIN_OUT"])
    prev = os.environ.get("ANJO_TRAIN_PREV") or None
    random.seed(cfg["seed"])
    numpy.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=cfg["revision"])
    rows = _rows(Path(os.environ["ANJO_TRAIN_DATASET"]))
    kept = [row for row in rows if _fits(tokenizer, row, cfg["max_seq_length"])]
    quant = {**settings["quantization"], "bnb_4bit_compute_dtype": torch.bfloat16,
             "bnb_4bit_quant_storage": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=cfg["revision"], quantization_config=BitsAndBytesConfig(**quant),
        torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = None
    if prev:
        model = PeftModel.from_pretrained(model, prev, is_trainable=True)
    else:
        peft_config = LoraConfig(**settings["lora"])
    data = Dataset.from_list([{"prompt": row["prompt"], "completion": row["completion"]} for row in kept])
    trainer = SFTTrainer(
        model=model, args=SFTConfig(output_dir=str(out / "run"), **settings["sft"]),
        train_dataset=data, processing_class=tokenizer, peft_config=peft_config,
    )
    started = time.monotonic()
    trainer.train()
    minutes = (time.monotonic() - started) / 60.0
    trainer.model.save_pretrained(str(out))
    receipt = {
        "steps": int(trainer.state.global_step), "epochs": cfg["epochs"], "train_minutes": minutes,
        "examples": len(kept), "dropped_truncated": len(rows) - len(kept),
        "gpu": torch.cuda.get_device_name(0),
    }
    (out / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - the VM's kernel runs the file as __main__
    main()
