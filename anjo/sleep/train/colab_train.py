"""The training script the GPU runner sends to its VM (Task 5): TRL SFT with a PEFT QLoRA adapter.

It is standalone on purpose. The Colab runner executes this one file on the VM (``colab exec -f``),
so it imports nothing from Anjo, and torch, transformers, peft, trl and datasets load only inside
``main``. Importing it on the night's host loads none of them, and ``build_settings`` turns the
night's recipe into the library settings offline, which the tests check.

The recipe (``anjo.sleep.train.lora``): Glimmer in 4-bit NF4 with double quantization and bf16
compute, a LoRA of rank r (at most 32) and alpha 2r on the attention and MLP projections, the loss
on the completion (her reply) only, no packing, and every row longer than ``max_seq_length``
dropped and counted rather than cut. A warm start loads the previous adapter as trainable.

The ``sft_dpo`` objective (Task 7) adds a DPO pass after SFT on the dataset's pair rows (prompt,
chosen, rejected), at Meta's TRL settings (beta 0.1, lr 5e-7). The trained PEFT model goes in with
``ref_model=None``, so TRL's reference is the same model with the adapter switched off and no
second model loads. Pairs longer than ``max_seq_length`` on either side are dropped and counted.

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
OBJECTIVES = ("sft", "sft_dpo")
MAX_DPO_LR = 1e-5
_KEYS = (
    "r", "alpha", "dropout", "target_modules", "learning_rate", "seed", "max_seq_length", "packing",
    "accumulation", "epochs", "batch_size", "warm_start", "revision", "objective",
)


def _int(cfg: dict[str, Any], key: str, low: int, high: int) -> int:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return value


def _rate(cfg: dict[str, Any], key: str, high: float) -> float:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= high:
        raise ValueError(f"{key} must be a number in (0, {high:g}]")
    return float(value)


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
    if cfg["packing"] is not False or cfg["objective"] not in OBJECTIVES:
        raise ValueError(f"packing stays off and the objective is one of {', '.join(OBJECTIVES)}")
    if cfg["objective"] == "sft_dpo":
        _rate(cfg, "dpo_beta", 1.0)
        _rate(cfg, "dpo_learning_rate", MAX_DPO_LR)
    if not isinstance(cfg["revision"], str) or not cfg["revision"]:
        raise ValueError("the base model's revision is pinned")
    return cfg


def build_settings(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The library settings for ``cfg``: bitsandbytes, PEFT's LoraConfig and TRL's SFTConfig, and
    TRL's DPOConfig for ``sft_dpo``."""
    settings = {
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
    if cfg["objective"] == "sft_dpo":
        settings["dpo"] = {
            "beta": cfg["dpo_beta"], "learning_rate": cfg["dpo_learning_rate"], "max_length": cfg["max_seq_length"],
            "bf16": True, "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["accumulation"], "num_train_epochs": 1,
            "seed": cfg["seed"], "data_seed": cfg["seed"], "gradient_checkpointing": True,
            "precompute_ref_log_probs": False, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
            "logging_steps": 1, "save_strategy": "no", "report_to": "none",
        }
    return settings


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The SFT rows (prompt, completion) and the pair rows (prompt, chosen, rejected)."""
    return [row for row in rows if "completion" in row], [row for row in rows if "chosen" in row]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fits(tokenizer: Any, row: dict[str, Any], max_length: int) -> bool:
    sides = [row["completion"]] if "completion" in row else [row["chosen"], row["rejected"]]
    return all(len(tokenizer.apply_chat_template(row["prompt"] + side, tokenize=True)) <= max_length for side in sides)


def _dpo(model: Any, tokenizer: Any, pairs: list[dict[str, Any]], settings: dict[str, Any], out: Path) -> int:  # pragma: no cover - GPU VM only
    """The DPO pass on the night's pairs; its reference is ``model`` with the adapter switched off."""
    from datasets import Dataset  # type: ignore[import-not-found, unused-ignore]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-not-found, unused-ignore]

    data = Dataset.from_list([{k: row[k] for k in ("prompt", "chosen", "rejected")} for row in pairs])
    trainer = DPOTrainer(model=model, ref_model=None, args=DPOConfig(output_dir=str(out / "dpo"), **settings),
                         train_dataset=data, processing_class=tokenizer)
    trainer.train()
    return int(trainer.state.global_step)


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
    rows, pair_rows = split_rows(_rows(Path(os.environ["ANJO_TRAIN_DATASET"])))
    kept = [row for row in rows if _fits(tokenizer, row, cfg["max_seq_length"])]
    pairs = [row for row in pair_rows if _fits(tokenizer, row, cfg["max_seq_length"])] if "dpo" in settings else []
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
    dpo_steps = _dpo(trainer.model, tokenizer, pairs, settings["dpo"], out) if pairs else 0
    minutes = (time.monotonic() - started) / 60.0
    trainer.model.save_pretrained(str(out))
    receipt = {
        "steps": int(trainer.state.global_step), "epochs": cfg["epochs"], "train_minutes": minutes,
        "examples": len(kept), "dropped_truncated": len(rows) - len(kept),
        "gpu": torch.cuda.get_device_name(0), "objective": cfg["objective"], "dpo_steps": dpo_steps,
        "pairs": len(pairs), "pairs_dropped_truncated": len(pair_rows) - len(pairs) if "dpo" in settings else 0,
    }
    (out / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - the VM's kernel runs the file as __main__
    main()
