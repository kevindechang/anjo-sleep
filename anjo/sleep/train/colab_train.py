"""The training script the GPU runner sends to its VM (Task 5): TRL SFT with a PEFT QLoRA adapter.

It is standalone on purpose. The Colab runner executes this one file on the VM (``colab exec -f``),
so it imports nothing from Anjo, and torch, transformers, peft, trl and datasets load only inside
``main``. Importing it on the night's host loads none of them, and ``build_settings`` turns the
night's recipe into the library settings offline, which the tests check.

The recipe (``anjo.sleep.train.lora``): Glimmer in 4-bit NF4 with double quantization and bf16
compute, a LoRA of rank r (at most 32) and alpha 2r on the attention and MLP projections, the loss
on the completion (her reply) only, no packing, and every row longer than ``max_seq_length``
dropped and counted rather than cut. A warm start loads the previous adapter as trainable.

Glimmer is multimodal (``MuseGlimmerForConditionalGeneration``, which transformers maps under
``AutoModelForImageTextToText``, not ``AutoModelForCausalLM``). Its vision tower has ``q_proj``,
``k_proj`` and ``v_proj`` too, and PEFT matches a list of names as suffixes anywhere, so the LoRA's
targets are a pattern anchored to the text decoder's layers (``lora_targets``). In Glimmer the name
``gate_proj`` is both the MLP's gate and the attention's output gate, and both are trained. The
vision tower is quantized with the rest: training is text only and never runs it.

The ``sft_dpo`` objective (Task 7) adds a DPO pass after SFT on the dataset's pair rows (prompt,
chosen, rejected), at Meta's TRL settings (beta 0.1, lr 5e-7). The trained PEFT model goes in with
``ref_model=None``, so no second model loads. TRL 1.x would copy the adapter as a frozen ``ref``
adapter and use that copy as the reference; the copy is removed, so the reference is the same model
with the adapter switched off. Pairs longer than ``max_seq_length`` on either side are dropped and
counted.

The runner sets four environment variables: ``ANJO_TRAIN_CONFIG`` (the recipe as JSON),
``ANJO_TRAIN_DATASET`` (the dataset file), ``ANJO_TRAIN_OUT`` (where the adapter and
``receipt.json`` go) and, for a warm start, ``ANJO_TRAIN_PREV`` (the previous adapter's folder).
The receipt holds counts and timings only. A fifth, ``ANJO_TRAIN_SMOKE_MODEL``, is set only by the
CPU smoke run (``anjo.sleep.train.smoke``) and never by a runner: it names a tiny local model, which
loads unquantized in fp32 on the CPU in place of Glimmer.

``TRAIN_PACKAGES`` are the versions every runner installs (Colab's image brings its own torch), and
``TORCH`` the torch for a runner that brings none. They live here, in a file that imports only the
standard library, so a runner that copies this one file gets the pins with it.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# transformers 5.15 is the first release with Glimmer's architecture (``muse_glimmer``). The set is the
# newest whose PyPI metadata agrees: peft and trl bound transformers only from below, and trl needs
# accelerate >=1.10.1 and datasets >=4.7.
TRAIN_PACKAGES = (
    "transformers==5.21.0", "peft==0.21.2", "trl==1.16.0", "bitsandbytes==0.50.2",
    "accelerate==1.15.0", "datasets==5.1.0",
)
# The set needs torch >=2.6,<3 (transformers' and bitsandbytes' bounds). Colab's GPU image has torch
# 2.11.0+cu130 (googlecolab/backend-info, 2026-10-26), and a runner without one installs the same.
TORCH = "torch==2.11.0"
MODEL = "meta-models/Muse-Glimmer-30B"
MAX_RANK = 32
MIN_SEQ_LENGTH = 8192
RECEIPT = "receipt.json"
OBJECTIVES = ("sft", "sft_dpo")
MAX_DPO_LR = 1e-5
MAX_ACCUMULATION = 16
MAX_EPOCHS = 4
_KEYS = (
    "r", "alpha", "dropout", "target_modules", "learning_rate", "seed", "max_seq_length", "packing",
    "accumulation", "epochs", "batch_size", "warm_start", "revision", "objective",
)
# A 3% linear warmup: transformers 5 dropped ``warmup_ratio``, and ``warmup_steps`` under 1 is the share.
WARMUP = 0.03
SMOKE_ENV = "ANJO_TRAIN_SMOKE_MODEL"
# The text decoder's layers in MuseGlimmerForConditionalGeneration; the vision tower, its adapter and
# projection sit beside them under ``model.vision_*``.
TEXT_DECODER = r"model\.language_model\.layers\.\d+\.(?:self_attn|mlp)"
_MODULE_NAME = re.compile(r"[a-z][a-z0-9_]*")


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
        _int(cfg, "min_optimizer_steps", 1, 10_000)
    if not isinstance(cfg["revision"], str) or not cfg["revision"]:
        raise ValueError("the base model's revision is pinned")
    return cfg


def lora_targets(names: Sequence[str]) -> str:
    """PEFT's ``target_modules`` for the projections ``names`` (q_proj, ..., down_proj): a pattern
    that only the text decoder's attention and MLP match, never a module of the vision tower."""
    if not names or not all(isinstance(name, str) and _MODULE_NAME.fullmatch(name) for name in names):
        raise ValueError("target_modules are module names such as q_proj")
    return rf"{TEXT_DECODER}\.(?:{'|'.join(names)})"


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
            "target_modules": lora_targets(cfg["target_modules"]), "bias": "none", "task_type": "CAUSAL_LM",
        },
        "sft": {
            "max_length": cfg["max_seq_length"], "packing": False, "completion_only_loss": True,
            "bf16": True, "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["accumulation"], "num_train_epochs": cfg["epochs"],
            "learning_rate": cfg["learning_rate"], "seed": cfg["seed"], "data_seed": cfg["seed"],
            "gradient_checkpointing": True, "lr_scheduler_type": "cosine", "warmup_steps": WARMUP,
            "logging_steps": 1, "save_strategy": "no", "report_to": "none",
        },
    }
    if cfg["objective"] == "sft_dpo":
        settings["dpo"] = {
            "beta": cfg["dpo_beta"], "learning_rate": cfg["dpo_learning_rate"], "max_length": cfg["max_seq_length"],
            "bf16": True, "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["accumulation"], "num_train_epochs": 1,
            "seed": cfg["seed"], "data_seed": cfg["seed"], "gradient_checkpointing": True,
            "precompute_ref_log_probs": False, "lr_scheduler_type": "cosine", "warmup_steps": WARMUP,
            "logging_steps": 1, "save_strategy": "no", "report_to": "none",
        }
    return settings


def smoke_settings(settings: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``settings`` for the CPU smoke run: the trainers in fp32 on the CPU, everything else kept."""
    cpu = {key: {**settings[key], "bf16": False, "use_cpu": True} for key in ("sft", "dpo") if key in settings}
    return {**settings, **cpu}


def dpo_plan(n_pairs: int, cfg: dict[str, Any]) -> tuple[int, int]:
    """The DPO pass's accumulation and epochs, by the SFT rule (``lora.plan_steps``): at least
    ``min_optimizer_steps`` updates, accumulation shrinking to ``n // min_optimizer_steps`` (1 to
    16) and, if that still falls short, epochs rising to at most 4."""
    if n_pairs < 1:
        raise ValueError("a DPO pass has at least one pair")
    steps = cfg["min_optimizer_steps"]
    accumulation = max(1, min(MAX_ACCUMULATION, n_pairs // steps))
    per_epoch = math.ceil(n_pairs / (cfg["batch_size"] * accumulation))
    epochs = 1 if per_epoch >= steps else min(MAX_EPOCHS, math.ceil(steps / per_epoch))
    return accumulation, epochs


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The SFT rows (prompt, completion) and the pair rows (prompt, chosen, rejected)."""
    return [row for row in rows if "completion" in row], [row for row in rows if "chosen" in row]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fits(tokenizer: Any, row: dict[str, Any], max_length: int) -> bool:
    sides = [row["completion"]] if "completion" in row else [row["chosen"], row["rejected"]]
    # transformers 5 returns a dict by default when it tokenizes; the length of the ids is wanted
    return all(len(tokenizer.apply_chat_template(row["prompt"] + side, tokenize=True, return_dict=False)) <= max_length
               for side in sides)


def _smoke_model() -> str | None:
    """The tiny local model the smoke run names, or None on a real night. A name that isn't a local
    folder is refused: the smoke run never falls back to anything."""
    smoke = os.environ.get(SMOKE_ENV) or None
    if smoke is not None and not Path(smoke).is_dir():
        raise ValueError(f"{SMOKE_ENV} names a local model folder")
    return smoke


def _load_model(cfg: dict[str, Any], settings: dict[str, Any], smoke: str | None) -> Any:  # pragma: no cover - needs the training stack
    """Glimmer in 4-bit on the GPU or, for the smoke run only, the tiny model in fp32 on the CPU; then
    PEFT's preparation for k-bit training (frozen base, casts, gradient checkpointing)."""
    import torch  # type: ignore[import-not-found, unused-ignore]
    from peft import prepare_model_for_kbit_training  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig  # type: ignore[import-not-found, unused-ignore]

    if smoke is None:
        quant = {**settings["quantization"], "bnb_4bit_compute_dtype": torch.bfloat16,
                 "bnb_4bit_quant_storage": torch.bfloat16}
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, revision=cfg["revision"], quantization_config=BitsAndBytesConfig(**quant),
            dtype=torch.bfloat16, device_map={"": 0},
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(smoke, dtype=torch.float32)
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def _dpo(model: Any, tokenizer: Any, pairs: list[dict[str, Any]], settings: dict[str, Any], out: Path) -> int:  # pragma: no cover - GPU VM only
    """The DPO pass on the night's pairs; its reference is ``model`` with the adapter switched off."""
    from datasets import Dataset  # type: ignore[import-not-found, unused-ignore]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-not-found, unused-ignore]

    data = Dataset.from_list([{k: row[k] for k in ("prompt", "chosen", "rejected")} for row in pairs])
    trainer = DPOTrainer(model=model, ref_model=None, args=DPOConfig(output_dir=str(out / "dpo"), **settings),
                         train_dataset=data, processing_class=tokenizer)
    # TRL 1.x adds a frozen copy of the trained adapter ("ref") and scores the reference with it. With
    # only the trained adapter left, it scores the reference with the adapter switched off.
    for name in [name for name in trainer.model.peft_config if name != "default"]:
        trainer.model.delete_adapter(name)
    trainer.train()
    return int(trainer.state.global_step)


def main() -> int:  # pragma: no cover - runs only on the GPU VM and in the CPU smoke run
    import numpy  # type: ignore[import-not-found, unused-ignore]
    import torch  # type: ignore[import-not-found, unused-ignore]
    from datasets import Dataset  # type: ignore[import-not-found, unused-ignore]
    from peft import LoraConfig, PeftModel  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoTokenizer, set_seed  # type: ignore[import-not-found, unused-ignore]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found, unused-ignore]

    cfg = load_config(os.environ["ANJO_TRAIN_CONFIG"])
    smoke = _smoke_model()
    settings = build_settings(cfg) if smoke is None else smoke_settings(build_settings(cfg))
    out = Path(os.environ["ANJO_TRAIN_OUT"])
    prev = os.environ.get("ANJO_TRAIN_PREV") or None
    random.seed(cfg["seed"])
    numpy.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(smoke or MODEL, revision=None if smoke else cfg["revision"])
    rows, pair_rows = split_rows(_rows(Path(os.environ["ANJO_TRAIN_DATASET"])))
    kept = [row for row in rows if _fits(tokenizer, row, cfg["max_seq_length"])]
    pairs = [row for row in pair_rows if _fits(tokenizer, row, cfg["max_seq_length"])] if "dpo" in settings else []
    model = _load_model(cfg, settings, smoke)
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
    dpo_steps = 0
    if pairs:
        accumulation, epochs = dpo_plan(len(pairs), cfg)
        dpo_settings = {**settings["dpo"], "gradient_accumulation_steps": accumulation, "num_train_epochs": epochs}
        dpo_steps = _dpo(trainer.model, tokenizer, pairs, dpo_settings, out)
    minutes = (time.monotonic() - started) / 60.0
    trainer.model.save_pretrained(str(out))
    receipt = {
        "steps": int(trainer.state.global_step), "epochs": cfg["epochs"], "train_minutes": minutes,
        "examples": len(kept), "dropped_truncated": len(rows) - len(kept),
        "gpu": torch.cuda.get_device_name(0) if smoke is None else "cpu", "objective": cfg["objective"],
        "dpo_steps": dpo_steps, "pairs": len(pairs),
        "pairs_dropped_truncated": len(pair_rows) - len(pairs) if "dpo" in settings else 0,
    }
    (out / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - the VM's kernel runs the file as __main__
    main()
