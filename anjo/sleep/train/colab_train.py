"""The training script the GPU runner sends to its VM (Task 5): TRL SFT with a PEFT QLoRA adapter.

It is standalone on purpose. The Colab runner executes this one file on the VM (``colab exec -f``),
so it imports nothing from Anjo, and torch, transformers, peft, trl and datasets load only inside
``main``. Importing it on the night's host loads none of them, and ``build_settings`` turns the
night's recipe into the library settings offline, which the tests check.

The recipe (``anjo.sleep.train.lora``): Glimmer in 4-bit NF4 with double quantization and bf16
compute, a LoRA of rank r (at most 32) and alpha 2r on the attention and MLP projections, the loss
on the completion (her reply) only, no packing, and every row longer than ``max_seq_length``
dropped and counted rather than cut; a recipe with ``keep_every_row`` (Task 17b) instead stops
before the model loads if any row would be dropped (``check_every_row_kept``). A warm start loads the previous adapter as trainable.

**Train as served.** Each row goes to TRL as plain text, rendered the way the server renders a
request: Glimmer's chat template at the night's ``reasoning_strength`` and ``current_date`` (or the
row's own ``current_date``, for replay's rows, each served on its own night) with the
generation prompt (``<|start|>assistant``), then the reply's channel header ``" to=user<|message|>"``
on the prompt side, and the reply closed by ``<|eot|>``. So the loss falls on the reply and
``<|eot|>`` only. (TRL's conversational form would train the header too.) The host cleaned every
text as the server's prompts are (``anjo.sleep.serving.framing``), and a row that still carries a
special token's text, or the template's reasoning phrase in its system turn, stops the run. Before a
step is taken, the tokenizer's chat template is checked against the recipe's ``template_sha256``,
and every row's tokens, as TRL prepared them, against the template's own render (``_check_tokens``).
The adapter records the recipe's ``rendering`` in its ``adapter_config.json`` (``anjo_rendering``),
which vLLM and PEFT ignore and the host checks.

Glimmer is multimodal (``MuseGlimmerForConditionalGeneration``, which transformers maps under
``AutoModelForImageTextToText``, not ``AutoModelForCausalLM``). Its vision tower has ``q_proj``,
``k_proj`` and ``v_proj`` too, and PEFT matches a list of names as suffixes anywhere, so the LoRA's
targets are a pattern anchored to the text decoder's layers (``lora_targets``). In Glimmer the name
``gate_proj`` is both the MLP's gate and the attention's output gate, and both are trained. The
vision tower is quantized with the rest: training is text only and never runs it.

The ``sft_dpo`` objective (Task 7) adds a DPO pass after SFT on the dataset's pair rows (prompt,
chosen, rejected), at Meta's TRL settings (beta 0.1, lr 5e-6, the LoRA scale). The trained PEFT model goes in with
``ref_model=None``, so no second model loads. TRL 1.x would copy the adapter as a frozen ``ref``
adapter and use that copy as the reference; the copy is removed, so the reference is the same model
with the adapter switched off; the receipt's ``dpo_reference`` says which reference TRL used.
Pairs longer than ``max_seq_length`` on either side are dropped and counted.

The ``sft_dpo_seed`` objective (2026-10-29) trains no TRL pass: ``sleep_loss.py``, which ships beside
this file (Modal's image only; the Colab runner refuses the objective), runs the one weighted loss over
the SFT rows, the pairs and the dataset's ``kind`` rows (seeding, dreams, anchors), each tokenized as
the server renders it (``served_ids``). Its recipe carries ``terms``: ``fold_rank`` (fold the night into
the long-term adapter, ``ANJO_TRAIN_PREV``, at that rank) and ``dreams``. ``match_minutes``, the live
comparison's control, lets today's ``sft_dpo`` keep training SFT until the job has used that many minutes.

The runner sets four environment variables: ``ANJO_TRAIN_CONFIG`` (the recipe as JSON),
``ANJO_TRAIN_DATASET`` (the dataset file), ``ANJO_TRAIN_OUT`` (where the adapter and
``receipt.json`` go) and, for a warm start, ``ANJO_TRAIN_PREV`` (the previous adapter's folder).
The receipt holds counts and timings only. The Modal job (``modal_job.py``) also sets
``ANJO_TRAIN_BASE``, the folder on its weights volume that holds the pinned revision: Glimmer and its
tokenizer load from there, with nothing downloaded, and the saved adapter names Glimmer as its base,
as one loaded by name does (``record_base_model``). A sixth, ``ANJO_TRAIN_SMOKE_MODEL``, is set only by the
CPU smoke run (``anjo.sleep.train.smoke``) and never by a runner: it names a tiny local model, which
loads unquantized in fp32 on the CPU in place of Glimmer, and the run then also writes the prompt ids
TRL prepared for each row (``smoke_tokens.json``) for the smoke to hold against the server's.

``TRAIN_PACKAGES`` are the versions every runner installs (Colab's image brings its own torch), and
``TORCH`` the torch for a runner that brings none. They live here, in a file that imports only the
standard library, so a runner that copies this one file gets the pins with it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Mapping, Sequence
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
OBJECTIVES = ("sft", "sft_dpo", "sft_dpo_seed")
SEED_OBJECTIVE = "sft_dpo_seed"
LOSS_FILE = "sleep_loss.py"  # sft_dpo_seed's loss, beside this file
KINDS = ("seed", "dream", "anchor")
MAX_MATCH_MINUTES = 100.0
MAX_DPO_LR = 1e-5
MAX_ACCUMULATION = 16
MAX_EPOCHS = 4
_KEYS = (
    "r", "alpha", "dropout", "target_modules", "learning_rate", "seed", "max_seq_length", "packing",
    "accumulation", "epochs", "batch_size", "warm_start", "revision", "objective", "reasoning_strength",
    "current_date", "template_sha256", "rendering",
)
# Standalone copies of anjo.sleep.serving.config's and framing's (a test holds them equal): Glimmer's
# reasoning levels, the reply's channel header after the generation prompt, the end of its turn,
# every special token's text, the template's reasoning phrase, and where an adapter records its rendering.
REASONING_LEVELS = ("low", "medium", "high", "xhigh")
USER_CHANNEL = " to=user<|message|>"
REPLY_END = "<|eot|>"
SPECIAL_TOKEN = re.compile(
    r"<\|(?:reserved_special_token_[0-9]{1,4}|vid_frame_separator|finetune_right_pad|begin_of_text|end_of_text"
    r"|image_start|image_end|vid_start|message|vid_end|start|image|video|patch|eom|eot)\|>")
REASONING_PHRASE = re.compile(r"(?i)(reasoning)\s+(strength|effort)")
RENDERING_KEY = "anjo_rendering"
SMOKE_TOKENS = "smoke_tokens.json"
# A 3% linear warmup: transformers 5 dropped ``warmup_ratio``, and ``warmup_steps`` under 1 is the share.
WARMUP = 0.03
SMOKE_ENV = "ANJO_TRAIN_SMOKE_MODEL"
BASE_ENV = "ANJO_TRAIN_BASE"  # the Modal job's local copy of the pinned revision
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
    if "max_steps" in cfg:
        _int(cfg, "max_steps", 1, 100_000)
    if not isinstance(cfg.get("keep_every_row", False), bool):
        raise ValueError("keep_every_row is true or false")
    if cfg["objective"] in ("sft_dpo", SEED_OBJECTIVE):
        _rate(cfg, "dpo_beta", 1.0)
        _rate(cfg, "dpo_learning_rate", MAX_DPO_LR)
        _int(cfg, "min_optimizer_steps", 1, 10_000)
    _check_terms(cfg)
    if not isinstance(cfg["revision"], str) or not cfg["revision"]:
        raise ValueError("the base model's revision is pinned")
    if cfg["reasoning_strength"] not in REASONING_LEVELS:
        raise ValueError(f"reasoning_strength is one of {', '.join(REASONING_LEVELS)}")
    if not isinstance(cfg["current_date"], str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", cfg["current_date"]):
        raise ValueError("current_date is the night's date, YYYY-MM-DD")
    if not isinstance(cfg["template_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", cfg["template_sha256"]):
        raise ValueError("template_sha256 is the chat template's sha256")
    if not isinstance(cfg["rendering"], str) or not cfg["rendering"].endswith("/" + cfg["reasoning_strength"]):
        raise ValueError("the rendering names the reasoning strength the rows are rendered at")
    return cfg


def _check_terms(cfg: dict[str, Any]) -> None:
    """``terms`` comes with ``sft_dpo_seed`` and only then; ``match_minutes`` only without it."""
    seeded, terms = cfg["objective"] == SEED_OBJECTIVE, cfg.get("terms")
    if seeded != (terms is not None):
        raise ValueError(f"terms come with the {SEED_OBJECTIVE} objective, and only with it")
    if seeded:
        fold = terms.get("fold_rank") if isinstance(terms, dict) else 0
        if (not isinstance(terms, dict) or set(terms) != {"fold_rank", "dreams"} or not isinstance(terms["dreams"], bool)
                or (fold is not None and (isinstance(fold, bool) or not isinstance(fold, int)
                                          or not cfg["r"] <= fold <= MAX_RANK))):
            raise ValueError(f"terms are fold_rank (none, or from r to {MAX_RANK}) and dreams (true or false)")
    if "match_minutes" in cfg and (seeded or isinstance(cfg["match_minutes"], bool)
                                   or not isinstance(cfg["match_minutes"], int | float)
                                   or not 0 < cfg["match_minutes"] <= MAX_MATCH_MINUTES):
        raise ValueError(f"match_minutes pads today's objective only, up to {MAX_MATCH_MINUTES:g} minutes")


def lora_targets(names: Sequence[str]) -> str:
    """PEFT's ``target_modules`` for the projections ``names`` (q_proj, ..., down_proj): a pattern
    that only the text decoder's attention and MLP match, never a module of the vision tower."""
    if not names or not all(isinstance(name, str) and _MODULE_NAME.fullmatch(name) for name in names):
        raise ValueError("target_modules are module names such as q_proj")
    return rf"{TEXT_DECODER}\.(?:{'|'.join(names)})"


def build_settings(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The library settings for ``cfg``: bitsandbytes, PEFT's LoraConfig and TRL's SFTConfig, and
    TRL's DPOConfig for ``sft_dpo``. A recipe's ``max_steps`` (Task 17b's two adapters) caps SFT at
    that many updates, which TRL then takes over the epochs as needed."""
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
            **({"max_steps": cfg["max_steps"]} if "max_steps" in cfg else {}),
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
    """The SFT rows (prompt, completion) and the pair rows (prompt, chosen, rejected); the rows
    with a ``kind`` belong to ``sft_dpo_seed`` alone (``term_rows``)."""
    plain = [row for row in rows if "kind" not in row]
    return [row for row in plain if "completion" in row], [row for row in plain if "chosen" in row]


def term_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """``sft_dpo_seed``'s rows by kind, each checked as the host cleaned it."""
    found: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}
    for row in rows:
        if "kind" not in row:
            continue
        if row["kind"] not in KINDS:
            raise ValueError(f"row {row.get('id')}'s kind is one of {', '.join(KINDS)}")
        if row["kind"] == "anchor":
            check_clean({"id": row.get("id"), "prompt": row["prompt"], "chosen": [], "rejected": []})
            choices, answer = row.get("choices"), row.get("answer")
            if (not isinstance(choices, list) or not all(isinstance(c, str) and not SPECIAL_TOKEN.search(c) for c in choices)
                    or isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices)):
                raise ValueError(f"anchor {row.get('id')} has clean choices and an answer among them")
        else:
            check_clean(row)
            if row["kind"] == "seed":
                check_clean({**row, "prompt": row.get("teacher")})
        found[row["kind"]].append(row)
    return found


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sides(row: dict[str, Any]) -> tuple[str, ...]:
    return ("completion",) if "completion" in row else ("chosen", "rejected")


def _reply(side: Any) -> str:
    """A row's reply: one assistant message whose content is text."""
    if (not isinstance(side, list) or len(side) != 1 or not isinstance(side[0], dict)
            or side[0].get("role") != "assistant" or not isinstance(side[0].get("content"), str)):
        raise ValueError("a row's reply is one assistant message of text")
    return str(side[0]["content"])


def template_kwargs(cfg: dict[str, Any]) -> dict[str, str]:
    """The chat template's arguments for every row of the run, as the server is sent them: the
    night's reasoning strength and its date. The template dates a prompt that has no system turn
    from the clock unless it is told, so the date never comes from the VM's clock."""
    return {"reasoning_strength": cfg["reasoning_strength"], "current_date": cfg["current_date"]}


def row_template(template: dict[str, str], row: dict[str, Any]) -> dict[str, str]:
    """``template`` for one row: a row that carries its own ``current_date`` (replay's rows, each
    served on its own night, Task 17b) is rendered on that date, any other on the run's."""
    if "current_date" not in row:
        return template
    day = row["current_date"]
    if not isinstance(day, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day):
        raise ValueError(f"row {row.get('id')}'s current_date is YYYY-MM-DD")
    return {**template, "current_date": day}


def check_clean(row: dict[str, Any]) -> None:
    """A row as the host cleaned it: no special token's text anywhere, and no reasoning phrase in a
    system turn, which would stop the template writing the night's strength."""
    if not isinstance(row.get("prompt"), list) or not all(isinstance(row.get(side), list) for side in _sides(row)):
        raise ValueError(f"row {row.get('id')}'s prompt and reply are lists of messages")
    for message in [*row["prompt"], *(message for side in _sides(row) for message in row[side])]:
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise ValueError("every message of a row is text")
        if SPECIAL_TOKEN.search(text) or (message.get("role") == "system" and REASONING_PHRASE.search(text)):
            raise ValueError(f"row {row.get('id')} wasn't cleaned as the server's prompts are")


def plain_row(tokenizer: Any, row: dict[str, Any], template: dict[str, str]) -> dict[str, str]:
    """``row`` as TRL's plain-text prompt and reply (or chosen and rejected): the chat template's
    prompt (with ``template``'s arguments) and the generation prompt and the reply's channel header,
    each reply closed by ``<|eot|>``. TRL tokenizes plain text with the tokenizer's special tokens
    on, which puts the BOS first itself, so the template's own BOS is left off: one BOS, as the
    server has."""
    check_clean(row)
    prompt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False,
                                           **row_template(template, row))
    if not isinstance(prompt, str) or not prompt.startswith(tokenizer.bos_token):
        raise ValueError("the chat template's prompt starts with the BOS")
    return {"prompt": prompt[len(tokenizer.bos_token):] + USER_CHANNEL,
            **{side: _reply(row[side]) + REPLY_END for side in _sides(row)}}


def _fits(tokenizer: Any, plain: dict[str, str], max_length: int) -> bool:
    """Whether every side of a plain row is at most ``max_length`` tokens, counted as TRL counts."""
    return all(len(tokenizer(text=plain["prompt"] + plain[side])["input_ids"]) <= max_length
               for side in _sides(plain))


def _rendered(tokenizer: Any, rows: list[dict[str, Any]], max_length: int,
              template: dict[str, str]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    """Each row with its plain form, the rows longer than ``max_length`` left out."""
    both = [(row, plain_row(tokenizer, row, template)) for row in rows]
    return [(row, plain) for row, plain in both if _fits(tokenizer, plain, max_length)]


def check_every_row_kept(cfg: dict[str, Any], rows: int, kept: int) -> None:
    """A recipe with ``keep_every_row`` (Task 17b, whose rows are whole trajectories) trains on
    every row or not at all: a row too long for ``max_seq_length``, counted with the real
    tokenizer, stops the run here, before the model loads or a step is taken."""
    if cfg.get("keep_every_row") and kept != rows:
        raise ValueError(f"{rows - kept} of {rows} rows are longer than max_seq_length, and this run keeps every row")


def _check_tokens(tokenizer: Any, row: dict[str, Any], side: str, prompt_ids: Sequence[int],
                  reply_ids: Sequence[int], template: dict[str, str]) -> None:  # pragma: no cover - needs the training stack
    """What TRL prepared for one side of ``row`` is what the server renders: the prompt is the
    template's prompt with the generation prompt, then the header (as the night's client asks vLLM
    to tokenize them), the trained tokens are the reply and ``<|eot|>``, and the two together are
    the template's own render of the conversation, reply included."""
    ids = {"tokenize": True, "return_dict": False, **row_template(template, row)}
    served = [*tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, **ids),
              *tokenizer(USER_CHANNEL, add_special_tokens=False)["input_ids"]]
    reply = [*tokenizer(_reply(row[side]), add_special_tokens=False)["input_ids"],
             tokenizer.convert_tokens_to_ids(REPLY_END)]
    whole = list(tokenizer.apply_chat_template(row["prompt"] + row[side], **ids))
    if list(prompt_ids) != served or list(reply_ids) != reply or [*prompt_ids, *reply_ids] != whole:
        raise ValueError(f"the {side} tokens TRL prepared differ from the chat template's render")


def _check_sft(tokenizer: Any, kept: list[tuple[dict[str, Any], dict[str, str]]], prepared: Any,
               template: dict[str, str], prompts: list[dict[str, Any]]) -> int:  # pragma: no cover - needs the training stack
    """Every SFT row's tokens and labels, as TRL prepared them, against the template: the labels are
    -100 (no loss) on the prompt and the tokens themselves from the reply's first token on. Each
    row's prompt ids go into ``prompts``."""
    for (row, _), done in zip(kept, prepared, strict=True):
        ids, labels = list(done["input_ids"]), list(done["labels"])
        start = next((i for i, label in enumerate(labels) if label != -100), len(labels))
        if labels != [-100] * start + ids[start:]:
            raise ValueError("the loss falls on more than the reply")
        _check_tokens(tokenizer, row, "completion", ids[:start], ids[start:], template)
        prompts.append({"id": row["id"], "prompt_ids": ids[:start]})
    return len(kept)


def check_template(tokenizer: Any, cfg: dict[str, Any]) -> None:
    """The tokenizer's chat template is the one the rendering names (by its sha256)."""
    found = hashlib.sha256(str(tokenizer.chat_template).encode("utf-8")).hexdigest()
    if found != cfg["template_sha256"]:
        raise ValueError(f"the chat template's sha256 is {found}, not the rendering's {cfg['template_sha256']}")


def record_rendering(out: Path, rendering: str) -> None:
    """The rendering, written into the saved adapter's ``adapter_config.json``."""
    meta_path = out / "adapter_config.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, RENDERING_KEY: rendering}, indent=2, sort_keys=True), encoding="utf-8")


def record_base_model(out: Path) -> None:
    """Glimmer's name as the saved adapter's base, where PEFT wrote the folder it loaded from: the
    server resolves an adapter of ``MODEL`` only, and the upload checks it."""
    meta_path = out / "adapter_config.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, "base_model_name_or_path": MODEL}, indent=2, sort_keys=True),
                         encoding="utf-8")


def base_model(environ: Mapping[str, str]) -> str | None:
    """The local folder holding Glimmer's pinned revision that the Modal job names (``BASE_ENV``), or
    None to load Glimmer by name at the recipe's revision (Colab). A name that isn't a folder is
    refused, and so is a base beside the smoke model: neither ever falls back to the other."""
    base = environ.get(BASE_ENV) or None
    if base is None:
        return None
    if environ.get(SMOKE_ENV):
        raise ValueError(f"{BASE_ENV} and the smoke model ({SMOKE_ENV}) never go together")
    if not Path(base).is_dir():
        raise ValueError(f"{BASE_ENV} names a local folder holding the pinned revision")
    return base


def _smoke_model() -> str | None:
    """The tiny local model the smoke run names, or None on a real night. A name that isn't a local
    folder is refused: the smoke run never falls back to anything."""
    smoke = os.environ.get(SMOKE_ENV) or None
    if smoke is not None and not Path(smoke).is_dir():
        raise ValueError(f"{SMOKE_ENV} names a local model folder")
    return smoke


def _load_model(cfg: dict[str, Any], settings: dict[str, Any], smoke: str | None,
                base: str | None) -> Any:  # pragma: no cover - needs the training stack
    """Glimmer in 4-bit on the GPU (by name at the recipe's revision, or from ``base``, the Modal job's
    copy of that revision) or, for the smoke run only, the tiny model in fp32 on the CPU; then PEFT's
    preparation for k-bit training (frozen base, casts, gradient checkpointing)."""
    import torch  # type: ignore[import-not-found, unused-ignore]
    from peft import prepare_model_for_kbit_training  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig  # type: ignore[import-not-found, unused-ignore]

    if smoke is None:
        quant = {**settings["quantization"], "bnb_4bit_compute_dtype": torch.bfloat16,
                 "bnb_4bit_quant_storage": torch.bfloat16}
        model = AutoModelForImageTextToText.from_pretrained(
            base or MODEL, revision=None if base else cfg["revision"], quantization_config=BitsAndBytesConfig(**quant),
            dtype=torch.bfloat16, device_map={"": 0},
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(smoke, dtype=torch.float32)
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def dpo_reference(trainer: Any) -> str:
    """The reference TRL 1.16's DPOTrainer scores against, by its own rule: with no ``ref_model``, a
    PEFT model's ``ref`` adapter if it has one, else the model with its adapters switched off."""
    if trainer.ref_model is not None:
        return "ref_model"
    return "ref_adapter" if "ref" in trainer.model.peft_config else "adapter_off"


def _dpo(model: Any, tokenizer: Any, pairs: list[tuple[dict[str, Any], dict[str, str]]], settings: dict[str, Any],
         out: Path, template: dict[str, str], prompts: list[dict[str, Any]]) -> tuple[int, str]:  # pragma: no cover - GPU VM only
    """The DPO pass on the night's pairs; its reference is ``model`` with the adapter switched off.
    Returns the steps taken and the reference used."""
    from datasets import Dataset  # type: ignore[import-not-found, unused-ignore]
    from trl import DPOConfig, DPOTrainer  # type: ignore[import-not-found, unused-ignore]

    data = Dataset.from_list([plain for _, plain in pairs])
    trainer = DPOTrainer(model=model, ref_model=None, args=DPOConfig(output_dir=str(out / "dpo"), **settings),
                         train_dataset=data, processing_class=tokenizer)
    for (row, _), done in zip(pairs, trainer.train_dataset, strict=True):
        for side in ("chosen", "rejected"):
            _check_tokens(tokenizer, row, side, done["prompt_ids"], done[f"{side}_ids"], template)
        prompts.append({"id": row["id"], "prompt_ids": list(done["prompt_ids"])})
    # TRL 1.x adds a frozen copy of the trained adapter ("ref") and scores the reference with it. With
    # only the trained adapter left, it scores the reference with the adapter switched off.
    for name in [name for name in trainer.model.peft_config if name != "default"]:
        trainer.model.delete_adapter(name)
    reference = dpo_reference(trainer)
    if reference != "adapter_off":
        raise ValueError(f"the DPO reference is {reference}, not the model with its adapter switched off")
    trainer.train()
    return int(trainer.state.global_step), reference


def load_loss(folder: Path | None = None) -> Any:
    """``sleep_loss.py`` from beside this file. The job runs this script under ``python -I``, which
    puts no folder on the import path, so the module loads by its file and is registered by name
    (its dataclasses look themselves up there)."""
    import importlib.util
    import sys

    path = (folder or Path(__file__).resolve().parent) / LOSS_FILE
    spec = importlib.util.spec_from_file_location("sleep_loss", path)
    if spec is None or spec.loader is None or not path.is_file():
        raise ValueError(f"{SEED_OBJECTIVE} needs {LOSS_FILE} beside the training script")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sleep_loss"] = module
    spec.loader.exec_module(module)
    return module


def served_ids(tokenizer: Any, messages: list[dict[str, Any]], template: dict[str, str]) -> list[int]:
    """A prompt's ids as the server renders it: the template with the generation prompt, then the header."""
    return [*tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False,
                                           **template),
            *tokenizer(USER_CHANNEL, add_special_tokens=False)["input_ids"]]


def reply_ids(tokenizer: Any, text: str, *, end: bool = True) -> list[int]:
    """A reply's ids, closed by ``<|eot|>`` (an anchor's choice is scored open, as the gate scores it)."""
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return [*ids, tokenizer.convert_tokens_to_ids(REPLY_END)] if end else ids


def seed_batch(loss: Any, tokenizer: Any, kept: list[tuple[dict[str, Any], dict[str, str]]],
               pairs: list[tuple[dict[str, Any], dict[str, str]]], terms: dict[str, list[dict[str, Any]]],
               cfg: dict[str, Any], template: dict[str, str]) -> tuple[Any, dict[str, int]]:
    """The loss's batch, every row as the server renders it, and the term rows too long to train."""
    def prompt(row: dict[str, Any], key: str = "prompt") -> tuple[int, ...]:
        return tuple(served_ids(tokenizer, row[key], row_template(template, row)))

    def reply(row: dict[str, Any], side: str = "completion") -> tuple[int, ...]:
        return tuple(reply_ids(tokenizer, _reply(row[side])))

    limit = cfg["max_seq_length"]
    seeds = [loss.SeedSeq(prompt(row), prompt(row, "teacher"), reply(row)) for row in terms["seed"]]
    dreams = ([(loss.Seq(prompt(row), reply(row)), str(row.get("seed_kind", ""))) for row in terms["dream"]]
              if cfg["terms"]["dreams"] else [])  # dream rows train only when the recipe asks for them
    fit = [(seq, kind) for seq, kind in dreams if len(seq.prompt) + len(seq.reply) <= limit]
    batch = loss.Batch(
        sft=tuple(loss.Seq(prompt(row), reply(row)) for row, _ in kept),
        pairs=tuple(loss.PairSeq(prompt(row), reply(row, "chosen"), reply(row, "rejected")) for row, _ in pairs),
        seeds=tuple(row for row in seeds if len(row.teacher) + len(row.reply) <= limit),
        dreams=tuple(seq for seq, _ in fit), dream_groups=tuple(kind for _, kind in fit),
        anchors=tuple(loss.AnchorSeq(prompt(row), tuple(tuple(reply_ids(tokenizer, c, end=False)) for c in row["choices"]),
                                     row["answer"]) for row in terms["anchor"]),
    )
    return batch, {"seed": len(seeds) - len(batch.seeds), "dream": len(dreams) - len(batch.dreams)}


def _seed_model(model: Any, settings: dict[str, Any], prev: str | None, fold: int | None,
                loss: Any) -> tuple[Any, int]:  # pragma: no cover - needs the training stack
    """The PEFT model ``sft_dpo_seed`` trains, and how many leading ranks stay frozen. Folding, the
    long-term adapter fills the first ranks of a LoRA of its rank plus the night's, and the rest of B
    starts at zero (the reset); otherwise a warm start loads the previous adapter as today's does."""
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found, unused-ignore]

    if fold is None and prev:
        return PeftModel.from_pretrained(model, prev, is_trainable=True), 0
    if fold is None or not prev:
        return get_peft_model(model, LoraConfig(**settings["lora"])), 0
    meta = json.loads((Path(prev) / "adapter_config.json").read_text(encoding="utf-8"))
    r_prev = meta.get("r")
    if not isinstance(r_prev, int) or meta.get("lora_alpha") != 2 * r_prev or meta.get("rank_pattern"):
        raise ValueError("the long-term adapter is a plain LoRA with alpha 2r")
    width = r_prev + settings["lora"]["r"]
    peft = get_peft_model(model, LoraConfig(**{**settings["lora"], "r": width, "lora_alpha": 2 * width}))
    loss.load_leading(peft, loss.read_safetensors(Path(prev) / "adapter_model.safetensors"), r_prev)
    return peft, r_prev


def _train_seed(cfg: dict[str, Any], settings: dict[str, Any], tokenizer: Any, template: dict[str, str],
                kept: list[tuple[dict[str, Any], dict[str, str]]], pairs: list[tuple[dict[str, Any], dict[str, str]]],
                rows: list[dict[str, Any]], model: Any, prev: str | None, out: Path) -> dict[str, Any]:  # pragma: no cover - needs the training stack
    """``sft_dpo_seed``: the one weighted loss over every term in one loop, then the fold. Returns the
    receipt's own part: the terms' receipt, the steps and the pairs trained."""
    loss = load_loss()
    fold = cfg["terms"]["fold_rank"]
    by_kind = term_rows(rows)
    batch, too_long = seed_batch(loss, tokenizer, kept, pairs, by_kind, cfg, template)
    checked = _check_seed_tokens(tokenizer, batch, kept, pairs, by_kind, template)
    model, frozen = _seed_model(model, settings, prev, fold, loss)
    eot = tokenizer.convert_tokens_to_ids(REPLY_END)
    plan = loss.Plan(lr=cfg["learning_rate"], min_steps=cfg["min_optimizer_steps"], beta=cfg["dpo_beta"],
                     seed=cfg["seed"], eot=eot, pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eot,
                     long_term_rank=frozen, dreams=cfg["terms"]["dreams"])
    terms = loss.train_terms(model, batch, plan)
    model.save_pretrained(str(out))
    terms |= {"dropped_truncated": too_long, "fold_rank": fold,
              "folded_to": loss.fold_folder(out, fold) if fold is not None else None}
    return {"terms": terms, "steps": terms["steps"], "pairs": len(batch.pairs), "rows_checked": checked}


def _check_seed_tokens(tokenizer: Any, batch: Any, kept: list[tuple[dict[str, Any], dict[str, str]]],
                       pairs: list[tuple[dict[str, Any], dict[str, str]]], by_kind: dict[str, list[dict[str, Any]]],
                       template: dict[str, str]) -> int:  # pragma: no cover - needs the training stack
    """Each trained row's ids against the template's own render of the whole conversation, as the
    SFT and DPO passes' rows are checked: the teacher's form of a seeding row too."""
    done = 0
    for (row, _), seq in zip(kept, batch.sft, strict=True):
        _check_tokens(tokenizer, row, "completion", seq.prompt, seq.reply, template)
        done += 1
    for (row, _), pair in zip(pairs, batch.pairs, strict=True):
        _check_tokens(tokenizer, row, "chosen", pair.prompt, pair.chosen, template)
        _check_tokens(tokenizer, row, "rejected", pair.prompt, pair.rejected, template)
        done += 1
    for row in [*by_kind["seed"], *by_kind["dream"]]:
        forms = [row, {**row, "prompt": row["teacher"]}] if row["kind"] == "seed" else [row]
        for form in forms:
            _check_tokens(tokenizer, form, "completion", served_ids(tokenizer, form["prompt"], row_template(template, form)),
                          reply_ids(tokenizer, _reply(form["completion"])), template)
        done += 1
    return done


def _match_sft(trainer: Any, minutes: float, started: float) -> int:  # pragma: no cover - needs the training stack
    """The live comparison's control: after today's SFT and DPO, another SFT pass on today's rows (the
    recipe's own epochs and schedule) while the job has trained under ``minutes``; none starts past
    the mark. Returns the extra steps."""
    extra = 0
    while (time.monotonic() - started) / 60.0 < minutes:
        trainer.train()
        extra += int(trainer.state.global_step)
    return extra


def main() -> int:  # pragma: no cover - runs only on the GPU VM and in the CPU smoke run
    import numpy  # type: ignore[import-not-found, unused-ignore]
    import torch  # type: ignore[import-not-found, unused-ignore]
    from datasets import Dataset  # type: ignore[import-not-found, unused-ignore]
    from peft import LoraConfig, PeftModel  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoTokenizer, set_seed  # type: ignore[import-not-found, unused-ignore]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found, unused-ignore]

    cfg = load_config(os.environ["ANJO_TRAIN_CONFIG"])
    smoke = _smoke_model()
    base = base_model(os.environ)
    settings = build_settings(cfg) if smoke is None else smoke_settings(build_settings(cfg))
    out = Path(os.environ["ANJO_TRAIN_OUT"])
    prev = os.environ.get("ANJO_TRAIN_PREV") or None
    random.seed(cfg["seed"])
    numpy.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    set_seed(cfg["seed"])
    local = smoke or base
    tokenizer = AutoTokenizer.from_pretrained(local or MODEL, revision=None if local else cfg["revision"])
    # TRL ends a plain-text reply with the tokenizer's EOS unless it ends with it already. Glimmer's
    # turn ends with <|eot|> (its EOS is <|end_of_text|>), and each reply is closed by it.
    tokenizer.eos_token = REPLY_END
    check_template(tokenizer, cfg)
    template = template_kwargs(cfg)
    every = _rows(Path(os.environ["ANJO_TRAIN_DATASET"]))
    rows, pair_rows = split_rows(every)
    kept = _rendered(tokenizer, rows, cfg["max_seq_length"], template)
    check_every_row_kept(cfg, len(rows), len(kept))  # before the model loads: only the boot is spent
    seeded = cfg["objective"] == SEED_OBJECTIVE
    pairs = _rendered(tokenizer, pair_rows, cfg["max_seq_length"], template) if "dpo" in settings or seeded else []
    model = _load_model(cfg, settings, smoke, base)
    if seeded:
        started = time.monotonic()
        done = _train_seed(cfg, settings, tokenizer, template, kept, pairs, every, model, prev, out)
        minutes = (time.monotonic() - started) / 60.0
        record_rendering(out, cfg["rendering"])
        if base is not None:
            record_base_model(out)
        receipt = {
            **done, "epochs": done["terms"]["epochs"], "train_minutes": minutes, "examples": len(kept),
            "dropped_truncated": len(rows) - len(kept), "gpu": torch.cuda.get_device_name(0) if smoke is None else "cpu",
            "objective": cfg["objective"], "dpo_steps": 0, "pairs_dropped_truncated": len(pair_rows) - len(pairs),
            "dpo_reference": "adapter_off", "rendering": cfg["rendering"],
        }
        (out / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
        return 0
    peft_config = None
    if prev:
        model = PeftModel.from_pretrained(model, prev, is_trainable=True)
    else:
        peft_config = LoraConfig(**settings["lora"])
    data = Dataset.from_list([plain for _, plain in kept])
    trainer = SFTTrainer(
        model=model, args=SFTConfig(output_dir=str(out / "run"), **settings["sft"]),
        train_dataset=data, processing_class=tokenizer, peft_config=peft_config,
    )
    prompts: list[dict[str, Any]] = []
    checked = _check_sft(tokenizer, kept, trainer.train_dataset, template, prompts)
    started = time.monotonic()
    trainer.train()
    dpo_steps, reference = 0, "none"
    if pairs:
        accumulation, epochs = dpo_plan(len(pairs), cfg)
        dpo_settings = {**settings["dpo"], "gradient_accumulation_steps": accumulation, "num_train_epochs": epochs}
        dpo_steps, reference = _dpo(trainer.model, tokenizer, pairs, dpo_settings, out, template, prompts)
    matched = _match_sft(trainer, cfg["match_minutes"], started) if "match_minutes" in cfg else 0
    minutes = (time.monotonic() - started) / 60.0
    trainer.model.save_pretrained(str(out))
    record_rendering(out, cfg["rendering"])
    if base is not None:
        record_base_model(out)
    if smoke is not None:
        (out / SMOKE_TOKENS).write_text(json.dumps(prompts), encoding="utf-8")
    receipt = {
        "steps": int(trainer.state.global_step), "epochs": cfg["epochs"], "train_minutes": minutes,
        "examples": len(kept), "dropped_truncated": len(rows) - len(kept),
        "gpu": torch.cuda.get_device_name(0) if smoke is None else "cpu", "objective": cfg["objective"],
        "dpo_steps": dpo_steps, "pairs": len(pairs),
        "pairs_dropped_truncated": len(pair_rows) - len(pairs) if "dpo" in settings else 0,
        "rows_checked": checked + len(pairs), "dpo_reference": reference, "rendering": cfg["rendering"],
        **({"match_minutes": cfg["match_minutes"], "match_steps": matched} if "match_minutes" in cfg else {}),
    }
    (out / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - the VM's kernel runs the file as __main__
    main()
