"""The LoRA recipe and ``train_lora`` (Task 5).

The recipe starts from Meta's TRL QLoRA recipe for Glimmer (NF4 with double quantization, bf16
compute) with Kevin's 2026-10-27 adjustments, each a dated ledger entry the coordinator registers:

- rank ``r`` 16 by default and at most 32, the serving cap (``config.MAX_LORA_RANK``); alpha is
  always 2r and dropout 0.05;
- the attention projections and the MLP (q, k, v, o, gate, up, down) of the text decoder only: the
  training script anchors the names there, since Glimmer's vision tower has a q, k and v too;
- lr 2e-4, or 5e-5 for a warm start on the last promoted adapter; fixed seeds;
- at least ``min_optimizer_steps`` updates: a tiny night with batch 1 and accumulation 16 would
  make one or two, so accumulation shrinks to ``n // min_optimizer_steps`` (between 1 and 16) and,
  if that still falls short, epochs rise to at most 4 (``plan_steps``);
- ``max_seq_length`` at least 8,192 with ``packing=False``;
- ``objective``: ``sft`` (the default), or ``sft_dpo`` (Task 7), which adds a DPO pass on the
  night's (best, worst) practice pairs after SFT, at Meta's TRL settings (beta 0.1, lr 5e-7), with
  the adapter switched off as the reference so no second model loads. It's a development ablation,
  registered at the freeze only if it wins.

``train_lora`` writes the dataset (with the pair rows only for ``sft_dpo``), runs the ``train_lora`` job on a ``GpuRunner`` (Colab for the
night) under the training hard timeout, copies the finished adapter into the local adapter cache
and onto the Modal volume through Task 4's upload, which refuses a name already there, and returns
the ``CheckpointRef``, which carries the rendering the adapter was trained under (an adapter whose
``adapter_config.json`` records another is refused before it is published). Adapter names are
``p-<hash>-n<night>`` and never reused.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal

from anjo.sleep.budget import TRAINING_TIMEOUT_S
from anjo.sleep.ports import GpuRunner
from anjo.sleep.serving import config, framing
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.types import CheckpointRef

MAX_RANK: Final = config.MAX_LORA_RANK
DEFAULT_TARGET_MODULES: Final = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
MIN_SEQ_LENGTH: Final = 8192
MAX_ACCUMULATION: Final = 16
MAX_EPOCHS: Final = 4
TRAIN_SEED: Final = 20261028
JOB: Final = "train_lora"
DATASET_FILE: Final = "dataset.jsonl"
VOLUME_URI: Final = f"modal-volume://{config.ADAPTERS_VOLUME}"
Objective = Literal["sft", "sft_dpo"]
OBJECTIVES: Final[tuple[Objective, ...]] = ("sft", "sft_dpo")
DPO_BETA: Final = 0.1
DPO_LR: Final = 5e-7
MAX_DPO_LR: Final = 1e-5


class LoraCfgError(ValueError):
    """A recipe setting outside what Meta's ranges and the server allow."""


class TrainError(RuntimeError):
    """The training job didn't return a usable adapter."""


@dataclass(frozen=True)
class LoraCfg:
    r: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    lr: float = 2e-4
    warm_lr: float = 5e-5
    seed: int = TRAIN_SEED
    max_seq_length: int = MIN_SEQ_LENGTH
    packing: bool = False
    min_optimizer_steps: int = 8
    batch_size: int = 1
    max_accumulation: int = MAX_ACCUMULATION
    max_epochs: int = MAX_EPOCHS
    objective: Objective = "sft"
    warm_start: bool = True
    dpo_beta: float = DPO_BETA
    dpo_lr: float = DPO_LR

    def __post_init__(self) -> None:
        if isinstance(self.r, bool) or not isinstance(self.r, int) or not 1 <= self.r <= MAX_RANK:
            raise LoraCfgError(f"rank is 1 to {MAX_RANK}, the most the server takes")
        if not 0.0 <= self.dropout < 1.0:
            raise LoraCfgError("dropout is in [0, 1)")
        if not self.target_modules:
            raise LoraCfgError("a LoRA targets at least one module")
        if not (0 < self.warm_lr <= self.lr <= 1e-3):
            raise LoraCfgError("learning rates are positive, warm start no higher than a fresh one")
        if self.max_seq_length < MIN_SEQ_LENGTH:
            raise LoraCfgError(f"max_seq_length is at least {MIN_SEQ_LENGTH}")
        if self.packing:
            raise LoraCfgError("packing stays off")
        if self.min_optimizer_steps < 1 or self.batch_size != 1:
            raise LoraCfgError("at least one optimizer step, at batch 1")
        if not 1 <= self.max_accumulation <= MAX_ACCUMULATION or not 1 <= self.max_epochs <= MAX_EPOCHS:
            raise LoraCfgError(f"accumulation is 1 to {MAX_ACCUMULATION}, epochs 1 to {MAX_EPOCHS}")
        if self.objective not in OBJECTIVES:
            raise LoraCfgError(f"the objective is one of {', '.join(OBJECTIVES)}")
        if not 0 < self.dpo_beta <= 1 or not 0 < self.dpo_lr <= MAX_DPO_LR:
            raise LoraCfgError(f"DPO's beta is in (0, 1] and its learning rate in (0, {MAX_DPO_LR:g}]")

    @property
    def alpha(self) -> int:
        return 2 * self.r


@dataclass(frozen=True)
class StepPlan:
    accumulation: int
    epochs: int
    steps_per_epoch: int

    @property
    def steps(self) -> int:
        return self.steps_per_epoch * self.epochs


def plan_steps(n_examples: int, cfg: LoraCfg) -> StepPlan:
    """Accumulation and epochs that give at least ``min_optimizer_steps`` updates, within caps."""
    if n_examples < 1:
        raise LoraCfgError("a training night has at least one example")
    accumulation = max(1, min(cfg.max_accumulation, n_examples // cfg.min_optimizer_steps))
    per_epoch = math.ceil(n_examples / (cfg.batch_size * accumulation))
    epochs = 1
    if per_epoch < cfg.min_optimizer_steps:
        epochs = min(cfg.max_epochs, math.ceil(cfg.min_optimizer_steps / per_epoch))
    return StepPlan(accumulation=accumulation, epochs=epochs, steps_per_epoch=per_epoch)


def adapter_name(user_id: str, night_number: int, attempt: int) -> str:
    """``p-<hash>-n<night>``: the hash keeps the user id off the volume and tells attempts apart."""
    digest = hashlib.sha256(f"{user_id}\x1f{night_number}\x1f{attempt}".encode()).hexdigest()[:12]
    return config.check_adapter_name(f"p-{digest}-n{night_number}")


def fresh_adapter_name(user_id: str, night_number: int, used: frozenset[str]) -> str:
    """The first name for this night that was never trained under."""
    attempt = 0
    while (name := adapter_name(user_id, night_number, attempt)) in used:
        attempt += 1
    return name


def training_config(
    cfg: LoraCfg, n_examples: int, *, warm: bool, revision: str | None, night: date,
) -> dict[str, object]:
    """What the training script is told: the recipe, the step plan, the learning rate, and how the
    rows are rendered, so the adapter trains on the prompt it is served: the reasoning strength the
    server renders every request at (``config.reasoning_strength``), the night's date, the chat
    template's hash, and the rendering they make (``framing.rendering``), which the adapter records."""
    plan = plan_steps(n_examples, cfg)
    strength = config.reasoning_strength()
    return {
        "r": cfg.r, "alpha": cfg.alpha, "dropout": cfg.dropout,
        "target_modules": list(cfg.target_modules),
        "learning_rate": cfg.warm_lr if warm else cfg.lr, "seed": cfg.seed,
        "max_seq_length": cfg.max_seq_length, "packing": cfg.packing,
        "accumulation": plan.accumulation, "epochs": plan.epochs, "batch_size": cfg.batch_size,
        "warm_start": warm, "revision": revision, "objective": cfg.objective,
        "dpo_beta": cfg.dpo_beta, "dpo_learning_rate": cfg.dpo_lr, "min_optimizer_steps": cfg.min_optimizer_steps,
        "reasoning_strength": strength, "current_date": night.isoformat(),
        "template_sha256": framing.TEMPLATE_SHA256, "rendering": framing.rendering(strength),
    }


@dataclass(frozen=True)
class TrainReceipt:
    """What the night log records about training (Task 5)."""

    adapter: str
    warm_start: bool
    examples: int
    planned_steps: int
    steps: int
    epochs: int
    accumulation: int
    gpu: str
    gpu_minutes: float
    train_minutes: float
    compute_units: float
    dropped_truncated: int


AdapterPublisher = Callable[[Path, str], None]


def _number(outputs: Mapping[str, object], key: str) -> float:
    value = outputs.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrainError(f"the training job didn't report {key}")
    return float(value)


def _rendering(folder: Path) -> object:
    """The rendering an adapter folder records in its ``adapter_config.json``, or None."""
    try:
        meta = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta.get(framing.RENDERING_KEY) if isinstance(meta, dict) else None


def _copy_adapter(source: Path, target: Path) -> None:
    if target.exists():
        raise TrainError(f"adapter {target.name} is already in the cache; a name is never reused")
    target.mkdir(parents=True)
    for file in config.ADAPTER_FILES:
        if not (source / file).is_file():
            raise TrainError(f"the training job returned no {file}")
        shutil.copyfile(source / file, target / file)


def train_lora_with_receipt(
    prev: CheckpointRef | None,
    data: Dataset,
    cfg: LoraCfg,
    runner: GpuRunner,
    *,
    person_id: str,
    night: date,
    name: str,
    workdir: Path,
    adapter_cache: Path,
    publish: AdapterPublisher,
    revision: str | None = None,
    timeout_s: float = TRAINING_TIMEOUT_S,
) -> tuple[CheckpointRef, TrainReceipt]:
    """Train ``name`` on ``data``, warm from ``prev`` when the recipe says so, and publish it."""
    config.check_adapter_name(name)
    if prev is not None and prev.person_id != person_id:
        raise TrainError("a warm start is only ever from the person's own adapter")
    if not data.examples:
        raise TrainError("a training night has at least one example")
    if revision is None or not revision.strip():
        raise TrainError("the training job names Glimmer's revision; refused before any GPU is paid for")
    warm = prev is not None and cfg.warm_start
    prev_dir = adapter_cache / prev.id if warm and prev is not None else None
    if prev_dir is not None:
        config.check_adapter(prev_dir, prev_dir.name)  # the last promoted adapter, whole (RF1)
    workdir.mkdir(parents=True, exist_ok=True)
    dataset_path = workdir / DATASET_FILE
    settings = training_config(cfg, len(data.examples), warm=warm, revision=revision, night=night)
    try:  # the dataset is the person's words in plain text: it never outlives the run
        dataset_path.write_text(data.to_jsonl(pairs=cfg.objective == "sft_dpo"), encoding="utf-8")
        outputs = runner.run(JOB, {
            "person_id": person_id, "adapter_name": name, "night": night.isoformat(), "dataset_path": str(dataset_path),
            "config": settings, "prev_adapter_dir": None if prev_dir is None else str(prev_dir),
            "workdir": str(workdir),
        }, timeout_s=timeout_s)
    finally:
        dataset_path.unlink(missing_ok=True)
    returned = outputs.get("adapter_dir")
    if not isinstance(returned, str):
        raise TrainError("the training job returned no adapter folder")
    if _rendering(Path(returned)) != settings["rendering"]:
        raise TrainError(f"the adapter doesn't record the rendering it was trained under ({settings['rendering']})")
    cached = adapter_cache / name
    _copy_adapter(Path(returned), cached)
    publish(cached, name)
    plan = plan_steps(len(data.examples), cfg)
    receipt = TrainReceipt(
        adapter=name, warm_start=warm, examples=len(data.examples), planned_steps=plan.steps,
        steps=int(_number(outputs, "steps")), epochs=int(_number(outputs, "epochs")),
        accumulation=plan.accumulation, gpu=str(outputs.get("gpu", "")),
        gpu_minutes=_number(outputs, "gpu_minutes"), train_minutes=_number(outputs, "train_minutes"),
        compute_units=_number(outputs, "compute_units"),
        dropped_truncated=int(_number(outputs, "dropped_truncated")),
    )
    ckpt = CheckpointRef(
        id=name, person_id=person_id, night=night, uri=f"{VOLUME_URI}/{name}",
        parent_id=prev.id if warm and prev is not None else None, rendering=str(settings["rendering"]),
    )
    return ckpt, receipt


def train_lora(
    prev: CheckpointRef | None, data: Dataset, cfg: LoraCfg, runner: GpuRunner, **kw: object,
) -> CheckpointRef:
    """``train_lora_with_receipt`` without the receipt (the spec's signature)."""
    ckpt, _ = train_lora_with_receipt(prev, data, cfg, runner, **kw)  # type: ignore[arg-type]
    return ckpt
