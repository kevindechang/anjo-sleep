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
  if that still falls short, epochs rise to at most 4 (``plan_steps``); with ``passes`` (capacity,
  ``objective.CAPACITY_ENV``) every row is its own step, ``passes`` times over;
- ``max_seq_length`` at least 8,192 with ``packing=False``;
- ``objective``: ``sft`` (the recipe's default), or ``sft_dpo`` (Task 7), which adds a DPO pass on the
  night's (best, worst) practice pairs after SFT, at Meta's TRL settings (beta 0.1, lr 5e-6: the LoRA scale), with
  the adapter switched off as the reference so no second model loads. The nights ask for ``sft_dpo``
  (``objective.py``, Kevin 2026-10-28); a night with no usable pair trains ``sft`` (``objective_used``);
  or ``sft_dpo_seed`` (2026-10-29): one weighted loss in one job over SFT, DPO, knowledge seeding, the
  imitation term and, with ``dreams``, dreams (``sleep_loss.py``). ``fold_rank`` folds the night into the
  person's long-term adapter at that rank, a fresh slot training beside it at ``r`` and the fresh rate.
  ``protect`` (``anchors`` or ``random``, with the fold only) trains that slot off the forgetting anchors'
  activation basis, or a random one of its shape (``sleep_protect.py``); ``off`` sends the recipe unchanged.

``train_lora`` writes the dataset (with the pair rows only for ``sft_dpo``), runs the ``train_lora`` job on a ``GpuRunner`` (Modal for the
night by default, or Colab) under the training hard timeout, copies the finished adapter into the local
adapter cache and publishes it to the server: Task 4's upload, which refuses a name already there, or
nothing more when Modal's job landed it on the adapters volume already (``runners/choice.py``). It returns
the ``CheckpointRef``, which carries the rendering the adapter was trained under (an adapter whose
``adapter_config.json`` records another is refused before it is published). Adapter names are
``p-<hash>-n<night>`` and never reused.
"""
from __future__ import annotations

import dataclasses
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
PROJECTOR_SUFFIX: Final = ".projector.npz"  # the thought channel's projector, beside its night's adapter
Objective = Literal["sft", "sft_dpo", "sft_dpo_seed"]
OBJECTIVES: Final[tuple[Objective, ...]] = ("sft", "sft_dpo", "sft_dpo_seed")
Protection = Literal["off", "anchors", "random"]
PROTECTIONS: Final[tuple[Protection, ...]] = ("off", "anchors", "random")
WITH_PAIRS: Final = ("sft_dpo", "sft_dpo_seed")
MAX_MATCH_MINUTES: Final = 100.0  # inside the job's two-hour cap, with its start and landing
DPO_BETA: Final = 0.1
DPO_LR: Final = 5e-6  # 10x the spec's full-fine-tune 5e-7: on LoRA weights over a few dozen steps that is inert (alignment-handbook QLoRA DPO scale)
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
    fold_rank: int | None = None  # sft_dpo_seed: fold the night into a long-term adapter of this rank
    dreams: bool = False  # sft_dpo_seed: train on the night's dream rows
    protect: Protection = "off"  # sft_dpo_seed with the fold: tonight's slot off the anchors' basis (or a random one)
    # The live comparison's matched control only: each night's training budget in minutes (its reference
    # arm's, serving included), which the job fills by the clock; every real night leaves it empty.
    match_by_night: tuple[tuple[date, float], ...] = ()
    # Capacity (objective.CAPACITY_ENV): the main block's passes over its rows at one row a step, in place
    # of the step plan; None is the step plan. Never with a matched control, which trains by the clock.
    passes: int | None = None

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
        seeded = self.objective == "sft_dpo_seed"
        if (self.fold_rank is not None or self.dreams) and not seeded:
            raise LoraCfgError("the fold and dreams train only under sft_dpo_seed")
        if self.protect not in PROTECTIONS:
            raise LoraCfgError(f"protect is one of {', '.join(PROTECTIONS)}")
        if self.protect != "off" and (not seeded or self.fold_rank is None):
            raise LoraCfgError("the protected slot trains only under sft_dpo_seed with the fold, which keeps it apart")
        if self.fold_rank is not None and (isinstance(self.fold_rank, bool) or not self.r <= self.fold_rank <= MAX_RANK):
            raise LoraCfgError(f"the long-term rank is from the night's rank {self.r} to {MAX_RANK}")
        nights = [night for night, _ in self.match_by_night]
        if self.match_by_night and (self.dreams or len(set(nights)) != len(nights)
                                    or not all(0 < minutes <= MAX_MATCH_MINUTES for _, minutes in self.match_by_night)):
            raise LoraCfgError(f"a matched control has one budget a night, up to {MAX_MATCH_MINUTES:g} minutes, and no dreams")
        if self.passes is not None and (isinstance(self.passes, bool) or not 1 <= self.passes <= self.max_epochs
                                        or self.match_by_night):
            raise LoraCfgError(f"passes are 1 to {self.max_epochs}, and never with a matched control's budget")

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
    if cfg.passes is not None:  # capacity: every row its own step, each pass
        return StepPlan(accumulation=1, epochs=cfg.passes, steps_per_epoch=n_examples)
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
    cfg: LoraCfg, n_examples: int, *, warm: bool, revision: str | None, night: date, steps: StepPlan | None = None,
    keep_every_row: bool = False, channel: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """What the training script is told: the recipe, the step plan, the learning rate, and how the
    rows are rendered, so the adapter trains on the prompt it is served: the reasoning strength the
    server renders every request at (``config.reasoning_strength``), the night's date, the chat
    template's hash, and the rendering they make (``framing.rendering``), which the adapter records.
    ``steps`` fixes the step plan instead of deriving it from ``n_examples``, and is sent as
    ``max_steps``, so two runs on sets of different sizes take the same updates; ``keep_every_row``
    stops the run on the VM before training if any row is too long (Task 17b). ``channel`` (the thought
    channel's settings, ``channel_train.check_channel``) trains the projector with the LoRA, on rows that
    carry the core state they were made under."""
    plan = steps or plan_steps(n_examples, cfg)
    strength = config.reasoning_strength()
    fixed: dict[str, object] = {} if steps is None else {"max_steps": steps.steps}
    fixed |= {"keep_every_row": True} if keep_every_row else {}
    fixed |= {} if channel is None else {"channel": dict(channel)}
    if cfg.objective == "sft_dpo_seed":
        fixed |= {"terms": {"fold_rank": cfg.fold_rank, "dreams": cfg.dreams,
                            **({} if cfg.protect == "off" else {"protect": cfg.protect}),
                            **({} if cfg.passes is None else {"passes": cfg.passes})}}
    # A folded night loads the long-term adapter frozen and trains a fresh slot, at the fresh rate.
    rate = cfg.warm_lr if warm and cfg.fold_rank is None else cfg.lr
    if cfg.match_by_night:
        budget = dict(cfg.match_by_night).get(night)
        if budget is None:
            raise LoraCfgError(f"the matched control has no budget for the night of {night.isoformat()}")
        fixed |= {"match_minutes": budget}
    return {**fixed,
        "r": cfg.r, "alpha": cfg.alpha, "dropout": cfg.dropout,
        "target_modules": list(cfg.target_modules),
        "learning_rate": rate, "seed": cfg.seed,
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
    objective: str = "sft"  # what the job trained under: sft_dpo falls back to sft for a night with no pair
    pairs: int = 0  # the (best, worst) pairs sent when it trained sft_dpo; a count, never the words
    terms: Mapping[str, object] | None = None  # sft_dpo_seed's receipt: rows, minutes and loss per term, dreams kept
    surprise_weights: str | None = None  # whether the rows' surprise weights reached the loss; None when none had one
    match: Mapping[str, object] | None = None  # a matched control's budget, the minutes it used and its steps
    projector: str = ""  # the thought channel's projector this night trained, kept beside the adapter ("": off)
    channel_receipt: str = ""  # the channel's counts and margins (JSON), as the job reported them ("": off)


AdapterPublisher = Callable[[Path, str], None]


def objective_used(cfg: LoraCfg, data: Dataset) -> Objective:
    """``cfg``'s objective, except that ``sft_dpo`` trains ``sft`` for a night with no usable pair.
    ``sft_dpo_seed`` is one loss whatever rows the night has: a term without rows adds nothing."""
    if cfg.objective == "sft_dpo_seed":
        return "sft_dpo_seed"
    return "sft_dpo" if cfg.objective == "sft_dpo" and data.pairs else "sft"


def _number(outputs: Mapping[str, object], key: str) -> float:
    value = outputs.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrainError(f"the training job didn't report {key}")
    return float(value)


def _pairs_trained(outputs: Mapping[str, object], data: Dataset) -> int:
    """The pairs the DPO pass trained: the job's own count, else those sent less those it dropped as too long."""
    reported = outputs.get("pairs")
    if isinstance(reported, int) and not isinstance(reported, bool):
        return max(0, reported)
    dropped = outputs.get("pairs_dropped_truncated")
    return max(0, len(data.pairs) - (dropped if isinstance(dropped, int) and not isinstance(dropped, bool) else 0))


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


def _keep_projector(outputs: Mapping[str, object], adapter_cache: Path, name: str) -> str:
    """The channel run's projector, copied beside the cached adapter (never inside its folder, which
    holds the adapter's two files and nothing else)."""
    returned = outputs.get("projector_path")
    if not isinstance(returned, str) or not Path(returned).is_file():
        raise TrainError("the channel run returned no projector")
    kept = adapter_cache / f"{name}{PROJECTOR_SUFFIX}"
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(returned, kept)
    return str(kept)


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
    steps: StepPlan | None = None,
    keep_every_row: bool = False,
    channel: Mapping[str, object] | None = None,
    prev_projector: Path | None = None,
) -> tuple[CheckpointRef, TrainReceipt]:
    """Train ``name`` on ``data``, warm from ``prev`` when the recipe says so, and publish it.
    ``steps`` and ``keep_every_row`` go to the script as ``training_config`` says. With ``channel``
    the same job trains the thought channel's projector too, from ``prev_projector`` (the person's
    last, or the shared one), and the one it trained is kept at ``<adapter_cache>/<name>.projector.npz``,
    beside the adapter: one unit with the core state the rows carry."""
    config.check_adapter_name(name)
    if prev is not None and prev.person_id != person_id:
        raise TrainError("a warm start is only ever from the person's own adapter")
    if not data.examples:
        raise TrainError("a training night has at least one example")
    if revision is None or not revision.strip():
        raise TrainError("the training job names Glimmer's revision; refused before any GPU is paid for")
    cfg = dataclasses.replace(cfg, objective=objective_used(cfg, data))
    warm = prev is not None and cfg.warm_start
    prev_dir = adapter_cache / prev.id if warm and prev is not None else None
    if prev_dir is not None:
        config.check_adapter(prev_dir, prev_dir.name)  # the last promoted adapter, whole (RF1)
    workdir.mkdir(parents=True, exist_ok=True)
    dataset_path = workdir / DATASET_FILE
    settings = training_config(cfg, len(data.examples), warm=warm, revision=revision, night=night, steps=steps,
                               keep_every_row=keep_every_row, channel=channel)
    if channel is None and prev_projector is not None:
        raise TrainError("a projector warm start without the channel on")
    try:  # the dataset is the person's words in plain text: it never outlives the run
        dataset_path.write_text(data.to_jsonl(pairs=cfg.objective in WITH_PAIRS, terms=cfg.objective == "sft_dpo_seed"),
                                encoding="utf-8")
        outputs = runner.run(JOB, {
            "person_id": person_id, "adapter_name": name, "night": night.isoformat(), "dataset_path": str(dataset_path),
            "config": settings, "prev_adapter_dir": None if prev_dir is None else str(prev_dir),
            "workdir": str(workdir),
            **({} if prev_projector is None else {"prev_projector": str(prev_projector)}),
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
    projector = _keep_projector(outputs, adapter_cache, name) if channel is not None else ""
    publish(cached, name)
    plan = steps or plan_steps(len(data.examples), cfg)
    trained_pairs = _pairs_trained(outputs, data) if cfg.objective in WITH_PAIRS else 0
    trained_objective: Objective = "sft_dpo" if trained_pairs else "sft"  # all pairs dropped: only SFT ran
    if cfg.objective == "sft_dpo_seed":
        trained_objective = "sft_dpo_seed"
    terms = outputs.get("terms")
    receipt = TrainReceipt(
        adapter=name, warm_start=warm, examples=len(data.examples), planned_steps=plan.steps,
        steps=int(_number(outputs, "steps")), epochs=int(_number(outputs, "epochs")),
        accumulation=plan.accumulation, gpu=str(outputs.get("gpu", "")),
        gpu_minutes=_number(outputs, "gpu_minutes"), train_minutes=_number(outputs, "train_minutes"),
        compute_units=_number(outputs, "compute_units"),
        dropped_truncated=int(_number(outputs, "dropped_truncated")),
        objective=trained_objective, pairs=trained_pairs, projector=projector,
        terms=dict(terms) if cfg.objective == "sft_dpo_seed" and isinstance(terms, Mapping) else None,
        surprise_weights=str(outputs["surprise_weights"]) if outputs.get("surprise_weights") else None,
        match=dict(match) if cfg.match_by_night and isinstance(match := outputs.get("match"), Mapping) else None,
        channel_receipt=json.dumps(outputs["channel"], sort_keys=True) if isinstance(outputs.get("channel"), dict) else "",
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
