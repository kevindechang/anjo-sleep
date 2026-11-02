"""The judges' warm trainer (2026-11-02): the part of it that runs in the container, and its shared terms.

The night's cold trainer (``modal_app.py``, ``anjo-sleep-train``) starts the training script in a new
process on every call, so each call pays the container's boot and Glimmer's load and 4-bit
quantization again: about 1.4 to 1.9 minutes of a judge's night before the first step. The warm
trainer (``modal_warm_app.py``, its own app ``anjo-sleep-judge-train``, so the nights' app and both
serving apps stay untouched) is a Modal class: its container loads Glimmer once, on enter
(``load``), and every call trains on the model it holds (``train_loaded``), then takes the adapter
out again (PEFT's ``unload``), so the next call starts from the base as it was loaded. A call that
fails, or a model that still holds an adapter's layers afterwards (``clean``), makes the container
stop taking calls after this one, so the next call gets a fresh one: never a call on a dirty base.

It trains plain SFT only, which is what a judge's night asks for (``night/demo.py``: 8 optimizer
steps); any other recipe (pairs, the sleep objective's terms, the thought channel) is refused here
and, before anything is spawned, sent to the cold path by the host (``runners/warm.py``).

``train_warm_on_volume`` is what one call does, and it is the cold job's steps with the script's
process swapped for the loaded model: refuse a name the adapters volume has, refuse a recipe that
isn't pinned to the app's revision, write the inputs to a scratch folder, train, hold the adapter's
two files to the upload's checks and its rendering, and land them on the adapters volume the server
loads from (``modal_job``'s own helpers, so the two can't drift). It returns what the cold job
returns, with this app's digest of the files its image carries (``files_digest`` over ``WARM_FILES``).

The container stays up ``scaledown_seconds`` after its last call (``SCALEDOWN_ENV``, read when the app
is deployed: 90 s by default, 30 to 600), so a burst of judges' nights shares one load and the GPU
isn't left idling once they end. The host keeps it up from the start of a training night until that
night's training ends (``runners/warm.py``'s ``Warmer``, one ``warm`` call every ``PING_S``).

It imports the standard library, ``serving/config.py``, ``colab_train.py`` and ``modal_job.py`` only, and
the training stack inside the functions that run on the GPU.
"""
from __future__ import annotations

import gc
import random
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.serving import config
from anjo.sleep.train import colab_train, modal_job

APP_NAME: Final = "anjo-sleep-judge-train"  # never the nights' app or a serving app
CLASS: Final = "WarmTrainer"
TRAIN_METHOD: Final = "train"
WARM_METHOD: Final = "warm"
OBJECTIVES: Final = ("sft",)
SCALEDOWN_ENV: Final = "ANJO_SLEEP_JUDGE_TRAIN_SCALEDOWN_S"
DEFAULT_SCALEDOWN_S: Final = 90
SCALEDOWN_RANGE: Final = (30, 600)
PING_S: Final = 45.0  # the host's keep-warm call, inside the scale-down window
MAX_CONTAINERS: Final = 1  # one loaded Glimmer: a second judge's training waits for the first's
_TRAIN: Final = Path(__file__).resolve().parent
WARM_FILES: Final[tuple[tuple[Path, str], ...]] = (
    *modal_job.TRAIN_FILES,
    (_TRAIN / "warm_job.py", f"{modal_job.TRAIN_ROOT}/sleep/train/warm_job.py"),
    (_TRAIN / "modal_warm_app.py", f"{modal_job.TRAIN_ROOT}/sleep/train/modal_warm_app.py"),
)
# colab_train.build_settings' quantization block, as the model is loaded once, before any recipe exists
# (a test holds them equal).
QUANTIZATION: Final[Mapping[str, object]] = {
    "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_storage": "bfloat16",
}

Train = Callable[[dict[str, Any], Path, "Path | None", Path], dict[str, Any]]


class WarmJobError(RuntimeError):
    """The warm trainer refused its inputs, or the training didn't produce an adapter the server takes."""


def scaledown_seconds(environ: Mapping[str, str]) -> int:
    """How long the container stays up after its last call: ``SCALEDOWN_ENV``, 90 s when unset."""
    raw = environ.get(SCALEDOWN_ENV, "").strip()
    if not raw:
        return DEFAULT_SCALEDOWN_S
    try:
        value = int(raw)
    except ValueError:
        raise WarmJobError(f"{SCALEDOWN_ENV} is a whole number of seconds") from None
    low, high = SCALEDOWN_RANGE
    if not low <= value <= high:
        raise WarmJobError(f"{SCALEDOWN_ENV} is {low} to {high} seconds")
    return value


def refusal(recipe: Mapping[str, object]) -> str | None:
    """Why the warm trainer won't run ``recipe`` (the host sends it to the cold path), or None."""
    if recipe.get("objective") not in OBJECTIVES:
        return f"the warm trainer runs plain SFT only, not {recipe.get('objective')}"
    if any(recipe.get(key) is not None for key in ("channel", "terms", "match_minutes")):
        return "the warm trainer runs no thought channel, sleep-objective terms or matched clock"
    return None


def files_digest(files: Sequence[tuple[Path, str]] = WARM_FILES) -> str:
    """``modal_job.files_digest`` over this app's files: the host holds the call's to the checkout's."""
    return modal_job.files_digest(files)


def deployed_files() -> tuple[tuple[Path, str], ...]:
    """``WARM_FILES`` as the container has them: each at its remote path."""
    return tuple((Path(remote), remote) for _, remote in WARM_FILES)


def train_warm_on_volume(
    *, dataset: str, config_json: str, prev: Mapping[str, bytes], name: str, pinned: str, weights_root: Path,
    adapters_root: Path, train: Train, train_files: Sequence[tuple[Path, str]] | None = None,
) -> dict[str, Any]:
    """One call in the container: train ``name`` on the loaded model (``train``) and land it on the
    adapters volume. Returns the receipt, the adapter's two files and this app's files digest."""
    digest = files_digest(deployed_files() if train_files is None else train_files)
    target = adapters_root / config.check_adapter_name(name)
    if target.exists():
        raise WarmJobError(f"adapter {name} is already on the volume; a night's adapter never changes")
    recipe = colab_train.load_config(config_json)
    if (why := refusal(recipe)) is not None:
        raise WarmJobError(why)
    modal_job.base_folder(weights_root, recipe.get("revision"), pinned)  # the pinned revision, on the volume
    with tempfile.TemporaryDirectory(prefix="anjo-warm-") as scratch:
        root = Path(scratch)
        warm = modal_job._write_inputs(root, dataset, prev)
        out = root / "out"
        out.mkdir()
        receipt = train(recipe, root / modal_job.DATASET_FILE, warm, out)
        stage = root / "stage" / name
        try:
            modal_job._checked(out, stage, name, recipe.get("rendering"))
        except modal_job.ModalJobError as error:
            raise WarmJobError(str(error)) from None
        modal_job._land(stage, target)
        files = {file: (stage / file).read_bytes() for file in config.ADAPTER_FILES}
    return {"adapter": name, "receipt": receipt, "files": files, modal_job.DIGEST_KEY: digest}


@dataclass
class Loaded:
    """Glimmer as the container loaded it once: the 4-bit model (prepared for k-bit training), its
    tokenizer, how long the load took, and whether the model is still the base (``clean``)."""

    model: Any
    tokenizer: Any
    seconds: float
    clean: bool = True


def load(base: Path) -> Loaded:  # pragma: no cover - runs only in the GPU container
    """Glimmer from the weights volume's pinned revision, in 4-bit NF4 on the GPU, prepared for k-bit
    training, exactly as the training script loads it (``colab_train._load_model``)."""
    from transformers import AutoTokenizer  # type: ignore[import-not-found, unused-ignore]

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    tokenizer.eos_token = colab_train.REPLY_END  # Glimmer's turn ends with <|eot|> (colab_train.main)
    model = colab_train._load_model({}, {"quantization": dict(QUANTIZATION)}, None, str(base))
    return Loaded(model=model, tokenizer=tokenizer, seconds=time.monotonic() - started)


def _seeded(seed: int) -> None:  # pragma: no cover - needs the training stack
    import numpy  # type: ignore[import-not-found, unused-ignore]
    import torch  # type: ignore[import-not-found, unused-ignore]
    from transformers import set_seed  # type: ignore[import-not-found, unused-ignore]

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def train_loaded(  # pragma: no cover - runs only in the GPU container
    loaded: Loaded, cfg: dict[str, Any], dataset_path: Path, prev: Path | None, out: Path,
) -> dict[str, Any]:
    """Plain SFT on the loaded model, as the training script's SFT pass runs it (the same settings,
    rendering and token checks), then the adapter is saved to ``out`` and taken out of the model."""
    import torch  # type: ignore[import-not-found, unused-ignore]
    from datasets import Dataset  # type: ignore[import-not-found, import-untyped, unused-ignore]
    from peft import LoraConfig, PeftModel  # type: ignore[import-not-found, unused-ignore]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found, unused-ignore]

    ct, tokenizer = colab_train, loaded.tokenizer
    settings = ct.build_settings(cfg)
    _seeded(cfg["seed"])
    ct.check_template(tokenizer, cfg)
    template = ct.template_kwargs(cfg)
    rows, pairs = ct.split_rows(ct._rows(dataset_path))
    if pairs:
        raise WarmJobError("the warm trainer runs plain SFT; a night with pairs trains on the cold path")
    surprise = ct.surprise_weights(cfg["objective"], rows)
    kept = ct._rendered(tokenizer, rows, cfg["max_seq_length"], template)
    ct.check_every_row_kept(cfg, len(rows), len(kept))
    loaded.clean = False  # until the adapter is out again
    model = PeftModel.from_pretrained(loaded.model, str(prev), is_trainable=True) if prev is not None else loaded.model
    trainer = None
    try:
        trainer = SFTTrainer(
            model=model, args=SFTConfig(output_dir=str(out / "run"), **settings["sft"]),
            train_dataset=Dataset.from_list([plain for _, plain in kept]), processing_class=tokenizer,
            peft_config=None if prev is not None else LoraConfig(**settings["lora"]),
        )
        checked = ct._check_sft(tokenizer, kept, trainer.train_dataset, template, [])
        started = time.monotonic()
        trainer.train()
        minutes = (time.monotonic() - started) / 60.0
        trainer.model.save_pretrained(str(out))
        ct.record_rendering(out, cfg["rendering"])
        ct.record_base_model(out)
        steps = int(trainer.state.global_step)
    finally:
        wrapped = trainer.model if trainer is not None else model
        del trainer
        loaded.model = wrapped.unload() if isinstance(wrapped, PeftModel) else wrapped  # the base as it was loaded
        loaded.clean = not any("lora_" in name for name, _ in loaded.model.named_modules())
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "steps": steps, "epochs": cfg["epochs"], "train_minutes": minutes, "examples": len(kept),
        "dropped_truncated": len(rows) - len(kept), "gpu": torch.cuda.get_device_name(0),
        "objective": cfg["objective"], "dpo_steps": 0, "pairs": 0, "pairs_dropped_truncated": 0,
        "rows_checked": checked, "dpo_reference": "none", "rendering": cfg["rendering"],
        "load_minutes": round(loaded.seconds / 60.0, 3),
        **({"surprise_weights": surprise} if surprise else {}),
    }
