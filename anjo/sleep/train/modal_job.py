"""The Modal training job's shared terms, and the part of it that runs in the container (Task 5).

The training app (``modal_app.py``, ``anjo-sleep-train``) and the night's runner
(``runners/modal.py``) both read the names, the GPU and the time limits here, so they can't drift.
``train_on_volume`` is what the container does with one call: the same training script the Colab VM
runs (``colab_train.py``), on the same inputs, with Glimmer read from the weights volume:

1. refuse the adapter's name if the adapters volume has it already: a night's adapter never changes;
2. refuse the recipe unless its revision is a commit, the one the app was deployed pinned to, and on
   the weights volume (``download_weights`` put it there);
3. write the dataset and a warm start's two adapter files to a scratch folder, and run the script
   under the job's own hard timeout, with Glimmer's folder in ``ANJO_TRAIN_BASE`` and Hugging Face
   offline, so nothing downloads. Its environment is the container's minus anything named like a
   key and minus Anjo's and Modal's settings: the app holds no secret, and nothing else is passed;
4. take the adapter's two files only, hold them to the upload's checks (``config.check_adapter``)
   and to the rendering the recipe names, then land them on the adapters volume under the name the
   server loads. A copy that fails partway is removed.

A thought-channel recipe (``channel_train``) runs through the same steps: its warm start may add the
projector (``projector.npz``) beside the adapter's files, or send it alone, and the projector the run
trained comes back with the adapter (``projector``), for the host to keep beside the person's core
state; only the adapter lands on the volume.

It imports the standard library, ``serving/config.py``, ``serving/projector_math.py``, ``colab_train.py`` and ``channel_train.py``
only: the image carries those files and this one, and nothing else of Anjo.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from anjo.sleep.serving import config
from anjo.sleep.train import channel_train, colab_train

APP_NAME: Final = "anjo-sleep-train"
FUNCTION: Final = "train"
GPU: Final = "A100-80GB"  # config.BF16_A100's GPU: one A100-80GB
FUNCTION_TIMEOUT_S: Final = 2 * 3600  # §12.1's training cap (budget.TRAINING_TIMEOUT_S; a test holds them equal)
STARTUP_TIMEOUT_S: Final = 15 * 60  # the container's start, before the call runs (Modal's startup_timeout)
RETURN_S: Final = 5 * 60  # landing the adapter and sending its files back
CANCEL_S: Final = 60.0  # cancelling a call whose wait ran out: the runner stops waiting this much early
MIN_EXEC_S: Final = 1800.0
REMOTE_SCRIPT: Final = "/root/anjo/sleep/train/colab_train.py"
BASE_ENV: Final = "ANJO_TRAIN_BASE"  # colab_train.BASE_ENV
DATASET_FILE: Final = "dataset.jsonl"  # lora.DATASET_FILE
OFFLINE: Final = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
# One call's arguments, in order: the dataset's text, the recipe as JSON, a warm start's adapter files,
# the adapter's name and the seconds the script may run. Nothing else crosses into the container.
JOB_FIELDS: Final = ("dataset", "config_json", "prev", "name", "exec_s")
_KEYLIKE: Final = re.compile(r"(?i)token|key|secret|passw|credential|auth")
_DROPPED_PREFIXES: Final = ("ANJO_", "MODAL_")
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")

Run = Callable[..., "subprocess.CompletedProcess[Any]"]


class ModalJobError(RuntimeError):
    """The job refused its inputs, or the training didn't produce an adapter the server takes."""


def exec_seconds(timeout_s: float) -> float:
    """How long the script may run inside a job whose hard timeout is ``timeout_s``: the container's
    start, the adapter's landing and a cancel's margin come out of it, so the call ends inside the hold."""
    if not math.isfinite(timeout_s):
        raise ModalJobError("every job has a finite timeout")
    exec_s = timeout_s - STARTUP_TIMEOUT_S - RETURN_S - CANCEL_S
    if exec_s < MIN_EXEC_S:
        raise ModalJobError("the job's timeout leaves too little time to train")
    return exec_s


def image_env(revision: str) -> dict[str, str]:
    """The image's environment: the pin the app was deployed with, and Hugging Face offline. No key."""
    return {config.REVISION_ENV: revision, **OFFLINE}


def base_folder(weights_root: Path, revision: object, pinned: str) -> Path:
    """Glimmer's folder on the weights volume for the recipe's ``revision``, which must be the pin."""
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise ModalJobError("the recipe names no pinned revision (a 40-character commit)")
    if revision != pinned:
        raise ModalJobError("the recipe's revision isn't the one this app was deployed pinned to")
    folder = weights_root / config.MODEL / revision
    if not (folder / "config.json").is_file():
        raise ModalJobError("the weights volume holds no copy of the pinned revision; run download_weights")
    return folder


def script_env(environ: Mapping[str, str], train: Mapping[str, str]) -> dict[str, str]:
    """The script's environment: the container's, minus anything named like a key and minus Anjo's
    and Modal's own settings, with Hugging Face offline and the job's ``ANJO_TRAIN_*`` values."""
    kept = {name: value for name, value in environ.items()
            if not _KEYLIKE.search(name) and not name.startswith(_DROPPED_PREFIXES)}
    return {**kept, **OFFLINE, **train}


def _write_inputs(root: Path, dataset: str, prev: Mapping[str, bytes]) -> Path | None:
    (root / DATASET_FILE).write_text(dataset, encoding="utf-8")
    if not prev:
        return None
    adapter = sorted(set(prev) - {channel_train.PROJECTOR_FILE})
    if adapter not in ([], sorted(config.ADAPTER_FILES)):
        raise ModalJobError(f"a warm start sends exactly {', '.join(config.ADAPTER_FILES)}, the thought channel's "
                            f"{channel_train.PROJECTOR_FILE} beside them, or that alone")
    folder = root / "prev"
    folder.mkdir()
    for name, data in prev.items():
        (folder / name).write_bytes(data)
    return folder


def _checked(out: Path, stage: Path, name: str, rendering: object) -> None:
    """Copy the adapter's two files to ``stage`` and hold them to the upload's checks and the rendering."""
    stage.mkdir(parents=True)
    for file in config.ADAPTER_FILES:
        if not (out / file).is_file():
            raise ModalJobError(f"the training script wrote no {file}")
        shutil.copyfile(out / file, stage / file)
    config.check_adapter(stage, name)
    meta = json.loads((stage / "adapter_config.json").read_text(encoding="utf-8"))
    if meta.get(colab_train.RENDERING_KEY) != rendering:
        raise ModalJobError(f"the adapter doesn't record the rendering it was trained under ({rendering})")


def _land(stage: Path, target: Path) -> None:
    """Put the checked files at ``target`` on the adapters volume; a copy that fails partway is removed."""
    target.mkdir()  # refuses a name that appeared meanwhile
    try:
        for file in config.ADAPTER_FILES:
            shutil.copyfile(stage / file, target / file)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def train_on_volume(
    *, dataset: str, config_json: str, prev: Mapping[str, bytes], name: str, exec_s: float, pinned: str,
    weights_root: Path, adapters_root: Path, script: str = REMOTE_SCRIPT, python: str = sys.executable,
    run: Run = subprocess.run, environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One call in the container: train ``name`` and land it on the adapters volume. Returns the
    receipt and the adapter's two files, for the night's own copy."""
    target = adapters_root / config.check_adapter_name(name)
    if target.exists():
        raise ModalJobError(f"adapter {name} is already on the volume; a night's adapter never changes")
    recipe = json.loads(config_json)
    if not isinstance(recipe, dict):
        raise ModalJobError("the recipe is a JSON object")
    base = base_folder(weights_root, recipe.get("revision"), pinned)
    with tempfile.TemporaryDirectory(prefix="anjo-train-") as scratch:
        root = Path(scratch)
        warm = _write_inputs(root, dataset, prev)
        out = root / "out"
        train = {"ANJO_TRAIN_CONFIG": config_json, "ANJO_TRAIN_DATASET": str(root / DATASET_FILE),
                 "ANJO_TRAIN_OUT": str(out), BASE_ENV: str(base),
                 **({"ANJO_TRAIN_PREV": str(warm)} if warm is not None else {})}
        # The script's own output goes to the container's log: losses and counts, never a row's words.
        done = run([python, "-I", script], env=script_env(os.environ if environ is None else environ, train),
                   timeout=exec_s, check=False)
        if done.returncode != 0:
            raise ModalJobError(f"the training script exited with {done.returncode}")
        receipt = json.loads((out / colab_train.RECEIPT).read_text(encoding="utf-8"))
        stage = root / "stage" / name
        _checked(out, stage, name, recipe.get("rendering"))
        _land(stage, target)
        files = {file: (stage / file).read_bytes() for file in config.ADAPTER_FILES}
        projector = out / channel_train.PROJECTOR_FILE  # a channel run's projector: back to the host, never landed
        trained = {"projector": projector.read_bytes()} if projector.is_file() else {}
    return {"adapter": name, "receipt": receipt, "files": files, **trained}
