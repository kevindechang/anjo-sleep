# mypy: disallow-untyped-decorators=False
# (modal has no type stubs here, so its decorators are untyped)
"""The Modal app that trains the example's LoRA (one function, one A100-80GB).

It is the Modal twin of the Colab runner's VM: the same self-contained training script
(``anjo.sleep.train.colab_train``) runs in a container built from pinned packages. The dataset goes
up as text, the adapter's files come back as bytes, and the container is gone when the call returns.
The weights volume is the one the serving app's ``download_weights`` fills; here it holds the
Hugging Face cache (fetched with ``hf_transfer``), so the base weights download once, not every run.
``ModalTrainRunner`` imports this file only for a live run, so ``--fake`` never needs ``modal``.

The container imports nothing from Anjo but this file and ``colab_train`` (standard library only), so
the image carries the package markers and those two files at their own import paths. The package
pins and the training script are the engine's: ``colab_train.TRAIN_PACKAGES`` and ``TORCH``, and the
one copy of the script the image runs. The constants below are copies of the serving config's GPU
and volume and the training timeout; a test keeps them equal.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

from anjo.sleep.train import colab_train

modal: Any = importlib.import_module("modal")  # only a live run has the package

APP_NAME: Final = "anjo-sleep-example-train"
GPU: Final = "A100-80GB"
TIMEOUT_S: Final = 7200  # two hours, the engine's training timeout
REMOTE_SCRIPT: Final = "/root/anjo/sleep/train/colab_train.py"
OUTPUT_FILES: Final = ("adapter_config.json", "adapter_model.safetensors", "receipt.json")

_HERE = Path(__file__).resolve().parent
_SLEEP = _HERE.parents[1]
_MARKERS = (
    (_SLEEP / "__init__.py", "/root/anjo/sleep/__init__.py"),
    (_HERE.parent / "__init__.py", "/root/anjo/sleep/examples/__init__.py"),
    (_HERE / "__init__.py", "/root/anjo/sleep/examples/synthetic/__init__.py"),
    (_HERE / "train_app.py", "/root/anjo/sleep/examples/synthetic/train_app.py"),
    (_SLEEP / "train" / "__init__.py", "/root/anjo/sleep/train/__init__.py"),
    (_SLEEP / "train" / "colab_train.py", REMOTE_SCRIPT),
)

WEIGHTS_VOLUME: Final = "anjo-sleep-weights"  # the volume the serving app's download_weights fills
WEIGHTS_DIR: Final = "/weights"
HF_CACHE: Final = "/weights/hf-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(colab_train.TORCH, *colab_train.TRAIN_PACKAGES, "hf_transfer==0.1.9")
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
for _local, _remote in _MARKERS:
    image = image.add_local_file(str(_local), _remote)
app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={WEIGHTS_DIR: weights})
def train(dataset: str, config_json: str, prev: dict[str, bytes]) -> dict[str, bytes]:
    """Write the inputs, run the training script and return its output files."""
    import os
    import subprocess
    import sys
    import tempfile

    root = Path(tempfile.mkdtemp())
    out = root / "out"
    out.mkdir()
    (root / "dataset.jsonl").write_text(dataset, encoding="utf-8")
    env = {
        **os.environ, "ANJO_TRAIN_CONFIG": config_json,
        "ANJO_TRAIN_DATASET": str(root / "dataset.jsonl"), "ANJO_TRAIN_OUT": str(out),
    }
    if prev:
        (root / "prev").mkdir()
        for name, data in prev.items():
            (root / "prev" / name).write_bytes(data)
        env["ANJO_TRAIN_PREV"] = str(root / "prev")
    try:
        subprocess.run([sys.executable, REMOTE_SCRIPT], env=env, check=True)
    finally:
        weights.commit()  # keep the downloaded weights, so the next run doesn't fetch them again
    return {name: (out / name).read_bytes() for name in OUTPUT_FILES}
