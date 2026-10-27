# mypy: disallow-untyped-decorators=False
# (modal has no type stubs here, so its decorators are untyped)
"""The Modal app that trains the example's LoRA (one function, one A100-80GB).

It is the Modal twin of the Colab runner's VM: the same self-contained training script
(``anjo.sleep.train.colab_train``) runs in a container built from pinned packages. The dataset goes
up as text, the adapter's files come back as bytes, and the container is gone when the call returns.
``ModalTrainRunner`` imports this file only for a live run, so ``--fake`` never needs ``modal``.

The container imports nothing from Anjo but this file, so the image carries the three package
markers it needs to import it under its own name, and the training script. The constants below are
copies of the engine's (the Colab runner's packages, the serving config's GPU and the training
timeout); a test keeps them equal.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

modal: Any = importlib.import_module("modal")  # only a live run has the package

APP_NAME: Final = "anjo-sleep-example-train"
GPU: Final = "A100-80GB"
TIMEOUT_S: Final = 7200  # two hours, the engine's training timeout
TORCH: Final = "torch==2.8.0"
PACKAGES: Final = (
    "transformers==4.57.1", "peft==0.17.1", "trl==0.24.0", "bitsandbytes==0.48.1",
    "accelerate==1.10.1", "datasets==4.2.0",
)
REMOTE_SCRIPT: Final = "/root/colab_train.py"
OUTPUT_FILES: Final = ("adapter_config.json", "adapter_model.safetensors", "receipt.json")

_HERE = Path(__file__).resolve().parent
_SLEEP = _HERE.parents[1]
_MARKERS = (
    (_SLEEP / "__init__.py", "/root/anjo/sleep/__init__.py"),
    (_HERE.parent / "__init__.py", "/root/anjo/sleep/examples/__init__.py"),
    (_HERE / "__init__.py", "/root/anjo/sleep/examples/synthetic/__init__.py"),
    (_HERE / "train_app.py", "/root/anjo/sleep/examples/synthetic/train_app.py"),
)

image = modal.Image.debian_slim(python_version="3.12").pip_install(TORCH, *PACKAGES)
for _local, _remote in _MARKERS:
    image = image.add_local_file(str(_local), _remote)
image = image.add_local_file(str(_SLEEP / "train" / "colab_train.py"), REMOTE_SCRIPT)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_S)
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
    subprocess.run([sys.executable, REMOTE_SCRIPT], env=env, check=True)
    return {name: (out / name).read_bytes() for name in OUTPUT_FILES}
