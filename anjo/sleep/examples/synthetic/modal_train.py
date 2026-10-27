"""A ``GpuRunner`` for the training job, on your own Modal account.

It runs the same self-contained script the Colab runner runs (``anjo.sleep.train.colab_train``):
Glimmer in 4-bit NF4, a LoRA of rank 16 on the attention and MLP projections, the loss on her
replies only. One call is one container on one A100-80GB (``train_app``); Modal bills for the
minutes it ran. Modal is imported only when a live run reaches ``run``.
"""
from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.serving import config
from anjo.sleep.train import colab_train
from anjo.sleep.train.lora import JOB

APP_MODULE: Final = "anjo.sleep.examples.synthetic.train_app"
OUTPUT_FILES: Final = (*config.ADAPTER_FILES, colab_train.RECEIPT)


class ModalTrainError(RuntimeError):
    """The Modal job didn't return an adapter."""


@dataclass(frozen=True)
class RemoteJob:
    dataset: str
    config_json: str
    prev: Mapping[str, bytes]  # a warm start's adapter files; empty for a clean slate


Remote = Callable[[RemoteJob], Mapping[str, bytes]]


def modal_remote(job: RemoteJob) -> Mapping[str, bytes]:
    """Call ``train_app.train`` on Modal and wait for the adapter's files."""
    modal: Any = importlib.import_module("modal")
    app: Any = importlib.import_module(APP_MODULE)
    with modal.enable_output(), app.app.run():
        files: Mapping[str, bytes] = app.train.remote(job.dataset, job.config_json, dict(job.prev))
    return files


class ModalTrainRunner:
    """``GpuRunner`` for the ``train_lora`` job. ``remote`` is Modal's by default; tests pass a
    fake."""

    def __init__(self, remote: Remote = modal_remote) -> None:
        self._remote = remote

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        if job != JOB:
            raise ModalTrainError(f"this runner runs only {JOB}")
        prev_dir = inputs.get("prev_adapter_dir")
        prev = {} if prev_dir is None else {
            name: (Path(str(prev_dir)) / name).read_bytes() for name in config.ADAPTER_FILES
        }
        spec = RemoteJob(
            dataset=Path(str(inputs["dataset_path"])).read_text(encoding="utf-8"),
            config_json=json.dumps(inputs["config"], sort_keys=True), prev=prev,
        )
        started = time.monotonic()
        files = self._remote(spec)
        minutes = (time.monotonic() - started) / 60.0
        out = Path(str(inputs["workdir"])) / "out"
        out.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_FILES:
            if name not in files:
                raise ModalTrainError(f"the Modal job returned no {name}")
            (out / name).write_bytes(files[name])
        receipt = json.loads((out / colab_train.RECEIPT).read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ModalTrainError("the training receipt is not a record")
        return {**receipt, "adapter_dir": str(out), "gpu_minutes": minutes, "compute_units": 0.0}
