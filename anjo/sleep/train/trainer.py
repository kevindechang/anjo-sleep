"""The night's ``Trainer`` (Task 5): ``train_lora_with_receipt`` on a GPU runner (Colab for now),
publishing each finished adapter to the serving volume under its never-reused name.

The dataset holds the person's words in plain text, outside their encrypted folder, so it is
deleted as soon as the run ends, whether it trained or failed."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from anjo.sleep.night.deps import TrainingData
from anjo.sleep.ports import GpuRunner
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.train.lora import DATASET_FILE, AdapterPublisher, LoraCfg, train_lora_with_receipt
from anjo.sleep.types import CheckpointRef


class LoraTrainer:
    def __init__(
        self, runner: GpuRunner, *, root: Path, publish: AdapterPublisher, revision: str | None,
        cfg: LoraCfg | None = None,
    ) -> None:
        self._runner = runner
        self._root = root
        self._publish = publish
        self._revision = revision
        self._cfg = cfg if cfg is not None else LoraCfg()

    def train(
        self, user_id: str, night: date, prev: CheckpointRef | None, data: TrainingData, name: str,
    ) -> tuple[CheckpointRef, Mapping[str, object]]:
        if not isinstance(data, Dataset):
            raise TypeError("the trainer takes the night's dataset (build_night_dataset)")
        workdir = self._root / "work" / name
        try:
            ckpt, receipt = train_lora_with_receipt(
                prev, data, self._cfg, self._runner, person_id=user_id, night=night, name=name,
                workdir=workdir, adapter_cache=self._root / "adapters",
                publish=self._publish, revision=self._revision,
            )
        finally:
            (workdir / DATASET_FILE).unlink(missing_ok=True)
        return ckpt, dataclasses.asdict(receipt)
