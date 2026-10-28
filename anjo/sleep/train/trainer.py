"""The night's ``Trainer`` (Task 5): ``train_lora_with_receipt`` on a GPU runner (Modal by default, or
Colab: ``runners/choice.py``), publishing each finished adapter to the server under its never-reused name.

It is also the thought channel's ``ChannelTrainer`` (``night.channel_night``): ``train_with_channel``
trains the projector with the LoRA in the same job, warm from ``projector`` (the person's own, or the
shared one), its shape read from that file. The projector the night trained is kept beside the
adapter's local copy, ``<root>/adapters/<name>.projector.npz``, and named in the receipt."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.budget import TRAINING_TIMEOUT_S
from anjo.sleep.night.core_snapshot import ChannelArm
from anjo.sleep.night.deps import TrainingData
from anjo.sleep.ports import GpuRunner
from anjo.sleep.train import channel_train, modal_job
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.train.lora import AdapterPublisher, LoraCfg, train_lora_with_receipt
from anjo.sleep.types import CheckpointRef


# The receipt's keys that only a switch fills: the sleep objective's, the surprise weights', a matched
# control's and the thought channel's. Left out when empty.
OPTIONAL_KEYS: Final = ("terms", "surprise_weights", "match", "projector", "channel_receipt")


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
        return self._train(user_id, night, prev, data, name)

    def train_with_channel(
        self, user_id: str, night: date, prev: CheckpointRef | None, data: TrainingData, name: str, *,
        arm: ChannelArm, projector: Path,
    ) -> tuple[CheckpointRef, Mapping[str, object]]:
        _, meta = channel_train.load_params(projector.read_bytes())
        block = channel_train.block(k=int(meta["k"]), d=int(meta["d"]), width=int(meta["width"]), arm=arm,
                                    exec_s=modal_job.exec_seconds(TRAINING_TIMEOUT_S))
        return self._train(user_id, night, prev, data, name, channel=block, prev_projector=projector)

    def _train(
        self, user_id: str, night: date, prev: CheckpointRef | None, data: TrainingData, name: str, **channel: Any,
    ) -> tuple[CheckpointRef, Mapping[str, object]]:
        if not isinstance(data, Dataset):
            raise TypeError("the trainer takes the night's dataset (build_night_dataset)")
        ckpt, receipt = train_lora_with_receipt(
            prev, data, self._cfg, self._runner, person_id=user_id, night=night, name=name,
            workdir=self._root / "work" / name, adapter_cache=self._root / "adapters",
            publish=self._publish, revision=self._revision, **channel,
        )
        done = dataclasses.asdict(receipt)
        for key in OPTIONAL_KEYS:  # each only when it applies: a night with every switch off logs today's receipt
            if done[key] in (None, ""):
                del done[key]
        return ckpt, done
