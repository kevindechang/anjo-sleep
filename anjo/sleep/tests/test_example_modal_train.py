"""The example's Modal training runner: its local half, with the remote faked (Task 15).

The Modal app itself (``train_app``) needs the ``modal`` package and a GPU, so it is checked only
for staying equal to the engine's own constants.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import anjo.sleep
from anjo.sleep.budget import TRAINING_TIMEOUT_S
from anjo.sleep.examples.synthetic.modal_train import (
    ModalTrainError,
    ModalTrainRunner,
    RemoteJob,
)
from anjo.sleep.serving import config
from anjo.sleep.train import colab_train
from anjo.sleep.train.runners.colab import TRAIN_PACKAGES

APP = Path(anjo.sleep.__file__).parent / "examples" / "synthetic" / "train_app.py"
JOB = "train_lora"


def _constants() -> dict[str, object]:
    found: dict[str, object] = {}
    for node in ast.parse(APP.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                found[node.target.id] = ast.literal_eval(node.value) if node.value else None
            except ValueError:
                continue
    return found


def test_train_app_constants_match_the_engines() -> None:
    constants = _constants()

    assert constants["PACKAGES"] == TRAIN_PACKAGES
    assert constants["GPU"] == config.BF16_A100.gpu
    assert constants["WEIGHTS_VOLUME"] == config.WEIGHTS_VOLUME
    assert constants["WEIGHTS_DIR"] == config.WEIGHTS_DIR
    assert constants["TIMEOUT_S"] == TRAINING_TIMEOUT_S
    assert constants["OUTPUT_FILES"] == (*config.ADAPTER_FILES, colab_train.RECEIPT)


def _inputs(tmp_path: Path, *, prev: bool = False) -> dict[str, object]:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    (work / "dataset.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    previous = None
    if prev:
        folder = tmp_path / "prev"
        folder.mkdir(parents=True)
        for name in config.ADAPTER_FILES:
            (folder / name).write_bytes(name.encode())
        previous = str(folder)
    return {
        "dataset_path": str(work / "dataset.jsonl"), "workdir": str(work), "config": {"r": 16},
        "prev_adapter_dir": previous,
    }


def _files() -> dict[str, bytes]:
    return {
        "adapter_config.json": b"{}", "adapter_model.safetensors": b"w",
        colab_train.RECEIPT: json.dumps({"steps": 3, "epochs": 1, "gpu": "A", "train_minutes": 1.0,
                                         "dropped_truncated": 0}).encode(),
    }


def test_runner_sends_the_dataset_and_config_and_writes_the_returned_adapter(tmp_path: Path) -> None:
    seen: list[RemoteJob] = []

    def remote(job: RemoteJob) -> dict[str, bytes]:
        seen.append(job)
        return _files()

    out = ModalTrainRunner(remote).run(JOB, _inputs(tmp_path), timeout_s=60.0)

    assert seen[0].dataset == '{"id": "x"}\n' and json.loads(seen[0].config_json) == {"r": 16}
    assert seen[0].prev == {}
    folder = Path(str(out["adapter_dir"]))
    assert (folder / "adapter_model.safetensors").read_bytes() == b"w"
    assert out["steps"] == 3 and float(str(out["gpu_minutes"])) >= 0.0


def test_runner_sends_a_warm_starts_adapter_files(tmp_path: Path) -> None:
    seen: list[RemoteJob] = []

    def remote(job: RemoteJob) -> dict[str, bytes]:
        seen.append(job)
        return _files()

    ModalTrainRunner(remote).run(JOB, _inputs(tmp_path, prev=True), timeout_s=60.0)

    assert set(seen[0].prev) == set(config.ADAPTER_FILES)


def test_runner_refuses_another_job_and_a_missing_output(tmp_path: Path) -> None:
    with pytest.raises(ModalTrainError, match="only train_lora"):
        ModalTrainRunner(lambda job: _files()).run("gates", _inputs(tmp_path), timeout_s=60.0)
    with pytest.raises(ModalTrainError, match="adapter_model.safetensors"):
        ModalTrainRunner(lambda job: {"adapter_config.json": b"{}"}).run(
            JOB, _inputs(tmp_path / "again"), timeout_s=60.0,
        )
