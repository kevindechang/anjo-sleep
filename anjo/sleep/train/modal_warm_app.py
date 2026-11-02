# mypy: disallow-untyped-decorators=False
# (modal has no type stubs here, so its decorators are untyped)
"""The judges' warm trainer on Modal (2026-11-02): ``WarmTrainer``, its own app, one A100-80GB.

``warm_job.py`` has the why. The image is the cold app's (``modal_app.py``): the same pins, the same
files at the same paths, plus ``warm_job.py`` and this file. The weights volume is mounted
read-only and the adapters volume read-write, as there. No secret. Modal's hard timeout, a start
limit, no retries, one container at most (``warm_job.MAX_CONTAINERS``), and a scale-down window read
from the deploying machine's ``ANJO_SLEEP_JUDGE_TRAIN_SCALEDOWN_S`` (90 s by default).

The host calls the deployed class by name (``runners/warm.py``) and never imports this file. Only
the coordinator deploys it, on Kevin's go; deploying costs nothing until a call runs:

    ANJO_SLEEP_GLIMMER_REVISION=<commit> modal deploy anjo/sleep/train/modal_warm_app.py
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Final

from anjo.sleep.serving import config
from anjo.sleep.train import colab_train, modal_job, warm_job

modal: Any = importlib.import_module("modal")  # only the deploying machine has the package

REVISION: Final = config.ServeConfig.from_env().pinned_revision()  # at load: no deploy without a pinned revision
SCALEDOWN_S: Final = warm_job.scaledown_seconds(os.environ)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(colab_train.TORCH, *colab_train.TRAIN_PACKAGES)
    .env(modal_job.image_env(REVISION))
)
for _local, _remote in warm_job.WARM_FILES:
    image = image.add_local_file(str(_local), _remote)
app = modal.App(warm_job.APP_NAME, image=image)
weights = modal.Volume.from_name(config.WEIGHTS_VOLUME)  # filled by the serving app; never made here
adapters = modal.Volume.from_name(config.ADAPTERS_VOLUME, create_if_missing=True)


@app.cls(
    gpu=modal_job.GPU, timeout=modal_job.FUNCTION_TIMEOUT_S, startup_timeout=modal_job.STARTUP_TIMEOUT_S,
    scaledown_window=SCALEDOWN_S, max_containers=warm_job.MAX_CONTAINERS,
    volumes={config.WEIGHTS_DIR: weights.read_only(), config.ADAPTERS_DIR: adapters}, secrets=[],
)
class WarmTrainer:
    @modal.enter()
    def load(self) -> None:
        """Glimmer, once per container, from the pinned revision on the weights volume."""
        base = modal_job.base_folder(Path(config.WEIGHTS_DIR), REVISION, REVISION)
        self.loaded = warm_job.load(base)

    @modal.method()
    def warm(self) -> dict[str, Any]:
        """The host's keep-warm call: nothing but the container's state."""
        return {"loaded_s": round(self.loaded.seconds, 1), "clean": self.loaded.clean}

    @modal.method()
    def train(self, dataset: str, config_json: str, prev: dict[str, bytes], name: str) -> dict[str, Any]:
        """Train one night's adapter on the loaded model and land it on the adapters volume."""
        adapters.reload()  # see every adapter landed since this container started: a name is never reused
        try:
            done = warm_job.train_warm_on_volume(
                dataset=dataset, config_json=config_json, prev=prev, name=name, pinned=REVISION,
                weights_root=Path(config.WEIGHTS_DIR), adapters_root=Path(config.ADAPTERS_DIR),
                train=lambda cfg, data, warm, out: warm_job.train_loaded(self.loaded, cfg, data, warm, out),
            )
        finally:
            if not self.loaded.clean:  # never a second call on a base that kept an adapter's layers
                modal.experimental.stop_fetching_inputs()
        adapters.commit()
        return done
