# mypy: disallow-untyped-decorators=False
# (modal has no type stubs here, so its decorators are untyped)
"""Anjo Sleep's training app on Modal (Task 5): one function, ``train``, on one A100-80GB.

It is the Modal twin of the Colab VM: the image installs exactly the pins the training script names
(``colab_train.TORCH`` and ``TRAIN_PACKAGES``, the list the Colab runner installs too), and the call
runs that same script on the same inputs (``modal_job.train_on_volume``). Glimmer comes off the
serving app's weights volume, mounted read-only, at the revision this app was deployed pinned to, so
nothing downloads per night; the finished adapter, checked as an upload is, lands on the adapters
volume the server loads from, so nothing uploads either. The app holds no secret: no Hugging Face
token, nothing from ``.env``. The call has Modal's hard timeout (§12.1's two hours), a start limit,
and no retries (Modal's default), so a call runs at most once.

The night's runner (``runners/modal.py``) calls the deployed function by name and never imports
this file. Deploying costs nothing until a call runs; only the coordinator deploys, on Kevin's go
(the commands are in deploy/dev_host.md). Loading this file without a pinned revision raises, so a
deploy without one fails before anything is built:

    ANJO_SLEEP_GLIMMER_REVISION=<commit> modal deploy anjo/sleep/train/modal_app.py

The weights volume must already hold that revision (the serving app's ``download_weights``).
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

from anjo.sleep.serving import config
from anjo.sleep.train import colab_train, modal_job

modal: Any = importlib.import_module("modal")  # only the deploying machine has the package

REVISION: Final = config.ServeConfig.from_env().pinned_revision()  # at load: no deploy without a pinned revision
_TRAIN = Path(__file__).resolve().parent
_SLEEP = _TRAIN.parent
# The same import paths as on the deploying machine; anjo itself stays a namespace package.
_FILES: Final = (
    (_SLEEP / "__init__.py", "/root/anjo/sleep/__init__.py"),
    (_SLEEP / "serving" / "__init__.py", "/root/anjo/sleep/serving/__init__.py"),
    (_SLEEP / "serving" / "config.py", "/root/anjo/sleep/serving/config.py"),
    (_TRAIN / "__init__.py", "/root/anjo/sleep/train/__init__.py"),
    (_TRAIN / "colab_train.py", modal_job.REMOTE_SCRIPT),
    (_TRAIN / "modal_job.py", "/root/anjo/sleep/train/modal_job.py"),
    (_TRAIN / "modal_app.py", "/root/anjo/sleep/train/modal_app.py"),
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(colab_train.TORCH, *colab_train.TRAIN_PACKAGES)
    .env(modal_job.image_env(REVISION))
)
for _local, _remote in _FILES:
    image = image.add_local_file(str(_local), _remote)
app = modal.App(modal_job.APP_NAME, image=image)
weights = modal.Volume.from_name(config.WEIGHTS_VOLUME)  # filled by the serving app; never made here
adapters = modal.Volume.from_name(config.ADAPTERS_VOLUME, create_if_missing=True)


@app.function(
    gpu=modal_job.GPU, timeout=modal_job.FUNCTION_TIMEOUT_S, startup_timeout=modal_job.STARTUP_TIMEOUT_S,
    volumes={config.WEIGHTS_DIR: weights.read_only(), config.ADAPTERS_DIR: adapters}, secrets=[],
)
def train(dataset: str, config_json: str, prev: dict[str, bytes], name: str, exec_s: float) -> dict[str, Any]:
    """Train one night's adapter and land it on the adapters volume (``modal_job.train_on_volume``)."""
    adapters.reload()  # see every adapter landed since this container started: a name is never reused
    done = modal_job.train_on_volume(
        dataset=dataset, config_json=config_json, prev=prev, name=name, exec_s=exec_s, pinned=REVISION,
        weights_root=Path(config.WEIGHTS_DIR), adapters_root=Path(config.ADAPTERS_DIR),
    )
    adapters.commit()
    return done
