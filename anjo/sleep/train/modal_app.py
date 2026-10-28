# mypy: disallow-untyped-decorators=False
# (modal has no type stubs here, so its decorators are untyped)
"""Anjo Sleep's training app on Modal (Task 5): ``train``, on one A100-80GB, and MIND's encoder
``encode_latents`` beside it.

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

``encode_latents`` reads a job's events through Glimmer's hidden states on the same image and GPU,
with the weights read-only and the latents volume read-write (``latent_job``); the host's side is
``latent_runner``, which holds before it calls. ``embed_rows`` sends the raw rows of Glimmer's input
embedding table for some token ids (no GPU, weights read-only), for ``serving.soft_tokens``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Final

from anjo.sleep.serving import config
from anjo.sleep.train import colab_train, latent_job, modal_job

modal: Any = importlib.import_module("modal")  # only the deploying machine has the package

REVISION: Final = config.ServeConfig.from_env().pinned_revision()  # at load: no deploy without a pinned revision
_TRAIN = Path(__file__).resolve().parent
_SLEEP = _TRAIN.parent
# The same import paths as on the deploying machine; anjo itself stays a namespace package.
_FILES: Final = (
    (_SLEEP / "__init__.py", "/root/anjo/sleep/__init__.py"),
    (_SLEEP / "serving" / "__init__.py", "/root/anjo/sleep/serving/__init__.py"),
    (_SLEEP / "serving" / "config.py", "/root/anjo/sleep/serving/config.py"),
    (_SLEEP / "serving" / "projector_math.py", "/root/anjo/sleep/serving/projector_math.py"),  # channel_train loads it by path
    (_TRAIN / "__init__.py", "/root/anjo/sleep/train/__init__.py"),
    (_TRAIN / "colab_train.py", modal_job.REMOTE_SCRIPT),
    (_TRAIN / "sleep_job.py", "/root/anjo/sleep/train/sleep_job.py"),  # sft_dpo_seed's run, beside the script
    (_TRAIN / "sleep_loss.py", "/root/anjo/sleep/train/sleep_loss.py"),  # its one loss
    (_TRAIN / "sleep_fold.py", "/root/anjo/sleep/train/sleep_fold.py"),  # its fold
    (_TRAIN / "channel_train.py", "/root/anjo/sleep/train/channel_train.py"),  # beside the script, which loads it by path
    (_TRAIN / "modal_job.py", "/root/anjo/sleep/train/modal_job.py"),
    (_TRAIN / "modal_app.py", "/root/anjo/sleep/train/modal_app.py"),
    (_TRAIN / "latent_job.py", "/root/anjo/sleep/train/latent_job.py"),
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
latents = modal.Volume.from_name(latent_job.VOLUME, create_if_missing=True)  # MIND's event latents


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


@app.function(
    gpu=latent_job.GPU, timeout=latent_job.FUNCTION_TIMEOUT_S, startup_timeout=modal_job.STARTUP_TIMEOUT_S,
    scaledown_window=latent_job.SCALEDOWN_S,
    volumes={config.WEIGHTS_DIR: weights.read_only(), latent_job.VOLUME_DIR: latents}, secrets=[],
)
def encode_latents(job: str, spec_json: str, exec_s: float) -> dict[str, Any]:
    """Read one job's uploaded events through Glimmer's hidden states (``latent_job.encode_on_volume``),
    after leaving this call's id in the job's folder, where a sweep finds it by the job's name."""
    job_dir = Path(latent_job.VOLUME_DIR) / latent_job.check_job(job)
    latents.reload()  # see the events the host uploaded after this container started
    latent_job.mark_call(job_dir, modal.current_function_call_id())
    latents.commit()
    done = latent_job.encode_on_volume(
        job_dir=job_dir, spec_json=spec_json, pinned=REVISION,
        weights_root=Path(config.WEIGHTS_DIR), exec_s=exec_s,
    )
    latents.commit()
    return done


@app.function(
    timeout=latent_job.EMBED_TIMEOUT_S, scaledown_window=latent_job.SCALEDOWN_S,
    volumes={config.WEIGHTS_DIR: weights.read_only()}, secrets=[],
)
def embed_rows(ids: list[int]) -> bytes:
    """The raw bf16 rows of Glimmer's input embedding table for ``ids`` (``latent_job.embedding_rows``).
    No hold of its own: only a caller holding one that covers it calls it (``tools.soft_tokens_live``)."""
    return latent_job.embedding_rows(modal_job.base_folder(Path(config.WEIGHTS_DIR), REVISION, REVISION), ids)
