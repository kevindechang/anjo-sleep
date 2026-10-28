# mypy: disallow-untyped-decorators=False
# (modal has no type stubs here, so its decorators are untyped)
"""Anjo Sleep's serving app on Modal: Glimmer and each person's adapter behind vLLM (Task 4).

This app only serves. Training is the training app's (``train/modal_app.py``, Task 5), which mounts
the weights volume read-only and lands each adapter on the adapters volume. Three things live here:

- ``download_weights``: fetches the pinned Glimmer revision onto the weights volume, once. It
  holds no secret unless the deploy names a Hugging Face one (the repo isn't gated).
- ``Serve``: vLLM's OpenAI-compatible server on one GPU at the precision ``config.SERVING``
  names, with LoRA on, so the base model and every personal adapter share it. It holds only the
  serving secret (VLLM_API_KEY). It runs one container at most (``config.MAX_CONTAINERS``), scales
  to zero after ten idle minutes, and every function has a timeout. vLLM loads an adapter only
  when the client asks it to (``/v1/load_lora_adapter``, ``serving.client``).
- ``put_adapter`` and the ``upload_adapter`` entrypoint: put one checked adapter folder on the
  adapters volume under its name, never over an adapter already there. A running container mounts
  the volume as it was when the container started, and nothing reloads it. On night 1c (2026-10-28)
  the server stopped answering, ``/v1/models`` included, before the night had sent any adapter load;
  the stall was in this container's Modal process, where the volume reload thread ran (on night 1b
  the adapter load itself succeeded). An adapter committed since the container started is copied
  onto its own disk by ``Serve.stage`` (``stage_adapter``), which the client calls when vLLM can't
  find the adapter on the mount.

Deploying and calling cost money, so only the coordinator runs them, on Kevin's go (the pinned
commit is in deploy/dev_host.md). Loading this file without a pinned revision raises, so a deploy
without one fails before anything is built:

    ANJO_SLEEP_GLIMMER_REVISION=<commit> modal deploy anjo/sleep/serving/modal_app.py
    ANJO_SLEEP_GLIMMER_REVISION=<commit> modal run anjo/sleep/serving/modal_app.py::download_weights
    ANJO_SLEEP_GLIMMER_REVISION=<commit> modal run anjo/sleep/serving/modal_app.py::upload_adapter \
        --folder <dir> --name p-<hash>-n<night>

The deploy-time settings (``config.ServeConfig.from_env``) are baked into the image's environment,
so the container reads the same values. The image carries only this file and ``config.py``.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any, Final

from anjo.sleep.serving import config

modal: Any = importlib.import_module("modal")  # only the deploying machine has the package

SETTINGS = config.ServeConfig.from_env()
REVISION: Final = SETTINGS.pinned_revision()  # at load: no deploy without a pinned revision
_HERE = Path(__file__).resolve().parent
_HEALTH_POLL_S = 5.0


def _settings_env(settings: config.ServeConfig) -> dict[str, str]:
    """The deploy's settings as the container's environment, so ``from_env`` reads them there."""
    env = {config.VLLM_VERSION_ENV: settings.vllm_version, config.REVISION_ENV: settings.pinned_revision()}
    if settings.tool_parser is not None:
        env[config.TOOL_PARSER_ENV] = settings.tool_parser
    if settings.gpu_snapshot:
        env[config.SNAPSHOT_ENV] = "1"
    if settings.hf_secret is not None:
        env[config.HF_SECRET_ENV] = settings.hf_secret
    return env


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={SETTINGS.vllm_version}", "hf_transfer==0.1.9")
    .env({**config.vllm_env(), **_settings_env(SETTINGS)})
    # The same import path as on the deploying machine; anjo itself stays a namespace package.
    .add_local_file(_HERE.parent / "__init__.py", "/root/anjo/sleep/__init__.py")
    .add_local_file(_HERE / "__init__.py", "/root/anjo/sleep/serving/__init__.py")
    .add_local_file(_HERE / "config.py", "/root/anjo/sleep/serving/config.py")
)
app = modal.App(config.APP_NAME, image=image)
weights = modal.Volume.from_name(config.WEIGHTS_VOLUME, create_if_missing=True)
adapters = modal.Volume.from_name(config.ADAPTERS_VOLUME, create_if_missing=True)
serve_secret = modal.Secret.from_name(config.SECRET_NAME)
download_secrets = [] if SETTINGS.hf_secret is None else [modal.Secret.from_name(SETTINGS.hf_secret)]

# Modal's GPU memory snapshot, tried to cut the cold start; the wake time with and without it is
# measured and recorded in deploy/dev_host.md.
_SNAPSHOT = SETTINGS.gpu_snapshot
_SNAPSHOT_OPTIONS: dict[str, Any] = (
    {"enable_memory_snapshot": True, "experimental_options": {"enable_gpu_snapshot": True}}
    if _SNAPSHOT else {}
)


@app.function(
    volumes={config.WEIGHTS_DIR: weights}, secrets=download_secrets, timeout=config.DOWNLOAD_TIMEOUT_S,
)
def download_weights() -> str:
    """Fetch the pinned revision onto the weights volume, with HF_TOKEN only when the deploy named
    a Hugging Face secret."""
    from huggingface_hub import snapshot_download

    path = config.weights_path(REVISION)
    snapshot_download(config.MODEL, revision=REVISION, local_dir=path)
    weights.commit()
    return path


def _wait_until_healthy(process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{config.PORT}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with {process.returncode} before it was healthy")
        try:
            with urllib.request.urlopen(url, timeout=_HEALTH_POLL_S) as reply:
                if reply.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass  # still loading: poll again until the deadline
        time.sleep(_HEALTH_POLL_S)
    raise TimeoutError(f"vLLM was not healthy within {timeout_s:.0f} s")


def stage_adapter(read: Callable[[str], Iterable[bytes]], name: str, root: Path) -> bool:
    """Copy adapter ``name`` from the adapters volume onto this container's disk under ``root``, so
    vLLM loads it from there. ``read`` is the volume's API (``Volume.read_file``), never its mount, so
    nothing reloads the mount vLLM reads. True once the adapter is under ``root``; False when the
    volume has no such adapter. The name is checked before anything is read, the copy is checked as
    an upload is, and it lands whole, by one rename, or not at all."""
    config.check_adapter_name(name)
    target = root / name
    if target.is_dir():
        return True
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".staging-") as scratch:
        folder = Path(scratch) / name
        folder.mkdir()
        try:
            for file in config.ADAPTER_FILES:
                with (folder / file).open("wb") as out:
                    for chunk in read(f"/{name}/{file}"):
                        out.write(chunk)
        except FileNotFoundError:
            return False
        config.check_adapter(folder, name)
        try:
            os.replace(folder, target)
        except OSError:
            if not target.is_dir():  # not a second stage of the same adapter that landed first
                raise
    return True


@app.cls(
    gpu=SETTINGS.precision.gpu,
    volumes={config.WEIGHTS_DIR: weights, config.ADAPTERS_DIR: adapters},
    secrets=[serve_secret],
    timeout=config.SERVE_TIMEOUT_S,
    scaledown_window=config.SCALEDOWN_S,
    max_containers=config.MAX_CONTAINERS,
    **_SNAPSHOT_OPTIONS,
)
@modal.concurrent(max_inputs=config.CONCURRENT_REQUESTS)
class Serve:
    @modal.enter(snap=_SNAPSHOT)
    def start(self) -> None:
        self.process = subprocess.Popen(config.vllm_command(SETTINGS))
        _wait_until_healthy(self.process, config.STARTUP_TIMEOUT_S)

    @modal.web_server(port=config.PORT, startup_timeout=config.STARTUP_TIMEOUT_S)
    def web(self) -> None:
        """The OpenAI-compatible server the dev host calls. vLLM checks VLLM_API_KEY."""

    @modal.method()
    def stage(self, name: str) -> bool:
        """Copy an adapter committed since this container started onto its own disk
        (``config.STAGED_ADAPTERS_DIR``); False when the volume has no such adapter. One container
        runs at most, so this is the container vLLM serves from."""
        return stage_adapter(adapters.read_file, name, Path(config.STAGED_ADAPTERS_DIR))

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


def put_adapter(folder: Path, name: str) -> None:
    """Upload one adapter folder under its name. It is checked first, and a name already on the
    volume is refused rather than overwritten, so a pinned night's adapter never changes. Modal's
    own upload refuses to overwrite as well (force=False), for two uploads racing."""
    config.check_adapter(folder, name)
    volume = modal.Volume.from_name(config.ADAPTERS_VOLUME, create_if_missing=True)
    if name in {PurePosixPath(entry.path).name for entry in volume.listdir("/")}:
        raise config.ServeConfigError(f"adapter {name} is already on the volume; a night's adapter never changes")
    with volume.batch_upload(force=False) as batch:
        batch.put_directory(str(folder), f"/{name}")


@app.local_entrypoint()
def upload_adapter(folder: str, name: str) -> None:
    put_adapter(Path(folder), name)
