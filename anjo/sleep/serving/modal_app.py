# mypy: disallow-untyped-decorators=False
# (modal has no type stubs here, so its decorators are untyped)
"""Anjo Sleep's serving app on Modal: Glimmer and each person's adapter behind vLLM (Task 4).

Modal only serves; training runs on Colab (Task 5). Three things live here:

- ``download_weights``: fetches the pinned Glimmer revision onto the weights volume, once.
- ``Serve``: vLLM's OpenAI-compatible server on one GPU at the precision ``config.SERVING``
  names, with LoRA on, so the base model and every personal adapter share it. It scales to zero
  after ten idle minutes, and every function has a timeout.
- ``put_adapter`` and the ``upload_adapter`` entrypoint: put one checked adapter folder on the
  adapters volume under its name. A running container sees a new adapter after its next start.

Deploying and calling cost money, so only the coordinator runs them, on Kevin's go:

    ANJO_SLEEP_GLIMMER_REVISION=<commit> modal deploy anjo/sleep/serving/modal_app.py
    modal run anjo/sleep/serving/modal_app.py::download_weights
    modal run anjo/sleep/serving/modal_app.py::upload_adapter --folder <dir> --name p-<hash>-n<night>

The deploy-time settings (``config.ServeConfig.from_env``) are baked into the image's environment,
so the container reads the same values. The image carries only this file and ``config.py``.
"""
from __future__ import annotations

import importlib
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from anjo.sleep.serving import config

modal: Any = importlib.import_module("modal")  # only the deploying machine has the package

SETTINGS = config.ServeConfig.from_env()
_HERE = Path(__file__).resolve().parent
_HEALTH_POLL_S = 5.0


def _settings_env(settings: config.ServeConfig) -> dict[str, str]:
    """The deploy's settings as the container's environment, so ``from_env`` reads them there."""
    env = {config.VLLM_VERSION_ENV: settings.vllm_version}
    if settings.revision is not None:
        env[config.REVISION_ENV] = settings.revision
    if settings.tool_parser is not None:
        env[config.TOOL_PARSER_ENV] = settings.tool_parser
    if settings.gpu_snapshot:
        env[config.SNAPSHOT_ENV] = "1"
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
secret = modal.Secret.from_name(config.SECRET_NAME)

# Modal's GPU memory snapshot, tried to cut the cold start; the wake time with and without it is
# measured and recorded in deploy/dev_host.md.
_SNAPSHOT = SETTINGS.gpu_snapshot
_SNAPSHOT_OPTIONS: dict[str, Any] = (
    {"enable_memory_snapshot": True, "experimental_options": {"enable_gpu_snapshot": True}}
    if _SNAPSHOT else {}
)


@app.function(
    volumes={config.WEIGHTS_DIR: weights}, secrets=[secret], timeout=config.DOWNLOAD_TIMEOUT_S,
)
def download_weights() -> str:
    """Fetch the pinned revision onto the weights volume; HF_TOKEN comes from the secret."""
    from huggingface_hub import snapshot_download

    revision = SETTINGS.pinned_revision()
    path = config.weights_path(revision)
    snapshot_download(config.MODEL, revision=revision, local_dir=path)
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


@app.cls(
    gpu=SETTINGS.precision.gpu,
    volumes={config.WEIGHTS_DIR: weights, config.ADAPTERS_DIR: adapters},
    secrets=[secret],
    timeout=config.SERVE_TIMEOUT_S,
    scaledown_window=config.SCALEDOWN_S,
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

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()


def put_adapter(folder: Path, name: str) -> None:
    """Upload one adapter folder under its name. It is checked first, and a name already on the
    volume is refused rather than overwritten: each night's adapter has its own name."""
    config.check_adapter(folder, name)
    volume = modal.Volume.from_name(config.ADAPTERS_VOLUME, create_if_missing=True)
    with volume.batch_upload() as batch:
        batch.put_directory(str(folder), f"/{name}")


@app.local_entrypoint()
def upload_adapter(folder: str, name: str) -> None:
    put_adapter(Path(folder), name)
