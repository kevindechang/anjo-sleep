"""Runs on the Colab VM, through ``colab exec -f`` (``colab_vllm``): Glimmer's weights and its vLLM
server. Standard library only, and it imports nothing from Anjo: the VM holds vLLM and this file.

Everything but the key arrives in the kernel's environment (``colab exec --env``):

- ``ANJO_COLAB_STEP``: ``download`` (the pinned revision into the weights folder, blocking),
  ``start`` (vLLM in the background, under the session's hard timeout) or ``mkdir`` (an adapter
  folder under the LoRA directory);
- ``ANJO_GLIMMER_MODEL``, ``ANJO_GLIMMER_REVISION`` and ``ANJO_GLIMMER_DIR`` for the download;
- ``ANJO_VLLM_COMMAND`` (the server's argv, JSON), ``ANJO_VLLM_ENV`` (its environment beside the
  key, JSON), ``ANJO_VLLM_SECONDS`` (its hard timeout) and ``ANJO_VLLM_KEY_FILE`` for the start:
  the file the session's own random key was uploaded to. The step reads it, deletes it, and puts
  the key in vLLM's environment as ``VLLM_API_KEY``, so it is never in a command line;
- ``ANJO_COLAB_DIR`` for ``mkdir``.

Each step prints ``ANJO_STEP_OK <step>`` when it finished, which the caller checks.

The server binds the VM's loopback only; the tunnel's far end is the one way in.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

STEP_ENV = "ANJO_COLAB_STEP"
STEP_OK = "ANJO_STEP_OK"  # colab_vllm.STEP_OK
LOG = "/content/vllm.log"


def _download() -> None:
    from huggingface_hub import snapshot_download  # installed with vLLM

    snapshot_download(os.environ["ANJO_GLIMMER_MODEL"], revision=os.environ["ANJO_GLIMMER_REVISION"],
                      local_dir=os.environ["ANJO_GLIMMER_DIR"])


def _start() -> None:
    command = json.loads(os.environ["ANJO_VLLM_COMMAND"])
    extra = json.loads(os.environ["ANJO_VLLM_ENV"])
    seconds = int(float(os.environ["ANJO_VLLM_SECONDS"]))
    key = _take_key(os.environ["ANJO_VLLM_KEY_FILE"])
    if not (isinstance(command, list) and command[:2] == ["vllm", "serve"] and key):
        raise SystemExit("the start step runs vllm serve, with its key set")
    env = {**os.environ, **{str(k): str(v) for k, v in extra.items()}, "VLLM_API_KEY": key}
    with open(LOG, "ab") as log:  # noqa: SIM115 - the server outlives this step and keeps writing
        process = subprocess.Popen(["timeout", str(seconds), *command], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"pid": process.pid}))


def _take_key(path: str) -> str:
    """The uploaded key, its file deleted."""
    try:
        with open(path, encoding="utf-8") as file:
            return file.read().strip()
    finally:
        os.remove(path)


def main() -> None:
    step = os.environ.get(STEP_ENV)
    if step == "download":
        _download()
    elif step == "start":
        _start()
    elif step == "mkdir":
        os.makedirs(os.environ["ANJO_COLAB_DIR"], exist_ok=True)
    else:
        sys.exit(f"{STEP_ENV} is download, start or mkdir")
    print(f"{STEP_OK} {step}")


if __name__ == "__main__":
    main()
