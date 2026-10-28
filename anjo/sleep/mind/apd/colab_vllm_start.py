"""Runs on the Colab VM, through ``colab exec -f`` (``colab_vllm``): Glimmer's weights and its vLLM
server. Standard library only, and it imports nothing from Anjo: the VM holds vLLM and this file.

Everything arrives in the kernel's environment (``colab exec --env``), never as a file:

- ``ANJO_COLAB_STEP``: ``download`` (the pinned revision into the weights folder, blocking),
  ``start`` (vLLM in the background, under the session's hard timeout) or ``mkdir`` (an adapter
  folder under the LoRA directory);
- ``ANJO_GLIMMER_MODEL``, ``ANJO_GLIMMER_REVISION`` and ``ANJO_GLIMMER_DIR`` for the download;
- ``ANJO_VLLM_COMMAND`` (the server's argv, JSON), ``ANJO_VLLM_ENV`` (its environment beside the
  key, JSON), ``ANJO_VLLM_SECONDS`` (its hard timeout) and ``VLLM_API_KEY`` (the session's own
  random key, which vLLM reads from its environment) for the start;
- ``ANJO_COLAB_DIR`` for ``mkdir``.

The server binds the VM's loopback only; the tunnel's far end is the one way in.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

STEP_ENV = "ANJO_COLAB_STEP"
LOG = "/content/vllm.log"


def _download() -> None:
    from huggingface_hub import snapshot_download  # installed with vLLM

    snapshot_download(os.environ["ANJO_GLIMMER_MODEL"], revision=os.environ["ANJO_GLIMMER_REVISION"],
                      local_dir=os.environ["ANJO_GLIMMER_DIR"])


def _start() -> None:
    command = json.loads(os.environ["ANJO_VLLM_COMMAND"])
    extra = json.loads(os.environ["ANJO_VLLM_ENV"])
    seconds = int(float(os.environ["ANJO_VLLM_SECONDS"]))
    if not (isinstance(command, list) and command[:2] == ["vllm", "serve"] and os.environ.get("VLLM_API_KEY")):
        raise SystemExit("the start step runs vllm serve, with its key set")
    env = {**os.environ, **{str(k): str(v) for k, v in extra.items()}}
    os.makedirs(env.get("VLLM_LORA_RESOLVER_CACHE_DIR", "/content/adapters"), exist_ok=True)
    with open(LOG, "ab") as log:  # noqa: SIM115 - the server outlives this step and keeps writing
        process = subprocess.Popen(["timeout", str(seconds), *command], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"pid": process.pid}))


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


if __name__ == "__main__":
    main()
