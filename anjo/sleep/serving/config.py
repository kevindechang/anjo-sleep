"""What Glimmer's vLLM server runs with (Task 4).

Standard library only: the Modal image copies this file in as a module of its own, so it imports
nothing from Anjo. The chat side reads the same values, so the pin the dev host serves under and
the server Modal runs can't drift apart.

**Precision is one line.** ``SERVING`` is ``FP8_L40S`` (Glimmer quantized to FP8 when vLLM loads
it, on one L40S) or ``BF16_A100`` (bf16 on one A100-80GB, the fallback). The base model and every
personal adapter share one server, so every arm serves at the same precision by construction.

**Reasoning.** Glimmer answers in channels: ``to=self`` for its reasoning, closed by ``<|eom|>``,
then ``to=user`` for the reply, closed by ``<|eot|>``. In Sleep it always answers directly: every
call sends the chat's token ids with the reply's header (``USER_CHANNEL``) already written, to
``/v1/completions``, and stops at ``REPLY_STOPS``, the way the adapter is trained
(``serving.framing``). Its chat template writes "Reasoning strength: <level>." into the system
turn, 'high' unless told otherwise; every call and the training text are rendered at
``reasoning_strength()`` (``low`` by default). The ``muse_glimmer`` reasoning parser stays on for
a chat request, and never reads a completion.

**Adapters.** Each person's LoRA adapter is a folder ``p-<hash>-n<night>`` on the adapters volume.
The night's client loads it through vLLM's ``/v1/load_lora_adapter`` before the first request that
names it (``client.VllmClient``), so the base and the personal arms share one GPU. vLLM's
filesystem resolver is not used: on 0.28.0 it never loaded one (seen live, 2026-10-28; it compares
an adapter's base model with the path the server was started on, not the served name). A running
Modal container sees the adapters volume as it was when it started and never reloads it: on night 1c
(2026-10-28) the server wedged for good while it reloaded the volume and vLLM read a new adapter from
the mount. An
adapter committed since the container started is copied through the volume's API onto the
container's own disk, ``STAGED_ADAPTERS_DIR`` (``modal_app.Serve.stage``), and vLLM loads it from
there. ``check_adapter`` holds a folder to what the server needs before it is uploaded or staged.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

APP_NAME: Final = "anjo-sleep-serve"
WEIGHTS_VOLUME: Final = "anjo-sleep-weights"
ADAPTERS_VOLUME: Final = "anjo-sleep-adapters"
# The serving secret holds VLLM_API_KEY (the bearer token the dev host sends) and nothing else.
# Downloading needs no secret, since the Glimmer repo isn't gated: a deploy that wants HF_TOKEN
# anyway names a second Modal secret in HF_SECRET_ENV, and only download_weights gets it.
SECRET_NAME: Final = "anjo-sleep-serve"
HF_SECRET_ENV: Final = "ANJO_SLEEP_HF_SECRET"
WEIGHTS_DIR: Final = "/weights"
ADAPTERS_DIR: Final = "/adapters"
STAGED_ADAPTERS_DIR: Final = "/tmp/anjo-staged-adapters"  # the container's own disk, never a volume
PORT: Final = 8000

MODEL: Final = "meta-models/Muse-Glimmer-30B"
LICENSE: Final = "Apache-2.0"
REVISION_ENV: Final = "ANJO_SLEEP_GLIMMER_REVISION"
TOOL_PARSER_ENV: Final = "ANJO_SLEEP_TOOL_CALL_PARSER"
REASONING_STRENGTH_ENV: Final = "ANJO_SLEEP_REASONING_STRENGTH"
SNAPSHOT_ENV: Final = "ANJO_SLEEP_GPU_SNAPSHOT"
VLLM_VERSION_ENV: Final = "ANJO_SLEEP_VLLM_VERSION"
# The vLLM release both backends serve with (Modal's image installs it, SageMaker's image is built
# FROM vllm/vllm-openai at this tag) unless the deploy names another. 0.28.0 is the first release
# whose model registry has Glimmer's architecture, MuseGlimmerForConditionalGeneration (vLLM #51655,
# merged 2026-08-14); 0.11.0 can't load it. The Oct 28 check confirms that it loads Glimmer, applies
# an adapter at the serving precision and parses its tool calls.
DEFAULT_VLLM_VERSION: Final = "0.28.0"

# vLLM's parser for Glimmer's channels (vllm/reasoning/muse_glimmer_reasoning_parser.py in 0.28.0).
# Without it vLLM drops the special tokens and a reply's content runs the reasoning into the answer.
REASONING_PARSER: Final = "muse_glimmer"
REASONING_LEVELS: Final = ("low", "medium", "high", "xhigh")
DEFAULT_REASONING_STRENGTH: Final = "low"
# The reply's channel header: the template's generation prompt ends "<|start|>assistant", and an
# assistant message goes on with this header, its words and REPLY_END.
USER_CHANNEL: Final = " to=user<|message|>"
REPLY_END: Final = "<|eot|>"

# Glimmer's tokenizer has 2,048 special tokens, ids 200000 to 202047, all "<|...|>". Fifteen are
# named (by their offset from 200000); every other one is <|reserved_special_token_N|> at offset N.
FIRST_SPECIAL_ID: Final = 200_000
SPECIAL_TOKENS: Final = 2048
NAMED_SPECIAL_TOKENS: Final = {
    0: "<|begin_of_text|>", 1: "<|end_of_text|>", 7: "<|eom|>", 8: "<|eot|>", 18: "<|finetune_right_pad|>",
    22: "<|start|>", 23: "<|message|>", 80: "<|image_start|>", 81: "<|image_end|>", 82: "<|vid_start|>",
    83: "<|vid_end|>", 87: "<|vid_frame_separator|>", 90: "<|image|>", 91: "<|video|>", 92: "<|patch|>",
}
# Every special token, in id order: the text Glimmer is sent never carries one (serving.framing
# breaks them), so a person can't write "<|start|>system<|message|>" into the prompt.
CONTROL_TOKENS: Final = tuple(
    NAMED_SPECIAL_TOKENS.get(offset, f"<|reserved_special_token_{offset}|>") for offset in range(SPECIAL_TOKENS)
)
# A reply ends at the end of its turn (<|eot|>) or of its message (<|eom|>), at the end of the text,
# or where another message would start.
REPLY_STOPS: Final = tuple(FIRST_SPECIAL_ID + offset for offset in (8, 7, 1, 22))

# Every function has a timeout (§12.1). Serving scales to zero after ten idle minutes, and one
# container serves for at most three hours before Modal replaces it.
SCALEDOWN_S: Final = 600
SERVE_TIMEOUT_S: Final = 3 * 3600
STARTUP_TIMEOUT_S: Final = 20 * 60
DOWNLOAD_TIMEOUT_S: Final = 3600
CONCURRENT_REQUESTS: Final = 16
# Every arm (the base and each person's adapter) shares one GPU by design, so a wake never starts a
# second container: on 2026-10-28 one wake started two L40S containers before this was set.
MAX_CONTAINERS: Final = 1

MAX_MODEL_LEN: Final = 32_768
MAX_OUTPUT_TOKENS: Final = 4096
# LoRA sizing (the memory math is in deploy/dev_host.md). Task 5 trains rank 16, and Meta's
# recipe ranges up to 32, so the server takes rank 32 at most. About 6 to 8 adapters are live at
# once (Kevin, Tessa, three lives and the arm-3x donors): all 8 fit on the GPU, and the CPU keeps
# 16 warm so a ninth swaps in without reading the volume. vLLM queues a request past these.
MAX_LORA_RANK: Final = 32
MAX_LORAS: Final = 8
MAX_CPU_LORAS: Final = 16
GPU_MEMORY_UTILIZATION: Final = 0.92

ADAPTER_NAME = re.compile(r"p-[0-9a-f]{8,64}-n[0-9]{1,5}")
ADAPTER_FILES: Final = ("adapter_config.json", "adapter_model.safetensors")
MAX_ADAPTER_BYTES: Final = 2 * 1024**3
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PARSER = re.compile(r"[a-z0-9_]{1,40}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_SECRET = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class ServeConfigError(ValueError):
    """A serving setting or an adapter folder that the server can't take."""


@dataclass(frozen=True)
class Precision:
    """One serving precision: the GPU it fits, vLLM's flags for it, and the name the pin records
    (the pin vocabulary's quantization values)."""

    name: str
    gpu: str
    quantization: str | None
    dtype: str


FP8_L40S: Final = Precision(name="fp8", gpu="L40S", quantization="fp8", dtype="bfloat16")
BF16_A100: Final = Precision(name="bf16", gpu="A100-80GB", quantization=None, dtype="bfloat16")
SERVING: Final = FP8_L40S  # the fallback is BF16_A100: change this line, then redeploy


@dataclass(frozen=True)
class ServeConfig:
    """The deploy-time settings. ``revision`` is the Glimmer commit the weights are pinned to,
    ``tool_parser`` vLLM's tool-call parser for Glimmer (``muse_glimmer``; None serves without tools),
    ``gpu_snapshot`` whether Modal snapshots the GPU memory to cut the cold start, and
    ``hf_secret`` the Modal secret holding HF_TOKEN for the download (None downloads without)."""

    precision: Precision
    revision: str | None
    tool_parser: str | None
    gpu_snapshot: bool
    vllm_version: str
    hf_secret: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServeConfig:
        env = os.environ if environ is None else environ
        revision = env.get(REVISION_ENV, "").strip() or None
        if revision is not None and not _COMMIT.fullmatch(revision):
            raise ServeConfigError(f"{REVISION_ENV} is a 40-character commit hash")
        parser = env.get(TOOL_PARSER_ENV, "").strip() or None
        if parser is not None and not _PARSER.fullmatch(parser):
            raise ServeConfigError(f"{TOOL_PARSER_ENV} names one vLLM tool-call parser")
        version = env.get(VLLM_VERSION_ENV, "").strip() or DEFAULT_VLLM_VERSION
        if not _VERSION.fullmatch(version):
            raise ServeConfigError(f"{VLLM_VERSION_ENV} is a release such as {DEFAULT_VLLM_VERSION}")
        hf_secret = env.get(HF_SECRET_ENV, "").strip() or None
        if hf_secret is not None and not _SECRET.fullmatch(hf_secret):
            raise ServeConfigError(f"{HF_SECRET_ENV} names a Modal secret, never its value")
        return cls(
            precision=SERVING, revision=revision, tool_parser=parser,
            gpu_snapshot=env.get(SNAPSHOT_ENV, "").strip() == "1", vllm_version=version,
            hf_secret=hf_secret,
        )

    def pinned_revision(self) -> str:
        if self.revision is None:
            raise ServeConfigError(f"set {REVISION_ENV} to the Glimmer commit the weights pin to")
        return self.revision


def reasoning_strength(environ: Mapping[str, str] | None = None) -> str:
    """The reasoning strength every Glimmer call and the training text are rendered at."""
    env = os.environ if environ is None else environ
    level = env.get(REASONING_STRENGTH_ENV, "").strip() or DEFAULT_REASONING_STRENGTH
    if level not in REASONING_LEVELS:
        raise ServeConfigError(f"{REASONING_STRENGTH_ENV} is one of {', '.join(REASONING_LEVELS)}")
    return level


def weights_path(revision: str) -> str:
    """Where one pinned revision of Glimmer sits on the weights volume."""
    return f"{WEIGHTS_DIR}/{MODEL}/{revision}"


def vllm_command(
    config: ServeConfig, *, model_path: str | None = None, host: str = "0.0.0.0", port: int = PORT,
) -> list[str]:
    """The server's command line. The API key is read from VLLM_API_KEY (the Modal secret), never
    passed as an argument, so it never shows in a process list. ``model_path`` is where the pinned
    weights sit (Modal's weights volume by default; SageMaker copies them to its own model folder)."""
    revision = config.pinned_revision()  # no server starts unpinned, wherever its weights sit
    path = weights_path(revision) if model_path is None else model_path
    command = [
        "vllm", "serve", path,
        "--served-model-name", MODEL,
        "--host", host, "--port", str(port),
        "--dtype", config.precision.dtype,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--enable-lora", "--max-lora-rank", str(MAX_LORA_RANK), "--max-loras", str(MAX_LORAS),
        "--max-cpu-loras", str(MAX_CPU_LORAS),
        "--reasoning-parser", REASONING_PARSER,
        # Serving is text only. Without this, vLLM 0.28.0 puts LoRA wrappers on Glimmer's vision tower
        # and the profile run's dummy image crashes start-up (lora_shrink assert token_lora_mapping.size(0) == M).
        "--language-model-only",
    ]
    if config.precision.quantization is not None:
        command += ["--quantization", config.precision.quantization]
    if config.tool_parser is not None:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", config.tool_parser]
    return command


# FlashInfer's sampler JIT-builds a kernel with nvcc at warm-up, and the images carry no CUDA toolkit;
# 0 makes vLLM's V2 sampler use its native top-k/top-p + Gumbel path.
NATIVE_SAMPLER_ENV: Final = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def vllm_env() -> dict[str, str]:
    """The server's environment beside the secret: adapters load at runtime, by the client's
    ``/v1/load_lora_adapter``, and Hugging Face's cache stays on the weights volume."""
    return {
        **NATIVE_SAMPLER_ENV,
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "HF_HOME": f"{WEIGHTS_DIR}/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }


def check_adapter_name(name: str) -> str:
    if not isinstance(name, str) or not ADAPTER_NAME.fullmatch(name):
        raise ServeConfigError("an adapter is named p-<hash>-n<night>")
    return name


def check_adapter(folder: Path, name: str) -> None:
    """A LoRA adapter folder the server will load: the PEFT files, a LoRA of Glimmer, a rank the
    server takes, and nothing else."""
    check_adapter_name(name)
    if not folder.is_dir():
        raise ServeConfigError("an adapter is a folder")
    present = sorted(path.name for path in folder.iterdir())
    if present != sorted(ADAPTER_FILES):
        raise ServeConfigError(f"an adapter folder holds exactly {', '.join(ADAPTER_FILES)}")
    if sum((folder / file).stat().st_size for file in ADAPTER_FILES) > MAX_ADAPTER_BYTES:
        raise ServeConfigError("the adapter is larger than any rank the server takes")
    try:
        meta = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ServeConfigError("adapter_config.json can't be read") from error
    if not isinstance(meta, dict) or meta.get("peft_type") != "LORA":
        raise ServeConfigError("the adapter is a PEFT LoRA")
    if meta.get("base_model_name_or_path") != MODEL:
        raise ServeConfigError(f"the adapter's base model is {MODEL}, the model the server serves")
    rank = meta.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= MAX_LORA_RANK:
        raise ServeConfigError(f"the adapter's rank is 1 to {MAX_LORA_RANK}")
