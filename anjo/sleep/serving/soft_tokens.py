"""Soft tokens: K vectors from MIND's core read by Glimmer as if they were tokens of the latest user
turn (a stretch, out of the demo's critical path, off by default).

**What vLLM 0.28 takes.** The server runs ``--language-model-only`` (LoRA crashes on the vision
tower otherwise, ``config.vllm_command``), and on that path a request's ``prompt_embeds`` is its
whole prompt: vLLM reads every prompt position from the tensor and only the generated tokens as
ids. So a turn with soft tokens is sent as the whole rendered prompt's embeddings:

1. the prompt's token ids, rendered exactly as today (``client.reply_prompt``);
2. each id's row of Glimmer's embedding table after its weightless RMSNorm (``EmbeddingRows``):
   what vLLM's own ``embed_input_ids`` would compute, since supplied embeddings skip that norm;
3. the K soft vectors spliced in just before the ``<|eot|>`` that closes the latest user turn,
   where every layer sees them (39 of Glimmer's 52 layers see only a 2,048-token window);
4. the matrix as a bf16 tensor, ``torch.save``d and base64-encoded, in ``/v1/completions``'s
   ``prompt_embeds`` (``encode_embeds``; torch loads only here, on the host that sends one).

A soft vector must be post-norm, as a token's row is: RMS 1 within ``RMS_RANGE``. ``SoftTokens.none()``
(K = 0) sends a prompt as embeddings and nothing else, the parity check against its token ids.

**The rows** come from the training app's ``embed_rows`` (a CPU call on the weights volume at the
pinned revision, which sends each id's raw bf16 row; ``ModalEmbeddingRows``), and each id is fetched
once per process (``CachedRows``). Nothing here is used unless a client is switched on with
``VllmClient.with_soft_tokens`` and a turn passes ``soft=``; the server must also run with
``ANJO_SLEEP_PROMPT_EMBEDS=1``.
"""
from __future__ import annotations

import base64
import importlib
import io
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from anjo.sleep.serving import config

Matrix: TypeAlias = NDArray[np.float32]
HIDDEN: Final = 6656  # Glimmer's text hidden size
VOCAB: Final = 202_048
EOT_ID: Final = config.FIRST_SPECIAL_ID + 8  # <|eot|>
MAX_SOFT_TOKENS: Final = 32
RMS_RANGE: Final = (0.5, 2.0)  # a token's post-norm row has RMS 1
RMS_EPS: Final = 1e-5  # Glimmer's rms_norm_eps, its embedding norm's epsilon
TRAIN_APP: Final = "anjo-sleep-train"  # train.modal_job.APP_NAME (serving never imports train)
EMBED_FUNCTION: Final = "embed_rows"  # train.latent_job.EMBED_FUNCTION
EMBED_TIMEOUT_S: Final = 300.0  # train.latent_job.EMBED_TIMEOUT_S, the call's hard timeout
EMBED_COLD_START_S: Final = 200.0  # Modal starting the CPU container before that timeout begins, about
EMBED_USD_PER_HOUR: Final = 0.10  # the CPU container that sends the embedding rows


class SoftTokenError(ValueError):
    """Soft tokens that aren't post-norm vectors of Glimmer's width, or a prompt they can't go in."""


@dataclass(frozen=True, eq=False)
class SoftTokens:
    """K post-norm vectors of Glimmer's width, 0 to ``MAX_SOFT_TOKENS`` of them."""

    vectors: Matrix

    def __post_init__(self) -> None:
        vectors = self.vectors
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[1] != HIDDEN:
            raise SoftTokenError(f"soft tokens are rows of Glimmer's width, {HIDDEN}")
        if vectors.shape[0] > MAX_SOFT_TOKENS:
            raise SoftTokenError(f"a turn carries at most {MAX_SOFT_TOKENS} soft tokens")
        if not np.isfinite(vectors).all():
            raise SoftTokenError("a soft token is not finite")
        rms = np.sqrt((vectors.astype(np.float64) ** 2).mean(axis=1))
        if ((rms < RMS_RANGE[0]) | (rms > RMS_RANGE[1])).any():
            raise SoftTokenError(f"a soft token is post-norm, as a token's row is: RMS within {RMS_RANGE}")
        object.__setattr__(self, "vectors", vectors.astype(np.float32))

    @classmethod
    def none(cls) -> SoftTokens:
        return cls(np.zeros((0, HIDDEN), dtype=np.float32))

    @property
    def k(self) -> int:
        return int(self.vectors.shape[0])


class EmbeddingRows(Protocol):
    def rows(self, ids: Sequence[int]) -> Matrix:
        """Each id's post-norm embedding row, float32, ``(len(ids), HIDDEN)``."""
        ...


def normalize(raw: NDArray[Any]) -> Matrix:
    """Glimmer's weightless RMSNorm, as vLLM's ``MuseGlimmerRMSNorm`` computes it, in float32."""
    x = np.asarray(raw, dtype=np.float32)
    return (x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(RMS_EPS))).astype(np.float32)


def bf16_rows(data: bytes, count: int) -> Matrix:
    """``count`` rows of raw little-endian bf16 bits as float32 (exact)."""
    bits: NDArray[np.uint16] = np.frombuffer(data, dtype="<u2")
    if bits.size != count * HIDDEN:
        raise SoftTokenError(f"expected {count} bf16 rows of width {HIDDEN}")
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(count, HIDDEN)


class CachedRows:
    """``EmbeddingRows`` over ``fetch`` (ids to their raw bf16 rows), each id fetched once."""

    def __init__(self, fetch: Callable[[list[int]], bytes]) -> None:
        self._fetch = fetch
        self._rows: dict[int, Matrix] = {}
        self._lock = threading.Lock()

    def rows(self, ids: Sequence[int]) -> Matrix:
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < VOCAB for i in ids):
            raise SoftTokenError(f"a token id is 0 to {VOCAB - 1}")
        with self._lock:
            missing = list(dict.fromkeys(i for i in ids if i not in self._rows))
            if missing:
                fresh = normalize(bf16_rows(self._fetch(missing), len(missing)))
                self._rows.update(zip(missing, fresh, strict=True))
            return np.stack([self._rows[i] for i in ids]).astype(np.float32)


class ModalEmbeddingRows(CachedRows):
    """The rows from the training app's ``embed_rows`` (a CPU call on the weights volume). It takes no
    hold of its own: use it only under a hold that covers it, as ``tools.soft_tokens_live`` does."""

    def __init__(self, app_name: str = TRAIN_APP, function: str = EMBED_FUNCTION) -> None:
        def fetch(ids: list[int]) -> bytes:
            modal: Any = importlib.import_module("modal")
            return bytes(modal.Function.from_name(app_name, function).remote(ids))

        super().__init__(fetch)


def splice(prompt: Sequence[int], rows: EmbeddingRows, soft: SoftTokens) -> Matrix:
    """The prompt's rows with the soft vectors just before the latest user turn's ``<|eot|>``: the
    last one in the prompt, since the generation prompt and the reply's header carry none."""
    ids = list(prompt)
    if EOT_ID not in ids:
        raise SoftTokenError("the prompt has no closed user turn to put soft tokens in")
    cut = len(ids) - 1 - ids[::-1].index(EOT_ID)
    matrix = rows.rows(ids)
    if matrix.shape != (len(ids), HIDDEN):
        raise SoftTokenError("the embedding rows don't match the prompt")
    return np.concatenate([matrix[:cut], soft.vectors, matrix[cut:]]).astype(np.float32)


def encode_embeds(matrix: Matrix) -> str:  # pragma: no cover - needs torch, on the host that sends soft tokens
    """vLLM's ``prompt_embeds`` wire form: a bf16 tensor, ``torch.save``d, base64-encoded."""
    torch: Any = importlib.import_module("torch")
    buffer = io.BytesIO()
    torch.save(torch.from_numpy(np.ascontiguousarray(matrix)).to(torch.bfloat16), buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
