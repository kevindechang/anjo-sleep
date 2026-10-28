"""The serving GPUs a night could wake (``config.SERVE_GPUS``), priced and sized for the cost model
and the gate bench. Numbers only: nothing here calls anything.

- **Price:** Modal's on-demand list price an hour (the L40S at $1.95 is the rate the 2026-10-28
  nights were billed at; the H100's and the A100-80GB's are Modal's list rates, to check before quoting).
- **KV room:** what vLLM keeps for the KV cache at FP8 Glimmer, ``config.GPU_MEMORY_UTILIZATION`` and
  eight rank-32 LoRA slots. Measured on the L40S (5.78 GiB, 2026-10-28: the weights take 31.8 GiB of
  its 45 GiB); on the 80 GB GPUs it is that plus their extra usable memory, an estimate until a deploy
  prints vLLM's own "GPU KV cache size" line.
- **Decode speed:** one stream's tokens a second. Decoding a dense 30B model is bound by memory
  bandwidth, so the L40S's measured 25.6 tokens a second (a full 1,024-token reply in 40 s, 16 at
  once; ``serving.client``) scales by bandwidth: an estimate until ``tools.gate_bench`` measures it.
  Streams add up while the GPU stays bandwidth-bound, up to what the KV room holds.

**Glimmer's KV** (``deploy/dev_host.md`` §1.5): 52 layers, 2 KV heads of 208 (a KV width of 416), bf16:
1,664 bytes a layer and token. Thirteen layers keep every token; the other 39 keep a 2,048-token
window. So a gate request (a short prompt and a reply of at most 1,024 tokens, about 1,200) takes
about 104 MB, and a practice candidate (F of up to 8k tokens, the message and its reply, about 9,000)
about 328 MB before prefix sharing.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from anjo.sleep.serving import config

LAYER_TOKEN_BYTES: Final = 2 * 416 * 2  # K and V, 416 wide, bf16
FULL_LAYERS: Final = 13
WINDOW_LAYERS: Final = 39
WINDOW_TOKENS: Final = 2048
GATE_REQUEST_TOKENS: Final = 1200  # a gate's prompt and its reply at the client's cap
PRACTICE_REQUEST_TOKENS: Final = 9000  # F at practice's cap, the message and a full reply
L40S_KV_GIB: Final = 5.78  # measured, 2026-10-28
L40S_STREAM_TOKENS_S: Final = 1024 / 40  # serving.client: a full reply in 40 s, 16 at once
GIB: Final = 1024**3


@dataclass(frozen=True)
class Gpu:
    name: str
    usd_per_hour: float
    memory_gib: float
    bandwidth_gb_s: float
    native_fp8: bool
    kv_measured: bool = False

    @property
    def kv_gib(self) -> float:
        """vLLM's KV room: the L40S's measured room, plus this GPU's extra usable memory."""
        extra = (self.memory_gib - L40S.memory_gib) * config.GPU_MEMORY_UTILIZATION
        return L40S_KV_GIB + extra

    @property
    def stream_tokens_s(self) -> float:
        """One stream's decode speed, scaled from the L40S's by bandwidth (an estimate)."""
        return L40S_STREAM_TOKENS_S * self.bandwidth_gb_s / L40S.bandwidth_gb_s

    def fits(self, request_tokens: int, kv_gib: float | None = None) -> int:
        """How many requests of ``request_tokens`` its KV room holds at once."""
        room = (self.kv_gib if kv_gib is None else kv_gib) * GIB
        return int(room // request_bytes(request_tokens))


def request_bytes(tokens: int) -> int:
    """One request's KV at its longest."""
    return LAYER_TOKEN_BYTES * (FULL_LAYERS * tokens + WINDOW_LAYERS * min(tokens, WINDOW_TOKENS))


L40S: Final = Gpu("L40S", usd_per_hour=1.95, memory_gib=44.99, bandwidth_gb_s=864.0, native_fp8=True, kv_measured=True)
H100: Final = Gpu("H100", usd_per_hour=3.95, memory_gib=79.65, bandwidth_gb_s=3350.0, native_fp8=True)
A100_80GB: Final = Gpu("A100-80GB", usd_per_hour=2.50, memory_gib=80.0, bandwidth_gb_s=2039.0, native_fp8=False)
GPUS: Final[Mapping[str, Gpu]] = MappingProxyType({gpu.name: gpu for gpu in (L40S, H100, A100_80GB)})
