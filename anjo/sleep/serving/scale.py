"""A scaled copy of a person's LoRA adapter, for the weight-strength read (adapter-scale).

Arm 4 scored below arm 2C on knowledge in the development read while arm 3 beat arm 1, so the
cheapest test of "the adapter overrides what C says" is arm 4 with the adapter's delta multiplied
by ``s``. This module writes that copy; it never touches the original.

**What is scaled.** A PEFT LoRA adds ``(lora_B @ lora_A) * (lora_alpha / r)`` to its base weights.
Multiplying every ``lora_B`` tensor by ``s`` multiplies that delta by ``s`` exactly, before the
stored dtype's rounding, and leaves ``adapter_config.json`` a valid LoRA the server already knows
how to load. Scaling ``lora_alpha`` would need the server to accept a fractional alpha; scaling B
does not. ``s=1`` is a byte-for-byte copy; ``s=0`` is a delta of zero, the base model's answer.

The safetensors container is read and written here rather than through ``safetensors``, which the
sleep host's venv does not carry: the header is JSON after a little-endian length, then the tensors'
bytes in ``data_offsets`` order. Only the B tensors' bytes change, so every offset, shape and the
config stay as they were.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from anjo.sleep.serving import config

_HEADER_BYTES: Final = 8
_B_MARK: Final = ".lora_B."
# safetensors' float dtypes that a trained LoRA can land in. A B tensor in another dtype is refused
# rather than silently left at full strength.
_FLOATS: Final = frozenset({"F64", "F32", "F16", "BF16"})


class ScaleError(ValueError):
    """An adapter or scale that this writer can't copy exactly."""


@dataclass(frozen=True)
class Tensors:
    """One safetensors file split into its header bytes, parsed header and data blob."""

    header: bytes
    meta: dict[str, dict[str, object]]
    data: bytes


def check_scale(scale: float) -> float:
    """``s`` in [0, 1]: shrinking only, and a whole-number check so a bool is not a scale."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ScaleError("a scale is a number from 0 to 1")
    found = float(scale)
    if not 0.0 <= found <= 1.0:
        raise ScaleError("a scale is from 0 (no adapter) to 1 (the adapter as trained)")
    return found


def content_digest(source: Path) -> str:
    """The adapter's content hash: both files, in ``ADAPTER_FILES`` order.

    The copy's name carries this digest, so a name already on the server names the same source bytes
    at the same ``s``: the copy may be reused instead of uploaded again.
    """
    digest = hashlib.sha256()
    try:
        for name in config.ADAPTER_FILES:
            with (source / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
    except OSError as error:
        raise ScaleError(f"the adapter {source.name} can't be read ({type(error).__name__})") from None
    return digest.hexdigest()


def scaled_name(source: str, scale: float, *, digest: str | None = None) -> str:
    """A new ``p-<hash>-n<night>`` for ``source`` at ``scale``: deterministic, so a rerun reuses it,
    and distinct from the source and from every other scale. ``digest`` is the source's
    ``content_digest``; with it the name is content-addressed, so an existing name is the same copy."""
    try:
        config.check_adapter_name(source)
    except config.ServeConfigError as error:
        raise ScaleError(str(error)) from None
    night = source.rsplit("-n", 1)[1]
    material = f"adapter-scale\x1f{source}\x1f{check_scale(scale)!r}"
    if digest is not None:
        material += f"\x1f{digest}"
    found = hashlib.sha256(material.encode()).hexdigest()[:12]
    return config.check_adapter_name(f"p-{found}-n{night}")


def split(raw: bytes) -> Tensors:
    """The safetensors container's three parts. The header is kept byte-for-byte: only data moves."""
    if len(raw) < _HEADER_BYTES:
        raise ScaleError("adapter_model.safetensors is shorter than its header")
    (size,) = struct.unpack("<Q", raw[:_HEADER_BYTES])
    if size <= 0 or _HEADER_BYTES + size > len(raw):
        raise ScaleError("adapter_model.safetensors has an unreadable header length")
    header = raw[_HEADER_BYTES:_HEADER_BYTES + size]
    try:
        meta = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ScaleError("adapter_model.safetensors has an unreadable header") from None
    if not isinstance(meta, dict) or not all(isinstance(item, dict) for item in meta.values()):
        raise ScaleError("adapter_model.safetensors has an unreadable header")
    return Tensors(header, meta, raw[_HEADER_BYTES + size:])


def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 to bfloat16, as a uint16 array."""
    bits = values.astype("<f4").view("<u4")
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return rounded.astype("<u2")


def _bf16_to_f32(words: np.ndarray) -> np.ndarray:
    return (words.astype("<u4") << np.uint32(16)).view("<f4")


def _scaled_bytes(dtype: str, raw: bytes, scale: float) -> bytes:
    """``raw`` (one tensor) multiplied by ``scale``, in the same dtype."""
    if dtype not in _FLOATS:
        raise ScaleError(f"a LoRA B tensor in {dtype} can't be scaled exactly")
    if dtype == "BF16":
        return _f32_to_bf16(_bf16_to_f32(np.frombuffer(raw, dtype="<u2")) * np.float32(scale)).tobytes()
    kind = {"F64": "<f8", "F32": "<f4", "F16": "<f2"}[dtype]
    return (np.frombuffer(raw, dtype=kind).astype(np.float64) * scale).astype(kind).tobytes()


def scaled_model(raw: bytes, scale: float) -> bytes:
    """The safetensors bytes with every ``lora_B`` tensor multiplied by ``scale``, all else intact.

    A delta stored anywhere else (an embedding's ``lora_embedding_B``, DoRA's
    ``lora_magnitude_vector``) is refused: scaling its B would not be a scale of that delta, and a
    sweep that silently returned the same answer at every ``s`` would prove nothing.
    """
    found = split(raw)
    if not any(_B_MARK in name for name in found.meta):
        raise ScaleError("adapter_model.safetensors holds no lora_B tensor to scale")
    for name in found.meta:
        if "lora_" in name and _B_MARK not in name and ".lora_A." not in name:
            raise ScaleError(f"tensor {name} is not a LoRA A or B this writer can scale exactly")
    data = bytearray(found.data)
    for name, meta in found.meta.items():
        if _B_MARK not in name:
            continue
        try:
            dtype = str(meta["dtype"])
            offsets = meta["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise TypeError
            start, end = int(offsets[0]), int(offsets[1])
        except (KeyError, TypeError, ValueError):
            raise ScaleError(f"tensor {name} has an unreadable description") from None
        if not 0 <= start <= end <= len(data):
            raise ScaleError(f"tensor {name} points outside the data blob")
        data[start:end] = _scaled_bytes(dtype, bytes(data[start:end]), scale)
    return struct.pack("<Q", len(found.header)) + found.header + bytes(data)


def write_scaled(source: Path, cache: Path, scale: float) -> Path:
    """Write ``source`` at ``scale`` under a new name in ``cache`` and return its folder.

    An existing folder of that name is reused, so a rerun pays nothing; the name is derived from the
    source's content and the scale, so an existing one is the copy asked for. Neither the source nor
    any other adapter is ever written, and a partial write lands under a temporary name first.
    """
    value = check_scale(scale)
    try:
        config.check_adapter(source, source.name)
    except config.ServeConfigError as error:
        raise ScaleError(str(error)) from None
    target_name = scaled_name(source.name, value, digest=content_digest(source))
    target = cache / target_name
    if target.resolve() == source.resolve() or target_name == source.name:
        raise ScaleError("a scaled copy never overwrites its original")
    if target.exists():
        config.check_adapter(target, target_name)
        return target
    model = (source / config.ADAPTER_FILES[1]).read_bytes()
    if value != 1.0:
        model = scaled_model(model, value)
    cache.mkdir(parents=True, exist_ok=True)
    pending = cache / f".{target_name}.{os.getpid()}.tmp"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    try:
        shutil.copyfile(source / config.ADAPTER_FILES[0], pending / config.ADAPTER_FILES[0])
        (pending / config.ADAPTER_FILES[1]).write_bytes(model)
        os.replace(pending, target)  # a complete folder appears at once, or none does
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return target


__all__ = ["ScaleError", "Tensors", "content_digest", "scaled_model", "scaled_name", "split", "write_scaled"]
