"""The fold and the adapter file, for ``sft_dpo_seed``'s job (``sleep_job.py``): with ``long_term_rank`` set
the job trains a LoRA whose first ranks hold the person's long-term adapter (frozen) and whose last ranks
are tonight's fresh slot; after training, each module's product B A is cut to the best rank-R
approximation (QR, then an SVD of the small core) and written back as a plain LoRA of rank R, alpha 2R,
which the server takes as any other (``fold_folder``). Scale is 2 before and after (alpha is always 2r).

``read_safetensors`` and ``write_safetensors`` read and write an adapter's weights file with torch alone
(bfloat16 has no numpy type). Standalone, as the training script is: the standard library at the top and
torch inside the functions; Modal's image carries it beside ``colab_train.py``, which loads it by its path.
"""
from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

_DTYPES: Final = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}


def fold(tensors: Mapping[str, Any], rank: int) -> dict[str, Any]:
    """Each LoRA pair's product cut to its best rank ``min(rank, k)`` approximation, as a plain pair."""
    import torch

    a_names = sorted(name for name in tensors if name.endswith("lora_A.weight"))
    if not a_names or len(tensors) != 2 * len(a_names):
        raise ValueError("a plain LoRA holds lora_A and lora_B pairs and nothing else")
    out: dict[str, Any] = {}
    for a_name in a_names:
        b_name = a_name[: -len("lora_A.weight")] + "lora_B.weight"
        a, b = tensors[a_name].double(), tensors[b_name].double()  # (k, d_in) and (d_out, k)
        keep = min(rank, a.shape[0])
        qb, rb = torch.linalg.qr(b)
        qa, ra = torch.linalg.qr(a.T)
        u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
        m = min(keep, s.numel())
        root = s[:m].sqrt()
        new_b = torch.zeros(b.shape[0], keep, dtype=b.dtype)
        new_a = torch.zeros(keep, a.shape[1], dtype=a.dtype)
        new_b[:, :m] = (qb @ u[:, :m]) * root
        new_a[:m] = root[:, None] * (vh[:m] @ qa.T)
        dtype = tensors[a_name].dtype
        out[a_name], out[b_name] = new_a.to(dtype).contiguous(), new_b.to(dtype).contiguous()
    return out


def read_safetensors(path: Path) -> dict[str, Any]:
    import torch

    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + size])
    data = raw[8 + size:]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] not in _DTYPES:
            raise ValueError(f"{name} is {meta['dtype']}, not a float type an adapter is saved in")
        begin, end = meta["data_offsets"]
        flat = torch.frombuffer(bytearray(data[begin:end]), dtype=getattr(torch, _DTYPES[meta["dtype"]]))
        out[name] = flat.reshape(meta["shape"]).clone()
    return out


def write_safetensors(tensors: Mapping[str, Any], path: Path) -> None:
    import torch

    codes = {name: code for code, name in _DTYPES.items()}
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    blobs, offset = [], 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu()
        blob = tensor.view(-1).view(torch.uint8).numpy().tobytes()  # bfloat16 has no numpy type
        header[name] = {"dtype": codes[str(tensor.dtype).removeprefix("torch.")], "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def fold_folder(folder: Path, rank: int) -> int:
    """Fold the adapter saved in ``folder`` to rank ``min(rank, r)`` in place, as a plain LoRA with
    alpha 2r: the two files, nothing else. Returns the rank it holds."""
    config_path, weights_path = folder / "adapter_config.json", folder / "adapter_model.safetensors"
    meta = json.loads(config_path.read_text(encoding="utf-8"))
    r = meta.get("r")
    if (meta.get("peft_type") != "LORA" or meta.get("use_dora") or meta.get("rank_pattern")
            or meta.get("alpha_pattern") or not isinstance(r, int) or meta.get("lora_alpha") != 2 * r):
        raise ValueError("the fold takes a plain LoRA with alpha 2r")
    tensors = read_safetensors(weights_path)
    if any(t.shape[0 if name.endswith("lora_A.weight") else 1] != r for name, t in tensors.items()):
        raise ValueError("the adapter's tensors don't hold the rank its config names")
    folded, kept = fold(tensors, rank), min(rank, r)
    pending = folder / f".{weights_path.name}.tmp"
    write_safetensors(folded, pending)
    os.replace(pending, weights_path)
    meta.update({"r": kept, "lora_alpha": 2 * kept, "rank_pattern": {}, "alpha_pattern": {}})
    config_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return kept
