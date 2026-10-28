"""The thought channel's projector as math: the same code on numpy (the host that sends soft tokens,
``mind.core.projector``) and on torch (the night's training job, ``train.channel_train``).

**The projector** (``forward``): one token from the core's global vector and speak probability, then
one per routed slot, in the core's own choice order. Each slot's features go through a two-layer MLP
to Glimmer's width and are added, scaled by the core's choice gate, to that rank's anchor row; the
global token mixes a "speak" and a "silence" anchor by the speak probability. Every token is
RMS-normalized, as a token's own row is after Glimmer's weightless embedding norm. ``forward`` uses
only ``@``, arithmetic and the ``tanh`` and ``sqrt`` it is given, so numpy and torch run the same
code. A fresh projector's anchors are the rows of ``ANCHOR_WORDS``, so Glimmer reads them as words
before any training.

It imports the standard library only (numpy loads inside the functions that need it), and lives on
the serving side because serving runs it: the training job's isolated script loads it by path from
the image (``train.channel_train``), and the night's practice reads it without loading training.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from typing import Any, Final

HIDDEN: Final = 6656  # Glimmer's text hidden size (serving.soft_tokens.HIDDEN; a test holds them equal)
RMS_EPS: Final = 1e-5  # Glimmer's embedding norm's epsilon (serving.soft_tokens.RMS_EPS)
PROJECTOR_FILE: Final = "projector.npz"
PROJECTOR_VERSION: Final = 1
SCALARS: Final = ("usage", "risk", "missing", "self_error", "choice", "age", "cited")
MAX_K: Final = 12
GATE_FLOOR: Final = 0.25  # a routed slot's gate never falls below this, so a low-ranked thread still speaks
INIT_OUT: Final = 0.4  # a fresh output layer's gain: soft tokens start near their anchors, already state-dependent
ANCHOR_WORDS: Final = (" concern", " nothing", " first", " second", " third", " fourth", " fifth", " sixth",
                       " seventh", " eighth", " ninth", " tenth", " eleventh")  # speak, silence, then each rank
PARAMS: Final = ("w1", "b1", "u1", "c1", "w2", "b2", "anchors", "speak", "silence",
                 "x_mean", "x_std", "g_mean", "g_std")
_GELU: Final = math.sqrt(2.0 / math.pi)


def features(d: int) -> int:
    """A routed slot's feature count: its vector, the fast memory's answer to it, the global vector,
    and ``SCALARS``."""
    return 3 * d + len(SCALARS)


def _gelu(x: Any, tanh: Callable[[Any], Any]) -> Any:
    return 0.5 * x * (1.0 + tanh(_GELU * (x + 0.044715 * x * x * x)))


def rms(x: Any, ones: Any, sqrt: Callable[[Any], Any]) -> Any:
    """Glimmer's weightless RMSNorm over the last axis; ``ones`` is a ``(H, 1)`` column of ones."""
    return x / sqrt(((x * x) @ ones) / ones.shape[0] + RMS_EPS)


def forward(
    p: Mapping[str, Any], slot_x: Any, gates: Any, global_x: Any, speak: Any, *,
    tanh: Callable[[Any], Any], sqrt: Callable[[Any], Any], ones: Any,
) -> tuple[Any, Any]:
    """The speak token ``(B, H)`` and the slot tokens ``(B, K-1, H)`` for readings stacked in a batch:
    ``slot_x (B, K-1, F)``, ``gates (B, K-1)``, ``global_x (B, 2d)``, ``speak (B,)``."""
    hidden = _gelu(((slot_x - p["x_mean"]) / p["x_std"]) @ p["w1"] + p["b1"], tanh)
    slots = rms(p["anchors"] + gates[..., None] * (hidden @ p["w2"] + p["b2"]), ones, sqrt)
    whole = _gelu(((global_x - p["g_mean"]) / p["g_std"]) @ p["u1"] + p["c1"], tanh) @ p["w2"] + p["b2"]
    s = speak[..., None]
    return rms(s * p["speak"] + (1.0 - s) * p["silence"] + whole, ones, sqrt), slots


def fresh_params(k: int, d: int, width: int, anchor_rows: Any, seed: int, *, hidden: int = HIDDEN) -> dict[str, Any]:
    """A new projector (numpy float32): ``anchor_rows`` are ``k + 1`` post-norm rows of the model's
    width ``hidden`` (speak, silence, then each rank's), the first layers scaled for unit features,
    the output layer small."""
    import numpy as np

    rows = np.asarray(anchor_rows, dtype=np.float32)
    if rows.shape != (k + 1, hidden):
        raise ValueError(f"a projector of K = {k} takes {k + 1} anchor rows of width {hidden}")
    rng = np.random.default_rng(seed)
    f, g = features(d), 2 * d

    def normal(shape: tuple[int, ...], std: float) -> Any:
        return (rng.standard_normal(shape) * std).astype(np.float32)

    return {
        "w1": normal((f, width), 1 / math.sqrt(f)), "b1": np.zeros(width, np.float32),
        "u1": normal((g, width), 1 / math.sqrt(g)), "c1": np.zeros(width, np.float32),
        "w2": normal((width, hidden), INIT_OUT / math.sqrt(width)), "b2": np.zeros(hidden, np.float32),
        "anchors": rows[2:].copy(), "speak": rows[0].copy(), "silence": rows[1].copy(),
        "x_mean": np.zeros(f, np.float32), "x_std": np.ones(f, np.float32),
        "g_mean": np.zeros(g, np.float32), "g_std": np.ones(g, np.float32),
    }


def save_params(params: Mapping[str, Any], meta: Mapping[str, Any]) -> bytes:
    """The projector as ``.npz`` bytes (float32 arrays and a JSON ``meta``, which records the width
    it was made for, ``hidden``; no pickle)."""
    import io

    import numpy as np

    if sorted(params) != sorted(PARAMS):
        raise ValueError(f"a projector holds exactly {', '.join(PARAMS)}")
    out = io.BytesIO()
    arrays = {name: np.asarray(params[name], dtype=np.float32) for name in PARAMS}
    full = {**meta, "hidden": int(arrays["speak"].shape[0]), "version": PROJECTOR_VERSION}
    np.savez(out, meta=np.array(json.dumps(full, sort_keys=True)), **arrays)
    return out.getvalue()


def load_params(blob: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """``save_params``' bytes back: the arrays and the meta; ``ValueError`` for anything else."""
    import io
    import zipfile

    import numpy as np

    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as held:
            meta = json.loads(str(held["meta"]))
            params = {name: np.array(held[name], dtype=np.float32) for name in PARAMS}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"not a saved projector: {type(error).__name__}") from None
    if meta.get("version") != PROJECTOR_VERSION:
        raise ValueError(f"projector version {meta.get('version')!r}; this build reads {PROJECTOR_VERSION}")
    return params, meta


def params_sha(params: Mapping[str, Any]) -> str:
    """A projector's id: a hash of its arrays."""
    import numpy as np

    digest = hashlib.sha256()
    for name in PARAMS:
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(params[name], dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def decode_reading(obj: Mapping[str, Any]) -> tuple[Any, Any, Any, float]:
    """A row's ``core.reading`` (``mind.core.projector.Reading.to_json``) as numpy arrays:
    ``slot_x (K-1, F)``, ``gates (K-1,)``, ``global_x (2d,)`` and the speak probability."""
    import numpy as np

    k, d = int(obj["k"]), int(obj["d"])

    def arr(key: str, shape: tuple[int, ...]) -> Any:
        found: Any = np.frombuffer(base64.b64decode(obj[key]), dtype="<f4")
        if found.size != math.prod(shape) or not np.isfinite(found).all():
            raise ValueError(f"a reading's {key} isn't {shape} finite floats")
        return found.reshape(shape).astype(np.float32)

    return (arr("slot_x", (k - 1, features(d))), arr("gates", (k - 1,)), arr("global_x", (2 * d,)),
            float(obj["speak"]))
