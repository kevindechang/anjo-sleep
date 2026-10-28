"""The thought channel's projector: MIND's state of one person as K soft tokens Glimmer reads in
place of a pasted memory (torch-free; the host serves it, the night's training job trains it).

**The choice: the core routes the channel.** Soft-prompt and memory-token methods (prompt tuning,
gist tokens, in-context autoencoders, xRAG) compress text, or a retrieved document, into vectors
and leave it to the language model to decide what matters. Here a separate recurrent network
holds the memory, and its own trained decision heads decide what Glimmer hears and how loudly:

- token 0 is the **speak token**: a mix of a "speak" and a "silence" anchor by the core's speak
  probability, plus what its global vector and fast memory say;
- tokens 1 to K-1 are the open slots that cite real events, **in the core's own choice order**
  (rank r gets rank r's anchor, so position means rank, not an arbitrary slot index); each is its
  rank's anchor plus the projection of the slot's features, **scaled by the core's choice share**
  (relative to the top slot, never below ``GATE_FLOOR``). An empty rank is its anchor alone.

So a transplanted state changes the tokens through the core's beliefs before the projector learns
anything; when the core sees nothing worth raising, the channel says so explicitly (the silence
anchor, quiet slots), which gives the reader a signal to stay quiet; and a fresh projector starts
from real word rows (``channel_train.ANCHOR_WORDS``), so Glimmer can read it on night 0.

**A slot's features** (``channel_train.SCALARS``): the slot vector, the fast memory's answer to it
(``memory @ slot``), the global vector, then its usage, risk, missing-step and self-error heads,
its choice share, its age in horizons and its cited count. The global token reads the global
vector and the memory's answer to it.

The math is ``channel_train.forward``, the same code the GPU trains with (numpy here, torch there).
``Reading`` is the projector's input; a training row carries it inside a ``CoreSnapshot``
(``night.core_snapshot``), so the row records the state it was made under.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from anjo.sleep.serving.soft_tokens import HIDDEN, SoftTokens
from anjo.sleep.train import channel_train

K_DEFAULT: Final = 8
GATE_FLOOR: Final = channel_train.GATE_FLOOR
HORIZON_DAYS: Final = 7.0  # the core's hazard horizon (types.HORIZON_DAYS): a slot's age is counted in these
MAX_AGE: Final = 4.0
CITE_KEEP: Final = 8.0  # CoreConfig.cite_keep's default: the cited count is a share of it
Array: TypeAlias = NDArray[np.float32]


class ProjectorError(ValueError):
    """A reading or projector of the wrong shape, or a saved projector this build can't read."""


class StateLike(Protocol):
    """What the projector reads of a ``mind.core.state.CoreState``."""

    @property
    def slots(self) -> Array: ...
    @property
    def opened(self) -> Array: ...
    @property
    def memory(self) -> Array: ...
    @property
    def glob(self) -> Array: ...
    @property
    def now(self) -> float | None: ...


class SlotLike(Protocol):
    @property
    def slot(self) -> int: ...
    @property
    def usage(self) -> float: ...
    @property
    def risk(self) -> float: ...
    @property
    def missing(self) -> float: ...
    @property
    def self_error(self) -> float: ...
    @property
    def choice(self) -> float: ...
    @property
    def cites(self) -> tuple[str, ...]: ...


class PredictionsLike(Protocol):
    """What the projector reads of ``mind.core.types.Predictions`` (``Core.predict``)."""

    @property
    def speak(self) -> float: ...
    @property
    def slots(self) -> Sequence[SlotLike]: ...


def _b64(array: NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


@dataclass(frozen=True, eq=False)
class Reading:
    """The projector's input for one moment: each routed slot's features and gate (rank order),
    the global features, the speak probability, and which slots were routed and what they cite."""

    slot_x: Array  # (K-1, F)
    gates: Array  # (K-1,)
    global_x: Array  # (2d,)
    speak: float
    routed: tuple[int, ...] = ()
    cites: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("slot_x", "gates", "global_x"):
            array = np.array(getattr(self, name), dtype=np.float32)
            if not np.isfinite(array).all():
                raise ProjectorError(f"a reading's {name} isn't finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        r, f = self.slot_x.shape
        d = self.global_x.shape[0] // 2
        if self.gates.shape != (r,) or f != channel_train.features(d) or len(self.routed) > r:
            raise ProjectorError("a reading's slots, gates and global features disagree")
        if len(self.cites) != len(self.routed) or not 0.0 <= self.speak <= 1.0:
            raise ProjectorError("a reading names a cite list per routed slot and a speak probability in [0, 1]")

    @property
    def k(self) -> int:
        return int(self.slot_x.shape[0]) + 1

    @property
    def d(self) -> int:
        return int(self.global_x.shape[0]) // 2

    @classmethod
    def empty(cls, k: int = K_DEFAULT, d: int = 64) -> Reading:
        """The channel with nothing in it: the zero-tokens arm (same K positions, no state)."""
        return cls(np.zeros((k - 1, channel_train.features(d)), np.float32), np.zeros(k - 1, np.float32),
                   np.zeros(2 * d, np.float32), 0.0)

    def to_json(self) -> dict[str, Any]:
        """The reading on a training row: arrays as base64 little-endian float32 (``channel_train.decode_reading``
        reads it back in the training job). Cited event ids stay off the row."""
        return {"k": self.k, "d": self.d, "slot_x": _b64(self.slot_x), "gates": _b64(self.gates),
                "global_x": _b64(self.global_x), "speak": float(self.speak), "routed": list(self.routed)}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> Reading:
        try:
            slot_x, gates, global_x, speak = channel_train.decode_reading(obj)
            routed = tuple(int(i) for i in obj.get("routed", ()))
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectorError(f"a row's reading can't be read: {error}") from None
        return cls(slot_x, gates, global_x, speak, routed, ((),) * len(routed))


def read(state: StateLike, predictions: PredictionsLike, *, k: int = K_DEFAULT) -> Reading:
    """The core's state as the projector's input: the open slots that cite real events, ranked by
    the core's choice share (ties to the lower slot), the top ``k - 1`` of them routed."""
    if not 2 <= k <= channel_train.MAX_K:
        raise ProjectorError(f"K is 2 to {channel_train.MAX_K}")
    slots, memory, glob = (np.asarray(a, dtype=np.float32) for a in (state.slots, state.memory, state.glob))
    d = int(glob.shape[0])
    ranked = sorted((s for s in predictions.slots if s.cites), key=lambda s: (-s.choice, s.slot))[: k - 1]
    top = max((s.choice for s in ranked), default=0.0)
    slot_x: Array = np.zeros((k - 1, channel_train.features(d)), np.float32)
    gates: Array = np.zeros(k - 1, np.float32)
    now = 0.0 if state.now is None else float(state.now)
    for rank, s in enumerate(ranked):
        vec = slots[s.slot]
        age = min(max((now - float(state.opened[s.slot])) / HORIZON_DAYS, 0.0), MAX_AGE)
        scalars = [s.usage, s.risk, s.missing, s.self_error, s.choice, age, min(len(s.cites) / CITE_KEEP, 1.0)]
        slot_x[rank] = np.concatenate([vec, memory @ vec, glob, np.asarray(scalars, np.float32)])
        share = s.choice / top if top > 0 else 0.0
        gates[rank] = GATE_FLOOR + (1.0 - GATE_FLOOR) * share
    return Reading(slot_x, gates, np.concatenate([glob, memory @ glob]), float(predictions.speak),
                   tuple(s.slot for s in ranked), tuple(tuple(s.cites) for s in ranked))


@dataclass(frozen=True, eq=False)
class Projector:
    """A trained (or fresh) projector: its K, the core's width ``d``, its own width, and its arrays."""

    k: int
    d: int
    width: int
    params: Mapping[str, Array]

    def __post_init__(self) -> None:
        if sorted(self.params) != sorted(channel_train.PARAMS):
            raise ProjectorError(f"a projector holds exactly {', '.join(channel_train.PARAMS)}")
        shapes = {"w1": (channel_train.features(self.d), self.width), "u1": (2 * self.d, self.width),
                  "w2": (self.width, HIDDEN), "anchors": (self.k - 1, HIDDEN), "speak": (HIDDEN,)}
        for name, shape in shapes.items():
            if tuple(self.params[name].shape) != shape:
                raise ProjectorError(f"the projector's {name} isn't {shape}")
        frozen = {}
        for name, value in self.params.items():
            array = np.array(value, dtype=np.float32)
            array.setflags(write=False)
            frozen[name] = array
        object.__setattr__(self, "params", frozen)

    @classmethod
    def fresh(cls, *, anchor_rows: NDArray[Any], k: int = K_DEFAULT, d: int = 64, width: int = 256,
              seed: int = 0) -> Projector:
        """A new projector whose anchors are ``anchor_rows`` (speak, silence, then each rank's)."""
        return cls(k, d, width, channel_train.fresh_params(k, d, width, anchor_rows, seed))

    @property
    def id(self) -> str:
        return channel_train.params_sha(self.params)

    def project(self, reading: Reading) -> Array:
        """The K soft vectors for ``reading``: post-norm rows of Glimmer's width."""
        if (reading.k, reading.d) != (self.k, self.d):
            raise ProjectorError(f"a reading for K = {reading.k}, d = {reading.d} given to a projector for "
                                 f"K = {self.k}, d = {self.d}")
        first, rest = channel_train.forward(
            self.params, reading.slot_x[None], reading.gates[None], reading.global_x[None],
            np.array([reading.speak], np.float32), tanh=np.tanh, sqrt=np.sqrt, ones=np.ones((HIDDEN, 1), np.float32))
        return np.concatenate([first, rest[0]]).astype(np.float32)

    def tokens(self, reading: Reading) -> SoftTokens:
        """``project``'s vectors as the soft tokens a turn carries (``VllmClient.complete(soft=...)``)."""
        return SoftTokens(self.project(reading))

    def save(self) -> bytes:
        return channel_train.save_params(self.params, {"k": self.k, "d": self.d, "width": self.width})

    @classmethod
    def load(cls, blob: bytes) -> Projector:
        try:
            params, meta = channel_train.load_params(blob)
            return cls(int(meta["k"]), int(meta["d"]), int(meta["width"]), params)
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectorError(f"not a projector this build reads: {error}") from None
