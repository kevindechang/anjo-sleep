"""One person's core state, and its versioned save and load. Torch-free.

The state is everything the shared core keeps for one person: the slots and their usage and
opening times, the fast-weight memory and its momentum, the global vector, the clock, the real
events each slot cites, and the person's head adaptation (``delta``, empty until a night adapts
it). It also keeps the surprise sleep hasn't taken in yet (``pending``), the last event it observed
(``through``: its time in days and its id, so a night observes only what comes after it, and a
killed night's day is caught up the next), and whether any event reached it through the fake
encoder (``fake_latents``: sticky, so a state built on stand-in latents always says so). It names
the core it belongs to (``model_id``): the slots and the memory mean something only
under those weights, so ``load`` refuses a state of another core; a person's state for a new core
is rebuilt by replaying their events.

``save`` writes a zip of ``.npy`` arrays plus a JSON ``meta`` entry, with fixed zip times, so the
same state always gives the same bytes (``allow_pickle`` stays off on the way back).
"""
from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

STATE_VERSION: Final = 2  # 2: pending, through and fake_latents
_ARRAYS: Final = ("slots", "usage", "opened", "memory", "momentum", "glob")
_DELTA: Final = "delta__"
_ZIP_TIME: Final = (1980, 1, 1, 0, 0, 0)
Cite = tuple[str, float]  # an event id and the share it wrote into the slot


class StateError(ValueError):
    """A saved state this build or this core can't read."""


@dataclass(frozen=True, eq=False)
class CoreState:
    person: str
    model_id: str
    slots: NDArray[np.float32]  # (K, d)
    usage: NDArray[np.float32]  # (K,)
    opened: NDArray[np.float32]  # (K,) days
    memory: NDArray[np.float32]  # (d, d)
    momentum: NDArray[np.float32]  # (d, d)
    glob: NDArray[np.float32]  # (d,)
    now: float | None  # the latest event, in days; None before any
    observed: int
    cites: tuple[tuple[Cite, ...], ...]  # per slot, the strongest write first
    delta: Mapping[str, NDArray[np.float32]] = field(default_factory=dict)
    pending: float = 0.0  # surprise observed since the last consolidation: what sleep hasn't taken in
    through: tuple[float, str] | None = None  # the last event observed: (days, id); None before any
    fake_latents: bool = False  # any event observed through the fake encoder

    def __post_init__(self) -> None:
        for name in _ARRAYS:
            array = np.array(getattr(self, name), dtype=np.float32)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        k, d = self.slots.shape
        if self.usage.shape != (k,) or self.opened.shape != (k,) or self.glob.shape != (d,):
            raise StateError("a state's slots, usage, opening times and global vector disagree")
        if self.memory.shape != (d, d) or self.momentum.shape != (d, d) or len(self.cites) != k:
            raise StateError("a state's memory or cites don't fit its slots")
        frozen = {}
        for name, value in self.delta.items():
            array = np.array(value, dtype=np.float32)
            array.setflags(write=False)
            frozen[name] = array
        object.__setattr__(self, "delta", MappingProxyType(frozen))

    def cited(self, slot: int) -> tuple[str, ...]:
        return tuple(event_id for event_id, _ in self.cites[slot])

    @property
    def known(self) -> frozenset[str]:
        """Every event id a slot cites: all real, since only ``observe`` adds cites."""
        return frozenset(event_id for row in self.cites for event_id, _ in row)

    def after(self, at: float, event_id: str) -> bool:
        """Whether an event at ``at`` days with ``event_id`` comes after every one observed."""
        return self.through is None or (at, event_id) > self.through


def equal(a: CoreState, b: CoreState) -> bool:
    """Same person, core, arrays, clock, cites and adaptation."""
    same_meta = (a.person, a.model_id, a.now, a.observed, a.cites, a.pending, a.through, a.fake_latents) == (
        b.person, b.model_id, b.now, b.observed, b.cites, b.pending, b.through, b.fake_latents)
    same_arrays = all(np.array_equal(getattr(a, name), getattr(b, name)) for name in _ARRAYS)
    same_delta = set(a.delta) == set(b.delta) and all(np.array_equal(a.delta[n], b.delta[n]) for n in a.delta)
    return same_meta and same_arrays and same_delta


def _npy(array: NDArray[np.generic]) -> bytes:
    out = io.BytesIO()
    np.lib.format.write_array(out, np.asarray(array, order="C"), allow_pickle=False)
    return out.getvalue()


def save(state: CoreState) -> bytes:
    """The state as bytes: the same state, the same bytes."""
    meta = {
        "version": STATE_VERSION, "person": state.person, "model_id": state.model_id, "now": state.now,
        "observed": state.observed, "cites": [[[eid, share] for eid, share in row] for row in state.cites],
        "pending": state.pending, "through": None if state.through is None else list(state.through),
        "fake_latents": state.fake_latents,
    }
    entries = {name: getattr(state, name) for name in _ARRAYS}
    entries.update({_DELTA + name: value for name, value in sorted(state.delta.items())})
    entries["meta"] = np.array(json.dumps(meta, sort_keys=True))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, array in entries.items():
            zf.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIME), _npy(array))
    return out.getvalue()


def load(blob: bytes, *, model_id: str) -> CoreState:
    """The state ``save`` wrote; ``StateError`` for another version, another core, or bad bytes."""
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as held:
            arrays = {name: held[name] for name in held.files}
        meta = json.loads(str(arrays.pop("meta")))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise StateError(f"not a saved core state: {type(error).__name__}") from None
    if meta.get("version") != STATE_VERSION:
        raise StateError(f"state version {meta.get('version')!r}; this build reads version {STATE_VERSION}")
    if meta.get("model_id") != model_id:
        raise StateError("this state belongs to another core model; replay the person's events")
    try:
        return CoreState(
            person=str(meta["person"]), model_id=str(meta["model_id"]),
            **{name: arrays[name] for name in _ARRAYS},
            now=None if meta["now"] is None else float(meta["now"]), observed=int(meta["observed"]),
            cites=tuple(tuple((str(eid), float(share)) for eid, share in row) for row in meta["cites"]),
            delta={name[len(_DELTA):]: value for name, value in arrays.items() if name.startswith(_DELTA)},
            pending=float(meta["pending"]),
            through=None if meta["through"] is None else (float(meta["through"][0]), str(meta["through"][1])),
            fake_latents=bool(meta["fake_latents"]),
        )
    except (KeyError, TypeError) as error:
        raise StateError(f"a saved state is missing a part: {error}") from None
