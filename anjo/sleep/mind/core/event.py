"""MIND's event record, the one the core reads and the latents module writes: numpy only, so the
night and the app can name it without loading the latent encoder or any training code."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Matrix: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, eq=False)
class EventLatent:
    """The design's interface: one event and its latent ``vec`` (float32, shape ``(D,)``)."""

    event_id: str
    life_or_person: str
    t: datetime
    kind: str
    vec: Matrix
