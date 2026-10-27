"""The nightly gates (Task 5): general skills, collapse and safety. Each takes a checkpoint and
returns a ``GateResult`` against the registered thresholds in ``prereg/thresholds.json``, which
this package reads and never writes.

Every gate compares the checkpoint with the base model on the same items, sampled on the same
server at the same precision (``anjo.sleep.serving.client``). The base side of a comparison is
the same for everyone in a batch, so a gate keeps it once per run. Requests go out in parallel,
up to the server's concurrency, and come back in item order, so a gate's verdict never depends on
timing. Reasons name item ids or case indices, never their words.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeVar

from anjo.sleep.serving import config

THRESHOLDS: Final = Path(__file__).parents[1] / "prereg" / "thresholds.json"
EVAL_SEED: Final = 20261028
TEMPERATURE: Final = 1.0  # registered for every arm and gate, with top_p 0.95 and top_k 64

In = TypeVar("In")
Out = TypeVar("Out")


@dataclass(frozen=True)
class GateThresholds:
    general_points: float
    collapse_points: float
    safety_regressions: int


def load_thresholds(path: Path = THRESHOLDS) -> GateThresholds:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GateThresholds(
        general_points=float(data["general_skills"]["reject_if_accuracy_drops_more_than_points"]),
        collapse_points=float(data["collapse"]["reject_if_above_base_rate_by_points"]),
        safety_regressions=int(data["safety"]["max_regressions"]),
    )


def in_order(fn: Callable[[In], Out], items: Sequence[In], workers: int = config.CONCURRENT_REQUESTS) -> list[Out]:
    """``fn`` over ``items`` on up to ``workers`` threads, results in item order. The first error
    raises once every call has ended."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        return list(pool.map(fn, items))


def listed(indices: Sequence[int] | Sequence[str], limit: int = 20) -> str:
    """Ids or indices for a reason, the first ``limit`` of them and how many more."""
    shown = ", ".join(str(item) for item in indices[:limit])
    more = len(indices) - limit
    return shown + (f" and {more} more" if more > 0 else "")
