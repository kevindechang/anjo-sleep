"""Replay's registered setting (Task 17b): λ, the score's token penalty, read from
``prereg/thresholds.json``, the file the night reads its token cap from. It is a placeholder there,
dated: chosen a priori on fakes, to be confirmed on development lives before registration. The
pure modules (``score``, ``dataset``) take it as an argument."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from anjo.sleep.gates import THRESHOLDS

TOKEN_PENALTY_KEY: Final = "replay_token_penalty"


def token_penalty(path: Path = THRESHOLDS) -> float:
    """λ: score points per 1,000 tokens."""
    value = json.loads(path.read_text(encoding="utf-8"))[TOKEN_PENALTY_KEY]
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{TOKEN_PENALTY_KEY} is a number of score points per 1,000 tokens, at least 0")
    return float(value)
