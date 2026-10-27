"""Which model outputs may train her weights (spec §3).

Every evidence item and every training example carries its ancestry: each model whose output it
derives from, through everything that model read. Training takes an item only when every
ancestor is cleared: Spark Open (licensed for training) and Glimmer (open weights). The person's
own words have no ancestor. Hosted Muse Spark and Wonder outputs train only once Meta allows it
in writing, which ``hosted_in_training`` stands for. Unknown ancestry (``None``) and every other
model (Claude, Gemini, OpenAI) never train.
"""
from __future__ import annotations

from typing import Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.meta_api import PINNED_MODELS

GLIMMER: Final = "meta-models/Muse-Glimmer-30B"
CLEARED: Final = frozenset({SPARK_OPEN, GLIMMER})
# A hosted output records its pinned, versioned id; the bare names are listed too.
HOSTED: Final = frozenset(PINNED_MODELS) | frozenset(PINNED_MODELS.values())


def may_train(ancestry: frozenset[str] | None, *, hosted_in_training: bool) -> bool:
    """Whether text with this ancestry may train: an allowlist, so an unnamed model never does."""
    if ancestry is None:
        return False
    allowed = CLEARED | HOSTED if hosted_in_training else CLEARED
    return ancestry <= allowed
