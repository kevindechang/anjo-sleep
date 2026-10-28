"""How replay's thinker is served (Task 17b): Glimmer on the vLLM server, as the night serves MIND.

The thinker's whole prompt is one user message, the brief and the rendered workspace
(``BriefedThinker``), sent through ``VllmClient.complete``, which cleans it, renders it with the
chat template at the server's reasoning strength and the client's day, and adds the reply's
header. That is how the night's thinker is called (``app/mind_night.py``); the dataset's rows hold
the same message, and the training script renders them the same way, so a trained adapter sees
the prompt it is served.

Every replay prompt, and every prompt of the three arms of the 17b test, is dated
``RENDER_DAY``: the training script renders one date per run, so one date for all keeps the rows
and the served prompts identical. Branches sample at ``BRANCH_TEMPERATURE``, one seed each; the
arms (M on base Glimmer, M+replay and M+shuffled on their adapters) run at temperature 0, as the
night does. A failed call is empty text charged its cap, as in the night. A reply records the
model that wrote it: base Glimmer, or the adapter's name, which no replay dataset accepts.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from anjo.sleep.mind.inquiry.types import ModelReply
from anjo.sleep.mind.replay.types import BRANCH_TEMPERATURE, RENDER_DAY
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.spark_inquiry import THINK_MAX_TOKENS
from anjo.sleep.types import CheckpointRef

logger = logging.getLogger(__name__)
ARM_TEMPERATURE: Final = 0.0


@dataclass(frozen=True)
class ServedThinker:
    """The ``Thinker`` port on the vLLM server: ``adapter`` ``None`` is base Glimmer."""

    client: VllmClient
    temperature: float
    seed: int | None = None
    adapter: CheckpointRef | None = None

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        model = GLIMMER if self.adapter is None else self.adapter.id
        cap = min(max_tokens, THINK_MAX_TOKENS)
        if cap <= 0:
            return ModelReply(text="", tokens=0, model=model)
        try:
            result = self.client.complete(
                self.adapter, [{"role": "user", "content": rendered}], temperature=self.temperature, seed=self.seed,
                max_tokens=cap,
            )
        except ServingError as error:
            logger.warning("replay's Glimmer call failed, charged its cap of %d tokens: %s", cap, error)
            return ModelReply(text="", tokens=cap, model=model)
        return ModelReply(text=result.text, tokens=result.output_tokens, model=model)


def branch_thinkers(client: VllmClient) -> Callable[[int], ServedThinker]:
    """Each branch's thinker (``ReplayModels.thinker_for``): base Glimmer at ``BRANCH_TEMPERATURE``
    with the branch's seed."""
    dated = client.on(RENDER_DAY)

    def thinker_for(seed: int) -> ServedThinker:
        return ServedThinker(dated, BRANCH_TEMPERATURE, seed)

    return thinker_for


def arm_thinker(client: VllmClient, adapter: CheckpointRef | None, *, seed: int = 0) -> ServedThinker:
    """An arm's thinker for the 17b test: base Glimmer for M, an adapter for M+replay or M+shuffled."""
    return ServedThinker(client.on(RENDER_DAY), ARM_TEMPERATURE, seed, adapter)
