"""How replay's thinker is served (Task 17b): Glimmer on the vLLM server, as the night serves MIND.

The thinker's whole prompt is one user message, the brief and the rendered workspace
(``BriefedThinker``), sent through ``VllmClient.complete``, which cleans it, renders it with the
chat template at the server's reasoning strength and the client's day, and adds the reply's
header. That is how the night's thinker is called (``app/mind_night.py``); the dataset's rows hold
the same message, and the training script renders each row the same way, on its own day, so a
trained adapter sees the prompt it is served.

- **The date:** the night dates its prompts on the night's clock (``app/night.py``), so every
  call here is dated on the night whose evidence it sees: an arm on each night of the life in turn
  (``dated_core`` sets the ``NightClock`` before each loop call), and a branch on its start state's
  night, which its tools are masked to (its ``RunStart`` names the next night only to restart the
  stall count). A call with no night set is refused, never dated today.
- **Sampling:** branches at ``BRANCH_TEMPERATURE``, one seed each; the arms (M on base Glimmer,
  M+replay and M+shuffled on their adapters) at temperature 0, as the night does.
- **Failures are counted, never turned into text:** a failed call is empty text charged its cap,
  as the thinker port asks, and it is recorded (``calls``) with how every call ended (vLLM's
  ``finish_reason``), so a branch with a failed call is incomplete, a reply cut at its token cap
  never trains, and an arm with a failed call is never scored (``compare``). An adapter trained
  under another rendering is refused before any call (``arm_thinker``).

A reply records the model that wrote it: base Glimmer, or the adapter's name, which no replay
dataset accepts.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final

from anjo.sleep.mind.apd.arms import MindCore
from anjo.sleep.mind.inquiry.ports import InquiryPorts
from anjo.sleep.mind.inquiry.types import Budget, InquiryRun, ModelReply, Workspace
from anjo.sleep.mind.replay.spend import GlimmerServer, colab_client
from anjo.sleep.mind.replay.types import BRANCH_TEMPERATURE, CallEnd
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.spark_inquiry import THINK_MAX_TOKENS
from anjo.sleep.types import CheckpointRef, Evidence

logger = logging.getLogger(__name__)
ARM_TEMPERATURE: Final = 0.0


@dataclass
class NightClock:
    """The night the calls being made belong to; ``None`` before any is set."""

    day: date | None = None


@dataclass(frozen=True)
class ServedThinker:
    """The ``Thinker`` port on the vLLM server: ``adapter`` ``None`` is base Glimmer. ``calls``
    records how each call ended, in call order (one per ``Turn``)."""

    client: VllmClient
    clock: NightClock
    temperature: float
    seed: int | None = None
    adapter: CheckpointRef | None = None
    calls: list[CallEnd] = field(default_factory=list, compare=False)

    def propose(self, rendered: str, max_tokens: int) -> ModelReply:
        model = GLIMMER if self.adapter is None else self.adapter.id
        cap = min(max_tokens, THINK_MAX_TOKENS)
        if self.clock.day is None:
            raise ValueError("a replay prompt is dated on its run's night, and none is set")
        if cap <= 0:  # no allowance left: no call, and nothing to train on
            self.calls.append(CallEnd(None))
            return ModelReply(text="", tokens=0, model=model)
        try:
            result = self.client.on(self.clock.day).complete(
                self.adapter, [{"role": "user", "content": rendered}], temperature=self.temperature, seed=self.seed,
                max_tokens=cap,
            )
        except ServingError as error:
            logger.warning("replay's Glimmer call failed, charged its cap of %d tokens: %s", cap, error)
            self.calls.append(CallEnd(None, failed=True))
            return ModelReply(text="", tokens=cap, model=model)
        self.calls.append(CallEnd(result.finish_reason))
        return ModelReply(text=result.text, tokens=result.output_tokens, model=model)


def branch_thinkers(server: GlimmerServer) -> Callable[[int, date], ServedThinker]:
    """Each branch's thinker (``ReplayModels.thinker_for``): base Glimmer on Colab
    (``spend.colab_client``) at ``BRANCH_TEMPERATURE`` with the branch's seed, dated on the
    branch's night."""
    client = colab_client(server)

    def thinker_for(seed: int, day: date) -> ServedThinker:
        return ServedThinker(client, NightClock(day), BRANCH_TEMPERATURE, seed)

    return thinker_for


def arm_thinker(server: GlimmerServer, adapter: CheckpointRef | None, clock: NightClock, *,
                seed: int = 0) -> ServedThinker:
    """An arm's thinker for the 17b test, on Glimmer on Colab: base Glimmer for M, an adapter for
    M+replay or M+shuffled. An adapter trained under another rendering than the server's is
    refused here, before any call."""
    client = colab_client(server)
    if adapter is not None and adapter.rendering != client.rendering:
        raise ServingError(f"adapter {adapter.id} was trained under rendering {adapter.rendering}, "
                           f"not {client.rendering}; it is never served under another")
    return ServedThinker(client, clock, ARM_TEMPERATURE, seed, adapter)


def failed_calls(calls: Sequence[CallEnd]) -> int:
    return sum(1 for end in calls if end.failed)


def dated_core(core: MindCore, clock: NightClock) -> MindCore:
    """``core`` with ``clock`` set to each loop call's night before it runs, so an arm's thinker
    dates every prompt on the night it belongs to."""

    def notice(ws: Workspace, new: Sequence[Evidence], ports: InquiryPorts, budget: Budget, *, night: date,
               **rest: Any) -> InquiryRun:
        clock.day = night
        return core.notice(ws, new, ports, budget, night=night, **rest)

    def run_inquiry(ws: Workspace, ports: InquiryPorts, budget: Budget, *, night: date, as_of: datetime,
                    **rest: Any) -> InquiryRun:
        clock.day = night
        return core.run_inquiry(ws, ports, budget, night=night, as_of=as_of, **rest)

    return dataclasses.replace(core, notice=notice, run_inquiry=run_inquiry)
