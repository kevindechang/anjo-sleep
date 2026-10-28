"""MIND's core in the night's steps: the one system's loop, where each step feeds the next.

With ``ANJO_SLEEP_MIND_CORE=1`` and MIND's inquiry on, the app gives the night a ``MindCore`` port
(``deps.core``) that serves the demo persona's people only; for anyone else, and with the switch
off (``NoCore``), every hook here leaves the night as it was, and her night log has no dream.

- **Dream** (step 2d, after the notice pass, before the probes and the choice): the core observes
  her stored evidence it has not observed yet (a killed night's day too), from her last kept
  state, dreams, opens a question through the kernel for each ignition that cites her real
  evidence, picks the threads she is least sure of or most at risk on, consolidates, and keeps the
  state the night's rows are made under. Its one log record holds the dream: the surprise
  weights, the ignitions, the questions opened and the ignitions dropped (why), the curriculum's
  picks and the imagined states. A nap dreams too: every part of it is CPU work, so it wakes no
  Glimmer and holds nothing on ``modal_usd``; its questions wait, with the deferred inquiry, for
  her next deep sleep. A person the core refuses (fake latents for anyone not synthetic) gets a
  no-op dream record saying why, and the rest of her night as it would be without the core.
- **Inquire** reads the dream: the call carries the core's view, so the core picks the question.
- **Rehearse**: practice is seeded on the picked evidence, at the same scenes and token cap; with
  nothing picked, on the newest evidence as before.
- **Learn**: each of tonight's training rows carries its loss weight, the mean surprise of the
  evidence it was made from (the Sleep link); the core state it was made under is the thought
  channel's row field (``core``, its ``state`` is ``p.dreamt.state_id``). The training step's
  record says what the training job did with the weights (``surprise_weights``), as its receipt
  reports it (applied in ``sft_dpo_seed``'s one loss, not under another objective), or that the job
  did not say.
- **Read**: the thought channel's ``mind_state`` is the core port's ``reading``: this dream's
  state, its id and the core's predictions, and the projector her last unit kept. Practice and
  training ask for it in the GPU session, after every CPU step, so always after the dream.
- **Keep**: her unit (``mind.core.store``) holds the core state, the projector and the adapter
  that serve her, together. A night that promotes keeps it first, with the adapter it trained and
  the projector trained with it: if it can't be kept, nothing is promoted (the promote step fails),
  so a new adapter never serves beside an old projector. Any other night keeps it when the night
  ends, however it ends (unless the person was deleted), with the adapter that serves and the
  projector the dream read her through. The dream reads her through her last unit's projector only
  when that unit names the adapter that serves; else through the newest unit that does, or the
  shared projector, and its record says so (``projector_fallback``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from anjo.sleep.night.deps import MindCall, NightDeps
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.types import Evidence

if TYPE_CHECKING:
    from anjo.sleep.night.state import PersonNight


class CoreSteps:
    """The night runner's hooks for MIND's core."""

    deps: NightDeps

    def _job(self, p: PersonNight, step: str) -> str:
        raise NotImplementedError

    def _call(self, p: PersonNight, step: str) -> MindCall:
        return MindCall(user_id=p.uid, night=p.night, as_of=p.as_of, at=p.stamp, store=p.store, targets=p.targets,
                        spend=p.spend, wonder=p.wonder, actions_left=p.actions_left, job=self._job(p, step),
                        cancelled=lambda: self.deps.cancel(p.uid), core_view=p.dreamt.view)

    def _dreams(self, p: PersonNight) -> tuple[tuple[str, Callable[[PersonNight], Mapping[str, Any]]], ...]:
        """The dream step, only for a person the core serves: for anyone else the night's log is
        today's to the byte."""
        return (("dream", self._dream),) if self.deps.core.serves(p.uid) else ()

    def _dream(self, p: PersonNight) -> Mapping[str, Any]:
        """Her stored evidence through tonight, never a practice scene or a held-out item; the core
        observes only what its state has not."""
        evidence = [item for item in p.store.stored(p.night) if not item.held_out and item.kind != "practice"]
        serving = self.deps.registry.last_promoted(p.uid)
        p.dreamt = self.deps.core.dream(replace(self._call(p, "dream"), serving=None if serving is None else serving.id),
                                        evidence)
        return dict(p.dreamt.receipt)

    def _seeded(self, p: PersonNight) -> list[Evidence]:
        """What practice is seeded on: the dream's picks it may train on, else the trainable evidence as before."""
        trainable = list(p.store.trainable(p.night))
        picked = frozenset(p.dreamt.rehearse)
        return [item for item in trainable if item.id in picked] or trainable

    def _weighted(self, p: PersonNight, data: Dataset) -> Dataset:
        """Tonight's rows weighted by the mean surprise of the evidence they were made from; a row
        made from none of today's evidence, or on another night, keeps 1.0."""
        if not p.dreamt.weights:
            return data
        made, weights = self.deps.transcripts(p.uid), p.dreamt.weights
        found = {}
        for example in data.examples:
            transcript = made.get(example.id)
            known = [] if transcript is None or transcript.night != p.night else [
                weights[i] for i in transcript.f_ids if i in weights]
            if known:
                found[example.id] = math.fsum(known) / len(known)
        return data.weighted(found)

    def _surprise(self, p: PersonNight, receipt: Mapping[str, Any]) -> dict[str, Any]:
        """For a night whose rows carry surprise weights: whether the training job applied them,
        as it reported (``train.colab_train``), else that it did not say."""
        if p.data is None or all(example.weight == 1.0 for example in p.data.examples):
            return {}
        said = receipt.get("surprise_weights")
        return {"surprise_weights": said if isinstance(said, str) else "not applied (the training job did not report)"}

    def _promoted_unit(self, p: PersonNight, adapter: str) -> dict[str, Any]:
        """Her unit with the adapter about to serve and the projector trained with it (none when the
        channel was off), kept before the promotion; a failure to keep it raises, and nothing is promoted."""
        if p.dreamt.state_id is None or self.deps.cancel(p.uid):
            return {}
        trained = p.receipts.get("train", {}).get("receipt", {}).get("projector")
        unit = self.deps.core.keep(self._call(p, "promote"), p.dreamt, lora=adapter,
                                   projector=trained if isinstance(trained, str) else None)
        return {"mind_unit": dict(unit)}

    def _kept(self, p: PersonNight) -> dict[str, Any]:
        """The night's unit, for the summary: the one its promotion kept, else kept now; a disk or store
        failure to keep it is recorded, not raised."""
        if "mind_unit" in p.receipts.get("promote", {}):
            return {"mind_unit": p.receipts["promote"]["mind_unit"]}
        if p.dreamt.state_id is None or self.deps.cancel(p.uid):  # a deleted person's mind is never written again
            return {}
        serving = self.deps.registry.last_promoted(p.uid)
        try:
            unit = self.deps.core.keep(self._call(p, "finish"), p.dreamt, lora=None if serving is None else serving.id,
                                       projector=p.dreamt.projector)
        except (OSError, ValueError) as error:  # the night already ended: its summary says the unit wasn't kept
            return {"mind_unit": None, "mind_unit_error": type(error).__name__}
        return {"mind_unit": dict(unit)}
