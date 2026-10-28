"""MIND's core in a person's night: the night's ``MindCore`` port (``night/deps.py``), one step of
the one system. The app builds it under ``ANJO_SLEEP_MIND_CORE=1`` and loads this module by name
(it reaches torch and the latents module's training code).

**Dream** (step 2c). Her core state starts from the newest unit of an earlier night
(``MindStore.before``), so a rerun never observes a day twice; with none, or one kept under
another core model or unreadable, she starts fresh and the receipt says why. The day's evidence
is read through the encoder (``latents.Encoder``: the fake encoder until the live one serves)
and observed in time order, which gives each item its surprise weight (``Core.surprise_weights``,
the Sleep link). Then one dream (``Core.dream_states``, the night's seeded rng): its ignitions and
the core's predictions rank her threads, and the curriculum picks the evidence practice rehearses
(``curriculum.pick``; off with ``ANJO_SLEEP_DREAM_CURRICULUM=0``). The imagined states about the
picked threads, and the states where an imagined turn first flips her decision (``boundaries``:
where she is fragile), are kept by id with the real events they rest on, for the sleep objective. Last, she consolidates (fast memory into
slow, ``Core.consolidate``): the core's night is consolidation only, since rehearsing picks into
the core's own memory lowered recall on generated lives (``sleep_eval.ablations``); rehearsal is
Glimmer's practice. The consolidated state (``Dreamt.state``, her state after every event so far)
is what practice reads through the thought channel and what the night's rows are made under; its
id is the rows' ``core`` snapshot's ``state``. Consolidation leaves the core's predictions as they
were (its heads read no memory), so ``Dreamt.view.predictions`` is ``Core.predict`` of that state.

**Keep.** The night's unit: the state, the core's model id, the thought channel's projector version
and the adapter serving her (``MindStore.keep``). Nothing is written once the night is cancelled.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np

from anjo.sleep.mind.core import curriculum, latents, store
from anjo.sleep.mind.core import state as core_state
from anjo.sleep.mind.core.api import Core, Dream, DreamState
from anjo.sleep.mind.core.bridge import CoreView
from anjo.sleep.mind.core.evaluate import DREAM_STEPS
from anjo.sleep.mind.core.event import EventLatent
from anjo.sleep.mind.core.types import Decision, Ignition
from anjo.sleep.night.deps import Dreamt, MindCall
from anjo.sleep.stops import Cancelled
from anjo.sleep.types import Evidence

CURRICULUM_ENV: Final = "ANJO_SLEEP_DREAM_CURRICULUM"  # "0": practice seeds on the newest evidence, as before
REHEARSE_BUDGET: Final = 12  # evidence items the curriculum may pick for practice
ROLLOUTS: Final = 8  # the dream's imagined continuations


def curriculum_on(env: Mapping[str, str]) -> bool:
    return env.get(CURRICULUM_ENV, "").strip() != "0"


def from_env(env: Mapping[str, str], model: Path, sleep_root: Path) -> CoreNight:
    """The app's core night: the trained core at ``model``, over ``<sleep root>/mind_state/``,
    reading evidence through the fake encoder (of the core's width) until the live one serves."""
    core = Core.load(model)
    return CoreNight(core, sleep_root / store.STORE_DIR, latents.FakeEncoder(core.config.dim_in),
                     curriculum=curriculum_on(env))


@dataclass(frozen=True)
class _Start:
    state: core_state.CoreState
    unit: store.Unit | None
    fresh: str | None  # why she starts fresh, or None


def _seed(seed: int, person: str, night: date) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{person}:{night.isoformat()}".encode()).digest()[:8], "little")


def _ignition(ignition: Ignition) -> dict[str, Any]:
    return {"slot": ignition.slot, "other": ignition.other, "kind": ignition.kind, "score": round(ignition.score, 6),
            "cites": list(ignition.cited_event_ids)}


def _decision(decision: Decision) -> dict[str, Any]:
    return {"speak": decision.speak, "slot": decision.slot, "p_speak": round(decision.p_speak, 6)}


def _guard(call: MindCall) -> None:
    if call.cancelled():
        raise Cancelled("the night was cancelled")


class CoreNight:
    """The night's ``MindCore``: ``core`` over one store root, reading evidence through ``encoder``."""

    available = True

    def __init__(
        self, core: Core, root: Path, encoder: latents.Encoder, *, curriculum: bool = True, seed: int = 0,
        budget: int = REHEARSE_BUDGET,
    ) -> None:
        if encoder.spec.d != core.config.dim_in:
            raise ValueError(f"the encoder writes {encoder.spec.d} dims; the core reads {core.config.dim_in}")
        self.core, self.root, self.encoder = core, root, encoder
        self.curriculum, self.seed, self.budget = curriculum, seed, budget

    def _store(self, person: str) -> store.MindStore:
        return store.MindStore(self.root, person)

    def _start(self, held: store.MindStore, person: str, night: date) -> _Start:
        unit = held.before(night)
        if unit is None:
            return _Start(self.core.init_state(person), None, "first night")
        if unit.model != self.core.model_id:
            return _Start(self.core.init_state(person), unit, "another core model")
        try:
            return _Start(core_state.load(held.state(unit.state), model_id=self.core.model_id), unit, None)
        except (store.StoreError, core_state.StateError):
            return _Start(self.core.init_state(person), unit, "state unreadable")

    def _events(self, person: str, day: Sequence[Evidence]) -> list[EventLatent]:
        ordered = sorted(day, key=lambda item: (item.created_at, item.id))
        inputs = [latents.EventInput(item.id, person, item.created_at, item.kind, item.text) for item in ordered]
        matrix = latents.encode_events(inputs, self.encoder).matrix
        return [EventLatent(e.event_id, person, e.t, e.kind, matrix[n]) for n, e in enumerate(inputs)]

    def _targets_of(self, call: MindCall, cited: set[str]) -> dict[str, str]:
        return {item.id: item.target_id for item in call.store.stored(call.night)
                if item.id in cited and item.target_id is not None}

    def dream(self, call: MindCall, day: Sequence[Evidence]) -> Dreamt:
        held = self._store(call.user_id)
        start = self._start(held, call.user_id, call.night)
        events = self._events(call.user_id, day)
        lived, weights = self.core.surprise_weights(start.state, events)
        dream = self.core.dream_states(lived, DREAM_STEPS, np.random.default_rng(_seed(self.seed, call.user_id, call.night)),
                                       rollouts=ROLLOUTS)
        seen = self.core.predict(lived)
        decision = self.core.decide(lived, dream.ignitions)
        picks = curriculum.pick(seen, dream.ignitions, weights, budget=self.budget) if self.curriculum else None
        pressure = self.core.sleep_pressure(lived)
        _guard(call)  # nothing is written for a person deleted mid-night
        slept = self.core.consolidate(lived)
        state_id = held.put_state(core_state.save(slept))
        dreams = self._imagined(held, dream, picks)
        cited = {c for slot in seen.slots for c in slot.cites}
        view = CoreView(decision, seen, self._targets_of(call, cited))
        receipt = {
            "state": state_id, "from_version": None if start.unit is None else start.unit.version,
            "fresh": start.fresh, "model": self.core.model_id, "encoder": self.encoder.spec.spec_id(),
            "events": len(events), "weights": {k: round(v, 6) for k, v in weights.items()},
            "ignitions": [_ignition(i) for i in dream.ignitions], "decision": _decision(decision),
            "picks": None if picks is None else curriculum.record(picks), "dreams": dreams,
            "boundaries": [{**self._kept_state(held, b.dream), "steps": b.steps, "speak": b.speak, "slot": b.slot}
                           for b in dream.boundaries],
            "pressure": round(pressure, 6),
        }
        return Dreamt(state_id=state_id, state=slept, rehearse=() if picks is None else picks.evidence_ids,
                      weights=weights, view=view, receipt=receipt)

    def _imagined(self, held: store.MindStore, dream: Dream, picks: curriculum.Picks | None) -> list[dict[str, Any]]:
        """Each imagined state about the picked threads (every one, with no curriculum), kept by id."""
        states: Sequence[DreamState] = dream.states if picks is None else dream.on([t.slot for t in picks.threads])
        return [self._kept_state(held, d) for d in states]

    @staticmethod
    def _kept_state(held: store.MindStore, dreamt: DreamState) -> dict[str, Any]:
        return {"state": held.put_state(core_state.save(dreamt.state)), "threads": list(dreamt.threads),
                "rests_on": list(dreamt.rests_on)}

    def keep(self, call: MindCall, dreamt: Dreamt, *, lora: str | None, projector: str | None) -> dict[str, object]:
        if dreamt.state_id is None:
            raise ValueError("a night that never dreamt has no unit to keep")
        _guard(call)
        unit = self._store(call.user_id).keep(call.night, dreamt.state_id, model=self.core.model_id,
                                              projector=projector, lora=lora)
        return store.unit_record(unit)
