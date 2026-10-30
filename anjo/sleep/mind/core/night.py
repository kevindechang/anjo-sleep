"""MIND's core in a person's night: the night's ``MindCore`` port (``night/deps.py``), one step of
the one system. The app builds it under ``ANJO_SLEEP_MIND_CORE=1`` for the people it ``serves``
(the demo persona's) and loads this module by name (it reaches torch and the latents module's
training code).

**Dream** (step 2c). Her core state starts from the newest unit of an earlier night
(``MindStore.before``); with none, or one kept under another core model or unreadable, she starts
fresh and the receipt says why. Of the evidence the night hands it (all of hers through tonight),
the core observes, in time order, what its state has not: each item after the last one it
observed (``CoreState.through``) and not after the night's clock. So a rerun never observes a day
twice, and a night killed before its unit was kept is caught up the next night
(``events_before_tonight`` counts those). The evidence is read through the encoder
(``latents.Encoder``: the fake encoder until the live one serves); a state that read any fake
latent says so (``CoreState.fake_latents``). Fake latents are refused for anyone but a synthetic
person unless ``ANJO_SLEEP_MIND_FAKE_LATENTS=1`` (development only): her night then skips the core,
the dream step's record (a no-op) and the log saying why, and runs on as it would without it. Observing gives
each item its surprise weight (``Core.surprise_weights``, bounded and averaging 1: the Sleep link).
Then one dream (``Core.dream_states``, the night's seeded rng). Each ignition opens a question in
her workspace through the kernel (``bridge.ignite``, citing her real evidence only; a collision
only with ``ANJO_SLEEP_MIND_CONTRADICTIONS=1``, off by default), logged under tonight's notice pass (a notice ``RunStart`` for tonight is logged first when
there was none); once tonight's inquiry has begun (a rerun of a killed night) nothing is opened,
so its resume reads the log it left. Its ignitions and the core's predictions rank her threads,
and the curriculum picks the evidence practice rehearses (``curriculum.pick``; off with
``ANJO_SLEEP_DREAM_CURRICULUM=0``). The imagined states about the picked threads, and the states
where an imagined turn first flips her decision (``boundaries``: where she is fragile), are kept
by id with the real events they rest on, for the sleep objective. Last, she consolidates (fast
memory into slow, ``Core.consolidate``): the core's night is consolidation only, since rehearsing picks into
the core's own memory lowered recall on generated lives (``sleep_eval.ablations``); rehearsal is
Glimmer's practice. The consolidated state (``Dreamt.state``, her state after every event so far)
is what practice reads through the thought channel and what the night's rows are made under; its
id is the rows' ``core`` snapshot's ``state``. Consolidation leaves the core's predictions as they
were (its heads read no memory), so ``Dreamt.view.predictions`` is ``Core.predict`` of that state.

**Read.** ``reading(person, night)`` is the thought channel's ``MindState``: tonight's dream as
``Dreamt.reading`` gives it (the consolidated state, its id, the core's predictions on it, and the
projector her last kept unit names, when that unit is this core's), None for anyone it doesn't
serve or a night it skipped. Asked before that night's dream step, it refuses (``ValueError``).
``imagined(person, night, state)`` is one of the states tonight's dream kept (in its receipt's
``dreams`` or ``boundaries``), read back from her store, with the core's predictions on it and
tonight's projector, for the dreams through the thought channel; any other id is refused.

**Keep.** The night's unit: the state, the core's model id, the thought channel's projector version
and the adapter serving her (``MindStore.keep``). Nothing is written once the night is cancelled.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from anjo.sleep.mind.core import bridge, curriculum, latents, store
from anjo.sleep.mind.core import state as core_state
from anjo.sleep.mind.core.api import Core, Dream, DreamState
from anjo.sleep.mind.core.evaluate import DREAM_STEPS
from anjo.sleep.mind.core.event import EventLatent
from anjo.sleep.mind.core.types import Decision, Ignition, days_of
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.loop_mark import is_loop_start
from anjo.sleep.mind.inquiry.ports import WorkspaceLog
from anjo.sleep.mind.inquiry.types import FULL_RULES, Budget, Rejected, RunStart, Step, WorkspaceEvent
from anjo.sleep.night.channel_night import MindReading
from anjo.sleep.night.deps import Dreamt, MindCall
from anjo.sleep.stops import Cancelled
from anjo.sleep.types import Evidence

CURRICULUM_ENV: Final = "ANJO_SLEEP_DREAM_CURRICULUM"  # "0": practice seeds on the newest evidence, as before
FAKE_LATENTS_ENV: Final = "ANJO_SLEEP_MIND_FAKE_LATENTS"  # "1": fake latents for anyone (development only)
REHEARSE_BUDGET: Final = 12  # evidence items the curriculum may pick for practice
ROLLOUTS: Final = 8  # the dream's imagined continuations

logger = logging.getLogger(__name__)


def curriculum_on(env: Mapping[str, str]) -> bool:
    return env.get(CURRICULUM_ENV, "").strip() != "0"


def from_env(
    env: Mapping[str, str], model: Path, sleep_root: Path, *, people: frozenset[str],
    workspace: Callable[[str], WorkspaceLog], synthetic: Callable[[str], bool],
) -> CoreNight:
    """The app's core night for ``people``: the trained core at ``model``, over
    ``<sleep root>/mind_state/``, opening questions in each one's ``workspace`` log, reading
    evidence through the fake encoder (of the core's width) until the live one serves."""
    core = Core.load(model)
    return CoreNight(core, sleep_root / store.STORE_DIR, latents.FakeEncoder(core.config.dim_in),
                     curriculum=curriculum_on(env), people=people, workspace=workspace, synthetic=synthetic,
                     fake_for_anyone=env.get(FAKE_LATENTS_ENV, "").strip() == "1",
                     contradictions=bridge.contradictions_on(env))


@dataclass(frozen=True)
class _Start:
    state: core_state.CoreState
    unit: store.Unit | None
    fresh: str | None  # why she starts fresh, or None


@dataclass(frozen=True)
class _Guarded:
    """Her workspace log, with the cancel flag read before each append (RF2)."""

    inner: WorkspaceLog
    cancelled: Callable[[], bool]

    def append(self, event: WorkspaceEvent) -> None:
        if self.cancelled():
            raise Cancelled("the night was cancelled")
        self.inner.append(event)

    def events(self) -> tuple[WorkspaceEvent, ...]:
        return self.inner.events()


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


def _runs_tonight(events: Sequence[WorkspaceEvent], night: date) -> tuple[RunStart, ...]:
    """Tonight's notice and inquiry runs; the practice loop's own start is neither, so it neither
    blocks an ignition nor starts tonight's notice pass."""
    return tuple(e.step.entry for e in events
                 if isinstance(e.step.entry, RunStart) and e.step.entry.night == night
                 and not is_loop_start(e.step.entry))


def ignite(log: WorkspaceLog, ignitions: Sequence[Ignition], call: MindCall, *, collisions: bool = False) -> dict[str, Any]:
    """Open a question for each ignition in her workspace (``bridge.ignite``; a collision only with
    ``collisions``), under tonight's notice pass; the receipt holds the questions' ids and each
    ignition dropped, with why."""
    events = log.events()
    tonight = _runs_tonight(events, call.night)
    if any(start.call == "inquiry" for start in tonight):
        return {"questions_opened": [], "ignitions_dropped": [], "ignite": "skipped: tonight's inquiry began"}
    ws = kernel.fold(events)
    if not tonight and ignitions:  # the notice pass found nothing new: tonight starts here
        start = RunStart(call.night, call.as_of, "notice", FULL_RULES, "priority", None, Budget(), False,
                         kernel.DENYLIST_VERSION)
        begun = kernel.apply(ws, Step(seq=ws.seq, at=call.as_of, entry=start))
        if isinstance(begun, Rejected):
            raise RuntimeError(f"the kernel refused tonight's start ({begun.code})")
        log.append(begun.event)
        ws = begun.ws
    before = {q.id for q in ws.questions}
    done = bridge.ignite(ws, ignitions, log, targets=call.targets.open_targets(call.as_of), as_of=call.as_of,
                         collisions=collisions)
    return {"questions_opened": [q.id for q in done.ws.questions if q.id not in before],
            "ignitions_dropped": [{"slot": i.slot, "other": i.other, "kind": i.kind, "why": why}
                                  for i, why in done.dropped],
            "ignite": "opened"}


class CoreNight:
    """The night's ``MindCore``: ``core`` over one store root, reading evidence through ``encoder``,
    for ``people`` only, opening questions in each one's ``workspace`` log (none: it opens none)."""

    def __init__(
        self, core: Core, root: Path, encoder: latents.Encoder, *, curriculum: bool = True, seed: int = 0,
        budget: int = REHEARSE_BUDGET, people: frozenset[str] = frozenset(),
        workspace: Callable[[str], WorkspaceLog] | None = None, synthetic: Callable[[str], bool] = lambda _: False,
        fake_for_anyone: bool = False, contradictions: bool = False,
    ) -> None:
        if encoder.spec.d != core.config.dim_in:
            raise ValueError(f"the encoder writes {encoder.spec.d} dims; the core reads {core.config.dim_in}")
        self.core, self.root, self.encoder = core, root, encoder
        self.curriculum, self.seed, self.budget = curriculum, seed, budget
        self.people, self.workspace, self.synthetic = people, workspace, synthetic
        self.fake_for_anyone, self.contradictions = fake_for_anyone, contradictions
        self._tonight: dict[tuple[str, date], Dreamt] = {}  # each night's dream, for ``dreamt`` and ``reading``

    @property
    def fake(self) -> bool:
        return self.encoder.spec.encoder == "fake"

    def serves(self, user_id: str) -> bool:
        return user_id in self.people

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

    def _events(self, call: MindCall, state: core_state.CoreState, evidence: Sequence[Evidence]) -> list[EventLatent]:
        """What the state has not observed, through the night's clock, in time order."""
        unseen = [item for item in evidence if item.created_at <= call.as_of
                  and state.after(days_of(item.created_at), item.id)]
        ordered = sorted(unseen, key=lambda item: (item.created_at, item.id))
        inputs = [latents.EventInput(item.id, call.user_id, item.created_at, item.kind, item.text) for item in ordered]
        matrix = latents.encode_events(inputs, self.encoder).matrix
        return [EventLatent(e.event_id, call.user_id, e.t, e.kind, matrix[n]) for n, e in enumerate(inputs)]

    def _targets_of(self, call: MindCall, cited: set[str]) -> dict[str, str]:
        return {item.id: item.target_id for item in call.store.stored(call.night)
                if item.id in cited and item.target_id is not None}

    def _refusal(self, user_id: str) -> str | None:
        if self.fake and not (self.fake_for_anyone or self.synthetic(user_id)):
            return f"skipped: fake latents serve only a synthetic person's night ({FAKE_LATENTS_ENV}=1 allows them)"
        return None

    def dreamt(self, user_id: str, night: date) -> Dreamt | None:
        if not self.serves(user_id):
            return None
        done = self._tonight.get((user_id, night))
        if done is None:
            raise ValueError("MIND's state was asked for before the night's dream step")
        return done

    def reading(self, user_id: str, night: date) -> MindReading | None:
        dreamt = self.dreamt(user_id, night)
        return None if dreamt is None else dreamt.reading()

    def imagined(self, user_id: str, night: date, state_ref: str) -> MindReading | None:
        dreamt = self.dreamt(user_id, night)
        if dreamt is None or dreamt.state_id is None:
            return None
        kept = {str(d["state"]) for key in ("dreams", "boundaries")
                for d in cast("Sequence[Mapping[str, object]]", dreamt.receipt.get(key) or ())}
        if state_ref not in kept:
            raise ValueError("that state isn't one tonight's dream kept")
        state = core_state.load(self._store(user_id).state(state_ref), model_id=self.core.model_id)
        return MindReading(state, state_ref, self.core.predict(state), dreamt.projector)

    def dream(self, call: MindCall, evidence: Sequence[Evidence]) -> Dreamt:
        if (why := self._refusal(call.user_id)) is not None:
            logger.warning("MIND's core %s", why)
            done = Dreamt(receipt={"noop": True, "why": why})
        else:
            done = self._dream(call, evidence)
        self._tonight[(call.user_id, call.night)] = done
        return done

    def _dream(self, call: MindCall, evidence: Sequence[Evidence]) -> Dreamt:
        held = self._store(call.user_id)
        start = self._start(held, call.user_id, call.night)
        events = self._events(call, start.state, evidence)
        lived, weights = self.core.surprise_weights(start.state, events)
        lived = replace(lived, fake_latents=lived.fake_latents or (self.fake and bool(events)))
        dream = self.core.dream_states(lived, DREAM_STEPS, np.random.default_rng(_seed(self.seed, call.user_id, call.night)),
                                       rollouts=ROLLOUTS)
        seen = self.core.predict(lived)
        decision = self.core.decide(lived, dream.ignitions)
        picks = curriculum.pick(seen, dream.ignitions, weights, budget=self.budget) if self.curriculum else None
        pressure = self.core.sleep_pressure(lived)
        _guard(call)  # nothing is written for a person deleted mid-night
        opened = ({"ignite": "no workspace"} if self.workspace is None
                  else ignite(_Guarded(self.workspace(call.user_id), call.cancelled), dream.ignitions, call,
                              collisions=self.contradictions))
        slept = self.core.consolidate(lived)
        state_id = held.put_state(core_state.save(slept))
        cited = {c for slot in seen.slots for c in slot.cites}
        receipt = {
            "state": state_id, "from_version": None if start.unit is None else start.unit.version,
            "fresh": start.fresh, "model": self.core.model_id, "encoder": self.encoder.spec.spec_id(),
            "fake_latents": slept.fake_latents, "events": len(events),
            "events_before_tonight": len({e.event_id for e in events} & {i.id for i in evidence if i.day < call.night}),
            "weights": {k: round(v, 6) for k, v in weights.items()},
            "ignitions": [_ignition(i) for i in dream.ignitions], **opened, "decision": _decision(decision),
            "picks": None if picks is None else curriculum.record(picks), "dreams": self._imagined(held, dream, picks),
            "boundaries": [{**self._kept_state(held, b.dream), "steps": b.steps, "speak": b.speak, "slot": b.slot}
                           for b in dream.boundaries],
            "pressure": round(pressure, 6),
        }
        projector, fallback = self._paired(held, start.unit, call.serving)
        return Dreamt(state_id=state_id, state=slept, rehearse=() if picks is None else picks.evidence_ids,
                      weights=weights, view=bridge.CoreView(decision, seen, self._targets_of(call, cited)),
                      projector=projector, receipt={**receipt, **fallback})

    def _paired(self, held: store.MindStore, unit: store.Unit | None, serving: str | None) -> tuple[str | None, dict[str, Any]]:
        """The projector she is read through: her last unit's, when it reads this core's states and names
        the adapter that serves her; else the newest unit's that does, or the shared one (None), logged."""
        if unit is None or unit.model != self.core.model_id:
            return None, {}
        if unit.lora == serving:
            return unit.projector, {}
        found = next((u for u in reversed(held.units()) if u.lora == serving and u.model == self.core.model_id), None)
        logger.warning("MIND's last unit names another adapter than the one serving; its projector is not used")
        return (None if found is None else found.projector), {"projector_fallback": {
            "unit": unit.version, "unit_lora": unit.lora, "serving": serving, "from_unit": None if found is None else found.version}}

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
