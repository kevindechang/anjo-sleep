"""MIND's core as one person meets it: ``observe``, ``predict``, ``dream``, ``decide`` (torch, CPU).

A ``Core`` is the shared network with its locked thresholds; every call takes the person's
``CoreState`` and returns a new one, so the state is a value and the core holds nothing per person.

- ``observe(state, event)`` runs one step: the fast weights learn from the event's surprise, the
  event binds into slots, and each slot it wrote into at least ``cite_min`` (and always the one it
  wrote most) cites its id. Returns the new state and the surprise.
- ``predict(state)``: every open slot's forecasts and the global speak and choice.
- ``dream(state, steps, rng)`` rolls the core forward on its own imagined events, ``rollouts`` at a
  time, on the CPU. Only slots open in the real state can ignite, and only with the real events they
  cite; a real slot reset inside a dream stops counting there. The real state is never touched.
  A slot ignites on its mean peak risk (``risk``) or missing step (``missing``); two slots collide
  when an imagined event writes into both while either's risk is high (``collision``: the mean peak
  of write x write x the larger risk).
- ``decide(state)``: the slot with the top choice share among the open slots that cite something;
  speak when the speak probability reaches the threshold locked on validation.

At night (the one system's loop, ``night``):

- ``consolidate(state, rank, residue)``: the fast memory moves into the person's slow memory
  (``SLOW_MEMORY`` in her adaptation), pruned to its ``rank`` strongest directions, and the fast
  level keeps only ``residue`` of itself, its momentum none (consolidate, then reset: Behrouz et
  al., "Language Models Need Sleep", 2026). The slow memory is read with the fast one ever after.
- ``rehearse(state, events)``: replays chosen events into the fast memory only; slots, cites and
  the clock stay as they were. Offline, it is what the night's rehearsal does to her memory.
- ``dream_states(state, steps, rng)``: the night's one dream (``dream`` keeps only its
  ignitions). The same rollouts give the ignitions, which the curriculum reads to pick what she
  rehearses, and each rollout's imagined state with the real events it rests on (``Dream.on``),
  which dream data speaks from (Glimmer reading it through the thought channel). The same
  rollouts also give her decision's boundaries (``Dream.boundaries``): in each rollout, the first
  imagined state whose decision differs from today's (speak or wait, or which thread), after the
  fewest imagined events, so the smallest change found that flips it. Both branches rest on the
  same real events; the imagined one is hypothetical.
"""
from __future__ import annotations

import hashlib
import math
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import torch
from torch import Tensor

from anjo.sleep.mind.core import curriculum
from anjo.sleep.mind.core.net import CoreNet, Heads, Tensors, param_count
from anjo.sleep.mind.core.state import Cite, CoreState
from anjo.sleep.mind.core.types import (
    CoreConfig,
    Decision,
    EventLatent,
    Ignition,
    IgnitionKind,
    Predictions,
    SlotPrediction,
    days_of,
    kind_index,
)

DREAM_NOISE: Final = 0.5  # the spread of an imagined event around the core's own expectation
MODEL_VERSION: Final = 1
PRESSURE_RISK: Final = 0.5  # an open thread adds to sleep pressure by its risk above this line
# Sleep pressure's weights until a model locks fitted ones: an uncalibrated stand-in where 1.0 is an
# ordinary day's surprise (1.8 to 2.3 on the toy cores). ``sleep_eval.fit_pressure`` found nothing
# to calibrate against on toy lives: skipping a night's consolidation cost no next-day prediction
# (R^2 <= 0 on three seeds), since consolidation there loses nothing and the fast memory never fills.
DEFAULT_PRESSURE: Final = (0.5, 0.0, 0.0)
SLOW_MEMORY: Final = "memory"  # the person's consolidated memory, kept in her adaptation (``CoreState.delta``)


@dataclass(frozen=True)
class DreamState:
    """One imagined continuation of a person's state: the state at the dream's end (its cites are
    the real events of the threads still alive in it, nothing imagined), the threads it is about,
    and the real events they rest on."""

    state: CoreState
    threads: tuple[int, ...]
    rests_on: tuple[str, ...]


@dataclass(frozen=True)
class Boundary:
    """Where one imagined continuation first flips her decision: the imagined state there, after
    ``steps`` imagined events, and the decision it flips to (``speak``, and the thread ``slot``)."""

    dream: DreamState
    steps: int
    speak: bool
    slot: int | None


@dataclass(frozen=True)
class Dream:
    """The night's one dream: its ignitions (the curriculum and the workspace read them), each
    rollout's imagined state (dream data reads them) and where her decision flips (``boundaries``),
    from the same rollouts."""

    ignitions: tuple[Ignition, ...]
    states: tuple[DreamState, ...]
    boundaries: tuple[Boundary, ...] = ()

    def on(self, threads: Sequence[int]) -> tuple[DreamState, ...]:
        """Each imagined state about ``threads``, those still alive in it; one holding none is left out."""
        out = []
        for dream in self.states:
            kept = tuple(k for k in dict.fromkeys(threads) if k in dream.threads)
            if kept:
                rests = tuple(dict.fromkeys(e for k in kept for e in dream.state.cited(k)))
                out.append(DreamState(dream.state, kept, rests))
        return tuple(out)


@dataclass(frozen=True)
class Thresholds:
    speak: float = 0.5  # locked on validation (``evaluate.lock_threshold``)
    ignite: float = 0.5
    pressure: tuple[float, float, float] = DEFAULT_PRESSURE  # ``sleep_pressure``'s weights (``sleep_eval.fit_pressure``)


class PressureFeatures(NamedTuple):
    """What sleep pressure reads: the surprise since the last consolidation, the fast memory's load
    (its Frobenius norm) and the open threads' risk above ``PRESSURE_RISK``, summed."""

    surprise: float
    load: float
    risk: float


def model_id(net: CoreNet) -> str:
    """A hash of the config and every weight: a state names the core it was built under."""
    digest = hashlib.sha256(json.dumps(asdict(net.cfg), sort_keys=True).encode())
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


class _Rolled(NamedTuple):
    peaks: tuple[np.ndarray, np.ndarray, np.ndarray]  # mean peak risk, missing step, collision
    end: Tensors  # each rollout's state at the dream's end
    alive: np.ndarray  # (rollouts, K): the real slots never reset in that rollout
    flips: tuple[tuple[int, Tensors, np.ndarray, bool, int | None] | None, ...]  # per rollout: step, state, alive, decision


class Core:
    def __init__(self, net: CoreNet, thresholds: Thresholds = Thresholds()) -> None:
        self.net = net.eval()
        self.config: CoreConfig = net.cfg
        self.thresholds = thresholds
        self.model_id = model_id(net)

    @classmethod
    def fresh(cls, config: CoreConfig, seed: int) -> Core:
        torch.manual_seed(seed)
        return cls(CoreNet(config))

    @property
    def threshold(self) -> float:
        return self.thresholds.speak

    def with_thresholds(self, thresholds: Thresholds) -> Core:
        return Core(self.net, thresholds)

    @property
    def params(self) -> int:
        return param_count(self.net)

    # ── state ────────────────────────────────────────────────────────────────

    def init_state(self, person: str) -> CoreState:
        return self._state(person, self.net.initial(1), now=None, observed=0,
                           cites=((),) * self.config.slots, delta={})

    def _state(
        self, person: str, st: Tensors, *, now: float | None, observed: int, cites: tuple[tuple[Cite, ...], ...],
        delta: Mapping[str, np.ndarray], pending: float = 0.0, through: tuple[float, str] | None = None,
        fake_latents: bool = False,
    ) -> CoreState:
        rows = [part[0].detach().numpy() for part in st]
        return CoreState(
            person=person, model_id=self.model_id, slots=rows[0], usage=rows[1], opened=rows[2], memory=rows[3],
            momentum=rows[4], glob=rows[5], now=now, observed=observed, cites=cites, delta=delta, pending=pending,
            through=through, fake_latents=fake_latents,
        )

    def tensors(self, state: CoreState, copies: int = 1) -> Tensors:
        def t(a: np.ndarray) -> Tensor:
            return torch.from_numpy(np.array(a, dtype=np.float32))[None].repeat(copies, *([1] * a.ndim))

        now = torch.full((copies,), 0.0 if state.now is None else state.now)
        return Tensors(t(state.slots), t(state.usage), t(state.opened), t(state.memory), t(state.momentum),
                       t(state.glob), now)

    def _delta(self, state: CoreState) -> dict[str, Tensor]:
        return {name: torch.from_numpy(np.array(v)) for name, v in state.delta.items() if name != SLOW_MEMORY}

    def _slow(self, state: CoreState, copies: int = 1) -> Tensor | None:
        held = state.delta.get(SLOW_MEMORY)
        return None if held is None else torch.from_numpy(np.array(held, dtype=np.float32))[None].repeat(copies, 1, 1)

    def _check(self, state: CoreState) -> None:
        if state.model_id != self.model_id:
            raise ValueError("this state belongs to another core model")

    # ── observe ──────────────────────────────────────────────────────────────

    def observe(self, state: CoreState, event: EventLatent) -> tuple[CoreState, float]:
        self._check(state)
        if event.vec.shape != (self.config.dim_in,):
            raise ValueError(f"event {event.event_id} has {event.vec.size} dims; the core reads {self.config.dim_in}")
        now = days_of(event.t)
        gap = 0.0 if state.now is None else max(now - state.now, 0.0)
        with torch.no_grad():
            st = self.tensors(state) if self.config.memory else self.net.initial(1, torch.full((1,), now))
            e = self.net.encode(torch.from_numpy(np.array(event.vec))[None], torch.tensor([kind_index(event.kind)]),
                                torch.tensor([gap]))
            slow = self._slow(state) if self.config.memory else None
            new, written = self.net.step(st, e, torch.tensor([gap]), torch.tensor([now]), slow)
        prior = state.cites if self.config.memory else ((),) * self.config.slots
        cites = self._cite(prior, written.write[0].numpy(), written.fresh[0].numpy(), event.event_id)
        surprise = float(written.surprise[0])
        through = (now, event.event_id) if state.after(now, event.event_id) else state.through
        out = self._state(state.person, new, now=now, observed=state.observed + 1, cites=cites, delta=state.delta,
                          pending=state.pending + surprise, through=through, fake_latents=state.fake_latents)
        return out, surprise

    def surprise_weights(self, state: CoreState, events: Sequence[EventLatent]) -> tuple[CoreState, dict[str, float]]:
        """The Sleep link: observe ``events`` in order, and give each a training weight for the
        night's LoRA data from its surprise, bounded and averaging 1 (``curriculum.bounded_weights``),
        so what surprised her is what the weights learn first, and no one event takes the night."""
        surprises: dict[str, float] = {}
        for event in events:
            state, surprise = self.observe(state, event)
            surprises[event.event_id] = surprise
        return state, curriculum.bounded_weights(surprises)

    def _cite(
        self, cites: tuple[tuple[Cite, ...], ...], write: np.ndarray, fresh: np.ndarray, event_id: str,
    ) -> tuple[tuple[Cite, ...], ...]:
        cfg = self.config
        top = int(np.argmax(write))
        rows = []
        for slot, row in enumerate(cites):
            kept = [] if fresh[slot] > 0.5 else [c for _, c in sorted(enumerate(row), key=lambda p: (-p[1][1], -p[0]))]
            cite = (event_id, round(float(write[slot]), 6))
            if slot == top:  # the event's own thread always cites it, however full the slot
                kept = [*kept[: cfg.cite_keep - 1], cite]
            elif cite[1] >= cfg.cite_min:
                kept = [*kept, cite]
            rows.append(tuple(sorted(kept, key=lambda c: -c[1])[: cfg.cite_keep]))
        return tuple(rows)

    # ── predict and decide ───────────────────────────────────────────────────

    def predict(self, state: CoreState) -> Predictions:
        self._check(state)
        with torch.no_grad():
            heads = self.net.heads(self.tensors(state), self._delta(state))
        choice = torch.softmax(heads.choice_logits[0], dim=-1).numpy()
        hazard = torch.exp(heads.log_hazard[0]).numpy()
        missing = torch.sigmoid(heads.missing_logit[0]).numpy()
        slots = tuple(
            SlotPrediction(
                slot=k, usage=float(state.usage[k]), risk=float(heads.risk[0, k]),
                hazard=tuple(float(h) for h in hazard[k]), missing=float(missing[k]),
                self_error=float(heads.self_error[0, k]), choice=float(choice[k]), cites=state.cited(k),
            )
            for k in range(self.config.slots) if state.usage[k] >= self.config.free_below
        )
        return Predictions(speak=float(torch.sigmoid(heads.speak_logit[0])), none=float(choice[-1]), slots=slots)

    def decide(self, state: CoreState, ops: Sequence[Ignition] = ()) -> Decision:
        seen = self.predict(state)
        cited = [s for s in seen.slots if s.cites]
        slot = max(cited, key=lambda s: (s.choice, -s.slot)).slot if cited else None
        speak = slot is not None and seen.speak >= self.thresholds.speak
        return Decision(speak=speak, slot=slot, p_speak=seen.speak, ops=tuple(ops))

    # ── dream ────────────────────────────────────────────────────────────────

    def dream(
        self, state: CoreState, steps: int, rng: np.random.Generator, *, rollouts: int = 8,
        ignite_at: float | None = None,
    ) -> tuple[Ignition, ...]:
        return self.dream_states(state, steps, rng, rollouts=rollouts, ignite_at=ignite_at).ignitions

    def dream_states(
        self, state: CoreState, steps: int, rng: np.random.Generator, *, rollouts: int = 8,
        ignite_at: float | None = None,
    ) -> Dream:
        self._check(state)
        line = self.thresholds.ignite if ignite_at is None else ignite_at
        real = np.array([state.usage[k] >= self.config.free_below and bool(state.cites[k])
                         for k in range(self.config.slots)])
        if steps < 1 or not real.any():
            return Dream((), ())
        rolled = self._roll(state, steps, rng, rollouts, real, self.decide(state))
        peak_risk, peak_missing, peak_collide = rolled.peaks
        found: list[Ignition] = []
        for k in np.flatnonzero(real):
            singles: tuple[tuple[IgnitionKind, float], ...] = (("risk", peak_risk[k]), ("missing", peak_missing[k]))
            for kind, score in singles:
                if score >= line:
                    found.append(self._ignition(state, int(k), None, kind, float(score)))
            for j in np.flatnonzero(real):
                if j > k and peak_collide[k, j] >= line:
                    found.append(self._ignition(state, int(k), int(j), "collision", float(peak_collide[k, j])))
        ignitions = tuple(sorted(found, key=lambda i: (-i.score, i.slot, i.other or -1, i.kind)))
        boundaries = tuple(
            Boundary(self._imagined(state, st, alive, r), step, speak, slot)
            for r, flip in enumerate(rolled.flips) if flip is not None
            for step, st, alive, speak, slot in (flip,)
        )
        return Dream(ignitions, tuple(self._imagined(state, rolled.end, rolled.alive[r], r) for r in range(rollouts)),
                     boundaries)

    def _imagined(self, state: CoreState, st: Tensors, alive: np.ndarray, r: int) -> DreamState:
        threads = tuple(int(k) for k in np.flatnonzero(alive > 0.5))
        cites = tuple(state.cites[k] if k in threads else () for k in range(self.config.slots))
        imagined = replace(state, **{name: getattr(st, name)[r].numpy() for name in
                                     ("slots", "usage", "opened", "memory", "momentum", "glob")},
                           now=float(st.now[r]), cites=cites)
        return DreamState(imagined, threads, tuple(dict.fromkeys(e for k in threads for e in state.cited(k))))

    def _ignition(self, state: CoreState, slot: int, other: int | None, kind: IgnitionKind, score: float) -> Ignition:
        cites = state.cited(slot) + (state.cited(other) if other is not None else ())
        return Ignition(slot=slot, other=other, kind=kind, score=round(score, 6),
                        cited_event_ids=tuple(dict.fromkeys(cites)))

    def _decisions(self, heads: Heads, st: Tensors, alive: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Each rollout's decision as ``decide`` makes it: the top choice among the live real threads,
        spoken when the speak probability reaches the line (slot -1: no thread to raise)."""
        k = self.config.slots
        open_ = (alive > 0.5) & (st.usage >= self.config.free_below)
        share = torch.softmax(heads.choice_logits, dim=-1)[:, :k].masked_fill(~open_, -1.0)
        slot = torch.where(open_.any(dim=1), share.argmax(dim=1), torch.full_like(share[:, 0], -1, dtype=torch.long))
        speak = (slot >= 0) & (torch.sigmoid(heads.speak_logit) >= self.thresholds.speak)
        return speak.numpy(), slot.numpy()

    def _roll(
        self, state: CoreState, steps: int, rng: np.random.Generator, rollouts: int, real: np.ndarray, today: Decision,
    ) -> _Rolled:
        cfg, net = self.config, self.net
        flips: list[tuple[int, Tensors, np.ndarray, bool, int | None] | None] = [None] * rollouts
        was = (today.speak, today.slot if today.speak else None)
        k = cfg.slots
        alive = torch.from_numpy(real.astype(np.float32))[None].repeat(rollouts, 1)
        peak_risk, peak_missing = torch.zeros(rollouts, k), torch.zeros(rollouts, k)
        peak_collide = torch.zeros(rollouts, k, k)
        delta, slow = self._delta(state), self._slow(state, rollouts)
        gap = torch.full((rollouts,), cfg.dream_gap_days)
        with torch.no_grad():
            st = self.tensors(state, rollouts)
            for step in range(steps):
                noise = torch.from_numpy(rng.normal(size=(rollouts, cfg.width)).astype(np.float32))
                imagined = net.heads(st, delta).imagined + DREAM_NOISE * noise
                st, written = net.step(st, imagined, gap, st.now + gap, slow)
                alive = alive * (1.0 - written.fresh)
                heads = net.heads(st, delta)
                for r, (speak, slot) in enumerate(zip(*self._decisions(heads, st, alive), strict=True)):
                    picked = None if slot < 0 else int(slot)
                    if flips[r] is None and (bool(speak), picked if speak else None) != was:
                        flips[r] = (step + 1, st, alive.numpy().copy()[r], bool(speak), picked)
                peak_risk = torch.maximum(peak_risk, heads.risk * alive)
                peak_missing = torch.maximum(peak_missing, torch.sigmoid(heads.missing_logit) * alive)
                both = written.write[:, :, None] * written.write[:, None, :] * alive[:, :, None] * alive[:, None, :]
                joint = torch.maximum(heads.risk[:, :, None], heads.risk[:, None, :])
                peak_collide = torch.maximum(peak_collide, both * joint)
        peaks = (peak_risk.mean(0).numpy(), peak_missing.mean(0).numpy(), peak_collide.mean(0).numpy())
        return _Rolled(peaks, st, alive.numpy(), tuple(flips))

    # ── at night ─────────────────────────────────────────────────────────────

    def consolidate(self, state: CoreState, *, rank: int | None = None, residue: float = 0.0) -> CoreState:
        """The fast memory into the slow, pruned to its ``rank`` strongest directions (all when
        ``None``); the fast memory keeps ``residue`` of itself and no momentum."""
        self._check(state)
        if not 0.0 <= residue <= 1.0 or (rank is not None and rank < 1):
            raise ValueError("a residue is in [0, 1] and a rank is positive")
        fast = np.array(state.memory, dtype=np.float64)
        moved = fast
        if rank is not None:
            u, sv, vt = np.linalg.svd(fast)
            moved = (u[:, :rank] * sv[:rank]) @ vt[:rank]
        slow = np.array(state.delta.get(SLOW_MEMORY, np.zeros_like(fast)), dtype=np.float64) + moved
        return replace(state, memory=(residue * fast).astype(np.float32), momentum=np.zeros_like(state.momentum),
                       delta={**state.delta, SLOW_MEMORY: slow.astype(np.float32)}, pending=0.0)

    def pressure_features(self, state: CoreState) -> PressureFeatures:
        risky = math.fsum(max(0.0, s.risk - PRESSURE_RISK) for s in self.predict(state).slots if s.cites)
        return PressureFeatures(state.pending, float(np.linalg.norm(state.memory)), risky)

    def sleep_pressure(self, state: CoreState) -> float:
        """How much unconsolidated life she carries: 1.0 means consolidate now (what skipping an
        ordinary day's consolidation costs her next-day prediction), 0.0 right after it."""
        weights = self.thresholds.pressure
        return max(0.0, math.fsum(w * f for w, f in zip(weights, self.pressure_features(state), strict=True)))

    def rehearse(self, state: CoreState, events: Sequence[EventLatent]) -> CoreState:
        """``events`` replayed, in order, into the fast memory alone."""
        self._check(state)
        if not events:
            return state
        with torch.no_grad():
            st, slow, gap = self.tensors(state), self._slow(state), torch.zeros(1)
            for event in events:
                e = self.net.encode(torch.from_numpy(np.array(event.vec))[None], torch.tensor([kind_index(event.kind)]), gap)
                memory, momentum, _, _ = self.net._remember(st, e, gap, slow)
                st = st._replace(memory=memory, momentum=momentum)
        return replace(state, memory=st.memory[0].numpy(), momentum=st.momentum[0].numpy())

    # ── the shared weights on disk ───────────────────────────────────────────

    def save(self, path: Path) -> None:
        """The weights and thresholds as ``.npz`` (no pickle): run data, never in git."""
        meta = {"version": MODEL_VERSION, "config": asdict(self.config), "model_id": self.model_id,
                "thresholds": asdict(self.thresholds)}
        arrays = {f"w__{name}": t.detach().numpy() for name, t in self.net.state_dict().items()}
        with path.open("wb") as handle:
            np.savez(handle, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)

    @classmethod
    def load(cls, path: Path) -> Core:
        with np.load(path, allow_pickle=False) as held:
            meta = json.loads(str(held["meta"]))
            weights = {name[3:]: torch.from_numpy(np.array(held[name])) for name in held.files if name.startswith("w__")}
        if meta.get("version") != MODEL_VERSION:
            raise ValueError(f"core model version {meta.get('version')!r}; this build reads {MODEL_VERSION}")
        net = CoreNet(CoreConfig(**meta["config"]))
        net.load_state_dict(weights)
        held_thresholds = dict(meta["thresholds"])
        if "pressure" in held_thresholds:
            held_thresholds["pressure"] = tuple(float(w) for w in held_thresholds["pressure"])
        core = cls(net, Thresholds(**held_thresholds))
        if core.model_id != meta["model_id"]:
            raise ValueError("the saved weights don't hash to the model id they were saved under")
        return core


def adapted(state: CoreState, delta: Mapping[str, np.ndarray]) -> CoreState:
    """The state with the person's head adaptation set (``train.adapt`` computes it); her slow
    memory, from her nights, stays."""
    slow = {SLOW_MEMORY: state.delta[SLOW_MEMORY]} if SLOW_MEMORY in state.delta else {}
    return replace(state, delta={**dict(delta), **slow})
