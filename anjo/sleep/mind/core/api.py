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
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor

from anjo.sleep.mind.core.net import CoreNet, Tensors, param_count
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


@dataclass(frozen=True)
class Thresholds:
    speak: float = 0.5  # locked on validation (``evaluate.lock_threshold``)
    ignite: float = 0.5


def model_id(net: CoreNet) -> str:
    """A hash of the config and every weight: a state names the core it was built under."""
    digest = hashlib.sha256(json.dumps(asdict(net.cfg), sort_keys=True).encode())
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


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
        delta: Mapping[str, np.ndarray],
    ) -> CoreState:
        rows = [part[0].detach().numpy() for part in st]
        return CoreState(
            person=person, model_id=self.model_id, slots=rows[0], usage=rows[1], opened=rows[2], memory=rows[3],
            momentum=rows[4], glob=rows[5], now=now, observed=observed, cites=cites, delta=delta,
        )

    def tensors(self, state: CoreState, copies: int = 1) -> Tensors:
        def t(a: np.ndarray) -> Tensor:
            return torch.from_numpy(np.array(a, dtype=np.float32))[None].repeat(copies, *([1] * a.ndim))

        now = torch.full((copies,), 0.0 if state.now is None else state.now)
        return Tensors(t(state.slots), t(state.usage), t(state.opened), t(state.memory), t(state.momentum),
                       t(state.glob), now)

    def _delta(self, state: CoreState) -> dict[str, Tensor]:
        return {name: torch.from_numpy(np.array(v)) for name, v in state.delta.items()}

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
            new, written = self.net.step(st, e, torch.tensor([gap]), torch.tensor([now]))
        prior = state.cites if self.config.memory else ((),) * self.config.slots
        cites = self._cite(prior, written.write[0].numpy(), written.fresh[0].numpy(), event.event_id)
        out = self._state(state.person, new, now=now, observed=state.observed + 1, cites=cites, delta=state.delta)
        return out, float(written.surprise[0])

    def surprise_weights(self, state: CoreState, events: Sequence[EventLatent]) -> tuple[CoreState, dict[str, float]]:
        """The Sleep link: observe ``events`` in order, and give each a training weight for the
        night's LoRA data, its surprise over their mean (all 1.0 when nothing surprised more), so
        what surprised her is what the weights learn first."""
        surprises: dict[str, float] = {}
        for event in events:
            state, surprise = self.observe(state, event)
            surprises[event.event_id] = surprise
        mean = sum(surprises.values()) / max(len(surprises), 1)
        return state, {k: (v / mean if mean > 0 else 1.0) for k, v in surprises.items()}

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
        self._check(state)
        line = self.thresholds.ignite if ignite_at is None else ignite_at
        real = np.array([state.usage[k] >= self.config.free_below and bool(state.cites[k])
                         for k in range(self.config.slots)])
        if steps < 1 or not real.any():
            return ()
        peak_risk, peak_missing, peak_collide = self._roll(state, steps, rng, rollouts, real)
        found: list[Ignition] = []
        for k in np.flatnonzero(real):
            singles: tuple[tuple[IgnitionKind, float], ...] = (("risk", peak_risk[k]), ("missing", peak_missing[k]))
            for kind, score in singles:
                if score >= line:
                    found.append(self._ignition(state, int(k), None, kind, float(score)))
            for j in np.flatnonzero(real):
                if j > k and peak_collide[k, j] >= line:
                    found.append(self._ignition(state, int(k), int(j), "collision", float(peak_collide[k, j])))
        return tuple(sorted(found, key=lambda i: (-i.score, i.slot, i.other or -1, i.kind)))

    def _ignition(self, state: CoreState, slot: int, other: int | None, kind: IgnitionKind, score: float) -> Ignition:
        cites = state.cited(slot) + (state.cited(other) if other is not None else ())
        return Ignition(slot=slot, other=other, kind=kind, score=round(score, 6),
                        cited_event_ids=tuple(dict.fromkeys(cites)))

    def _roll(
        self, state: CoreState, steps: int, rng: np.random.Generator, rollouts: int, real: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg, net = self.config, self.net
        k = cfg.slots
        alive = torch.from_numpy(real.astype(np.float32))[None].repeat(rollouts, 1)
        peak_risk, peak_missing = torch.zeros(rollouts, k), torch.zeros(rollouts, k)
        peak_collide = torch.zeros(rollouts, k, k)
        delta = self._delta(state)
        gap = torch.full((rollouts,), cfg.dream_gap_days)
        with torch.no_grad():
            st = self.tensors(state, rollouts)
            for _ in range(steps):
                noise = torch.from_numpy(rng.normal(size=(rollouts, cfg.width)).astype(np.float32))
                imagined = net.heads(st, delta).imagined + DREAM_NOISE * noise
                st, written = net.step(st, imagined, gap, st.now + gap)
                alive = alive * (1.0 - written.fresh)
                heads = net.heads(st, delta)
                peak_risk = torch.maximum(peak_risk, heads.risk * alive)
                peak_missing = torch.maximum(peak_missing, torch.sigmoid(heads.missing_logit) * alive)
                both = written.write[:, :, None] * written.write[:, None, :] * alive[:, :, None] * alive[:, None, :]
                joint = torch.maximum(heads.risk[:, :, None], heads.risk[:, None, :])
                peak_collide = torch.maximum(peak_collide, both * joint)
        return peak_risk.mean(0).numpy(), peak_missing.mean(0).numpy(), peak_collide.mean(0).numpy()

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
        core = cls(net, Thresholds(**meta["thresholds"]))
        if core.model_id != meta["model_id"]:
            raise ValueError("the saved weights don't hash to the model id they were saved under")
        return core


def adapted(state: CoreState, delta: Mapping[str, np.ndarray]) -> CoreState:
    """The state with the person's head adaptation set (``train.adapt`` computes it)."""
    return replace(state, delta=dict(delta))
