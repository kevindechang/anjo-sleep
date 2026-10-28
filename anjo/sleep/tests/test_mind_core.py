# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's neural core: the fast-weight memory, state round trips, dreaming, deterministic
training, the counterfactual transplant and the planted dependency it must learn (toy lives with
deterministic latents; CPU only)."""
from __future__ import annotations

import dataclasses
import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anjo.sleep.mind.core import api, evaluate, net, toy, train  # noqa: E402
from anjo.sleep.mind.core import state as core_state  # noqa: E402
from anjo.sleep.mind.core.types import CoreConfig, EventLatent, LabelledStream, ThreadLabel  # noqa: E402

T0 = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
SMALL = CoreConfig(dim_in=toy.DIM, width=16, slots=4)
TOY = CoreConfig(dim_in=toy.DIM, width=64, slots=8)
NO_FAST = dataclasses.replace(TOY, fast=False)
EPOCHS, LIVES_PER_EPOCH, BATCH = 60, 128, 16  # fresh lives every epoch: ~35 s on one CPU thread
# Measured on seed 0 (seeds 1 and 2 in the report): the planted dependency separates at 0.94 AUC
# with the fast memory and at chance without it.
MIN_SEPARATION = 0.85
SEPARATION_MARGIN = 0.3  # over the same core with no fast weights
RECALL_MARGIN = 0.25  # recall at the locked threshold, over no fast weights
MAX_FALSE_PER_100 = 2.0
TRANSPLANT_TOLERANCE = 0.1  # mean risk shift of unrelated slots when only the history moves
MIN_RELEVANT_GAP = 0.25  # the booking's risk, after cancellations minus after deliveries
MIN_FLIPPED = 3  # of 16 pairs: speak about the booking after cancellations, not after deliveries


def _event(event_id: str, minutes: int, vec: np.ndarray) -> EventLatent:
    return EventLatent(event_id, "toy-1", T0 + timedelta(minutes=minutes), "turn", vec)


def _vec(seed: int) -> np.ndarray:
    out: np.ndarray = np.random.default_rng(seed).normal(size=toy.DIM).astype(np.float32)
    return out


def _observed(core: api.Core, n: int = 6) -> core_state.CoreState:
    state = core.init_state("toy-1")
    for i in range(n):
        state, _ = core.observe(state, _event(f"e{i}", 30 * i, _vec(i)))
    return state


# ── The fast-weight memory ────────────────────────────────────────────────────


def test_surprise_updates_fast_weights_and_decays() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    empty = core.init_state("toy-1")
    first, surprise = core.observe(empty, _event("e1", 0, _vec(1)))
    assert surprise > 0 and np.abs(first.memory).sum() > 0
    assert np.abs(empty.memory).sum() == 0  # observe returns a new state
    _, again = core.observe(first, _event("e2", 1, _vec(1)))
    assert again < surprise  # the same event surprises less once the memory holds it

    # The rule itself: a surprise-driven step lowers the error on what it saw; with nothing left
    # to learn and no momentum, the forget gate decays the memory by (1 - alpha).
    gen = torch.Generator().manual_seed(0)
    memory = torch.randn(1, 8, 8, generator=gen)
    k = torch.nn.functional.normalize(torch.randn(1, 8, generator=gen), dim=-1)
    v = torch.randn(1, 8, generator=gen)
    zero, quarter, fifth = torch.zeros(1, 8, 8), torch.full((1,), 0.25), torch.full((1,), 0.2)
    stepped, _, before = net.fast_update(memory, zero, k, v, quarter, torch.zeros(1), torch.zeros(1))
    _, _, after = net.fast_update(stepped, zero, k, v, quarter, torch.zeros(1), torch.zeros(1))
    assert float(after) < float(before)
    settled = torch.bmm(memory, k[:, :, None])[:, :, 0]
    decayed, _, nothing = net.fast_update(memory, zero, k, settled, quarter, torch.zeros(1), fifth)
    assert float(nothing) == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(decayed, 0.8 * memory)


# ── State ─────────────────────────────────────────────────────────────────────


def test_state_round_trips_and_versions() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    state = _observed(core)
    blob = core_state.save(state)
    back = core_state.load(blob, model_id=core.model_id)
    assert core_state.equal(back, state)
    nxt = _event("e9", 600, _vec(9))
    assert core.observe(back, nxt)[1] == core.observe(state, nxt)[1]
    assert core.predict(back) == core.predict(state)

    with pytest.raises(core_state.StateError, match="model"):
        core_state.load(blob, model_id="another-core")
    with np.load(io.BytesIO(blob), allow_pickle=False) as held:
        arrays = {name: held[name] for name in held.files}
    meta = json.loads(str(arrays["meta"]))
    arrays["meta"] = np.array(json.dumps({**meta, "version": core_state.STATE_VERSION + 1}))
    out = io.BytesIO()
    np.savez(out, **arrays)
    with pytest.raises(core_state.StateError, match="version"):
        core_state.load(out.getvalue(), model_id=core.model_id)


# ── Dreaming ──────────────────────────────────────────────────────────────────


def test_dream_never_cites_imagined_events() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    state = _observed(core)
    real = {f"e{i}" for i in range(6)}
    before = core_state.save(state)
    ignitions = core.dream(state, 8, np.random.default_rng(0), rollouts=4, ignite_at=0.0)
    assert ignitions, "every open slot ignites at a zero threshold"
    assert all(set(i.cited_event_ids) <= real for i in ignitions)
    assert core_state.save(state) == before  # dreaming leaves the real state as it was
    assert core.dream(state, 8, np.random.default_rng(0), rollouts=4, ignite_at=0.0) == ignitions
    assert core.dream(state, 8, np.random.default_rng(0), rollouts=4, ignite_at=1.01) == ()


# ── Training ──────────────────────────────────────────────────────────────────


def test_training_is_deterministic_on_cpu() -> None:
    lives = toy.lives(6, seed=11)
    first = train.train_shared(lives[:4], lives[4:], SMALL, seed=3, epochs=2)
    second = train.train_shared(lives[:4], lives[4:], SMALL, seed=3, epochs=2)
    assert first.core.model_id == second.core.model_id
    assert first.log == second.log and first.core.threshold == second.core.threshold
    other = train.train_shared(lives[:4], lives[4:], SMALL, seed=4, epochs=2)
    assert other.core.model_id != first.core.model_id


def _is_gap(label: ThreadLabel) -> bool:
    return label.id.endswith("-gap")


def _train(config: CoreConfig) -> api.Core:
    def fresh(epoch: int) -> tuple[LabelledStream, ...]:
        return toy.lives(LIVES_PER_EPOCH, seed=1000 + epoch)

    return train.train_shared(fresh, toy.lives(48, seed=2), config, seed=0, epochs=EPOCHS, batch_size=BATCH).core


@pytest.fixture(scope="module")
def trained() -> api.Core:
    return _train(TOY)


@pytest.fixture(scope="module")
def test_lives() -> tuple[LabelledStream, ...]:
    return toy.lives(64, seed=3)


def test_transplant_flips_only_the_relevant_slot(trained: api.Core) -> None:
    result = evaluate.transplant(trained, toy.pairs(16, seed=5), tolerance=TRANSPLANT_TOLERANCE)
    assert result.pairs == 16 and result.unrelated_found >= 8, result  # the rest share a slot with the history
    assert result.relevant_gap >= MIN_RELEVANT_GAP, result
    assert result.flipped >= MIN_FLIPPED, result
    assert result.mean_unrelated_shift <= TRANSPLANT_TOLERANCE, result


def test_core_learns_a_planted_dependency(trained: api.Core, test_lives: tuple[LabelledStream, ...]) -> None:
    blank = _train(NO_FAST)
    runs, flat = evaluate.traces(trained, test_lives), evaluate.traces(blank, test_lives)
    core, baseline = evaluate.separation(runs, test_lives, _is_gap), evaluate.separation(flat, test_lives, _is_gap)
    assert core is not None and baseline is not None
    assert core >= MIN_SEPARATION and core - baseline >= SEPARATION_MARGIN, (core, baseline)
    with_memory = evaluate.score(runs, test_lives, trained.threshold)
    without = evaluate.score(flat, test_lives, blank.threshold)
    assert with_memory.false_per_100 <= MAX_FALSE_PER_100, with_memory
    assert with_memory.recall - without.recall >= RECALL_MARGIN, (with_memory, without)
