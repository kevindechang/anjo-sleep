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
SEEDS = (0, 1, 2)
# Fresh lives every epoch, until validation stops improving (77 to 153 epochs on these seeds), within
# a fixed cap: up to ~1.5 min a core on one CPU thread.
MAX_EPOCHS, PATIENCE, LIVES_PER_EPOCH, BATCH = 200, 20, 128, 16
# Measured on seeds 0 / 1 / 2 (64 test lives): the planted dependency separates at 0.93 / 0.95 /
# 0.88 AUC with the fast memory, at chance without it (0.43 / 0.37 / 0.40); recall 0.61 / 0.61 /
# 0.63 against 0.02 / 0.24 / 0.00. Contradictions stay near chance either way, so nothing here
# claims them.
MIN_SEPARATION = 0.85
SEPARATION_MARGIN = 0.3  # over the same core with no fast weights
RECALL_MARGIN = 0.25  # recall at the locked threshold, over no fast weights
MIN_GAP_RECALL = 0.5  # the gap family alone: 24 of 32 on each seed
MAX_FALSE_PER_100 = 2.0
TRANSPLANT_TOLERANCE = 0.1  # mean risk shift of the unrelated open thread's slot: 0.047 / 0.066 / 0.036
MIN_RELEVANT_GAP = 0.25  # the booking's risk, after cancellations minus after deliveries: 0.56 / 0.76 / 0.31
# Of 16 pairs, speak about the booking after cancellations and not after deliveries: half of them.
# Measured 10 / 15 / 14; a core without its fast memory flips 1 / 0 / 0.
MIN_FLIPPED = 8
MIN_UNRELATED_FOUND = 8  # pairs whose open unrelated thread holds a slot of its own: 10 / 14 / 13


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


def test_training_stops_once_validation_stops_improving() -> None:
    lives = toy.lives(6, seed=11)
    flat = train.train_shared(lives[:4], lives[4:], SMALL, seed=3, epochs=10, patience=2, lr=0.0)
    assert len(flat.log) == 3 and flat.best_epoch == 0  # nothing moves, so two epochs without a gain end it
    capped = train.train_shared(lives[:4], lives[4:], SMALL, seed=3, epochs=2, patience=50)
    assert len(capped.log) == 2  # the cap holds whatever the patience


def test_deterministic_restores_the_callers_settings() -> None:
    before, warn_only, threads = (torch.are_deterministic_algorithms_enabled(),
                                  torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_num_threads())
    try:
        for setting in (True, False):
            torch.use_deterministic_algorithms(setting)
            with train.deterministic(0):
                assert torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 1
            assert torch.are_deterministic_algorithms_enabled() is setting and torch.get_num_threads() == threads
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn_only)


def test_the_dream_threshold_is_locked_on_validation() -> None:
    lives = toy.lives(6, seed=11)
    run = train.train_shared(lives[:4], lives[4:], SMALL, seed=3, epochs=2)
    assert run.core.thresholds.ignite == evaluate.lock_ignite(run.core, lives[4:], seed=3)
    dreamt = evaluate.dream_traces(run.core, lives[4:], seed=3)
    assert all(call.p_speak >= 0.0 and call.cites for trace in dreamt for call in trace.calls)
    assert evaluate.score(dreamt, lives[4:], run.core.thresholds.ignite).false_per_100 <= evaluate.MAX_FALSE_PER_100


def test_recall_is_reported_per_family() -> None:
    life = toy.life(0, 0)
    gap = ThreadLabel("g", "gap", (life.events[0].event_id,), (life.events[0].event_id,), life.events[0].t,
                      life.events[0].t + timedelta(days=3), failed=True)
    clash = dataclasses.replace(gap, id="c", family="contradiction", event_ids=(life.events[1].event_id,),
                                origin_ids=(life.events[1].event_id,))
    stream = dataclasses.replace(life, labels=(gap, clash))
    run = evaluate.LifeTrace(life.life_id, (evaluate.DayCall(life.events[0].t.date(), 0.9, gap.origin_ids),), ())
    result = evaluate.score([run], [stream], 0.5)
    assert result.by_family == (("contradiction", 1, 0), ("gap", 1, 1)) and result.recall == 0.5


def test_transplant_pairs_differ_only_in_the_history_and_hold_an_open_unrelated_thread() -> None:
    for pair in toy.pairs(6, seed=5):
        moved = {a.event_id for a, b in zip(pair.problem, pair.benign, strict=True) if not np.array_equal(a.vec, b.vec)}
        assert moved and moved == set(pair.moved) and moved <= set(pair.history)
        assert all(i.endswith("-end") for i in moved)  # only how the earlier bookings ended
        held = {e.event_id for e in pair.problem}
        assert pair.unrelated and set(pair.unrelated) <= held and not set(pair.unrelated) & set(pair.history)
        assert any(i.endswith("-constraint") for i in pair.unrelated)  # a conflicting constraint, not yet raised
        assert not any(i.endswith("-raised") for i in held)


def _is_gap(label: ThreadLabel) -> bool:
    return label.id.endswith("-gap")


def _train(config: CoreConfig, seed: int) -> api.Core:
    def fresh(epoch: int) -> tuple[LabelledStream, ...]:
        return toy.lives(LIVES_PER_EPOCH, seed=1000 + epoch)

    return train.train_shared(fresh, toy.lives(48, seed=2), config, seed=seed, epochs=MAX_EPOCHS, patience=PATIENCE,
                              batch_size=BATCH).core


@pytest.fixture(scope="module", params=SEEDS)
def seed(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture(scope="module")
def trained(seed: int) -> api.Core:
    return _train(TOY, seed)


@pytest.fixture(scope="module")
def blank(seed: int) -> api.Core:
    return _train(NO_FAST, seed)


@pytest.fixture(scope="module")
def test_lives() -> tuple[LabelledStream, ...]:
    return toy.lives(64, seed=3)


def test_transplant_flips_only_the_relevant_slot(trained: api.Core) -> None:
    result = evaluate.transplant(trained, toy.pairs(16, seed=5), tolerance=TRANSPLANT_TOLERANCE)
    assert result.pairs == 16 and result.unrelated_found >= MIN_UNRELATED_FOUND, result
    assert result.relevant_gap >= MIN_RELEVANT_GAP, result
    assert result.flipped >= MIN_FLIPPED, result
    assert result.mean_unrelated_shift <= TRANSPLANT_TOLERANCE, result


def test_core_learns_a_planted_dependency(
    trained: api.Core, blank: api.Core, test_lives: tuple[LabelledStream, ...],
) -> None:
    runs, flat = evaluate.traces(trained, test_lives), evaluate.traces(blank, test_lives)
    core, baseline = evaluate.separation(runs, test_lives, _is_gap), evaluate.separation(flat, test_lives, _is_gap)
    assert core is not None and baseline is not None
    assert core >= MIN_SEPARATION and core - baseline >= SEPARATION_MARGIN, (core, baseline)
    with_memory = evaluate.score(runs, test_lives, trained.threshold)
    without = evaluate.score(flat, test_lives, blank.threshold)
    assert with_memory.false_per_100 <= MAX_FALSE_PER_100, with_memory
    assert with_memory.recall - without.recall >= RECALL_MARGIN, (with_memory, without)
    gaps = dict((family, (caught, n)) for family, n, caught in with_memory.by_family)["gap"]
    assert gaps[0] >= MIN_GAP_RECALL * gaps[1], with_memory.by_family
