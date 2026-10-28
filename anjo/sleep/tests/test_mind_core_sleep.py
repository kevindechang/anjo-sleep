# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's core at night: consolidation into the slow memory, rehearsal of chosen events, and the
perturbed states dreams speak from, and where they flip her decision (toy lives, CPU only)."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anjo.sleep.mind.core import api, toy  # noqa: E402
from anjo.sleep.mind.core.event import EventLatent  # noqa: E402
from anjo.sleep.mind.core import state as core_state  # noqa: E402
from anjo.sleep.mind.core.types import CoreConfig  # noqa: E402

SMALL = CoreConfig(dim_in=toy.DIM, width=16, slots=4)


def _lived(core: api.Core, n: int = 12) -> tuple[core_state.CoreState, list[EventLatent]]:
    life = toy.lives(1, seed=4)[0]
    state = core.init_state(life.life_id)
    for event in life.events[:n]:
        state, _ = core.observe(state, event)
    return state, list(life.events[n:n + 4])


def test_consolidation_moves_the_fast_memory_into_the_slow_and_keeps_what_it_recalls() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    state, upcoming = _lived(core)
    assert np.abs(state.memory).sum() > 0 and api.SLOW_MEMORY not in state.delta
    slept = core.consolidate(state, rank=None, residue=0.0)
    assert np.abs(slept.memory).sum() == 0 and np.abs(slept.momentum).sum() == 0
    np.testing.assert_allclose(slept.delta[api.SLOW_MEMORY], state.memory, atol=1e-6)
    for event in upcoming:  # the next event surprises her exactly as much: nothing recalled is lost
        assert core.observe(slept, event)[1] == pytest.approx(core.observe(state, event)[1], rel=1e-4, abs=1e-6)
    again = core.consolidate(core.observe(slept, upcoming[0])[0], rank=None, residue=0.0)
    assert np.abs(again.delta[api.SLOW_MEMORY] - slept.delta[api.SLOW_MEMORY]).sum() > 0  # nights add up


def test_pruned_consolidation_keeps_the_strongest_directions_and_a_residue() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    state, _ = _lived(core)
    slept = core.consolidate(state, rank=1, residue=0.1)
    kept = slept.delta[api.SLOW_MEMORY]
    assert np.linalg.matrix_rank(kept, tol=1e-5) == 1
    top = np.linalg.svd(state.memory, compute_uv=False)[0]
    assert np.linalg.svd(kept, compute_uv=False)[0] == pytest.approx(top, rel=1e-4)
    np.testing.assert_allclose(slept.memory, 0.1 * state.memory, atol=1e-6)
    assert core_state.equal(core_state.load(core_state.save(slept), model_id=core.model_id), slept)


def test_a_slow_memory_survives_the_persons_head_adaptation() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    state, _ = _lived(core)
    slept = core.consolidate(state, rank=None, residue=0.0)
    adapted = api.adapted(slept, {"speak.bias": np.ones(1, dtype=np.float32)})
    assert set(adapted.delta) == {api.SLOW_MEMORY, "speak.bias"}


def test_rehearsal_replays_into_the_memory_only() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    state, upcoming = _lived(core)
    chosen = upcoming[:2]
    before = [core.observe(state, e)[1] for e in chosen]
    rehearsed = core.rehearse(state, chosen)
    assert rehearsed.cites == state.cites and rehearsed.observed == state.observed and rehearsed.now == state.now
    for name in ("slots", "usage", "opened", "glob"):
        np.testing.assert_array_equal(getattr(rehearsed, name), getattr(state, name))
    after = [core.observe(rehearsed, e)[1] for e in chosen]
    assert sum(after) < sum(before)  # what she rehearsed surprises her less
    assert core.rehearse(state, ()) is state


def test_one_dream_gives_the_ignitions_and_the_imagined_states_they_rest_on() -> None:
    core = api.Core.fresh(SMALL, seed=0).with_thresholds(api.Thresholds(speak=0.5, ignite=0.0))
    state, _ = _lived(core)
    dream = core.dream_states(state, 3, np.random.default_rng(5), rollouts=4)
    again = core.dream_states(state, 3, np.random.default_rng(5), rollouts=4)
    assert dream.ignitions and dream.ignitions == core.dream(state, 3, np.random.default_rng(5), rollouts=4)
    assert len(dream.states) == 4
    for imagined, twin in zip(dream.states, again.states, strict=True):
        assert core_state.equal(imagined.state, twin.state)
        assert not np.array_equal(imagined.state.slots, state.slots)  # she imagined on from today
        assert imagined.state.known <= state.known  # every cite is still a real event
        assert set(imagined.rests_on) == imagined.state.known
        assert all(imagined.state.cites[k] == state.cites[k] for k in imagined.threads)
    ignited = dream.ignitions[0].slot
    on = dream.on((ignited,))
    assert on and all(d.threads == (ignited,) and d.rests_on == state.cited(ignited) for d in on)
    assert dream.on((SMALL.slots + 1,)) == ()


def test_a_dream_finds_where_an_imagined_turn_first_flips_her_decision() -> None:
    lived, _ = _lived(api.Core.fresh(SMALL, seed=0))
    edge = api.Core.fresh(SMALL, seed=0).predict(lived).speak  # her speak line set at today's value: fragile
    core = api.Core.fresh(SMALL, seed=0).with_thresholds(api.Thresholds(speak=edge, ignite=0.0))
    real = core.decide(lived)
    dream = core.dream_states(lived, 4, np.random.default_rng(2), rollouts=8)
    assert dream.ignitions == core.dream(lived, 4, np.random.default_rng(2), rollouts=8)  # the same rollouts
    assert dream.boundaries and len(dream.boundaries) <= 8
    for edge_of in dream.boundaries:
        imagined = edge_of.dream.state
        flipped = core.decide(imagined)
        assert 1 <= edge_of.steps <= 4 and (flipped.speak, flipped.slot) == (edge_of.speak, edge_of.slot)
        assert (edge_of.speak, edge_of.slot if edge_of.speak else None) != (real.speak, real.slot if real.speak else None)
        assert imagined.known <= lived.known and set(edge_of.dream.rests_on) == imagined.known  # real events only
    again = core.dream_states(lived, 4, np.random.default_rng(2), rollouts=8).boundaries
    assert [(b.steps, b.speak, b.slot) for b in again] == [(b.steps, b.speak, b.slot) for b in dream.boundaries]


def test_a_dream_with_nothing_real_imagines_nothing() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    blank = core.init_state("toy-0")
    assert core.dream_states(blank, 3, np.random.default_rng(0)) == api.Dream((), ())


def test_chained_lives_run_many_nights_with_their_labels_in_step() -> None:
    long = toy.chained(2, seed=9, links=3)
    assert len(long) == 2 and len(long[0].days) == 3 * toy.DAYS
    first = toy.life(9, 0)
    held = {e.event_id: e for e in long[0].events}
    shift = 2 * toy.DAYS
    third = toy.life(9, 2)
    for event in third.events:
        assert (held[event.event_id].t - event.t).days == shift
    for label in third.labels:
        moved = next(lab for lab in long[0].labels if lab.id == label.id)
        assert (moved.outcome_at - label.outcome_at).days == shift
    assert {e.event_id for e in first.events} <= set(held)


def test_a_night_hook_runs_between_days() -> None:
    from anjo.sleep.mind.core import evaluate

    core = api.Core.fresh(SMALL, seed=0)
    life = toy.lives(1, seed=4)[0]
    seen: list[int] = []

    def night(state: core_state.CoreState, weights: Mapping[str, float]) -> core_state.CoreState:
        seen.append(state.observed)
        assert all(w > 0 for w in weights.values())
        return core.consolidate(state)

    run = evaluate.trace(core, life, nightly=night)
    assert len(seen) == len(life.days) - 1 and seen == sorted(seen)
    assert len(run.calls) == len(life.days)


def test_sleep_pressure_builds_through_the_day_and_falls_after_consolidation() -> None:
    core = api.Core.fresh(SMALL, seed=0)
    blank = core.init_state("toy-1")
    assert core.sleep_pressure(blank) == 0.0 and blank.pending == 0.0
    life = toy.lives(1, seed=4)[0]
    state, rising = blank, []
    for event in life.events[:10]:
        state, surprise = core.observe(state, event)
        rising.append(core.sleep_pressure(state))
    assert state.pending == pytest.approx(sum(core.observe(s, e)[1] for s, e in _steps(core, blank, life.events[:10])))
    assert rising[-1] > rising[0] > 0.0
    slept = core.consolidate(state)
    assert slept.pending == 0.0 and core.sleep_pressure(slept) < core.sleep_pressure(state)
    assert core_state.equal(core_state.load(core_state.save(state), model_id=core.model_id), state)
    features = core.pressure_features(state)
    assert features.surprise == state.pending and features.load == pytest.approx(float(np.linalg.norm(state.memory)))


def _steps(
    core: api.Core, state: core_state.CoreState, events: Sequence[EventLatent],
) -> Iterator[tuple[core_state.CoreState, EventLatent]]:
    for event in events:
        yield state, event
        state, _ = core.observe(state, event)


def test_pressure_weights_fit_what_skipping_consolidation_costs() -> None:
    from anjo.sleep.mind.core import sleep_eval

    core = api.Core.fresh(SMALL, seed=0)
    samples = sleep_eval.pressure_samples(core, toy.lives(3, seed=6), horizon=2)
    assert samples and all(len(s.features) == 3 for s in samples)
    weights, fit = sleep_eval.fit_pressure(samples)
    assert len(weights) == 3 and all(w >= 0 for w in weights)
    assert set(fit) >= {"r2", "samples", "day_cost", "mean_cost"}


def test_the_ablations_run_the_same_lives_under_each_arm() -> None:
    from anjo.sleep.mind.core import sleep_eval

    core = api.Core.fresh(SMALL, seed=0)
    lives = toy.lives(2, seed=6)
    scores = sleep_eval.ablations(core, lives, budget=3)
    assert [s.arm for s in scores] == list(sleep_eval.ABLATIONS)
    assert len({s.problems for s in scores}) == 1


def test_rehearsal_gains_pair_every_arm_on_the_same_nights() -> None:
    from anjo.sleep.mind.core import sleep_eval

    core = api.Core.fresh(SMALL, seed=0).with_thresholds(api.Thresholds(speak=0.5, ignite=0.0))
    samples, gains = sleep_eval.rehearsal_gain(core, toy.lives(2, seed=6), budget=3, seed=1)
    assert samples and all(set(s.all) == set(sleep_eval.ARMS) for s in samples)
    assert all(0 < s.rehearsed <= 3 for s in samples)
    assert {(g.arm, g.base) for g in gains if g.scope == "all"} >= {("dream", "random"), ("dream", "recent")}
