# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's continuum memory (``mind.core.cms``, ``ANJO_SLEEP_MIND_CMS``): each level steps at its own
period and only then (the night level only when Sleep consolidates), the switch off leaves every
output as it was, and the measurement script runs on a tiny config (toy lives, CPU only)."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import platform
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anjo.sleep.mind.core import api, cms, evaluate, net, night, sleep_eval, toy, train  # noqa: E402
from anjo.sleep.mind.core import state as core_state  # noqa: E402
from anjo.sleep.mind.core.types import CoreConfig, EventLatent, ThreadLabel, config_record  # noqa: E402
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog  # noqa: E402

T0 = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
SMALL = CoreConfig(dim_in=toy.DIM, width=16, slots=4)
CMS = dataclasses.replace(SMALL, cms=True)


def _event(event_id: str, minutes: float, seed: int) -> EventLatent:
    vec = np.random.default_rng(seed).normal(size=toy.DIM).astype(np.float32)
    return EventLatent(event_id, "toy-1", T0 + timedelta(minutes=minutes), "turn", vec)


def _levels(state: core_state.CoreState) -> np.ndarray:
    assert state.levels is not None
    levels: np.ndarray = np.array(state.levels)
    return levels


# ── The switch ────────────────────────────────────────────────────────────────


def test_the_switch_is_one_and_off_by_default() -> None:
    assert not cms.cms_on({}) and not cms.cms_on({cms.CMS_ENV: "0"}) and cms.cms_on({cms.CMS_ENV: " 1 "})
    assert not CoreConfig(dim_in=4).cms
    with pytest.raises(ValueError, match="fast"):
        CoreConfig(dim_in=4, fast=False, cms=True)
    with pytest.raises(ValueError, match="continuum"):
        net.CoreNet(SMALL).sleep(net.CoreNet(SMALL).initial(1))


def test_with_the_switch_off_the_core_records_and_saves_as_before() -> None:
    assert config_record(SMALL) == {k: v for k, v in dataclasses.asdict(SMALL).items() if k != "cms"}
    assert config_record(CMS)["cms"] is True
    off, on = net.CoreNet(SMALL), net.CoreNet(CMS)
    assert not any(name.startswith("cms") for name, _ in off.named_parameters())
    assert {name for name, _ in on.named_parameters()} - {name for name, _ in off.named_parameters()} == {
        "cms_rates.weight", "cms_rates.bias", "cms_mix"}
    core = api.Core.fresh(SMALL, seed=0)
    state = core.observe(core.init_state("toy-1"), _event("e0", 0, 0))[0]
    assert state.levels is None and core.tensors(state).levels.shape == (1, 0, 16, 16)
    with np.load(io.BytesIO(core_state.save(state)), allow_pickle=False) as held:
        assert "levels" not in held.files


# Captured on trunk (e5f21c86), before the continuum memory, with this same function: every output
# of a small core trained on fixed seeds, digested (training, observing, dreaming, the night, the
# traces, the score, the transplant, the sleep evaluations, adaptation). The digests are this
# platform's float kernels; elsewhere the test skips and the structural test above still runs.
TRUNK: dict[str, str] = {
    "train": "f92646db32a71b7a2baa091e8996c02691fd0538ca5b075c81ddeec1d7ee1a19",
    "observe": "e71cd8ee5bd6a4817118063ab0821d97c95675b1879b85e6c17c57d3d851c969",
    "dream": "70d825c2284684858c08674bc85d1b148734af2342a71d73b0a60708351369a9",
    "night": "8efa5dac2227011efde2658eda14bd1486c923977b34b2e172647e2aecbf39a5",
    "traces": "5000b04f96bb542ac8453f2d763f90ba45aa62587a1eca3626aedb5ef13de722",
    "score": "7b1fa4005612a0b67d9e923bdd9a3229f7b515230c3397d2512681fe08d8d36b",
    "transplant": "a002cd430ecf33e36d12b74f2b0ce61e357588133508dc3c6a9b782dab1a7df5",
    "consolidation": "c15bf2195dac473bd83dcd55382774d9ce25b567890e87ad15e8f8f576499f6f",
    "rehearsal": "568ffcaa087a2b08d5e732c9181477fc3f80db17509525e5bcb6e27bea81a050",
    "ablations": "dc472942cab48185ccf253604c42269632a8da884befdc7722d7a92f6c19987e",
    "adapt": "e832e4b16f679abeb8d6a7fe681c3e3d44cc11434f42c4af712e488b7beac8e3",
}
TRUNK_PLATFORM = ("2.11.0", "arm64")


def _digest(*parts: object) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else repr(part).encode())
    return h.hexdigest()


def fingerprints() -> dict[str, str]:
    lives = toy.lives(6, seed=11)
    run = train.train_shared(lives[:4], lives[4:], SMALL, seed=3, epochs=2,
                             pairs=lambda epoch: toy.contradiction_pairs(2, seed=50 + epoch, revisions=True))
    core = run.core
    out = {"train": _digest(core.model_id, run.log, run.best_epoch, dataclasses.astuple(core.thresholds))}
    life = toy.life(4, 0)
    state, weights = core.surprise_weights(core.init_state(life.life_id), life.events[:-6])
    out["observe"] = _digest(core_state.save(state), sorted(weights.items()), core.predict(state), core.decide(state))
    dream = core.dream_states(state, 8, np.random.default_rng(0), rollouts=4, ignite_at=0.0)
    out["dream"] = _digest(dream.ignitions, [core_state.save(d.state) for d in dream.states],
                           [(core_state.save(b.dream.state), b.steps, b.speak, b.slot) for b in dream.boundaries])
    slept = core.consolidate(state)
    pruned = core.consolidate(state, rank=2, residue=0.1)
    rehearsed = core.rehearse(slept, life.events[:5])
    after = [core.observe(slept, e)[1] for e in life.events[-6:]]
    out["night"] = _digest(core_state.save(slept), core_state.save(pruned), core_state.save(rehearsed), after,
                           core.pressure_features(state), core.sleep_pressure(state))
    test = toy.lives(8, seed=3)
    runs = evaluate.traces(core, test)
    out["traces"] = _digest(runs, evaluate.calibration(runs), evaluate.separation(runs, test))
    out["score"] = _digest([evaluate.score(runs, test, t) for t in (0.0, 0.2, core.threshold)])
    out["transplant"] = _digest(evaluate.transplant(core, toy.pairs(6, seed=5), tolerance=0.1))
    chained = toy.chained(2, seed=2, links=2)
    out["consolidation"] = _digest(sleep_eval.consolidation(core, chained, sleep_eval.night_ops(core)))
    out["rehearsal"] = _digest(sleep_eval.rehearsal_gain(core, toy.lives(2, seed=6), budget=3, seed=1))
    out["ablations"] = _digest(sleep_eval.ablations(core, lives[:2], budget=3))
    out["adapt"] = _digest(sorted((k, v.tobytes()) for k, v in train.adapt(core, life).items()))
    return out


def test_with_the_switch_off_every_output_is_byte_identical_to_trunk() -> None:
    if (torch.__version__, platform.machine()) != TRUNK_PLATFORM:
        pytest.skip(f"the trunk digests were captured on torch {TRUNK_PLATFORM[0]}, {TRUNK_PLATFORM[1]}")
    assert fingerprints() == TRUNK


# ── Each level at its own period ──────────────────────────────────────────────


def test_the_session_level_steps_only_when_a_session_ends() -> None:
    core = api.Core.fresh(CMS, seed=0)
    state = core.observe(core.init_state("toy-1"), _event("e0", 0, 0))[0]
    for n in range(1, 5):  # 30 minutes apart: one session
        before = _levels(state)
        state, _ = core.observe(state, _event(f"e{n}", 30 * n, n))
        now = _levels(state)
        assert np.array_equal(now[cms.SESSION], before[cms.SESSION])  # the level waits for its period
        assert np.abs(now[cms.SESSION_PENDING] - before[cms.SESSION_PENDING]).sum() > 0  # its step builds up
    assert np.abs(_levels(state)[cms.SESSION]).sum() == 0
    before = _levels(state)
    gap = 120 + 60 * 24 * cms.SESSION_GAP_DAYS  # three hours and more after the last: a new session
    state, _ = core.observe(state, _event("e5", gap, 5))
    now = _levels(state)
    np.testing.assert_allclose(now[cms.SESSION], before[cms.SESSION] - before[cms.SESSION_PENDING], atol=1e-6)
    assert np.abs(now[cms.SESSION]).sum() > 0 and np.abs(now[cms.SESSION_PENDING]).sum() > 0  # the new session's
    assert np.array_equal(now[cms.NIGHT], before[cms.NIGHT])


def test_the_night_level_steps_only_when_sleep_consolidates() -> None:
    core = api.Core.fresh(CMS, seed=0)
    state = core.init_state("toy-1")
    for n in range(12):  # four days, three events each: sessions end, no night runs
        state, _ = core.observe(state, _event(f"e{n}", 60 * 24 * (n // 3) + 30 * (n % 3), n))
    held = _levels(state)
    assert np.abs(held[cms.NIGHT]).sum() == 0 and np.abs(held[cms.NIGHT_PENDING]).sum() > 0
    assert np.abs(held[cms.SESSION]).sum() > 0 and np.abs(state.memory).sum() > 0
    slept = core.consolidate(state)
    after = _levels(slept)
    np.testing.assert_allclose(after[cms.NIGHT], held[cms.NIGHT] - held[cms.NIGHT_PENDING], atol=1e-6)
    np.testing.assert_allclose(after[cms.SESSION], held[cms.SESSION] - held[cms.SESSION_PENDING], atol=1e-6)
    assert np.abs(after[cms.NIGHT_PENDING]).sum() == 0 and np.abs(after[cms.SESSION_PENDING]).sum() == 0
    assert np.abs(slept.memory).sum() == 0 and np.abs(slept.momentum).sum() == 0 and slept.pending == 0.0
    assert api.SLOW_MEMORY not in slept.delta  # the night level is the slow memory under the switch
    assert core.predict(slept) == core.predict(state)  # the heads read no memory: sleep leaves the forecasts
    later = slept
    for n in range(12, 18):
        later, _ = core.observe(later, _event(f"e{n}", 60 * 24 * (4 + (n - 12) // 3) + 30 * n, n))
        assert np.array_equal(_levels(later)[cms.NIGHT], after[cms.NIGHT])  # nothing else moves it
    pruned = core.consolidate(later, rank=1)
    step = _levels(pruned)[cms.NIGHT] - after[cms.NIGHT]
    assert np.linalg.matrix_rank(step, tol=1e-5) == 1


def test_a_cms_state_round_trips_and_its_night_and_dreams_leave_the_levels_as_the_rules_say() -> None:
    core = api.Core.fresh(CMS, seed=0)
    state = core.init_state("toy-1")
    for n in range(6):
        state, _ = core.observe(state, _event(f"e{n}", 300 * n, n))
    back = core_state.load(core_state.save(state), model_id=core.model_id)
    assert core_state.equal(back, state) and back.levels is not None
    assert not core_state.equal(dataclasses.replace(back, levels=np.zeros_like(back.levels)), state)
    with pytest.raises(core_state.StateError, match="continuum"):
        dataclasses.replace(state, levels=np.zeros((2, 16, 16), np.float32))
    rehearsed = core.rehearse(state, [_event("r0", 0, 9)])
    assert np.array_equal(_levels(rehearsed), _levels(state))  # rehearsal reaches the fast level alone
    before = core_state.save(state)
    dream = core.dream_states(state, 4, np.random.default_rng(0), rollouts=2, ignite_at=0.0)
    assert core_state.save(state) == before and dream.states
    assert all(d.state.levels is not None and d.state.levels.shape == (cms.LEVELS, 16, 16) for d in dream.states)


# ── Training ──────────────────────────────────────────────────────────────────


def test_training_under_the_switch_sleeps_between_days_so_the_night_level_learns() -> None:
    lives = toy.lives(4, seed=11)
    data = train.batch(list(lives[:2]), CMS.horizon)
    marks = train.nights(data)
    for b, life in enumerate(lives[:2]):
        days = [e.t.date() for e in life.events]
        expected = [float(n > 0 and days[n] > days[n - 1]) for n in range(len(days))]
        assert marks[b, : len(days)].tolist() == expected and marks[b, len(days):].sum() == 0
    torch.manual_seed(0)
    model = net.CoreNet(CMS)
    total, _ = train.loss(model, data)
    total.backward()
    night_rate = model.cms_rates.weight.grad[1]  # the night level acts only through the unroll's sleeps
    assert night_rate.abs().sum() > 0 and model.cms_mix.grad[1].abs() > 0
    first = train.train_shared(lives[:3], lives[3:], CMS, seed=3, epochs=2)
    second = train.train_shared(lives[:3], lives[3:], CMS, seed=3, epochs=2)
    assert first.core.model_id == second.core.model_id and first.log == second.log
    assert first.core.model_id != train.train_shared(lives[:3], lives[3:], SMALL, seed=3, epochs=2).core.model_id


def test_the_night_runs_a_cms_core_only_under_the_switch(tmp_path: Path) -> None:
    for config, wrong in ((CMS, {}), (SMALL, {cms.CMS_ENV: "1"})):
        path = tmp_path / f"core-{config.cms}.npz"
        api.Core.fresh(config, seed=0).save(path)
        with pytest.raises(ValueError, match=cms.CMS_ENV):
            night.from_env(wrong, path, tmp_path, people=frozenset(), workspace=lambda _: MemoryWorkspaceLog(),
                           synthetic=lambda _: True)
    built = night.from_env({cms.CMS_ENV: "1"}, tmp_path / "core-True.npz", tmp_path, people=frozenset(),
                           workspace=lambda _: MemoryWorkspaceLog(), synthetic=lambda _: True)
    assert built.core.config.cms
    assert api.Core.load(tmp_path / "core-False.npz").model_id == api.Core.fresh(SMALL, seed=0).model_id


# ── The measurement ───────────────────────────────────────────────────────────


def test_a_probe_lives_other_lives_between_a_lifes_halves() -> None:
    from anjo.sleep.tools import mind_cms_eval as eval_cms

    life = next(s for s in toy.lives(8, seed=3) if any(lab.id.endswith("-gap") for lab in s.labels))
    fillers = toy.lives(2, seed=7)
    cut = life.days[len(life.days) // 2]
    alone, loaded = eval_cms.probe(life, ()), eval_cms.probe(life, fillers)
    assert len(loaded.events) == len(life.events) + sum(len(f.events) for f in fillers)
    assert len(loaded.days) == len(life.days) + sum(len(f.days) for f in fillers)
    assert {lab.id for lab in loaded.labels} == {lab.id for lab in alone.labels}
    assert alone.labels and all(lab.knowable_at.date() >= cut for lab in alone.labels)  # the second half's only
    first_half = [e.event_id for e in loaded.events if e.event_id in {x.event_id for x in life.events}
                  and e.t.date() < cut]
    assert first_half == [e.event_id for e in life.events if e.t.date() < cut]  # the first half as it was
    shift = loaded.labels[0].knowable_at - alone.labels[0].knowable_at
    assert shift == timedelta(days=sum(len(f.days) for f in fillers))


def test_the_measurement_runs_on_a_tiny_config(tmp_path: Path) -> None:
    from anjo.sleep.tools import mind_cms_eval as eval_cms

    plan = eval_cms.Plan(seeds=(0, 1), epochs=1, patience=1, width=8, slots=2, lives_per_epoch=4, val_lives=2,
                         test_lives=4, pairs=2, chains=1, links=2, fillers=(0, 1), probes=3, rehearsal_lives=1)
    results, text = eval_cms.run(plan)
    assert sorted((arm, seed) for arm, seed, _ in results) == [("off", 0), ("off", 1), ("on", 0), ("on", 1)]
    for _, _, metrics in results:
        assert {"separation", "recall_nightly", "transplant_flipped", "chained_recall_consolidate",
                "rehearsal_dream_minus_random", "retention_auc_nightly@1", "retention_recall_no_nights@0"} <= set(metrics)
        assert metrics["stored@1"] is not None and metrics["stored@0"] is not None
        assert metrics["stored@1"] > metrics["stored@0"]  # each filler life is more stored before the ask
    assert "on - off (paired)" in text and "retention_auc_nightly@1" in text
    summary = eval_cms.summarize(results)
    assert summary["params"]["on_minus_off"] is not None and summary["params"]["on_minus_off"][0] > 0
    assert eval_cms.interval([1.0, 3.0]) == pytest.approx((2.0, 2.0 - 12.706, 2.0 + 12.706))
    out: list[str] = []
    assert eval_cms.main(["--export", str(tmp_path)], emit=out.append) == 2 and "together" in out[0]


def test_one_call_citing_two_fresh_threads_catches_both() -> None:
    life = toy.life(0, 0)
    first, second = life.events[0], life.events[1]
    labels = tuple(
        ThreadLabel(name, "gap", (e.event_id,), (e.event_id,), e.t, e.t + timedelta(days=3), failed=True)
        for name, e in (("a", first), ("b", second)))
    stream = dataclasses.replace(life, labels=labels)
    both = (first.event_id, second.event_id)
    day = first.t.date()
    calls = (evaluate.DayCall(day, 0.9, both), evaluate.DayCall(date.fromordinal(day.toordinal() + 1), 0.9, both))
    result = evaluate.score([evaluate.LifeTrace(life.life_id, calls, ())], [stream], 0.5)
    assert (result.caught, result.recall, result.repeats, result.false) == (2, 1.0, 1, 0)
    assert result.by_family == (("gap", 2, 2),)


def test_the_fast_switch_measures_the_cores_memory_on_against_off_with_cms_off(tmp_path: Path) -> None:
    # 2026-11-02: MIND's memory (fast weights) on against N-nofast on the real latents, pre-registered
    from anjo.sleep.tools import mind_cms_eval as eval_cms

    cms_plan, fast_plan = eval_cms.Plan(), eval_cms.Plan(switch="fast")
    assert cms_plan.switch == "cms" and eval_cms.SWITCHES == ("cms", "fast")
    assert eval_cms.arm_config(cms_plan, "on", 4) == CoreConfig(dim_in=4, width=64, slots=8, cms=True)
    assert eval_cms.arm_config(cms_plan, "off", 4) == CoreConfig(dim_in=4, width=64, slots=8)
    assert eval_cms.arm_config(fast_plan, "on", 4) == CoreConfig(dim_in=4, width=64, slots=8)
    assert eval_cms.arm_config(fast_plan, "off", 4) == CoreConfig(dim_in=4, width=64, slots=8, fast=False)
    tiny = dataclasses.replace(fast_plan, seeds=(0,), epochs=1, patience=1, width=8, slots=2, lives_per_epoch=4,
                               val_lives=2, test_lives=4, pairs=2, chains=1, links=2, fillers=(0,), probes=3,
                               rehearsal_lives=1)
    results, text = eval_cms.run(tiny)
    assert sorted(arm for arm, _, _ in results) == ["off", "on"]
    assert text.startswith("MIND fast weights (the core's memory) off vs on")
    with pytest.raises(SystemExit):
        eval_cms._parser().parse_args(["--switch", "memory"])
    assert eval_cms._parser().parse_args(["--switch", "fast"]).switch == "fast"
