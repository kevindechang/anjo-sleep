# mypy: disallow-untyped-decorators=False
"""The cog core (``anjo.sleep.mind.cog.core``) on the toy world (``cog.toy_life``): the fast-weight
rule, shapes and determinism, then a tiny trained core against a per-person counter, its
uncertainty over a life, the fast-memory and ensemble lesions, the night after a shift, and the
self-forecast. One tiny core is trained once per session (``trained_tiny``) and shared with
``test_cog_voi``."""
from __future__ import annotations

import functools
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from anjo.sleep.mind.cog import core, train_cog
from anjo.sleep.mind.cog.encode import Vocab, activity_kinds, build_vocab, collate, encode_life, encode_queries
from anjo.sleep.mind.cog.night import (
    NO_GAIN, Adapter, Replayed, night, paid, priorities, served, transfer, update_gain,
)
from anjo.sleep.mind.cog.toy_life import ToyLife
from anjo.sleep.mind.cog.types import Event, Life, Prediction, Query, Resolution
from anjo.sleep.mind.core.net import fast_update

DEV = range(10_000, 10_024)


@functools.lru_cache(maxsize=1)
def trained_tiny() -> train_cog.Trained:
    """A small core trained for a few seconds on toy lives; shared by this file and test_cog_voi. Every
    life gets asks: with half of 384 lives asking, it hardly learns what a work answer says."""
    cfg = train_cog.TrainConfig(lives=384, epochs=6, batch=32, dim=32, hidden=64, tbptt=28, dev_lives=16, lr=6e-3,
                                n_days=56, seed=0, threads=1, patience=3, asking_share=1.0)
    return train_cog.train(cfg, world=train_cog.toy_world)


# ── the rule and the shapes ──────────────────────────────────────────────────


def test_multi_key_update_with_one_key_is_net_fast_update() -> None:
    g = torch.Generator().manual_seed(0)
    b, d = 3, 8
    m, s = torch.randn(b, d, d, generator=g), torch.randn(b, d, d, generator=g)
    k = torch.nn.functional.normalize(torch.randn(b, d, generator=g), dim=-1)
    v = torch.randn(b, d, generator=g)
    theta, eta, alpha = torch.rand(b, generator=g) * 0.5, torch.rand(b, generator=g), torch.rand(b, generator=g)
    want = fast_update(m, s, k, v, theta, eta, alpha)
    got = core.multi_fast_update(m, s, k[:, None], v[:, None], theta[:, None], eta, alpha)
    for a, w in zip(got[:2], want[:2], strict=True):
        assert torch.allclose(a, w, atol=1e-6)
    assert torch.allclose(got[2][:, 0], want[2], atol=1e-6)


def test_the_hazard_gate_writes_a_confident_miss_harder_and_keeps_no_evidence_without_a_probe() -> None:
    """With ``hazard`` the write's strength also reads ``hz``, the recent excess of the probe's
    surprise over the entropy it expected, and ``u / H``, the ensemble's share of that entropy."""

    def make(hazard: bool) -> core.CogNet:
        torch.manual_seed(0)
        return core.CogNet(core.CogConfig(vocab_size=20, dim=8, hidden=16, hazard=hazard))

    plain, hazard = make(False), make(True)
    assert all(torch.equal(v, hazard.state_dict()[k]) for k, v in plain.state_dict().items())  # built last
    key = torch.nn.functional.normalize(torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(1)), dim=-1)
    sure = plain.initial(1, 2)._replace(probe_logits=torch.full((1, 2, 5), 4.0), probe_key=key,
                                        probe_ok=torch.tensor([[1.0, 0.0]]))
    y, mask = torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)
    off, _ = plain._write(sure, y, mask, True, False)
    on, _ = hazard._write(sure, y, mask, True, False)
    assert torch.equal(off.hz, sure.hz)  # without the hazard nothing is kept
    assert float(on.hz[0, 0]) > 1.0 and float(on.hz[0, 1]) == 0.0  # activity 1 had no probe
    assert float(on.mem.norm()) > float(off.mem.norm())
    calm, _ = hazard._write(sure._replace(probe_logits=torch.full((1, 2, 5), -4.0)), y, mask, True, False)
    assert float(calm.hz[0, 0]) < 0.0  # sure it would not happen, and it did not: less surprise than expected


def _tiny_net(seed: int, **kw: object) -> tuple[core.CogNet, list[Life]]:
    lives: list[Life] = [ToyLife(s, n_days=21) for s in range(4)]
    vocab = build_vocab(lives)
    torch.manual_seed(seed)
    return core.CogNet(core.CogConfig(vocab_size=len(vocab), dim=16, hidden=32, **kw)), lives  # type: ignore[arg-type]


def test_unroll_shapes_and_determinism_by_seed() -> None:
    net_a, lives = _tiny_net(7)
    net_b, _ = _tiny_net(7)
    vocab = build_vocab(lives)
    batch = collate([encode_life(vocab, life, np.random.default_rng(1)) for life in lives])
    out_a = train_cog.unroll(net_a, batch)
    out_b = train_cog.unroll(net_b, batch)
    b, dmax, q = batch.q_mask.shape
    assert out_a.query_logits.shape == (b, dmax, q, 5)
    assert out_a.probe_logits.shape == (b, dmax, 2, 5)
    assert torch.equal(out_a.query_logits, out_b.query_logits)
    loss, parts = train_cog.loss(out_a, batch)
    assert torch.isfinite(loss) and {"query", "u", "probe", "answer", "self"} <= set(parts)
    net_c, _ = _tiny_net(8)
    assert not torch.equal(train_cog.unroll(net_c, batch).query_logits, out_a.query_logits)


def test_the_oracle_is_the_target_only_in_lives_without_asks() -> None:
    net, lives = _tiny_net(3)
    vocab = build_vocab(lives)
    rng = np.random.default_rng(0)
    arrays = [encode_life(vocab, life, rng, 0.5, asking=(i % 2 == 0)) for i, life in enumerate(lives)]
    assert not any(a is not None for a in arrays[1].asks)
    batch = collate(arrays)
    out = train_cog.unroll(net, batch)
    p = core.mean_prob(out.query_logits)
    m = batch.q_mask.float()
    asking = batch.asking.float()[:, None, None]
    target = asking * batch.q_y + (1 - asking) * batch.q_oracle
    want = (-(target * p.log() + (1 - target) * (1 - p).log()) * m).sum() / m.sum()
    _, parts = train_cog.loss(out, batch, soft=1.0)
    assert np.isclose(parts["query"], float(want), rtol=1e-5)
    _, plain = train_cog.loss(out, batch)
    assert not np.isclose(plain["query"], parts["query"])


def test_dev_is_scored_without_asks_like_the_eval_and_also_with_them() -> None:
    lives = [ToyLife(s, n_days=21) for s in DEV[:6]]
    passive, asking = train_cog.dev_batches(build_vocab(lives), lives, batch=4, ask_rate=0.5)
    assert [b.asking.shape[0] for b in passive] == [4, 2] == [b.asking.shape[0] for b in asking]
    assert not any(bool(b.asking.any()) or int(b.ask_mask.sum()) for b in passive)
    assert all(bool(b.asking.all()) for b in asking) and sum(int(b.ask_mask.sum()) for b in asking) > 0


def test_the_trainer_knows_the_main_and_the_independent_worlds() -> None:
    assert {"toy", "real", "indep"} <= set(train_cog.WORLDS)
    life = train_cog.WORLDS["indep"](10_000, 7)
    assert life.meta.n_days == 7 and life.queries(0)


def test_some_training_lives_never_hear_an_answer_whatever_the_target() -> None:
    assert train_cog.asking_share(train_cog.TrainConfig(soft=0.0)) == train_cog.TrainConfig().asking_share < 1.0


def test_tallies_count_each_activity_by_weekday_and_both_paths_agree() -> None:
    net, lives = _tiny_net(5, tallies=True)
    net.eval()
    vocab = build_vocab(lives)
    life = lives[0]
    batch = collate([encode_life(vocab, life, np.random.default_rng(0), 0.0, asking=False)])
    with torch.no_grad():
        out = train_cog.unroll(net, batch)
    acts, n = life.meta.activities, life.meta.n_days
    want = np.zeros((len(acts), 7, 4))
    for d in range(n):
        want[:, (life.meta.start_weekday + d) % 7] += activity_kinds(life.events(d), acts)
    assert want[:, :, 0].sum() > 0 and want[:, :, 1].sum() > 0
    assert np.allclose(out.final.tally[0].numpy(), want)
    assert float(out.final.tdays[0].sum()) == n
    rows = _run(core.MindPredictor(core.Checkpoint(net, vocab), night="off"), life)
    p_batch = core.mean_prob(out.query_logits)[batch.q_mask].numpy()
    assert np.allclose([r[2] for r in rows], np.clip(p_batch, core.P_FLOOR, 1 - core.P_FLOOR), atol=1e-5)


def test_a_query_carries_its_family_when_the_vocabulary_knows_it() -> None:
    lives = [ToyLife(s, n_days=21) for s in range(2)]
    vocab = build_vocab(lives)
    q = lives[0].queries(3)[0]
    assert vocab.id(f"family={q.family}") not in (0, 1)
    assert vocab.id(f"family={q.family}") in encode_queries(vocab, [q]).tok[0]
    old = Vocab(tuple(t for t in vocab.tokens if not t.startswith("family=")))  # a checkpoint from before
    assert encode_queries(old, [q]).tok[0].tolist() == [old.id(x) for x in q.tokens] + [0] * (
        encode_queries(old, [q]).tok.shape[1] - len(q.tokens))


def test_fast_off_never_writes_and_reads_zero() -> None:
    net, lives = _tiny_net(0, fast=False)
    vocab = build_vocab(lives)
    batch = collate([encode_life(vocab, lives[0], np.random.default_rng(0))])
    out = train_cog.unroll(net, batch)
    assert float(out.final.mem.abs().sum()) == 0.0


def test_the_continuum_memory_accumulates_by_day_and_steps_at_night() -> None:
    net, lives = _tiny_net(0, cms=True)
    vocab = build_vocab(lives)
    batch = collate([encode_life(vocab, lives[0], np.random.default_rng(0))])
    st = net.initial(1, 2)
    out, _ = train_cog.run_days(net, batch, st, 0, 5, train_cog._acc(batch))  # nights between days
    loss, _ = train_cog.loss(out, batch)
    loss.backward()
    assert net.cms_rate.grad is not None and net.cms_mix.grad is not None
    st = out.final
    assert float(st.levels[:, 0].detach().abs().sum()) > 0.0
    day, _ = train_cog.run_days(net, batch, st, 5, 6, train_cog._acc(batch), cms=True)
    pending = day.final.levels[:, 1]
    after = net.consolidate(day.final)
    assert torch.allclose(after.levels[:, 0], day.final.levels[:, 0] - pending)
    assert float(after.levels[:, 1].detach().abs().sum()) == 0.0


def test_the_predictor_refuses_a_continuum_memory_the_core_lacks() -> None:
    net, lives = _tiny_net(0)
    ck = core.Checkpoint(net, build_vocab(lives))
    with pytest.raises(ValueError):
        core.MindPredictor(ck, cms=True)
    with pytest.raises(ValueError):
        core.MindPredictor(ck, ensemble=6)


# ── a trained core on the toy world ──────────────────────────────────────────


def _counter_p(history: dict[tuple[str, int], list[float]], subject: str, weekday: int) -> float:
    seen = history.get((subject, weekday), [])
    return (sum(seen) + 2 * 0.35) / (len(seen) + 2)


def _run(predictor: core.MindPredictor, life: Life) -> list[tuple[int, str, float, float | None, float, float]]:
    """(asked day, subject, p, u, outcome, oracle_p) per query, the protocol run plainly."""
    rows = []
    predictor.begin(life.meta)
    for d in range(life.meta.n_days):
        predictor.observe(d, life.events(d))
        queries = life.queries(d)
        for q, pr in zip(queries, predictor.predict(queries), strict=True):
            r = life.resolve(q.qid)
            rows.append((d, q.subject, pr.p, pr.u, float(r.outcome), r.oracle_p))
        predictor.end_day(d)
    return rows


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def test_training_beats_a_per_person_counter_on_the_regime_activity() -> None:
    trained = trained_tiny()
    assert trained.history[-1]["dev"] < trained.history[0]["dev"]
    mind, counter, ys = [], [], []
    pred = core.MindPredictor(trained, night="off")
    for seed in DEV:
        life = ToyLife(seed)
        history: dict[tuple[str, int], list[float]] = {}
        pred.begin(life.meta)
        for d in range(life.meta.n_days):
            events = life.events(d)
            pred.observe(d, events)
            for q, pr in zip(life.queries(d), pred.predict(life.queries(d)), strict=True):
                if q.subject == "run" and d >= 14:
                    wd = (life.meta.start_weekday + q.target_day) % 7
                    mind.append(pr.p)
                    counter.append(_counter_p(history, "run", wd))
                    ys.append(float(life.resolve(q.qid).outcome))
            for act in life.meta.activities:
                did = float(any(e.kind == "did" and e.subject == act for e in events))
                history.setdefault((act, (life.meta.start_weekday + d) % 7), []).append(did)
            pred.end_day(d)
    y = np.array(ys)
    assert _logloss(np.array(mind), y) < _logloss(np.array(counter), y)


def test_uncertainty_is_higher_in_the_first_days_than_after_a_month() -> None:
    pred = core.MindPredictor(trained_tiny(), night="off")
    early, late = [], []
    for seed in DEV:
        for d, _, _, u, _, _ in _run(pred, ToyLife(seed)):
            assert u is not None and u >= 0.0
            (early if d < 5 else late if d >= 30 else []).append(u)
    assert np.mean(early) > np.mean(late)


def test_fast_memory_off_hurts_a_person_with_a_reversed_effect() -> None:
    trained = trained_tiny()
    on, off = core.MindPredictor(trained, night="off"), core.MindPredictor(trained, fast=False, night="off")
    loss_on, loss_off = [], []
    for seed in DEV:
        life = ToyLife(seed, reverse=True, shift_day=False)
        for rows, out in ((_run(on, life), loss_on), (_run(off, life), loss_off)):
            run = [(p, y) for d, s, p, _, y, _ in rows if s == "run" and d >= 21]
            out.append(_logloss(np.array([p for p, _ in run]), np.array([y for _, y in run])))
    assert np.mean(loss_on) < np.mean(loss_off)


def test_one_head_gives_no_uncertainty() -> None:
    pred = core.MindPredictor(trained_tiny(), ensemble=1, night="off")
    life = ToyLife(10_001)
    rows = _run(pred, life)
    assert rows and all(r[3] is None for r in rows)
    assert all(0.0 < r[2] < 1.0 for r in rows)


def test_predictor_is_deterministic_and_bounded() -> None:
    trained = trained_tiny()
    life = ToyLife(10_002)
    a = _run(core.MindPredictor(trained), life)
    b = _run(core.MindPredictor(trained), life)
    assert a == b
    assert all(0.0 < r[2] < 1.0 for r in a)


def test_self_forecast_is_a_positive_loss() -> None:
    pred = core.MindPredictor(trained_tiny())
    life = ToyLife(10_003)
    pred.begin(life.meta)
    pred.observe(0, life.events(0))
    first = pred.self_forecast()
    for d in range(1, 30):
        pred.end_day(d - 1)
        pred.observe(d, life.events(d))
        pred.predict(life.queries(d))
    assert 0.0 < first < 5.0 and 0.0 < pred.self_forecast() < 5.0


# ── the night ────────────────────────────────────────────────────────────────


def test_priorities_follow_surprise_times_need() -> None:
    lg = np.zeros(3)
    items = (Replayed(lg, 1.0, 2.0, "run", 9), Replayed(lg, 1.0, 0.1, "run", 9),
             Replayed(lg, 1.0, 2.0, "gym", 9), Replayed(lg, 1.0, 2.0, "run", 2))
    pr = priorities(items, today=9, counts={"run": 9, "gym": 1})
    assert pr[0] > pr[1] and pr[0] > pr[2] and pr[0] > pr[3]
    assert np.isclose(pr.sum(), 1.0)


def test_night_moves_the_adapter_toward_what_surprised_it() -> None:
    rng = np.random.default_rng(0)
    items = tuple(Replayed(np.full(3, -2.0), 1.0, 2.1, "run", i, (0, i % 7)) for i in range(40))
    start = Adapter.zero(2)
    cells = np.array([(0, wd) for wd in range(7)])
    for mode in ("uniform", "surprise"):
        after = night(start, items, today=40, mode=mode, rng=np.random.default_rng(1), counts={"run": 40})
        assert float(np.min(after.shift(cells))) > 0.0
    t0 = time.perf_counter()
    night(start, items, today=40, mode="surprise", rng=np.random.default_rng(1), counts={"run": 40})
    assert time.perf_counter() - t0 < 0.2
    with pytest.raises(ValueError):
        night(start, items, today=40, mode="sometimes", rng=rng, counts={})


def _p(base: np.ndarray, adapter: Adapter, wd: int) -> float:
    return float(np.mean(1 / (1 + np.exp(-(base + adapter.shift(np.array([[0, wd]]))[0])))))


def _shifted_stream(mode: str, seed: int, *, gated: bool = False, shift: bool = True) -> list[float]:
    """The night alone after a shift, on a base that cannot adapt: the base logits know the old
    habit (run on weekdays 0-2, logits 1.73 and -2.44: the habit's true 0.85 and 0.08) for good;
    from day 21 the person runs on weekdays 4-6 (or, ``shift=False``, keeps the habit, so the base
    is exactly calibrated). Each day one run query per weekday is predicted (base plus the adapter
    served: with ``gated``, only while its own predictions have beaten the base's), resolves, and
    the night runs. The log losses served on days 21-34, the two weeks after the shift. The outcomes
    have their own stream, so every variant sees the same ones."""
    rng, world = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    adapter, items, counts, losses, gain = Adapter.zero(2), [], {"run": 0}, [], NO_GAIN
    for day in range(42):
        habit = (4, 5, 6) if shift and day >= 21 else (0, 1, 2)
        today = []
        for wd in range(7):
            base = np.full(3, 1.73 if wd < 3 else -2.44)
            p = _p(base, served(adapter, gain) if gated else adapter, wd)
            y = float(world.random() < (0.85 if wd in habit else 0.08))
            if 21 <= day < 35:
                losses.append(-np.log(p if y else 1 - p))
            p_fit = _p(base, adapter, wd)
            today.append(Replayed(base, y, -np.log(p_fit if y else 1 - p_fit), "run", day, (0, wd)))
            counts["run"] += 1
        items.extend(today)
        gain = update_gain(gain, today)
        adapter = night(adapter, items, today=day, mode=mode, rng=rng, counts=counts)
    return losses


def test_priority_night_recovers_faster_than_no_night_and_uniform_after_a_shift() -> None:
    after = {mode: np.mean([np.mean(_shifted_stream(mode, s)) for s in range(8)])
             for mode in ("off", "uniform", "surprise")}
    assert after["surprise"] < after["uniform"] < after["off"]


def test_the_gated_night_still_recovers_after_a_shift() -> None:
    after = {(mode, gated): np.mean([np.mean(_shifted_stream(mode, s, gated=gated)) for s in range(8)])
             for mode in ("off", "surprise") for gated in (False, True)}
    assert after[("surprise", True)] < after[("off", False)] - 0.1
    assert after[("surprise", True)] < after[("surprise", False)] + 0.1  # the gate opens a day or two late


def _learning_stream(mode: str, seed: int, *, gated: bool) -> float:
    """The night over a base that learns the person itself, exactly: each weekday's rate is drawn
    from Beta(0.5, 0.5) and the base predicts its posterior mean from that weekday's outcomes so
    far (a calibrated online learner, as a trained core is meant to be). The outcomes the night
    replays are ones the base has already counted. The mean log loss served over 42 days."""
    rng, world = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    rate, k, n = world.beta(0.5, 0.5, size=7), np.zeros(7), np.zeros(7)
    adapter, items, counts, losses, gain = Adapter.zero(2), [], {"run": 0}, [], NO_GAIN
    for day in range(42):
        today = []
        for wd in range(7):
            q = (k[wd] + 0.5) / (n[wd] + 1.0)
            base = np.full(3, np.log(q / (1 - q)))
            p = _p(base, served(adapter, gain) if gated else adapter, wd)
            y = float(world.random() < rate[wd])
            losses.append(-np.log(p if y else 1 - p))
            p_fit = _p(base, adapter, wd)
            today.append(Replayed(base, y, -np.log(p_fit if y else 1 - p_fit), "run", day, (0, wd)))
            counts["run"] += 1
            k[wd], n[wd] = k[wd] + y, n[wd] + 1
        items.extend(today)
        gain = update_gain(gain, today)
        adapter = night(adapter, items, today=day, mode=mode, rng=rng, counts=counts)
    return float(np.mean(losses))


def test_the_gate_keeps_the_night_from_counting_twice_what_the_base_has_learned() -> None:
    after = {(mode, gated): np.mean([_learning_stream(mode, s, gated=gated) for s in range(16)])
             for mode in ("off", "uniform", "surprise") for gated in (False, True)}
    off = after[("off", False)]
    assert after[("surprise", False)] > off and after[("uniform", False)] > off
    assert after[("surprise", True)] < off + 1e-3 and after[("uniform", True)] < off + 1e-3


def test_the_gain_is_the_decayed_log_loss_the_adapter_saved() -> None:
    base = np.zeros(3)  # p = 0.5 from the base alone
    good = Replayed(base, 1.0, -np.log(0.8), "run", 0)
    assert np.allclose(update_gain(NO_GAIN, [good]), (np.log(1.6), np.log(1.6) ** 2))
    assert np.allclose(update_gain((1.0, 2.0), []), (0.9, 1.8))
    adapter = Adapter(0.5, np.zeros(1), np.zeros((1, 7)))
    assert paid((3.0, 1.0)) and not paid((1.0, 1.0)) and not paid(NO_GAIN)  # two standard errors
    assert served(adapter, (3.0, 1.0)) is adapter
    assert float(served(adapter, (1.0, 1.0)).shift(np.array([[0, 3]]))[0]) == 0.0


def test_the_predictor_serves_the_adapter_only_while_it_has_paid() -> None:
    trained = trained_tiny()
    life = ToyLife(DEV[1])
    for gate in (True, False):
        pred = core.MindPredictor(trained, night="surprise", gate=gate)
        pred.begin(life.meta)
        for d in range(30):
            pred.observe(d, life.events(d))
            pred.predict(life.queries(d))
            pred.end_day(d)
        pred.observe(30, life.events(30))
        live = pred.live
        bare = replace(live, adapter=Adapter.zero(len(live.adapter.c)))
        same = pred.predict_on(live, life.queries(30))[0] == pred.predict_on(bare, life.queries(30))[0]
        assert live.gain[1] > 0.0 and np.abs(live.adapter.c).sum() > 0
        assert same == (gate and not paid(live.gain))


def test_transfer_adds_each_cells_mean_moved_logit_and_its_subjects_mean_on_other_weekdays() -> None:
    start = Adapter(0.1, np.zeros(3), np.zeros((3, 7)))
    after = transfer(start, np.array([[0, 1], [0, 1], [0, 2], [1, 4]]), np.array([0.2, 0.4, 0.6, -1.0]))

    def shift(slot: int, wd: int) -> float:
        return float(after.shift(np.array([[slot, wd]]))[0])

    assert np.isclose(shift(0, 1), 0.1 + 0.3) and np.isclose(shift(0, 2), 0.1 + 0.6)
    assert np.isclose(shift(0, 5), 0.1 + 0.4)  # a weekday not measured: the subject's mean
    assert np.isclose(shift(1, 4), 0.1 - 1.0) and np.isclose(shift(2, 3), 0.1)  # a subject not measured: none
    assert transfer(start, np.zeros((0, 2), dtype=np.int64), np.zeros(0)) is start


def test_the_transfer_night_moves_fast_memory_into_the_adapter_and_keeps_the_predictions() -> None:
    """A transfer night scales the fast memory by ``1 - move`` and adds what that removed from the
    day's queries' logits to the adapter: the evidence moves, it is not counted twice."""
    trained = trained_tiny()
    life = ToyLife(DEV[2])
    pred = core.MindPredictor(trained, night="transfer", move=0.5)
    pred.begin(life.meta)
    for d in range(20):
        pred.observe(d, life.events(d))
        pred.predict(life.queries(d))
        pred.end_day(d)
    pred.observe(20, life.events(20))
    queries = life.queries(20)
    before = np.array([x.p for x in pred.predict(queries)])
    live = pred.live
    pred.end_day(20)
    after = pred.live
    assert torch.allclose(after.cur.mem, 0.5 * live.cur.mem) and torch.allclose(after.cur.mom, 0.5 * live.cur.mom)
    moved = np.array([x.p for x in pred.predict_on(after, queries)[0]])
    dropped = np.array([x.p for x in pred.predict_on(replace(after, adapter=live.adapter), queries)[0]])
    assert np.abs(dropped - before).mean() > 1e-3  # the memory spoke
    assert np.abs(moved - before).mean() < 0.25 * np.abs(dropped - before).mean()
    with pytest.raises(ValueError):
        core.MindPredictor(trained, night="transfer", move=1.5)


def test_the_night_does_not_forget_stable_habits() -> None:
    trained = trained_tiny()
    loss: dict[str, list[float]] = {"off": [], "surprise": []}
    for seed in DEV:
        life = ToyLife(seed, shift_day=False)
        for mode, out in loss.items():
            rows = _run(core.MindPredictor(trained, night=mode), life)
            out.append(_logloss(np.array([r[2] for r in rows]), np.array([r[4] for r in rows])))
    assert np.mean(loss["surprise"]) < np.mean(loss["off"]) + 0.02


class _Renamed:
    """A life whose every name and token the core has never seen (as in another world)."""

    def __init__(self, life: Life) -> None:
        self.life = life
        m = life.meta
        self.meta = replace(m, activities=tuple(f"zz_{a}" for a in m.activities),
                            people=tuple(f"zz_{p}" for p in m.people), topics=tuple(f"zz_{t}" for t in m.topics))

    def events(self, day: int) -> list[Event]:
        return [replace(e, subject=f"zz_{e.subject}", tokens=tuple(f"zz_{t}" for t in e.tokens))
                for e in self.life.events(day)]

    def queries(self, day: int) -> list[Query]:
        return [replace(q, subject=f"zz_{q.subject}", tokens=tuple(f"zz_{t}" for t in q.tokens))
                for q in self.life.queries(day)]

    def resolve(self, qid: str) -> Resolution:
        return self.life.resolve(qid)


def test_the_prior_is_the_train_lives_base_rate_per_family(tmp_path: Path) -> None:
    lives = [ToyLife(s, n_days=21) for s in range(3)]
    ys: dict[str, list[float]] = {}
    for life in lives:
        for d in range(21):
            for q in life.queries(d):
                ys.setdefault(q.family, []).append(float(life.resolve(q.qid).outcome))
    prior = train_cog.fit_prior(lives)
    assert prior.keys() == ys.keys() and all(np.isclose(prior[k], np.mean(v)) for k, v in ys.items())
    net, _ = _tiny_net(0)
    core.save_checkpoint(tmp_path / "c.pt", net, build_vocab(lives), {"prior": prior})
    assert core.MindPredictor(tmp_path / "c.pt").prior == prior


def test_platt_recovers_a_known_miscalibration() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 2.0, 50_000)
    y = (rng.random(z.shape) < 1 / (1 + np.exp(-(0.8 * z + 0.3)))).astype(float)
    a, b = train_cog.fit_platt(1 / (1 + np.exp(-z)), y)
    assert abs(a - 0.8) < 0.03 and abs(b - 0.3) < 0.03
    with pytest.raises(ValueError):
        train_cog.calibrate(Path("x.pt"), range(9_990, 10_010), Path("y.pt"))  # never on dev seeds


def test_the_predictor_applies_its_platt_map() -> None:
    trained = trained_tiny()
    life = ToyLife(DEV[3])
    plain = np.array([r[2] for r in _run(core.MindPredictor(trained), life)])
    shifted = np.array([r[2] for r in _run(core.MindPredictor(trained, platt=(1.0, 1.0)), life)])
    want = 1 / (1 + np.exp(-(np.log(plain / (1 - plain)) + 1.0)))
    assert np.allclose(shifted, np.clip(want, core.P_FLOOR, 1 - core.P_FLOOR), atol=1e-6)


def test_known_inputs_leave_predictions_alone() -> None:
    trained = trained_tiny()
    life = ToyLife(DEV[2])
    plain = _run(core.MindPredictor(trained, fallback=False), life)
    guarded = _run(core.MindPredictor(trained, fallback=True, prior={"count": 0.3, "hidden": 0.3}), life)
    assert np.allclose([r[2] for r in plain], [r[2] for r in guarded])


def test_unknown_inputs_fall_back_to_the_prior_and_say_so() -> None:
    trained = trained_tiny()
    life = _Renamed(ToyLife(DEV[2]))
    prior = {"count": 0.3, "hidden": 0.4}
    rows = _run(core.MindPredictor(trained, prior=prior), life)  # type: ignore[arg-type]
    fam = [q.family for d in range(life.meta.n_days) for q in life.queries(d)]
    assert np.allclose([r[2] for r in rows], [prior[f] for f in fam])
    h = {k: -(v * np.log(v) + (1 - v) * np.log(1 - v)) for k, v in prior.items()}
    assert np.allclose([r[3] for r in rows], [h[f] for f in fam])  # all of it is epistemic


def test_predict_has_no_side_effects() -> None:
    trained = trained_tiny()
    life = ToyLife(10_005)
    plain = _run(core.MindPredictor(trained), life)
    busy = core.MindPredictor(trained)
    busy.begin(life.meta)
    rows = []
    for d in range(life.meta.n_days):
        busy.observe(d, life.events(d))
        later = [q for dd in range(d, min(d + 5, life.meta.n_days)) for q in life.queries(dd)]
        busy.predict(later)  # an inquirer peeking at many queries, many times
        busy.predict(later)
        for q, pr in zip(life.queries(d), busy.predict(life.queries(d)), strict=True):
            r = life.resolve(q.qid)
            rows.append((d, q.subject, pr.p, pr.u, float(r.outcome), r.oracle_p))
        busy.end_day(d)
    assert rows == plain


def test_the_cli_builds_mind_and_its_lesions_from_a_checkpoint(tmp_path: Path) -> None:
    import pickle

    from anjo.sleep.mind.cog import cli
    from anjo.sleep.mind.cog.rivals import PopulationPrior

    trained = trained_tiny()
    ckpt = tmp_path / "cog_core.pt"
    core.save_checkpoint(ckpt, trained.net, trained.vocab)
    ctx = cli.BuildContext(population="base", train_seeds=(), n_days=56, make_life=ToyLife, out_dir=tmp_path,
                           options={"ckpt": str(ckpt), "lam": "0.5"})
    built = {name: cli.PREDICTORS[name](ctx)() for name in
             ("mind", "mind_nofast", "mind_k1", "mind_nonight", "mind_uniform")}
    assert all(isinstance(p, core.MindPredictor) and p.name == n for n, p in built.items())
    assert not built["mind_nofast"].fast and built["mind_k1"].k == 1
    assert built["mind_nonight"].night_mode == "off" and built["mind_uniform"].night_mode == "uniform"
    pickle.dumps(cli.PREDICTORS["mind"](ctx))
    make_voi = cli.INQUIRERS["mind_voi"](ctx)
    assert make_voi(built["mind"]).lam == 0.5
    with pytest.raises(ValueError):
        make_voi(PopulationPrior.__new__(PopulationPrior))
    with pytest.raises(ValueError):
        cli.PREDICTORS["mind"](cli.BuildContext("base", (), 56, ToyLife, tmp_path, {"ckpt": str(tmp_path / "no.pt")}))


def test_predictions_are_predictions() -> None:
    pred = core.MindPredictor(trained_tiny())
    life = ToyLife(10_004)
    pred.begin(life.meta)
    pred.observe(0, life.events(0))
    out = pred.predict(life.queries(0))
    assert all(isinstance(p, Prediction) for p in out)
    assert [p.qid for p in out] == [q.qid for q in life.queries(0)]
