# mypy: disallow-untyped-decorators=False
"""The cog core (``anjo.sleep.mind.cog.core``) on the toy world (``cog.toy_life``): the fast-weight
rule, shapes and determinism, then a tiny trained core against a per-person counter, its
uncertainty over a life, the fast-memory and ensemble lesions, the night after a shift, and the
self-forecast. One tiny core is trained once per session (``trained_tiny``) and shared with
``test_cog_voi``."""
from __future__ import annotations

import functools
import time

import numpy as np
import pytest
import torch

from anjo.sleep.mind.cog import core, train_cog
from anjo.sleep.mind.cog.encode import build_vocab, collate, encode_life
from anjo.sleep.mind.cog.night import Adapter, Replayed, night, priorities
from anjo.sleep.mind.cog.toy_life import ToyLife
from anjo.sleep.mind.cog.types import Life, Prediction
from anjo.sleep.mind.core.net import fast_update

DEV = range(10_000, 10_024)


@functools.lru_cache(maxsize=1)
def trained_tiny() -> train_cog.Trained:
    """A small core trained for a few seconds on toy lives; shared by this file and test_cog_voi."""
    cfg = train_cog.TrainConfig(lives=384, epochs=6, batch=32, dim=32, hidden=64, tbptt=28, dev_lives=16, lr=6e-3,
                                n_days=56, seed=0, threads=4, patience=3)
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


def _shifted_stream(mode: str, seed: int) -> list[float]:
    """The night alone after a shift, on a base that cannot adapt: the base logits know the old
    habit (run on weekdays 0-2) for good; from day 21 the person runs on weekdays 4-6. Each day one
    run query per weekday is predicted (base plus the adapter), resolves, and the night runs. The
    log losses of days 21-34, the two weeks after the shift."""
    rng = np.random.default_rng(seed)
    adapter, items, counts, losses = Adapter.zero(2), [], {"run": 0}, []
    for day in range(42):
        habit = (0, 1, 2) if day < 21 else (4, 5, 6)
        for wd in range(7):
            base = np.full(3, 1.5 if wd < 3 else -2.0)
            p = float(np.mean(1 / (1 + np.exp(-(base + adapter.shift(np.array([[0, wd]]))[0])))))
            y = float(rng.random() < (0.85 if wd in habit else 0.08))
            loss = -np.log(p if y else 1 - p)
            if 21 <= day < 35:
                losses.append(loss)
            items.append(Replayed(base, y, loss, "run", day, (0, wd)))
            counts["run"] += 1
        adapter = night(adapter, items, today=day, mode=mode, rng=rng, counts=counts)
    return losses


def test_priority_night_recovers_faster_than_no_night_and_uniform_after_a_shift() -> None:
    after = {mode: np.mean([np.mean(_shifted_stream(mode, s)) for s in range(8)])
             for mode in ("off", "uniform", "surprise")}
    assert after["surprise"] < after["uniform"] < after["off"]


def test_the_night_does_not_forget_stable_habits() -> None:
    trained = trained_tiny()
    loss: dict[str, list[float]] = {"off": [], "surprise": []}
    for seed in DEV:
        life = ToyLife(seed, shift_day=False)
        for mode, out in loss.items():
            rows = _run(core.MindPredictor(trained, night=mode), life)
            out.append(_logloss(np.array([r[2] for r in rows]), np.array([r[4] for r in rows])))
    assert np.mean(loss["surprise"]) < np.mean(loss["off"]) + 0.02


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


def test_predictions_are_predictions() -> None:
    pred = core.MindPredictor(trained_tiny())
    life = ToyLife(10_004)
    pred.begin(life.meta)
    pred.observe(0, life.events(0))
    out = pred.predict(life.queries(0))
    assert all(isinstance(p, Prediction) for p in out)
    assert [p.qid for p in out] == [q.qid for q in life.queries(0)]
