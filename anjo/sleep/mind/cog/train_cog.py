"""Meta-training the cog core across many lives (``python -m anjo.sleep.mind.cog.train_cog``).

Each batch is ``batch`` train lives (seeds ``0 .. lives-1``, generated on the fly and kept), each
with a random question asked on ``ask_rate`` of days (its answer joins that day's events, and the
answer head learns it). The core runs the days in order; every ``tbptt`` days the loss of that
window is backpropagated and the state is detached (truncated BPTT). The loss:

- ``query``: log loss of ``p`` (the heads' mean probability) on the day's queries. A share ``1 -
  asking_share`` of lives has no asks (as in the eval without an inquirer: a core that always heard
  answers in training reads their absence wrongly), and with ``soft > 0`` their target is ``soft *
  oracle_p + (1 - soft) * outcome``. The oracle is the Bayes filter given the person's true parameters on the
  passive events, so in a life without asks ``E[oracle_p | history] = P(y | history)``: the same
  minimizer as the outcome with much less variance (distilling privileged information the core
  never sees at test time). Lives with asks keep the outcome, since the oracle never hears answers
  (with soft targets everywhere the toy core learned to ignore them);
- ``u``: how the heads learn to disagree. ``u_mode="oracle"`` (default): the fair CRPS of the K
  heads' logits against the logit of the query's ``oracle_p`` (clipped to +-6), which makes the
  heads draws from the core's belief about what the oracle (the person's true parameters) would
  say. Its MI then estimates the expected excess log loss over the oracle: zero once the person is
  known, large on day 0. ``u_mode="boot"``: the bootstrap ensemble, each head's log loss on a
  Bernoulli(0.5) mask of the queries (and no ``query`` term);
- ``probe``: log loss of tomorrow's per-activity probes (the next-day-event auxiliary, which is also
  what gates the fast memory's writes);
- ``answer``: cross-entropy of the answer head on the asked questions, from the belief before the
  answer; ``kept``: the same from the belief after it, so the belief learns to keep what it heard
  (without it the toy core's predictions hardly moved with an answer: 1e-6 nats of information);
- ``self``: squared error of the self-forecast against the mean log loss of the queries that
  resolve the next day.

Early stopping on the dev seeds' query log loss without asks (``dev``, from ``DEV_SEED``), as the
eval scores ``mind`` alone; ``dev_ask`` is the same lives with random asks. The best weights are kept.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import torch
from torch import Tensor

from anjo.sleep.mind.cog.core import (
    CogConfig, CogNet, CogState, DayInput, mean_prob, mutual_information, param_count, save_checkpoint,
)
from anjo.sleep.mind.cog.encode import Batch, LifeArrays, Vocab, build_vocab, collate, encode_life
from anjo.sleep.mind.cog.types import Life

DEV_SEED: Final = 10_000
DEFAULT_OUT: Final = Path("/private/tmp/claude-501/-Users-k/f8dbbf52-df21-4aab-acb6-f342415fbeb3/scratchpad/mind_research/runs")
WEIGHTS: Final = {"query": 1.0, "u": 0.3, "probe": 0.5, "answer": 0.3, "kept": 0.3, "self": 0.1}
U_LOGIT_CLIP: Final = 6.0

World = Callable[[int, int], Life]  # (seed, n_days) -> Life


def toy_world(seed: int, n_days: int) -> Life:
    from anjo.sleep.mind.cog.toy_life import ToyLife

    return ToyLife(seed, n_days=n_days)


def real_world(seed: int, n_days: int) -> Life:
    from anjo.sleep.mind.cog import world

    return world.make_life(seed, population="base", n_days=n_days)


def indep_world(seed: int, n_days: int) -> Life:
    """The independent world (``world_indep``, written apart from MIND), for protocol 2: trained there."""
    from anjo.sleep.mind.cog import world_indep

    return world_indep.make_life(seed, n_days=n_days)


WORLDS: Final[dict[str, World]] = {"toy": toy_world, "real": real_world, "indep": indep_world}


@dataclass(frozen=True)
class TrainConfig:
    lives: int = 2000
    epochs: int = 20
    batch: int = 32
    dim: int = 64
    hidden: int = 128
    heads: int = 5
    tbptt: int = 28
    dev_lives: int = 200
    n_days: int = 84
    seed: int = 0
    threads: int = 1  # the ops are small: more threads do not help (22.5 vs 22.8 lives/s); run seeds in parallel
    patience: int = 3
    lr: float = 3e-3
    clip: float = 1.0
    ask_rate: float = 0.5
    u_mode: str = "oracle"
    fast: bool = True
    cms: bool = False
    vocab_lives: int = 300
    soft: float = 0.0  # the oracle's share of the query target in lives without asks (``loss``)
    asking_share: float = 0.5  # the share of train lives that get random asks; the rest get none
    tallies: bool = True  # the core keeps the counter's sufficient statistics (``core``, item 6)
    weights: dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))


@dataclass(frozen=True)
class Trained:
    net: CogNet
    vocab: Vocab
    history: list[dict[str, float]]
    config: TrainConfig
    lives_per_sec: float


class Unrolled(NamedTuple):
    """Per day of the unrolled window (B, D_w, ...)."""

    query_logits: Tensor  # (B, Dw, Q, K)
    probe_logits: Tensor  # (B, Dw, A, K) made on each day, about the next
    answer_logits: Tensor  # (B, Dw, O) from the belief before the answer
    kept_logits: Tensor  # (B, Dw, O) from the belief after it: does the belief keep what it heard?
    self_pred: Tensor  # (B, Dw)
    self_target: Tensor  # (B, Dw)
    self_mask: Tensor  # (B, Dw)
    final: CogState


def _day(batch: Batch, d: int) -> DayInput:
    return DayInput(batch.ev_tok[:, d], batch.ev_kind[:, d], batch.ev_val[:, d], batch.ev_subj[:, d],
                    batch.ev_plan[:, d], batch.ev_span[:, d], batch.ev_answer[:, d], batch.ev_mask[:, d])


def _bce(p: Tensor, y: Tensor) -> Tensor:
    return -(y * p.log() + (1 - y) * (1 - p).log())


def run_days(
    net: CogNet, batch: Batch, st: CogState, d0: int, d1: int, acc: tuple[Tensor, Tensor], *,
    fast: bool | None = None, cms: bool | None = None,
) -> tuple[Unrolled, tuple[Tensor, Tensor]]:
    """Days ``d0 .. d1-1`` from state ``st``. ``acc`` sums each life's query log loss per target day
    (detached), for the self-forecast's target."""
    wd0 = batch.start_weekday
    b, o = batch.ask_opt_mask.shape[0], batch.ask_opt_mask.shape[2]
    use_cms = net.cfg.cms if cms is None else cms
    acc_sum, acc_cnt = acc
    ql_all, pl_all, al_all, kept_all, sp_all, st_all, sm_all = [], [], [], [], [], [], []
    for d in range(d0, d1):
        asked = bool(batch.ask_mask[:, d].any())
        out = net.day_step(st, _day(batch, d), d, wd0, batch.act, batch.act_mask, batch.probe_y[:, d],
                           kinds=batch.act_kinds[:, d], fast=fast, cms=cms, want_pre=asked)
        st = out.state
        if out.h_pre is not None:
            ask = (batch.ask_tok[:, d], batch.ask_subj[:, d], batch.ask_opt[:, d], batch.ask_opt_mask[:, d])
            al = net.answer_logits(out.h_pre, *ask)
            kept = net.answer_logits(st.h, *ask)
        else:
            al = kept = torch.zeros(b, o)
        ql, _ = net.query_logits(st, batch.q_tok[:, d], batch.q_subj[:, d], batch.q_target[:, d], d, wd0,
                                 fast=fast, cms=cms)
        with torch.no_grad():
            qm = batch.q_mask[:, d].float()
            ll = _bce(mean_prob(ql), batch.q_y[:, d]) * qm
            acc_sum = acc_sum.scatter_add(1, batch.q_target[:, d], ll)
            acc_cnt = acc_cnt.scatter_add(1, batch.q_target[:, d], qm)
        sp = net.self_forecast(st, d)
        nxt = min(d + 1, acc_sum.shape[1] - 1)
        ql_all.append(ql)
        pl_all.append(out.probe_logits)
        al_all.append(al)
        kept_all.append(kept)
        sp_all.append(sp)
        st_all.append(acc_sum[:, nxt] / acc_cnt[:, nxt].clamp(min=1.0))
        sm_all.append((acc_cnt[:, nxt] > 0).float())
        if use_cms:
            st = net.consolidate(st)
    stack = torch.stack
    return Unrolled(stack(ql_all, 1), stack(pl_all, 1), stack(al_all, 1), stack(kept_all, 1), stack(sp_all, 1),
                    stack(st_all, 1), stack(sm_all, 1), st), (acc_sum, acc_cnt)


def _acc(batch: Batch) -> tuple[Tensor, Tensor]:
    shape = (batch.q_mask.shape[0], batch.q_mask.shape[1] + 1)
    return torch.zeros(shape), torch.zeros(shape)


def unroll(net: CogNet, batch: Batch, **kw: bool | None) -> Unrolled:
    """Whole lives in one pass (no truncation): for evaluation and tests."""
    st = net.initial(batch.act.shape[0], batch.act.shape[1])
    out, _ = run_days(net, batch, st, 0, batch.q_mask.shape[1], _acc(batch), **kw)  # type: ignore[arg-type]
    return out


def fair_crps(x: Tensor, y: Tensor) -> Tensor:
    """The fair CRPS of an ensemble ``x`` (..., K) against ``y`` (...)."""
    k = x.shape[-1]
    spread = (x[..., :, None] - x[..., None, :]).abs().sum(dim=(-1, -2)) / (2 * k * max(k - 1, 1))
    return (x - y[..., None]).abs().mean(dim=-1) - spread


def _masked_mean(v: Tensor, m: Tensor) -> Tensor:
    return (v * m).sum() / m.sum().clamp(min=1.0)


def loss(
    out: Unrolled, batch: Batch, d0: int = 0, *, u_mode: str = "oracle", weights: dict[str, float] | None = None,
    gen: torch.Generator | None = None, soft: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    w = WEIGHTS if weights is None else weights
    dw = out.query_logits.shape[1]
    sl = slice(d0, d0 + dw)
    qm, y = batch.q_mask[:, sl].float(), batch.q_y[:, sl]
    ql = out.query_logits
    passive = (~batch.asking).float()[:, None, None] * soft
    target = y if soft == 0.0 else passive * batch.q_oracle[:, sl] + (1.0 - passive) * y
    query = _masked_mean(_bce(mean_prob(ql), target), qm)
    if u_mode == "oracle":
        tgt = torch.logit(batch.q_oracle[:, sl].clamp(1e-6, 1 - 1e-6)).clamp(-U_LOGIT_CLIP, U_LOGIT_CLIP)
        u = _masked_mean(fair_crps(ql, tgt), qm)
    else:
        boot = torch.bernoulli(torch.full(ql.shape, 0.5), generator=gen)
        pk = torch.sigmoid(ql).clamp(1e-4, 1 - 1e-4)
        u = _masked_mean(_bce(pk, y[..., None]), boot * qm[..., None])
    days = torch.arange(d0, d0 + dw)
    valid_next = (days[None, :] + 1 < batch.n_days[:, None]).float()
    nxt = torch.clamp(days + 1, max=batch.probe_y.shape[1] - 1)
    py = batch.probe_y[:, nxt]
    pm = valid_next[:, :, None] * batch.act_mask[:, None, :].float()
    probe = _masked_mean(_bce(mean_prob(out.probe_logits), py), pm)
    am = batch.ask_mask[:, sl].float()
    logp = torch.log_softmax(out.answer_logits, dim=-1)
    ans_ll = -logp.gather(-1, batch.ask_answer[:, sl, None])[..., 0]
    answer = _masked_mean(ans_ll, am)
    kept_ll = -torch.log_softmax(out.kept_logits, dim=-1).gather(-1, batch.ask_answer[:, sl, None])[..., 0]
    kept = _masked_mean(kept_ll, am)
    selfl = _masked_mean((out.self_pred - out.self_target).pow(2), out.self_mask)
    parts = {"query": query, "u": u, "probe": probe, "answer": answer, "kept": kept, "self": selfl}
    q_w = w["query"] if u_mode == "oracle" else 0.0
    total = (q_w * query + w["u"] * u + w["probe"] * probe + w["answer"] * answer + w["kept"] * kept
             + w["self"] * selfl)
    if u_mode == "boot":
        total = total + (1.0 - w["u"]) * u  # the bootstrap heads' loss is the main term
    return total, {k: float(v.detach()) for k, v in parts.items()}


def _detach(st: CogState) -> CogState:
    return CogState(*(t.detach() for t in st))


# ── training ─────────────────────────────────────────────────────────────────


def _encode(
    vocab: Vocab, lives: Sequence[Life], rng: np.random.Generator, ask_rate: float, asking_share: float = 1.0,
) -> list[LifeArrays]:
    return [encode_life(vocab, life, rng, ask_rate, asking=asking_share >= 1.0 or bool(rng.random() < asking_share))
            for life in lives]


def fit_prior(lives: Sequence[Life]) -> dict[str, float]:
    """The train lives' base rate per query family: what the predictor falls back to on unknown inputs."""
    ys: dict[str, list[float]] = {}
    for life in lives:
        for d in range(life.meta.n_days):
            for q in life.queries(d):
                ys.setdefault(q.family, []).append(float(life.resolve(q.qid).outcome))
    return {k: float(np.clip(np.mean(v), 1e-3, 1 - 1e-3)) for k, v in ys.items()}


def fit_platt(p: np.ndarray, y: np.ndarray, iters: int = 100) -> tuple[float, float]:
    """``(a, b)`` minimizing the log loss of ``sigmoid(a logit(p) + b)`` against ``y`` (Newton steps)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    z, a, b = np.log(p / (1 - p)), 1.0, 0.0
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(a * z + b)))
        w = q * (1 - q)
        hess = np.array([[np.mean(w * z * z), np.mean(w * z)], [np.mean(w * z), np.mean(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, np.array([np.mean((q - y) * z), np.mean(q - y)]))
        a, b = a - float(step[0]), b - float(step[1])
        if np.abs(step).max() < 1e-10:
            break
    return a, b


def calibrate(ckpt: Path, seeds: Sequence[int], out: Path, world: World = real_world, n_days: int = 84) -> tuple[float, float]:
    """Fit the predictor's Platt map on train-range lives without asks (never dev seeds) and save
    the checkpoint with it as ``out``."""
    if any(s >= DEV_SEED for s in seeds):
        raise ValueError(f"calibrate on train seeds (< {DEV_SEED}), never on dev")
    from anjo.sleep.mind.cog.core import load_checkpoint

    ck = load_checkpoint(ckpt)
    passive, _ = dev_batches(ck.vocab, [world(s, n_days) for s in seeds], 32, 0.0)
    ps, ys = [], []
    with torch.no_grad():
        for b in passive:
            m = b.q_mask
            ps.append(mean_prob(unroll(ck.net, b).query_logits)[m].numpy())
            ys.append(b.q_y[m].numpy())
    platt = fit_platt(np.concatenate(ps), np.concatenate(ys))
    save_checkpoint(out, ck.net, ck.vocab, {**dict(ck.extra), "platt": list(platt),
                                            "platt_seeds": [int(seeds[0]), int(seeds[-1])]})
    return platt


def asking_share(cfg: TrainConfig) -> float:
    """The share of train lives with random asks (the rest hear no answer, whatever the target)."""
    return cfg.asking_share


def dev_batches(
    vocab: Vocab, lives: Sequence[Life], batch: int, ask_rate: float,
) -> tuple[list[Batch], list[Batch]]:
    """The dev lives twice: without asks (what early stopping reads) and with random asks."""
    rng = np.random.default_rng(DEV_SEED)
    chunks = [lives[i:i + batch] for i in range(0, len(lives), batch)]
    passive = [collate([encode_life(vocab, life, rng, 0.0, asking=False) for life in c]) for c in chunks]
    asking = [collate([encode_life(vocab, life, rng, ask_rate, asking=True) for life in c]) for c in chunks]
    return passive, asking


def _dev_metrics(net: CogNet, passive: Sequence[Batch], asking: Sequence[Batch], u_mode: str) -> dict[str, float]:
    return {**evaluate(net, passive, u_mode), "dev_ask": evaluate(net, asking, u_mode)["dev"]}


def evaluate(net: CogNet, batches: Sequence[Batch], u_mode: str) -> dict[str, float]:
    """Dev metrics: query log loss, the oracle's on the same queries, the u-vs-excess rank correlation."""
    net.eval()
    ll, oracle, us, excess = [], [], [], []
    with torch.no_grad():
        for batch in batches:
            out = unroll(net, batch)
            m = batch.q_mask
            p = mean_prob(out.query_logits)[m]
            y, po = batch.q_y[m], batch.q_oracle[m].clamp(1e-4, 1 - 1e-4)
            ll.append(_bce(p, y))
            oracle.append(_bce(po, y))
            us.append(mutual_information(out.query_logits)[m])
            excess.append(po * (po / p).log() + (1 - po) * ((1 - po) / (1 - p)).log())
    net.train()
    lls, us_all, ex = torch.cat(ll), torch.cat(us), torch.cat(excess)
    return {"dev": float(lls.mean()), "dev_oracle": float(torch.cat(oracle).mean()),
            "dev_u_mean": float(us_all.mean()), "dev_kl_oracle": float(ex.mean()),
            "dev_u_rank": _spearman(us_all.numpy(), ex.numpy())}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def train(cfg: TrainConfig, world: World = toy_world, out: Path | None = None, log: Callable[[str], None] | None = None) -> Trained:
    if not 0 < cfg.lives <= DEV_SEED:
        raise ValueError(f"train lives are seeds 0..{DEV_SEED - 1}")
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    cache: dict[int, Life] = {}

    def life(seed: int) -> Life:
        if seed not in cache:
            cache[seed] = world(seed, cfg.n_days)
        return cache[seed]

    vocab = build_vocab(life(s) for s in range(min(cfg.vocab_lives, cfg.lives)))
    prior = fit_prior([life(s) for s in range(min(cfg.vocab_lives, cfg.lives))])
    net = CogNet(CogConfig(vocab_size=len(vocab), dim=cfg.dim, hidden=cfg.hidden, heads=cfg.heads, fast=cfg.fast,
                           cms=cfg.cms, day_scale=float(cfg.n_days), u_mode=cfg.u_mode, tallies=cfg.tallies))
    dev_lives = [world(DEV_SEED + i, cfg.n_days) for i in range(cfg.dev_lives)]
    dev, dev_ask = dev_batches(vocab, dev_lives, cfg.batch, cfg.ask_rate)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    history: list[dict[str, float]] = [{"epoch": 0.0, **_dev_metrics(net, dev, dev_ask, cfg.u_mode)}]
    best, best_state, stale = history[0]["dev"], {k: v.clone() for k, v in net.state_dict().items()}, 0
    seen, t_train = 0, 0.0
    say = log or (lambda _msg: None)
    say(f"params {param_count(net)} vocab {len(vocab)} start {history[0]}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(cfg.lives)
        parts_sum: dict[str, float] = {}
        n_steps = 0
        for start in range(0, len(order), cfg.batch):
            seeds = order[start:start + cfg.batch]
            batch = collate(_encode(vocab, [life(int(s)) for s in seeds], rng, cfg.ask_rate, asking_share(cfg)))
            st = net.initial(batch.act.shape[0], batch.act.shape[1])
            acc = _acc(batch)
            dmax = batch.q_mask.shape[1]
            for d0 in range(0, dmax, cfg.tbptt):
                d1 = min(d0 + cfg.tbptt, dmax)
                window, acc = run_days(net, batch, st, d0, d1, acc)
                total, parts = loss(window, batch, d0, u_mode=cfg.u_mode, weights=cfg.weights, gen=gen, soft=cfg.soft)
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
                opt.step()
                st = _detach(window.final)
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v
                n_steps += 1
            seen += len(seeds)
        t_train += time.perf_counter() - t0
        row = {"epoch": float(epoch), **{f"train_{k}": v / max(n_steps, 1) for k, v in parts_sum.items()},
               **_dev_metrics(net, dev, dev_ask, cfg.u_mode), "lives_per_sec": seen / max(t_train, 1e-9)}
        history.append(row)
        say(json.dumps({k: round(v, 4) for k, v in row.items()}))
        if row["dev"] < best - 1e-4:
            best, best_state, stale = row["dev"], {k: v.clone() for k, v in net.state_dict().items()}, 0
            if out is not None:  # the best so far is on disk, whatever happens to the run
                save_checkpoint(out / "cog_core.pt", net, vocab, {"train": asdict(cfg), "history": history,
                                                                  "prior": prior})
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    trained = Trained(net, vocab, history, cfg, seen / max(t_train, 1e-9))
    if out is not None:
        save_checkpoint(out / "cog_core.pt", net, vocab, {"train": asdict(cfg), "history": history, "prior": prior})
        (out / "history.json").write_text(json.dumps({"config": asdict(cfg), "history": history,
                                                      "lives_per_sec": trained.lives_per_sec}, indent=1))
    return trained


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m anjo.sleep.mind.cog.train_cog", description=__doc__.splitlines()[0])
    ap.add_argument("--lives", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "cog_core")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", choices=sorted(WORLDS), default="real")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--tbptt", type=int, default=28)
    ap.add_argument("--dev-lives", type=int, default=200)
    ap.add_argument("--n-days", type=int, default=84)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--u-mode", choices=("oracle", "boot"), default="oracle")
    ap.add_argument("--soft", type=float, default=0.0, help="the oracle's share of the query target")
    ap.add_argument("--asking-share", type=float, default=0.5, help="the share of train lives with random asks")
    ap.add_argument("--no-fast", action="store_true", help="train without the fast memory (the trained lesion)")
    ap.add_argument("--cms", action="store_true", help="train with the continuum memory's night level")
    ap.add_argument("--no-tallies", action="store_true", help="train without the tallies (the old core)")
    ap.add_argument("--calibrate", metavar="FIRST-LAST", help="no training: fit the Platt map of OUT/cog_core.pt on "
                    "these train seeds and write OUT/cog_core_platt.pt")
    a = ap.parse_args(argv)
    if a.calibrate:
        first, last = (int(x) for x in a.calibrate.split("-"))
        platt = calibrate(a.out / "cog_core.pt", range(first, last + 1), a.out / "cog_core_platt.pt",
                          WORLDS[a.world], a.n_days)
        print(json.dumps({"platt": platt, "out": str(a.out / "cog_core_platt.pt")}))  # noqa: T201
        return 0
    cfg = TrainConfig(lives=a.lives, epochs=a.epochs, batch=a.batch, dim=a.dim, hidden=a.hidden, tbptt=a.tbptt,
                      dev_lives=a.dev_lives, n_days=a.n_days, seed=a.seed, threads=a.threads, patience=a.patience,
                      lr=a.lr, u_mode=a.u_mode, fast=not a.no_fast, cms=a.cms, soft=a.soft,
                      asking_share=a.asking_share, tallies=not a.no_tallies)
    a.out.mkdir(parents=True, exist_ok=True)
    trained = train(cfg, WORLDS[a.world], a.out, log=lambda m: print(m, flush=True))  # noqa: T201
    print(json.dumps({"out": str(a.out), "lives_per_sec": round(trained.lives_per_sec, 2),  # noqa: T201
                      "best_dev": min(h["dev"] for h in trained.history),
                      "params": param_count(trained.net)}))
    return 0 if math.isfinite(trained.lives_per_sec) else 1


if __name__ == "__main__":
    raise SystemExit(main())
