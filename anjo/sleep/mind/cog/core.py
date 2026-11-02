"""MIND's cog core: a learned state that expects, doubts, asks and sleeps (see DESIGN.md).

**One day, one step** (``CogNet.day_step``; every tensor leads with B, the lives):

1. **Events.** Each event is the mean of its token embeddings (one table over the train
   vocabulary, unknown tokens share ``UNK``) plus its kind and value features, normalized. The day's
   events are pooled into one vector by four attention heads; a day with none gets a learned vector.
2. **The fast memory learns from surprise** (Titans, as ``core.net.fast_update``, with one key per
   activity). Yesterday the core predicted, for each activity, whether it would happen today (the
   probes: the query path with a learned probe vector for tokens). Today's events give the outcome
   ``y`` (``encode.outcomes``). The write's key is the probe's key, its value ``W_v [y, 1 - y]``, and
   its strength is gated by the core's surprise, the probe's log loss ``s = -log p(y)``:
   ``theta_a = theta_max * sigmoid(g_w * s_a + g_b)`` (``g_w`` starts at 2, ``g_b`` at -2, both
   learned). Then ``S = eta S - sum_a theta_a grad_a`` and ``M = (1 - alpha) M + S``, the gradient of
   ``|M k_a - v_a|^2`` (``multi_fast_update``), with learned scalar momentum ``eta`` and forgetting ``alpha``.
   With the continuum memory (``cms``) a night level is read beside it; each write adds its rated
   step to the level's pending sum, applied at the night (``consolidate``; ``core.cms`` eq. 71).
3. **The belief.** A GRU over days reads the pooled day, the weekday and the day index.
4. **The episodic buffer.** The last ``window`` days' events stay as keys and values for the query
   to attend (four heads), with structured match features: same subject, an event pointing at the
   query's target day (a plan, a cancel, an answer's ``in=k``), the same weekday, age, the products
   of the subject match with the pointing and weekday ones, and the target inside an event's span
   (``left=k``).
5. **Tomorrow's probes** are predicted from the new state, for the next day's write.
6. **The tallies** (``tallies``): the counter's sufficient statistics, kept exactly: per activity
   and weekday, the ``did``, ``skipped``, ``cancel`` and ``plan`` events so far, the days of that
   weekday so far, and a recency-weighted rate of ``did`` on them (``TALLY_DECAY`` per occurrence).
   A query reads its subject's cell for its target weekday and the subject's totals, as counts,
   smoothed rates and the recent rate (``TALLY_FEATURES``); the meta-trained trunk does the
   shrinkage toward the population that a hand-built counter fixes by a prior.

**A query** reads its tokens, its subject, its target weekday and horizon, the belief, the fast
memory (keyed by subject, target weekday and belief: ``key_proj``) and the episodic buffer; a trunk
of two layers feeds **K heads**. ``p`` is the mean of the heads' probabilities. ``u`` is the
ensemble's mutual information ``H(mean p_k) - mean H(p_k)`` in nats (BALD). How the heads are
trained to disagree is the training's ``u_mode`` (``train_cog``): by default each head is a draw
from the core's belief about the oracle's prediction (fair CRPS of their logits against the
oracle's), so their MI estimates the expected excess log loss over the oracle.

**The answer head** gives a distribution over a question's options from the belief before the
answer; **the self head** predicts the core's own mean log loss on tomorrow's resolving queries.
``MindPredictor`` runs one person through the ``Predictor`` protocol with the night (``night``).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from anjo.sleep.mind.cog import night as night_mod
from anjo.sleep.mind.cog.encode import (
    KIND_INDEX, NO_PLAN, PAD, UNK, Vocab, activity_kinds, encode_day, encode_queries, encode_question, outcomes,
)
from anjo.sleep.mind.cog.types import Event, LifeMeta, Prediction, Query, Question

_EPS: Final = 1e-6
P_FLOOR: Final = 1e-4  # predictions are kept in [P_FLOOR, 1 - P_FLOOR]
POOL_HEADS: Final = 4
EPI_HEADS: Final = 4
N_FLAGS: Final = 7
N_KINDS: Final = 4  # encode.TALLY_KINDS: did, skipped, cancel, plan
TALLY_FEATURES: Final = 15
TALLY_DECAY: Final = 0.7
U_MODES: Final = ("oracle", "boot")


@dataclass(frozen=True)
class CogConfig:
    vocab_size: int
    dim: int = 64
    hidden: int = 128
    heads: int = 5
    window: int = 14  # days the episodic buffer keeps
    theta_max: float = 0.5  # 2 theta <= 1 per unit key: one write never overshoots
    fast: bool = True
    cms: bool = False
    day_scale: float = 84.0
    u_mode: str = "oracle"
    tallies: bool = False

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.dim, self.hidden, self.heads, self.window) < 1:
            raise ValueError("sizes must be at least 1")
        if not 0.0 < self.theta_max <= 0.5:
            raise ValueError("theta_max is in (0, 0.5]")
        if self.u_mode not in U_MODES:
            raise ValueError(f"u_mode is one of {U_MODES}")
        if self.cms and not self.fast:
            raise ValueError("the continuum memory builds on the fast memory")


def multi_fast_update(
    memory: Tensor, momentum: Tensor, keys: Tensor, values: Tensor, theta: Tensor, eta: Tensor, alpha: Tensor,
    slow: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """``core.net.fast_update`` with A keys at once (B, A, d), each with its own step ``theta`` (B, A):
    the gradient of ``sum_a theta_a |M k_a - v_a|^2`` taken at the old ``M``, one momentum and one
    forget step. With A = 1 it is ``fast_update`` exactly. Returns ``(memory, momentum, surprise)``,
    the surprise per key ``mean((M k - v)^2)``."""
    read = torch.einsum("bij,baj->bai", memory if slow is None else memory + slow, keys)
    err = read - values
    grad = 2.0 * torch.einsum("ba,bai,baj->bij", theta, err, keys)
    momentum = eta[:, None, None] * momentum - grad
    memory = (1.0 - alpha[:, None, None]) * memory + momentum
    return memory, momentum, err.pow(2).mean(dim=-1)


class CogState(NamedTuple):
    """One batch of per-person state (B leads). Pure: every step returns a new one."""

    h: Tensor  # (B, d) the belief
    mem: Tensor  # (B, d, d) the fast weights
    mom: Tensor  # (B, d, d) their momentum
    levels: Tensor  # (B, 2, d, d) the night level and its pending sum; (B, 0, d, d) without cms
    probe_logits: Tensor  # (B, A, K) yesterday's prediction of each activity today
    probe_key: Tensor  # (B, A, d) the keys those predictions read
    probe_ok: Tensor  # (B, A) 1 where a prediction was made
    epi_key: Tensor  # (B, N, d)
    epi_val: Tensor  # (B, N, d)
    epi_day: Tensor  # (B, N) long
    epi_wd: Tensor  # (B, N) long
    epi_subj: Tensor  # (B, N) long
    epi_plan: Tensor  # (B, N) float, NO_PLAN where none
    epi_span: Tensor  # (B, N) float, NO_PLAN where none
    epi_mask: Tensor  # (B, N) bool
    act: Tensor  # (B, A) the activities' ids (set by the first day)
    tally: Tensor  # (B, A, 7, 4) did / skipped / cancel / plan events per activity and weekday
    tdays: Tensor  # (B, 7) days of each weekday so far
    tema: Tensor  # (B, A, 7, 2) the recent rate of did per cell: decayed sum of outcomes, of days


class DayInput(NamedTuple):
    tok: Tensor  # (B, E, T)
    kind: Tensor  # (B, E)
    val: Tensor  # (B, E, 3)
    subj: Tensor  # (B, E)
    plan: Tensor  # (B, E)
    span: Tensor  # (B, E)
    answer: Tensor  # (B, E) bool
    mask: Tensor  # (B, E) bool


class StepOut(NamedTuple):
    state: CogState
    probe_logits: Tensor  # (B, A, K) tomorrow's probes, with their graph
    h_pre: Tensor | None  # (B, d) the belief from today's events without the answers
    surprise: Tensor  # (B, A) the probe log loss that gated today's write


def mutual_information(logits: Tensor) -> Tensor:
    """BALD over the last axis (K heads): ``H(mean p) - mean H(p)``, nats, >= 0."""
    p = torch.sigmoid(logits).clamp(P_FLOOR, 1 - P_FLOOR)
    pbar = p.mean(dim=-1)

    def ent(q: Tensor) -> Tensor:
        return -(q * q.log() + (1 - q) * (1 - q).log())

    return (ent(pbar) - ent(p).mean(dim=-1)).clamp(min=0.0)


def mean_prob(logits: Tensor) -> Tensor:
    return torch.sigmoid(logits).mean(dim=-1).clamp(P_FLOOR, 1 - P_FLOOR)


class CogNet(nn.Module):  # type: ignore[misc]  # torch's types are skipped (follow_imports)
    def __init__(self, cfg: CogConfig) -> None:
        super().__init__()
        d, hdim = cfg.dim, cfg.hidden
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, d, padding_idx=PAD)
        self.kind = nn.Embedding(len(KIND_INDEX), d)
        self.val = nn.Linear(3, d)
        self.ev_norm = nn.LayerNorm(d)
        self.pool_score = nn.Linear(d, POOL_HEADS)
        self.pool_out = nn.Linear(POOL_HEADS * d, d)
        self.empty_day = nn.Parameter(torch.zeros(d))
        self.gru = nn.GRUCell(d + 7 + 3, d)
        self.h0 = nn.Parameter(torch.zeros(d))
        # the fast memory
        self.key_proj = nn.Linear(d + 7 + d, d)
        self.value = nn.Linear(2, d, bias=False)
        self.gate = nn.Parameter(torch.tensor([2.0, -2.0]))  # g_w, g_b
        self.eta_alpha = nn.Parameter(torch.tensor([-2.0, -4.0]))  # momentum and forgetting, as logits
        self.read_proj = nn.Linear(d, d)
        # the episodic buffer
        self.epi_k = nn.Linear(d, d, bias=False)
        self.epi_v = nn.Linear(d, d, bias=False)
        self.epi_q = nn.Linear(d + 7 + 2, d)
        self.epi_flag_score = nn.Linear(N_FLAGS, EPI_HEADS)
        self.epi_flag_val = nn.Linear(N_FLAGS, d, bias=False)
        self.epi_null = nn.Parameter(torch.zeros(EPI_HEADS))
        self.epi_out = nn.Linear(EPI_HEADS * d, d)
        # queries
        self.probe_vec = nn.Parameter(torch.zeros(d))
        self.trunk = nn.Sequential(nn.Linear(5 * d + 7 + 2 + 2, hdim), nn.GELU(), nn.Linear(hdim, hdim), nn.GELU())
        self.heads = nn.Linear(hdim, cfg.heads)
        # answers and the self-forecast
        self.ans_ctx = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ans_opt = nn.Linear(d, d)
        self.self_head = nn.Linear(d + 4, 1)
        if cfg.tallies:
            self.tally_proj = nn.Sequential(nn.Linear(TALLY_FEATURES, d), nn.GELU(), nn.Linear(d, d))
            self.trunk[0] = nn.Linear(6 * d + 7 + 2 + 2, hdim)
        if cfg.cms:  # built last, so a core without it draws the same weights
            self.cms_rate = nn.Parameter(torch.tensor(-1.0))  # the night level's rate, a share of each write
            self.cms_mix = nn.Parameter(torch.tensor(1.0))  # its weight in the read

    # ── state ────────────────────────────────────────────────────────────────

    def initial(self, batch: int, n_act: int) -> CogState:
        d, k = self.cfg.dim, self.cfg.heads
        z = torch.zeros
        return CogState(
            h=self.h0.expand(batch, d).clone(), mem=z(batch, d, d), mom=z(batch, d, d),
            levels=z(batch, 2 if self.cfg.cms else 0, d, d), probe_logits=z(batch, n_act, k),
            probe_key=z(batch, n_act, d), probe_ok=z(batch, n_act), epi_key=z(batch, 0, d), epi_val=z(batch, 0, d),
            epi_day=z(batch, 0, dtype=torch.long), epi_wd=z(batch, 0, dtype=torch.long),
            epi_subj=z(batch, 0, dtype=torch.long), epi_plan=z(batch, 0), epi_span=z(batch, 0),
            epi_mask=z(batch, 0, dtype=torch.bool), act=z(batch, n_act, dtype=torch.long),
            tally=z(batch, n_act, 7, N_KINDS), tdays=z(batch, 7), tema=z(batch, n_act, 7, 2),
        )

    def count(self, st: CogState, kinds: Tensor, y: Tensor, weekday: Tensor) -> CogState:
        """Today's events into the tallies: ``kinds`` (B, A, 4) and outcomes ``y`` (B, A) on ``weekday`` (B,)."""
        wd = F.one_hot(weekday, 7).to(y.dtype)  # (B, 7)
        cell = wd[:, None, :, None]  # (B, 1, 7, 1)
        tema = st.tema * (1.0 - cell * (1.0 - TALLY_DECAY)) + cell * torch.stack([y, torch.ones_like(y)], -1)[:, :, None]
        return st._replace(tally=st.tally + cell * kinds[:, :, None, :], tdays=st.tdays + wd, tema=tema)

    def tally_features(self, st: CogState, subj: Tensor, wd_idx: Tensor) -> Tensor:
        """(B, Q, TALLY_FEATURES) for queries about ``subj`` (B, Q) on weekdays ``wd_idx`` (B, Q)."""
        match = ((st.act[:, None, :] == subj[:, :, None]) & (subj[:, :, None] != UNK)).to(st.tally.dtype)
        def pick(x: Tensor) -> Tensor:  # (B, A, 7, F) -> the query's subject and weekday (B, Q, F)
            w = F.one_hot(wd_idx, 7).to(x.dtype)
            return torch.einsum("bqa,bqw,bawf->bqf", match, w, x)
        cell, days = pick(st.tally), torch.einsum("bqw,bw->bq", F.one_hot(wd_idx, 7).to(st.tdays.dtype), st.tdays)
        tot = torch.einsum("bqa,baf->bqf", match, st.tally.sum(dim=2))
        all_days = st.tdays.sum(dim=1, keepdim=True).expand_as(days)
        ema = pick(st.tema)
        seen_cell, seen_tot = cell[..., :3].sum(-1), tot[..., :3].sum(-1)
        f = [torch.log1p(cell) / 3.0, torch.log1p(days)[..., None] / 3.0,
             ((cell[..., 0] + 0.5) / (days + 1.0))[..., None], ((cell[..., 0] + 0.5) / (seen_cell + 1.0))[..., None],
             torch.log1p(tot) / 4.0, ((tot[..., 0] + 0.5) / (all_days + 1.0))[..., None],
             ((tot[..., 0] + 0.5) / (seen_tot + 1.0))[..., None], ((ema[..., 0] + 0.5) / (ema[..., 1] + 1.0))[..., None],
             match.sum(-1, keepdim=True)]
        return torch.cat(f, dim=-1)

    # ── pieces ───────────────────────────────────────────────────────────────

    def bag(self, tok: Tensor) -> Tensor:
        """Mean token embedding over the last axis's non-PAD tokens (zeros when none)."""
        mask = (tok != PAD).to(self.tok.weight.dtype)
        return (self.tok(tok) * mask[..., None]).sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

    def encode_events(self, ev: DayInput) -> Tensor:
        return self.ev_norm(self.bag(ev.tok) + self.kind(ev.kind) + self.val(ev.val))

    def pool(self, e: Tensor, mask: Tensor) -> Tensor:
        """The day's events (B, E, d) to one vector (B, d), four attention heads; none: ``empty_day``."""
        score = self.pool_score(e).masked_fill(~mask[..., None], -1e9)
        w = torch.softmax(score, dim=1) * mask[..., None]
        pooled = self.pool_out(torch.einsum("bep,bed->bpd", w, e).flatten(1))
        any_ev = mask.any(dim=1, keepdim=True).to(e.dtype)
        return any_ev * pooled + (1.0 - any_ev) * self.empty_day

    def day_features(self, day: int, batch: int) -> Tensor:
        s = self.cfg.day_scale
        f = torch.tensor([min(day / s, 2.0), math.log1p(max(day, 0)) / math.log1p(s)])
        return f.expand(batch, 2)

    def eta_alpha_now(self, batch: int) -> tuple[Tensor, Tensor]:
        ea = torch.sigmoid(self.eta_alpha)
        return ea[0].expand(batch), ea[1].expand(batch)

    def memory_read(self, st: CogState, key: Tensor, fast: bool, cms: bool) -> Tensor:
        """``(M_fast + w M_night) k`` for keys (B, Q, d); zeros with the fast memory off."""
        if not fast:
            return torch.zeros_like(key)
        m = st.mem
        if cms and st.levels.shape[1]:
            m = m + self.cms_mix * st.levels[:, 0]
        return torch.einsum("bij,bqj->bqi", m, key)

    def _write(self, st: CogState, y: Tensor, act_mask: Tensor, fast: bool, cms: bool) -> tuple[CogState, Tensor]:
        """Today's outcomes ``y`` (B, A) into the fast memory, gated by yesterday's probe surprise."""
        p = mean_prob(st.probe_logits.detach())
        surprise = -(y * p.log() + (1 - y) * (1 - p).log())
        active = st.probe_ok * act_mask.to(y.dtype)
        if not fast:
            return st, surprise * active
        g = self.gate
        theta = self.cfg.theta_max * torch.sigmoid(g[0] * surprise + g[1]) * active
        v = self.value(torch.stack([y, 1.0 - y], dim=-1))
        eta, alpha = self.eta_alpha_now(y.shape[0])
        use_cms = cms and st.levels.shape[1] > 0
        slow = self.cms_mix * st.levels[:, 0] if use_cms else None
        mem, mom, _ = multi_fast_update(st.mem, st.mom, st.probe_key, v, theta, eta, alpha, slow)
        levels = st.levels
        if use_cms:
            read = torch.einsum("bij,baj->bai", st.mem + slow, st.probe_key)  # type: ignore[operator]
            grad = 2.0 * torch.einsum("ba,bai,baj->bij", theta, read - v, st.probe_key)
            rate = torch.sigmoid(self.cms_rate) * self.cms_mix
            levels = torch.stack([st.levels[:, 0], st.levels[:, 1] + rate * grad], dim=1)
        return st._replace(mem=mem, mom=mom, levels=levels), surprise * active

    def consolidate(self, st: CogState) -> CogState:
        """The night under the continuum memory: the night level takes its pending step (cms eq. 71)."""
        if not st.levels.shape[1]:
            return st
        lv = st.levels
        return st._replace(levels=torch.stack([lv[:, 0] - lv[:, 1], torch.zeros_like(lv[:, 1])], dim=1))

    def _append(self, st: CogState, e: Tensor, ev: DayInput, day: int, weekday: Tensor) -> CogState:
        b, n = e.shape[:2]
        keep_from = day - self.cfg.window + 1
        cat = torch.cat
        st = st._replace(
            epi_key=cat([st.epi_key, self.epi_k(e)], 1), epi_val=cat([st.epi_val, self.epi_v(e)], 1),
            epi_day=cat([st.epi_day, torch.full((b, n), day, dtype=torch.long)], 1),
            epi_wd=cat([st.epi_wd, weekday[:, None].expand(b, n)], 1), epi_subj=cat([st.epi_subj, ev.subj], 1),
            epi_plan=cat([st.epi_plan, ev.plan], 1), epi_span=cat([st.epi_span, ev.span], 1),
            epi_mask=cat([st.epi_mask, ev.mask], 1),
        )
        stale = ((st.epi_day < keep_from) | ~st.epi_mask).all(dim=0)
        drop = int(torch.cumprod(stale.to(torch.long), dim=0).sum())
        if drop:
            st = st._replace(**{f: getattr(st, f)[:, drop:] for f in CogState._fields if f.startswith("epi_")})
        return st

    def episodic(self, st: CogState, qv_in: Tensor, subj: Tensor, target: Tensor, wd_t: Tensor) -> Tensor:
        """Attention of the queries (B, Q) over the buffer, with the structured match features."""
        b, q = subj.shape
        d, nh = self.cfg.dim, EPI_HEADS
        n = st.epi_key.shape[1]
        if n == 0:
            return self.epi_out(torch.zeros(b, q, nh * d))
        dh = d // nh
        qv = self.epi_q(qv_in).view(b, q, nh, dh)
        kv = st.epi_key[:, :, : nh * dh].reshape(b, n, nh, dh)
        score = torch.einsum("bqhc,bnhc->bqhn", qv, kv) / math.sqrt(dh)
        same = ((st.epi_subj[:, None, :] == subj[:, :, None]) & (subj[:, :, None] != UNK)).float()
        match = (st.epi_plan[:, None, :] == target[:, :, None].float()).float()
        same_wd = (st.epi_wd[:, None, :] == wd_t[:, :, None]).float()
        age = (target[:, :, None] - st.epi_day[:, None, :]).float() / self.cfg.window
        tf = target[:, :, None].float()
        within = ((st.epi_span[:, None, :] >= tf) & (st.epi_day[:, None, :].float() <= tf)).float()
        flags = torch.stack([same, match, same_wd, age, same * same_wd, same * match, within], dim=-1)  # (B, Q, N, F)
        score = score + self.epi_flag_score(flags).permute(0, 1, 3, 2)
        score = score.masked_fill(~st.epi_mask[:, None, None, :], -1e9)
        null = self.epi_null.view(1, 1, nh, 1).expand(b, q, nh, 1)
        w = torch.softmax(torch.cat([score, null], dim=-1), dim=-1)[..., :n]
        out = torch.einsum("bqhn,bnd->bqhd", w, st.epi_val)
        out = out + self.epi_flag_val(torch.einsum("bqhn,bqnf->bqhf", w, flags))
        return self.epi_out(out.flatten(2))

    def features(
        self, st: CogState, tok: Tensor, subj: Tensor, target: Tensor, day: int, weekday0: Tensor, *,
        probe: bool = False, fast: bool | None = None, cms: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The trunk features (B, Q, H) of queries, and the memory keys they read (B, Q, d)."""
        fast = self.cfg.fast if fast is None else fast
        cms = self.cfg.cms if cms is None else cms
        b, q = subj.shape
        qb = self.bag(tok) + (self.probe_vec if probe else 0.0)
        s = self.tok(subj)
        wd_idx = (weekday0[:, None] + target) % 7
        wd_t = F.one_hot(wd_idx, 7).float()
        hz = (target - day).clamp(min=0).float()
        hzf = torch.stack([hz / 7.0, torch.log1p(hz) / math.log(8.0)], dim=-1)
        h = st.h[:, None, :].expand(b, q, -1)
        key = F.normalize(self.key_proj(torch.cat([s, wd_t, h], dim=-1)), dim=-1)
        read = self.memory_read(st, key, fast, cms)
        epi = self.episodic(st, torch.cat([qb + s, wd_t, hzf], dim=-1), subj, target, wd_idx)
        dayf = self.day_features(day, b)[:, None, :].expand(b, q, 2)
        parts = [qb, s, wd_t, hzf, h, self.read_proj(read), epi, dayf]
        if self.cfg.tallies:
            parts.append(self.tally_proj(self.tally_features(st, subj, wd_idx)))
        z = torch.cat(parts, dim=-1)
        return self.trunk(z), key

    def query_logits(
        self, st: CogState, tok: Tensor, subj: Tensor, target: Tensor, day: int, weekday0: Tensor, **kw: bool | None,
    ) -> tuple[Tensor, Tensor]:
        """The K heads' logits (B, Q, K) and the trunk features (B, Q, H)."""
        x, _ = self.features(st, tok, subj, target, day, weekday0, **kw)  # type: ignore[arg-type]
        return self.heads(x), x

    # ── one day ──────────────────────────────────────────────────────────────

    def day_step(
        self, st: CogState, ev: DayInput, day: int, weekday0: Tensor, act: Tensor, act_mask: Tensor, y: Tensor, *,
        kinds: Tensor | None = None, fast: bool | None = None, cms: bool | None = None, want_pre: bool = False,
    ) -> StepOut:
        """Day ``day``'s events: write today's outcomes ``y`` (B, A) by surprise, count them with the
        day's ``kinds`` (B, A, 4; ``did`` alone when None) in the tallies, step the belief, keep the
        events, and predict each activity tomorrow."""
        fast = self.cfg.fast if fast is None else fast
        cms = self.cfg.cms if cms is None else cms
        b = ev.mask.shape[0]
        st, surprise = self._write(st, y, act_mask, fast, cms)
        e = self.encode_events(ev)
        weekday = (weekday0 + day) % 7
        ctx = torch.cat([F.one_hot(weekday, 7).float(), self.day_features(day, b),
                         torch.log1p(ev.mask.sum(dim=1, keepdim=True).float()) / 3.0], dim=-1)
        pooled = self.pool(e, ev.mask)
        h_pre = None
        if want_pre:
            pre_mask = ev.mask & ~ev.answer
            both = self.gru(torch.cat([torch.cat([pooled, self.pool(e, pre_mask)], 0), ctx.repeat(2, 1)], -1),
                            st.h.repeat(2, 1))
            h, h_pre = both[:b], both[b:]
        else:
            h = self.gru(torch.cat([pooled, ctx], dim=-1), st.h)
        st = self._append(st._replace(h=h, act=act), e, ev, day, weekday)
        if kinds is None:
            kinds = F.pad(y[..., None], (0, N_KINDS - 1))
        st = self.count(st, kinds * act_mask[..., None].to(y.dtype), y * act_mask.to(y.dtype), weekday)
        a = act.shape[1]
        empty = torch.zeros(b, a, 1, dtype=torch.long)
        tomorrow = torch.full((b, a), day + 1, dtype=torch.long)
        x, key = self.features(st, empty, act, tomorrow, day, weekday0, probe=True, fast=fast, cms=cms)
        logits = self.heads(x)
        st = st._replace(probe_logits=logits.detach(), probe_key=key, probe_ok=act_mask.to(h.dtype))
        return StepOut(st, logits, h_pre, surprise)

    # ── answers and the self-forecast ────────────────────────────────────────

    def answer_logits(self, h: Tensor, q_tok: Tensor, q_subj: Tensor, opt: Tensor, opt_mask: Tensor) -> Tensor:
        """Logits (B, O) over a question's options, from the belief before the answer."""
        z = self.ans_ctx(torch.cat([h, self.bag(q_tok), self.tok(q_subj)], dim=-1))
        o = self.ans_opt(self.bag(opt))
        return (torch.einsum("bd,bod->bo", z, o) / math.sqrt(self.cfg.dim)).masked_fill(~opt_mask, -1e9)

    def self_forecast(self, st: CogState, day: int) -> Tensor:
        """The core's predicted mean log loss (B,) on the queries that resolve tomorrow."""
        p = mean_prob(st.probe_logits)
        ent = -(p * p.log() + (1 - p) * (1 - p).log())
        ok = st.probe_ok.clamp(min=0.0)
        n = ok.sum(dim=1).clamp(min=1.0)
        feats = torch.stack([(ent * ok).sum(1) / n, (mutual_information(st.probe_logits) * ok).sum(1) / n], dim=-1)
        z = torch.cat([st.h, feats, self.day_features(day, st.h.shape[0])], dim=-1)
        return F.softplus(self.self_head(z))[:, 0]


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ── checkpoints ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Checkpoint:
    net: CogNet
    vocab: Vocab


def save_checkpoint(path: Path, net: CogNet, vocab: Vocab, extra: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(net.cfg), "vocab": vocab.to_json(), "state_dict": net.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path: Path | str) -> Checkpoint:
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    net = CogNet(CogConfig(**blob["config"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return Checkpoint(net, Vocab.from_json(blob["vocab"]))


# ── the predictor ────────────────────────────────────────────────────────────

NightMode = Literal["off", "uniform", "surprise"]


def day_input(vocab: Vocab, events: Sequence[Event]) -> DayInput:
    a = encode_day(vocab, events)
    t = torch.from_numpy
    return DayInput(t(a.tok)[None], t(a.kind)[None], t(a.val)[None], t(a.subj)[None], t(a.plan)[None],
                    t(a.span)[None], t(a.answer)[None], torch.ones(1, len(events), dtype=torch.bool))


@dataclass(frozen=True)
class _Pending:
    subject: str
    target_day: int
    logits: np.ndarray  # (K,) the heads' logits from the end of the asked day
    p: float  # the prediction then, adapter included
    cell: tuple[int, int]


@dataclass(frozen=True)
class PersonState:
    """Everything one person's predictor holds; frozen, so a clone is the same object."""

    day: int
    start: CogState  # after the night before ``day``
    cur: CogState  # after ``day``'s events so far
    today: tuple[Event, ...]
    adapter: night_mod.Adapter
    seen: tuple[Query, ...]  # today's queries (asked today) that ``predict`` was shown
    pending: tuple[_Pending, ...]  # predicted, not yet resolved
    replay: tuple[night_mod.Replayed, ...]  # resolved: the night's buffer
    counts: tuple[tuple[str, int], ...]  # queries asked per subject (the night's need)
    gain: night_mod.Gain = night_mod.NO_GAIN  # what the night's adapter has saved (``night.update_gain``)


class MindPredictor:
    """The cog core as a ``Predictor`` for one person at a time.

    ``predict`` returns the same predictions however often and by whom it is called, and changes
    no later prediction: it only notes which of today's queries (asked today) it was shown, a set
    that is the same whoever asks. At ``end_day`` each of them is predicted once more from the
    day's final state, and that prediction is what the night replays once it resolves (a ``did``
    event for its subject on its target day means it happened).
    Lesion switches: ``fast`` (the fast memory read and written), ``ensemble`` (heads used; 1 gives
    ``u = None``, and ``p`` is then head 0's), ``night`` (``off``, ``uniform`` or ``surprise``
    replay into the per-person adapter), ``gate`` (serve the adapter only while it has paid,
    ``night.served``; off, it is always served) and ``cms`` (the night level; only for a core
    trained with it).
    """

    def __init__(
        self, checkpoint: Checkpoint | Path | str | object, *, fast: bool = True, ensemble: int | None = None,
        night: NightMode = "surprise", cms: bool | None = None, seed: int = 0, name: str = "mind",
        replay_keep: int = 512, gate: bool = True,
    ) -> None:
        ck = checkpoint if isinstance(checkpoint, Checkpoint) else _as_checkpoint(checkpoint)
        self.net, self.vocab = ck.net.eval(), ck.vocab
        cfg = self.net.cfg
        self.k = cfg.heads if ensemble is None else ensemble
        if not 1 <= self.k <= cfg.heads:
            raise ValueError(f"ensemble is 1..{cfg.heads}")
        if night not in night_mod.NIGHT_MODES:
            raise ValueError(f"night is one of {night_mod.NIGHT_MODES}")
        if cms and not cfg.cms:
            raise ValueError("this core was trained without the continuum memory")
        self.fast = fast and cfg.fast
        self.cms = cfg.cms if cms is None else cms
        self.night_mode: NightMode = night
        self.seed, self.name, self.replay_keep, self.gate = seed, name, replay_keep, gate
        self._meta: LifeMeta | None = None
        self._state: PersonState | None = None

    # ── the protocol ─────────────────────────────────────────────────────────

    def begin(self, meta: LifeMeta) -> None:
        self._meta = meta
        self._act = torch.tensor([[self.vocab.id(a) for a in meta.activities]], dtype=torch.long)
        self._act_mask = torch.ones_like(self._act, dtype=torch.bool)
        self._wd0 = torch.tensor([meta.start_weekday])
        self._rng = np.random.default_rng(self.seed)
        with torch.no_grad():
            st = self.net.initial(1, len(meta.activities))
        adapter = night_mod.Adapter.zero(len(meta.activities) + 1)
        self._state = PersonState(-1, st, st, (), adapter, (), (), (), ())

    def observe(self, day: int, events: Sequence[Event]) -> None:
        self._state = self.peek(day, events)

    def predict(self, queries: Sequence[Query]) -> Sequence[Prediction]:
        ps = self.live
        known = {q.qid for q in ps.seen}
        fresh = tuple(q for q in queries if q.asked_day == ps.day and q.qid not in known)
        if fresh:
            self._state = replace(ps, seen=(*ps.seen, *dict((q.qid, q) for q in fresh).values()))
        return self.predict_on(ps, queries)[0]

    def end_day(self, day: int) -> None:
        """The night: today's queries are predicted from the day's final state and wait to resolve;
        those about today resolve into the replay buffer; the adapter takes its steps; the
        continuum memory (if on) consolidates."""
        ps = self.live
        counts = dict(ps.counts)
        pending = list(ps.pending)
        if ps.seen:
            _, _, base = self.predict_on(ps, ps.seen)
            cells = np.array([self._cell(q) for q in ps.seen], dtype=np.int64)
            p = _mean_sigmoid(base + ps.adapter.shift(cells)[:, None])
            for i, q in enumerate(ps.seen):
                pending.append(_Pending(q.subject, q.target_day, base[i], float(p[i]), self._cell(q)))
                counts[q.subject] = counts.get(q.subject, 0) + 1
        did = {e.subject for e in ps.today if e.kind == "did"} if ps.day == day else None
        resolved, waiting = [], []
        for pend in pending:
            if pend.target_day == day and did is not None:
                y = 1.0 if pend.subject in did else 0.0
                surprise = -math.log(pend.p if y else 1.0 - pend.p)
                resolved.append(night_mod.Replayed(pend.logits, y, surprise, pend.subject, day, pend.cell))
            elif pend.target_day > day:
                waiting.append(pend)
        replay = (*ps.replay, *resolved)[-self.replay_keep:]
        gain = night_mod.update_gain(ps.gain, resolved)
        adapter = ps.adapter
        if self.night_mode != "off" and replay:
            adapter = night_mod.night(adapter, replay, today=day, mode=self.night_mode, rng=self._rng, counts=counts)
        cur = ps.cur
        if self.cms:
            with torch.no_grad():
                cur = self.net.consolidate(cur)
        self._state = PersonState(ps.day, ps.start, cur, ps.today, adapter, (), tuple(waiting), replay,
                                  tuple(counts.items()), gain)

    # ── pure views, for the inquirer ─────────────────────────────────────────

    @property
    def live(self) -> PersonState:
        if self._state is None or self._meta is None:
            raise RuntimeError("begin() first")
        return self._state

    def peek(self, day: int, events: Sequence[Event], state: PersonState | None = None) -> PersonState:
        """The state after observing ``events`` on ``day``, without changing the live one."""
        ps = self.live if state is None else state
        start, cur, today = ps.start, ps.cur, ps.today
        with torch.no_grad():
            for gap in range(ps.day + 1, day):  # days nobody observed step with no events
                start = cur = self._step(cur, gap, ())
            if day != ps.day:
                start, today = cur, ()
            today = (*today, *events)
            cur = self._step(start, day, today)
        seen = ps.seen if day == ps.day else ()
        return PersonState(day, start, cur, today, ps.adapter, seen, ps.pending, ps.replay, ps.counts, ps.gain)

    def _step(self, st: CogState, day: int, events: Sequence[Event]) -> CogState:
        assert self._meta is not None
        y = torch.from_numpy(outcomes(events, self._meta.activities))[None]
        kinds = torch.from_numpy(activity_kinds(events, self._meta.activities))[None]
        out = self.net.day_step(st, day_input(self.vocab, events), day, self._wd0, self._act, self._act_mask, y,
                                kinds=kinds, fast=self.fast, cms=self.cms)
        return out.state

    def predict_on(self, ps: PersonState, queries: Sequence[Query]) -> tuple[list[Prediction], np.ndarray, np.ndarray]:
        """Predictions from a state, with the trunk features and base logits; nothing is recorded."""
        if not queries:
            return [], np.zeros((0, self.net.cfg.hidden)), np.zeros((0, self.k))
        qa = encode_queries(self.vocab, queries)
        t = torch.from_numpy
        with torch.no_grad():
            logits, x = self.net.query_logits(ps.cur, t(qa.tok)[None], t(qa.subj)[None], t(qa.target)[None],
                                              max(ps.day, 0), self._wd0, fast=self.fast, cms=self.cms)
        base = logits[0, :, : self.k].numpy().astype(np.float64)
        feats = x[0].numpy().astype(np.float64)
        cells = np.array([self._cell(q) for q in queries], dtype=np.int64)
        adapter = night_mod.served(ps.adapter, ps.gain) if self.gate else ps.adapter
        shifted = torch.from_numpy(base + adapter.shift(cells)[:, None])
        p = mean_prob(shifted).numpy()
        u = mutual_information(shifted).numpy() if self.k > 1 else None
        preds = [Prediction(q.qid, float(p[i]), None if u is None else float(u[i])) for i, q in enumerate(queries)]
        return preds, feats, base

    def _cell(self, q: Query) -> tuple[int, int]:
        """The adapter's (subject slot, target weekday) for a query; slot A is any other subject."""
        assert self._meta is not None
        acts = self._meta.activities
        slot = acts.index(q.subject) if q.subject in acts else len(acts)
        return slot, (self._meta.start_weekday + q.target_day) % 7

    def answer_probs(self, question: Question, state: PersonState | None = None) -> np.ndarray:
        """The answer head's distribution over ``question.options``."""
        ps = self.live if state is None else state
        qa = encode_question(self.vocab, question)
        t = torch.from_numpy
        with torch.no_grad():
            logits = self.net.answer_logits(ps.cur.h, t(qa.tok)[None], torch.tensor([qa.subj]), t(qa.options)[None],
                                            torch.ones(1, qa.options.shape[0], dtype=torch.bool))
        return torch.softmax(logits[0], dim=-1).numpy()

    def self_forecast(self) -> float:
        """The core's predicted mean log loss on the queries that resolve tomorrow."""
        ps = self.live
        with torch.no_grad():
            return float(self.net.self_forecast(ps.cur, max(ps.day, 0))[0])


def _mean_sigmoid(logits: np.ndarray) -> np.ndarray:
    p = (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).mean(axis=-1)
    return np.clip(p, P_FLOOR, 1 - P_FLOOR)


def _as_checkpoint(source: object) -> Checkpoint:
    if isinstance(source, (str, Path)):
        return load_checkpoint(source)
    net, vocab = getattr(source, "net", None), getattr(source, "vocab", None)
    if isinstance(net, CogNet) and isinstance(vocab, Vocab):
        return Checkpoint(net, vocab)
    raise TypeError("a checkpoint path, a Checkpoint, or a trained result with .net and .vocab")


__all__ = [
    "NO_PLAN", "CogConfig", "CogNet", "CogState", "Checkpoint", "DayInput", "MindPredictor", "PersonState",
    "load_checkpoint", "multi_fast_update", "mutual_information", "save_checkpoint",
]
