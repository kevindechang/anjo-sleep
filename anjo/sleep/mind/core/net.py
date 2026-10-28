"""The shared recurrent network of MIND's core (torch, CPU). Batched: every tensor leads with B.

**One event, one step.** The event's latent is RMS-normalized, projected to the width and joined
with its kind and the gap since the event before (``encode``). Then:

1. **Slot binding by learned attention.** The event's address scores every slot (a dot product
   with the slot's normalized state plus its log usage, so a free slot is hardly addressed) and a
   learned "new thread" logit; one softmax splits the write, kept to its ``TOP_WRITES`` largest
   entries and renormalized, so an event writes where it belongs and usage marks real threads. The
   new-thread share writes into the least-used slot. The only rule is bookkeeping: a slot whose usage fell below ``free_below`` is
   free, and a slot is reset to the learned initial state just before a new thread is written into
   it. Each slot is updated by a shared GRU cell in proportion to its write; usage rises with
   writes, fades with idle time at a learned rate, and a learned close gate lets an event that
   ends a thread release its slot.
2. **Fast-weight memory (Titans, linear memory).** The key and value are projections of the event
   itself; the memory is read with the event's key before it is written, and the read joins the
   event in the slot update. So a new booking with a party recalls, per person, what was said with
   that party before (two cancellations, say), even after those threads closed, and the slot it
   opens carries that recall. Surprise is the memory's error on the event, ``mean((M k - v)^2)``;
   the update is a gradient step on that error with momentum and a forget gate, each gate a
   function of the event (``fast_update``): ``S = eta S - theta grad``, ``M = (1 - alpha) M + S``.
   Keys are unit vectors and ``theta <= theta_max <= 0.5``, so a step never overshoots.
3. **Global state:** a GRU cell over the event, the memory read and what was written.
4. **The stance probe** (``stance``, for contradictions). A plain key is a projection of the whole
   event, so a constraint that flips a plan's attribute cancels the very part of the key it
   shares with the plan, and recalls nothing of it. With ``stance`` the key is the centered square
   of that projection: what an event is about, whatever its sign, so the constraint recalls the
   plan. The slot update then also reads the elementwise product of the recalled value and the
   event's own (normalized): a learned probe of whether the two agree or flip, the pair feature
   natural-language inference reads, applied to her own memory.

**Heads** read each slot with the global state and the slot's age: the daily hazard over ``horizon`` days (risk is one minus survival), a missing
prerequisite, the predicted squared error of its own risk (the self-model), and a choice logit.
Globally: speak or wait, the "raise nothing" logit, and an imagined next event for dreaming.
Per-person adaptation overrides only the head parameters (``HEAD_PARAMS``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from anjo.sleep.mind.core.types import KINDS, CoreConfig

_EPS: Final = 1e-6
_LOG_FLOOR: Final = 1e-4  # log usage floor: a free slot's address logit sits near log(1e-4)
TOP_WRITES: Final = 2  # an event writes into its two best addresses only, so usage marks real threads
HEAD_PARAMS: Final = (
    "hazard.weight", "hazard.bias", "missing.weight", "missing.bias", "self_err.weight", "self_err.bias",
    "choice.weight", "choice.bias", "none.weight", "none.bias", "speak.weight", "speak.bias",
)


class Tensors(NamedTuple):
    """One batch of per-person state. Times are days on the core's clock."""

    slots: Tensor  # (B, K, d)
    usage: Tensor  # (B, K)
    opened: Tensor  # (B, K) when each slot's thread began
    memory: Tensor  # (B, d, d) the fast weights
    momentum: Tensor  # (B, d, d)
    glob: Tensor  # (B, d)
    now: Tensor  # (B,) the latest event's time


class Written(NamedTuple):
    """What a step did: each slot's write, the slots it reset, and the surprise."""

    write: Tensor  # (B, K)
    fresh: Tensor  # (B, K) 1 where a free slot was reset for a new thread
    surprise: Tensor  # (B,)
    address: Tensor  # (B, K + 1) the softmax, the new-thread share last


class Heads(NamedTuple):
    log_hazard: Tensor  # (B, K, H) log h
    log_survive: Tensor  # (B, K, H) log (1 - h)
    risk: Tensor  # (B, K) failure within the horizon
    missing_logit: Tensor  # (B, K)
    self_error: Tensor  # (B, K) predicted (outcome - risk)^2
    choice_logits: Tensor  # (B, K + 1) the last is "raise nothing"
    speak_logit: Tensor  # (B,)
    imagined: Tensor  # (B, d) the next event's encoding, for dreams


def fast_update(
    memory: Tensor, momentum: Tensor, k: Tensor, v: Tensor, theta: Tensor, eta: Tensor, alpha: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """One Titans step of a linear memory on (k, v): ``(memory, momentum, surprise)``. The gradient
    of ``|M k - v|^2`` in ``M`` is ``2 (M k - v) k^T``."""
    err = torch.bmm(memory, k[:, :, None])[:, :, 0] - v
    surprise = err.pow(2).mean(dim=-1)
    grad = 2.0 * err[:, :, None] * k[:, None, :]
    momentum = eta[:, None, None] * momentum - theta[:, None, None] * grad
    memory = (1.0 - alpha[:, None, None]) * memory + momentum
    return memory, momentum, surprise


class CoreNet(nn.Module):  # type: ignore[misc]  # torch's types are skipped (follow_imports)
    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        d = cfg.width
        self.cfg = cfg
        self.inp = nn.Linear(cfg.dim_in, d)
        self.inp_norm = nn.LayerNorm(d)
        self.kind = nn.Embedding(len(KINDS), d)
        self.time = nn.Linear(2, d)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.read_norm = nn.LayerNorm(d)
        self.gates = nn.Linear(d + 1, 3)
        self.addr = nn.Linear(d, d, bias=False)
        self.slot_norm = nn.LayerNorm(d)
        self.new = nn.Linear(2 * d, 1)
        self.slot_init = nn.Parameter(torch.zeros(d))
        self.cell = nn.GRUCell((3 if cfg.stance else 2) * d, d)
        self.stance_norm = nn.LayerNorm(d) if cfg.stance else None
        self.close = nn.Linear(2 * d, 1)
        self.fade = nn.Parameter(torch.zeros(()))  # usage fades at softplus(fade) per idle day
        self.glob_cell = nn.GRUCell(3 * d, d)
        self.trunk = nn.Linear(2 * d + 1, d)
        self.hazard = nn.Linear(d, cfg.horizon)
        self.missing = nn.Linear(d, 1)
        self.self_err = nn.Linear(d, 1)
        self.choice = nn.Linear(d, 1)
        self.none = nn.Linear(d, 1)
        self.speak = nn.Linear(2 * d, 1)
        self.imagine = nn.Linear(2 * d, d)
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([0.0, -2.0, -3.0]))  # start: half steps, little momentum or forgetting
            self.new.bias.fill_(1.0)  # start: an event rather opens a thread than joins one
            self.speak.bias.fill_(-2.0)
            self.hazard.bias.fill_(-3.0)

    # ── state ────────────────────────────────────────────────────────────────

    def initial(self, batch: int, now: Tensor | None = None) -> Tensors:
        d, k = self.cfg.width, self.cfg.slots
        when = torch.zeros(batch) if now is None else now
        return Tensors(
            slots=self.slot_init.expand(batch, k, d).clone(), usage=torch.zeros(batch, k),
            opened=when[:, None].expand(batch, k).clone(), memory=torch.zeros(batch, d, d),
            momentum=torch.zeros(batch, d, d), glob=torch.zeros(batch, d), now=when.clone(),
        )

    # ── one event ────────────────────────────────────────────────────────────

    def encode(self, vec: Tensor, kind: Tensor, gap: Tensor) -> Tensor:
        """Events (..., D) with their kind index and gap in days to the event before -> (..., d)."""
        rms = vec.pow(2).mean(dim=-1, keepdim=True).add(_EPS).sqrt()
        timing = torch.stack([torch.log1p(gap.clamp(min=0.0)), torch.ones_like(gap)], dim=-1)
        return self.inp_norm(self.inp(vec / rms)) + self.kind(kind) + self.time(timing)

    def step(self, st: Tensors, e: Tensor, gap: Tensor, now: Tensor) -> tuple[Tensors, Written]:
        cfg = self.cfg
        faded = st.usage * torch.exp(-F.softplus(self.fade) * gap.clamp(min=0.0))[:, None]
        st = st._replace(usage=faded)
        address = self._address(st, e)
        free = F.one_hot(torch.argmin(st.usage, dim=1), cfg.slots).to(e.dtype)
        fresh = free * (st.usage < cfg.free_below).to(e.dtype)
        slots = st.slots * (1.0 - fresh[:, :, None]) + self.slot_init * fresh[:, :, None]
        opened = st.opened * (1.0 - fresh) + now[:, None] * fresh
        write = address[:, : cfg.slots] + address[:, cfg.slots:] * free
        memory, momentum, surprise, read = self._remember(st, e, gap)
        parts = [e, self.read_norm(read)]
        if self.stance_norm is not None:
            parts.append(self.stance_norm(read * self.wv(e)))
        joined = torch.cat(parts, dim=-1)[:, None, :].expand(-1, cfg.slots, -1)
        cand = self.cell(joined.reshape(cfg.slots * e.shape[0], -1), slots.reshape(-1, cfg.width)).view_as(slots)
        slots = slots + write[:, :, None] * (cand - slots)
        usage = st.usage + write * (1.0 - st.usage)
        closing = torch.sigmoid(self.close(torch.cat([slots, e[:, None, :].expand_as(slots)], dim=-1)))[:, :, 0]
        usage = usage * (1.0 - write * closing)
        written = (write[:, :, None] * slots).sum(dim=1)
        glob = self.glob_cell(torch.cat([e, read, written], dim=-1), st.glob)
        out = Tensors(slots, usage, opened, memory, momentum, glob, now)
        return out, Written(write, fresh, surprise, address)

    def _remember(self, st: Tensors, e: Tensor, gap: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """What the memory recalls for the event (read before the write), then one surprise step."""
        cfg = self.cfg
        k = self._key(e)
        read = torch.bmm(st.memory, k[:, :, None])[:, :, 0]
        if not cfg.fast:
            return st.memory, st.momentum, torch.zeros(e.shape[0]), torch.zeros_like(read)
        g = torch.sigmoid(self.gates(torch.cat([e, torch.log1p(gap.clamp(min=0.0))[:, None]], dim=-1)))
        memory, momentum, surprise = fast_update(
            st.memory, st.momentum, k, self.wv(e), g[:, 0] * cfg.theta_max, g[:, 1], g[:, 2])
        return memory, momentum, surprise, read

    def _key(self, e: Tensor) -> Tensor:
        """The memory's key: the event's projection, or under ``stance`` its centered square (its topic)."""
        q = self.wk(e)
        if self.cfg.stance:
            q = q.pow(2)
            q = q - q.mean(dim=-1, keepdim=True)
        return F.normalize(q, dim=-1)

    def _address(self, st: Tensors, e: Tensor) -> Tensor:
        a = self.addr(e)
        logits = torch.einsum("bkd,bd->bk", self.slot_norm(st.slots), a) / math.sqrt(self.cfg.width)
        logits = logits + torch.log(st.usage + _LOG_FLOOR)
        new = self.new(torch.cat([e, st.glob], dim=-1))
        soft = torch.softmax(torch.cat([logits, new], dim=-1), dim=-1)
        kept = torch.zeros_like(soft).scatter(1, soft.topk(TOP_WRITES, dim=-1).indices, 1.0)
        sparse = soft * kept
        return sparse / sparse.sum(dim=-1, keepdim=True)

    # ── heads ────────────────────────────────────────────────────────────────

    def heads(self, st: Tensors, delta: Mapping[str, Tensor] | None = None) -> Heads:
        cfg = self.cfg

        def lin(name: str, x: Tensor) -> Tensor:
            layer = getattr(self, name)
            w, b = layer.weight, layer.bias
            if delta:
                w = w + delta.get(f"{name}.weight", 0.0)
                b = b + delta.get(f"{name}.bias", 0.0)
            return F.linear(x, w, b)

        h = self.features(st)
        hazard = lin("hazard", h)
        log_hazard, log_survive = F.logsigmoid(hazard), F.logsigmoid(-hazard)
        risk = 1.0 - torch.exp(log_survive.sum(dim=-1))
        usage_log = torch.log(st.usage + _LOG_FLOOR)
        choice = torch.cat([lin("choice", h)[:, :, 0] + usage_log, lin("none", st.glob)], dim=-1)
        pooled = (torch.softmax(choice, dim=-1)[:, : cfg.slots, None] * h).sum(dim=1)
        return Heads(
            log_hazard=log_hazard, log_survive=log_survive, risk=risk, missing_logit=lin("missing", h)[:, :, 0],
            self_error=F.softplus(lin("self_err", h)[:, :, 0]), choice_logits=choice,
            speak_logit=lin("speak", torch.cat([st.glob, pooled], dim=-1))[:, 0],
            imagined=self.imagine(torch.cat([st.glob, pooled], dim=-1)),
        )

    def features(self, st: Tensors) -> Tensor:
        """Each slot's trunk features (B, K, d): its state, the global state and its age in horizons."""
        cfg = self.cfg
        age = ((st.now[:, None] - st.opened) / cfg.horizon).clamp(0.0, 4.0)[:, :, None]
        glob = st.glob[:, None, :].expand(-1, cfg.slots, -1)
        return F.gelu(self.trunk(torch.cat([self.slot_norm(st.slots), glob, age], dim=-1)))


def blend(keep: Tensor, new: Tensors, old: Tensors) -> Tensors:
    """``new`` where ``keep`` (B,) is 1, else ``old``: a padded step changes nothing."""

    def mix(a: Tensor, b: Tensor) -> Tensor:
        shape = (-1,) + (1,) * (a.dim() - 1)
        m = keep.view(shape).to(a.dtype)
        return a * m + b * (1.0 - m)

    return Tensors(*(mix(a, b) for a, b in zip(new, old)))


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))
