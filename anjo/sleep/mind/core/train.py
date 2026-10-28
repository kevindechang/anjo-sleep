"""Training MIND's core (torch, CPU, deterministic): shared across lives, then per person at night.

**Shared training** unrolls whole lives in batches (backpropagation through every event, the
fast-weight updates included) on labelled streams. Targets come from the labels and from the
core's own writes: a failing thread is live from ``knowable_at`` until ``outcome_at``; a slot's
target is the share of the thread's events it holds (so binding, not a string rule, decides which
slot answers for a thread). The losses:

- the hazard heads: survival likelihood of the day the thread surfaces (censored past the horizon,
  or for a slot that holds no failing thread);
- missing step, speak (any failing thread live) and choice (the slots holding the live threads, or
  "raise nothing");
- the self-model: the predicted squared error of each slot's own risk against the realized one;
- binding: events of one thread write into the same slot, events of different threads apart;
- imagination: the next event's encoding, for dreams.

Validation keeps the best epoch; training stops once ``patience`` epochs pass without a better
validation loss, or at the ``epochs`` cap. The speak threshold is then locked on validation
(``evaluate.lock_threshold``), and the dream's own ignition threshold on the same lives
(``evaluate.lock_ignite``). Seeds, one CPU thread and deterministic algorithms make a run
repeatable bit for bit.

**Per-person adaptation** (``adapt``) is conservative: the person's events are replayed through
the frozen core, and only the head parameters move, by a few gradient steps on that person's
resolved threads with an L2 prior toward the shared heads. A step that doesn't lower the
person's loss is refused: the delta is then empty.
"""
from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from anjo.sleep.mind.core import evaluate
from anjo.sleep.mind.core.api import Core, Thresholds
from anjo.sleep.mind.core.net import HEAD_PARAMS, CoreNet, Heads, Tensors, blend
from anjo.sleep.mind.core.types import CoreConfig, LabelledStream, ThreadLabel, days_of, kind_index

WEIGHTS: Final = {"hazard": 1.0, "speak": 1.0, "choice": 1.0, "missing": 0.5, "self": 1.0, "bind": 0.5,
                  "imagine": 0.1}
_EPS: Final = 1e-6
FOCUS: Final = 20.0  # extra weight on the slots of open labelled threads: a problem against its decoy


@dataclass(frozen=True)
class Trained:
    core: Core
    log: tuple[tuple[int, float, float], ...]  # (epoch, train loss, validation loss)
    best_epoch: int


class Batch(NamedTuple):
    vec: Tensor  # (B, T, D)
    kind: Tensor  # (B, T)
    gap: Tensor  # (B, T) days since the event before
    now: Tensor  # (B, T) days
    mask: Tensor  # (B, T)
    member: Tensor  # (B, L, T) event t belongs to label l
    live: Tensor  # (B, L, T) label l is a failing thread, live at t
    open: Tensor  # (B, L, T) label l (failing or not) is between knowable and outcome at t
    fail_bin: Tensor  # (B, L, T) the day it surfaces, as a hazard bin; horizon = beyond it
    missing: Tensor  # (B, L)
    same: Tensor  # (P, 3) b, t1, t2: events of one thread
    apart: Tensor  # (Q, 3) b, t1, t2: events of two threads


# ── Streams to tensors ────────────────────────────────────────────────────────


def _fail_bin(label: ThreadLabel, t: datetime, horizon: int) -> int:
    days = (label.outcome_at.date() - t.date()).days
    return min(max(days - 1, 0), horizon)


def batch(streams: Sequence[LabelledStream], horizon: int) -> Batch:
    b, t = len(streams), max(len(s.events) for s in streams)
    width = streams[0].events[0].vec.size
    lmax = max(1, max(len(s.labels) for s in streams))
    vec: np.ndarray = np.zeros((b, t, width), np.float32)
    kind: np.ndarray = np.zeros((b, t), np.int64)
    gap: np.ndarray = np.zeros((b, t), np.float32)
    now: np.ndarray = np.zeros((b, t), np.float32)
    mask: np.ndarray = np.zeros((b, t), np.float32)
    member: np.ndarray = np.zeros((b, lmax, t), np.float32)
    live: np.ndarray = np.zeros((b, lmax, t), np.float32)
    opened: np.ndarray = np.zeros((b, lmax, t), np.float32)
    fail_bin: np.ndarray = np.full((b, lmax, t), horizon, np.int64)
    missing: np.ndarray = np.zeros((b, lmax), np.float32)
    same: list[tuple[int, int, int]] = []
    apart: list[tuple[int, int, int]] = []
    for i, stream in enumerate(streams):
        index = {e.event_id: n for n, e in enumerate(stream.events)}
        days = [days_of(e.t) for e in stream.events]
        for n, event in enumerate(stream.events):
            vec[i, n], kind[i, n], now[i, n], mask[i, n] = event.vec, kind_index(event.kind), days[n], 1.0
            gap[i, n] = days[n] - days[n - 1] if n else 0.0
        groups = []
        for j, label in enumerate(stream.labels):
            rows = [index[eid] for eid in label.event_ids]
            groups.append(rows)
            member[i, j, rows] = 1.0
            missing[i, j] = float(label.missing)
            for n, event in enumerate(stream.events):
                if not label.knowable_at <= event.t < label.outcome_at:
                    continue
                opened[i, j, n] = 1.0
                if label.failed:
                    live[i, j, n] = 1.0
                    fail_bin[i, j, n] = _fail_bin(label, event.t, horizon)
        for a, rows in enumerate(groups):
            same.extend((i, x, y) for x in rows for y in rows if x < y)
            apart.extend((i, x, y) for other in groups[a + 1:] for x in rows for y in other)
    def pairs(rows: list[tuple[int, int, int]]) -> Tensor:
        return torch.tensor(rows, dtype=torch.long).view(-1, 3)

    return Batch(
        torch.from_numpy(vec), torch.from_numpy(kind), torch.from_numpy(gap), torch.from_numpy(now),
        torch.from_numpy(mask), torch.from_numpy(member), torch.from_numpy(live), torch.from_numpy(opened),
        torch.from_numpy(fail_bin),
        torch.from_numpy(missing), pairs(same), pairs(apart),
    )


# ── The unroll and the loss ───────────────────────────────────────────────────


class Unrolled(NamedTuple):
    heads: Heads  # every field (B, T, ...)
    write: Tensor  # (B, T, K)
    usage: Tensor  # (B, T, K)
    encoded: Tensor  # (B, T, d)


def unroll(net: CoreNet, data: Batch) -> Unrolled:
    encoded = net.encode(data.vec, data.kind, data.gap)
    st = net.initial(data.vec.shape[0], data.now[:, 0])
    heads, writes, usages = [], [], []
    for step in range(data.vec.shape[1]):
        start = net.initial(data.vec.shape[0], data.now[:, step]) if not net.cfg.memory else st
        new, written = net.step(start, encoded[:, step], data.gap[:, step], data.now[:, step])
        st = blend(data.mask[:, step], new, st)
        heads.append(net.heads(st))
        writes.append(written.write * data.mask[:, step, None])
        usages.append(st.usage)
    stacked = Heads(*(torch.stack(parts, dim=1) for parts in zip(*heads)))
    return Unrolled(stacked, torch.stack(writes, dim=1), torch.stack(usages, dim=1), encoded)


class Targets(NamedTuple):
    slot: Tensor  # (B, T, K) the share of a live failing thread the slot holds
    fail_bin: Tensor  # (B, T, K)
    missing: Tensor  # (B, T, K)
    speak: Tensor  # (B, T)
    choice: Tensor  # (B, T, K + 1) a distribution
    focus: Tensor  # (B, T, K) the share of an open labelled thread (a problem or its decoy) the slot holds


def targets(write: Tensor, data: Batch, horizon: int) -> Targets:
    held = torch.cumsum(data.member[..., None] * write.detach()[:, None], dim=2)  # (B, L, T, K)
    share = held / (held.sum(dim=-1, keepdim=True) + _EPS)
    shares = data.live[..., None] * share
    best, which = shares.max(dim=1)  # (B, T, K)
    focus = (data.open[..., None] * share).max(dim=1).values
    bins = _pick(data.fail_bin, which)
    missing = best * _pick(data.missing[:, :, None].expand(-1, -1, data.fail_bin.shape[-1]), which)
    speak = data.live.max(dim=1).values
    pooled = shares.sum(dim=1)
    total = pooled.sum(dim=-1, keepdim=True)
    choice = torch.cat([pooled, (total <= _EPS).float()], dim=-1)
    choice = choice / choice.sum(dim=-1, keepdim=True)
    return Targets(best, bins, missing, speak, choice, focus)


def _pick(per_label: Tensor, which: Tensor) -> Tensor:
    """``per_label`` (B, L, T) read at label ``which`` (B, T, K) -> (B, T, K)."""
    b, _, t = per_label.shape
    k = which.shape[-1]
    gathered = torch.gather(per_label.permute(0, 2, 1)[:, :, None, :].expand(b, t, k, -1), 3, which[..., None])
    return gathered[..., 0]


def _survival(log_hazard: Tensor, log_survive: Tensor, fail_bin: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
    """Negative log-likelihoods of surfacing on bin ``fail_bin``, and of surviving the horizon."""
    before = torch.cumsum(log_survive, dim=-1) - log_survive
    idx = fail_bin.clamp(max=horizon - 1)[..., None]
    fail = -(torch.gather(before, -1, idx) + torch.gather(log_hazard, -1, idx))[..., 0]
    survive = -log_survive.sum(dim=-1)
    return fail, survive


def head_loss(heads: Heads, usage: Tensor, tg: Targets, mask: Tensor, horizon: int) -> dict[str, Tensor]:
    weight = (usage.detach() + 0.05 + FOCUS * tg.focus) * mask[..., None]
    fail, survive = _survival(heads.log_hazard, heads.log_survive, tg.fail_bin, horizon)
    within = (tg.fail_bin < horizon).float()
    hazard = tg.slot * within * fail + (1.0 - tg.slot * within) * survive
    realized = tg.slot * within
    errs = (realized - heads.risk.detach()).pow(2)
    n = mask.sum() + _EPS
    return {
        "hazard": (hazard * weight).sum() / (weight.sum() + _EPS),
        "missing": (F.binary_cross_entropy_with_logits(heads.missing_logit, tg.missing, reduction="none")
                    * weight).sum() / (weight.sum() + _EPS),
        "self": ((heads.self_error - errs).pow(2) * weight).sum() / (weight.sum() + _EPS),
        "speak": (F.binary_cross_entropy_with_logits(heads.speak_logit, tg.speak, reduction="none") * mask).sum() / n,
        "choice": (-(tg.choice * F.log_softmax(heads.choice_logits, dim=-1)).sum(-1) * mask).sum() / n,
    }


def loss(net: CoreNet, data: Batch) -> tuple[Tensor, dict[str, float]]:
    cfg = net.cfg
    run = unroll(net, data)
    parts = head_loss(run.heads, run.usage, targets(run.write, data, cfg.horizon), data.mask, cfg.horizon)
    parts["bind"] = _binding(run.write, data)
    nxt = data.mask[:, 1:]
    guess = F.normalize(run.heads.imagined[:, :-1], dim=-1)
    truth = F.normalize(run.encoded[:, 1:].detach(), dim=-1)
    parts["imagine"] = ((guess - truth).pow(2).sum(-1) * nxt).sum() / (nxt.sum() + _EPS)
    total = sum((WEIGHTS[name] * value for name, value in parts.items()), torch.zeros(()))
    return total, {name: float(value.detach()) for name, value in parts.items()}


def _binding(write: Tensor, data: Batch) -> Tensor:
    def overlap(pairs: Tensor) -> Tensor:
        return (write[pairs[:, 0], pairs[:, 1]] * write[pairs[:, 0], pairs[:, 2]]).sum(-1)

    # bounded on purpose: a log here explodes when two writes barely overlap, and gradient clipping
    # then silences every other loss in the batch
    together = (1.0 - overlap(data.same)).mean() if len(data.same) else torch.zeros(())
    apart = overlap(data.apart).mean() if len(data.apart) else torch.zeros(())
    return together + apart


# ── Shared training ───────────────────────────────────────────────────────────


@contextlib.contextmanager
def deterministic(seed: int) -> Iterator[np.random.Generator]:
    """Seeded, one thread, deterministic algorithms; the caller's settings come back on exit."""
    threads = torch.get_num_threads()
    was, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield np.random.default_rng(seed)
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn_only)
        torch.set_num_threads(threads)


def _chunks(streams: Sequence[LabelledStream], order: np.ndarray, size: int) -> Iterator[list[LabelledStream]]:
    for start in range(0, len(order), size):
        yield [streams[i] for i in order[start:start + size]]


def validation_loss(net: CoreNet, batches: Sequence[Batch]) -> float:
    with torch.no_grad():
        return math.fsum(float(loss(net, data)[0]) for data in batches) / max(len(batches), 1)


LifeSource = Sequence[LabelledStream] | Callable[[int], Sequence[LabelledStream]]


def train_shared(
    fit: LifeSource, val: Sequence[LabelledStream], config: CoreConfig, *, seed: int,
    epochs: int, patience: int | None = None, batch_size: int = 16, lr: float = 1e-3, weight_decay: float = 0.01,
    clip: float = 1.0,
) -> Trained:
    """The shared core, trained on ``fit``, its best epoch and thresholds chosen on ``val``.
    ``fit`` is a fixed set of lives, or a source of each epoch's lives (``fit(epoch)``): fresh
    generated lives each epoch, so nothing can be learned by heart. ``epochs`` caps the run;
    ``patience`` ends it early, once that many epochs pass without a better validation loss."""
    if not val or (not callable(fit) and not fit):
        raise ValueError("shared training needs training and validation lives")
    with deterministic(seed) as rng:
        net = CoreNet(config)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        held = [batch(list(val[i:i + batch_size]), config.horizon) for i in range(0, len(val), batch_size)]
        fit_batches: dict[tuple[str, ...], Batch] = {}
        best, best_loss, best_epoch, log = None, math.inf, 0, []
        for epoch in range(epochs):
            if patience is not None and epoch - best_epoch > patience:
                break
            net.train()
            seen = []
            lives = fit(epoch) if callable(fit) else fit
            for chunk in _chunks(lives, rng.permutation(len(lives)), batch_size):
                key = tuple(s.life_id for s in chunk)
                data = batch(chunk, config.horizon) if callable(fit) else fit_batches.setdefault(
                    key, batch(chunk, config.horizon))
                total, _ = loss(net, data)
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
                seen.append(float(total.detach()))
            net.eval()
            checked = validation_loss(net, held)
            log.append((epoch, round(math.fsum(seen) / len(seen), 6), round(checked, 6)))
            if checked < best_loss:
                best, best_loss, best_epoch = {k: v.clone() for k, v in net.state_dict().items()}, checked, epoch
        assert best is not None
        net.load_state_dict(best)
        core = Core(net)
        thresholds = Thresholds(speak=evaluate.lock_threshold(core, val), ignite=evaluate.lock_ignite(core, val, seed=seed))
    return Trained(core.with_thresholds(thresholds), tuple(log), best_epoch)


# ── Per-person adaptation ─────────────────────────────────────────────────────


def replay(core: Core, stream: LabelledStream) -> tuple[Tensors, Tensor, Tensor]:
    """The person's states after each event, stacked (T, ...), with the writes and usage."""
    data = batch([stream], core.config.horizon)
    net = core.net
    with torch.no_grad():
        encoded = net.encode(data.vec, data.kind, data.gap)
        st = net.initial(1, data.now[:, 0])
        states, writes = [], []
        for step in range(data.vec.shape[1]):
            start = st if core.config.memory else net.initial(1, data.now[:, step])
            st, written = net.step(start, encoded[:, step], data.gap[:, step], data.now[:, step])
            states.append(st)
            writes.append(written.write)
    stacked = Tensors(*(torch.cat(parts, dim=0) for parts in zip(*states)))
    return stacked, torch.cat(writes, dim=0)[None], stacked.usage[None]


def adapt(
    core: Core, stream: LabelledStream, *, steps: int = 5, lr: float = 0.01, prior: float = 1.0, seed: int = 0,
) -> dict[str, np.ndarray]:
    """The person's head delta from their resolved threads (``stream.labels``); empty when no step
    lowered their loss."""
    if not stream.labels or not stream.events:
        return {}
    cfg = core.config
    with deterministic(seed):
        states, write, usage = replay(core, stream)
        data = batch([stream], cfg.horizon)
        tg = targets(write, data, cfg.horizon)
        params = dict(core.net.named_parameters())
        delta = {name: torch.zeros_like(params[name], requires_grad=True) for name in HEAD_PARAMS}
        opt = torch.optim.Adam(list(delta.values()), lr=lr)

        def person_loss(with_delta: Mapping[str, Tensor]) -> Tensor:
            flat = core.net.heads(states, with_delta)
            heads = Heads(*(part[None] for part in flat))
            parts = head_loss(heads, usage, tg, data.mask, cfg.horizon)
            return sum((WEIGHTS[n] * parts[n] for n in parts), torch.zeros(()))

        with torch.no_grad():
            start = float(person_loss({}))
        for _ in range(steps):
            total = person_loss(delta) + prior * sum((d.pow(2).sum() for d in delta.values()), torch.zeros(()))
            opt.zero_grad()
            total.backward()
            opt.step()
        with torch.no_grad():
            end = float(person_loss(delta))
    if not end < start:
        return {}
    return {name: d.detach().numpy().copy() for name, d in delta.items()}
