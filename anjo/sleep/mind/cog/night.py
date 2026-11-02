"""The cog core's night: a few gradient steps on a small per-person adapter over replayed queries.

**The adapter** shifts every head's logit by ``delta = w . z + b + c[s] + t[s, wd]``: ``z`` is the
query's trunk features when it was first predicted, standardized by this person's replay buffer
(``z = (x - mu) / (sd sqrt(H))``, ``mu`` and ``sd`` refreshed each night), so ``w`` is a rank-one
delta on the heads' weights; ``c`` is a bias per subject (each activity, and one for any other
subject) and ``t`` one per subject and target weekday: what a new schedule or a friend moving away
changes. ``H + 1 + 8 S`` numbers per person for ``S`` subjects, shared by the K heads. It is
warm-started night to night and held by a Gaussian prior of weight ``prior / n`` for ``n`` resolved
queries (a MAP fit; the step is proximal, so any weight is stable): with little data it stays near
zero and the slow weights speak.

**What a night replays** (Mattar & Daw 2018, "Prioritized memory access explains planning and
hippocampal replay": replay where the gain is large and the state will be needed). Each night
draws ``samples`` resolved queries with replacement and takes ``steps`` plain gradient steps:

- ``surprise``: drawn with probability ``P_i`` proportional to ``surprise_i x need_i + eps``.
  Surprise is the log loss when the query was first predicted. Need is how much that item speaks
  for the person's coming days: the share of their queries about its subject, times
  ``decay ** (today - day)`` (a successor-style discount; an old surprise says less about them now).
  Each draw is weighted by ``need_i / P_i`` (normalized), so the update is unbiased for the
  need-weighted loss ``sum_i need_i loss_i``: surprise only decides where the samples go (the large
  gradients, so less variance), and does not tilt predictions toward the outcomes it got wrong
  (uncorrected, it shrinks a calibrated core toward 0.5, which we measured).
- ``uniform``: the same number of draws, uniform over the buffer, unweighted, the same steps: the
  equal-compute control.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

NIGHT_MODES: Final = ("off", "uniform", "surprise")
SAMPLES: Final = 128
STEPS: Final = 10
LR: Final = 0.5
PRIOR: Final = 8.0
DECAY: Final = 0.9
_EPS: Final = 1e-3


@dataclass(frozen=True)
class Replayed:
    x: np.ndarray  # (H,) trunk features at the first prediction
    logits: np.ndarray  # (K,) the heads' logits then
    y: float
    surprise: float  # log loss when first predicted
    subject: str
    day: int  # the day it resolved
    cell: tuple[int, int] = (0, 0)  # (subject slot, target weekday)


@dataclass(frozen=True)
class Adapter:
    w: np.ndarray  # (H,)
    b: float
    mu: np.ndarray  # (H,) the buffer's mean features
    sd: np.ndarray  # (H,) and their spread
    c: np.ndarray  # (S,) per subject
    t: np.ndarray  # (S, 7) per subject and target weekday

    @staticmethod
    def zero(h: int, subjects: int = 1) -> Adapter:
        return Adapter(np.zeros(h), 0.0, np.zeros(h), np.ones(h), np.zeros(subjects), np.zeros((subjects, 7)))

    def z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / (self.sd * np.sqrt(len(self.w)))

    def shift(self, x: np.ndarray, cells: np.ndarray | None = None) -> np.ndarray:
        """The logit shift for features (N, H) and cells (N, 2) -> (N,)."""
        out = self.z(x) @ self.w + self.b
        if cells is not None and len(cells):
            out = out + self.c[cells[:, 0]] + self.t[cells[:, 0], cells[:, 1]]
        return out


def need(items: Sequence[Replayed], today: int, counts: Mapping[str, int], decay: float = DECAY) -> np.ndarray:
    total = max(sum(counts.values()), 1)
    return np.array([(counts.get(it.subject, 0) / total) * decay ** max(today - it.day, 0) for it in items])


def priorities(items: Sequence[Replayed], today: int, counts: Mapping[str, int], decay: float = DECAY) -> np.ndarray:
    """Each item's replay probability in ``surprise`` mode: surprise x need, normalized."""
    raw = np.array([it.surprise for it in items]) * need(items, today, counts, decay) + _EPS
    return raw / raw.sum()


def night(
    adapter: Adapter, items: Sequence[Replayed], *, today: int, mode: str, rng: np.random.Generator,
    counts: Mapping[str, int], samples: int = SAMPLES, steps: int = STEPS, lr: float = LR, prior: float = PRIOR,
    decay: float = DECAY,
) -> Adapter:
    """One night's adapter, from the last one and the replay buffer."""
    if mode not in NIGHT_MODES:
        raise ValueError(f"night mode is one of {NIGHT_MODES}")
    if mode == "off" or not items:
        return adapter
    xs = np.stack([it.x for it in items])
    fresh = Adapter(adapter.w, adapter.b, xs.mean(axis=0), xs.std(axis=0) + 1e-3, adapter.c, adapter.t)
    if mode == "surprise":
        prob = priorities(items, today, counts, decay)
        pick = rng.choice(len(items), size=samples, replace=True, p=prob)
        iw = need(items, today, counts, decay)[pick] / prob[pick]
        iw = iw / iw.sum()
    else:
        pick = rng.choice(len(items), size=samples, replace=True)
        iw = np.full(samples, 1.0 / samples)
    z = fresh.z(xs[pick])
    base = np.stack([items[i].logits for i in pick])
    y = np.array([items[i].y for i in pick])
    cells = np.array([items[i].cell for i in pick], dtype=np.int64)
    s_idx, wd_idx = cells[:, 0], cells[:, 1]
    shrink = 1.0 / (1.0 + 2.0 * lr * prior / len(items))  # the prior's proximal step
    w, b, c, t = fresh.w.copy(), fresh.b, fresh.c.copy(), fresh.t.copy()
    for _ in range(steps):
        shift = z @ w + b + c[s_idx] + t[s_idx, wd_idx]
        pk = 1.0 / (1.0 + np.exp(-np.clip(base + shift[:, None], -30.0, 30.0)))
        p = np.clip(pk.mean(axis=1), 1e-6, 1 - 1e-6)
        g = iw * (p - y) / (p * (1 - p)) * (pk * (1 - pk)).mean(axis=1)  # d loss / d shift, per draw
        gc, gt = np.zeros_like(c), np.zeros_like(t)
        np.add.at(gc, s_idx, g)
        np.add.at(gt, (s_idx, wd_idx), g)
        w, b = (w - lr * (z.T @ g)) * shrink, (b - lr * float(g.sum())) * shrink
        c, t = (c - lr * gc) * shrink, (t - lr * gt) * shrink
    return Adapter(w, float(b), fresh.mu, fresh.sd, c, t)
