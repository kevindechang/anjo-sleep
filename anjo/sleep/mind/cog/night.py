"""The cog core's night: a few gradient steps on a small per-person adapter over replayed predictions.

**What is replayed.** Each day the core predicts, for every activity, whether it happens tomorrow
(its probes, made inside ``observe``, so they never depend on who called ``predict`` or how often);
the next day's events resolve them (``encode.outcomes``). The replay buffer holds those resolved
probes: the heads' logits then, the outcome, the surprise (the log loss of the prediction then,
adapter included) and the cell (activity, weekday).

**The adapter** shifts every head's logit by ``delta = b + c[s] + t[s, wd]``: a bias, one per subject
(each activity, and one slot for any other subject) and one per subject and target weekday. That is
what a new schedule or a friend moving away changes: ``1 + 8 S`` numbers per person for ``S``
subjects, shared by the K heads, the same for probes and queries. It is warm-started night to night
and held by a Gaussian prior of weight ``prior / n`` for ``n`` replayed items (a MAP fit; the
prior's step is proximal, so any weight is stable): with little data it stays near zero and the
slow weights speak.

**Which items a night replays** (Mattar & Daw 2018, "Prioritized memory access explains planning
and hippocampal replay": replay where the gain is large and the state will be needed). Each night
draws ``samples`` items with replacement and takes ``steps`` plain gradient steps:

- ``surprise``: drawn with probability ``P_i`` proportional to ``surprise_i x need_i + eps``. Need is
  how much the item speaks for the person's coming days: the share of their events about its
  subject, times ``decay ** (today - day)`` (a successor-style discount: an old surprise says less
  about them now). Each draw is weighted by ``need_i / P_i`` (normalized), so the update is unbiased
  for the need-weighted loss ``sum_i need_i loss_i``: surprise decides where the samples go (the
  large gradients, so less variance) and does not tilt predictions toward the outcomes the core got
  wrong (uncorrected, it shrank a calibrated core toward 0.5 on the toy world).
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
    logits: np.ndarray  # (K,) the heads' logits when predicted
    y: float
    surprise: float  # log loss when predicted
    subject: str
    day: int  # the day it resolved
    cell: tuple[int, int] = (0, 0)  # (subject slot, weekday of the day it was about)


@dataclass(frozen=True)
class Adapter:
    b: float
    c: np.ndarray  # (S,) per subject
    t: np.ndarray  # (S, 7) per subject and weekday

    @staticmethod
    def zero(subjects: int = 1) -> Adapter:
        return Adapter(0.0, np.zeros(subjects), np.zeros((subjects, 7)))

    def shift(self, cells: np.ndarray) -> np.ndarray:
        """The logit shift for cells (N, 2) of (subject slot, weekday) -> (N,)."""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        return self.b + self.c[cells[:, 0]] + self.t[cells[:, 0], cells[:, 1]]


def need(items: Sequence[Replayed], today: int, counts: Mapping[str, int], decay: float = DECAY) -> np.ndarray:
    """Share of the person's events about the item's subject (add-one, so a subject not yet seen
    still counts a little) times the recency discount."""
    total = sum(counts.values()) + len({it.subject for it in items})
    return np.array([((counts.get(it.subject, 0) + 1) / total) * decay ** max(today - it.day, 0) for it in items])


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
    if mode == "surprise":
        prob = priorities(items, today, counts, decay)
        pick = rng.choice(len(items), size=samples, replace=True, p=prob)
        iw = need(items, today, counts, decay)[pick] / prob[pick]
        iw = iw / iw.sum()
    else:
        pick = rng.choice(len(items), size=samples, replace=True)
        iw = np.full(samples, 1.0 / samples)
    base = np.stack([items[i].logits for i in pick])
    y = np.array([items[i].y for i in pick])
    cells = np.array([items[i].cell for i in pick], dtype=np.int64)
    s_idx, wd_idx = cells[:, 0], cells[:, 1]
    shrink = 1.0 / (1.0 + 2.0 * lr * prior / len(items))  # the prior's proximal step
    b, c, t = adapter.b, adapter.c.copy(), adapter.t.copy()
    for _ in range(steps):
        shift = b + c[s_idx] + t[s_idx, wd_idx]
        pk = 1.0 / (1.0 + np.exp(-np.clip(base + shift[:, None], -30.0, 30.0)))
        p = np.clip(pk.mean(axis=1), 1e-6, 1 - 1e-6)
        g = iw * (p - y) / (p * (1 - p)) * (pk * (1 - pk)).mean(axis=1)  # d loss / d shift, per draw
        gc, gt = np.zeros_like(c), np.zeros_like(t)
        np.add.at(gc, s_idx, g)
        np.add.at(gt, (s_idx, wd_idx), g)
        b = (b - lr * float(g.sum())) * shrink
        c, t = (c - lr * gc) * shrink, (t - lr * gt) * shrink
    return Adapter(float(b), c, t)
