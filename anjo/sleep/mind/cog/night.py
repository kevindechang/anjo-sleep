"""The cog core's night: a few gradient steps on a small per-person adapter over replayed predictions.

**What is replayed.** The person's resolved queries (``core.MindPredictor``: each of a day's
queries is predicted from that day's final state, and resolves on its target day): the heads'
logits then, the outcome, the surprise (the log loss of the prediction then, adapter included) and
the cell (subject, target weekday). Replaying the core's own next-day probes instead was tried and
dropped: the adapter then carried the probes' errors to queries whose errors differ (it raised a
weakly trained core's log loss on a real life from 0.687 to 0.769).

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
  how much the item speaks for the person's coming days: the share of their queries about its
  subject, times ``decay ** (today - day)`` (a successor-style discount: an old surprise says less
  about them now). Each draw is weighted by ``need_i / P_i`` (normalized), so the update is unbiased
  for the need-weighted loss ``sum_i need_i loss_i``: surprise decides where the samples go (the
  large gradients, so less variance) and does not tilt predictions toward the outcomes the core got
  wrong (uncorrected, it shrank a calibrated core toward 0.5 on the toy world).
- ``uniform``: the same number of draws, uniform over the buffer, unweighted, the same steps: the
  equal-compute control.

**The adapter is served only while it has paid** (``served``, ``update_gain``, ``paid``). The night
always fits it, but predictions use it only while its gain is ``Z`` standard errors above zero: the
log loss its own predictions saved over the core's alone on the items resolved so far, each
counted from before its outcome was known (prequential), decayed by ``GAIN_DECAY`` a day, against
the root of the decayed sum of the squared savings. A core whose state already learns the person
online has seen the outcomes the night replays, so a bias fit to them counts that evidence twice:
on 40 real-world lives no core trained on (seeds 7000-7039, a core trained on 3000 lives) the
ungated night raised the log loss by 0.020 nats (uniform replay by 0.006), and gated by 0.004
(0.002). The gate starts closed, so a person's first predictions are the core's own.

**The transfer night** (``transfer``; complementary learning systems, McClelland, McNaughton &
O'Reilly 1995) replays nothing. It moves a share ``move`` of the fast memory into the adapter: the
predictor scales the fast weights and their momentum by ``1 - move``, and each of the day's queries
gives the logit that removed (the heads' mean, the full memory's minus the scaled one's). Each
measured cell's adapter gains the mean of its queries, and the subject's other weekdays the mean
of the subject's (``transfer``). The moved part is always served: it is no fit to outcomes that
the gate would have to vouch for. It conserves the evidence only if the memory stays scaled down,
and this core's does not: its write moves ``M k`` toward the fixed value ``W_v [y, 1 - y]``
whatever the adapter holds, so a scaled-down memory relearns the same content by the next night
and the adapter takes it again. On 40 real-world lives no core trained on (seeds 7000-7039, the
6000-life tally core), the memory's norm settled at 1.2 instead of 4.4, the adapter's subject
shifts reached 2-3 logits, and the log loss rose by 0.145 nats at ``move`` 0.5 (0.082 at 0.2,
0.180 at 1; hidden queries, 0.62 -> 0.88). A transfer that conserves needs a core trained with
it, whose memory learns the residual of the adapter.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

NIGHT_MODES: Final = ("off", "uniform", "surprise", "transfer")
SAMPLES: Final = 128
STEPS: Final = 10
LR: Final = 0.5
PRIOR: Final = 8.0
MOVE: Final = 0.5  # the share of the fast memory a transfer night moves
DECAY: Final = 0.9
GAIN_DECAY: Final = 0.9  # per day: the gate weighs about the last ten days
Z: Final = 2.0  # the gate opens on a saving two standard errors above zero
Gain = tuple[float, float]  # (decayed sum of the savings, decayed sum of their squares)
NO_GAIN: Final[Gain] = (0.0, 0.0)
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


def _loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def update_gain(gain: Gain, resolved: Sequence[Replayed], decay: float = GAIN_DECAY) -> Gain:
    """The gate's evidence after a day: the old sums decayed, plus each resolved item's saving, its
    base log loss minus the loss it had with the adapter (its ``surprise``), both from before the
    outcome (and its square)."""
    total, sq = decay * gain[0], decay * gain[1]
    if not resolved:
        return total, sq
    base = np.stack([it.logits for it in resolved])
    p_base = (1.0 / (1.0 + np.exp(-np.clip(base, -30.0, 30.0)))).mean(axis=1)
    y = np.array([it.y for it in resolved])
    saved = _loss(p_base, y) - np.array([it.surprise for it in resolved])
    return total + float(saved.sum()), sq + float((saved ** 2).sum())


def paid(gain: Gain, z: float = Z) -> bool:
    return gain[0] > z * float(np.sqrt(gain[1])) and gain[0] > 0.0


def served(adapter: Adapter, gain: Gain) -> Adapter:
    """The adapter predictions use: the night's while it has paid, else none."""
    return adapter if paid(gain) else Adapter.zero(len(adapter.c))


def transfer(adapter: Adapter, cells: np.ndarray, moved: np.ndarray) -> Adapter:
    """The adapter after a transfer night: ``moved`` (N,) is the logit each query (cells (N, 2))
    lost when the fast memory was scaled down. A measured cell gains its queries' mean; the other
    weekdays of a measured subject gain the subject's mean; other subjects, nothing."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    moved = np.asarray(moved, dtype=np.float64).reshape(-1)
    if not len(moved):
        return adapter
    c, t = adapter.c.copy(), adapter.t.copy()
    for slot in np.unique(cells[:, 0]):
        mine = cells[:, 0] == slot
        mean = float(moved[mine].mean())
        c[slot] += mean
        for wd in np.unique(cells[mine, 1]):
            t[slot, wd] += float(moved[mine & (cells[:, 1] == wd)].mean()) - mean
    return Adapter(adapter.b, c, t)


def night(
    adapter: Adapter, items: Sequence[Replayed], *, today: int, mode: str, rng: np.random.Generator,
    counts: Mapping[str, int], samples: int = SAMPLES, steps: int = STEPS, lr: float = LR, prior: float = PRIOR,
    decay: float = DECAY,
) -> Adapter:
    """One night's adapter, from the last one and the replay buffer."""
    if mode not in NIGHT_MODES:
        raise ValueError(f"night mode is one of {NIGHT_MODES}")
    if mode in ("off", "transfer") or not items:  # the transfer is the predictor's (it reads the net)
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
