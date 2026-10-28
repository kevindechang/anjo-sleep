"""The conflict-energy pair scorer: a contradiction method apart from the core, measured against
``contra_bar``'s bar. Research code (torch, CPU): nothing in a night imports it.

A constraint ``y`` is compared with each event it may speak to (its **sources**: every earlier event
of the life, oldest first, the upper bound of what keeping each slot's source events would retain).
Each pair gets two low-rank pair features, ``(Px) * (Py)`` and ``(Px - Py) ** 2`` of a rank ``r``
projection ``P`` of the layer-normed latents, read twice: a **match** (is this source what ``y``
speaks to) and a **conflict energy** (does ``y`` oppose it). The newest matching source carries the
comparison: each source's weight is its match times the miss of every newer one, so a revision
outweighs the plan it revised. The score is a logit, the weighted energies plus a bias. Trained
with binary cross-entropy on labelled candidates, revision lives among them (a constraint that
follows a revision is a negative against the plan it contradicts); the best epoch is the one with
the lowest validation loss.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from anjo.sleep.mind.core.contra_bar import Candidate

CandidateSource = Callable[[int], Sequence[Candidate]]


class PairScorer(nn.Module):  # type: ignore[misc]  # torch's types are skipped (follow_imports)
    def __init__(self, d: int, rank: int) -> None:
        super().__init__()
        self.d, self.rank = d, rank
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.match_proj = nn.Linear(d, rank, bias=False)
        self.match_head = nn.Linear(2 * rank, 1)
        self.energy_proj = nn.Linear(d, rank, bias=False)
        self.energy_head = nn.Linear(2 * rank, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _pair(proj: nn.Linear, head: nn.Linear, sources: Tensor, y: Tensor) -> Tensor:
        a, b = proj(sources), proj(y).unsqueeze(1)
        out: Tensor = head(torch.cat([a * b, (a - b) ** 2], dim=-1)).squeeze(-1)
        return out

    def forward(self, sources: Tensor, mask: Tensor, y: Tensor) -> Tensor:
        """``sources`` ``(B, N, d)`` oldest first, ``mask`` ``(B, N)`` true for a real source, ``y``
        ``(B, d)``: one logit per constraint."""
        s, q = self.norm(sources), self.norm(y)
        match = torch.sigmoid(self._pair(self.match_proj, self.match_head, s, q)) * mask
        energy = self._pair(self.energy_proj, self.energy_head, s, q)
        newest_first = match.flip(1)
        missed = torch.cumprod(torch.cat([torch.ones_like(newest_first[:, :1]), 1 - newest_first[:, :-1]], 1), 1)
        weight = (newest_first * missed).flip(1)
        out: Tensor = self.bias + (weight * energy).sum(1)
        return out


def _batch(cands: Sequence[Candidate], d: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sources padded at the front (the newest last), their mask, the constraints and the labels."""
    width = max(1, max(len(c.earlier) for c in cands))
    sources: np.ndarray = np.zeros((len(cands), width, d), dtype=np.float32)
    mask: np.ndarray = np.zeros((len(cands), width), dtype=np.float32)
    for n, cand in enumerate(cands):
        if cand.earlier:
            sources[n, width - len(cand.earlier):] = np.stack([e.vec for e in cand.earlier])
            mask[n, width - len(cand.earlier):] = 1.0
    y = np.stack([c.knowable.vec for c in cands]).astype(np.float32)
    label = np.array([float(c.failed) for c in cands], dtype=np.float32)
    return torch.from_numpy(sources), torch.from_numpy(mask), torch.from_numpy(y), torch.from_numpy(label)


@dataclass(frozen=True)
class Fitted:
    scorer: PairScorer
    best_epoch: int
    val_loss: float
    epochs: int

    def scores(self, cands: Sequence[Candidate], *, batch_size: int = 64) -> dict[str, float]:
        """Each candidate's logit, by label id."""
        self.scorer.eval()
        out: dict[str, float] = {}
        with torch.no_grad():
            for start in range(0, len(cands), batch_size):
                chunk = cands[start:start + batch_size]
                logits = self.scorer(*_batch(chunk, self.scorer.d)[:3])
                out.update({c.label_id: float(v) for c, v in zip(chunk, logits.tolist(), strict=True)})
        return out


def _loss(scorer: PairScorer, cands: Sequence[Candidate], batch_size: int) -> float:
    scorer.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(cands), batch_size):
            sources, mask, y, label = _batch(cands[start:start + batch_size], scorer.d)
            total += float(nn.functional.binary_cross_entropy_with_logits(scorer(sources, mask, y), label,
                                                                           reduction="sum"))
    return total / max(1, len(cands))


def fit(
    source: CandidateSource, val: Sequence[Candidate], *, d: int, rank: int, seed: int, epochs: int,
    patience: int, batch_size: int = 32, lr: float = 3e-3, weight_decay: float = 0.01,
) -> Fitted:
    """A scorer trained on ``source(epoch)``'s candidates, kept at its best validation loss."""
    if not val:
        raise ValueError("the pair scorer needs validation candidates")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scorer = PairScorer(d, rank)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_loss, best_epoch, epoch = None, math.inf, 0, 0
    for epoch in range(epochs):
        if epoch - best_epoch > patience:
            break
        cands = list(source(epoch))
        order = rng.permutation(len(cands))
        scorer.train()
        for start in range(0, len(cands), batch_size):
            chunk = [cands[int(i)] for i in order[start:start + batch_size]]
            sources, mask, y, label = _batch(chunk, d)
            loss = nn.functional.binary_cross_entropy_with_logits(scorer(sources, mask, y), label)
            opt.zero_grad()
            loss.backward()
            opt.step()
        held = _loss(scorer, val, batch_size)
        if held < best_loss:
            best, best_loss, best_epoch = {k: v.clone() for k, v in scorer.state_dict().items()}, held, epoch
    if best is not None:
        scorer.load_state_dict(best)
    return Fitted(scorer, best_epoch, best_loss, epoch + 1)
