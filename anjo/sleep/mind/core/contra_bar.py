"""The bar a contradiction method must clear, and the sanity checks on the data it is measured on.
Torch-free research code: nothing in a night imports it.

**Candidates.** Each labelled thread a contradiction method is asked about (``select``): in toy
lives the ones ending ``-contradiction`` (a conflict or its decoy, ``in_toy``); in an export's lives
every contradiction, decoy and reversal (``in_export``), the same code on real latents. A candidate is
read at the event that makes it knowable (the constraint), against every event of the life before
it; its own earlier events (the plan, a revision) are kept apart for the timing scores. A method
scores each candidate; ``auc`` is the chance a conflict outscores a non-conflict (ties half), and
``margin`` the 95% interval of a method's AUC over the bar's on the same candidates (paired
bootstrap): a method beats the bar only when the whole interval lies above zero.

**The untrained scores** need no training and set the bar:

- ``unmatched``: the constraint matches no earlier event (one minus its best cosine to any). The bar:
  a method counts only if it beats it.
- ``gap_first`` and ``gap_latest``: hours from the thread's first, or latest, earlier event to the
  constraint. Timing alone: on data where it says nothing, both sit near 0.5.
- ``match_age``: hours back to the earlier event the constraint most resembles. A stale assumption
  resembles an older version of the plan by construction, so on revisions this is a second bar.

A toy life also has an oracle with its generating parts (``toy.oracle``): near 1.0, the latents
carry the conflict.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from anjo.sleep.mind.core.types import EventLatent, LabelledStream, ThreadLabel

Select = Callable[[ThreadLabel], bool]
Score = Callable[["Candidate"], float]
EXPORT_FAMILIES: Final = frozenset({"contradiction", "decoy", "reversal"})
HOUR: Final = 3600.0
BOOTSTRAP: Final = 1000  # resamples for a method's margin over the bar


def in_toy(label: ThreadLabel) -> bool:
    return label.id.endswith("-contradiction")


def in_export(label: ThreadLabel) -> bool:
    return label.family in EXPORT_FAMILIES


@dataclass(frozen=True)
class Candidate:
    """One thread a method is asked about, at the event that makes it knowable."""

    life_id: str
    label_id: str
    failed: bool  # a conflict
    knowable: EventLatent
    own: tuple[EventLatent, ...]  # the thread's events before it, oldest first
    earlier: tuple[EventLatent, ...]  # every event of the life before it, oldest first


def _knowable(stream: LabelledStream, label: ThreadLabel) -> int | None:
    """The stream index of the label's event at its knowable time, else its latest event by then."""
    mine = [n for n, e in enumerate(stream.events) if e.event_id in label.event_ids and e.t <= label.knowable_at]
    exact = [n for n in mine if stream.events[n].t == label.knowable_at]
    found = exact or mine
    return found[-1] if found else None


def candidates(streams: Sequence[LabelledStream], select: Select) -> tuple[Candidate, ...]:
    """Every selected label of every life with an event to read it at, in life and label order."""
    out = []
    for stream in streams:
        for label in stream.labels:
            at = _knowable(stream, label) if select(label) else None
            if at is None:
                continue
            earlier = stream.events[:at]
            out.append(Candidate(stream.life_id, label.id, label.failed, stream.events[at],
                                 tuple(e for e in earlier if e.event_id in label.event_ids), earlier))
    return tuple(out)


def auc(scores: Mapping[str, float], cands: Sequence[Candidate]) -> float | None:
    """The chance a conflict outscores a non-conflict (ties count half); None when a side is empty.
    A candidate the method gave no score counts as 0.0."""
    pos = np.array([scores.get(c.label_id, 0.0) for c in cands if c.failed], dtype=np.float64)
    neg = np.array([scores.get(c.label_id, 0.0) for c in cands if not c.failed], dtype=np.float64)
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def margin(scores: Mapping[str, float], line: Mapping[str, float], cands: Sequence[Candidate], *,
           draws: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float] | None:
    """How far ``scores``' AUC lies above ``line``'s on the same candidates: the 2.5th and 97.5th
    percentiles of the difference over ``draws`` resamples of the candidates (paired). None when a
    side is empty."""
    if auc(scores, cands) is None:
        return None
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        drawn = [cands[int(i)] for i in rng.integers(0, len(cands), len(cands))]
        ours, theirs = auc(scores, drawn), auc(line, drawn)
        if ours is not None and theirs is not None:
            diffs.append(ours - theirs)
    bounds: np.ndarray = np.percentile(diffs, [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])


def _cosines(cand: Candidate) -> np.ndarray:
    if not cand.earlier:
        return np.zeros(0)
    rows = np.stack([e.vec for e in cand.earlier]).astype(np.float64)
    y = cand.knowable.vec.astype(np.float64)
    norms = np.linalg.norm(rows, axis=1) * np.linalg.norm(y)
    out: np.ndarray = rows @ y / np.where(norms > 0, norms, 1.0)
    return out


def unmatched(cand: Candidate) -> float:
    cos = _cosines(cand)
    return 1.0 - float(cos.max()) if len(cos) else 1.0


def _hours(later: EventLatent, earlier: EventLatent) -> float:
    return (later.t - earlier.t).total_seconds() / HOUR


def gap_first(cand: Candidate) -> float:
    return _hours(cand.knowable, cand.own[0]) if cand.own else 0.0


def gap_latest(cand: Candidate) -> float:
    return _hours(cand.knowable, cand.own[-1]) if cand.own else 0.0


def match_age(cand: Candidate) -> float:
    cos = _cosines(cand)
    return _hours(cand.knowable, cand.earlier[int(cos.argmax())]) if len(cos) else 0.0


UNTRAINED: Final[Mapping[str, Score]] = {
    "unmatched": unmatched, "gap_first": gap_first, "gap_latest": gap_latest, "match_age": match_age,
}


def scored(score: Score, cands: Sequence[Candidate]) -> dict[str, float]:
    return {c.label_id: score(c) for c in cands}


def bar(cands: Sequence[Candidate], extra: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, object]:
    """The candidates' counts and each untrained score's AUC (and any ``extra`` score's, by name)."""
    aucs = {name: auc(scored(score, cands), cands) for name, score in UNTRAINED.items()}
    aucs.update({name: auc(scores, cands) for name, scores in (extra or {}).items()})
    return {"candidates": len(cands), "conflicts": sum(c.failed for c in cands),
            "auc": {name: None if value is None else round(value, 4) for name, value in aucs.items()}}
