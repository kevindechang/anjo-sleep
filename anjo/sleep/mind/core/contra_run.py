"""``python -m anjo.sleep.mind.core contra``: contradiction methods against the untrained bar, chosen
on validation and tested once. Research code (torch, CPU, no network): nothing in a night imports it.

**Splits.** Toy (no export), the stance family throughout (``toy``): validation is
``toy.lives(VAL_LIVES, seed=2, stance=True)`` and ``toy.revision_lives(REV_LIVES, seed=2, stance=True)``,
test the same at seed 3, and the pair scorer trains on fresh lives each epoch (both kinds, at their
own seeds). An export
(real latents, ``core-train``): its lives split by template as ``cli train`` splits them (the same
``--val-share`` and ``--seed``), the held-back lives then split by template again into validation
and test halves; the pair scorer trains on the fitted lives.

**Methods.** The pair scorer (``pair_scorer``), one fit per ``--seeds``; and each trained core
named ``--core NAME=PATH`` (repeat a name for its seeds), scored by its peak risk on a thread while
open (``evaluate.traces``). Per method, the run with the best mean validation AUC across the splits
is chosen and tested once. Beside them, always: the untrained scores (``contra_bar``) and, on toy
lives, the oracle with the generating parts. A method counts only if it beats ``unmatched`` (the
constraint matches no earlier event) on the same test candidates: the 95% interval of its AUC over
the bar's (``contra_bar.margin``) lies above zero.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.mind.core import contra_bar, evaluate, pair_scorer, toy
from anjo.sleep.mind.core.api import Core
from anjo.sleep.mind.core.contra_bar import Candidate
from anjo.sleep.mind.core.types import LabelledStream

VAL_LIVES: Final = 512  # toy lives a split (about half hold a contradiction or its decoy)
REV_LIVES: Final = 256  # revision lives a split (every one holds a stale assumption or a follow)
FIT_LIVES: Final = 1024  # fresh toy lives an epoch for the pair scorer, and half as many revision lives
BAR: Final = "unmatched"


@dataclass(frozen=True)
class Split:
    """One split's lives and candidates, by part (toy: ``std`` and ``rev``; an export: ``lives``)."""

    streams: Mapping[str, tuple[LabelledStream, ...]]
    cands: Mapping[str, tuple[Candidate, ...]]
    oracle: Mapping[str, Mapping[str, float]] | None = None  # toy only, by part


def toy_split(seed: int) -> Split:
    streams = {"std": toy.lives(VAL_LIVES, seed=seed, stance=True),
               "rev": toy.revision_lives(REV_LIVES, seed=seed, stance=True)}
    cands = {part: contra_bar.candidates(lives, contra_bar.in_toy) for part, lives in streams.items()}
    parts = {"std": toy.oracle_parts(VAL_LIVES, seed=seed), "rev": toy.oracle_parts(REV_LIVES, seed=seed, revisions=True)}
    oracle = {part: {c.label_id: toy.oracle(c.knowable, c.own, parts[part][c.label_id]) for c in cands[part]}
              for part in cands}
    return Split(streams, cands, oracle)


def export_split(streams: Sequence[LabelledStream]) -> Split:
    lives = tuple(streams)
    return Split({"lives": lives}, {"lives": contra_bar.candidates(lives, contra_bar.in_export)})


def toy_source(epoch: int) -> tuple[Candidate, ...]:
    return (contra_bar.candidates(toy.lives(FIT_LIVES, seed=100_000 + epoch, stance=True), contra_bar.in_toy)
            + contra_bar.candidates(toy.revision_lives(FIT_LIVES // 2, seed=200_000 + epoch, stance=True),
                                    contra_bar.in_toy))


def bar(split: Split) -> dict[str, Any]:
    return {part: contra_bar.bar(cands, None if split.oracle is None else {"oracle": split.oracle[part]})
            for part, cands in split.cands.items()}


def plan(splits: Mapping[str, Split], *, d: int, seeds: Sequence[int], cores: Mapping[str, Sequence[Path]],
         fit_note: str) -> dict[str, Any]:
    """What the run will do, before it trains anything."""
    return {
        "latent_dim": d, "fit": fit_note,
        "splits": {name: {part: {"lives": len(split.streams[part]), "candidates": len(cands),
                                 "conflicts": sum(c.failed for c in cands)} for part, cands in split.cands.items()}
                   for name, split in splits.items()},
        "methods": {"pair_scorer": {"seeds": list(seeds)}, **{name: {"models": [str(p) for p in paths]}
                                                              for name, paths in cores.items()}},
        "selection": "per method, the run with the best mean validation AUC over the splits' parts; tested once",
        "counts_only_if": f"the 95% interval of its test AUC over the untrained '{BAR}' score's lies above zero",
    }


def _aucs(scores: Mapping[str, float], split: Split) -> dict[str, float | None]:
    return {part: contra_bar.auc(scores, cands) for part, cands in split.cands.items()}


def _mean(aucs: Mapping[str, float | None]) -> float:
    known = [v for v in aucs.values() if v is not None]
    return sum(known) / len(known) if known else 0.0


def _core_scores(core: Core, split: Split) -> dict[str, float]:
    return {name: peak for lives in split.streams.values() for run in evaluate.traces(core, lives)
            for name, peak in run.peaks}


def _chosen(runs: Sequence[tuple[str, dict[str, float | None]]]) -> str:
    return max(runs, key=lambda run: _mean(run[1]))[0]


def _verdict(scores: Mapping[str, float], split: Split) -> dict[str, Any]:
    """Per part: the 95% interval of the method's AUC over the bar's, and whether it lies above zero."""
    out: dict[str, Any] = {}
    for part, cands in split.cands.items():
        found = contra_bar.margin(scores, contra_bar.scored(contra_bar.UNTRAINED[BAR], cands), cands)
        out[part] = None if found is None else {"margin": [round(found[0], 4), round(found[1], 4)], "beats": found[0] > 0}
    return out


def run(val: Split, test: Split, *, fit_source: pair_scorer.CandidateSource, d: int, seeds: Sequence[int],
        rank: int, epochs: int, patience: int, cores: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
    """Every method on validation, the chosen run of each on test once, beside the bar."""
    bars = {"val": bar(val), "test": bar(test)}
    flat_val = [c for cands in val.cands.values() for c in cands]
    methods: dict[str, Any] = {}
    fitted = {}
    for seed in seeds:
        fitted[str(seed)] = pair_scorer.fit(fit_source, flat_val, d=d, rank=rank, seed=seed, epochs=epochs,
                                            patience=patience)
    runs = [(seed, _aucs(f.scores(flat_val), val)) for seed, f in fitted.items()]
    pick = _chosen(runs)
    scores = fitted[pick].scores([c for cands in test.cands.values() for c in cands])
    methods["pair_scorer"] = {"val": dict(runs), "chosen": pick, "test": _aucs(scores, test),
                              "beats_bar": _verdict(scores, test),
                              "epochs": {s: [f.best_epoch, f.epochs] for s, f in fitted.items()}, "rank": rank}
    for name, paths in cores.items():
        loaded = {str(path): Core.load(path) for path in paths}
        runs = [(key, _aucs(_core_scores(core, val), val)) for key, core in loaded.items()]
        pick = _chosen(runs)
        scores = _core_scores(loaded[pick], test)
        methods[name] = {"val": dict(runs), "chosen": pick, "test": _aucs(scores, test), "beats_bar": _verdict(scores, test)}
    return {"bar": bars, "methods": methods}
