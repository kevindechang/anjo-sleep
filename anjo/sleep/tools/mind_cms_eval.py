"""MIND's continuum memory measured: ``ANJO_SLEEP_MIND_CMS`` off against on (``mind.core.cms``), offline.

    python -m anjo.sleep.tools.mind_cms_eval [--seeds 0 1 2 3 4] [--jobs 4] [--out FILE] [--json FILE]
    python -m anjo.sleep.tools.mind_cms_eval --export DIR --latents DIR [...]   # real latents

Each seed trains two cores on the same lives with the same recipe (``test_mind_core``'s: fresh toy
lives each epoch until validation stops improving, thresholds locked on validation), the switch off
and on, and scores both on the same held-out lives. Every number is per seed; the table gives the
mean over seeds with a 95% t-interval, and the paired difference (on minus off, seed by seed) with its own.

- **The existing evaluations**, as their tests run them: the planted dependency's AUC (``separation``,
  gap threads against their decoys) and recall at the locked threshold, with no night between days
  (the tests' protocol) and with a consolidation each night (the live night's); the transplant
  (flips of 16, the booking's risk gap, the unrelated thread's mean shift); the night's rehearsal
  (dream picks against random and against the newest events, the needed threads' next-day
  surprise); consolidation on long chained lives (recall each night consolidating, never resetting,
  or resetting: with the switch on, ``reset`` clears the fast level only and the night level never steps).
- **Retention under load** (new): each held-out life is cut at its middle day and ``F`` whole other
  lives are lived in between (``probe``), every event of theirs one more association stored before
  the life's second half asks for its first: the booking whose risk only the memory of a party's
  earlier cancellations tells. Swept over ``F`` (``--fillers``), nights consolidating (the live
  protocol) and not; scored by the AUC of the second half's threads and their recall at the
  locked threshold. Nested Learning's claim for a continuum memory is that the slower levels keep
  what the fast one has overwritten, so this is where it should show.

``--export DIR --latents DIR`` runs the same on an export's lives read through the real encoder:
``latents`` is a folder that ``python -m anjo.sleep.mind.core.latents encode`` wrote on the GPU (it
plans first and holds the budget; this script never loads a model or calls a service). The lives
are split as the core CLI's ``train`` and ``contra`` split them (fit, validation, test by template);
the transplant needs toy pairs and is left out, and long lives chain the test lives.
CPU only, deterministic per seed; ``--jobs`` trains seeds and arms in parallel processes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Final

import numpy as np
import torch

from anjo.sleep.mind.core import cli, data, evaluate, sleep_eval, toy, train
from anjo.sleep.mind.core.api import Core
from anjo.sleep.mind.core.state import CoreState
from anjo.sleep.mind.core.types import CoreConfig, LabelledStream, ThreadLabel

ARMS: Final = ("off", "on")
# Student's t at 0.975 by degrees of freedom; past the table, the normal 1.96.
_T975: Final = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)
Metrics = dict[str, float | None]


@dataclass(frozen=True)
class Plan:
    """What one run measures. ``TOY`` is the recipe the core's tests train; the test runs a tiny one."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 200
    patience: int = 20
    width: int = 64
    slots: int = 8
    lives_per_epoch: int = 128
    val_lives: int = 48
    test_lives: int = 64
    pairs: int = 16
    chains: int = 16
    links: int = 5
    fillers: tuple[int, ...] = (0, 1, 2, 4, 8)
    probes: int = 48
    rehearsal_lives: int = 32
    export: Path | None = None
    latents: Path | None = None


@dataclass(frozen=True)
class Lives:
    fit: train.LifeSource
    val: tuple[LabelledStream, ...]
    test: tuple[LabelledStream, ...]
    pairs: tuple[toy.TransplantPair, ...]
    chained: tuple[LabelledStream, ...]
    fillers: tuple[LabelledStream, ...]


# ── Lives ─────────────────────────────────────────────────────────────────────


def _toy_epoch(n: int, epoch: int) -> tuple[LabelledStream, ...]:
    lives: tuple[LabelledStream, ...] = toy.lives(n, seed=1000 + epoch)
    return lives


def _shift(stream: LabelledStream, person: str, by: timedelta) -> LabelledStream:
    events = tuple(replace(e, life_or_person=person, t=e.t + by) for e in stream.events)
    labels = tuple(replace(lab, knowable_at=lab.knowable_at + by, outcome_at=lab.outcome_at + by) for lab in stream.labels)
    return LabelledStream(person, tuple(d + by for d in stream.days), events, labels)


def chain(streams: Sequence[LabelledStream], person: str) -> LabelledStream:
    """``streams`` lived one after another by one person, each from the day after the last one's end."""
    parts, at = [], streams[0].days[0]
    for stream in streams:
        moved = _shift(stream, person, timedelta(days=(at - stream.days[0]).days))
        parts.append(moved)
        at = moved.days[-1] + timedelta(days=1)
    return LabelledStream(person, tuple(d for p in parts for d in p.days), tuple(e for p in parts for e in p.events),
                          tuple(lab for p in parts for lab in p.labels))


def _late(label: ThreadLabel, times: Mapping[str, date], cut: date) -> bool:
    return label.knowable_at.date() >= cut and all(times[eid] >= cut for eid in label.event_ids)


def probe(stream: LabelledStream, fillers: Sequence[LabelledStream]) -> LabelledStream:
    """``stream`` cut at its middle day with ``fillers`` lived in between (their labels dropped): the
    second half's threads, the only labels kept, ask for what the first half stored."""
    cut = stream.days[len(stream.days) // 2]
    times = {e.event_id: e.t.date() for e in stream.events}
    early = LabelledStream(stream.life_id, tuple(d for d in stream.days if d < cut),
                           tuple(e for e in stream.events if e.t.date() < cut), ())
    late = LabelledStream(stream.life_id, tuple(d for d in stream.days if d >= cut),
                          tuple(e for e in stream.events if e.t.date() >= cut),
                          tuple(lab for lab in stream.labels if _late(lab, times, cut)))
    between = tuple(replace(f, labels=()) for f in fillers)
    whole = chain([early, *between, late], stream.life_id) if early.days else chain([*between, late], stream.life_id)
    return replace(whole, days=tuple(sorted(set(whole.days))))


def _toy_lives(plan: Plan) -> Lives:
    pool = toy.lives(plan.probes * max(plan.fillers, default=0), seed=7)
    return Lives(partial(_toy_epoch, plan.lives_per_epoch), toy.lives(plan.val_lives, seed=2),
                 toy.lives(plan.test_lives, seed=3), toy.pairs(plan.pairs, seed=5),
                 toy.chained(plan.chains, seed=3, links=plan.links), pool)


def _export_lives(plan: Plan, seed: int) -> Lives:
    if plan.export is None or plan.latents is None:
        raise ValueError("--export and --latents go together")
    streams = data.load_export(plan.export, plan.latents)
    fit, held = cli.split_lives(streams, 0.2, seed)
    val, test = cli.split_lives(held, 0.5, seed + 1)
    chained = tuple(chain(test[i:i + plan.links], f"chain{i}") for i in range(0, len(test) - plan.links + 1, plan.links))
    return Lives(tuple(fit), tuple(val), tuple(test), (), chained, tuple(fit))


# ── One core ──────────────────────────────────────────────────────────────────


def _is_gap(label: ThreadLabel) -> bool:
    return label.id.endswith("-gap") or label.family == "gap"


def _nightly(core: Core) -> evaluate.NightOp:
    def night(state: CoreState, weights: Mapping[str, float]) -> CoreState:
        return core.consolidate(state)

    return night


def _headline(core: Core, lives: Lives) -> Metrics:
    out: Metrics = {}
    for protocol, op in (("", None), ("_nightly", _nightly(core))):
        runs = evaluate.traces(core, lives.test, nightly=op)
        head = evaluate.score(runs, lives.test, core.threshold)
        out[f"separation{protocol}"] = evaluate.separation(runs, lives.test, _is_gap)
        out[f"recall{protocol}"], out[f"false_per_100{protocol}"] = head.recall, head.false_per_100
    return out


def _transplant(core: Core, lives: Lives) -> Metrics:
    if not lives.pairs:
        return {}
    result = evaluate.transplant(core, lives.pairs, tolerance=0.1)
    return {"transplant_flipped": float(result.flipped), "transplant_relevant_gap": result.relevant_gap,
            "transplant_unrelated_shift": result.mean_unrelated_shift,
            "transplant_unrelated_shift_all": result.mean_unrelated_shift_all}


def _nights(core: Core, lives: Lives, seed: int, rehearsal_lives: int) -> Metrics:
    out: Metrics = {}
    for score in sleep_eval.consolidation(core, lives.chained, sleep_eval.night_ops(core)):
        out[f"chained_recall_{score.arm}"], out[f"chained_auc_{score.arm}"] = score.recall, score.auc
    _, gains = sleep_eval.rehearsal_gain(core, lives.test[:rehearsal_lives], seed=seed)
    for gain in gains:
        if gain.scope == "needed" and gain.arm == "dream" and gain.base in ("random", "recent"):
            out[f"rehearsal_dream_minus_{gain.base}"] = gain.mean
    return out


def _retention(core: Core, lives: Lives, fillers: Sequence[int], probes: int) -> Metrics:
    asked = [s for s in lives.test if probe(s, ()).labels][:probes]
    out: Metrics = {}
    for n in fillers:
        streams = [probe(s, lives.fillers[i * n:(i + 1) * n]) for i, s in enumerate(asked)]
        out[f"stored@{n}"] = float(np.mean([len(s.events) for s in streams]))
        for protocol, op in (("nightly", _nightly(core)), ("no_nights", None)):
            runs = evaluate.traces(core, streams, nightly=op)
            out[f"retention_auc_{protocol}@{n}"] = evaluate.separation(runs, streams)
            out[f"retention_recall_{protocol}@{n}"] = evaluate.score(runs, streams, core.threshold).recall
    return out


def job(plan: Plan, arm: str, seed: int) -> tuple[str, int, Metrics]:
    """One core, trained and scored: ``arm`` is the switch, off or on."""
    torch.set_num_threads(1)
    lives = _toy_lives(plan) if plan.export is None else _export_lives(plan, seed)
    dim = lives.val[0].events[0].vec.shape[0]
    config = CoreConfig(dim_in=int(dim), width=plan.width, slots=plan.slots, cms=arm == "on")
    start = time.perf_counter()
    trained = train.train_shared(lives.fit, lives.val, config, seed=seed, epochs=plan.epochs, patience=plan.patience,
                                 batch_size=16)
    core = trained.core
    metrics: Metrics = {"params": float(core.params), "epochs": float(len(trained.log)),
                        "train_s": time.perf_counter() - start}
    metrics.update(_headline(core, lives))
    metrics.update(_transplant(core, lives))
    metrics.update(_nights(core, lives, seed, plan.rehearsal_lives))
    metrics.update(_retention(core, lives, plan.fillers, plan.probes))
    return arm, seed, metrics


# ── Across seeds ──────────────────────────────────────────────────────────────


def interval(values: Sequence[float]) -> tuple[float, float, float] | None:
    """The mean and its 95% t-interval; ``None`` for no values, the mean alone for one."""
    if not values:
        return None
    mean = math.fsum(values) / len(values)
    if len(values) < 2:
        return mean, mean, mean
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))
    t = _T975[len(values) - 2] if len(values) - 2 < len(_T975) else 1.96
    half = t * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def summarize(results: Sequence[tuple[str, int, Metrics]]) -> dict[str, dict[str, tuple[float, float, float] | None]]:
    """Per metric: each arm's mean and interval over seeds, and the paired difference (on - off)."""
    by = {(arm, seed): m for arm, seed, m in results}
    seeds = sorted({seed for _, seed, _ in results})
    names = list(dict.fromkeys(name for _, _, m in results for name in m))
    out = {}
    for name in names:
        def got(arm: str, seed: int) -> float | None:
            return by.get((arm, seed), {}).get(name)

        row = {arm: interval([v for s in seeds if (v := got(arm, s)) is not None]) for arm in ARMS}
        paired = [b - a for s in seeds if (a := got("off", s)) is not None and (b := got("on", s)) is not None]
        out[name] = {**row, "on_minus_off": interval(paired)}
    return out


def _cell(held: tuple[float, float, float] | None, sign: str = "") -> str:
    if held is None:
        return "n/a"
    mean, low, high = held
    return f"{mean:{sign}.3f} [{low:{sign}.3f}, {high:{sign}.3f}]" if low != high else f"{mean:{sign}.3f}"


def table(summary: Mapping[str, Mapping[str, tuple[float, float, float] | None]], seeds: Sequence[int]) -> str:
    lines = [f"MIND continuum memory (ANJO_SLEEP_MIND_CMS) off vs on; seeds {list(seeds)}; mean [95% t-interval]",
             f"{'metric':<38} {'off':<28} {'on':<28} {'on - off (paired)':<28}"]
    lines.extend(f"{name:<38} {_cell(row['off']):<28} {_cell(row['on']):<28} {_cell(row['on_minus_off'], '+'):<28}"
                 for name, row in summary.items())
    return "\n".join(lines)


def run(plan: Plan, *, jobs: int = 1) -> tuple[list[tuple[str, int, Metrics]], str]:
    tasks = [(arm, seed) for seed in plan.seeds for arm in ARMS]
    if jobs <= 1:
        results = [job(plan, arm, seed) for arm, seed in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            results = list(pool.map(job, [plan] * len(tasks), *zip(*tasks)))
    return results, table(summarize(results), plan.seeds)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m anjo.sleep.tools.mind_cms_eval", description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(Plan.seeds))
    p.add_argument("--epochs", type=int, default=Plan.epochs)
    p.add_argument("--patience", type=int, default=Plan.patience)
    p.add_argument("--fillers", type=int, nargs="+", default=list(Plan.fillers), help="lives lived between a probe's halves")
    p.add_argument("--probes", type=int, default=Plan.probes)
    p.add_argument("--export", type=Path, default=None, help="an export's lives instead of toy lives")
    p.add_argument("--latents", type=Path, default=None, help="the export's latents, from the real encoder")
    p.add_argument("--jobs", type=int, default=1, help="cores trained at once, one process each")
    p.add_argument("--out", type=Path, default=None, help="the table, as text")
    p.add_argument("--json", type=Path, default=None, help="every seed's metrics, as JSON")
    return p


def _write(path: Path | None, text: str) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None, *, emit: Callable[[str], object] = print) -> int:
    args = _parser().parse_args(argv)
    if (args.export is None) != (args.latents is None):
        emit("error: --export and --latents go together")
        return 2
    plan = Plan(seeds=tuple(args.seeds), epochs=args.epochs, patience=args.patience, fillers=tuple(args.fillers),
                probes=args.probes, export=args.export, latents=args.latents)
    try:
        results, text = run(plan, jobs=args.jobs)
    except (data.DataError, OSError, ValueError) as error:
        emit(f"error: {error}")
        return 2
    _write(args.out, text + "\n")
    _write(args.json, json.dumps({"plan": asdict(plan), "results": results}, indent=1, default=str) + "\n")
    emit(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
