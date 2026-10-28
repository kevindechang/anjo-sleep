"""``python -m anjo.sleep.mind.core``: train MIND's core and run it offline (torch, CPU, no network).

- ``toy --out DIR``: train on generated toy lives (fresh ones each epoch) and evaluate on held-out
  ones, the transplant pairs included. A smoke run of the whole chain, minutes on one CPU thread.
- ``train --export DIR --latents DIR --out DIR``: train on an export's lives (a labelled split,
  ``core-train``). About ``--val-share`` of its lives are held back by template (``split_lives``),
  for the best epoch and the locked thresholds, since ``core-val``'s labels stay sealed.
  ``--no-fast`` trains N-nofast. Both train until validation stops improving (``--patience``
  epochs), within ``--epochs``.
- ``findings --model FILE --export DIR --latents DIR --out FILE``: run the core over each life,
  night by night, and write its calls as APD finding records (one JSON object a line, with the
  life's id and the arm label: ``core-N``, ``core-N-reset``, ``core-N-nofast``, kept apart from APD's
  pre-registered arm names), quoting the cited evidence as each record's problem.
  ``--reset-each-night`` is N-reset; ``--dream`` adds each night's ignitions, kept at the dream's
  own threshold (locked on the same held-back lives as the speak threshold). ``--threshold``
  reports at another operating point than the locked one, and every record and the summary then
  say so (``threshold_by_hand``): such a run is exploration, not a score. Prints the compute it took.
- ``render --runs DIR --arm B --export DIR --out FILE``: another arm's stored APD runs
  (``DIR/<life>.<arm>.json``) as finding records in the core's form (``findings.in_core_form``:
  the first two cited items quoted, a fixed conclusion), for one blind judge pass over both arms.
  Their cites are the export's event ids. The headline for both is ``prescore``; the judge's
  reading of B's own prose stays secondary.
- ``prescore --findings FILE --labels DIR``: the deterministic pre-score against the planted items
  in ``DIR/<life>.json`` (the export's ``labels``, or the sealed items in the same shape).

``toy`` and ``train`` write ``core.npz`` (the weights and thresholds; run data, never in git) and
``model_card.json``: parameters, data, seed, losses, thresholds, metrics and the time per event.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from anjo.sleep.mind.apd.store import loads
from anjo.sleep.mind.apd.types import ArmRun, FindingReport
from anjo.sleep.mind.core import data, evaluate, findings, toy, train
from anjo.sleep.mind.core.api import Core
from anjo.sleep.mind.core.types import CoreConfig, LabelledStream

MODEL_FILE: Final = "core.npz"
CARD_FILE: Final = "model_card.json"
TOY_LIVES_PER_EPOCH: Final = 128
QUOTE_CHARS: Final = 160  # of each cited item, in a finding's problem
ARM_NAME: Final = re.compile(r"[A-Za-z0-9-]{1,32}")


def timing(core: Core, streams: Sequence[LabelledStream], *, limit: int = 200) -> dict[str, float]:
    """Milliseconds per observed event and per prediction, on this CPU, over up to ``limit`` events."""
    events = [e for s in streams for e in s.events][:limit]
    state = core.init_state("timing")
    start = time.perf_counter()
    for event in events:
        state, _ = core.observe(state, event)
    observed = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(20):
        core.predict(state)
    return {"observe_ms_per_event": round(1000 * observed / max(len(events), 1), 3),
            "predict_ms": round(1000 * (time.perf_counter() - start) / 20, 3), "events_timed": len(events)}


def metrics(core: Core, streams: Sequence[LabelledStream]) -> dict[str, Any]:
    runs = evaluate.traces(core, streams)
    return {"lives": len(streams), "headline": dataclasses.asdict(evaluate.score(runs, streams, core.threshold)),
            "separation": evaluate.separation(runs, streams)}


def _card(trained: train.Trained, seconds: float, data_note: dict[str, Any], seed: int, held: Sequence[LabelledStream],
          extra: dict[str, Any]) -> dict[str, Any]:
    core = trained.core
    return {
        "model_id": core.model_id, "params": core.params, "config": dataclasses.asdict(core.config),
        "data": data_note, "seed": seed, "epochs": len(trained.log), "best_epoch": trained.best_epoch,
        "log": [list(row) for row in trained.log], "thresholds": dataclasses.asdict(core.thresholds),
        "metrics": {"held_back": metrics(core, held), **extra},
        "timing": {"train_s": round(seconds, 1), **timing(core, held)}, "torch": torch.__version__, "device": "cpu",
    }


def _write(out: Path, core: Core, card: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    core.save(out / MODEL_FILE)
    (out / CARD_FILE).write_text(json.dumps(card, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run_toy(args: argparse.Namespace) -> dict[str, Any]:
    config = CoreConfig(dim_in=toy.DIM, width=args.width, slots=args.slots, fast=not args.no_fast)
    start = time.perf_counter()
    trained = train.train_shared(lambda epoch: toy.lives(TOY_LIVES_PER_EPOCH, seed=1000 + epoch),
                                 toy.lives(48, seed=2), config, seed=args.seed, epochs=args.epochs,
                                 patience=args.patience, batch_size=16)
    seconds = time.perf_counter() - start
    test = toy.lives(64, seed=3)
    pairs = evaluate.transplant(trained.core, toy.pairs(16, seed=5), tolerance=0.1)
    note = {"source": "toy", "lives_per_epoch": TOY_LIVES_PER_EPOCH, "validation": 48, "test": 64, "dim": toy.DIM}
    card = _card(trained, seconds, note, args.seed, test, {"transplant": dataclasses.asdict(pairs)})
    _write(args.out, trained.core, card)
    return card


def _templates(stream: LabelledStream) -> frozenset[str]:
    return frozenset(label.template for label in stream.labels if label.template)


def split_lives(streams: Sequence[LabelledStream], share: float, seed: int) -> tuple[list[LabelledStream], list[LabelledStream]]:
    """``streams`` into fit and held-back lives, held back by template as ``core-test`` is (which
    is held out by field as well). Templates are drawn in the seed's order until the lives using
    a drawn one reach ``share`` of the lives with any template; every such life is held back, so no
    fitted life holds a drawn template. A template that would leave nothing to fit on is skipped.
    Lives with no template (problem-free ones) are held back at the same share, picked by the seed.
    Both sides keep the export's order."""
    rng = np.random.default_rng(seed)
    templated = [s for s in streams if _templates(s)]
    names = sorted(frozenset().union(*(_templates(s) for s in templated)))
    drawn: set[str] = set()
    for name in (names[int(i)] for i in rng.permutation(len(names))):
        if sum(1 for s in templated if _templates(s) & drawn) >= share * len(templated):
            break
        if not all(_templates(s) & (drawn | {name}) for s in templated):
            drawn.add(name)
    plain = [s for s in streams if not _templates(s)]
    count = round(share * len(plain)) if templated else min(max(1, round(share * len(plain))), len(plain) - 1)
    held_ids = ({s.life_id for s in templated if _templates(s) & drawn}
                | {plain[int(i)].life_id for i in rng.permutation(len(plain))[:max(count, 0)]})
    fit, held = [s for s in streams if s.life_id not in held_ids], [s for s in streams if s.life_id in held_ids]
    if not fit or not held:
        raise data.DataError(f"can't hold lives back by template: {len(fit)} to fit, {len(held)} held back")
    return fit, held


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    streams = data.load_export(args.export, args.latents)
    if len(streams) < 2 or not any(s.labels for s in streams):
        raise data.DataError("training needs at least two lives of a labelled split")
    header, _ = data.read_latents(args.latents)
    fit, held = split_lives(streams, args.val_share, args.seed)
    config = CoreConfig(dim_in=int(header["d"]), width=args.width, slots=args.slots, fast=not args.no_fast)
    start = time.perf_counter()
    trained = train.train_shared(fit, held, config, seed=args.seed, epochs=args.epochs, patience=args.patience,
                                 batch_size=args.batch)
    note = {"source": str(args.export.name), "fit": len(fit), "held_back": len(held), "held_back_by": "template",
            "latents": {k: header.get(k) for k in ("format", "model", "revision", "layers", "d", "pooling")}}
    card = _card(trained, time.perf_counter() - start, note, args.seed, held, {})
    _write(args.out, trained.core, card)
    return card


def _texts(export: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted((export / "events").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                texts[str(row["event_id"])] = str(row.get("text", ""))
    return texts


def quoting(texts: dict[str, str]) -> Callable[[tuple[str, ...]], str]:
    def topic(cites: tuple[str, ...]) -> str:
        quoted = [texts[c][:QUOTE_CHARS] for c in cites[:2] if texts.get(c)]
        return " / ".join(quoted) if quoted else findings.default_topic(cites)

    return topic


def life_calls(core: Core, stream: LabelledStream, *, reset: bool, dream: bool, seed: int) -> list[findings.Raised]:
    """The core's calls over one life: each night's decision, and when ``dream`` its ignitions at the
    dream's locked threshold (``evaluate.DREAM_STEPS`` steps, as locked)."""
    run = evaluate.trace(core, stream, reset_each_night=reset)
    calls = [findings.Raised(c.day, c.p_speak, c.cites, "speak") for c in run.calls]
    if not dream:
        return calls
    rng = np.random.default_rng(seed)
    state = core.init_state(stream.life_id)
    for day in stream.days:
        if reset:
            state = core.init_state(stream.life_id)
        for event in (e for e in stream.events if e.t.date() == day):
            state, _ = core.observe(state, event)
        calls.extend(findings.Raised(day, i.score, i.cited_event_ids, i.kind)
                     for i in core.dream(state, evaluate.DREAM_STEPS, rng))
    return sorted(calls, key=lambda c: c.day)


def arm_label(core: Core, reset: bool) -> str:
    """The core's own arm labels, apart from APD's pre-registered arm names: ``core-N``,
    ``core-N-reset`` (its state reset each night), ``core-N-nofast`` (trained without fast weights)."""
    return "core-N" + ("-nofast" if not core.config.fast else "") + ("-reset" if reset else "")


def run_findings(args: argparse.Namespace) -> dict[str, Any]:
    core = Core.load(args.model)
    arm = args.arm or arm_label(core, args.reset_each_night)
    streams = data.load_export(args.export, args.latents)
    topic = quoting(_texts(args.export))
    line, by_hand = (core.threshold, False) if args.threshold is None else (args.threshold, True)
    ignite = core.thresholds.ignite if args.dream else None
    start, rows, events = time.perf_counter(), list[dict[str, Any]](), 0
    for stream in streams:
        calls = life_calls(core, stream, reset=args.reset_each_night, dream=args.dream, seed=args.seed)
        events += len(stream.events)
        rows.extend(_row(arm, stream.life_id, report, threshold_by_hand=by_hand)
                    for report in findings.reports(calls, threshold=line, ignite=ignite, topic=topic))
    seconds = time.perf_counter() - start
    _write_rows(args.out, rows)
    return {"arm": arm, "model_id": core.model_id, "threshold": line, "threshold_by_hand": by_hand,
            "ignite_threshold": core.thresholds.ignite, "lives": len(streams), "events": events, "findings": len(rows),
            "reset_each_night": args.reset_each_night, "dream_steps": evaluate.DREAM_STEPS if args.dream else 0,
            "cpu_s": round(seconds, 2), "ms_per_event": round(1000 * seconds / max(events, 1), 3)}


def _row(arm: str, life_id: str, report: FindingReport, **extra: Any) -> dict[str, Any]:
    return {"arm": arm, "life_id": life_id, **dataclasses.asdict(report), "day": report.day.isoformat(),
            "cites": list(report.cites), **extra}


def _write_rows(out: Path, rows: Sequence[dict[str, Any]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def run_render(args: argparse.Namespace) -> dict[str, Any]:
    if not ARM_NAME.fullmatch(args.arm):
        raise ValueError(f"not an arm name: {args.arm!r}")
    topic = quoting(_texts(args.export))
    rows, runs = list[dict[str, Any]](), 0
    for path in sorted(args.runs.glob(f"*.{args.arm}.json")):
        run: ArmRun = loads(path.read_text(encoding="utf-8"), ArmRun)
        if run.arm != args.arm:
            raise ValueError(f"{path.name} holds another arm's run")
        runs += 1
        rows.extend(_row(args.arm, run.life_id, findings.in_core_form(report, topic)) for report in run.reports)
    _write_rows(args.out, rows)
    return {"arm": args.arm, "runs": runs, "findings": len(rows), "form": "core"}


def run_prescore(args: argparse.Namespace) -> dict[str, Any]:
    by_life: dict[str, list[dict[str, Any]]] = {}
    for line in args.findings.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_life.setdefault(str(row["life_id"]), []).append(row)
    total = findings.PreScore(0, 0, 0, 0, 0, 0)
    for path in sorted(args.labels.glob("*.json")):
        held = json.loads(path.read_text(encoding="utf-8"))
        rows = by_life.get(str(held["life_id"]), [])
        reports = [FindingReport(
            finding_id=r["finding_id"], day=date.fromisoformat(r["day"]), seq=r["seq"], problem=r["problem"],
            conclusion=r["conclusion"], p=r["p"], cites=tuple(r["cites"]), kind=r["kind"]) for r in rows]
        one = findings.prescore(reports, [findings.item_of(item) for item in held["items"]])
        total = findings.PreScore(*(a + b for a, b in zip(dataclasses.astuple(total), dataclasses.astuple(one))))
    return {**dataclasses.asdict(total), "recall": total.recall}


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m anjo.sleep.mind.core", description=__doc__.split("\n")[0])
    sub = top.add_subparsers(dest="command", required=True)
    for name in ("toy", "train"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--epochs", type=int, default=200, help="the cap")
        p.add_argument("--patience", type=int, default=20, help="epochs without a better validation loss before stopping")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--width", type=int, default=64)
        p.add_argument("--slots", type=int, default=8 if name == "toy" else 16)
        p.add_argument("--no-fast", action="store_true")
        if name == "train":
            p.add_argument("--export", type=Path, required=True)
            p.add_argument("--latents", type=Path, required=True)
            p.add_argument("--val-share", type=float, default=0.2)
            p.add_argument("--batch", type=int, default=16)
    f = sub.add_parser("findings")
    for flag in ("--model", "--export", "--latents", "--out"):
        f.add_argument(flag, type=Path, required=True)
    f.add_argument("--reset-each-night", action="store_true")
    f.add_argument("--dream", action="store_true", help="add each night's ignitions, at the dream's locked threshold")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--threshold", type=float, default=None, help="report at this speak threshold, not the locked one")
    f.add_argument("--arm", default=None, help="the label each record carries (default: core-N, -nofast, -reset)")
    r = sub.add_parser("render")
    for flag in ("--runs", "--export", "--out"):
        r.add_argument(flag, type=Path, required=True)
    r.add_argument("--arm", default="B")
    s = sub.add_parser("prescore")
    s.add_argument("--findings", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True)
    return top


COMMANDS: Final[dict[str, Callable[[argparse.Namespace], dict[str, Any]]]] = {
    "toy": run_toy, "train": run_train, "findings": run_findings, "prescore": run_prescore, "render": run_render,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    torch.set_num_threads(1)
    try:
        result = COMMANDS[args.command](args)
    except (data.DataError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    summary = result if args.command in ("findings", "prescore", "render") else {
        k: result[k] for k in ("model_id", "params", "best_epoch", "thresholds", "metrics", "timing")}
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0
