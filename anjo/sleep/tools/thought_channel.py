"""The thought channel's shared pretraining, step by step, plan first (``mind.core.projector``).

    python -m anjo.sleep.tools.thought_channel moments  --export TRAIN --out DIR [--seed 0] [--drop-share 0.3]
    python -m anjo.sleep.tools.thought_channel readings --core CORE.npz --export DIR --latents DIR --keys FILE --out FILE
                                                        [--pairs DIR --pair-latents DIR] [--k 8]
    python -m anjo.sleep.tools.thought_channel targets  --export TRAIN --plans DIR/plans.jsonl --out DIR/targets.jsonl [--yes]
    python -m anjo.sleep.tools.thought_channel rows     --plans FILE --readings FILE --targets FILE --out DIR/rows.jsonl
    python -m anjo.sleep.tools.thought_channel train    --rows FILE --out WORKDIR --revision COMMIT [--hours 1] [--yes]
    python -m anjo.sleep.tools.thought_channel cases    --export TEST --out FILE [--pairs DIR]

- ``moments`` (free): core-train's raise, quiet, decoy and settled moments and each row's contrast
  state (``reader_data``), to ``DIR/plans.jsonl``. It opens core-train only.
- ``readings`` (free, CPU, needs torch): the trained core run over each life's events, its state
  read just before each key's message, written as snapshots (``night.core_snapshot``); ``--keys`` is a
  plans or a cases file. Each pair history reads its own latents folder, ``--pair-latents/<pair>.<side>/``
  (the core's latents-folder format): the two share their last day's event ids, but not the context
  those events were read in.
- ``targets`` (paid, Spark Open): one call per moment (``reader_targets``), held on the provider's
  pool at the worst case before the first call (every prompt's characters as tokens, and the reply's
  cap), settled at what was used; moments already in ``--out`` are kept and not asked again.
- ``rows`` (free): the training rows: the night's row form, each with the state it was made under.
- ``train`` (paid, Modal A100): the night's own training job (``train`` on ``anjo-sleep-train``)
  with a ``channel`` block, through the night's Modal runner: held on ``modal_usd`` at the job's
  hard timeout before the call, settled at its wall time. The shared reader adapter lands on the
  adapters volume; the projector comes back to ``WORKDIR/channel/projector.npz``. The receipt's
  held-out margins say whether the channel carries anything (``val_state_margin`` against
  ``val_zero_margin``).
- ``cases`` (free): the evaluation's cases from a core-test export's events (never its labels) and the
  counterfactual pairs (``reader_eval``), for ``tools.thought_channel_eval``.

Without ``--yes`` the paid steps print their plan and hold and touch nothing. Every paid step, and
the evaluation's, runs under MIND's lock (``mind.sweep.mind_lock``) and first sweeps what a killed run
of these tools left (``swept_channel``): the pretraining's recorded Modal calls cancelled, then every
open hold of their own kind (``thought_channel:``) settled at its full bound. Neither the night's sweep
(``night.sweep.NIGHT_KINDS``) nor MIND's (``mind.sweep.MIND_KINDS``) names that kind, so a night or an
APD command never settles a live pretraining's hold, and these tools never settle theirs.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.budget import DATA_DIR_ENV, Bound, Ledger, default_ledger
from anjo.sleep.mind import spark_provider
from anjo.sleep.mind.core import reader_data, reader_eval, reader_targets
from anjo.sleep.mind.core.projector import K_DEFAULT, read
from anjo.sleep.mind.sweep import MindBusy, lock_path, mind_lock
from anjo.sleep.night import core_snapshot
from anjo.sleep.night.holds import hold, usd
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train import channel_train, colab_train, lora, modal_job
from anjo.sleep.train.runners import modal as modal_runner

SHARED_PERSON: Final = "shared-reader"  # the pretraining's synthetic person: generated lives only
HOLD_KIND: Final = "thought_channel"  # these tools' holds: no other sweep names this kind
PRICES_ENV: Final = "ANJO_SLEEP_PRICES"  # "spark_in,spark_out,train_gpu_hour,serve_gpu_hour"
CALLS_DIR: Final = "reader_calls"  # under <data dir>/sleep/: the pretraining's own call records
TOKENS_PER_S: Final = 350.0  # an estimate: QLoRA forward and backward on Glimmer, one A100-80GB
CHARS_PER_TOKEN: Final = 3.5  # the estimate; the hold counts every character as a token
TEMPLATE_TOKENS: Final = 64  # the chat template's own tokens around a call's messages, at most
TYPICAL_REPLY_TOKENS: Final = 250  # the estimate of a target call's reply; the hold takes its cap
DEFAULT_D: Final = 64  # the core's state width, when no row carries a reading
WIDTH: Final = 256  # the projector's hidden width


def _prices(environ: Mapping[str, str]) -> tuple[float, float, float, float]:
    parts = environ.get(PRICES_ENV, "").split(",")
    try:
        found = tuple(float(p) for p in parts)
    except ValueError:
        found = ()
    if len(found) != 4 or any(p < 0 for p in found):
        raise ValueError(f"{PRICES_ENV} holds four prices (Spark Open in and out per million, the GPUs' hours)")
    return found


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def records_root(environ: Mapping[str, str], records: Path | None = None) -> Path:
    """Where the pretraining records its Modal calls: never the night's records."""
    return records or Path(environ.get(DATA_DIR_ENV, ".")) / "sleep" / CALLS_DIR


@contextmanager
def swept_channel(ledger: Ledger, root: Path, records: Path, calls: Callable[[], Any]) -> Iterator[int]:
    """MIND's lock for one paid step of these tools, after a killed run's leftovers are cleared: its
    recorded Modal calls cancelled (``calls`` is asked for a client only when one is recorded; a
    cancel that fails raises and every hold stays open), then each open ``thought_channel:`` hold
    settled at its full bound, the honest charge for work that may have run. Yields how many."""
    with mind_lock(lock_path(ledger, root)):
        found = modal_runner.CallRecords(records)
        if found.items():
            modal_runner.cancel_leftover_calls(found, calls())
        stale = [held for held in ledger.open_holds() if held.job.startswith(f"{HOLD_KIND}:")]
        for held in stale:
            held.settle(held.bound.amount)
        yield len(stale)


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# ── Free steps ────────────────────────────────────────────────────────────────


def _moments(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    export = reader_data.TrainingExport.open(args.export)
    plans = reader_data.plan_rows(reader_data.all_moments(export, seed=args.seed), seed=args.seed,
                                  drop_share=args.drop_share)
    args.out.mkdir(parents=True, exist_ok=True)
    reader_data.write_plans(args.out / "plans.jsonl", plans)
    raised = sum(bool(p.moment.open_items) for p in plans)
    out(f"{len(plans)} moments from {len(export.lives)} lives: {raised} raise a live problem (each with a contrast "
        f"state), {len(plans) - raised} earn the plain reply; written to {args.out / 'plans.jsonl'}")
    return 0


def _keys(path: Path) -> set[str]:
    rows = _jsonl(path)
    keys = {str(r["reading_key"]) for r in rows if "reading_key" in r}
    keys |= {f"{r['life_id']}:{r['event_id']}" for r in rows if "row_id" in r}
    keys |= {str(r["negative"]) for r in rows if r.get("negative") not in (None, reader_data.EMPTY_KEY)}
    return keys


def _streams(data: Any, args: argparse.Namespace) -> list[Any]:
    streams = list(data.load_export(args.export, args.latents))
    if args.pairs is not None:
        for pair in json.loads((args.pairs / "pairs_manifest.json").read_text(encoding="utf-8"))["pairs"]:
            for side in ("failed", "kept"):
                rows = _jsonl(args.pairs / "events" / f"{pair}.{side}.jsonl")
                _, latents = data.read_latents(args.pair_latents / f"{pair}.{side}")
                entry = {"life_id": f"{pair}.{side}", "first_day": rows[0]["day"], "last_day": rows[-1]["day"]}
                streams.append(data.stream_of(entry, rows, [], latents))
    return streams


def _readings(args: argparse.Namespace, out: Callable[[str], None]) -> int:  # pragma: no cover - needs torch
    api = importlib.import_module("anjo.sleep.mind.core.api")
    data = importlib.import_module("anjo.sleep.mind.core.data")
    core = api.Core.load(args.core)
    wanted = _keys(args.keys)
    found: dict[str, tuple[core_snapshot.CoreSnapshot, tuple[tuple[str, ...], ...]]] = {}
    for stream in _streams(data, args):
        mine = {key.split(":", 1)[1] for key in wanted if key.split(":", 1)[0] == stream.life_id}
        state = core.init_state(stream.life_id)
        for event in stream.events:
            if event.event_id in mine:
                seen = core.predict(state)
                snap = core_snapshot.snapshot(state, seen, arm="state", k=args.k)
                found[f"{stream.life_id}:{event.event_id}"] = (snap, read(state, seen, k=args.k).cites)
            state, _ = core.observe(state, event)
    missing = wanted - set(found)
    reader_data.write_readings(args.out, found)
    out(f"{len(found)} readings written to {args.out}; {len(missing)} keys had no event in the streams")
    return 0 if not missing else 1


def _rows(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    plans = reader_data.read_plans(args.plans)
    readings = {k: s for k, (s, _) in reader_data.read_readings(args.readings).items()}
    targets = reader_data.read_targets(args.targets)
    rows, skipped = [], 0
    for plan in plans:
        try:
            rows.append(reader_data.training_row(plan, readings, targets))
        except reader_data.ReaderDataError:
            skipped += 1  # no target (dropped by the writer's checks) or no reading
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    contrast = sum("contrast" in r for r in rows)
    out(f"{len(rows)} rows ({contrast} with a contrast state), {skipped} moments skipped; written to {args.out}")
    return 0


def _cases(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    cases: list[reader_eval.Case] = []
    for entry in manifest["lives"]:  # events only: a core-test export's labels are never read here
        cases += reader_eval.life_cases(str(entry["life_id"]), _jsonl(args.export / "events" / f"{entry['life_id']}.jsonl"))
    pairs = 0
    if args.pairs is not None:
        for pair in json.loads((args.pairs / "pairs_manifest.json").read_text(encoding="utf-8"))["pairs"]:
            record = json.loads((args.pairs / "pairs" / f"{pair}.json").read_text(encoding="utf-8"))
            cases += reader_eval.pair_cases(record, *(_jsonl(args.pairs / "events" / f"{pair}.{side}.jsonl")
                                                      for side in ("failed", "kept")))
            pairs += 1
    args.out.write_text("".join(json.dumps(c.to_json(), ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    out(f"{len(cases)} cases ({len(cases) - 2 * pairs} person-days, {pairs} transplant pairs) written to {args.out}")
    return 0


# ── Paid steps ────────────────────────────────────────────────────────────────


def _targets(args: argparse.Namespace, out: Callable[[str], None], ledger: Ledger | None,
             chat: reader_targets.Chat | None, environ: Mapping[str, str], calls: Any, records: Path | None) -> int:
    export = reader_data.TrainingExport.open(args.export)
    moments = reader_data.moments_by_life(reader_data.read_plans(args.plans))
    done = reader_data.read_targets(args.out)
    todo = reader_targets.asks(export, moments, done=done)
    spark_in, spark_out = _prices(environ)[:2]
    chars = [sum(len(m["content"]) for m in ask.messages) + TEMPLATE_TOKENS for ask in todo]
    worst = (sum(chars) * spark_in + len(todo) * reader_targets.MAX_TOKENS * spark_out) / 1e6
    expected = (sum(chars) / CHARS_PER_TOKEN * spark_in + len(todo) * TYPICAL_REPLY_TOKENS * spark_out) / 1e6
    provider = spark_provider.from_env(environ)
    out(f"targets: {len(todo)} Spark Open calls on {provider.label} ({len(done)} already written), about "
        f"${expected:.2f}; hold ${worst:.2f} on {provider.pool} (every character a token, every reply at its cap)")
    if not args.yes or not todo:
        out("run again with --yes to write them" if todo else "nothing to write")
        return 0
    led = ledger or default_ledger()
    writer = reader_targets.TargetWriter(chat or provider.chat)
    with swept_channel(led, args.out.parent, records_root(environ, records), lambda: calls or modal_runner.SdkCalls()):
        reservation = hold(led, f"{HOLD_KIND}:targets:{stamp()}", (provider.pool,), usd(max(worst, 0.01)))
        try:
            found = writer.write(export, moments, done=done,
                                 keep=lambda key, t: reader_data.append_target(args.out, key, t))
        finally:
            spent = (writer.tokens_in * spark_in + writer.tokens_out * spark_out) / 1e6
            reservation.settle(min(spent, reservation.bound.amount))
    reader_data.write_targets(args.out, found)
    out(f"targets: {writer.calls} calls, {writer.dropped} dropped by the checks, ${spent:.2f} settled on {provider.pool}")
    return 0


def _recipe(rows: list[dict[str, Any]], args: argparse.Namespace, exec_s: float) -> dict[str, object]:
    d = next((int(r["core"]["reading"]["d"]) for r in rows if r.get("core")), DEFAULT_D)
    channel = channel_train.block(k=args.k, d=d, width=WIDTH, arm="state", exec_s=exec_s)
    cfg = lora.LoraCfg(r=args.rank, lr=args.lr, objective="sft", warm_start=False)
    settings = lora.training_config(cfg, len(rows), warm=False, revision=args.revision, night=reader_data.READER_DAY,
                                    channel=channel)
    settings |= {"epochs": args.epochs, "accumulation": args.accumulation}
    colab_train.load_config(json.dumps(settings))  # the script's own checks, here, before anything is held
    return settings


def _pass_tokens(rows: list[dict[str, Any]], k: int) -> int:
    def tokens(row: dict[str, Any]) -> float:
        prompt = sum(len(m["content"]) for m in row["prompt"]) / CHARS_PER_TOKEN + 60 + k
        reply = len(row["completion"][0]["content"]) / CHARS_PER_TOKEN
        if "contrast" in row:
            return 4 * prompt + 2 * reply + 2 * len(row["contrast"]["plain"]) / CHARS_PER_TOKEN
        return prompt + reply

    return int(sum(tokens(r) for r in rows))


def _train(args: argparse.Namespace, out: Callable[[str], None], ledger: Ledger | None, calls: Any,
           records: Path | None, environ: Mapping[str, str]) -> int:
    rows = _jsonl(args.rows)
    timeout_s = args.hours * 3600.0
    exec_s = modal_job.exec_seconds(timeout_s)
    settings = _recipe(rows, args, exec_s)
    name = lora.adapter_name(SHARED_PERSON, 0, args.attempt)
    per_pass = _pass_tokens(rows, args.k)
    channel = settings["channel"]
    assert isinstance(channel, dict)
    bound = Bound(modal_runner.A100_USD_PER_HOUR, timeout_s)
    out(f"pretraining: {len(rows)} rows ({sum('contrast' in r for r in rows)} with a contrast state), about "
        f"{per_pass:,} tokens a pass, {per_pass / TOKENS_PER_S / 60:.0f} min a pass at ~{TOKENS_PER_S:.0f} tokens/s; "
        f"{args.epochs} passes, stopped at {channel['max_minutes']:.0f} min of training")
    out(f"job: the night's training job ({modal_job.APP_NAME}/{modal_job.FUNCTION}) with a channel block "
        f"(K={channel['k']}, projector width {channel['width']}), adapter {name}, LoRA r={settings['r']}")
    out(f"hold ${bound.amount:.2f} on {modal_runner.POOL} (A100-80GB ${modal_runner.A100_USD_PER_HOUR:.2f}/h for "
        f"{args.hours:g} h, the job's hard timeout); settled at its wall time")
    if not args.yes:
        out("run again with --yes to train")
        return 0
    led = ledger or default_ledger()
    root = records_root(environ, records)
    the_calls = calls or modal_runner.SdkCalls()
    runner = modal_runner.ModalRunner(calls=the_calls, ledger=led, records=modal_runner.CallRecords(root),
                                      is_synthetic=lambda person: person == SHARED_PERSON,
                                      hold_kind=f"{HOLD_KIND}:train")
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = args.out / lora.DATASET_FILE
    with swept_channel(led, args.out, root, lambda: the_calls):
        shutil.copyfile(args.rows, dataset)
        try:
            outputs = runner.run(lora.JOB, {
                "person_id": SHARED_PERSON, "adapter_name": name, "night": date.today().isoformat(),
                "dataset_path": str(dataset), "config": settings, "prev_adapter_dir": None, "workdir": str(args.out),
                **({} if args.prev_projector is None else {"prev_projector": str(args.prev_projector)}),
            }, timeout_s=timeout_s)
        finally:
            dataset.unlink(missing_ok=True)
    _report(outputs, name, out)
    return 0


def _report(outputs: Mapping[str, object], name: str, out: Callable[[str], None]) -> None:
    ch = outputs.get("channel") if isinstance(outputs.get("channel"), dict) else {}
    assert isinstance(ch, dict)
    out(f"trained {name}: {outputs.get('steps')} steps, ${float(str(outputs.get('usd', 0.0))):.2f} on "
        f"{modal_runner.POOL}; projector at {outputs.get('projector_path')}")
    state, zero = ch.get("val_state_margin"), ch.get("val_zero_margin")
    if isinstance(state, int | float) and isinstance(zero, int | float):
        out(f"held-out margin: state {state:+.3f} (right {ch.get('val_state_right')}), zero tokens {zero:+.3f} "
            f"(right {ch.get('val_zero_right')})" + ("; the channel carries nothing here" if state <= zero else ""))


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="thought_channel", description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    m = sub.add_parser("moments")
    m.add_argument("--export", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--drop-share", type=float, default=0.3)
    r = sub.add_parser("readings")
    for flag in ("--core", "--export", "--latents", "--keys", "--out"):
        r.add_argument(flag, type=Path, required=True)
    r.add_argument("--pairs", type=Path, default=None)
    r.add_argument("--pair-latents", type=Path, default=None)
    r.add_argument("--k", type=int, default=K_DEFAULT)
    t = sub.add_parser("targets")
    for flag in ("--export", "--plans", "--out"):
        t.add_argument(flag, type=Path, required=True)
    t.add_argument("--yes", action="store_true")
    w = sub.add_parser("rows")
    for flag in ("--plans", "--readings", "--targets", "--out"):
        w.add_argument(flag, type=Path, required=True)
    g = sub.add_parser("train")
    g.add_argument("--rows", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True, help="the run's workdir")
    g.add_argument("--revision", required=True, help="Glimmer's pinned commit, as the training app was deployed")
    g.add_argument("--hours", type=float, default=1.0, help="the job's hard timeout, and its hold")
    g.add_argument("--k", type=int, default=K_DEFAULT, choices=range(2, channel_train.MAX_K + 1), metavar="K")
    g.add_argument("--rank", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--epochs", type=int, default=2)
    g.add_argument("--accumulation", type=int, default=8)
    g.add_argument("--attempt", type=int, default=0, help="a new adapter name for a rerun")
    g.add_argument("--prev-projector", type=Path, default=None)
    g.add_argument("--yes", action="store_true")
    c = sub.add_parser("cases")
    c.add_argument("--export", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--pairs", type=Path, default=None)
    return parser.parse_args(list(argv))


def main(
    argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print, ledger: Ledger | None = None,
    calls: Any = None, chat: reader_targets.Chat | None = None, records: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    args = _parse([] if argv is None else argv)
    env = os.environ if environ is None else environ
    started = time.monotonic()
    try:
        if args.step == "moments":
            return _moments(args, out)
        if args.step == "readings":
            return _readings(args, out)
        if args.step == "rows":
            return _rows(args, out)
        if args.step == "cases":
            return _cases(args, out)
        if args.step == "targets":
            return _targets(args, out, ledger, chat, env, calls, records)
        return _train(args, out, ledger, calls, records, env)
    except (reader_data.ReaderDataError, ValueError, modal_job.ModalJobError) as error:
        out(f"not run: {error}")
        return 2
    except (BudgetStop, MindBusy) as stop:
        out(f"not run: {stop}")
        return 2
    finally:
        if args.step in ("targets", "train") and getattr(args, "yes", False):
            out(f"{args.step}: {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    raise SystemExit(main())
