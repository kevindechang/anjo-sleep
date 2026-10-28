"""``python -m anjo.sleep.mind.apd``: the APD pilot (Task 17, with 17b on the same server).

``--fake`` runs any command, or with no command the whole chain (generate, timing, the
development runs, the intervention test, a fake freeze, the held-out lives and runs with 17b's arms,
kill-resume, judge and report), on offline fakes: no key, no network, no spend. Their numbers mean
nothing. ``--parallel N`` (default 8) runs that many life-arms at once against the one server.

A real run makes paid calls, so it is refused without ``--live``; ``--live`` prints the command's
plan (its Colab session and hard timeout, and every hold on every pool) and stops, and only
``--live --yes`` runs it. Glimmer's pinned revision comes from the serving settings
(``ServeConfig.from_env``), the ledger from the data dir (``budget.default_ledger``).

A run with a failed model call, or a session that can't carry the next run, stops the command
with exit code 3: what finished is kept, and a rerun goes on from there. Every command runs under
MIND's lock, after MIND's sweep (``mind.sweep``): leftover ``anjo-mind-`` sessions stopped, killed
runs' journals settled, stale MIND holds settled at their bound; a second command at once stops
with exit code 4. SIGTERM and SIGHUP end a command as Ctrl-C does, through its cleanup, so its
session is stopped and its holds settled. On a Mac, a live command keeps the machine awake
(``caffeinate -i -w <pid>``) for as long as it runs. A command that serves first checks the Colab
CLI, ``ssh`` and ``ssh-keygen`` are here (``colab_vllm.preflight``), then Glimmer's revision,
vLLM's version, the precision and the reasoning strength against the pilot's first session
(``DataDir.pin_serving``), and 17b's adapters before the VM boots.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import date
from functools import partial
from pathlib import Path
from typing import Final

from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import colab_vllm, fakes, generate, intervene, pilot, plan, timing
from anjo.sleep.mind.apd.batch import PARALLEL, settle_orphans
from anjo.sleep.mind.apd.journal import SpendJournal
from anjo.sleep.mind.apd.judging import GEMINI, MUSE, Rate
from anjo.sleep.mind.apd.live import RunFailed
from anjo.sleep.mind.apd.pilot import Out
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import HELD_OUT_LIVES, KILL_RESUME_LIVES, ArmName, Life, Split
from anjo.sleep.mind.sweep import MindBusy, MindSweep, lock_path, swept_mind
from anjo.sleep.night.deps import Prices
from anjo.sleep.prereg.ledger import (
    LedgerEntry,
    LedgerError,
    ledger_path,
    read_entries,
    read_pushed_head,
    verify,
    verify_entries,
)
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.config import ServeConfig, ServeConfigError
from anjo.sleep.train.runners.colab import MIND_SESSION_PREFIX, stop_leftover_sessions
from anjo.sleep.types import CheckpointRef

PROG: Final = "python -m anjo.sleep.mind.apd"
REFUSED: Final = (f"A real APD run calls Glimmer on Colab, Spark Open, Muse Spark and Gemini, which costs money: "
                  f"it needs --live, which prints its plan, then --live --yes. Offline: {PROG} --fake.")
DEFAULT_HOURS: Final = 6.0
FAKE_DRAW_SEED: Final = "fake-draw-seed"
FAKE_KILL_AT: Final = 12
REFUSED_CODE: Final = 2
STOPPED: Final = 3
BUSY: Final = 4
EXIT_SIGNALS: Final = (signal.SIGTERM, signal.SIGHUP)
SERVING_COMMANDS: Final = ("run", "timing", "intervene", "kill-resume")
FAKE_PRICES: Final = Prices(spark_in_per_m=0.6, spark_out_per_m=2.4, train_gpu_per_hour=1.5, serve_gpu_per_hour=2.61)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="The APD pilot (Task 17).")
    parser.add_argument("--fake", action="store_true", help="run on offline fakes")
    parser.add_argument("--live", action="store_true", help="print the live plan; with --yes, run it")
    parser.add_argument("--yes", action="store_true", help="with --live, run the plan")
    parser.add_argument("--data", type=Path, help="the pilot's data dir (with --fake: a fresh temporary one)")
    parser.add_argument("--prices", help="spark_in,spark_out,train_gpu_hour,serve_gpu_hour (USD list prices)")
    parser.add_argument("--judge-prices", help="muse_in,muse_out,gemini_in,gemini_out (USD per million tokens)")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="the Colab session's hard timeout")
    parser.add_argument("--only", type=int, help="run a step on its first N lives only")
    parser.add_argument("--parallel", type=int, default=PARALLEL, help="life-arms at once against the one server")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("pilot", help="the whole chain (--fake only)")
    made = commands.add_parser("generate", help="the development lives; --held-out only at the freeze")
    made.add_argument("--held-out", action="store_true")
    made.add_argument("--draw-seed", help="the pre-committed draw seed, revealed at the freeze")
    made.add_argument("--lives", type=int, default=HELD_OUT_LIVES)
    made.add_argument("--prereg", type=Path, help="with --fake only: a stand-in prereg ledger (default: the data dir's)")
    ran = commands.add_parser("run", help="every arm on every life of a split")
    ran.add_argument("--held-out", action="store_true")
    ran.add_argument("--adapters", type=Path, help="17b's adapter cache: runs M+replay and M+shuffled")
    ran.add_argument("--replay", help="thinker-1's adapter name in the cache (M+replay)")
    ran.add_argument("--shuffled", help="thinker-shuffled's adapter name in the cache (M+shuffled)")
    commands.add_parser("timing", help="throughput at --parallel, 17b sampled, training planned, the projection")
    commands.add_parser("intervene", help="the intervention test on the development lives' M runs")
    judged = commands.add_parser("judge", help="judge every run of a split, blind")
    judged.add_argument("--held-out", action="store_true")
    commands.add_parser("report", help="the APD report, with 17b's section")
    killed = commands.add_parser("kill-resume", help="kill-resume on held-out lives")
    killed.add_argument("--kill-at", type=int, required=True)
    return parser


def _four(text: str | None, name: str) -> tuple[float, ...] | None:
    if text is None:
        return None
    parts = tuple(float(part) for part in text.split(","))
    if len(parts) != 4 or any(part < 0 for part in parts):
        raise SystemExit(f"{name} holds four prices, each at least 0")
    return parts


def _rates(text: str | None) -> dict[str, Rate] | None:
    parts = _four(text, "--judge-prices")
    return None if parts is None else {MUSE: Rate(parts[0], parts[1]), GEMINI: Rate(parts[2], parts[3])}


def _split(args: argparse.Namespace) -> Split:
    return "held_out" if getattr(args, "held_out", False) else "development"


def _lives(data: DataDir, split: Split, only: int | None) -> tuple[Life, ...]:
    found = data.lives(split)
    return found if only is None else found[:only]


# ── Plans ─────────────────────────────────────────────────────────────────────


def _plan(args: argparse.Namespace, data: DataDir, prices: Prices, rates: dict[str, Rate]) -> plan.Plan:
    session_s, parallel, measured = args.hours * 3600.0, args.parallel, timing.load(data)
    split = _split(args)
    on_disk = len(_lives(data, split, args.only))
    if args.command == "generate":
        count = args.lives if split == "held_out" else len(generate.development_specs())
        return plan.generate_plan(count if args.only is None else min(count, args.only), prices, rates, split)
    if args.command == "run":
        arms = pilot.HELD_OUT_ARMS + (pilot.REPLAY_ARMS if args.adapters else ()) if split == "held_out" else pilot.DEV_ARMS
        return plan.run_plan(on_disk, arms, prices, session_s, parallel, measured)
    if args.command == "timing":
        return plan.timing_plan(prices, session_s, parallel)
    if args.command == "intervene":
        return plan.intervene_plan(prices, session_s, parallel, measured)
    if args.command == "judge":
        return plan.judge_plan(rates)
    if args.command == "kill-resume":
        return plan.kill_plan(prices, session_s, measured)
    return plan.Plan("report: reads what is on disk", None, ())


# ── Steps ─────────────────────────────────────────────────────────────────────


def _settings(fake: bool) -> ServeConfig:
    return fakes.SETTINGS if fake else ServeConfig.from_env()


def _adapters(args: argparse.Namespace) -> tuple[pilot.ColabPublisher, dict[ArmName, CheckpointRef]]:
    """17b's two adapters from the local cache, by the names training gave them: thinker-1's for
    M+replay, thinker-shuffled's for M+shuffled, each with the rendering its folder records."""
    if not (args.replay and args.shuffled):
        raise SystemExit("17b's arms need --replay and --shuffled, the two adapters' names")
    publisher, refs = pilot.ColabPublisher(), dict[ArmName, CheckpointRef]()
    pairs: tuple[tuple[ArmName, str], ...] = (("M+replay", args.replay), ("M+shuffled", args.shuffled))
    for arm, name in pairs:
        folder = args.adapters / name
        publisher(folder, name)
        rendering = colab_vllm.adapter_rendering(folder)
        refs[arm] = CheckpointRef(name, "apd-replay", date.min, str(folder), None,
                                  rendering=rendering if isinstance(rendering, str) else None)
    return publisher, refs


def _recover_journals(data: DataDir, backend: backends.Backend) -> float:
    return sum(settle_orphans(backend, SpendJournal(path)) for path in data.journals())


def _swept(data: DataDir, backend: backends.Backend) -> AbstractContextManager[MindSweep | None]:
    return swept_mind(lock_path(backend.ledger, data.root), backend.ledger,
                      partial(stop_leftover_sessions, backend.cli, prefix=MIND_SESSION_PREFIX),
                      partial(_recover_journals, data, backend))


def _say_swept(swept: MindSweep | None, out: Out) -> None:
    if swept is None or not (swept.sessions_stopped or swept.journals_usd or swept.holds_settled):
        return
    out(f"MIND's sweep: stopped {swept.sessions_stopped} leftover MIND session(s) "
        f"({swept.sessions_not_ours} other session(s) up, not ours, left alone); settled ${swept.journals_usd:.4f} "
        f"from killed runs' journals and {swept.holds_settled} stale MIND hold(s) at their full bound.")


def _guarded(args: argparse.Namespace, data: DataDir, backend: backends.Backend, out: Out) -> int:
    """One command under MIND's lock, after its sweep; a failed call or a session that can't carry
    the next run stops it cleanly."""
    try:
        if args.command in SERVING_COMMANDS:
            backend.preflight()  # the CLI, ssh and ssh-keygen, before the sweep, a hold or a VM
        with _swept(data, backend) as swept:
            _say_swept(swept, out)
            return _step(args, data, backend, out)
    except MindBusy as error:
        out(f"{error}.")
        return BUSY
    except (pilot.FreezeError, LedgerError, StoreError, ServeConfigError, colab_vllm.PreflightError,
            intervene.DrawRefused) as error:
        out(f"Refused: {error}.")
        return REFUSED_CODE
    except (RunFailed, colab_vllm.SessionEnding) as error:
        out(f"Stopped: {error}. What finished is kept, and a rerun goes on from there.")
        return STOPPED


def _exit_on_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def _signals_exit() -> Iterator[None]:
    """SIGTERM and SIGHUP raise ``SystemExit``, so every ``finally`` runs: the session stops and
    the holds settle. Only the main thread installs handlers."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    before = {sig: signal.signal(sig, _exit_on_signal) for sig in EXIT_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


@contextmanager
def _awake(live: bool, start: Callable[[list[str]], object] = subprocess.Popen) -> Iterator[None]:
    """On a Mac, ``caffeinate`` keeps a live command's machine from sleeping under its session:
    tied to this process (``-w``), so it ends with it however it ends, and stopped on the way out."""
    tool = shutil.which("caffeinate") if live and sys.platform == "darwin" else None
    keeper = None if tool is None else start([tool, "-i", "-w", str(os.getpid())])
    try:
        yield
    finally:
        if isinstance(keeper, subprocess.Popen):
            keeper.terminate()


def _pins(settings_: ServeConfig) -> dict[str, str]:
    return {"glimmer_revision": settings_.pinned_revision(), "vllm_version": settings_.vllm_version,
            "precision": settings_.precision.name, "reasoning_strength": config.reasoning_strength()}


def _step(args: argparse.Namespace, data: DataDir, backend: backends.Backend, out: Out) -> int:
    settings_, session_s, split = _settings(backend.fake), args.hours * 3600.0, _split(args)
    run_s = timing.run_s(timing.load(data))
    if args.command == "kill-resume" or (args.command == "run" and split == "held_out"):  # held-out runs
        pilot.check_drawn(_entries(args, data, backend.fake), data.drawn())
    if args.command in SERVING_COMMANDS:
        data.pin_serving(_pins(settings_))
    if args.command == "generate":
        specs = generate.development_specs()
        if split == "held_out":
            specs = pilot.held_out_specs(_entries(args, data, backend.fake), args.draw_seed or "", args.lives)
            data.save_drawn(pilot.seed_digest(args.draw_seed or ""), args.lives)
        pilot.generate_lives(data, specs if args.only is None else specs[:args.only], backend, out)
    elif args.command == "run":
        arms = pilot.DEV_ARMS if split == "development" else pilot.HELD_OUT_ARMS
        replay = _adapters(args) if args.adapters is not None and split == "held_out" else None  # before the VM
        with pilot.serving(backend, settings_, session_s, out) as session:
            adapters = None
            if replay is not None:
                publisher, adapters = replay
                publisher.upload(session)
                arms = (*arms, *pilot.REPLAY_ARMS)
            pilot.run_lives(data, _lives(data, split, args.only), arms, session, backend, out, adapters=adapters,
                            parallel=args.parallel, run_s=run_s)
    elif args.command == "timing":
        started = time.monotonic()
        with pilot.serving(backend, settings_, session_s, out) as session:
            measured = timing.time_pilot(data, session, backend, out, setup_s=time.monotonic() - started,
                                         parallel=args.parallel)
        left = None if backend.ledger is None else backend.ledger.remaining("baseten_mind_usd")
        out(timing.describe(measured, spark_left=left))
    elif args.command == "intervene":
        snaps = intervene.draw(data)  # before the VM: refused until every development M run is on disk
        with pilot.serving(backend, settings_, session_s, out) as session:
            results, skipped = intervene.intervene(data, session, backend, out, parallel=args.parallel,
                                                   snapshots=snaps, run_s=run_s)
        out(f"the intervention test: {len(results)} resumed, {len(skipped)} skipped")
    elif args.command == "judge":
        pilot.judge_lives(data, _lives(data, split, args.only), backend, out)
    elif args.command == "report":
        judged = pilot.judge_lives(data, _lives(data, "held_out", args.only), _no_new_calls(backend), out)
        out(pilot.report(judged, intervene.on_disk(data)))
    elif args.command == "kill-resume":
        with pilot.serving(backend, settings_, session_s, out) as session:
            pilot.kill_lives(data, _lives(data, "held_out", KILL_RESUME_LIVES), session, args.kill_at, backend, out,
                             parallel=min(args.parallel, KILL_RESUME_LIVES), run_s=run_s)
    return 0


def _no_new_calls(backend: backends.Backend) -> backends.Backend:
    """The report reads the judges' cached replies only: a prompt not judged yet is an error."""

    def refuse(prompt: str) -> tuple[str, int, int]:
        raise SystemExit("a run isn't judged yet: run the judge step first")

    return dataclasses.replace(backend, muse=refuse, gemini=refuse, claude=None, ledger=None)


def _entries(args: argparse.Namespace, data: DataDir, fake: bool) -> list[LedgerEntry]:
    """The prereg ledger, its chain verified. Live, only the real one (``ANJO_DATA_DIR``), and only
    as far as the head last pushed to the public ``prereg`` branch: an entry never pushed, or a
    chain re-forged since, is refused. The fakes read the data dir's stand-in (or ``--prereg``)."""
    if fake:
        entries = read_entries(getattr(args, "prereg", None) or _fake_prereg(data))
        verify_entries(entries)
        return entries
    path = ledger_path()
    head = read_pushed_head(path)
    if head is None:
        raise LedgerError("the prereg ledger was never pushed; push it before the held-out lives")
    verify(path, expect_exists=True, remote_head=head)
    return read_entries(path)


# ── The fake chain ────────────────────────────────────────────────────────────


def _fake_prereg(data: DataDir) -> Path:
    return data.root / "prereg" / "ledger.jsonl"


def _fake_freeze(data: DataDir, lives: int) -> Path:
    """A local stand-in for the prereg ledger: the draw seed committed, then the freeze."""
    from anjo.sleep.prereg.ledger import register

    path = _fake_prereg(data)
    register(pilot.DRAW_SEED_KIND, hashlib.sha256(FAKE_DRAW_SEED.encode()).hexdigest(), "fake draw seed", path=path)
    register(pilot.FREEZE_KIND, "0" * 64, f"fake freeze held_out_lives={lives}", path=path)
    return path


FAKE_ADAPTERS: Final = ("p-0123456789ab-n0", "p-ba9876543210-n0")  # thinker-1's, thinker-shuffled's
FAKE_RATES: Final = {MUSE: Rate(1.25, 4.25), GEMINI: Rate(0.75, 3.75)}


def _fake_backend(data: DataDir) -> backends.Backend:
    """The fakes, held and priced on a ledger of the data dir's own at the real caps, so the
    holds run as they would live (the fake prices are list-like; the fakes' tokens are tiny)."""
    from anjo.sleep.budget import Ledger

    return dataclasses.replace(backends.fake_backend(), ledger=Ledger(data.root / "fake-budget.jsonl"),
                               prices=FAKE_PRICES, rates=FAKE_RATES)


def _fake_adapters(root: Path) -> Path:
    folder = root / "adapter_cache"
    rendering = framing.rendering(config.reasoning_strength())
    for name in FAKE_ADAPTERS:
        fakes.adapter_folder(folder / name, rendering)
    return folder


def _chain(parser: argparse.ArgumentParser, base: list[str], data: DataDir, out: Out, *, whole: bool = True) -> int:
    """The whole pilot on the fakes, step by step through the same commands. On some lives only
    (``whole`` off, ``--only``), the intervention test, which draws from all the development
    lives, is left out."""
    prereg = _fake_freeze(data, HELD_OUT_LIVES)
    adapters = _fake_adapters(data.root)
    if not whole:
        out("the intervention test draws from all the development lives' M runs: left out with --only")
    for argv in (
        ["generate"], ["timing"], ["run"], *([["intervene"]] if whole else []),
        ["generate", "--held-out", "--draw-seed", FAKE_DRAW_SEED, "--lives", str(HELD_OUT_LIVES), "--prereg", str(prereg)],
        ["run", "--held-out", "--adapters", str(adapters), "--replay", FAKE_ADAPTERS[0], "--shuffled", FAKE_ADAPTERS[1]],
        ["kill-resume", "--kill-at", str(FAKE_KILL_AT)],
        ["judge", "--held-out"], ["report"],
    ):
        out(f"$ {PROG} {' '.join([*base, *argv])}")
        code = _guarded(parser.parse_args([*base, *argv]), data, _fake_backend(data), out)
        if code:
            return code
    return 0


def main(argv: Sequence[str] | None = None, *, out: Out = print) -> int:
    with _signals_exit():
        return _main(argv, out)


def _main(argv: Sequence[str] | None, out: Out) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.fake:
        data = DataDir(args.data or Path(tempfile.mkdtemp(prefix="apd-fake-")))
        out(f"APD on fakes in {data.root}. The numbers mean nothing.")
        if args.command in (None, "pilot"):
            base = ["--fake", "--data", str(data.root), "--parallel", str(args.parallel),
                    *([] if args.only is None else ["--only", str(args.only)])]
            return _chain(parser, base, data, out, whole=args.only is None)
        return _guarded(args, data, _fake_backend(data), out)
    if args.command in (None, "pilot"):
        parser.error("the whole chain runs only with --fake; live, run each command")
    if getattr(args, "prereg", None) is not None:
        parser.error("--prereg is the fakes' stand-in; live, the prereg ledger is the pushed one in ANJO_DATA_DIR")
    if not args.live:
        out(REFUSED)
        return 2
    if args.data is None:
        parser.error("--live needs --data, the pilot's data dir")
    prices, rates = _four(args.prices, "--prices"), _rates(args.judge_prices)
    if prices is None or rates is None:
        parser.error("--live needs --prices and --judge-prices to price its holds")
    data = DataDir(args.data)
    out(plan.describe(_plan(args, data, Prices(*prices), rates)))
    if not args.yes:
        return 0
    from anjo.sleep.budget import default_ledger
    with _awake(live=True):
        return _guarded(args, data, backends.live_backend(data.root, default_ledger(), Prices(*prices), rates), out)


if __name__ == "__main__":
    sys.exit(main())
