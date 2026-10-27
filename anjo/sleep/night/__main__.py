"""``python -m anjo.sleep.night``: one night for the people named, on the sleep host (Task 5).

    ANJO_SLEEP_HOST=1 python -m anjo.sleep.night --wiring anjo.sleep.app.night:build \\
        --night 2026-10-28 --user <account id> --persona founder

With ``--simulate anjo.sleep.app.sim_transport:simulate --day <i>``, each persona account first
plays day ``i`` through the dev API, and the night runs on that day's window; a day cut short by a
turn Glimmer couldn't answer is logged as short, never as an error.

``--persona`` names the synthetic persona the account before it plays (its sealed probes and the
collapse gate's fields). An account named without one is a real person, and runs only with
``--allow-real``, which Claude passes only for Kevin with his OK.

Before anything else it takes the night lock and sweeps what a run killed hard left behind
(``sweep.py``): leftover Colab sessions are stopped and stale holds settled at their full bound. A
second command while one is live stops before it touches anything.

This is the composition root. The engine core never imports the app (the import contract), so the
app's wiring is named here and loaded by name. The trainer is built here: Colab trains (synthetic
people only, unless ``--allow-real``), and each finished adapter is uploaded to the serving volume
through Task 4's ``put_adapter``, which refuses a name that exists.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from anjo.sleep.budget import default_ledger
from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.night.deps import NightOptions, Wiring
from anjo.sleep.night.run import Window, run_night_batch_with
from anjo.sleep.night.sweep import NightBusy, swept_night
from anjo.sleep.serving.config import REVISION_ENV
from anjo.sleep.train.runners.colab import ColabCli, ColabRunner, stop_leftover_sessions
from anjo.sleep.train.trainer import LoraTrainer
from anjo.sleep.sim.day import DayRun
from anjo.sleep.types import NightResult


def _named(path: str) -> Any:
    module, _, name = path.partition(":")
    if not module or not name:
        raise SystemExit(f"{path} is not <module>:<function>")
    return getattr(importlib.import_module(module), name)


def _wiring(path: str) -> Wiring:
    found: Wiring = _named(path)
    return found


def _colab_cli() -> ColabCli:
    return ColabCli()


def _publish(folder: Path, name: str) -> None:
    from anjo.sleep.serving.modal_app import put_adapter  # needs the modal package: imported only to upload

    put_adapter(folder, name)


def execute(
    *, wiring: str, night: date, users: Sequence[str], personas: Mapping[str, str], allow_real: bool = False,
    practice_scenes: int = 16, teacher: bool = True, simulate: str | None = None, day: int | None = None,
    seed: int = 20261028, windows: Mapping[str, Window] | None = None, days: Mapping[str, DayRun] | None = None,
) -> dict[str, NightResult]:
    """The night for ``users`` under the lock and sweep, with Colab training and the adapter upload.
    ``windows`` gives each person's own window (the real-time schedule's previous day); a simulated
    ``day`` gives the window of the day it played. ``days`` are days already lived (or missed: a day
    with a ``failure`` fails the night at ``simulate``). Also used by ``night/daily.py``."""
    data_dir = Path(os.environ.get(DATA_DIR_ENV, "").strip() or _no_data_dir())
    options = NightOptions(night=night, personas=dict(personas), allow_real=allow_real,
                           practice_scenes=practice_scenes, practice_teacher=teacher)
    # Colab trains only the people named with a persona (the wiring refuses a persona on an account
    # that isn't registered synthetic), and anyone else only with --allow-real.
    cli, ledger = _colab_cli(), default_ledger()
    runner = ColabRunner(cli=cli, ledger=ledger, is_synthetic=lambda user_id: user_id in personas,
                         allow_real=allow_real)
    trainer = LoraTrainer(runner, root=data_dir / "sleep" / "training", publish=_publish,
                          revision=os.environ.get(REVISION_ENV) or None)
    try:
        with swept_night(ledger, lambda: stop_leftover_sessions(cli)) as swept:
            deps = _wiring(wiring)(trainer, options)
            play = _named(simulate) if simulate else None
            played = {user: play(user, persona, day, seed) for user, persona in personas.items()} if play else {}
            return run_night_batch_with(deps, {user: night for user in users}, windows=windows,
                                        days={**played, **(days or {})},
                                        sweep=swept)
    except NightBusy as busy:
        raise SystemExit(str(busy)) from None


def _no_data_dir() -> str:
    raise SystemExit(f"{DATA_DIR_ENV} is not set")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one night for the people named.")
    parser.add_argument("--wiring", required=True, help="the app's NightDeps builder, <module>:<function>")
    parser.add_argument("--night", required=True, type=date.fromisoformat)
    parser.add_argument("--user", action="append", required=True, dest="users")
    parser.add_argument("--persona", action="append", default=[], dest="personas",
                        help="the persona each --user plays, in order; omit for a real person")
    parser.add_argument("--allow-real", action="store_true")
    parser.add_argument("--practice-scenes", type=int, default=16, help="full practice's scenes, 16 to 20")
    parser.add_argument("--no-teacher", action="store_true", help="full practice without the teacher candidate")
    parser.add_argument("--simulate", help="the app's day runner, <module>:<function>, run per persona first")
    parser.add_argument("--day", type=int, help="the persona's day index to simulate")
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args(argv)
    if len(args.personas) > len(args.users):
        parser.error("more personas than users")
    personas = dict(zip(args.users, args.personas, strict=False))
    real = [user for user in args.users if user not in personas]
    if real and not args.allow_real:
        parser.error("a user without a persona is a real person: pass --allow-real only with their OK")
    if (args.simulate is None) != (args.day is None):
        parser.error("--simulate and --day go together")
    if not os.environ.get(DATA_DIR_ENV, "").strip():
        parser.error(f"{DATA_DIR_ENV} is not set")
    results = execute(
        wiring=args.wiring, night=args.night, users=args.users, personas=personas, allow_real=args.allow_real,
        practice_scenes=args.practice_scenes, teacher=not args.no_teacher, simulate=args.simulate, day=args.day,
        seed=args.seed,
    )
    print(json.dumps({user: {"status": r.status, "failed_at": r.failed_at, "log": str(r.log_path)}
                      for user, r in results.items()}, indent=1))
    return 0 if all(r.status != "failed" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
