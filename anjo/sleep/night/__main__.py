"""``python -m anjo.sleep.night``: one night for the people named, on the sleep host (Task 5).

    ANJO_SLEEP_HOST=1 python -m anjo.sleep.night --wiring anjo.sleep.app.night:build \\
        --night 2026-10-28 --user <account id> --persona founder

``--persona`` names the synthetic persona the account before it plays (its sealed probes and the
collapse gate's fields). An account named without one is a real person, and runs only with
``--allow-real``, which Claude passes only for Kevin with his OK.

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
from datetime import date
from pathlib import Path

from anjo.sleep.budget import default_ledger
from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.night.deps import NightOptions, Wiring
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.serving.config import REVISION_ENV
from anjo.sleep.train.runners.colab import ColabCli, ColabRunner
from anjo.sleep.train.trainer import LoraTrainer


def _wiring(path: str) -> Wiring:
    module, _, name = path.partition(":")
    if not module or not name:
        raise SystemExit("--wiring is <module>:<function>")
    found: Wiring = getattr(importlib.import_module(module), name)
    return found


def _publish(folder: Path, name: str) -> None:
    from anjo.sleep.serving.modal_app import put_adapter  # needs the modal package: imported only to upload

    put_adapter(folder, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one night for the people named.")
    parser.add_argument("--wiring", required=True, help="the app's NightDeps builder, <module>:<function>")
    parser.add_argument("--night", required=True, type=date.fromisoformat)
    parser.add_argument("--user", action="append", required=True, dest="users")
    parser.add_argument("--persona", action="append", default=[], dest="personas",
                        help="the persona each --user plays, in order; omit for a real person")
    parser.add_argument("--allow-real", action="store_true")
    args = parser.parse_args(argv)
    if len(args.personas) > len(args.users):
        parser.error("more personas than users")
    personas = dict(zip(args.users, args.personas, strict=False))
    real = [user for user in args.users if user not in personas]
    if real and not args.allow_real:
        parser.error("a user without a persona is a real person: pass --allow-real only with their OK")
    data_dir = Path(os.environ.get(DATA_DIR_ENV, "").strip() or parser.error(f"{DATA_DIR_ENV} is not set"))
    options = NightOptions(night=args.night, personas=personas, allow_real=args.allow_real)
    # Colab trains only the people named with a persona (the wiring refuses a persona on an account
    # that isn't registered synthetic), and anyone else only with --allow-real.
    runner = ColabRunner(cli=ColabCli(), ledger=default_ledger(), is_synthetic=lambda user_id: user_id in personas,
                         allow_real=args.allow_real)
    trainer = LoraTrainer(runner, root=data_dir / "sleep" / "training", publish=_publish,
                          revision=os.environ.get(REVISION_ENV) or None)
    deps = _wiring(args.wiring)(trainer, options)
    results = run_night_batch_with(deps, {user: args.night for user in args.users})
    print(json.dumps({user: {"status": r.status, "failed_at": r.failed_at, "log": str(r.log_path)}
                      for user, r in results.items()}, indent=1))
    return 0 if all(r.status != "failed" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
