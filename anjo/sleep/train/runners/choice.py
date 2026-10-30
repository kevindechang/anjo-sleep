"""Which runner trains Sleep's nights (Task 5): ``ANJO_SLEEP_TRAIN_RUNNER`` is ``modal`` (the default;
Kevin, 2026-10-28: Modal's hackathon credit first) or ``colab`` (the fallback).

The night command, the accelerated runs and the synthetic example all build their runner here, so
they can't drift: ``build`` gives the runner and its sweep, the step the night command runs under
its lock before it settles the stale holds. Whichever runner trains, the sweep stops what a killed
run left on either: Modal's recorded calls are cancelled (Modal is reached only if one is recorded),
and Sleep's Colab sessions are stopped wherever the Colab CLI is installed (without it, nothing here
can have started a VM), as a delete's stoppers do. So no hold is settled while its job may still run.

MIND's training (Tasks 17 and 17b, ``train/thinker.py``, the APD runner) never comes here and stays
on Colab whatever the setting says.

A Modal-trained adapter is already on the adapters volume Modal serves from, so ``publisher``
uploads it only where it didn't land (SageMaker's bucket); a Colab-trained one is uploaded as before.
"""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Ledger
from anjo.sleep.night.sweep import Stopped
from anjo.sleep.ports import GpuRunner
from anjo.sleep.train.lora import AdapterPublisher
from anjo.sleep.train.runners import colab as colab_runner
from anjo.sleep.train.runners.colab import ColabCli, ColabError, ColabRunner, stop_leftover_sessions
from anjo.sleep.train.runners.modal import CallRecords, ModalCalls, ModalRunner, cancel_leftover_calls

RUNNER_ENV: Final = "ANJO_SLEEP_TRAIN_RUNNER"
RUNNERS: Final = ("modal", "colab")
DEFAULT_RUNNER: Final = "modal"
# What a hiccup of the Colab CLI's session listing raises: a refused call, a hang, or a missing CLI.
LISTING_FAILURES: Final = (ColabError, subprocess.TimeoutExpired, OSError)
logger = logging.getLogger(__name__)


class RunnerChoiceError(ValueError):
    """``ANJO_SLEEP_TRAIN_RUNNER`` names no runner."""


@dataclass(frozen=True)
class Chosen:
    """The runner the setting picked, by name, and its sweep (how many stopped, how many not ours,
    and why a Colab listing it couldn't read was skipped)."""

    name: str
    runner: GpuRunner
    sweep: Callable[[], Stopped]


def runner_name(environ: Mapping[str, str]) -> str:
    """The runner ``environ`` names: ``modal`` when unset or blank."""
    chosen = environ.get(RUNNER_ENV, "").strip() or DEFAULT_RUNNER
    if chosen not in RUNNERS:
        raise RunnerChoiceError(f"{RUNNER_ENV} is {' or '.join(RUNNERS)}")
    return chosen


def colab_installed() -> bool:
    """Whether this machine has the Colab CLI: without it, nothing here can have started a VM."""
    return Path(colab_runner.COLAB_BIN).is_file()


def _sweep(name: str, records: CallRecords, calls: ModalCalls,
           cli: ColabCli | None) -> Callable[[], Stopped]:
    """Both runners' leftovers: Modal's recorded calls, and Sleep's Colab sessions when ``cli`` is given.
    Returns how many were stopped and how many Colab sessions weren't ours. A Colab listing that
    fails is retried once when Colab trains this night (its own VMs may be up, so the sweep must
    not be skipped); on any other runner it is logged and skipped, and the night goes on."""
    def sweep() -> Stopped:
        stopped = not_ours = 0
        skipped: str | None = None
        if cli is not None:
            try:
                stopped, not_ours = stop_leftover_sessions(cli)
            except LISTING_FAILURES as error:
                if name == "colab":
                    stopped, not_ours = stop_leftover_sessions(cli)  # one retry, then fail as today
                else:
                    skipped = f"colab listing failed ({type(error).__name__})"
                    logger.warning("the Colab session sweep was skipped on the %s runner: %s", name, skipped)
        cancelled, _ = cancel_leftover_calls(records, calls)
        return Stopped(stopped + cancelled, not_ours, skipped)

    return sweep


def build(
    name: str, *, ledger: Ledger, calls_dir: Path, is_synthetic: Callable[[str], bool],
    allow_real: Collection[str] = (), colab_cli: Callable[[], ColabCli], modal_calls: Callable[[], ModalCalls],
) -> Chosen:
    """The runner ``name`` (``runner_name``'s answer) and the sweep of both runners' leftovers.
    ``allow_real`` names the real people it may train (Kevin's own, §12.1); Colab takes it as a yes or
    no, as it always has. ``calls_dir`` holds Modal's call records. Colab's CLI is made only where it is
    installed or chosen, and Modal's client calls Modal only for a recorded call."""
    records, calls = CallRecords(calls_dir), modal_calls()
    if name == "modal":
        modal = ModalRunner(calls=calls, ledger=ledger, records=records, is_synthetic=is_synthetic,
                            allow_real=allow_real)
        return Chosen(name, modal, _sweep(name, records, calls, colab_cli() if colab_installed() else None))
    if name == "colab":
        cli = colab_cli()
        colab = ColabRunner(cli=cli, ledger=ledger, is_synthetic=is_synthetic, allow_real=bool(allow_real))
        return Chosen(name, colab, _sweep(name, records, calls, cli))
    raise RunnerChoiceError(f"{RUNNER_ENV} is {' or '.join(RUNNERS)}")


def publisher(runner: str, upload: AdapterPublisher, *, serving: str) -> AdapterPublisher:
    """How a finished adapter reaches the server: Modal's training landed it on Modal's volume already."""
    if runner == "modal" and serving == "modal":
        return _landed
    return upload


def _landed(folder: Path, name: str) -> None:
    """Nothing to upload: the training call landed ``name`` on the adapters volume, checked."""
