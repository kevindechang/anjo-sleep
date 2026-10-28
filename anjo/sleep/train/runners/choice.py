"""Which runner trains Sleep's nights (Task 5): ``ANJO_SLEEP_TRAIN_RUNNER`` is ``modal`` (the default;
Kevin, 2026-10-28: Modal's hackathon credit first) or ``colab`` (the fallback).

The night command, the accelerated runs and the synthetic example all build their runner here, so
they can't drift: ``build`` gives the runner and its sweep, the step the night command runs under
its lock before anything else (Modal: cancel the calls a killed run left; Colab: stop its leftover
sessions). Each sweeps its own runner only, so a night on one never reaches the other's service.

MIND's training (Tasks 17 and 17b, ``train/thinker.py``, the APD runner) never comes here and stays
on Colab whatever the setting says.

A Modal-trained adapter is already on the adapters volume Modal serves from, so ``publisher``
uploads it only where it didn't land (SageMaker's bucket); a Colab-trained one is uploaded as before.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Ledger
from anjo.sleep.ports import GpuRunner
from anjo.sleep.train.lora import AdapterPublisher
from anjo.sleep.train.runners.colab import ColabCli, ColabRunner, stop_leftover_sessions
from anjo.sleep.train.runners.modal import CallRecords, ModalCalls, ModalRunner, cancel_leftover_calls

RUNNER_ENV: Final = "ANJO_SLEEP_TRAIN_RUNNER"
RUNNERS: Final = ("modal", "colab")
DEFAULT_RUNNER: Final = "modal"


class RunnerChoiceError(ValueError):
    """``ANJO_SLEEP_TRAIN_RUNNER`` names no runner."""


@dataclass(frozen=True)
class Chosen:
    """The runner the setting picked, by name, and its sweep (how many stopped, how many not ours)."""

    name: str
    runner: GpuRunner
    sweep: Callable[[], tuple[int, int]]


def runner_name(environ: Mapping[str, str]) -> str:
    """The runner ``environ`` names: ``modal`` when unset or blank."""
    chosen = environ.get(RUNNER_ENV, "").strip() or DEFAULT_RUNNER
    if chosen not in RUNNERS:
        raise RunnerChoiceError(f"{RUNNER_ENV} is {' or '.join(RUNNERS)}")
    return chosen


def build(
    name: str, *, ledger: Ledger, calls_dir: Path, is_synthetic: Callable[[str], bool],
    allow_real: Collection[str] = (), colab_cli: Callable[[], ColabCli], modal_calls: Callable[[], ModalCalls],
) -> Chosen:
    """The runner ``name`` (``runner_name``'s answer). ``allow_real`` names the real people it may train
    (Kevin's own, §12.1); Colab takes it as a yes or no, as it always has. ``calls_dir`` holds Modal's
    call records. Only the chosen runner's client is made."""
    if name == "modal":
        calls, records = modal_calls(), CallRecords(calls_dir)
        modal = ModalRunner(calls=calls, ledger=ledger, records=records, is_synthetic=is_synthetic,
                            allow_real=allow_real)
        return Chosen(name, modal, lambda: cancel_leftover_calls(records, calls))
    if name == "colab":
        cli = colab_cli()
        colab = ColabRunner(cli=cli, ledger=ledger, is_synthetic=is_synthetic, allow_real=bool(allow_real))
        return Chosen(name, colab, lambda: stop_leftover_sessions(cli))
    raise RunnerChoiceError(f"{RUNNER_ENV} is {' or '.join(RUNNERS)}")


def publisher(runner: str, upload: AdapterPublisher, *, serving: str) -> AdapterPublisher:
    """How a finished adapter reaches the server: Modal's training landed it on Modal's volume already."""
    if runner == "modal" and serving == "modal":
        return _landed
    return upload


def _landed(folder: Path, name: str) -> None:
    """Nothing to upload: the training call landed ``name`` on the adapters volume, checked."""
