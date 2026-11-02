"""The ``colab_modal`` runner (``ANJO_SLEEP_TRAIN_RUNNER=colab_modal``; Kevin, 2026-10-27: "Colab trains,
Modal serves", with Modal as the fallback). Each night trains on Colab exactly as ``colab`` does, and
on Modal when Colab's infrastructure fails. The fallback is part of the one recorded setting, so the
recipe stays the same whichever runner trained a night, and a final night is never stuck on a VM
Colab can't give.

Which failures fall back is read off the Colab CLI's own calls. ``TracedCli`` notes each call that
failed: what it raised and how long it ran. Only these fall back:

- any failure of a call made before the training script starts: ``new`` (no A100 to assign, a VM
  that never came up), ``install``, ``mkdir`` (the exec that makes the remote folders) or ``upload``,
  whether it exited non-zero, hung past its hard
  timeout or found no CLI;
- a timeout: any later call that hung past its hard timeout, or an ``exec`` that failed at or past
  the kernel's own ``--timeout``;
- a VM lost mid-run: an ``exec`` the watch stopped (``ColabVmLost``) because the session's VM was
  missing from the account's assignments on two checks in a row.

Everything else fails the night as ``colab`` does:

- an ``exec`` that exits non-zero before its timeout. The CLI reports only an exit code, so a Python
  error or a NaN the script reports can't be told from a VM lost mid-run, and a script error would
  fail the same way on Modal;
- a ``download`` that exits non-zero, or a receipt that isn't one;
- a refusal before any VM: a real person, the thought channel, a timeout too short;
- ``BudgetStop``, ``Cancelled``, an interrupt;
- a VM stop that failed. That VM may still run, so stop it by hand before anything else.

A keep-alive ping or a VM check that fails (``KEEPALIVE``, ``VM_CHECK``, from the watch's thread while
the exec runs) is noted like any call but never decides the night: the exec's own outcome does.

By the time the job falls back, the Colab runner's own cleanup has run: the VM is stopped (a life's
VM too), and the ``colab_cu`` hold is settled for what the VM ran, or in full after a ``new`` that
may have made one. Then the same inputs (dataset, recipe, warm start, adapter name, timeout) go to
the Modal runner, which holds its own worst case on ``modal_usd``. Each attempt runs under the job's
timeout, so a night that falls back after a Colab timeout can train for up to twice it.

The outputs name the runner that trained (``runner``) and, after a fallback, why
(``fallback_reason``: the call and its failure, never its output). The night's receipt records both
(``TrainReceipt``). An adapter Modal trained is on the adapters volume already, so ``landed`` tells
the publisher not to upload it (``choice.publisher``).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Final

from anjo.sleep.budget import Ledger
from anjo.sleep.ports import GpuRunner
from anjo.sleep.serving import config
from anjo.sleep.train.runners.colab import (
    KEEPALIVE,
    KEEPALIVE_PERIOD_S,
    MKDIR,
    VM_CHECK,
    VM_CHECK_PERIOD_S,
    ColabCli,
    ColabError,
    ColabLives,
    ColabRunner,
    ColabVmLost,
    command_label,
)
from anjo.sleep.train.runners.modal import ModalRunner

# What one Colab CLI call raises when Colab fails: an exit code, a hang past its hard timeout, no CLI.
CLI_FAILURES: Final = (ColabError, subprocess.TimeoutExpired, OSError)
# The calls before the training script starts: the VM (an A100 to assign), its packages, its remote
# folders, the night's files.
BEFORE_TRAINING: Final = ("new", "install", MKDIR, "upload")
EXEC: Final = "exec"
STOP: Final = "stop"
# Calls whose failure is never the night's: the VM's stop, the watch's pings and checks.
NOT_THE_NIGHT: Final = (STOP, KEEPALIVE, VM_CHECK)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Failed:
    """One CLI call that failed: its command, what it raised, how long it ran and, for ``exec``, the
    kernel's own ``--timeout``."""

    command: str
    error: BaseException
    seconds: float
    kernel_timeout_s: float | None = None


def _kernel_timeout(args: Sequence[str]) -> float | None:
    if args[0] != EXEC or "--timeout" not in args:
        return None
    return float(args[args.index("--timeout") + 1])


class TracedCli(ColabCli):
    """The Colab CLI, noting every call that fails (``failed``, since the last ``begin``). Each call,
    its arguments and its outcome are the wrapped CLI's own."""

    def __init__(self, cli: ColabCli, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._cli = cli
        self._clock = clock
        self.failed: list[Failed] = []

    def begin(self) -> None:
        """Forget the calls of an earlier job."""
        self.failed = []

    def call(self, *args: str, timeout_s: float, vm_lost: threading.Event | None = None) -> str:
        started = self._clock()
        try:
            if vm_lost is None:
                return self._cli.call(*args, timeout_s=timeout_s)
            return self._cli.call(*args, timeout_s=timeout_s, vm_lost=vm_lost)
        except CLI_FAILURES as error:
            self.failed.append(Failed(command_label(args), error, self._clock() - started, _kernel_timeout(args)))
            raise


def fallback_reason(error: BaseException, failed: Sequence[Failed]) -> str | None:
    """Why Colab's ``error`` sends the job to Modal, or None when it fails the night as on ``colab``.

    Only a CLI call's own failure falls back: the first failed call other than the VM's stop and the
    watch's pings and checks must have raised ``error`` itself, so a refusal, a receipt that isn't
    one, a failed stop, a failed ping or a failed check never does.
    """
    first = next((call for call in failed if call.command not in NOT_THE_NIGHT), None)
    if first is None or first.error is not error:
        return None
    if isinstance(error, ColabVmLost):
        return "colab VM lost mid-run"
    if isinstance(error, subprocess.TimeoutExpired):
        return f"colab {first.command} timed out"
    if first.command == EXEC:  # past the kernel's timeout, or the script (or the VM under it) failed
        late = first.kernel_timeout_s is not None and first.seconds >= first.kernel_timeout_s
        return "colab exec ran past its timeout" if late else None
    if first.command in BEFORE_TRAINING:
        return str(error) if isinstance(error, ColabError) else f"colab {first.command} failed ({type(error).__name__})"
    return None


class ColabModalRunner:
    """``GpuRunner`` for the ``train_lora`` job: the Colab runner first (``colab``, built here on the
    traced CLI), and ``modal`` when Colab's infrastructure fails (``fallback_reason``)."""

    def __init__(
        self, *, cli: ColabCli, modal: ModalRunner, ledger: Ledger, is_synthetic: Callable[[str], bool],
        allow_real: bool = False, clock: Callable[[], float] = time.monotonic,
        keepalive_period_s: float = KEEPALIVE_PERIOD_S, vm_check_period_s: float = VM_CHECK_PERIOD_S,
    ) -> None:
        self._trace = TracedCli(cli, clock=clock)
        self.colab = ColabRunner(cli=self._trace, ledger=ledger, is_synthetic=is_synthetic, allow_real=allow_real,
                                 clock=clock, keepalive_period_s=keepalive_period_s,
                                 vm_check_period_s=vm_check_period_s)
        self.modal = modal
        self._landed: set[str] = set()

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        return self.attempt(self.colab, job, inputs, timeout_s)

    def lives(self, *, life_timeout_s: float) -> ColabModalLives:
        """One Colab VM per life (``ColabLives``, Task 12a), with the same fallback each night."""
        return ColabModalLives(self, self.colab.lives(life_timeout_s=life_timeout_s))

    def landed(self, name: str) -> bool:
        """Whether Modal trained ``name``, so the adapter is on the adapters volume already."""
        return name in self._landed

    def attempt(
        self, colab: GpuRunner, job: str, inputs: Mapping[str, object], timeout_s: float,
    ) -> Mapping[str, object]:
        """``job`` on ``colab`` (this runner's Colab runner, or its lives), and on Modal if that fails
        for a reason that falls back."""
        self._trace.begin()
        try:
            return {**colab.run(job, inputs, timeout_s=timeout_s), "runner": "colab"}
        except CLI_FAILURES as error:
            reason = fallback_reason(error, self._trace.failed)
            if reason is None:
                raise
        name = config.check_adapter_name(str(inputs["adapter_name"]))
        logger.warning("Colab couldn't train %s (%s); training it on Modal", name, reason)
        done = self.modal.run(job, inputs, timeout_s=timeout_s)
        self._landed.add(name)
        return {**done, "runner": "modal", "fallback_reason": reason}


class ColabModalLives:
    """``ColabLives`` with the fallback each night. A night Colab can't train stops the life's VM and
    settles its hold (``ColabLives``' own rule), then trains on Modal; the life's next night makes a
    fresh VM. Use it as a context manager, as ``ColabLives``, so every VM is stopped at the end."""

    def __init__(self, runner: ColabModalRunner, lives: ColabLives) -> None:
        self._runner = runner
        self._lives = lives

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        return self._runner.attempt(self._lives, job, inputs, timeout_s)

    def note_day(self, person_id: str, seconds: float) -> None:
        self._lives.note_day(person_id, seconds)

    def note_scoring(self, person_id: str, seconds: float) -> None:
        self._lives.note_scoring(person_id, seconds)

    def records(self, person_id: str) -> list[dict[str, object]]:
        """The life's Colab nights (``ColabLives.records``); a night Modal trained has none."""
        return self._lives.records(person_id)

    def __enter__(self) -> ColabModalLives:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._lives.close()
