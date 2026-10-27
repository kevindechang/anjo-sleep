"""The Colab ``GpuRunner`` the night trains on (Task 5; Kevin, 2026-10-27: "Colab trains, Modal
serves").

It drives the ``colab`` CLI (``~/.local/bin/colab``, OAuth). One training job is one VM:

1. refuse a person who isn't registered synthetic, unless ``allow_real=True`` (only Claude sets it,
   for Kevin, with his OK; the standing rule keeps real data off Colab otherwise);
2. reserve the job's worst case on ``colab_cu`` before anything starts: the job's hard timeout at
   the A100-80GB rate (``Bound``). With no room, the night stops (``BudgetStop``);
3. ``new`` an A100 session, ``install`` the pinned training packages, ``upload`` the night's
   dataset (and, for a warm start, the previous adapter's two files), ``exec`` the training script
   (``colab_train.py``) with the recipe in its environment, and ``download`` the adapter and its
   receipt;
4. ``stop`` the session in a ``finally``, on success, failure or cancel, then settle what the VM
   ran for. A stop that fails settles the whole hold and raises, so a VM that may still run is
   never silent.

Every CLI call runs under a hard subprocess timeout, because the CLI can hang on a dropped
connection, and the timeouts of all calls together fit inside the job's own. Nothing else is
uploaded: never ``.env``, never another file. The finished adapter's upload to the Modal volume
is ``train_lora``'s step, through Task 4's ``put_adapter``.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.serving import config
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train import colab_train
from anjo.sleep.train.lora import DATASET_FILE, JOB

COLAB_BIN: Final = Path.home() / ".local" / "bin" / "colab"
AUTH: Final = ("--auth", "oauth2")
GPU: Final = "A100"
# Compute units an hour for an A100-80GB high-RAM VM (Kevin's figure, 2026-10-27; `colab usage`
# on the first session checks it). The hold is the job's hard timeout at this rate.
A100_CU_PER_HOUR: Final = 5.3
POOL: Final = "colab_cu"
REMOTE_DIR: Final = "/content/anjo"
SCRIPT: Final = Path(colab_train.__file__)
# Pinned on the VM, where torch comes with Colab's image. Checked on the first real night.
TRAIN_PACKAGES: Final = (
    "transformers==4.57.1", "peft==0.17.1", "trl==0.24.0", "bitsandbytes==0.48.1",
    "accelerate==1.10.1", "datasets==4.2.0",
)
NEW_TIMEOUT_S: Final = 600.0
INSTALL_TIMEOUT_S: Final = 900.0
UPLOAD_TIMEOUT_S: Final = 300.0
DOWNLOAD_TIMEOUT_S: Final = 300.0
STOP_TIMEOUT_S: Final = 180.0
EXEC_MARGIN_S: Final = 60.0  # the CLI's own wait past the kernel's timeout
MIN_EXEC_S: Final = 1800.0
_OUTPUT_FILES: Final = (*config.ADAPTER_FILES, colab_train.RECEIPT)

CliRun = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]


class ColabError(RuntimeError):
    """A CLI call failed, or the job didn't return what training promises."""


class RealPersonRefused(PermissionError):
    """A person who isn't registered synthetic, without ``allow_real``."""


def run_cli(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    """One CLI call under a hard timeout; ``TimeoutExpired`` kills it and propagates."""
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout_s, check=False)


class ColabCli:
    """``colab --auth oauth2 <command> ...``, every call with a finite timeout."""

    def __init__(self, binary: Path = COLAB_BIN, *, run: CliRun = run_cli) -> None:
        self._binary = binary
        self._run = run

    def call(self, *args: str, timeout_s: float) -> str:
        if not (0 < timeout_s < math.inf):
            raise ValueError("every colab call has a finite timeout")
        done = self._run([str(self._binary), *AUTH, *args], timeout_s)
        if done.returncode != 0:
            raise ColabError(f"colab {args[0]} failed with exit {done.returncode}")
        return done.stdout


def _uploads(inputs: Mapping[str, object]) -> list[tuple[Path, str]]:
    """The dataset, and a warm start's two adapter files: nothing else is ever uploaded."""
    workdir = Path(str(inputs["workdir"]))
    dataset = Path(str(inputs["dataset_path"]))
    if dataset.name != DATASET_FILE or dataset.parent != workdir:
        raise ColabError("only the night's dataset file is uploaded")
    files = [(dataset, f"{REMOTE_DIR}/{DATASET_FILE}")]
    prev = inputs.get("prev_adapter_dir")
    if prev is not None:
        files += [(Path(str(prev)) / name, f"{REMOTE_DIR}/prev/{name}") for name in config.ADAPTER_FILES]
    return files


def _setup_s(uploads: int) -> float:
    return (NEW_TIMEOUT_S + INSTALL_TIMEOUT_S + uploads * UPLOAD_TIMEOUT_S
            + len(_OUTPUT_FILES) * DOWNLOAD_TIMEOUT_S + STOP_TIMEOUT_S + EXEC_MARGIN_S)


def _receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ColabError("the training receipt is not a record")
    return receipt


class ColabRunner:
    """``GpuRunner`` for the ``train_lora`` job on a Colab A100."""

    def __init__(
        self, *, cli: ColabCli, ledger: Ledger, is_synthetic: Callable[[str], bool],
        allow_real: bool = False, clock: Callable[[], float] = time.monotonic,
        packages: Sequence[str] = TRAIN_PACKAGES,
    ) -> None:
        self._cli = cli
        self._ledger = ledger
        self._is_synthetic = is_synthetic
        self._allow_real = allow_real
        self._clock = clock
        self._packages = tuple(packages)

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        if job != JOB:
            raise ColabError(f"the Colab runner runs only {JOB}")
        if not self._allow_real and not self._is_synthetic(str(inputs["person_id"])):
            raise RealPersonRefused("Colab trains synthetic people only, unless allow_real is set")
        name = config.check_adapter_name(str(inputs["adapter_name"]))
        uploads = _uploads(inputs)
        exec_s = timeout_s - _setup_s(len(uploads))
        if exec_s < MIN_EXEC_S:
            raise ColabError("the job's timeout leaves too little time to train")
        reservation = self._ledger.reserve(f"colab_train:{name}", (POOL,), Bound(A100_CU_PER_HOUR, timeout_s))
        if reservation is None:
            raise BudgetStop(f"{POOL} can't hold this training's worst case")
        session = f"anjo-sleep-{name}"
        out = Path(str(inputs["workdir"])) / "out"
        started = self._clock()
        try:
            receipt = self._train(session, uploads, inputs, out, exec_s)
        finally:
            elapsed = self._finish(session, reservation, started)
        return {
            **receipt, "adapter_dir": str(out), "gpu_minutes": elapsed / 60.0,
            "compute_units": elapsed / 3600.0 * A100_CU_PER_HOUR,
        }

    def _train(
        self, session: str, uploads: list[tuple[Path, str]], inputs: Mapping[str, object],
        out: Path, exec_s: float,
    ) -> dict[str, object]:
        cli = self._cli
        cli.call("new", "-s", session, "--gpu", GPU, "--high-mem", timeout_s=NEW_TIMEOUT_S)
        cli.call("install", "-s", session, *self._packages, timeout_s=INSTALL_TIMEOUT_S)
        for local, remote in uploads:
            cli.call("upload", "-s", session, str(local), remote, timeout_s=UPLOAD_TIMEOUT_S)
        warm = inputs.get("prev_adapter_dir") is not None
        env = {
            "ANJO_TRAIN_CONFIG": json.dumps(inputs["config"], sort_keys=True),
            "ANJO_TRAIN_DATASET": f"{REMOTE_DIR}/{DATASET_FILE}",
            "ANJO_TRAIN_OUT": f"{REMOTE_DIR}/out",
            **({"ANJO_TRAIN_PREV": f"{REMOTE_DIR}/prev"} if warm else {}),
        }
        flags = [part for key, value in env.items() for part in ("--env", f"{key}={value}")]
        cli.call("exec", "-s", session, "-f", str(SCRIPT), "--timeout", str(exec_s), *flags,
                 timeout_s=exec_s + EXEC_MARGIN_S)
        out.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            cli.call("download", "-s", session, f"{REMOTE_DIR}/out/{name}", str(out / name),
                     timeout_s=DOWNLOAD_TIMEOUT_S)
        return _receipt(out / colab_train.RECEIPT)

    def _finish(self, session: str, reservation: Reservation, started: float) -> float:
        """Stop the VM, then settle what it ran for. A failed stop settles the whole hold."""
        try:
            self._cli.call("stop", "-s", session, timeout_s=STOP_TIMEOUT_S)
        except (ColabError, subprocess.TimeoutExpired, OSError) as error:
            reservation.settle(reservation.bound.amount)
            raise ColabError(f"stopping {session} failed; stop it by hand before anything else") from error
        elapsed = max(0.0, self._clock() - started)
        reservation.settle(elapsed / 3600.0 * A100_CU_PER_HOUR)
        return elapsed
