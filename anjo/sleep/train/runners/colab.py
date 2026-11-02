"""The Colab ``GpuRunner`` (Task 5; Kevin, 2026-10-27: "Colab trains, Modal serves"). Since 2026-10-28
the night trains on Modal by default (``modal.py``) and here with ``ANJO_SLEEP_TRAIN_RUNNER=colab``
(``choice.py``); MIND's jobs always train here.

It drives the ``colab`` CLI (``~/.local/bin/colab``, OAuth). One training job is one VM:

1. refuse a person who isn't registered synthetic, unless ``allow_real=True`` (only Claude sets it,
   for Kevin, with his OK; the standing rule keeps real data off Colab otherwise);
2. reserve the job's worst case on ``colab_cu`` before anything starts: the job's hard timeout at
   the A100-80GB rate (``Bound``), keyed by night and person with an attempt number
   (``holds.hold``). With no room, the night stops (``BudgetStop``);
3. ``new`` an A100 session, ``install`` the pinned training packages, make the remote folders
   (``mkdir``: a small ``exec``, once a VM), ``upload`` the night's
   dataset (and, for a warm start, the previous adapter's two files), ``exec`` the training script
   (``colab_train.py``) with the recipe in its environment, and ``download`` the adapter and its
   receipt. A seeded night (``sft_dpo_seed``) also uploads the files Modal's image carries as one
   archive, at Modal's paths, unpacked on the VM before the entry runs, so its loss, fold and
   protect files load beside it;
4. ``stop`` the session in a ``finally``, on success, failure or cancel, then settle what the VM
   ran for. A stop that fails settles the whole hold and raises, so a VM that may still run is
   never silent.

**The keep-alive.** 2026-11-02: an A100 running a training ``exec`` was reclaimed about 35 minutes
in, with the local ``colab exec`` left waiting, and the CLI sends no keep-alive. While the training
``exec`` runs, and only then, a thread pings the session's keep-alive every ``KEEPALIVE_PERIOD_S``,
the first one period in: each ping is one process, ``colab_ping.py`` on the CLI's own python (the
``colab`` entry point's shebang), under its own hard timeout. A ping that fails is logged and
counted (``keepalive_pings``, ``keepalive_failed`` in the job's outputs), never raised: the exec
decides the night. Its calls count as ``keepalive`` (``command_label``), so the fallback's tracing
never takes one for the night's failure. With no python on the shebang the night trains without
pings, and says so.

A run killed hard runs no ``finally``. Every session is named ``anjo-sleep-<adapter>``, so the
night command's sweep (``stop_leftover_sessions``) stops any a killed run left before it starts.
MIND's synthetic jobs (Tasks 17 and 17b) name theirs ``anjo-mind-<...>`` instead (the runner's
``session_prefix``, ``MIND_SESSION_PREFIX``), so a Sleep night that starts while one runs never
stops it, and MIND's own sweep (the APD command's) never stops Sleep's. Their holds are named
apart the same way (``HOLD_KINDS``: ``colab_train:`` and ``colab_life:`` for Sleep's,
``mind_train:`` and ``mind_life:`` for MIND's), so neither sweep settles the other's.

Every CLI call runs under a hard subprocess timeout, because the CLI can hang on a dropped
connection, and the timeouts of all calls together fit inside the job's own. Nothing else is
uploaded: never ``.env``, never another file. The finished adapter's upload to the Modal volume
is ``train_lora``'s step, through Task 4's ``put_adapter``.

An accelerated life keeps one VM for all its nights (``ColabRunner.lives``, ``ColabLives``, Task
12a): the VM is made and set up on the life's first night, so the base weights download once, and
it is stopped when the batch ends or when any of the life's nights fails. Every rule above holds per
night; the life holds its worst case in one reservation, settled at the stop.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import logging
import tarfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from pathlib import Path
from typing import Final, Literal

from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import config
from anjo.sleep.train import colab_train, modal_job
# The pins live with the training script (one source for every runner); torch comes with Colab's image.
from anjo.sleep.train.colab_train import TRAIN_PACKAGES as TRAIN_PACKAGES
from anjo.sleep.train.lora import DATASET_FILE, JOB

COLAB_BIN: Final = Path.home() / ".local" / "bin" / "colab"
AUTH: Final = ("--auth", "oauth2")
GPU: Final = "A100"
# Compute units an hour for an A100-80GB high-RAM VM, as `colab usage` read it on 2026-11-02 (spec
# §11.5's 7.5 was an estimate). The hold is the job's hard timeout at this rate.
A100_CU_PER_HOUR: Final = 8.88
POOL: Final = "colab_cu"
REMOTE_DIR: Final = "/content/anjo"
SCRIPT: Final = Path(colab_train.__file__)
# A seeded night ships Modal's file list as one archive, unpacked on the VM before the entry runs;
# the three hours cover that upload (at the call's hard timeout), the unpack and a 27-minute night,
# leaving a warm seed night (dataset, archive and two adapter files) over 100 minutes to train.
SEED_TIMEOUT_S: Final = 3 * 3600.0
SEED_ARCHIVE: Final = "train-files.tar"
SEED_ARCHIVE_ENTRY: Final = f"{REMOTE_DIR}/{SEED_ARCHIVE}"
SEED_ENTRY: Final = (
    "import runpy, tarfile\n"
    f"with tarfile.open({SEED_ARCHIVE_ENTRY!r}) as archive:\n"
    "    archive.extractall('/')\n"
    f"runpy.run_path({modal_job.REMOTE_SCRIPT!r}, run_name='__main__')\n"
)
# An accelerated life's VM is kept between nights only when the simulated day before was this short;
# otherwise idling costs more than downloading the base weights again. Revisited after the Oct 30 timing.
MAX_IDLE_BETWEEN_NIGHTS_S: Final = 600.0
NEW_TIMEOUT_S: Final = 600.0
INSTALL_TIMEOUT_S: Final = 900.0
UPLOAD_TIMEOUT_S: Final = 300.0
DOWNLOAD_TIMEOUT_S: Final = 300.0
STOP_TIMEOUT_S: Final = 180.0
LIST_TIMEOUT_S: Final = 120.0
SESSION_PREFIX: Final = "anjo-sleep-"
MIND_SESSION_PREFIX: Final = "anjo-mind-"  # MIND's sessions: APD serving and replay's training
SESSION_PREFIXES: Final = (SESSION_PREFIX, MIND_SESSION_PREFIX)  # neither starts with the other
HOLD_KINDS: Final = {SESSION_PREFIX: ("colab_train", "colab_life"), MIND_SESSION_PREFIX: ("mind_train", "mind_life")}
CLI_NOTE: Final = "[colab] "  # the CLI's own messages, such as an update notice
NO_SESSIONS: Final = "No active sessions"
_SESSION_LINE: Final = re.compile(r"^\[(?P<name>[^\]]+)\] \S+ \| Hardware: ")
EXEC_MARGIN_S: Final = 60.0  # the CLI's own wait past the kernel's timeout
# 2026-11-02: an upload into a folder the VM lacks fails (HTTP 500 from its contents API), and a
# new VM has no REMOTE_DIR: the first real Colab night failed at its first upload. Each VM makes
# the night's remote folders once, right after its packages, with this script (``colab exec``
# runs a file; the CLI has no mkdir); its call counts as ``mkdir`` (``command_label``).
MKDIR: Final = "mkdir"
MKDIR_FILE: Final = ".anjo-remote-dirs.py"
MKDIR_TIMEOUT_S: Final = 60.0
REMOTE_DIRS: Final = (REMOTE_DIR, f"{REMOTE_DIR}/prev")
MKDIR_SCRIPT: Final = f"import os\nfor path in {REMOTE_DIRS!r}:\n    os.makedirs(path, exist_ok=True)\n"
MIN_EXEC_S: Final = 1800.0
_OUTPUT_FILES: Final = (*config.ADAPTER_FILES, colab_train.RECEIPT)
# The keep-alive (see above): a ping a minute while the training exec runs, each its own process
# under a hard timeout; the exec's end waits for a ping in flight, so the job's time counts one.
KEEPALIVE: Final = "keepalive"  # a ping's command, as ``call`` and ``command_label`` name it
KEEPALIVE_PERIOD_S: Final = 60.0
KEEPALIVE_TIMEOUT_S: Final = 30.0
KEEPALIVE_THREAD: Final = "colab-keepalive"
PING_SCRIPT: Final = Path(__file__).with_name("colab_ping.py")
_PYTHON: Final = re.compile(r"python[0-9.]*")
logger = logging.getLogger(__name__)

CliRun = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]


class ColabError(RuntimeError):
    """A CLI call failed, or the job didn't return what training promises."""


class RealPersonRefused(PermissionError):
    """A person who isn't registered synthetic, without ``allow_real``."""


class KeepaliveOff(ColabError):
    """No python to run the ping on: the ``colab`` entry point's first line names none."""


def run_cli(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    """One CLI call under a hard timeout; ``TimeoutExpired`` kills it and propagates."""
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout_s, check=False)


def keepalive_command(binary: Path = COLAB_BIN) -> tuple[str, ...] | None:
    """The ping's command before its session's name: the python on the ``colab`` entry point's
    shebang (the CLI's own environment, where ``colab_cli`` imports), isolated (``-I``), running
    ``PING_SCRIPT``. None when the first line names no python file by its absolute path alone: a
    missing entry point, a shell launcher, ``/usr/bin/env python3``, a python with flags."""
    try:
        with binary.open("rb") as entry:
            first = entry.readline(4096).decode("utf-8", "replace").strip()
    except OSError:
        return None
    words = first[2:].split() if first.startswith("#!") else []
    if len(words) != 1:
        return None
    python = Path(words[0])
    if not (python.is_absolute() and _PYTHON.fullmatch(python.name) and python.is_file()):
        return None
    return (str(python), "-I", str(PING_SCRIPT))


class ColabCli:
    """``colab --auth oauth2 <command> ...``, every call with a finite timeout. ``KEEPALIVE`` is the
    one command the CLI lacks: ``call(KEEPALIVE, session)`` runs the ping, ``keepalive`` (the command
    before the session's name) or, by default, ``keepalive_command(binary)``."""

    def __init__(
        self, binary: Path = COLAB_BIN, *, run: CliRun = run_cli, keepalive: Sequence[str] | None = None,
    ) -> None:
        self._binary = binary
        self._run = run
        self._keepalive = None if keepalive is None else tuple(keepalive)

    def call(self, *args: str, timeout_s: float) -> str:
        if not (0 < timeout_s < math.inf):
            raise ValueError("every colab call has a finite timeout")
        argv = self._ping(args) if command_label(args) == KEEPALIVE else [str(self._binary), *AUTH, *args]
        done = self._run(argv, timeout_s)
        if done.returncode != 0:
            raise ColabError(f"colab {command_label(args)} failed with exit {done.returncode}")
        return done.stdout

    def _ping(self, args: Sequence[str]) -> list[str]:
        command = self._keepalive or keepalive_command(self._binary)
        if command is None:
            raise KeepaliveOff(f"{self._binary}'s first line names no python to ping the session with")
        return [*command, *args[1:]]


def hold_base(night: str, person_id: str, prefix: str = SESSION_PREFIX) -> str:
    """The training's hold key before its attempt number: one night, one person; named for the
    sessions' owner (``prefix``), so each owner's sweep settles only its own."""
    return f"{HOLD_KINDS[prefix][0]}:{night}:{person_id}"


def _live_sessions(cli: ColabCli) -> list[str]:
    """The names of every live session. A list the sweep can't read (the CLI's format changed)
    raises, rather than finding nothing."""
    lines = [line.strip() for line in cli.call("sessions", timeout_s=LIST_TIMEOUT_S).splitlines() if line.strip()]
    found = [(line, _SESSION_LINE.match(line)) for line in lines]
    names = [match.group("name") for _, match in found if match]
    unread = [line for line, match in found if match is None and not line.startswith(CLI_NOTE)]
    if unread or not (names or any(NO_SESSIONS in line for line in lines)):
        raise ColabError("colab sessions printed a list the sweep can't read; check for leftover VMs by hand")
    return names


def _stop_each(cli: ColabCli, sessions: Sequence[str]) -> None:
    for name in sessions:
        try:
            cli.call("stop", "-s", name, timeout_s=STOP_TIMEOUT_S)
        except (ColabError, subprocess.TimeoutExpired, OSError) as error:
            raise ColabError("stopping a leftover session failed; stop it by hand before anything else") from error


def stop_leftover_sessions(cli: ColabCli, *, prefix: str = SESSION_PREFIX) -> tuple[int, int]:
    """Stop every live session named with ``prefix``: Sleep's (``anjo-sleep-…``), as the night
    command's sweep, or MIND's (``anjo-mind-…``), as the APD command's.

    Returns how many were stopped and how many other sessions are up: another tool's, or one this
    machine has no name for. Those are only counted. A list or a stop that fails raises, so a VM
    that may run never goes unnoticed: stop it by hand before anything else. So does a list the
    sweep can't read (the CLI's format changed), rather than finding nothing.
    """
    if prefix not in SESSION_PREFIXES:
        raise ValueError(f"a sweep's prefix is one of {', '.join(SESSION_PREFIXES)}")
    names = _live_sessions(cli)
    ours = [name for name in names if name.startswith(prefix)]
    _stop_each(cli, ours)
    return len(ours), len(names) - len(ours)


def stop_person_sessions(cli: ColabCli, person_id: str, adapters: Iterable[str]) -> int:
    """Stop the live sessions training ``person_id`` (a delete, Task 9): one named for each of their
    adapters, and their life's. Nobody else's is touched. Returns how many were stopped."""
    wanted = {f"{SESSION_PREFIX}{name}" for name in adapters} | {_life_session(person_id)}
    mine = [name for name in _live_sessions(cli) if name in wanted]
    _stop_each(cli, mine)
    return len(mine)


def _seeded(inputs: Mapping[str, object]) -> bool:
    """Whether the recipe is ``sft_dpo_seed``: the one objective that loads the files beside the script."""
    recipe = inputs.get("config")
    return isinstance(recipe, Mapping) and recipe.get("objective") == colab_train.SEED_OBJECTIVE


def _job_timeout(inputs: Mapping[str, object], timeout_s: float) -> float:
    """A seeded night's hard timeout: its files' uploads and download leave too little of two hours."""
    return max(timeout_s, SEED_TIMEOUT_S) if _seeded(inputs) else timeout_s


def _seed_archive(
    workdir: Path, files: Sequence[tuple[Path, str]] = modal_job.TRAIN_FILES,
) -> tuple[Path, str]:
    """One archive of the shared training files, at their remote paths, unpacked at the VM's root.

    The member list must be exactly ``modal_job.TRAIN_FILES``: a file outside that list is refused
    before the archive is written, and the written archive is read back and checked again.
    """
    allowed = {remote.lstrip("/") for _, remote in modal_job.TRAIN_FILES}
    members = [(local, remote.lstrip("/")) for local, remote in files]
    if len(members) != len(allowed) or {remote for _, remote in members} != allowed:
        raise ColabError("a seeded night archives exactly the shared training files")
    archive = workdir / SEED_ARCHIVE
    with tarfile.open(archive, "w") as tar:
        for local, remote in members:
            if not local.is_file():
                raise ColabError(f"a shared training file is missing or not a file: {local}")
            tar.add(local, arcname=remote)
    with tarfile.open(archive) as tar:
        written = tar.getnames()
    if sorted(written) != sorted(allowed):
        raise ColabError("the seed archive's members are not exactly the shared training files")
    return archive, SEED_ARCHIVE_ENTRY


def _uploads(inputs: Mapping[str, object]) -> list[tuple[Path, str]]:
    """The dataset, a seeded night's shared training files in one archive, and a warm start's two
    adapter files: nothing else is ever uploaded."""
    workdir = Path(str(inputs["workdir"]))
    dataset = Path(str(inputs["dataset_path"]))
    if dataset.name != DATASET_FILE or dataset.parent != workdir:
        raise ColabError("only the night's dataset file is uploaded")
    files = [(dataset, f"{REMOTE_DIR}/{DATASET_FILE}")]
    if _seeded(inputs):  # before any hold or VM: the same files, at the same paths, as Modal's image
        files.append(_seed_archive(workdir))
    prev = inputs.get("prev_adapter_dir")
    if prev is not None:
        files += [(Path(str(prev)) / name, f"{REMOTE_DIR}/prev/{name}") for name in config.ADAPTER_FILES]
    return files


def _setup_s(uploads: int) -> float:
    return (NEW_TIMEOUT_S + INSTALL_TIMEOUT_S + MKDIR_TIMEOUT_S + EXEC_MARGIN_S + uploads * UPLOAD_TIMEOUT_S
            + len(_OUTPUT_FILES) * DOWNLOAD_TIMEOUT_S + STOP_TIMEOUT_S + EXEC_MARGIN_S + KEEPALIVE_TIMEOUT_S)


def command_label(args: Sequence[str]) -> str:
    """A CLI call's command (``args[0]``: ``KEEPALIVE`` for a keep-alive ping), or ``MKDIR`` for the
    ``exec`` that makes the remote folders."""
    if args and args[0] == "exec" and "-f" in args and Path(args[list(args).index("-f") + 1]).name == MKDIR_FILE:
        return MKDIR
    return args[0]


def _make_remote_dirs(cli: ColabCli, session: str, workdir: Path) -> None:
    script = workdir / MKDIR_FILE
    script.write_text(MKDIR_SCRIPT, encoding="utf-8")
    try:
        cli.call("exec", "-s", session, "-f", str(script), "--timeout", str(MKDIR_TIMEOUT_S),
                 timeout_s=MKDIR_TIMEOUT_S + EXEC_MARGIN_S)
    finally:
        script.unlink(missing_ok=True)


def _exec_script(workdir: Path, seed: bool) -> Path:
    """The file ``colab exec -f`` reads: the script itself, or a seeded night's entry that runs the
    uploaded script at its remote path, so ``__file__`` and its siblings' paths are the VM's."""
    if not seed:
        return SCRIPT
    path = workdir / ".seed-entry.py"
    path.write_text(SEED_ENTRY, encoding="utf-8")
    return path


@dataclass
class _Pings:
    sent: int = 0
    failed: int = 0


def _ping_each(cli: ColabCli, session: str, period_s: float, stop: threading.Event, pings: _Pings) -> None:
    due = time.monotonic() + period_s
    while not stop.wait(max(0.0, due - time.monotonic())):
        due += period_s
        pings.sent += 1
        try:
            cli.call(KEEPALIVE, session, timeout_s=KEEPALIVE_TIMEOUT_S)
        except KeepaliveOff as off:
            pings.failed += 1
            logger.warning("%s: %s trains without keep-alive pings", off, session)
            return
        except (ColabError, subprocess.TimeoutExpired, OSError) as error:
            pings.failed += 1
            logger.warning("a keep-alive ping for %s failed (%s)", session, type(error).__name__)


@contextmanager
def _keepalive(cli: ColabCli, session: str, period_s: float) -> Iterator[_Pings]:
    """Ping ``session``'s keep-alive every ``period_s`` from a thread while the block runs, the first
    one period in. The thread is stopped and joined when the block ends, however it ends."""
    pings, stop = _Pings(), threading.Event()
    thread = threading.Thread(target=_ping_each, args=(cli, session, period_s, stop, pings), name=KEEPALIVE_THREAD,
                              daemon=True)
    thread.start()
    try:
        yield pings
    finally:
        stop.set()
        thread.join()
        logger.info("keep-alive for %s: %d pings, %d failed", session, pings.sent, pings.failed)


def _receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ColabError("the training receipt is not a record")
    return receipt


@dataclass
class _Vm:
    state: Literal["none", "maybe", "up"] = "none"


class ColabRunner:
    """``GpuRunner`` for the ``train_lora`` job on a Colab A100. Its sessions are named with
    ``session_prefix``: Sleep's by default, ``MIND_SESSION_PREFIX`` for MIND's jobs."""

    def __init__(
        self, *, cli: ColabCli, ledger: Ledger, is_synthetic: Callable[[str], bool],
        allow_real: bool = False, clock: Callable[[], float] = time.monotonic,
        packages: Sequence[str] = TRAIN_PACKAGES, session_prefix: str = SESSION_PREFIX,
        keepalive_period_s: float = KEEPALIVE_PERIOD_S,
    ) -> None:
        if session_prefix not in SESSION_PREFIXES:
            raise ValueError(f"a session prefix is one of {', '.join(SESSION_PREFIXES)}")
        if not (0 < keepalive_period_s < math.inf):
            raise ValueError("the keep-alive has a finite period")
        self.session_prefix = session_prefix
        self._keepalive_period_s = keepalive_period_s
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
        recipe = inputs.get("config")
        if isinstance(recipe, Mapping) and recipe.get("channel") is not None:
            raise ColabError("the thought channel trains on Modal only: Colab runs the training script alone")
        timeout_s = _job_timeout(inputs, timeout_s)
        name = config.check_adapter_name(str(inputs["adapter_name"]))
        uploads = _uploads(inputs)
        exec_s = timeout_s - _setup_s(len(uploads))
        if exec_s < MIN_EXEC_S:
            raise ColabError("the job's timeout leaves too little time to train")
        base = hold_base(str(inputs["night"]), str(inputs["person_id"]), self.session_prefix)
        reservation = hold(self._ledger, base, (POOL,), Bound(A100_CU_PER_HOUR, timeout_s))
        session = f"{self.session_prefix}{name}"
        out = Path(str(inputs["workdir"])) / "out"
        started = self._clock()
        vm = _Vm()
        try:
            receipt = self._train(vm, session, uploads, inputs, out, exec_s)
        finally:
            elapsed = self._finish(vm, session, reservation, started)
        return {
            **receipt, "adapter_dir": str(out), "gpu_minutes": elapsed / 60.0,
            "compute_units": elapsed / 3600.0 * A100_CU_PER_HOUR,
        }

    def _train(
        self, vm: _Vm, session: str, uploads: list[tuple[Path, str]], inputs: Mapping[str, object],
        out: Path, exec_s: float,
    ) -> dict[str, object]:
        self._boot(vm, session, Path(str(inputs["workdir"])))
        return self._night(session, uploads, inputs, out, exec_s)

    def _boot(self, vm: _Vm, session: str, workdir: Path) -> None:
        """Make the VM, install the pinned packages and make the remote folders the uploads go to."""
        cli = self._cli
        vm.state = "maybe"  # a `new` that times out may still have made the VM
        try:
            cli.call("new", "-s", session, "--gpu", GPU, "--high-mem", timeout_s=NEW_TIMEOUT_S)
        except (ColabError, OSError):
            vm.state = "none"
            raise
        vm.state = "up"
        cli.call("install", "-s", session, *self._packages, timeout_s=INSTALL_TIMEOUT_S)
        _make_remote_dirs(cli, session, workdir)

    def _night(
        self, session: str, uploads: list[tuple[Path, str]], inputs: Mapping[str, object], out: Path,
        exec_s: float,
    ) -> dict[str, object]:
        """Upload, train and bring the results off a VM that is up."""
        cli = self._cli
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
        seed = _seeded(inputs)
        entry = _exec_script(Path(str(inputs["workdir"])), seed)
        try:
            with _keepalive(cli, session, self._keepalive_period_s) as pings:
                cli.call("exec", "-s", session, "-f", str(entry), "--timeout", str(exec_s), *flags,
                         timeout_s=exec_s + EXEC_MARGIN_S)
        finally:
            if seed:
                entry.unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            cli.call("download", "-s", session, f"{REMOTE_DIR}/out/{name}", str(out / name),
                     timeout_s=DOWNLOAD_TIMEOUT_S)
        return {**_receipt(out / colab_train.RECEIPT), "keepalive_pings": pings.sent, "keepalive_failed": pings.failed}

    def lives(self, *, life_timeout_s: float) -> ColabLives:
        """A runner that keeps one VM per life for all its nights (Task 12a); use it as a context
        manager, so every VM is stopped when the batch ends or fails."""
        return ColabLives(self, life_timeout_s=life_timeout_s)

    def _finish(self, vm: _Vm, session: str, reservation: Reservation, started: float) -> float:
        """Stop the VM, then settle what it ran for.

        - A VM that ``new`` never made: the stop is only tried, and the time spent is settled; the
          error ``new`` raised is the one that propagates.
        - A ``new`` that timed out may have made one: it is stopped, and the whole hold settled.
        - Any other failed stop settles the whole hold and says to stop the VM by hand.
        """
        try:
            self._cli.call("stop", "-s", session, timeout_s=STOP_TIMEOUT_S)
        except (ColabError, subprocess.TimeoutExpired, OSError) as error:
            if vm.state != "none":
                reservation.settle(reservation.bound.amount)
                raise ColabError(f"stopping {session} failed; stop it by hand before anything else") from error
        elapsed = max(0.0, self._clock() - started)
        if vm.state == "maybe":
            reservation.settle(reservation.bound.amount)
        else:
            reservation.settle(elapsed / 3600.0 * A100_CU_PER_HOUR)
        return elapsed


def _life_session(person_id: str, prefix: str = SESSION_PREFIX) -> str:
    """A life's session name: the sweep's prefix and a hash, never the person's id."""
    return f"{prefix}life-{hashlib.sha256(person_id.encode()).hexdigest()[:12]}"


def _night_setup_s(uploads: int) -> float:
    """A later night's fixed call time on a VM that is up: uploads, downloads, the exec margin and the
    last keep-alive ping."""
    return uploads * UPLOAD_TIMEOUT_S + len(_OUTPUT_FILES) * DOWNLOAD_TIMEOUT_S + EXEC_MARGIN_S + KEEPALIVE_TIMEOUT_S


@dataclass
class _Life:
    session: str
    reservation: Reservation
    started: float
    vm: _Vm = field(default_factory=_Vm)
    idle_since: float | None = None  # when the last night ended on this VM, if it was kept


class ColabLives:
    """``GpuRunner`` for the ``train_lora`` job that keeps one Colab VM per life (Task 12a).

    A life's first night makes the VM, installs the packages and trains; its later nights upload only
    that night's dataset and the previous adapter, and train on the same VM, so Glimmer's base
    weights download once. The life reserves ``life_timeout_s`` on ``colab_cu`` before its VM exists;
    a night whose own timeout would run past it is refused before it starts. The VM is stopped, and
    the hold settled, when a night of the life fails and when the context exits, whether the batch
    ended or broke. A stop that fails settles the whole hold and raises, as for a single night.
    """

    def __init__(self, runner: ColabRunner, *, life_timeout_s: float) -> None:
        if not (0 < life_timeout_s < math.inf):
            raise ValueError("a life has a finite timeout")
        self._runner = runner
        self._life_s = life_timeout_s
        self._lives: dict[str, _Life] = {}
        self._day_s: dict[str, float] = {}
        self._scoring_s: dict[str, float] = {}
        self._done: dict[str, int] = {}
        self._records: dict[str, list[dict[str, object]]] = {}

    def note_day(self, person_id: str, seconds: float) -> None:
        """How long the person's latest simulated day took: the guess for the gap before their next night."""
        self._day_s[person_id] = seconds

    def note_scoring(self, person_id: str, seconds: float) -> None:
        """How long the person's latest morning probe run took: it idles the VM too."""
        self._scoring_s[person_id] = seconds

    def records(self, person_id: str) -> list[dict[str, object]]:
        """Per night: whether the VM was kept after it, and the seconds it idled before it (content-free)."""
        return list(self._records.get(person_id, []))

    def __enter__(self) -> ColabLives:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None,
    ) -> None:
        self.close()

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        runner = self._runner
        if job != JOB:
            raise ColabError(f"the Colab runner runs only {JOB}")
        person = str(inputs["person_id"])
        if not runner._allow_real and not runner._is_synthetic(person):
            raise RealPersonRefused("Colab trains synthetic people only, unless allow_real is set")
        timeout_s = _job_timeout(inputs, timeout_s)
        name = config.check_adapter_name(str(inputs["adapter_name"]))
        uploads = _uploads(inputs)
        life = self._lives.get(person)
        first = life is None
        setup_s = _setup_s(len(uploads)) if first else _night_setup_s(len(uploads))
        exec_s = timeout_s - setup_s
        spent = 0.0 if life is None else runner._clock() - life.started
        refusal = None
        if exec_s < MIN_EXEC_S:
            refusal = "the job's timeout leaves too little time to train"
        elif spent + timeout_s > self._life_s:
            refusal = "this night would run past the life's hold"
        if refusal is not None:
            if life is not None:
                self._stop(person)  # a refused night never leaves the VM idling
            raise ColabError(refusal)
        if life is None:
            base = f"{HOLD_KINDS[runner.session_prefix][1]}:{inputs['night']}:{person}"
            reservation = hold(runner._ledger, base, (POOL,), Bound(A100_CU_PER_HOUR, self._life_s))
            life = self._lives[person] = _Life(_life_session(person, runner.session_prefix), reservation,
                                               runner._clock())
        out = Path(str(inputs["workdir"])) / "out"
        night_started = runner._clock()
        idle = None if first or life.idle_since is None else night_started - life.idle_since
        try:
            if first:
                runner._boot(life.vm, life.session, Path(str(inputs["workdir"])))
            receipt = runner._night(life.session, uploads, inputs, out, exec_s)
        except BaseException:
            self._stop(person)  # a failed night stops the life's VM; the error is the one that propagates
            raise
        elapsed = max(0.0, runner._clock() - night_started)
        done = self._done[person] = self._done.get(person, 0) + 1
        day_s = self._day_s.get(person)
        scoring_s = self._scoring_s.get(person)
        # the gap before the next night is a scoring run and a simulated day: both count
        keep = done > 1 and day_s is not None and day_s + (scoring_s or 0.0) <= MAX_IDLE_BETWEEN_NIGHTS_S
        self._records.setdefault(person, []).append(
            {"night": str(inputs["night"]), "vm_kept": keep, "idle_s_before": idle, "day_s": day_s,
             "scoring_s": scoring_s})
        if keep:
            life.idle_since = runner._clock()
        else:
            self._stop(person)  # nothing measured, or a long day ahead: the VM does not idle
        return {
            **receipt, "adapter_dir": str(out), "gpu_minutes": elapsed / 60.0,
            "compute_units": elapsed / 3600.0 * A100_CU_PER_HOUR, "adapter": name,
        }

    def _stop(self, person: str) -> None:
        life = self._lives.pop(person)
        self._runner._finish(life.vm, life.session, life.reservation, life.started)

    def close(self) -> None:
        """Stop every life's VM. Each is tried even if another fails; the first failure is raised."""
        failures: list[Exception] = []
        for person in list(self._lives):
            try:
                self._stop(person)
            except Exception as error:  # cleanup: every life is tried, and the first failure is raised below
                failures.append(error)
        if failures:
            raise failures[0]
