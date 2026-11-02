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

**The watch.** 2026-11-02: an A100 running a training ``exec`` was gone server-side about 35 minutes
in, while the local ``colab exec`` waited on toward its kernel timeout (two hours), and the CLI
sends no keep-alive. While the training ``exec`` runs, and only then, a thread watches the session,
each call one process, ``colab_ping.py`` on the CLI's own python (the ``colab`` entry point's
shebang, or the CLI's uv tool python behind a shell wrapper), under its own hard timeout:

- a keep-alive ping every ``KEEPALIVE_PERIOD_S``, the first one period in;
- a VM check every ``VM_CHECK_PERIOD_S``: whether the account's assignments still hold the
  session's VM. A VM missing on ``VM_LOST_CHECKS`` readings in a row stops the exec at once (the
  exec runs on the CLI's stoppable path, ``run_until``) and it raises ``ColabVmLost``; the night
  then lists the sessions (``colab sessions``, which drops the dead one, so the stop goes through)
  and stops the VM as always. A check that fails gives no reading: it neither counts nor resets.

A ping or check that fails is logged and counted (``keepalive_pings``, ``keepalive_failed``,
``vm_checks``, ``vm_checks_failed`` in the job's outputs), never raised: the exec decides the night.
Their calls count as ``keepalive`` and ``vmcheck`` (``command_label``), so the fallback's tracing
never takes one for the night's failure. With neither python the night trains unwatched, and says
so.

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

An accelerated life may keep one VM for several nights (``ColabRunner.lives``, ``ColabLives``, Task
12a): the VM is made and set up on the life's first night, so the base weights download once, and
it is stopped when the batch ends or when any of the life's nights fails. Every rule above holds per
night; each VM holds its worst case in one reservation, settled at its stop. A VM is kept for the
next night only when that night fits what is left of its hold; otherwise it is stopped, and the next
night makes a new VM with its own hold.
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

from anjo.core.best_effort import best_effort
from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import config
from anjo.sleep.train import colab_train, modal_job
# The pins live with the training script (one source for every runner); torch comes with Colab's image.
from anjo.sleep.train.colab_train import TRAIN_PACKAGES as TRAIN_PACKAGES
from anjo.sleep.train.lora import DATASET_FILE, JOB

COLAB_BIN: Final = Path.home() / ".local" / "bin" / "colab"
# The python of the CLI's own uv tool environment, for the watch when ``COLAB_BIN`` names none (a
# shell wrapper put in front of the CLI).
COLAB_TOOL_PYTHON: Final = Path.home() / ".local" / "share" / "uv" / "tools" / "google-colab-cli" / "bin" / "python"
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
# The watch (see above): a ping a minute and a VM check every two while the training exec runs, each
# its own process under a hard timeout. The exec's end waits for the one in flight, so the job's time
# counts one (``WATCH_CALL_S``). A lost VM is seen within two check periods of its last present
# reading, and the exec stopped within ``STOP_POLL_S`` of that: at most about 4.5 minutes, not 2 hours.
KEEPALIVE: Final = "keepalive"  # a ping's command, as ``call`` and ``command_label`` name it
KEEPALIVE_PERIOD_S: Final = 60.0
KEEPALIVE_TIMEOUT_S: Final = 30.0
VM_CHECK: Final = "vmcheck"  # a VM check's command, likewise
VM_CHECK_PERIOD_S: Final = 120.0
VM_CHECK_TIMEOUT_S: Final = 30.0
VM_LOST_CHECKS: Final = 2  # "missing" readings in a row that stop the exec
PRESENT: Final = "present"  # a check's two readings, as the ping script prints them
MISSING: Final = "missing"
WATCH_CALL_S: Final = max(KEEPALIVE_TIMEOUT_S, VM_CHECK_TIMEOUT_S)
WATCH_THREAD: Final = "colab-watch"
STOP_POLL_S: Final = 1.0  # how often a stoppable call looks at its stop
PING_SCRIPT: Final = Path(__file__).with_name("colab_ping.py")
_PYTHON: Final = re.compile(r"python[0-9.]*")
logger = logging.getLogger(__name__)

CliRun = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]
CliRunUntil = Callable[[Sequence[str], float, threading.Event], "subprocess.CompletedProcess[str]"]


class ColabError(RuntimeError):
    """A CLI call failed, or the job didn't return what training promises."""


class RealPersonRefused(PermissionError):
    """A person who isn't registered synthetic, without ``allow_real``."""


class KeepaliveOff(ColabError):
    """No python to run the ping on: the ``colab`` entry point's first line names none, and the CLI's
    tool environment has none."""


class ColabVmLost(ColabError):
    """The session's VM was missing from the account's assignments on ``VM_LOST_CHECKS`` checks in a
    row, so the training exec was stopped: Colab's infrastructure failed, as a timeout does."""


def run_cli(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    """One CLI call under a hard timeout; ``TimeoutExpired`` kills it and propagates."""
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout_s, check=False)


def run_cli_until(args: Sequence[str], timeout_s: float, stop: threading.Event) -> subprocess.CompletedProcess[str]:
    """``run_cli`` that ``stop`` also ends: the process is killed, and its exit is the kill's. The
    hard timeout still kills it and raises ``TimeoutExpired``, and so does anything else that ends the
    call early (an interrupt), as ``subprocess.run`` does. Its output is read as it comes."""
    deadline = time.monotonic() + timeout_s
    with subprocess.Popen(list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise subprocess.TimeoutExpired(list(args), timeout_s)
                try:
                    out, err = process.communicate(timeout=min(STOP_POLL_S, left))
                except subprocess.TimeoutExpired:
                    if not stop.is_set():
                        continue
                    process.kill()
                    out, err = process.communicate()
                return subprocess.CompletedProcess(list(args), process.returncode, out, err)
        finally:
            if process.poll() is None:
                process.kill()


def _unstoppable(run: CliRun) -> CliRunUntil:
    """A test's ``run`` with no stoppable path: the call runs on it, to its own end."""
    def until(args: Sequence[str], timeout_s: float, stop: threading.Event) -> subprocess.CompletedProcess[str]:
        return run(args, timeout_s)
    return until


def keepalive_command(binary: Path = COLAB_BIN, tool_python: Path = COLAB_TOOL_PYTHON) -> tuple[str, ...] | None:
    """The ping's command before its arguments: a python of the CLI's own environment (where
    ``colab_cli`` imports), isolated (``-I``), running ``PING_SCRIPT``. That python is the one the
    ``colab`` entry point's shebang names by its absolute path alone, or else ``tool_python`` (a
    shell launcher in front of the CLI, ``/usr/bin/env python3``, a python with flags, no entry
    point). None when neither is a file."""
    python = _shebang_python(binary) or (tool_python if tool_python.is_file() else None)
    return None if python is None else (str(python), "-I", str(PING_SCRIPT))


def _shebang_python(binary: Path) -> Path | None:
    try:
        with binary.open("rb") as entry:
            first = entry.readline(4096).decode("utf-8", "replace").strip()
    except OSError:
        return None
    words = first[2:].split() if first.startswith("#!") else []
    if len(words) != 1:
        return None
    python = Path(words[0])
    return python if python.is_absolute() and _PYTHON.fullmatch(python.name) and python.is_file() else None


class ColabCli:
    """``colab --auth oauth2 <command> ...``, every call with a finite timeout. ``KEEPALIVE`` and
    ``VM_CHECK`` are the commands the CLI lacks: ``call(KEEPALIVE, session)`` and ``call(VM_CHECK,
    session)`` run the ping script, on ``keepalive`` (the command before its arguments) or, by
    default, ``keepalive_command(binary, tool_python)``. A call given ``vm_lost`` runs on the stoppable path
    (``run_until``; by default ``run_cli_until``, or a test's ``run`` to its end) and is stopped when
    ``vm_lost`` is set: it then raises ``ColabVmLost``."""

    def __init__(
        self, binary: Path = COLAB_BIN, *, run: CliRun = run_cli, run_until: CliRunUntil | None = None,
        keepalive: Sequence[str] | None = None, tool_python: Path = COLAB_TOOL_PYTHON,
    ) -> None:
        self._binary = binary
        self._tool_python = tool_python
        self._run = run
        self._run_until = run_until or (run_cli_until if run is run_cli else _unstoppable(run))
        self._keepalive = None if keepalive is None else tuple(keepalive)

    def call(self, *args: str, timeout_s: float, vm_lost: threading.Event | None = None) -> str:
        if not (0 < timeout_s < math.inf):
            raise ValueError("every colab call has a finite timeout")
        label = command_label(args)
        argv = self._ping(args) if label in (KEEPALIVE, VM_CHECK) else [str(self._binary), *AUTH, *args]
        done = self._run(argv, timeout_s) if vm_lost is None else self._run_until(argv, timeout_s, vm_lost)
        if done.returncode != 0:
            if vm_lost is not None and vm_lost.is_set():
                raise ColabVmLost(f"colab {label} stopped: the session's VM is gone")
            raise ColabError(f"colab {label} failed with exit {done.returncode}")
        return done.stdout

    def _ping(self, args: Sequence[str]) -> list[str]:
        command = self._keepalive or keepalive_command(self._binary, self._tool_python)
        if command is None:
            raise KeepaliveOff(f"no python to watch the session with: {self._binary}'s first line names none, "
                               f"and {self._tool_python} is missing")
        return [*command, *(("--check",) if args[0] == VM_CHECK else ()), *args[1:]]


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
            + len(_OUTPUT_FILES) * DOWNLOAD_TIMEOUT_S + STOP_TIMEOUT_S + EXEC_MARGIN_S + WATCH_CALL_S)


def command_label(args: Sequence[str]) -> str:
    """A CLI call's command (``args[0]``: ``KEEPALIVE`` for a keep-alive ping, ``VM_CHECK`` for a VM
    check), or ``MKDIR`` for the ``exec`` that makes the remote folders."""
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
class _Watch:
    """What the watch did, as counts; ``lost`` is set when the VM was missing ``VM_LOST_CHECKS`` times
    in a row, and stops the exec."""

    lost: threading.Event = field(default_factory=threading.Event)
    off: bool = False  # no python for the ping script: nothing more is tried
    pings: int = 0
    pings_failed: int = 0
    checks: int = 0
    checks_failed: int = 0
    missing: int = 0  # "missing" readings in a row

    def outputs(self) -> dict[str, int]:
        return {"keepalive_pings": self.pings, "keepalive_failed": self.pings_failed, "vm_checks": self.checks,
                "vm_checks_failed": self.checks_failed}


_WATCH_STEPS: Final = {KEEPALIVE: "a keep-alive ping", VM_CHECK: "a VM check"}


def _watched_call(cli: ColabCli, watch: _Watch, *args: str, timeout_s: float) -> str | None:
    """One ping or check: its output, or None when it failed (logged by ``best_effort``)."""
    out: str | None = None
    with best_effort(_WATCH_STEPS[args[0]], session=args[1]):
        try:
            out = cli.call(*args, timeout_s=timeout_s)
        except KeepaliveOff as off:
            watch.off = True
            logger.warning("%s: %s trains without keep-alive pings or VM checks", off, args[1])
    return out


def _read_check(watch: _Watch, out: str | None) -> None:
    reading = None if out is None else out.strip()
    if reading == PRESENT:
        watch.missing = 0
    elif reading == MISSING:
        watch.missing += 1
        if watch.missing >= VM_LOST_CHECKS:
            watch.lost.set()
    else:  # no reading: it neither counts nor resets
        watch.checks_failed += 1
        if out is not None:
            logger.warning("a VM check printed something it never prints; no reading")


def _watch_each(cli: ColabCli, session: str, periods: tuple[float, float], stop: threading.Event,
                watch: _Watch) -> None:
    ping_s, check_s = periods
    started = time.monotonic()
    due = [started + ping_s, started + check_s]
    while not stop.wait(max(0.0, min(due) - time.monotonic())):
        if due[0] <= time.monotonic():
            due[0] += ping_s
            watch.pings += 1
            if _watched_call(cli, watch, KEEPALIVE, session, timeout_s=KEEPALIVE_TIMEOUT_S) is None:
                watch.pings_failed += 1
        if watch.off:
            return
        if due[1] <= time.monotonic() and not stop.is_set():
            due[1] += check_s
            watch.checks += 1
            _read_check(watch, _watched_call(cli, watch, VM_CHECK, session, timeout_s=VM_CHECK_TIMEOUT_S))
        if watch.off:
            return
        if watch.lost.is_set():
            logger.warning("%s's VM was missing on %d checks in a row; stopping its exec", session, VM_LOST_CHECKS)
            return


@contextmanager
def _watching(cli: ColabCli, session: str, ping_s: float, check_s: float) -> Iterator[_Watch]:
    """Watch ``session`` from a thread while the block runs: a ping every ``ping_s`` and a VM check every
    ``check_s``, each first one period in. The thread is stopped and joined when the block ends,
    however it ends."""
    watch, stop = _Watch(), threading.Event()
    thread = threading.Thread(target=_watch_each, args=(cli, session, (ping_s, check_s), stop, watch),
                              name=WATCH_THREAD, daemon=True)
    thread.start()
    try:
        yield watch
    finally:
        stop.set()
        thread.join()
        logger.info("watch of %s: %s", session, watch.outputs())


def _forget_lost(cli: ColabCli) -> None:
    """``colab sessions``: the CLI's own listing drops a session the server no longer has, so the stop
    that follows finds none to stop, rather than failing on a VM that is gone."""
    with best_effort("colab sessions after a lost VM"):
        cli.call("sessions", timeout_s=LIST_TIMEOUT_S)


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
        keepalive_period_s: float = KEEPALIVE_PERIOD_S, vm_check_period_s: float = VM_CHECK_PERIOD_S,
    ) -> None:
        if session_prefix not in SESSION_PREFIXES:
            raise ValueError(f"a session prefix is one of {', '.join(SESSION_PREFIXES)}")
        if not (0 < keepalive_period_s < math.inf and 0 < vm_check_period_s < math.inf):
            raise ValueError("the watch has finite periods")
        self.session_prefix = session_prefix
        self._watch_periods = (keepalive_period_s, vm_check_period_s)
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
            with _watching(cli, session, *self._watch_periods) as watch:
                cli.call("exec", "-s", session, "-f", str(entry), "--timeout", str(exec_s), *flags,
                         timeout_s=exec_s + EXEC_MARGIN_S, vm_lost=watch.lost)
        except ColabVmLost:
            _forget_lost(cli)
            raise
        finally:
            if seed:
                entry.unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        for name in _OUTPUT_FILES:
            cli.call("download", "-s", session, f"{REMOTE_DIR}/out/{name}", str(out / name),
                     timeout_s=DOWNLOAD_TIMEOUT_S)
        return {**_receipt(out / colab_train.RECEIPT), **watch.outputs()}

    def lives(self, *, life_timeout_s: float) -> ColabLives:
        """A runner that may keep one VM per life across its nights (Task 12a), each VM holding at least
        ``life_timeout_s``; use it as a context manager, so every VM is stopped when the batch ends or fails."""
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
    watch's last call. The VM's stop comes after it (``_Life.fits``)."""
    return uploads * UPLOAD_TIMEOUT_S + len(_OUTPUT_FILES) * DOWNLOAD_TIMEOUT_S + EXEC_MARGIN_S + WATCH_CALL_S


@dataclass
class _Life:
    session: str
    reservation: Reservation
    started: float
    vm: _Vm = field(default_factory=_Vm)
    idle_since: float | None = None  # when the last night ended on this VM, if it was kept

    @property
    def hold_s(self) -> float:
        """The seconds this VM's hold covers, from ``started``."""
        return self.reservation.bound.timeout_s

    def fits(self, at: float, timeout_s: float) -> bool:
        """Whether a night of ``timeout_s`` starting at ``at``, and the VM's stop after it, fit the hold."""
        return at - self.started + timeout_s + STOP_TIMEOUT_S <= self.hold_s


class ColabLives:
    """``GpuRunner`` for the ``train_lora`` job that may keep one Colab VM per life (Task 12a).

    A life's first night makes the VM, installs the packages and trains; a later night on a kept VM
    uploads only that night's dataset and the previous adapter, and trains on the same VM, so
    Glimmer's base weights download once. Each VM reserves its hold on ``colab_cu`` before it exists:
    ``life_timeout_s``, or its first night's own timeout when that is longer (a seeded night's).
    After a night the VM is kept only when the simulated day and scoring before the next night are
    short (``MAX_IDLE_BETWEEN_NIGHTS_S``) and another full night, and the stop after it, still fit its hold;
    otherwise it is stopped and its hold settled. A night that won't fit a kept VM's hold after all
    (its day ran longer than the last) stops that VM and makes a new one with its own hold, so no
    night is refused for its hold. The VM is stopped, and the hold settled, when a night of the life
    fails and when the context exits, whether the batch ended or broke. A stop that fails settles the
    whole hold and raises, as for a single night.

    2026-11-02: a life's VM once held all its nights and three more (75 units for a 2-night life), so
    a final's later holds didn't fit the cap, and a refused hold fails its night.
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
        if life is not None and not life.fits(runner._clock(), timeout_s):
            self._stop(person)  # past the kept VM's hold: this night makes a new VM with its own hold
            life = None
        first = life is None
        setup_s = _setup_s(len(uploads)) if first else _night_setup_s(len(uploads))
        exec_s = timeout_s - setup_s
        if exec_s < MIN_EXEC_S:
            if life is not None:
                self._stop(person)  # a refused night never leaves the VM idling
            raise ColabError("the job's timeout leaves too little time to train")
        if life is None:
            base = f"{HOLD_KINDS[runner.session_prefix][1]}:{inputs['night']}:{person}"
            bound = Bound(A100_CU_PER_HOUR, max(self._life_s, timeout_s))
            reservation = hold(runner._ledger, base, (POOL,), bound)
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
        ended = runner._clock()
        elapsed = max(0.0, ended - night_started)
        done = self._done[person] = self._done.get(person, 0) + 1
        day_s = self._day_s.get(person)
        scoring_s = self._scoring_s.get(person)
        # the gap before the next night is a scoring run and a simulated day: both count
        gap_s = None if day_s is None else day_s + (scoring_s or 0.0)
        keep = (done > 1 and gap_s is not None and gap_s <= MAX_IDLE_BETWEEN_NIGHTS_S
                and life.fits(ended + gap_s, timeout_s))  # and another night fits the hold
        self._records.setdefault(person, []).append(
            {"night": str(inputs["night"]), "vm_kept": keep, "idle_s_before": idle, "day_s": day_s,
             "scoring_s": scoring_s})
        if keep:
            life.idle_since = ended
        else:
            self._stop(person)  # nothing measured, a long day ahead or no room left: the VM does not idle
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
