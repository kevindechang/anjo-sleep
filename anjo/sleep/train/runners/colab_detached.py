"""The scripts of a detached Colab night: what the runner's short ``colab exec`` calls run on the VM.

2026-11-02: on two CPU VMs, one long ``colab exec`` died at about 58 minutes ("RuntimeError:
Connection was lost") while both VMs stayed assigned. The CLI's runtime proxy token expires about an
hour after ``new`` and is refreshed only when a CLI process starts, so no single exec outlives it.
The training therefore runs detached on the VM, and the runner only starts it and polls it, each
call a fresh CLI process:

- the launcher writes the training entry and a small runner beside it under the remote folder,
  starts the runner in its own session (``start_new_session``), its output in ``LOG``, writes its
  pid (``PID``) and returns at once. The runner runs the entry and, when it ends, writes its exit
  code (``EXIT``) atomically. The entry's environment is the kernel's, less every ``ANJO_TRAIN_``
  name, plus this night's names (the ones its exec set), so a kept VM's last night never leaks in;
- a poll prints ``running``, ``done <exit code>`` or ``missing``: no pid, or a runner that is gone
  without an exit code;
- a kill ends the runner's whole process group; a log tail prints the log's last lines.

Each prints one line that starts with ``MARK``. The CLI exits 0 even when the kernel's code raises,
so the runner reads that line, never the exit code alone. The scripts use only the standard library
and nothing here imports the runner, so they run, and are tested, anywhere.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

MARK: Final = "anjo-train:"
ENTRY: Final = "train-entry.py"
RUNNER: Final = "train-run.py"
PID: Final = "train.pid"
EXIT: Final = "train.exit"
LOG: Final = "train.log"
TRAIN_ENV_PREFIX: Final = "ANJO_TRAIN_"
LOG_TAIL_LINES: Final = 50
LOG_TAIL_CHARS: Final = 20_000
# The local files ``colab exec -f`` reads; ``command_label`` names each call by its file.
LAUNCH_FILE: Final = ".anjo-launch.py"
POLL_FILE: Final = ".anjo-poll.py"
KILL_FILE: Final = ".anjo-kill.py"
LOG_TAIL_FILE: Final = ".anjo-log-tail.py"

State = Literal["running", "done", "missing"]


@dataclass(frozen=True)
class Reading:
    """What a poll found: the training still running, done with its exit code, or missing."""

    state: State
    code: int | None = None


def _header(remote: str) -> str:
    names = {"REMOTE": remote, "ENTRY": ENTRY, "RUNNER": RUNNER, "PID": PID, "EXIT": EXIT, "LOG": LOG, "MARK": MARK}
    return "".join(f"{name} = {value!r}\n" for name, value in names.items())


_RUNNER_BODY: Final = """\
import os, subprocess, sys
code = subprocess.call([sys.executable, '-u', os.path.join(REMOTE, ENTRY)], cwd=REMOTE)
part = os.path.join(REMOTE, EXIT + '.part')
with open(part, 'w') as f:
    f.write(str(code))
os.replace(part, os.path.join(REMOTE, EXIT))
"""

_LAUNCH_BODY: Final = """\
def _launch():
    import os, subprocess, sys
    path = lambda name: os.path.join(REMOTE, name)
    os.makedirs(REMOTE, exist_ok=True)
    for stale in (PID, EXIT):  # a kept VM's last night
        if os.path.exists(path(stale)):
            os.remove(path(stale))
    with open(path(ENTRY), 'w', encoding='utf-8') as f:
        f.write(ENTRY_SOURCE)
    with open(path(RUNNER), 'w', encoding='utf-8') as f:
        f.write(RUNNER_SOURCE)
    env = {key: value for key, value in os.environ.items() if not key.startswith(PREFIX)}
    env.update({key: os.environ[key] for key in KEYS})
    with open(path(LOG), 'wb') as log:
        child = subprocess.Popen([sys.executable, path(RUNNER)], cwd=REMOTE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(path(PID), 'w') as f:
        f.write(str(child.pid))
    print(MARK, 'launched', child.pid)
_launch()
"""

_POLL_BODY: Final = """\
def _alive(pid):
    import os
    try:
        with open('/proc/%d/cmdline' % pid, 'rb') as f:
            return RUNNER.encode() in f.read()  # a zombie's is empty; a reused pid's is another's
    except OSError:
        if os.path.isdir('/proc'):
            return False
    try:  # no /proc (not Linux): whether the pid is up at all
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
def _reading():
    import os
    try:
        with open(os.path.join(REMOTE, PID)) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return 'missing'
    alive = _alive(pid)  # before the exit code: a runner writes it, then ends
    try:
        with open(os.path.join(REMOTE, EXIT)) as f:
            return 'done %d' % int(f.read().strip())
    except (OSError, ValueError):
        return 'running' if alive else 'missing'
print(MARK, _reading())
"""

_KILL_BODY: Final = """\
def _kill():
    import os, signal
    try:
        with open(os.path.join(REMOTE, PID)) as f:
            os.killpg(int(f.read().strip()), signal.SIGKILL)  # the runner's group: it and the training
    except (OSError, ValueError) as error:
        return 'not killed (%s)' % type(error).__name__
    return 'killed'
print(MARK, _kill())
"""

_LOG_TAIL_BODY: Final = """\
def _tail():
    import collections, os
    try:
        with open(os.path.join(REMOTE, LOG), encoding='utf-8', errors='replace') as f:
            return list(collections.deque(f, LINES))
    except OSError:
        return []
_lines = _tail()
print(MARK, 'log tail', len(_lines))
print(''.join(_lines)[-CHARS:], end='')
"""


def runner_script(remote: str) -> str:
    """The runner the launcher writes and starts: it runs the entry, then writes its exit code."""
    return _header(remote) + _RUNNER_BODY


def launch_script(remote: str, entry_source: str, keys: Sequence[str]) -> str:
    """The launcher: ``entry_source`` (the training script, or a seeded night's entry) run detached
    with this night's environment names ``keys`` (set in the kernel by the exec's ``--env``)."""
    if any(not key.startswith(TRAIN_ENV_PREFIX) for key in keys):
        raise ValueError(f"a night's environment names start with {TRAIN_ENV_PREFIX}")
    return (_header(remote) + f"ENTRY_SOURCE = {entry_source!r}\nKEYS = {tuple(keys)!r}\n"
            f"PREFIX = {TRAIN_ENV_PREFIX!r}\nRUNNER_SOURCE = {runner_script(remote)!r}\n" + _LAUNCH_BODY)


def poll_script(remote: str) -> str:
    return _header(remote) + _POLL_BODY


def kill_script(remote: str) -> str:
    return _header(remote) + _KILL_BODY


def log_tail_script(remote: str) -> str:
    return _header(remote) + f"LINES = {LOG_TAIL_LINES}\nCHARS = {LOG_TAIL_CHARS}\n" + _LOG_TAIL_BODY


def _marked(out: str) -> list[str]:
    """The words after ``MARK`` on the last line that starts with it, or none."""
    lines = [line[len(MARK):].split() for line in out.splitlines() if line.startswith(MARK)]
    return lines[-1] if lines else []


def read_launch(out: str) -> int | None:
    """The runner's pid the launcher printed, or None when it printed none."""
    words = _marked(out)
    return int(words[1]) if len(words) == 2 and words[0] == "launched" and words[1].isdigit() else None


def read_poll(out: str) -> Reading | None:
    """A poll's reading, or None when it printed none it could print."""
    words = _marked(out)
    if words == ["running"]:
        return Reading("running")
    if words == ["missing"]:
        return Reading("missing")
    if len(words) == 2 and words[0] == "done" and words[1].lstrip("-").isdigit():
        return Reading("done", int(words[1]))
    return None


@dataclass(frozen=True)
class Scripts:
    """A night's scripts, as the local files ``colab exec -f`` reads."""

    launch: Path
    poll: Path
    kill: Path
    log_tail: Path


@contextmanager
def written(workdir: Path, remote: str, entry_source: str, keys: Sequence[str]) -> Iterator[Scripts]:
    """The night's scripts in its folder while the block runs, and removed after, however it ends."""
    texts = {LAUNCH_FILE: launch_script(remote, entry_source, keys), POLL_FILE: poll_script(remote),
             KILL_FILE: kill_script(remote), LOG_TAIL_FILE: log_tail_script(remote)}
    paths = {name: workdir / name for name in texts}
    try:
        for name, text in texts.items():
            paths[name].write_text(text, encoding="utf-8")
        yield Scripts(paths[LAUNCH_FILE], paths[POLL_FILE], paths[KILL_FILE], paths[LOG_TAIL_FILE])
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)
