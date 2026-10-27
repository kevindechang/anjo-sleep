"""Delete everything Anjo Sleep keeps of a person (Task 9, RF2), through ports.

``delete_all`` runs the spec's order:

1. set the cancel flag, so a running night stops at its next step and writes nothing after;
2. stop the person's running GPU job, on whichever runner it is (``JobStopper``);
3. remove the checkpoints and adapters, remote copies first (``RemoteAdapters``: the SageMaker
   bucket's ``adapters/`` and the Modal volume), then the local files (``Eraser``): evidence,
   targets and their log, receipts, expectations and snapshots, night logs, sealed probes,
   registry entries and the forced-id entry;
4. clear MIND's workspace log and anything built from it (the last eraser).

The adapter names are read first, before any eraser removes the registry that lists them. Every
step is tried even when an earlier one failed, and a delete that left something behind raises
``DeleteIncomplete`` with the report: it never reports success on a partial delete. The steps are
idempotent, so a night that cancelled mid-step runs ``erase_stored`` again to remove what the step
wrote (``NightDeps.purge``).
"""
from __future__ import annotations

import shutil
from contextlib import AbstractContextManager, nullcontext
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

Names = frozenset[str]


class CancelFlag(Protocol):
    def set(self, user_id: str) -> None: ...


class JobStopper(Protocol):
    """Stops this person's running GPU job on one runner; returns how many it stopped."""

    name: str

    def stop(self, user_id: str, adapters: Names) -> int: ...


class RemoteAdapters(Protocol):
    """Removes adapters from one remote store; returns how many were there."""

    name: str

    def remove(self, adapters: Names) -> int: ...


class Eraser(Protocol):
    """Removes one kind of the person's local files; returns how many files it removed."""

    name: str

    def erase(self, user_id: str, adapters: Names) -> int: ...


class Inventory(Protocol):
    def read(self, user_id: str) -> Names: ...

    def write(self, user_id: str, adapters: Names) -> None: ...

    def clear(self, user_id: str) -> None: ...


NightLock = Callable[[float], AbstractContextManager[None]]


@dataclass(frozen=True)
class DeletePorts:
    cancel: CancelFlag
    adapter_names: Callable[[str], Names]
    inventory: Inventory
    stoppers: tuple[JobStopper, ...] = ()
    remotes: tuple[RemoteAdapters, ...] = ()
    erasers: tuple[Eraser, ...] = ()  # in order; the one that removes the registry goes last
    night_lock: NightLock | None = None  # None: no night shares this host's lock (tests, erasure)
    lock_timeout_s: float = 120.0


@dataclass(frozen=True)
class DeleteReport:
    cancelled: bool
    jobs_stopped: int
    adapters: int  # how many the registry named
    remote_removed: tuple[tuple[str, int], ...] = ()
    erased: tuple[tuple[str, int], ...] = ()
    failures: tuple[str, ...] = field(default=())


class DeleteIncomplete(RuntimeError):
    """Some step failed. The rest were still tried; ``report`` says what was removed."""

    def __init__(self, report: DeleteReport) -> None:
        super().__init__(f"delete left work undone: {', '.join(report.failures)}")
        self.report = report


def _names(user_id: str, ports: DeletePorts, failures: list[str]) -> Names:
    """Every adapter to remove: the registry's, and any a failed earlier delete left in the inventory."""
    found: set[str] = set()
    for name, read in (("adapter names", ports.adapter_names), ("cleanup inventory", ports.inventory.read)):
        try:
            found |= read(user_id)
        except Exception as error:  # every step is tried; the failure is reported, never swallowed
            failures.append(f"{name}: {type(error).__name__}")
    return frozenset(found)


def _flag(user_id: str, ports: DeletePorts, failures: list[str]) -> bool:
    try:
        ports.cancel.set(user_id)
    except Exception as error:
        failures.append(f"cancel flag: {type(error).__name__}")
        return False
    return True


def _stop(user_id: str, ports: DeletePorts, adapters: Names, failures: list[str]) -> int:
    stopped = 0
    for stopper in ports.stoppers:
        try:
            stopped += stopper.stop(user_id, adapters)
        except Exception as error:
            failures.append(f"stop {stopper.name}: {type(error).__name__}")
    return stopped


def _remove(ports: DeletePorts, adapters: Names, failures: list[str]) -> tuple[tuple[str, int], ...]:
    removed = []
    for remote in ports.remotes:
        try:
            removed.append((remote.name, remote.remove(adapters)))
        except Exception as error:
            failures.append(f"remove {remote.name}: {type(error).__name__}")
    return tuple(removed)


def _erase(user_id: str, ports: DeletePorts, adapters: Names, failures: list[str]) -> tuple[tuple[str, int], ...]:
    erased = []
    for eraser in ports.erasers:
        try:
            erased.append((eraser.name, eraser.erase(user_id, adapters)))
        except Exception as error:
            failures.append(f"erase {eraser.name}: {type(error).__name__}")
    return tuple(erased)


def _record(user_id: str, ports: DeletePorts, adapters: Names, failures: list[str]) -> None:
    """The durable list of remote work, written before any remote delete."""
    try:
        ports.inventory.write(user_id, adapters)
    except Exception as error:
        failures.append(f"cleanup inventory: {type(error).__name__}")


def erase_stored(user_id: str, ports: DeletePorts) -> DeleteReport:
    """Steps 3 and 4 alone: the inventory, the remote copies, then the local files (the registry last)."""
    failures: list[str] = []
    adapters = _names(user_id, ports, failures)
    _record(user_id, ports, adapters, failures)
    before = len(failures)
    remote = _remove(ports, adapters, failures)
    erased = _erase(user_id, ports, adapters, failures)
    if len(failures) == before:  # every remote delete succeeded: nothing is left to retry
        _clear(user_id, ports, failures)
    return _settled(DeleteReport(False, 0, len(adapters), remote, erased, tuple(failures)))


def _clear(user_id: str, ports: DeletePorts, failures: list[str]) -> None:
    try:
        ports.inventory.clear(user_id)
    except Exception as error:
        failures.append(f"cleanup inventory: {type(error).__name__}")


def _drain(user_id: str, ports: DeletePorts, failures: list[str]) -> AbstractContextManager[None] | None:
    """The night lock, once the cancel flag is up: a running night reaches its next step boundary,
    sees the cancel, purges and leaves. On a timeout the person's Colab VM is stopped and the lock
    tried once more; a lock still not free is a failure, and nothing is deleted."""
    if ports.night_lock is None:
        return nullcontext()
    for attempt in (1, 2):
        lock = ports.night_lock(ports.lock_timeout_s)
        try:
            lock.__enter__()
            return lock
        except TimeoutError:
            if attempt == 1:
                _stop(user_id, ports, _names(user_id, ports, []), failures)
    failures.append("night lock: still held")
    return None


def delete_all(user_id: str, ports: DeletePorts) -> DeleteReport:
    """Fence, drain, enumerate, delete: the flag first, then the night lock (so no night or day is
    writing), and only then the adapters, registry entries and files are listed and removed."""
    failures: list[str] = []
    if not _flag(user_id, ports, failures):
        return _settled(DeleteReport(False, 0, 0, (), (), tuple(failures)))
    lock = _drain(user_id, ports, failures)
    if lock is None:
        return _settled(DeleteReport(True, 0, 0, (), (), tuple(failures)))
    try:
        adapters = _names(user_id, ports, failures)  # after the drain: a late registration is included
        _record(user_id, ports, adapters, failures)
        stopped = _stop(user_id, ports, adapters, failures)
        before = len(failures)
        remote = _remove(ports, adapters, failures)
        erased = _erase(user_id, ports, adapters, failures)
        if len(failures) == before:
            _clear(user_id, ports, failures)
    finally:
        lock.__exit__(None, None, None)
    return _settled(DeleteReport(True, stopped, len(adapters), remote, erased, tuple(failures)))


def _settled(report: DeleteReport) -> DeleteReport:
    if report.failures:
        raise DeleteIncomplete(report)
    return report


def remove_tree(path: Path) -> int:
    """Remove a file or a folder and count the files; a path already gone counts none."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return 1
    if not path.is_dir():
        return 0
    count = sum(1 for entry in path.rglob("*") if entry.is_file() or entry.is_symlink())
    shutil.rmtree(path, ignore_errors=False, onexc=_ignore_gone)
    return count


def _ignore_gone(function: object, path: str, error: BaseException) -> None:
    if not isinstance(error, FileNotFoundError):
        raise error


class PathEraser:
    """An ``Eraser`` over the paths ``locate(user_id, adapters)`` names."""

    def __init__(self, name: str, locate: Callable[[str, Names], Sequence[Path]]) -> None:
        self.name = name
        self._locate = locate

    def erase(self, user_id: str, adapters: Names) -> int:
        return sum(remove_tree(path) for path in self._locate(user_id, adapters))
