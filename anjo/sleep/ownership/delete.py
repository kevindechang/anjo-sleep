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


@dataclass(frozen=True)
class DeletePorts:
    cancel: CancelFlag
    adapter_names: Callable[[str], Names]
    stoppers: tuple[JobStopper, ...] = ()
    remotes: tuple[RemoteAdapters, ...] = ()
    erasers: tuple[Eraser, ...] = ()


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
    try:
        return ports.adapter_names(user_id)
    except Exception as error:  # every step is tried; the failure is reported, never swallowed
        failures.append(f"adapter names: {type(error).__name__}")
        return frozenset()


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


def erase_stored(user_id: str, ports: DeletePorts) -> DeleteReport:
    """Steps 3 and 4 alone: the remote copies, then the local files, MIND's workspace last."""
    failures: list[str] = []
    adapters = _names(user_id, ports, failures)
    remote = _remove(ports, adapters, failures)
    erased = _erase(user_id, ports, adapters, failures)
    return _settled(DeleteReport(False, 0, len(adapters), remote, erased, tuple(failures)))


def delete_all(user_id: str, ports: DeletePorts) -> DeleteReport:
    failures: list[str] = []
    adapters = _names(user_id, ports, failures)  # before an eraser removes the registry that lists them
    cancelled = _flag(user_id, ports, failures)
    stopped = _stop(user_id, ports, adapters, failures)
    remote = _remove(ports, adapters, failures)
    erased = _erase(user_id, ports, adapters, failures)
    return _settled(DeleteReport(cancelled, stopped, len(adapters), remote, erased, tuple(failures)))


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
