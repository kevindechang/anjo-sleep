"""Undo, delete and what changed on the sleep host (Task 9): the pure engine in ``anjo.sleep.ownership``
over Anjo's own stores.

- ``undo``, ``delete_all`` and ``what_changed`` are the spec's interfaces, with the checkpoint registry,
  the cancel flag and the results folder filled in.
- ``delete_all`` cancels the person's night (``<data dir>/sleep/cancel/<id>``), stops their Colab VM,
  removes their adapters from the SageMaker bucket and the Modal volume when those are configured, and
  then removes the local files: the training copies, night logs and scores, sealed probes, registry,
  a pending note, the forced-id entry, the encrypted sleep folder, and last MIND's workspace log.
- ``remote_cleanup`` is account erasure's hook (``anjo.core.erasure``). Erasure already removes the
  person's folder under its own locks, so the hook does everything else and leaves that folder alone.

Left behind on purpose: the cancel counter (an id and a number, so a night that is still running sees
the delete), the budget ledger (spend; its hold keys carry the account id and a date), the shared
persona files of synthetic people, and the process environment's forced-id list outside this
process (a deploy setting).
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.storage import (
    SLEEP_DIR,
    WORKSPACE_FILE,
    checkpoint_registry,
    data_root,
    sleep_dir,
    writer_guard,
)
from anjo.sleep.note.hold import SOURCE as NOTE_SOURCE
from anjo.sleep.ownership import changes as _changes
from anjo.sleep.ownership import undo as _undo
from anjo.sleep.ownership.cancel import EpochFlag
from anjo.sleep.ownership.changes import Change
from anjo.sleep.ownership.delete import (
    DeletePorts,
    DeleteReport,
    Names,
    PathEraser,
    RemoteAdapters,
    erase_stored,
    remove_tree,
)
from anjo.sleep.ownership.delete import delete_all as _delete_all
from anjo.sleep.ownership.remote import ModalAdapters, S3Adapters
from anjo.sleep.types import CheckpointRef

CANCEL_DIR: Final = "cancel"
TRAINING_DIR: Final = "training"  # LoraTrainer's root: adapters/<name> and work/<name>
SEALED_ENV: Final = "ANJO_SLEEP_SEALED_DIR"
FORCED_ENV: Final = "ANJO_SLEEP_FORCED_USER_IDS"
BUCKET_ENV: Final = "ANJO_SLEEP_S3_BUCKET"
MODAL_URL_ENV: Final = "ANJO_SLEEP_MODAL_URL"
_USER: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _plain(user_id: str) -> str:
    if not isinstance(user_id, str) or not _USER.fullmatch(user_id):
        raise ValueError("a user id is a plain name")
    return user_id


def cancel_flag() -> EpochFlag:
    return EpochFlag(data_root() / SLEEP_DIR / CANCEL_DIR)


def results_trees() -> tuple[Path, Path]:
    """Where nights and scores go: a synthetic person's under the published tree, a real one's not."""
    base = data_root() / SLEEP_DIR
    return base / "results", base / "private_results"


def person_results(user_id: str) -> Path:
    published, private = results_trees()
    return (published if is_synthetic(user_id) else private) / _plain(user_id)


def night_logs(user_id: str) -> Path:
    return person_results(user_id) / "nights"


# ── The interfaces ────────────────────────────────────────────────────────────


def undo(user_id: str) -> CheckpointRef:
    require_sleep_host()
    return _undo.undo(user_id, checkpoint_registry())


def what_changed(user_id: str, since: date) -> list[Change]:
    require_sleep_host()
    return _changes.what_changed(user_id, since, results=person_results)


def delete_all(user_id: str) -> DeleteReport:
    require_sleep_host()
    return _delete_all(_plain(user_id), delete_ports())


def purge(user_id: str) -> None:
    """What a night that was cancelled mid-step runs: the delete's removals again, without its cancel."""
    erase_stored(_plain(user_id), delete_ports())


def has_footprint(user_id: str) -> bool:
    """Whether Anjo Sleep keeps anything of the person: a registry folder, a sleep folder or results."""
    return any(path.exists() for path in (
        checkpoint_registry().root / _plain(user_id), sleep_dir(user_id), *(tree / user_id for tree in results_trees()),
    ))


def remote_cleanup(user_id: str) -> None:
    """Account erasure's hook. It runs inside erasure's locks, which also remove the person's folder.
    A person Anjo Sleep keeps nothing of is left alone: no cancel flag, no client, no call."""
    if has_footprint(user_id):
        _delete_all(_plain(user_id), delete_ports(under_erasure=True))


# ── The ports ─────────────────────────────────────────────────────────────────


def adapter_names(user_id: str) -> Names:
    """Every adapter ever trained for the person, promoted or not, and the ones that serve."""
    registry = checkpoint_registry()
    return registry.trained_names(user_id) | {ckpt.id for ckpt in registry.history(user_id)}


def _training(user_id: str, adapters: Names) -> list[Path]:
    root = data_root() / SLEEP_DIR / TRAINING_DIR
    return [root / part / name for name in sorted(adapters) for part in ("adapters", "work")]


def _results(user_id: str, adapters: Names) -> list[Path]:
    return [tree / _plain(user_id) for tree in results_trees()]


def _sealed(user_id: str, adapters: Names) -> list[Path]:
    root = os.environ.get(SEALED_ENV, "").strip()
    return [Path(root) / _plain(user_id)] if root else []


def _registry(user_id: str, adapters: Names) -> list[Path]:
    return [checkpoint_registry().root / _plain(user_id)]


class _PendingNote:
    """A held note still waiting for the person's next open: it was written from what is deleted."""

    name = "pending note"

    def erase(self, user_id: str, adapters: Names) -> int:
        from anjo.core.outreach import get_pending_outreach_source, withdraw_pending_outreach

        if get_pending_outreach_source(user_id) == NOTE_SOURCE:
            return 1 if withdraw_pending_outreach(user_id) else 0
        return 0


class _ForcedId:
    """The person's entry in this process's forced-id list (the dev host's cohort)."""

    name = "forced-id entry"

    def erase(self, user_id: str, adapters: Names) -> int:
        ids = [part.strip() for part in os.environ.get(FORCED_ENV, "").split(",") if part.strip()]
        if user_id not in ids:
            return 0
        os.environ[FORCED_ENV] = ",".join(part for part in ids if part != user_id)
        return 1


class _SleepFiles:
    """The encrypted sleep folder except MIND's workspace log, removed while no writer is in flight:
    evidence, targets with their log, receipts, expectations and snapshots, extractions, briefs,
    practice transcripts, reply ancestry and cleared-session records."""

    name = "sleep files"

    def erase(self, user_id: str, adapters: Names) -> int:
        folder = sleep_dir(user_id)
        if not folder.is_dir():
            return 0
        with writer_guard(user_id)():
            return sum(remove_tree(entry) for entry in sorted(folder.iterdir()) if entry.name != WORKSPACE_FILE)


class _MindWorkspace:
    """MIND's workspace log, and the folder once it is empty."""

    name = "mind workspace"

    def erase(self, user_id: str, adapters: Names) -> int:
        folder = sleep_dir(user_id)
        if not folder.is_dir():
            return 0
        with writer_guard(user_id)():
            removed = remove_tree(folder / WORKSPACE_FILE)
            if not any(folder.iterdir()):
                shutil.rmtree(folder, ignore_errors=True)
        return removed


def _remotes() -> tuple[RemoteAdapters, ...]:
    stores: list[RemoteAdapters] = []
    bucket = os.environ.get(BUCKET_ENV, "").strip()
    if bucket:
        from anjo.sleep.serving.sagemaker.invoke import make_client

        stores.append(S3Adapters(bucket, make_client("s3", read_s=60.0, attempts=3)))
    if os.environ.get(MODAL_URL_ENV, "").strip():
        from anjo.sleep.serving.config import ADAPTERS_VOLUME

        volume: Any = importlib.import_module("modal").Volume.from_name(ADAPTERS_VOLUME)
        stores.append(ModalAdapters(volume))
    return tuple(stores)


class _ColabStopper:
    """Stops the person's Colab sessions. Loaded by name: the app never imports the training code."""

    name = "colab"

    def __init__(self, stop: Callable[[Any, str, Names], int], cli: Any) -> None:
        self._stop, self._cli = stop, cli

    def stop(self, user_id: str, adapters: Names) -> int:
        return self._stop(self._cli, user_id, adapters)


def _stoppers() -> tuple[_ColabStopper, ...]:
    colab = importlib.import_module("anjo.sleep.train.runners.colab")
    if not Path(colab.COLAB_BIN).is_file():
        return ()  # no Colab CLI on this machine: nothing here can have started a VM
    return (_ColabStopper(colab.stop_person_sessions, colab.ColabCli()),)


def local_erasers(*, under_erasure: bool = False) -> Sequence[Any]:
    files = [
        PathEraser("training copies", _training), PathEraser("night logs and scores", _results),
        PathEraser("sealed probes", _sealed), PathEraser("checkpoint registry", _registry),
        _PendingNote(), _ForcedId(),
    ]
    return [*files, *([] if under_erasure else [_SleepFiles(), _MindWorkspace()])]


def delete_ports(*, under_erasure: bool = False) -> DeletePorts:
    return DeletePorts(
        cancel=cancel_flag(), adapter_names=adapter_names, stoppers=_stoppers(), remotes=_remotes(),
        erasers=tuple(local_erasers(under_erasure=under_erasure)),
    )
