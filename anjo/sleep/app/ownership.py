"""Undo, delete and what changed on the sleep host (Task 9): the pure engine in ``anjo.sleep.ownership``
over Anjo's own stores.

- ``undo``, ``delete_all`` and ``what_changed`` are the spec's interfaces, with the checkpoint registry,
  the cancel flag and the results folder filled in.
- ``delete_all`` cancels the person's night (``<data dir>/sleep/cancel/<id>``), stops their Colab VM
  and cancels their running Modal training call, removes their adapters from the SageMaker bucket and
  the Modal volume when those are configured (the volume also whenever this host has trained on Modal,
  since the training job lands adapters there), and then removes the local files: the training copies,
  night logs and scores, sealed probes, MIND's latents and core state, registry, a pending note, the
  forced-id entry, the encrypted sleep folder, and last MIND's workspace log.
- ``remote_cleanup`` is account erasure's hook (``anjo.core.erasure``). Erasure already removes the
  person's folder under its own locks, so the hook does everything else and leaves that folder alone.
  It writes the cleanup inventory first (``<data dir>/sleep/cleanup/<id>.json``, adapter names only,
  outside the folder erasure removes); a remote delete that fails leaves it for ``retry_pending`` (the
  night's start) or a later ``delete_all``.
- ``delete_all`` also waits, up to ``LOCK_TIMEOUT_S``, for the night lock (``<data dir>/sleep/night.lock``,
  the one the night command and the simulated day hold) after raising the cancel flag.
- A delete leaves a tombstone, and the forced-id list is always the setting minus the tombstones.

Left behind on purpose: the cancel counter (an id and a number, so a night that is still running sees
the delete), the budget ledger (spend; its hold keys carry the account id and a date), the shared
persona files of synthetic people, and the deploy setting that holds the forced-id list (a restart
still reads it, and the tombstone takes the id out again).
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Final

from anjo.sleep.app.cleared_sessions import is_synthetic
from anjo.sleep.app.flags import FORCED_ENV, apply_tombstones, require_sleep_host
from anjo.sleep.app.storage import (
    SLEEP_DIR,
    WORKSPACE_FILE,
    checkpoint_registry,
    data_root,
    sleep_dir,
    writer_guard,
)
from anjo.sleep.mind.core.store import STORE_DIR
from anjo.sleep.note.hold import SOURCE as NOTE_SOURCE
from anjo.sleep.ownership import changes as _changes
from anjo.sleep.ownership import undo as _undo
from anjo.sleep.ownership.cancel import EpochFlag
from anjo.sleep.ownership.inventory import FileInventory
from anjo.sleep.ownership.lock import drain
from anjo.sleep.ownership.tombstone import Tombstones
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
CLEANUP_DIR: Final = "cleanup"
TOMBSTONE_DIR: Final = "deleted"
LOCK_NAME: Final = "night.lock"  # night/sweep.py's lock, beside the budget ledger
LOCK_TIMEOUT_S: Final = 120.0
TRAINING_DIR: Final = "training"  # LoraTrainer's root: adapters/<name> and work/<name>
PROJECTOR_SUFFIX: Final = ".projector.npz"  # train.lora's: a night's projector, beside its adapter (a test holds them equal)
SEALED_ENV: Final = "ANJO_SLEEP_SEALED_DIR"
BUCKET_ENV: Final = "ANJO_SLEEP_S3_BUCKET"
MODAL_URL_ENV: Final = "ANJO_SLEEP_MODAL_URL"
_USER: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _plain(user_id: str) -> str:
    if not isinstance(user_id, str) or not _USER.fullmatch(user_id):
        raise ValueError("a user id is a plain name")
    return user_id


def cancel_flag() -> EpochFlag:
    return EpochFlag(data_root() / SLEEP_DIR / CANCEL_DIR)


def inventory() -> FileInventory:
    return FileInventory(data_root() / SLEEP_DIR / CLEANUP_DIR)


def tombstones() -> Tombstones:
    return Tombstones(data_root() / SLEEP_DIR / TOMBSTONE_DIR)


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
    A person Anjo Sleep keeps nothing of is left alone: no cancel flag, no client, no call. The
    adapters are written to the inventory first, so a failure past this point loses nothing."""
    if not has_footprint(user_id):
        return
    inventory().write(user_id, adapter_names(user_id))
    _delete_all(_plain(user_id), delete_ports(under_erasure=True))


def retry_pending() -> list[str]:
    """Finish the remote deletes an earlier delete or erasure left in the inventory (the night's
    start). Returns the people whose work is still outstanding."""
    still: list[str] = []
    for user_id in inventory().pending():
        try:
            erase_stored(user_id, replace(delete_ports(under_erasure=True), erasers=()))
        except Exception:  # noqa: BLE001  (the inventory stays; the caller logs the people)
            still.append(user_id)
    return still


# ── The ports ─────────────────────────────────────────────────────────────────


def adapter_names(user_id: str) -> Names:
    """Every adapter ever trained for the person, promoted or not, and the ones that serve."""
    registry = checkpoint_registry()
    return registry.trained_names(user_id) | {ckpt.id for ckpt in registry.history(user_id)}


def _training(user_id: str, adapters: Names) -> list[Path]:
    """Each adapter's local copy and work folder, and the thought channel's projector its night trained,
    kept beside the copy (``<name>.projector.npz``, ``train.lora.PROJECTOR_SUFFIX``)."""
    root = data_root() / SLEEP_DIR / TRAINING_DIR
    return [path for name in sorted(adapters) for path in (
        root / "adapters" / name, root / "work" / name, root / "adapters" / f"{name}{PROJECTOR_SUFFIX}")]


def _results(user_id: str, adapters: Names) -> list[Path]:
    return [tree / _plain(user_id) for tree in results_trees()]


def _sealed(user_id: str, adapters: Names) -> list[Path]:
    root = os.environ.get(SEALED_ENV, "").strip()
    return [Path(root) / _plain(user_id)] if root else []


def _registry(user_id: str, adapters: Names) -> list[Path]:
    return [checkpoint_registry().root / _plain(user_id)]


def _latents(user_id: str, adapters: Names) -> list[Path]:
    """MIND's event latents of the person: their cache and their folder in every run (loaded by name:
    the module reaches the training code)."""
    latents = importlib.import_module("anjo.sleep.mind.core.latents")
    return list(latents.erase_paths(data_root() / SLEEP_DIR / latents.LATENTS_DIR, _plain(user_id)))


def _mind_state(user_id: str, adapters: Names) -> list[Path]:
    """MIND's core state of the person: every night's unit and state (``mind.core.store``)."""
    return [data_root() / SLEEP_DIR / STORE_DIR / _plain(user_id)]


class _PendingNote:
    """A held note still waiting for the person's next open: it was written from what is deleted."""

    name = "pending note"

    def erase(self, user_id: str, adapters: Names) -> int:
        from anjo.core.outreach import get_pending_outreach_source, withdraw_pending_outreach

        if get_pending_outreach_source(user_id) == NOTE_SOURCE:
            return 1 if withdraw_pending_outreach(user_id) else 0
        return 0


class _ForcedId:
    """The person's entry in the forced-id list (the dev host's cohort): a tombstone, so a restart
    can't bring it back, applied to this process now."""

    name = "forced-id entry"

    def erase(self, user_id: str, adapters: Names) -> int:
        listed = user_id in [part.strip() for part in os.environ.get(FORCED_ENV, "").split(",")]
        tombstones().add(user_id)
        apply_tombstones()
        return 1 if listed else 0


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


def _modal_calls() -> Any:
    """The Modal runner's call records (``runners/modal.py``), loaded by name like the stoppers."""
    modal = importlib.import_module("anjo.sleep.train.runners.modal")
    return modal.CallRecords(data_root() / SLEEP_DIR / modal.CALLS_DIR)


def _trained_on_modal() -> bool:
    """Whether this host has started a Modal training call: its call records' folder exists."""
    return bool(_modal_calls().used())


def _remotes() -> tuple[RemoteAdapters, ...]:
    stores: list[RemoteAdapters] = []
    bucket = os.environ.get(BUCKET_ENV, "").strip()
    if bucket:
        from anjo.sleep.serving.sagemaker.invoke import make_client

        stores.append(S3Adapters(bucket, make_client("s3", read_s=60.0, attempts=3)))
    if os.environ.get(MODAL_URL_ENV, "").strip() or _trained_on_modal():
        from anjo.sleep.serving.config import ADAPTERS_VOLUME

        volume: Any = importlib.import_module("modal").Volume.from_name(ADAPTERS_VOLUME)
        stores.append(ModalAdapters(volume))
    return tuple(stores)


class _JobStopper:
    """Stops the person's training jobs on one service. Loaded by name: the app never imports the
    training code."""

    def __init__(self, name: str, stop: Callable[[str, Names], int]) -> None:
        self.name, self._stop = name, stop

    def stop(self, user_id: str, adapters: Names) -> int:
        return self._stop(user_id, adapters)


def _stoppers() -> tuple[_JobStopper, ...]:
    found: list[_JobStopper] = []
    colab = importlib.import_module("anjo.sleep.train.runners.colab")
    if Path(colab.COLAB_BIN).is_file():  # with no Colab CLI on this machine, nothing here started a VM
        cli = colab.ColabCli()
        found.append(_JobStopper("colab", lambda user_id, names: colab.stop_person_sessions(cli, user_id, names)))
    records = _modal_calls()
    if records.used():  # with no call ever started here, nothing to cancel and no Modal client made
        stop = importlib.import_module("anjo.sleep.train.runners.modal").stop_person_calls
        found.append(_JobStopper("modal", lambda user_id, names: stop(records, user_id, names)))
    latent = importlib.import_module("anjo.sleep.train.latent_runner")
    encodes = latent.CallRecords(data_root() / SLEEP_DIR / latent.CALLS_DIR)
    if encodes.used():  # an encode killed hard may have left the person's events on the latents volume
        found.append(_JobStopper("modal latents", lambda user_id, names: latent.stop_person_jobs(encodes, user_id)))
    return tuple(found)


def local_erasers(*, under_erasure: bool = False) -> Sequence[Any]:
    """In order: the registry that names the adapters goes last of all."""
    files = [
        PathEraser("training copies", _training), PathEraser("night logs and scores", _results),
        PathEraser("sealed probes", _sealed), PathEraser("mind latents", _latents),
        PathEraser("mind state", _mind_state), _PendingNote(), _ForcedId(),
    ]
    registry = PathEraser("checkpoint registry", _registry)
    return [*files, *([] if under_erasure else [_SleepFiles(), _MindWorkspace()]), registry]


def delete_ports(*, under_erasure: bool = False) -> DeletePorts:
    """Under erasure the night lock is not taken: erasure's own guard already refuses a writer."""
    lock_path = data_root() / SLEEP_DIR / LOCK_NAME
    return DeletePorts(
        cancel=cancel_flag(), adapter_names=adapter_names, inventory=inventory(), stoppers=_stoppers(),
        remotes=_remotes(), erasers=tuple(local_erasers(under_erasure=under_erasure)),
        night_lock=None if under_erasure else (lambda timeout: drain(lock_path.resolve(), timeout)),
        lock_timeout_s=LOCK_TIMEOUT_S,
    )
