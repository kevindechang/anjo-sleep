"""The host's side of one encoder call on the Modal training app (``latent_job``): hold, record,
upload, spawn, wait, fetch, remove, settle.

1. Hold the call's worst case on ``modal_usd``, the Modal training pool, before anything touches
   Modal: the job's hard timeout (``latent_job.FUNCTION_TIMEOUT_S``) at Modal's A100-80GB rate,
   keyed by the job with an attempt number (``holds.hold``). With no room, ``BudgetStop``.
2. Record the job (``CallRecords``: its name, the lives it reads, and its Modal call id once there
   is one) before the upload, so a run killed at any point after it leaves a record that names
   the folder holding the events' text and the call that may still run. Modal looks a call up by
   its id only, so the container's first act is to write its own id into the job's folder
   (``latent_job.mark_call``): a run killed between the spawn and the record's id is still found by
   the job's name (``find_call``). One whose container hasn't started yet finds its folder gone and
   stops at its first read; either way its hold is charged in full.
3. Upload the events to the latents volume as ``<job>/events.jsonl``, then spawn the call with the
   job's name, the spec and the seconds its work may take, and nothing else: no key reaches the
   container, which reads Glimmer off the weights volume at the pinned revision.
4. Wait for what is left of the hard timeout, less a cancel's margin and the container's idle
   tail, then fetch the job's two files and check that they answer the keys sent, in order.
5. In a ``finally``, on success or failure: cancel a call that may still run, remove the job's
   folder from the volume (the events' text never stays there), then settle the wall time from the
   hold to the call's end plus the idle tail, never above the hold, and drop the record. A hold no
   call used is released; a spawn that failed partway settles all of it, since it may have started
   a container. A cancel that fails settles the whole hold and raises; a remove that fails raises.
   Either way the record stays.

**A run killed hard** runs no ``finally``. Every encode first takes MIND's lock (``mind.sweep``), so
while it holds it no other MIND command is live on this host, and then sweeps (``sweep``): it
cancels each recorded call, removes its folder, drops its record, and settles every open hold of
``HOLD_KIND`` at its full bound (work may have run, so the worst case is the honest charge). The new
hold then takes the next attempt number. MIND's own sweep settles these holds too
(``mind.sweep.MIND_KINDS``). A delete removes the person's recorded jobs the same way
(``stop_person_jobs``). The records hold a job name (a hash), a call id and life ids, under
``<data dir>/sleep/latent_calls/``.

``modal`` is imported only when a real call is made (``SdkLatentCalls``).
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np

from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.night.holds import hold
from anjo.sleep.train import latent_job, modal_job
from anjo.sleep.train.runners import modal as modal_runner

POOL: Final = modal_runner.POOL  # the Modal training pool
A100_USD_PER_HOUR: Final = modal_runner.A100_USD_PER_HOUR
HOLD_KIND: Final = "modal_latents"
CALLS_DIR: Final = "latent_calls"  # under <data dir>/sleep/
CALL_STATES = Literal["none", "maybe", "running", "done"]


class LatentJobError(RuntimeError):
    """The call failed, ran out of time, or sent back latents that don't answer the events sent."""


class LatentCalls(Protocol):
    def upload(self, job: str, data: bytes) -> None:
        """Put the events at ``<job>/events.jsonl`` on the latents volume."""
        ...

    def spawn(self, job: str, spec_json: str, exec_s: float) -> str:
        """Start the call; returns its id."""
        ...

    def wait(self, call_id: str, timeout_s: float) -> Mapping[str, object]:
        """The call's result; ``TimeoutError`` past ``timeout_s``."""
        ...

    def cancel(self, call_id: str) -> None:
        """Cancel the call and stop its container."""
        ...

    def fetch(self, job: str, dest: Path) -> None:
        """Copy the job's ``latents.npy`` and ``index.jsonl`` into ``dest``."""
        ...

    def remove(self, job: str) -> None:
        """Remove the job's folder from the latents volume; a folder already gone is no error."""
        ...

    def find_call(self, job: str) -> str | None:
        """The call id the job's container left in its folder (``latent_job.mark_call``), if any."""
        ...


class SdkLatentCalls:
    """``LatentCalls`` through the ``modal`` package, on the deployed training app."""

    def __init__(self, app_name: str = modal_job.APP_NAME, function: str = latent_job.FUNCTION) -> None:
        self._app, self._function = app_name, function

    def _volume(self) -> Any:
        modal: Any = importlib.import_module("modal")
        return modal.Volume.from_name(latent_job.VOLUME, create_if_missing=True)

    def upload(self, job: str, data: bytes) -> None:
        with self._volume().batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), f"/{latent_job.check_job(job)}/{latent_job.EVENTS_FILE}")

    def spawn(self, job: str, spec_json: str, exec_s: float) -> str:
        modal: Any = importlib.import_module("modal")
        return str(modal.Function.from_name(self._app, self._function).spawn(job, spec_json, exec_s).object_id)

    def wait(self, call_id: str, timeout_s: float) -> Mapping[str, object]:
        modal: Any = importlib.import_module("modal")
        try:
            found: Mapping[str, object] = modal.FunctionCall.from_id(call_id).get(timeout=timeout_s)
        except modal.exception.TimeoutError as error:
            raise TimeoutError("the encoder call ran past its hard timeout") from error
        return found

    def cancel(self, call_id: str) -> None:
        modal: Any = importlib.import_module("modal")
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def fetch(self, job: str, dest: Path) -> None:
        volume = self._volume()
        for name in (latent_job.MATRIX_FILE, latent_job.INDEX_FILE):
            with open(dest / name, "wb") as handle:
                for chunk in volume.read_file(f"/{latent_job.check_job(job)}/{name}"):
                    handle.write(chunk)

    def remove(self, job: str) -> None:
        modal: Any = importlib.import_module("modal")
        try:
            self._volume().remove_file(f"/{latent_job.check_job(job)}", recursive=True)
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass  # already gone: what a remove wants

    def find_call(self, job: str) -> str | None:
        modal: Any = importlib.import_module("modal")
        try:
            data = b"".join(self._volume().read_file(f"/{latent_job.check_job(job)}/{latent_job.CALL_FILE}"))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None  # its container hasn't started: removing the folder stops it at its first read
        return latent_job.read_call(data)


@dataclass(frozen=True)
class CallRecord:
    job: str
    call_id: str | None
    lives: tuple[str, ...]


class CallRecords:
    """The jobs that may have left text on the volume or a call running: one small file per job."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, job: str) -> Path:
        return self.root / f"{latent_job.check_job(job)}.json"

    def used(self) -> bool:
        """Whether this host has ever started an encoder job."""
        return self.root.is_dir()

    def add(self, job: str, call_id: str | None, lives: Iterable[str]) -> None:
        path = self._path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        pending.write_text(json.dumps({"call": call_id, "lives": sorted(set(lives))}), encoding="utf-8")
        os.replace(pending, path)

    def remove(self, job: str) -> None:
        self._path(job).unlink(missing_ok=True)

    def items(self) -> list[CallRecord]:
        if not self.root.is_dir():
            return []
        found = [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.root.glob("j-*.json"))]
        return [CallRecord(job, row.get("call"), tuple(row.get("lives", ()))) for job, row in found]


@dataclass(frozen=True)
class LatentSweep:
    calls_cancelled: int
    folders_removed: int
    holds_settled: int  # at their full bound


def _clear(records: CallRecords, calls: LatentCalls, found: Iterable[CallRecord]) -> int:
    cleared = 0
    for record in found:
        call_id = record.call_id or calls.find_call(record.job)  # killed before its id was recorded: by its job
        if call_id is not None:
            try:
                calls.cancel(call_id)
            except Exception as error:  # any failure of Modal's client: the call may still run, so say so
                raise LatentJobError(f"cancelling call {call_id} failed; cancel it by hand before "
                                     "anything else") from error
        calls.remove(record.job)
        records.remove(record.job)
        cleared += 1
    return cleared


def sweep(records: CallRecords, calls: LatentCalls, ledger: Ledger) -> LatentSweep:
    """Under MIND's lock, before any new hold: clear what killed runs left, then settle their holds."""
    found = records.items()
    _clear(records, calls, found)
    stale = [held for held in ledger.open_holds() if held.job.startswith(f"{HOLD_KIND}:")]
    for held in stale:
        held.settle(held.bound.amount)
    return LatentSweep(sum(record.call_id is not None for record in found), len(found), len(stale))


def stop_person_jobs(records: CallRecords, person_id: str, *, calls: LatentCalls | None = None) -> int:
    """A delete: cancel and remove every recorded job that read ``person_id``'s events. Modal is
    reached only if one is recorded."""
    mine = [record for record in records.items() if person_id in record.lives]
    if not mine:
        return 0
    return _clear(records, calls if calls is not None else SdkLatentCalls(), mine)


@dataclass(frozen=True)
class JobOutcome:
    """What one call returned: the latents in the order sent, whether each event was cut, the
    container's own counts, the wall time settled and its dollars."""

    matrix: np.ndarray
    truncated: tuple[bool, ...]
    stats: Mapping[str, object]
    elapsed_s: float
    usd: float


def hold_bound(timeout_s: float = latent_job.FUNCTION_TIMEOUT_S, rate: float = A100_USD_PER_HOUR) -> Bound:
    """The call's worst case: its hard timeout at the A100-80GB rate."""
    return Bound(rate, timeout_s)


def job_name(spec_json: str, keys: Sequence[str]) -> str:
    """The job's folder name: a hash of what it reads and how."""
    digest = hashlib.sha256("\n".join([spec_json, *keys]).encode("utf-8")).hexdigest()
    return f"j-{digest[:16]}"


@dataclass
class _Call:
    state: CALL_STATES = "none"
    call_id: str | None = None
    uploading: bool = False


@dataclass(frozen=True)
class JobInput:
    """What one call reads: the content keys in order, the uploaded events, the spec, and the lives
    the events belong to (for the record a delete reads)."""

    keys: tuple[str, ...]
    events_jsonl: bytes
    spec_json: str
    lives: tuple[str, ...]


def run_job(
    job_input: JobInput, *, calls: LatentCalls, ledger: Ledger, records: CallRecords, clock: Callable[[], float],
    scratch: Path, log: Callable[[str], None], timeout_s: float = latent_job.FUNCTION_TIMEOUT_S,
    rate: float = A100_USD_PER_HOUR,
) -> JobOutcome:
    """Encode the uploaded events in one call; see the module's steps."""
    job = job_name(job_input.spec_json, job_input.keys)
    exec_s = modal_job.exec_seconds(timeout_s)
    reservation = hold(ledger, f"{HOLD_KIND}:{job}", (POOL,), hold_bound(timeout_s, rate))
    started = clock()
    call = _Call()
    try:
        records.add(job, None, job_input.lives)  # before the upload: a killed run's record names the folder
        call.uploading = True
        calls.upload(job, job_input.events_jsonl)
        stats = _call(calls, call, job, job_input, exec_s, started, timeout_s, clock, records)
        matrix, truncated = _fetched(calls, job, job_input.keys, scratch)
    finally:
        elapsed, usd = _finish(calls, call, job, records, reservation, started, clock, rate)
    log(f"latents job {job}: {stats.get('events')} events, {stats.get('tokens')} tokens, {sum(truncated)} cut, "
        f"load {_seconds(stats.get('load_seconds'))} + encode {_seconds(stats.get('seconds'))} seconds in the "
        f"container; {elapsed:.0f} seconds settled, ${usd:.2f} on {POOL}")
    return JobOutcome(matrix, truncated, stats, elapsed, usd)


def _seconds(value: object) -> str:
    return f"{value:.0f}" if isinstance(value, int | float) else "?"


def _call(
    calls: LatentCalls, call: _Call, job: str, job_input: JobInput, exec_s: float, started: float, timeout_s: float,
    clock: Callable[[], float], records: CallRecords,
) -> Mapping[str, object]:
    call.state = "maybe"  # a spawn that fails may still have started the call
    call.call_id = calls.spawn(job, job_input.spec_json, exec_s)
    call.state = "running"
    records.add(job, call.call_id, job_input.lives)
    left = timeout_s - (clock() - started) - modal_job.CANCEL_S - latent_job.SCALEDOWN_S
    try:
        stats = calls.wait(call.call_id, max(0.0, left))
    except TimeoutError as error:
        raise LatentJobError("the encoder call ran out of its hard timeout") from error
    call.state = "done"
    return stats


def _fetched(calls: LatentCalls, job: str, keys: Sequence[str], scratch: Path) -> tuple[np.ndarray, tuple[bool, ...]]:
    dest = scratch / job
    dest.mkdir(parents=True, exist_ok=True)
    try:
        calls.fetch(job, dest)
        _, rows, matrix = latent_job.read_outputs(dest)
    except latent_job.LatentSpecError as error:
        raise LatentJobError(f"the encoder call's files can't be read: {error}") from error
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    if [row.get("key") for row in rows] != list(keys):
        raise LatentJobError("the encoder call's latents don't answer the keys sent, in order")
    if not all(isinstance(row.get("truncated"), bool) for row in rows):
        raise LatentJobError("the encoder call's index doesn't say which events were cut")
    return matrix, tuple(bool(row["truncated"]) for row in rows)


def _finish(
    calls: LatentCalls, call: _Call, job: str, records: CallRecords, reservation: Reservation, started: float,
    clock: Callable[[], float], rate: float,
) -> tuple[float, float]:
    """Cancel a call that may still run, remove the job's folder, then settle what it ran for (release
    a hold no call used). The record goes only once the call is stopped and the folder is gone."""
    if call.state == "running" and call.call_id is not None:
        try:
            calls.cancel(call.call_id)
        except Exception as error:  # any failure of Modal's client: the call may still run, so say so
            reservation.settle(reservation.bound.amount)
            raise LatentJobError(f"cancelling call {call.call_id} failed; cancel it by hand") from error
    removed: Exception | None = None
    if call.uploading:
        try:
            calls.remove(job)
        except Exception as error:  # any failure of Modal's client: the text may still be there, so say so
            removed = error
    elapsed = max(0.0, clock() - started) + latent_job.SCALEDOWN_S
    if call.state == "none":
        reservation.release()
        usd = 0.0
    else:
        usd = reservation.bound.amount if call.state == "maybe" else min(elapsed / 3600.0 * rate, reservation.bound.amount)
        reservation.settle(usd)
    if removed is not None:
        raise LatentJobError(f"removing job {job}'s events from the volume failed; the next encode removes "
                             "them") from removed
    if call.state != "maybe":  # a spawn that failed partway may have started a call: the sweep looks it up by its job
        records.remove(job)
    return elapsed, usd
