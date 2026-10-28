"""The host's side of one encoder call on the Modal training app (``latent_job``): hold, upload,
spawn, wait, fetch, settle.

1. Hold the call's worst case on ``modal_usd``, the Modal training pool, before anything touches
   Modal: the job's hard timeout (``latent_job.FUNCTION_TIMEOUT_S``) at Modal's A100-80GB rate,
   keyed by the job with an attempt number (``holds.hold``). With no room, ``BudgetStop``.
2. Upload the events to the latents volume as ``<job>/events.jsonl``, then spawn the call with the
   job's name, the spec and the seconds its work may take, and nothing else: no key reaches the
   container, which reads Glimmer off the weights volume at the pinned revision.
3. Wait for what is left of the hard timeout, less a cancel's margin and the container's idle
   tail, then fetch the job's two files and check that they answer the keys sent, in order.
4. Remove the job's folder from the volume once its files are fetched (a failed job's folder stays,
   and a rerun of the same job overwrites it). In a ``finally``: cancel a call that may still run,
   then settle the wall time from the hold to the call's end plus the idle tail, never above the
   hold. That upper-bounds the GPU seconds Modal bills for the call. A hold no call used is
   released; a spawn that failed partway settles all of it, since it may have started a container.

``modal`` is imported only when a real call is made (``SdkLatentCalls``).
"""
from __future__ import annotations

import hashlib
import importlib
import io
import shutil
from collections.abc import Callable, Mapping, Sequence
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
        """Remove the job's folder from the latents volume."""
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
        self._volume().remove_file(f"/{latent_job.check_job(job)}", recursive=True)


@dataclass(frozen=True)
class JobOutcome:
    """What one call returned: the latents in the order sent, the container's own counts, the wall
    time settled and its dollars."""

    matrix: np.ndarray
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


def run_job(
    keys: Sequence[str], events_jsonl: bytes, spec_json: str, *, calls: LatentCalls, ledger: Ledger,
    clock: Callable[[], float], scratch: Path, log: Callable[[str], None],
    timeout_s: float = latent_job.FUNCTION_TIMEOUT_S, rate: float = A100_USD_PER_HOUR,
) -> JobOutcome:
    """Encode the uploaded events in one call; see the module's steps."""
    job = job_name(spec_json, keys)
    exec_s = modal_job.exec_seconds(timeout_s)
    reservation = hold(ledger, f"{HOLD_KIND}:{job}", (POOL,), hold_bound(timeout_s, rate))
    started = clock()
    call = _Call()
    try:
        calls.upload(job, events_jsonl)
        stats = _call(calls, call, job, spec_json, exec_s, started, timeout_s, clock)
        matrix = _fetched(calls, job, keys, scratch)
        calls.remove(job)  # only once fetched; a failed job's folder stays, and a rerun overwrites it
    finally:
        elapsed, usd = _finish(calls, call, reservation, started, clock, rate)
    log(f"latents job {job}: {stats.get('events')} events, {stats.get('tokens')} tokens, "
        f"load {_seconds(stats.get('load_seconds'))} + encode {_seconds(stats.get('seconds'))} seconds in the "
        f"container; {elapsed:.0f} seconds settled, ${usd:.2f} on {POOL}")
    return JobOutcome(matrix, stats, elapsed, usd)


def _seconds(value: object) -> str:
    return f"{value:.0f}" if isinstance(value, int | float) else "?"


def _call(
    calls: LatentCalls, call: _Call, job: str, spec_json: str, exec_s: float, started: float, timeout_s: float,
    clock: Callable[[], float],
) -> Mapping[str, object]:
    call.state = "maybe"  # a spawn that fails may still have started the call
    call.call_id = calls.spawn(job, spec_json, exec_s)
    call.state = "running"
    left = timeout_s - (clock() - started) - modal_job.CANCEL_S - latent_job.SCALEDOWN_S
    try:
        stats = calls.wait(call.call_id, max(0.0, left))
    except TimeoutError as error:
        raise LatentJobError("the encoder call ran out of its hard timeout") from error
    call.state = "done"
    return stats


def _fetched(calls: LatentCalls, job: str, keys: Sequence[str], scratch: Path) -> np.ndarray:
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
    return matrix


def _finish(
    calls: LatentCalls, call: _Call, reservation: Reservation, started: float, clock: Callable[[], float], rate: float,
) -> tuple[float, float]:
    """Cancel a call that may still run, then settle what it ran for (release a hold no call used)."""
    if call.state == "running" and call.call_id is not None:
        try:
            calls.cancel(call.call_id)
        except Exception as error:  # any failure of Modal's client: the call may still run, so say so
            reservation.settle(reservation.bound.amount)
            raise LatentJobError(f"cancelling call {call.call_id} failed; cancel it by hand") from error
    elapsed = max(0.0, clock() - started) + latent_job.SCALEDOWN_S
    if call.state == "none":
        reservation.release()
        return elapsed, 0.0
    usd = reservation.bound.amount if call.state == "maybe" else min(elapsed / 3600.0 * rate, reservation.bound.amount)
    reservation.settle(usd)
    return elapsed, usd
