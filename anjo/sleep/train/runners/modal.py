"""The Modal ``GpuRunner`` the night trains on (Task 5; Kevin, 2026-10-28: Modal's $100 hackathon
credit first, Colab the fallback).

One training job is one call of the deployed training app (``anjo-sleep-train``,
``train/modal_app.py``) on one A100-80GB:

1. refuse a person who is neither registered synthetic nor one of the real people the runner names
   (``allow_real``: Kevin's own account, which only the night command's ``--allow-real`` names, §12.1).
   Anyone else real is refused, as on Colab (``RealPersonRefused``);
2. hold the job's worst case on ``modal_usd`` before anything starts: its hard timeout at Modal's
   A100-80GB rate (``A100_USD_PER_HOUR``), keyed by night and person with an attempt number
   (``holds.hold``). With no room, the night stops (``BudgetStop``);
3. spawn the call with the night's dataset text, its recipe, a warm start's two adapter files, the
   adapter's name and the script's share of the timeout (``RemoteJob``), and nothing else: no key and
   nothing from ``.env`` reaches the container. There Glimmer comes off the weights volume at the
   pinned revision, and the adapter, checked as an upload is, lands on the adapters volume
   (``modal_job.train_on_volume``);
4. record the call's id under the adapter's name (``CallRecords``), then wait for what is left of the
   hard timeout less a cancel's margin (``modal_job.CANCEL_S``), so the hold covers the cancel too;
5. in a ``finally``, on success, failure, timeout or interrupt, cancel a call that may still run,
   then settle what it ran for, never above the hold: an overage would stop ``modal_usd`` for
   everything, serving included, until it is cleared. A cancel that fails settles the whole hold and
   raises, so a call that may still run is never silent; its record stays for the next sweep.

A spawn that fails may have started a call with no id to cancel by, so it settles the whole hold;
one Modal refused outright (``SpawnRefused``: the app isn't deployed, the login is wrong) settles
only the time spent. A run killed hard runs no ``finally``: its record stays, and the night
command's sweep cancels the call before anything else (``cancel_leftover_calls``). A delete cancels
the person's running call through the same records (``stop_person_calls``, RF2). The records hold an
adapter name (a hash) and a Modal call id, nothing else, under ``<data dir>/sleep/modal_calls/``;
the folder exists once this host has trained on Modal, which tells a delete to clean the adapters
volume too. MIND never trains here (Tasks 17 and 17b stay on Colab).

A thought-channel recipe sends ``prev_projector`` (when given) up beside a warm start's files, and
its trained projector comes back to ``<workdir>/channel/projector.npz`` (``projector_path``).

The adapter comes back as its two files and the receipt, written where the Colab runner writes
them (``<workdir>/out``), so ``train_lora`` checks and caches it the same way; ``landed`` says it is
on the volume already. ``modal`` is imported only when a real call is made (``SdkCalls``).

The call also sends back the digest of the training files its image carries (``modal_job.DIGEST_KEY``),
and one that isn't the checkout's fails the night: the app was deployed before a training file
changed, so redeploy it. An app deployed before the digest existed sends none and is not checked.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.night.holds import hold
from anjo.sleep.serving import config
from anjo.sleep.train import channel_train, colab_train, modal_job
from anjo.sleep.train.lora import DATASET_FILE, JOB, VOLUME_URI
from anjo.sleep.train.runners.colab import RealPersonRefused

# Modal's list price for one A100-80GB, checked 2026-10-27 (about $2.50 an hour, billed by the second).
# The hold is the job's hard timeout at this rate; CPU and memory are billed beside it (see dev_host.md).
A100_USD_PER_HOUR: Final = 2.50
POOL: Final = "modal_usd"
HOLD_KIND: Final = "modal_train"
CALLS_DIR: Final = "modal_calls"  # under <data dir>/sleep/
CHANNEL_DIR: Final = "channel"  # under the workdir: a thought-channel run's projector
_CALL_ID: Final = re.compile(r"fc-[A-Za-z0-9]{1,64}")


class ModalTrainError(RuntimeError):
    """The call failed, or didn't return what training promises."""


class SpawnRefused(ModalTrainError):
    """Modal refused the call before it started (the app isn't deployed, or the login is wrong)."""


@dataclass(frozen=True)
class RemoteJob:
    """One call's arguments (``modal_job.JOB_FIELDS``): everything that goes up, and nothing else."""

    dataset: str
    config_json: str
    prev: Mapping[str, bytes] = field(repr=False)
    name: str
    exec_s: float

    def args(self) -> tuple[object, ...]:
        return tuple(getattr(self, name) if name != "prev" else dict(self.prev) for name in modal_job.JOB_FIELDS)


class ModalCalls(Protocol):
    def spawn(self, job: RemoteJob) -> str:
        """Start the call; returns its id. ``SpawnRefused`` if Modal refused it before it started."""
        ...

    def wait(self, call_id: str, timeout_s: float) -> Mapping[str, object]:
        """The call's result; ``TimeoutError`` past ``timeout_s``."""
        ...

    def cancel(self, call_id: str) -> None:
        """Cancel the call and stop its container."""
        ...


class SdkCalls:
    """``ModalCalls`` through the ``modal`` package, on the deployed training app."""

    def __init__(self, app_name: str = modal_job.APP_NAME, function: str = modal_job.FUNCTION) -> None:
        self._app = app_name
        self._function = function

    def spawn(self, job: RemoteJob) -> str:
        modal: Any = importlib.import_module("modal")
        refused = (modal.exception.NotFoundError, modal.exception.InvalidError, modal.exception.AuthError)
        try:
            call = modal.Function.from_name(self._app, self._function).spawn(*job.args())
        except refused as error:
            raise SpawnRefused(f"Modal refused the call ({type(error).__name__}); is {self._app} deployed?") from error
        return str(call.object_id)

    def wait(self, call_id: str, timeout_s: float) -> Mapping[str, object]:
        modal: Any = importlib.import_module("modal")
        try:
            found: Mapping[str, object] = modal.FunctionCall.from_id(call_id).get(timeout=timeout_s)
        except modal.exception.TimeoutError as error:
            raise TimeoutError("the training call ran past the job's hard timeout") from error
        return found

    def cancel(self, call_id: str) -> None:
        modal: Any = importlib.import_module("modal")
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)


class CallRecords:
    """The calls that may be running: one small file per adapter name holding its Modal call id."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, name: str) -> Path:
        return self.root / f"{config.check_adapter_name(name)}.json"

    def used(self) -> bool:
        """Whether this host has ever started a Modal training call."""
        return self.root.is_dir()

    def add(self, name: str, call_id: str) -> None:
        if not isinstance(call_id, str) or not _CALL_ID.fullmatch(call_id):
            raise ValueError("a Modal call id is fc-<letters and digits>")
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        pending.write_text(json.dumps({"call": call_id}), encoding="utf-8")
        os.replace(pending, path)

    def remove(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def items(self) -> list[tuple[str, str]]:
        """Every recorded (adapter name, call id), in name order."""
        if not self.root.is_dir():
            return []
        found = [(path.stem, json.loads(path.read_text(encoding="utf-8"))["call"])
                 for path in sorted(self.root.glob("p-*.json"))]
        return [(name, str(call)) for name, call in found]


def hold_base(night: str, person_id: str, kind: str = HOLD_KIND) -> str:
    """The training's hold key before its attempt number: one night, one person. ``kind`` is the
    night's own (``modal_train``, swept by the night command) unless a caller with its own sweep names
    another (the thought channel's pretraining)."""
    return f"{kind}:{night}:{person_id}"


def _cancel_each(records: CallRecords, calls: ModalCalls, found: Iterable[tuple[str, str]]) -> int:
    cancelled = 0
    for name, call_id in found:
        try:
            calls.cancel(call_id)
        except Exception as error:  # any failure of Modal's client: the call may still run, so say so
            raise ModalTrainError(f"cancelling call {call_id} failed; cancel it by hand before anything else") from error
        records.remove(name)
        cancelled += 1
    return cancelled


def cancel_leftover_calls(records: CallRecords, calls: ModalCalls) -> tuple[int, int]:
    """The night command's sweep: cancel every call a killed run left. Returns how many were cancelled,
    and 0 calls of anyone else's (every recorded call is the night's). A cancel that fails raises."""
    return _cancel_each(records, calls, records.items()), 0


def stop_person_calls(
    records: CallRecords, person_id: str, adapters: Iterable[str], *, calls: ModalCalls | None = None,
) -> int:
    """Cancel the running calls training ``person_id`` (a delete, Task 9): the ones recorded under their
    adapters' names. Nobody else's is touched. Modal is reached only if one is recorded."""
    wanted = set(adapters)
    mine = [(name, call_id) for name, call_id in records.items() if name in wanted]
    if not mine:
        return 0
    return _cancel_each(records, calls if calls is not None else SdkCalls(), mine)


@dataclass
class _Call:
    state: Literal["none", "maybe", "running", "done"] = "none"
    call_id: str | None = None


class ModalRunner:
    """``GpuRunner`` for the ``train_lora`` job on Modal's A100-80GB."""

    def __init__(
        self, *, calls: ModalCalls, ledger: Ledger, records: CallRecords, is_synthetic: Callable[[str], bool],
        allow_real: Collection[str] = (), clock: Callable[[], float] = time.monotonic,
        usd_per_hour: float = A100_USD_PER_HOUR, hold_kind: str = HOLD_KIND,
    ) -> None:
        self.allow_real = frozenset(allow_real)
        self._hold_kind = hold_kind
        self._calls = calls
        self._ledger = ledger
        self._records = records
        self._is_synthetic = is_synthetic
        self._clock = clock
        self._rate = usd_per_hour

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        if job != JOB:
            raise ModalTrainError(f"the Modal runner runs only {JOB}")
        person = str(inputs["person_id"])
        if not self._is_synthetic(person) and person not in self.allow_real:
            raise RealPersonRefused("Modal trains synthetic people and the real people named (Kevin's own), nobody else")
        name = config.check_adapter_name(str(inputs["adapter_name"]))
        try:
            exec_s = modal_job.exec_seconds(timeout_s)
        except modal_job.ModalJobError as error:
            raise ModalTrainError(str(error)) from None
        remote = _remote_job(inputs, name, exec_s)
        base = hold_base(str(inputs["night"]), person, self._hold_kind)
        reservation = hold(self._ledger, base, (POOL,), Bound(self._rate, timeout_s))
        started = self._clock()
        call = _Call()
        try:
            result = self._train(call, remote, started, timeout_s)
        finally:
            elapsed = self._finish(call, name, reservation, started)
        return self._outputs(result, inputs, name, elapsed)

    def _train(self, call: _Call, remote: RemoteJob, started: float, timeout_s: float) -> Mapping[str, object]:
        call.state = "maybe"  # a spawn that fails may still have started the call
        try:
            call.call_id = self._calls.spawn(remote)
        except SpawnRefused:
            call.state = "none"
            raise
        call.state = "running"
        self._records.add(remote.name, call.call_id)
        left = timeout_s - (self._clock() - started) - modal_job.CANCEL_S
        result = self._calls.wait(call.call_id, max(0.0, left))
        call.state = "done"
        return result

    def _finish(self, call: _Call, name: str, reservation: Reservation, started: float) -> float:
        """Cancel a call that may still run, then settle what it ran for.

        - A call that finished, or never started: its time is settled, at most the hold.
        - A spawn that failed without an id: nothing to cancel by, so the whole hold is settled.
        - A call that may still run is cancelled; a cancel that fails settles the whole hold and raises.
        """
        if call.state == "running" and call.call_id is not None:
            try:
                self._calls.cancel(call.call_id)
            except Exception as error:  # any failure of Modal's client: the call may still run, so say so
                reservation.settle(reservation.bound.amount)
                raise ModalTrainError(f"cancelling call {call.call_id} failed; cancel it by hand before anything "
                                      "else") from error
        if call.call_id is not None:
            self._records.remove(name)
        elapsed = max(0.0, self._clock() - started)
        if call.state == "maybe":
            reservation.settle(reservation.bound.amount)
        else:
            reservation.settle(min(elapsed / 3600.0 * self._rate, reservation.bound.amount))
        return elapsed

    def _outputs(
        self, result: Mapping[str, object], inputs: Mapping[str, object], name: str, elapsed: float,
    ) -> dict[str, object]:
        """The adapter's files and the receipt in ``<workdir>/out``, as the Colab runner leaves them."""
        files, receipt = result.get("files"), result.get("receipt")
        if result.get("adapter") != name or not isinstance(files, Mapping) or not isinstance(receipt, dict):
            raise ModalTrainError("the training call returned no adapter record")
        deployed = result.get(modal_job.DIGEST_KEY)
        if deployed is not None and deployed != modal_job.files_digest():
            raise ModalTrainError(f"{modal_job.APP_NAME} was deployed with other training files than this checkout's; "
                                  "redeploy it (deploy/dev_host.md §1.7)")
        out = Path(str(inputs["workdir"])) / "out"
        out.mkdir(parents=True, exist_ok=True)
        for file in config.ADAPTER_FILES:
            data = files.get(file)
            if not isinstance(data, bytes):
                raise ModalTrainError(f"the training call returned no {file}")
            (out / file).write_bytes(data)
        (out / colab_train.RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
        projector = result.get("projector")
        channel: dict[str, object] = {}
        if projector is not None:  # a thought-channel run: its projector, beside the adapter, never inside it
            if not isinstance(projector, bytes):
                raise ModalTrainError("the training call returned a projector that isn't a file")
            path = Path(str(inputs["workdir"])) / CHANNEL_DIR / channel_train.PROJECTOR_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(projector)
            channel = {"projector_path": str(path)}
        return {
            **receipt, "adapter_dir": str(out), "gpu_minutes": elapsed / 60.0, "compute_units": 0.0,
            "usd": elapsed / 3600.0 * self._rate, "landed": f"{VOLUME_URI}/{name}", **channel,
        }


def _remote_job(inputs: Mapping[str, object], name: str, exec_s: float) -> RemoteJob:
    """The call's arguments: the night's dataset file (and no other), the recipe, a warm start's files."""
    workdir = Path(str(inputs["workdir"]))
    dataset = Path(str(inputs["dataset_path"]))
    if dataset.name != DATASET_FILE or dataset.parent != workdir:
        raise ModalTrainError("only the night's dataset file goes up")
    prev_dir = inputs.get("prev_adapter_dir")
    prev = {} if prev_dir is None else {
        file: (Path(str(prev_dir)) / file).read_bytes() for file in config.ADAPTER_FILES
    }
    projector = inputs.get("prev_projector")
    if projector is not None:  # the thought channel's warm start: the person's projector, or the shared one
        prev[channel_train.PROJECTOR_FILE] = Path(str(projector)).read_bytes()
    return RemoteJob(dataset=dataset.read_text(encoding="utf-8"), config_json=json.dumps(inputs["config"], sort_keys=True),
                     prev=prev, name=name, exec_s=exec_s)
