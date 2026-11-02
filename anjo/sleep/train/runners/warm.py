"""The judges' trainer on the host side (2026-11-02): the warm app first, the cold app when it can't.

``train/warm_job.py`` has the warm app's why. Here:

- ``WarmCalls`` is ``ModalCalls`` on the deployed ``WarmTrainer`` class (``anjo-sleep-judge-train``):
  ``spawn`` calls its ``train`` method; ``wait`` and ``cancel`` are the cold calls' own, since a class
  method's call is a function call to Modal. ``warm`` spawns the keep-warm method and returns its call
  id at once; ``cancel_input`` drops a keep-warm call's input, never its container (another judge's
  training may be running in the one warm container).
- ``Warmer`` keeps the warm container up from the start of a training night until its training
  ends: one ``warm`` call now, then one every ``ping_s`` (``ANJO_SLEEP_JUDGE_WARM_PING_S``, 45 s by
  default, inside the container's 90 s scale-down window), on a daemon thread, until ``stop`` or its
  deadline, whichever comes first. The deadline is what its hold paid for, less the scale-down
  window: no ping starts after it, whatever the night is doing, so a slow night trains on a container
  that may have gone cold rather than paying for one nobody holds. A ping that fails is logged and
  the next one tries again: a cold container costs time, never the night. When the pings stop (``stop``,
  or the deadline), the warm thread cancels every keep-warm input it spawned, a ping still in flight
  then included once it lands, so none runs after. ``span_s`` counts from the first ping to the moment
  the pings stopped, so a ping in flight then is counted, however long its call takes to land.
- ``WarmFirstRunner`` is the ``GpuRunner`` a judge's night trains on. A recipe the warm app runs
  (``warm_job.refusal``: plain SFT, no pairs' terms, no thought channel) goes to the warm runner, a
  ``ModalRunner`` with this app's name and digest, so it holds, records, cancels and settles exactly
  as the cold one does. Any other recipe, ``ANJO_SLEEP_JUDGE_TRAINER=cold``, or a warm app Modal
  refuses outright (``SpawnRefused``: not deployed) trains on the cold runner instead, and the receipt
  says so (``runner``, ``fallback_reason``). A warm call that started and failed fails the training,
  as a cold one does: it may have spent, and a second call would spend again.
"""
from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Final

from anjo.sleep.ports import GpuRunner
from anjo.sleep.train import warm_job
from anjo.sleep.train.runners.modal import ModalRunner, RemoteJob, SdkCalls, SpawnRefused

TRAINER_ENV: Final = "ANJO_SLEEP_JUDGE_TRAINER"
TRAINERS: Final = ("warm", "cold")
PING_ENV: Final = "ANJO_SLEEP_JUDGE_WARM_PING_S"
PING_RANGE: Final = (10.0, 300.0)
WARM_RUNNER: Final = "modal_warm"
COLD_RUNNER: Final = "modal"
logger = logging.getLogger(__name__)


class WarmChoiceError(ValueError):
    """``ANJO_SLEEP_JUDGE_TRAINER`` or ``ANJO_SLEEP_JUDGE_WARM_PING_S`` holds a value it can't."""


def trainer_mode(environ: Mapping[str, str]) -> str:
    """``warm`` (the default: the warm app, the cold one when it can't) or ``cold``."""
    chosen = environ.get(TRAINER_ENV, "").strip() or "warm"
    if chosen not in TRAINERS:
        raise WarmChoiceError(f"{TRAINER_ENV} is {' or '.join(TRAINERS)}")
    return chosen


def ping_seconds(environ: Mapping[str, str]) -> float:
    """Seconds between keep-warm calls: ``PING_ENV``, ``warm_job.PING_S`` when unset."""
    raw = environ.get(PING_ENV, "").strip()
    if not raw:
        return warm_job.PING_S
    try:
        value = float(raw)
    except ValueError:
        raise WarmChoiceError(f"{PING_ENV} is a number of seconds") from None
    low, high = PING_RANGE
    if not low <= value <= high:
        raise WarmChoiceError(f"{PING_ENV} is {low:g} to {high:g} seconds")
    return value


class WarmCalls(SdkCalls):
    """``ModalCalls`` on the deployed warm class, and its keep-warm call."""

    def __init__(self, app_name: str = warm_job.APP_NAME, cls: str = warm_job.CLASS) -> None:
        super().__init__(app_name=app_name, function=f"{cls}.{warm_job.TRAIN_METHOD}")
        self._cls = cls

    def _trainer(self, modal: Any) -> Any:
        return modal.Cls.from_name(self._app, self._cls)()

    def spawn(self, job: RemoteJob) -> str:
        modal: Any = importlib.import_module("modal")
        refused = (modal.exception.NotFoundError, modal.exception.InvalidError, modal.exception.AuthError)
        try:
            call = self._trainer(modal).train.spawn(job.dataset, job.config_json, dict(job.prev), job.name)
        except refused as error:
            raise SpawnRefused(f"Modal refused the call ({type(error).__name__}); is {self._app} deployed?") from error
        return str(call.object_id)

    def warm(self) -> str | None:
        """Spawn one keep-warm call and return its id at once; None (logged) if Modal didn't take it."""
        modal: Any = importlib.import_module("modal")
        try:
            call = self._trainer(modal).warm.spawn()
        except (modal.exception.Error, OSError, TimeoutError) as error:
            logger.warning("the warm trainer's keep-warm call failed (%s)", type(error).__name__)
            return None
        return str(call.object_id)

    def cancel_input(self, call_id: str) -> bool:
        """Drop one keep-warm call's input, leaving its container alone; False (logged) if Modal didn't."""
        modal: Any = importlib.import_module("modal")
        try:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=False)
        except (modal.exception.Error, OSError, TimeoutError) as error:
            logger.warning("a keep-warm call couldn't be cancelled (%s)", type(error).__name__)
            return False
        return True


class Warmer:
    """Keep-warm calls on a daemon thread: one at ``start``, then one every ``ping_s`` until ``stop``,
    and none that would start ``deadline_s`` or more after the first; then every input it spawned is
    cancelled. ``warm`` returns a call's id (None when Modal didn't take it) and ``cancel`` never raises
    (``WarmCalls.warm`` and ``cancel_input``)."""

    def __init__(self, warm: Callable[[], str | None], ping_s: float = warm_job.PING_S, *, deadline_s: float,
                 cancel: Callable[[str], object] | None = None, clock: Callable[[], float] = time.monotonic,
                 join_s: float = 5.0) -> None:
        if deadline_s <= 0:
            raise WarmChoiceError("the keep-warm's deadline is a positive number of seconds")
        self._warm, self._ping_s, self._deadline_s, self._clock = warm, ping_s, deadline_s, clock
        self._cancel, self._join_s = cancel, join_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.pings = 0
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.first_ping_at: float | None = None
        self.stopped_at: float | None = None  # when ``stop`` was asked or the deadline ended the pings
        self.expired = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="judge-warm-trainer", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                now = self._clock()
                if self.first_ping_at is None:
                    self.first_ping_at = now
                elif now - self.first_ping_at >= self._deadline_s:
                    self.expired = True
                    self.stopped_at = self.stopped_at if self.stopped_at is not None else now
                    logger.warning("the warm trainer's keep-warm reached its %.0f s deadline and stops",
                                   self._deadline_s)
                    return
                self.pings += 1
                call = self._warm()  # a ping that lands after ``stop`` is cancelled below with the rest
                if call is not None:
                    self.calls.append(call)
                self._stop.wait(self._ping_s)
        finally:
            self._cancel_all()

    def _cancel_all(self) -> None:
        if self._cancel is None:
            return
        for call in self.calls:
            self._cancel(call)
            self.cancelled.append(call)

    def stop(self) -> None:
        if self.stopped_at is None and self.first_ping_at is not None:
            self.stopped_at = self._clock()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._join_s)  # a ping still in flight is counted to now, cancelled on landing

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def span_s(self) -> float | None:
        """Seconds the pings kept the container up, before its scale-down window and any start still
        under way: from the first ping to the moment the pings stopped (now, while they haven't). None
        before any ping."""
        if self.first_ping_at is None:
            return None
        end = self.stopped_at if self.stopped_at is not None else self._clock()
        return max(0.0, end - self.first_ping_at)


class WarmFirstRunner:
    """``GpuRunner``: the warm runner for a recipe it takes, the cold runner otherwise."""

    def __init__(self, *, warm: ModalRunner, cold: GpuRunner, mode: str = "warm") -> None:
        if mode not in TRAINERS:
            raise WarmChoiceError(f"{TRAINER_ENV} is {' or '.join(TRAINERS)}")
        self._warm, self._cold, self._mode = warm, cold, mode

    def run(self, job: str, inputs: Mapping[str, object], *, timeout_s: float) -> Mapping[str, object]:
        recipe = inputs.get("config")
        why = "the judge host chose the cold trainer" if self._mode == "cold" else (
            warm_job.refusal(recipe) if isinstance(recipe, Mapping) else "the recipe isn't a mapping")
        if why is None:
            try:
                return {**self._warm.run(job, inputs, timeout_s=timeout_s), "runner": WARM_RUNNER}
            except SpawnRefused as error:
                why = f"the warm app refused the call: {error}"
        logger.info("a judge's night trains on the cold app: %s", why)
        return {**self._cold.run(job, inputs, timeout_s=timeout_s), "runner": COLD_RUNNER, "fallback_reason": why}
