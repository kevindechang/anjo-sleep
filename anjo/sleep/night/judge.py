"""One judge's "Sleep now" night (2026-11-02): ``night/demo.py``'s profile, on a judge's own data.

The app's dispatcher (``app/judge_night.py``) runs this in a child process, one per night, and reads
its progress through the observer. What makes it a judge's night rather than Kevin's:

- **The recipe.** Plain SFT (``LoraCfg()``), whatever ``ANJO_SLEEP_OBJECTIVE`` says for the nights,
  at the profile's fixed step plan (``StepPlan``: 8 optimizer steps), with no dream terms.
- **The trainer.** ``runners/warm.py``'s ``WarmFirstRunner``: the warm app (``anjo-sleep-judge-train``)
  first, the cold app (``anjo-sleep-train``) when it can't, both through a ``ModalRunner`` that holds,
  records, cancels and settles as every night's does. The serving apps are never touched here: the
  night reaches Glimmer through the app's serving client as every night does, and the app's wiring
  leaves the serving GPU alone at the end (``app/night.py``: no ``gpu_stop``, no ``sleep`` on a
  profile).
- **Waking the serving GPU from the start.** Modal's snapshot restore of the Glimmer server took 43 s
  once and 279 s the next (a cold boot is about 351 s), so no wake is assumed to be quick: the night
  starts waking it on its own thread before its first step (``EarlyWake``), and phase 1's CPU-side
  steps run meanwhile. Every step that needs Glimmer still waits for its own wake, as on every night.
- **Keeping the trainer warm.** The ``Warmer`` starts when the night starts (``WARM_AT_ENV``: ``start``
  by default; ``extract``'s end or ``practice``'s on a host that wants a shorter warm span), so
  Glimmer's load on the trainer overlaps phase 1, and stops when ``train`` ends, when the night does,
  or at its deadline (``WARM_HOLD_S`` less the scale-down window), whichever is first. Its time is held
  on ``modal_usd`` (``WARM_HOLD_S`` at Modal's A100 rate) and settled at what the pings kept the
  container up, plus its scale-down window. A cost past the hold (the last ping's own latency) is
  never cut to fit: a second hold takes the rest, or, when the pool has no room, the whole cost is
  settled and logged, and the pool waits for the coordinator to clear it. A cold-only host
  (``ANJO_SLEEP_JUDGE_TRAINER=cold``) warms nothing.
- **The lock.** ``sweep.shared_night``: judges' nights run side by side; a night command waits for
  them, and they for it.

It imports nothing of Anjo's app: the wiring is handed in. Its process entry,
``python -m anjo.sleep.night.judge <module>:<function> <night id>``, calls the app's night runner by
name with ``execute_record`` (``app/judge_child.run_night``), as the night command calls its wiring.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep.budget import Bound, Ledger, Reservation
from anjo.sleep.night.demo import DemoProfile
from anjo.sleep.night.deps import NightOptions, StepObserver, Wiring
from anjo.sleep.night.holds import hold
from anjo.sleep.night.run import Window, run_night_batch_with
from anjo.sleep.night.sweep import shared_night
from anjo.sleep.serving.client import ServingError, WakeResult
from anjo.sleep.serving.config import REVISION_ENV
from anjo.sleep.train import warm_job
from anjo.sleep.train.lora import AdapterPublisher, LoraCfg, StepPlan
from anjo.sleep.train.runners import choice
from anjo.sleep.train.runners.colab import ColabCli
from anjo.sleep.train.runners.modal import A100_USD_PER_HOUR, CALLS_DIR, POOL, CallRecords, ModalCalls, ModalRunner, SdkCalls
from anjo.sleep.train.runners.warm import Warmer, WarmCalls, WarmFirstRunner, ping_seconds, trainer_mode
from anjo.sleep.train.trainer import LoraTrainer
from anjo.sleep.types import NightResult

WARM_AT_ENV: Final = "ANJO_SLEEP_JUDGE_WARM_AT"
WARM_AT: Final = ("start", "extract", "practice")
WARM_HOLD_S: Final = 30 * 60  # the longest the keep-warm runs: phase 1 and practice, with room
WARM_KIND: Final = "modal_train"  # a night kind: a killed run's hold is settled by the next sweep
STOP_AT: Final = "train"
OVER_EPSILON: Final = 1e-9  # dollars: below this a settlement is within its hold
logger = logging.getLogger(__name__)


class JudgeNightError(ValueError):
    """A judge's night was asked for with settings it can't run."""


@dataclass(frozen=True)
class JudgeOutcome:
    """The night's result, and what keeping the trainer warm cost beside it (dollars, at most its hold)."""

    result: NightResult
    warm_usd: float


@dataclass(frozen=True)
class JudgeNight:
    """Which night: the judge, the night's date, the window of chat it reads and its clock."""

    user: str
    night: date
    window: Window
    as_of: datetime


def warm_at(environ: Mapping[str, str]) -> str:
    chosen = environ.get(WARM_AT_ENV, "").strip() or WARM_AT[0]
    if chosen not in WARM_AT:
        raise JudgeNightError(f"{WARM_AT_ENV} is {' or '.join(WARM_AT)}")
    return chosen


def judge_options(night: date, profile: DemoProfile) -> NightOptions:
    """A judge is a real person without a persona; tonight is deep (it trains), with MIND's inquiry."""
    return NightOptions(night=night, personas={}, allow_real=True, practice_scenes=profile.practice_scenes,
                        mind_inquiry=True, force_deep=True, demo=profile)


def judge_steps(profile: DemoProfile) -> StepPlan:
    return StepPlan(accumulation=1, epochs=1, steps_per_epoch=profile.train_steps)


def warm_deadline(scaledown_s: float) -> float:
    """The latest a keep-warm ping may start after the first: the hold's span less the scale-down
    window the last ping keeps the container up for."""
    return WARM_HOLD_S - scaledown_s


class WarmWindow:
    """The keep-warm's span and its hold: ``begin`` at the warm-up step's end, ``end`` at training's end
    or the night's. Only the first ``begin`` and the first ``end`` count."""

    def __init__(self, warmer: Warmer | None, ledger: Ledger, base: str, *, scaledown_s: float) -> None:
        self._warmer, self._ledger, self._base, self._scaledown_s = warmer, ledger, base, scaledown_s
        self._held: Reservation | None = None
        self.ended = False
        self.spent = 0.0

    def begin(self) -> None:
        if self._warmer is None or self._held is not None or self.ended:
            return
        self._held = hold(self._ledger, self._base, (POOL,), Bound(A100_USD_PER_HOUR, WARM_HOLD_S))
        self._warmer.start()

    def end(self) -> None:
        if self.ended:
            return
        self.ended = True
        if self._warmer is None or self._held is None:
            return
        self._warmer.stop()
        span = self._warmer.span_s
        if span is None:  # stopped before its first ping: the container never started for it
            self._held.release()
            return
        self.spent = (span + self._scaledown_s) / 3600.0 * A100_USD_PER_HOUR
        self._settle(self._held, self.spent)

    def _settle(self, held: Reservation, spent: float) -> None:
        """Settle what the keep-warm cost, all of it: past the hold, a second hold takes the rest."""
        over = spent - held.bound.amount
        if over <= OVER_EPSILON:
            held.settle(spent)
            return
        extra = self._ledger.reserve(f"{held.job}:over", (POOL,),
                                     Bound(A100_USD_PER_HOUR, over / A100_USD_PER_HOUR * 3600.0))
        if extra is None:
            logger.warning("the keep-warm cost $%.2f past its hold and %s has no room for it: the whole "
                           "cost is settled, and the pool waits to be cleared", over, POOL)
            held.settle(spent)
            return
        logger.warning("the keep-warm cost $%.2f past its hold; a second hold took it", over)
        held.settle(held.bound.amount)
        extra.settle(over)


def watching(observer: StepObserver | None, window: WarmWindow, start_at: str) -> StepObserver:
    """The night's observer, with the keep-warm started and stopped on the steps' ends."""
    def observe(user_id: str, step: str, state: str, receipt: Mapping[str, object]) -> None:
        if step == STOP_AT and state != "running":
            window.end()  # training's end, however it ended: nothing after it needs the trainer
        elif start_at == "start" or (step == start_at and state == "ok"):
            window.begin()  # once, and never after the end
        if observer is not None:
            observer(user_id, step, state, receipt)

    return observe


class EarlyWake:
    """The serving GPU's wake, started on its own thread as the night starts. Its result (or the type
    of its failure) is kept for the log; a failure only means the night's own wakes do the work."""

    def __init__(self, wake: Callable[[], WakeResult] | None) -> None:
        self._wake = wake
        self.result: WakeResult | None = None
        self.error: str | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self._wake is None or self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name="judge-serving-wake", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self._wake is not None
        try:
            self.result = self._wake()
        except (ServingError, OSError, TimeoutError) as error:
            self.error = type(error).__name__
            logger.warning("the early wake failed (%s); the night's own wakes will try again", self.error)


def _cold_calls() -> ModalCalls:
    return SdkCalls()


def _upload(folder: Path, name: str) -> None:
    """The serving backend's upload, for an adapter the trainer didn't land where it is served."""
    from anjo.sleep.serving import backend

    backend.from_env().upload_adapter(folder, name)


def _publisher() -> AdapterPublisher:
    from anjo.sleep.serving import backend

    return choice.publisher("modal", _upload, serving=backend.backend_name())


def execute_judge(
    *, wiring: Wiring, job: JudgeNight, profile: DemoProfile, data_dir: Path, ledger: Ledger,
    observer: StepObserver | None = None, environ: Mapping[str, str] | None = None,
    warm_calls: Callable[[], WarmCalls] = WarmCalls, cold_calls: Callable[[], ModalCalls] = _cold_calls,
    publisher: Callable[[], AdapterPublisher] = _publisher, colab_cli: Callable[[], ColabCli] = ColabCli,
) -> JudgeOutcome:
    """One judge's night under the shared lock, on the warm-first trainer, told step by step."""
    env = os.environ if environ is None else environ
    mode, ping_s, start_at = trainer_mode(env), ping_seconds(env), warm_at(env)
    scaledown_s = warm_job.scaledown_seconds(env)
    calls_dir = data_dir / "sleep" / CALLS_DIR
    cold = choice.build("modal", ledger=ledger, calls_dir=calls_dir, is_synthetic=lambda _: False,
                        allow_real=frozenset({job.user}), colab_cli=colab_cli, modal_calls=cold_calls)
    warm = warm_calls()
    warm_runner = ModalRunner(calls=warm, ledger=ledger, records=CallRecords(calls_dir), is_synthetic=lambda _: False,
                              allow_real=frozenset({job.user}), digest=warm_job.files_digest, app_name=warm_job.APP_NAME)
    trainer = LoraTrainer(WarmFirstRunner(warm=warm_runner, cold=cold.runner, mode=mode),
                          root=data_dir / "sleep" / "training", publish=publisher(),
                          revision=env.get(REVISION_ENV) or None, cfg=LoraCfg(), steps=judge_steps(profile))
    warmer = Warmer(warm.warm, ping_s, deadline_s=warm_deadline(scaledown_s)) if mode == "warm" else None
    window = WarmWindow(warmer, ledger,
                        f"{WARM_KIND}:{job.night.isoformat()}:{job.user}:warm", scaledown_s=scaledown_s)
    with shared_night(ledger, cold.sweep) as swept:
        try:
            deps = replace(wiring(trainer, judge_options(job.night, profile)),
                           observer=watching(observer, window, start_at))
            EarlyWake(deps.wake).start()  # the serving GPU boots while phase 1 thinks
            done = run_night_batch_with(deps, {job.user: job.night}, windows={job.user: job.window},
                                        sweep=swept, as_of={job.user: job.as_of})
        finally:
            window.end()  # under the lock: no other night's sweep can settle the keep-warm's hold first
    return JudgeOutcome(done[job.user], window.spent)


def execute_record(*, job: Mapping[str, Any], **kwargs: Any) -> JudgeOutcome:
    """``execute_judge`` for the app's night runner, which names the night by its record's fields
    (``user``, ``night``, ``window``, ``as_of``) and never imports this module."""
    return execute_judge(job=JudgeNight(**job), **kwargs)


def main(argv: list[str]) -> int:
    """The night process: ``<module>:<function> <night id>``; the function runs the night with ``execute_record``."""
    if len(argv) != 2:
        raise SystemExit("usage: python -m anjo.sleep.night.judge <module>:<function> <night id>")
    module, _, name = argv[0].partition(":")
    if not module or not name:
        raise SystemExit(f"{argv[0]} is not <module>:<function>")
    run: Callable[[str, Callable[..., JudgeOutcome]], int] = getattr(importlib.import_module(module), name)
    return run(argv[1], execute_record)


if __name__ == "__main__":  # pragma: no cover - the judge's night process
    raise SystemExit(main(sys.argv[1:]))
