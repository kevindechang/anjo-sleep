"""What the example reaches outside through: one value, so a run is the same code live or fake.

``live.build_services`` wires the real ones (Baseten, the Meta API, the engine's training runner and
the Modal vLLM server). ``fakes.fake_services`` wires offline stand-ins. Nothing else differs.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.ports import Chooser, GpuRunner
from anjo.sleep.probes.clients import ModelClient, TokenLogprobs
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.train.lora import AdapterPublisher


class ExampleError(RuntimeError):
    """The example can't go on. The message says what to do about it, and never holds a key."""


class Glimmer(ModelClient, TokenLogprobs, Protocol):
    """Glimmer's vLLM server: sampled replies and forced-choice log-probabilities."""

    @property
    def rendering(self) -> str:
        """The rendering the server serves under; an adapter must be trained under it."""

    def on(self, day: date) -> Glimmer:
        """This client with its prompts dated ``day``, the date the rows were trained at."""

    def close(self) -> None: ...


class Serving(Protocol):
    """Waking the server, putting a trained adapter where it loads from, and restarting it."""

    def wake(self) -> WakeResult: ...

    def sleep_now(self) -> None:
        """Stop the server, so its next start sees every adapter on the volume."""

    def upload_adapter(self, folder: Path, name: str) -> None: ...

    def adapter_config(self, name: str) -> Mapping[str, Any] | None:
        """The ``adapter_config.json`` of an adapter on the server's volume, or None if there is none."""


@dataclass
class Activity:
    """When the example last called the server (on the services' clock). Modal scales an idle server
    to zero, and only a cold start loads a new adapter, so the example waits out this idle time."""

    last: float | None = None

    def idle_s(self, now: float) -> float | None:
        return None if self.last is None else now - self.last


class TrackedGlimmer:
    """A Glimmer client that records every call it makes in ``activity``."""

    def __init__(self, inner: Glimmer, activity: Activity, clock: Callable[[], float]) -> None:
        self._inner, self._activity, self._clock = inner, activity, clock

    @property
    def rendering(self) -> str:
        return self._inner.rendering

    def on(self, day: date) -> TrackedGlimmer:
        return TrackedGlimmer(self._inner.on(day), self._activity, self._clock)

    def close(self) -> None:
        self._inner.close()

    def complete(self, adapter: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        try:
            return self._inner.complete(adapter, messages, **kwargs)
        finally:
            self._activity.last = self._clock()

    def continuation_logprobs(self, adapter: Any, messages: list[dict[str, str]], continuation: str) -> Sequence[float]:
        try:
            return self._inner.continuation_logprobs(adapter, messages, continuation)
        finally:
            self._activity.last = self._clock()


@dataclass(frozen=True)
class Services:
    spark: ChatFn  # Muse Spark Open on Baseten: extraction, briefs, probes, practice scenes
    judge: ChatFn  # Muse Spark on the Meta API: the pairwise judge
    glimmer: Glimmer
    serving: Serving
    gpu: GpuRunner  # the training job: Modal's by default, or Colab's
    revision: str  # the Glimmer commit the training job pins
    # How a trained adapter reaches the server; None uploads it (``serving.upload_adapter``). Modal's
    # training job lands it on the server's volume itself, so nothing is uploaded after it.
    publish: AdapterPublisher | None = None
    chooser: Chooser = StateChooser()
    fake: bool = False
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
