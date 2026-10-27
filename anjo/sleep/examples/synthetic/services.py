"""What the example reaches outside through: one value, so a run is the same code live or fake.

``live.build_services`` wires the real ones (Baseten, the Meta API, a Modal training job and the
Modal vLLM server). ``fakes.fake_services`` wires offline stand-ins. Nothing else differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.ports import Chooser, GpuRunner
from anjo.sleep.probes.clients import ModelClient, TokenLogprobs
from anjo.sleep.serving.client import WakeResult


class ExampleError(RuntimeError):
    """The example can't go on. The message says what to do about it, and never holds a key."""


class Glimmer(ModelClient, TokenLogprobs, Protocol):
    """Glimmer's vLLM server: sampled replies and forced-choice log-probabilities."""

    def close(self) -> None: ...


class Serving(Protocol):
    """Waking the server, and putting a trained adapter where it loads from."""

    def wake(self) -> WakeResult: ...

    def upload_adapter(self, folder: Path, name: str) -> None: ...


@dataclass(frozen=True)
class Services:
    spark: ChatFn  # Muse Spark Open on Baseten: extraction, briefs, probes, practice scenes
    judge: ChatFn  # Muse Spark on the Meta API: the pairwise judge
    glimmer: Glimmer
    serving: Serving
    gpu: GpuRunner  # the training job, on Modal
    revision: str  # the Glimmer commit the training job pins
    chooser: Chooser = StateChooser()
    fake: bool = False
