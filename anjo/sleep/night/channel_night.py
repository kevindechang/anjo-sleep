"""The thought channel in the night (one system, ``brief_one_system``): practice reads the person's
MIND state as K soft tokens, and the night's one training job trains the projector with the LoRA on
the rows practice made under that state.

Two ports on ``NightDeps``, both ``None`` by default, which is exactly today's night:

- ``mind_state`` (``MindState``): the person's night as MIND has it (``MindReading``): the core state
  after the night's dream step, its id in MIND's store (``sha256(state.save)[:16]``), the core's
  predictions on it, and the projector the person's last kept unit names
  (``MindStore.before(night).projector``: the file the trainer kept beside that night's adapter), or
  ``None`` when the core is off for them. This module never loads the core or the store.
- ``channel`` (``ChannelSetup``): the arm (``state``, or ``zero``: the ablation, generated lives
  only), the shared pretraining's projector, the soft-token client and practice built over a client.

With both and a reading, practice (``practice``) replies through ``core_snapshot.Channel`` over
exactly that state (the snapshot's ``state`` is the reading's ``state_id``, checked), and every
transcript it makes is stamped, so the dataset's rows carry ``core``. The projector is the person's
own, or the shared one before their first channel night. The training step (``train``) hands a
dataset whose rows carry ``core`` to the trainer's ``train_with_channel``, warm from that same
projector; the receipt's ``projector`` is the file to keep as the night's unit's projector. Without
a reading the night is today's, practice and training alike.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

from anjo.sleep.mind.core.projector import PredictionsLike, Projector
from anjo.sleep.mind.core.state import CoreState
from anjo.sleep.night.core_snapshot import Channel, ChannelArm, SoftClient, snapshot
from anjo.sleep.practice.full import CappedChat
from anjo.sleep.practice.run import PracticeRun
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.types import CheckpointRef, Evidence

if TYPE_CHECKING:  # the types only: deps imports this module's ports
    from anjo.sleep.night.deps import NightDeps, PracticeRunner, TrainingData


class MindReading(NamedTuple):
    """One person's night as MIND has it, after the dream step."""

    state: CoreState  # the consolidated state (``p.dreamt.state``)
    state_id: str  # its id in MIND's store (``p.dreamt.state_id``)
    predictions: PredictionsLike  # the core's on it (``p.dreamt.view.predictions``)
    projector: str | None  # the last kept unit's projector file (``MindStore.before(night).projector``); None: the shared one


MindState = Callable[[str, date], MindReading | None]


@dataclass(frozen=True)
class ChannelSetup:
    """The night's channel: its arm, the shared projector (a person's first night warms from it), the
    client that sends soft tokens, and practice built over a ``ModelClient`` (the channel)."""

    arm: ChannelArm
    shared_projector: Path
    soft_client: SoftClient
    practice: Callable[[CappedChat, date, ModelClient], PracticeRunner]


@runtime_checkable
class ChannelTrainer(Protocol):
    """A ``Trainer`` that can train the projector with the LoRA in one job (``train.trainer.LoraTrainer``)."""

    def train_with_channel(
        self, user_id: str, night: date, prev: CheckpointRef | None, data: TrainingData, name: str, *,
        arm: ChannelArm, projector: Path,
    ) -> tuple[CheckpointRef, Mapping[str, object]]: ...


def check(deps: NightDeps, people: Iterable[str]) -> None:
    """Before any step: the channel's trainer, its shared projector, and the zero arm only on
    generated lives. Off, nothing is checked."""
    setup = deps.channel
    if setup is None:
        return
    if not isinstance(deps.trainer, ChannelTrainer):
        raise ValueError("the thought channel is on, but the trainer can't train the projector with the LoRA")
    if not setup.shared_projector.is_file():
        raise ValueError("the thought channel is on, but the shared projector isn't there")
    if setup.arm == "zero" and not all(deps.is_synthetic(uid) for uid in people):
        raise ValueError("the zero-tokens ablation runs on generated lives only; a real person runs the full system")


def _reading(deps: NightDeps, uid: str, night: date) -> MindReading | None:
    if deps.channel is None or deps.mind_state is None:
        return None
    return deps.mind_state(uid, night)


def projector_path(deps: NightDeps, reading: MindReading) -> Path:
    """The projector the person's night reads and trains from: their own, as their last unit kept it,
    or the shared pretraining's before their first channel night. A kept one that's gone fails."""
    assert deps.channel is not None
    return _projector_file(deps.channel, reading)


def _projector_file(setup: ChannelSetup, reading: MindReading) -> Path:
    if reading.projector is None:
        return setup.shared_projector
    own = Path(reading.projector)
    if not own.is_file():
        raise ValueError("the person's kept projector isn't there; the channel won't fall back to the shared one")
    return own


def channel_for(setup: ChannelSetup, reading: MindReading) -> Channel:
    """The channel over ``reading``'s state, through the projector the night reads (``projector_path``);
    the snapshot's state must be the one MIND kept under the reading's id."""
    projector = Projector.load(_projector_file(setup, reading).read_bytes())
    snap = snapshot(reading.state, reading.predictions, arm=setup.arm, k=projector.k)
    if snap.state_sha != reading.state_id:
        raise ValueError("the state the channel would read isn't the state MIND kept under that id")
    return Channel(snap, projector, setup.soft_client)


def _channel(deps: NightDeps, reading: MindReading) -> Channel:
    assert deps.channel is not None
    return channel_for(deps.channel, reading)


@dataclass(frozen=True)
class _Stamped:
    """Practice through the channel: its transcripts stamped with the state their replies were made under."""

    runner: PracticeRunner
    channel: Channel

    @property
    def max_spark_calls(self) -> int:
        return self.runner.max_spark_calls

    def run(
        self, user_id: str, night: date, evidence: Sequence[Evidence], hw: object | None, *, at: datetime,
    ) -> PracticeRun:
        run = self.runner.run(user_id, night, evidence, hw, at=at)
        return replace(run, transcripts=tuple(self.channel.stamp(run.transcripts)))


def practice(deps: NightDeps, uid: str, chat: CappedChat, night: date) -> tuple[PracticeRunner, dict[str, Any]]:
    """The night's practice, and the channel's record for the step's log (empty when it's off)."""
    reading = _reading(deps, uid, night)
    if reading is None:
        return deps.practice(chat, night), {}
    assert deps.channel is not None
    channel = _channel(deps, reading)
    snap = channel.snapshot
    record = {"arm": snap.arm, "core_model": snap.core_model, "state": snap.state_sha, "observed": snap.observed,
              "projector": channel.projector.id, "k": channel.projector.k, "own_projector": reading.projector is not None}
    return _Stamped(deps.channel.practice(chat, night, channel), channel), {"channel": record}


def train(
    deps: NightDeps, uid: str, night: date, prev: CheckpointRef | None, data: TrainingData, name: str,
) -> tuple[CheckpointRef, Mapping[str, object]]:
    """The night's training: the projector with the LoRA when the night has a reading and a row
    carries a state, else the LoRA alone, as today."""
    reading = _reading(deps, uid, night)
    if reading is None or not any(getattr(example, "core", None) for example in data.examples):
        return deps.trainer.train(uid, night, prev, data, name)
    assert deps.channel is not None and isinstance(deps.trainer, ChannelTrainer)
    ckpt, receipt = deps.trainer.train_with_channel(uid, night, prev, data, name, arm=deps.channel.arm,
                                                    projector=projector_path(deps, reading))
    channel = receipt.get("channel_receipt")
    return ckpt, {**receipt, **({"channel": json.loads(channel)} if isinstance(channel, str) and channel else {})}
