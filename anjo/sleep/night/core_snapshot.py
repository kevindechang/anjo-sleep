"""The core state a training row was made under: the row field (``core``) that ties a night's rows
to MIND (one system, ``brief_one_system``). Torch-free.

Practice replies while Glimmer reads the person's core state through the thought channel, so
every row it makes carries the state it was made under, as a ``CoreSnapshot``:

- ``core_model``: the shared core's model id (``Core.model_id``), so the row names the weights the
  state means something under;
- ``state_sha``: the person's saved state (``state.save``), hashed: the state itself stays in the
  person's versioned store, beside the LoRA and projector versions it was saved with, never on a row;
- ``observed``: how many events the state had read;
- ``arm``: ``state``, or ``zero`` for the zero-tokens ablation (the same nights, the channel's K
  positions with nothing in them), so a row says which arm it trained;
- ``reading``: the projector's input (``mind.core.projector.Reading``), which is all the training
  job needs: it trains the projector and the LoRA on it without the core's weights;
- ``dreamt``: ``true`` on a dream's snapshot only (a state MIND imagined, never hers), and absent
  on a real state's, whose row field is as it always was. Every reader that takes a snapshot for
  her real state refuses a dreamt one: ``Channel.stamp`` (practice), the dataset's examples, the
  pretraining's readings (``reader_data.read_readings``) and the training job's ``readings_of``
  for any row but a dream's own state (and a dream row's state must be dreamt).

**Practice reads MIND** through ``Channel``: a ``ModelClient`` over the soft-token client, so every
practice reply is made with the channel live, and ``Channel.stamp`` puts the snapshot on the
transcripts it made (``PracticeTranscript.core``), from which the dataset's rows take it.

On a dataset row it is ``row["core"] = snapshot.to_json()`` (an SFT row and a pair row alike); the
training job reads ``row["core"]["reading"]`` (``train.channel_train``). ``ANJO_SLEEP_CHANNEL``
(``channel_arm``) turns the channel on for a night: ``state``, ``zero`` (generated lives only, never a
real person's night), or unset for off, today's night with no soft tokens and no ``core`` field.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, get_args

if TYPE_CHECKING:  # the type only: practice imports nothing of the night
    from anjo.sleep.practice.transcripts import PracticeTranscript

from anjo.sleep.mind.core.projector import K_DEFAULT, PredictionsLike, Projector, Reading, read
from anjo.sleep.mind.core.state import CoreState, save
from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving.soft_tokens import SoftTokens
from anjo.sleep.types import CheckpointRef

ChannelArm = Literal["state", "zero"]
CHANNEL_ARMS: Final = frozenset(get_args(ChannelArm))
CHANNEL_ENV: Final = "ANJO_SLEEP_CHANNEL"
ROW_FIELD: Final = "core"
DREAMT: Final = "dreamt"  # the snapshot's mark of an imagined state (the training job reads the same key)
SNAPSHOT_VERSION: Final = 1


def channel_arm(env: Mapping[str, str], *, synthetic: bool) -> ChannelArm | None:
    """The night's channel: ``state``, or ``zero`` (the ablation), or None when it's off (unset or
    blank). The zero arm is for generated lives only: a real person's night runs the full system,
    so ``zero`` for one is refused."""
    raw = env.get(CHANNEL_ENV, "").strip().casefold()
    if not raw:
        return None
    if raw not in CHANNEL_ARMS:
        raise ValueError(f"{CHANNEL_ENV} is one of {', '.join(sorted(CHANNEL_ARMS))}, or unset for off")
    if raw == "zero" and not synthetic:
        raise ValueError("the zero-tokens ablation runs on generated lives only; a real person runs the full system")
    return "zero" if raw == "zero" else "state"


def state_sha(state: CoreState) -> str:
    """The id of a person's saved state: the same state, the same id (``state.save`` is byte-stable)."""
    return hashlib.sha256(save(state)).hexdigest()[:16]


@dataclass(frozen=True)
class CoreSnapshot:
    core_model: str
    state_sha: str
    observed: int
    arm: ChannelArm
    reading: Reading
    dreamt: bool = False  # an imagined state's, for a dream row: never her real state

    def __post_init__(self) -> None:
        if self.arm not in CHANNEL_ARMS:
            raise ValueError(f"a snapshot's arm is one of {', '.join(sorted(CHANNEL_ARMS))}")
        if not self.core_model or not self.state_sha or self.observed < 0:
            raise ValueError("a snapshot names its core, its state and how many events the state read")

    def to_json(self) -> dict[str, Any]:
        return {"v": SNAPSHOT_VERSION, "core_model": self.core_model, "state": self.state_sha,
                "observed": self.observed, "arm": self.arm, "reading": self.reading.to_json(),
                **({DREAMT: True} if self.dreamt else {})}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> CoreSnapshot:
        if obj.get("v") != SNAPSHOT_VERSION:
            raise ValueError(f"core snapshot version {obj.get('v')!r}; this build reads {SNAPSHOT_VERSION}")
        arm, dreamt = obj.get("arm"), obj.get(DREAMT, False)
        if arm not in CHANNEL_ARMS:
            raise ValueError("a snapshot's arm is state or zero")
        if not isinstance(dreamt, bool):
            raise ValueError("a snapshot's dreamt mark is true or absent")
        return cls(str(obj["core_model"]), str(obj["state"]), int(obj["observed"]),
                   "zero" if arm == "zero" else "state", Reading.from_json(obj["reading"]), dreamt=dreamt)


def snapshot(
    state: CoreState, predictions: PredictionsLike, *, arm: ChannelArm, k: int = K_DEFAULT, dreamt: bool = False,
) -> CoreSnapshot:
    """The row field for a row made now, under ``state`` (``predictions`` is ``Core.predict(state)``;
    ``dreamt`` when the state is one MIND imagined). The zero arm keeps the state's id and drops its
    reading: same nights, nothing in the channel."""
    reading = read(state, predictions, k=k) if arm == "state" else Reading.empty(k, int(state.glob.shape[0]))
    return CoreSnapshot(state.model_id, state_sha(state), state.observed, arm, reading, dreamt=dreamt)


class SoftClient(Protocol):
    """A client that can put soft tokens in the latest user turn (``VllmClient.with_soft_tokens``)."""

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None, soft: SoftTokens | None = None,
    ) -> ChatResult: ...


@dataclass(frozen=True, eq=False)
class Channel:
    """Practice reading MIND: a ``ModelClient`` whose every reply carries the person's state as soft
    tokens, and the snapshot each transcript it makes is stamped with (``stamp``)."""

    snapshot: CoreSnapshot
    projector: Projector
    client: SoftClient
    tokens: SoftTokens = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", self.projector.tokens(self.snapshot.reading))

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
        seed: int | None = None, max_tokens: int | None = None,
    ) -> ChatResult:
        return self.client.complete(adapter, messages, temperature=temperature, seed=seed, max_tokens=max_tokens,
                                    soft=self.tokens)

    @property
    def row_field(self) -> str:
        """The snapshot as a row's ``core`` (JSON)."""
        return json.dumps(self.snapshot.to_json(), sort_keys=True)

    def stamp(self, transcripts: Sequence[PracticeTranscript]) -> list[PracticeTranscript]:
        """The transcripts with the state their replies were made under: her real state, never a dreamt one."""
        if self.snapshot.dreamt:
            raise ValueError("practice replies under her real state; a dreamt state is never stamped on its transcripts")
        core = self.row_field
        return [replace(t, core=core) for t in transcripts]
