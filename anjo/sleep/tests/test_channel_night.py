# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The thought channel in the night (``night.channel_night``): with MIND's state of the person and the
channel's setup, practice replies through the state as soft tokens, its transcripts and rows carry
the state, and the night's one training job trains the projector with the LoRA; without either, the
night is today's. Fakes only: no torch, no network."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.context.brief_store import TrainingBriefs
from anjo.sleep.context.builder import RecallSettings
from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.context.training import TrainingContext
from anjo.sleep.mind.core.projector import Projector
from anjo.sleep.night import channel_night
from anjo.sleep.night.channel_night import ChannelSetup, MindReading
from anjo.sleep.night.core_snapshot import CoreSnapshot, state_sha
from anjo.sleep.night.nightlog import STORY
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.night.story import NO_CHANNEL_LOSS, Learn, NotRun, story_from
from anjo.sleep.practice.run import MinimalPractice
from anjo.sleep.tests.night_fakes import NIGHT, Trainer, World, result, step
from anjo.sleep.tests.test_thought_channel import K, _anchors, _predictions, _state
from anjo.sleep.types import CheckpointRef

UID = "p1"


class SoftGlimmer:
    """The soft-token client: every reply records whether soft tokens came with it."""

    def __init__(self) -> None:
        self.soft: list[bool] = []

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None, soft: Any = None) -> Any:
        self.soft.append(soft is not None)
        return result(f"Start with the infeed and keep it simple, step {len(self.soft)}.")


class ChannelTrainer(Trainer):
    """The night's trainer, able to train the projector with the LoRA."""

    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.channel: list[dict[str, Any]] = []

    def train_with_channel(self, user_id: str, night: date, prev: CheckpointRef | None, data: Any, name: str, *,
                           arm: str, projector: Path) -> tuple[CheckpointRef, dict[str, Any]]:
        states = {CoreSnapshot.from_json(json.loads(e.core)).state_sha for e in data.examples if e.core}
        self.channel.append({"arm": arm, "projector": projector, "states": states})
        ckpt, receipt = self.train(user_id, night, prev, data, name)
        kept = self.world.root / "adapters" / f"{name}.projector.npz"
        return ckpt, {**receipt, "projector": str(kept), "channel_receipt": json.dumps({"val_state_margin": 0.2})}


def _setup(world: World, glimmer: SoftGlimmer, shared: Path, arm: str = "state") -> ChannelSetup:
    def practice(chat: Any, night: date, client: Any) -> MinimalPractice:
        recall = Bm25Recall(world.evidence, until=night)
        context = TrainingContext(TrainingBriefs(world.briefs), recall, RecallSettings(k=6, budget_tokens=1200))
        return MinimalPractice(chat=chat, client=client, snapshot=context.snapshot, seed=7)

    return ChannelSetup(arm, shared, glimmer, practice)  # type: ignore[arg-type]


def _projector(path: Path) -> Path:
    path.write_bytes(Projector.fresh(anchor_rows=_anchors(), k=K, d=8, width=16).save())
    return path


def _reading(projector: str | None = None) -> MindReading:
    state = _state(3, person=UID)
    return MindReading(state, state_sha(state), _predictions(state, (0.1, 0.4, 0.2, 0.3, 0.9, 0.0)), projector)


def test_practice_reads_the_state_and_the_night_trains_the_projector_with_the_lora(tmp_path: Path) -> None:
    # Arrange: MIND holds the person's state for the night; the channel is on, from the shared projector
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    trainer, glimmer = ChannelTrainer(world), SoftGlimmer()
    shared = _projector(tmp_path / "shared.projector.npz")
    asked: list[tuple[str, date]] = []

    def mind_state(uid: str, night: date) -> MindReading:
        asked.append((uid, night))
        return _reading()

    deps = world.deps(trainer=trainer, mind_state=mind_state, channel=_setup(world, glimmer, shared))

    # Act
    night = run_night_batch_with(deps, {UID: NIGHT})[UID]

    # Assert: every practice reply carried soft tokens, every transcript the state, and the night's
    # one job trained the projector, warm from the shared one, on rows made under that state
    assert night.status == "promoted" and glimmer.soft and all(glimmer.soft) and world.glimmer.calls == 0
    expected = _reading().state_id
    kept = world.transcripts(UID).for_night(NIGHT)
    assert kept and {CoreSnapshot.from_json(json.loads(str(t.core))).state_sha for t in kept} == {expected}
    assert trainer.channel == [{"arm": "state", "projector": shared, "states": {expected}}]
    record = step(night.log_path, "practice")["channel"]
    assert record["state"] == expected and record["arm"] == "state" and record["own_projector"] is False
    assert record["projector"] == Projector.load(shared.read_bytes()).id
    train = step(night.log_path, "train")["receipt"]
    assert train["channel"] == {"val_state_margin": 0.2} and train["projector"].endswith(".projector.npz")
    assert asked == [(UID, NIGHT), (UID, NIGHT)]  # practice, then training: the same night's reading
    learned = story_from(json.loads((night.log_path.parent / STORY).read_text(encoding="utf-8"))).learn
    assert isinstance(learned, Learn) and learned.stamp == night.checkpoint.id  # type: ignore[union-attr]
    assert learned.channel_loss == NotRun(NO_CHANNEL_LOSS)  # this fake job reports a margin, no loss


def test_a_person_with_their_own_projector_trains_from_it(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    trainer, glimmer = ChannelTrainer(world), SoftGlimmer()
    own = _projector(tmp_path / "own.projector.npz")
    deps = world.deps(trainer=trainer, mind_state=lambda uid, night: _reading(str(own)),
                      channel=_setup(world, glimmer, _projector(tmp_path / "shared.projector.npz")))

    night = run_night_batch_with(deps, {UID: NIGHT})[UID]

    assert night.status == "promoted" and trainer.channel[0]["projector"] == own
    assert step(night.log_path, "practice")["channel"]["own_projector"] is True

    gone = World(tmp_path / "gone")  # a kept projector that's gone fails the night, never falls back
    gone.plan(UID, NIGHT)
    deps = gone.deps(trainer=ChannelTrainer(gone), mind_state=lambda uid, night: _reading(str(tmp_path / "x.npz")),
                     channel=_setup(gone, SoftGlimmer(), tmp_path / "shared.projector.npz"))
    assert run_night_batch_with(deps, {UID: NIGHT})[UID].status == "failed"


@pytest.mark.parametrize("off", ["no_channel", "no_mind", "no_state"])
def test_without_the_channel_or_a_state_the_night_is_todays(tmp_path: Path, off: str) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    trainer, glimmer = ChannelTrainer(world), SoftGlimmer()
    setup = _setup(world, glimmer, _projector(tmp_path / "shared.projector.npz"))
    every: dict[str, dict[str, Any]] = {"no_channel": {"mind_state": lambda uid, night: _reading()},
                                        "no_mind": {"channel": setup},
                                        "no_state": {"channel": setup, "mind_state": lambda uid, night: None}}
    ports = every[off]

    night = run_night_batch_with(world.deps(trainer=trainer, **ports), {UID: NIGHT})[UID]

    assert night.status == "promoted" and trainer.channel == [] and len(trainer.calls) == 1
    assert glimmer.soft == [] and world.glimmer.calls > 0  # today's client, no soft tokens
    assert all(t.core is None for t in world.transcripts(UID).for_night(NIGHT))
    assert "channel" not in step(night.log_path, "practice")


def test_the_batch_refuses_a_channel_it_cannot_run_before_any_step(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    shared = _projector(tmp_path / "shared.projector.npz")
    reading = {"mind_state": lambda uid, night: _reading()}
    cases = [
        ({"trainer": world.trainer, "channel": _setup(world, SoftGlimmer(), shared)}, "can't train the projector"),
        ({"trainer": ChannelTrainer(world), "channel": _setup(world, SoftGlimmer(), tmp_path / "none.npz")},
         "shared projector"),
        ({"trainer": ChannelTrainer(world), "channel": _setup(world, SoftGlimmer(), shared, arm="zero"),
          "is_synthetic": lambda uid: False}, "generated lives only"),
    ]
    for ports, why in cases:
        with pytest.raises(ValueError, match=why):
            run_night_batch_with(world.deps(**reading, **ports), {UID: NIGHT})
    assert not world.logs(UID).exists()  # refused before any step wrote anything


def test_a_state_that_isnt_the_one_mind_kept_fails_practice(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT)
    reading = _reading()._replace(state_id="0" * 16)
    deps = world.deps(trainer=ChannelTrainer(world), mind_state=lambda uid, night: reading,
                      channel=_setup(world, SoftGlimmer(), _projector(tmp_path / "shared.projector.npz")))

    night = run_night_batch_with(deps, {UID: NIGHT})[UID]

    assert night.status == "failed" and step(night.log_path, "practice")["status"] == "failed"


def test_the_trainer_reads_the_projectors_shape_and_trains_both_in_one_job(tmp_path: Path) -> None:
    from anjo.sleep.train import channel_train, lora
    from anjo.sleep.train.dataset import Dataset, Example
    from anjo.sleep.train.trainer import LoraTrainer

    class Runner:
        def __init__(self) -> None:
            self.inputs: dict[str, Any] = {}

        def run(self, job: str, inputs: Any, *, timeout_s: float) -> dict[str, object]:
            self.inputs = dict(inputs)
            raise RuntimeError("stop here: the inputs are what this test reads")

    shared = _projector(tmp_path / "shared.projector.npz")
    core = json.dumps({"v": 1})
    data = Dataset(examples=(Example(id="r0", prompt=(("user", "hi"),), completion="hey", context_dropped=True,
                                     tokens=10, new=True, core=core),), tokens=10, dropped={})
    runner = Runner()
    trainer = LoraTrainer(runner, root=tmp_path / "training", publish=lambda f, n: None, revision="a" * 40)
    assert isinstance(trainer, channel_night.ChannelTrainer)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer.train_with_channel(UID, NIGHT, None, data, lora.adapter_name(UID, 1, 0), arm="state", projector=shared)

    block = runner.inputs["config"]["channel"]
    assert (block["k"], block["d"], block["width"], block["arm"]) == (K, 8, 16, "state")
    assert channel_train.check_channel(block) == block and runner.inputs["prev_projector"] == str(shared)
