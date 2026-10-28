# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's core as the thought channel's ``mind_state`` (``CoreNight.reading``): practice reads
tonight's dream, the rows carry its state, the night's unit keeps the state, the projector trained
with the LoRA and that LoRA together, and the next night starts from that projector. With the
channel off the night keeps no projector. A nap dreams on CPU and holds nothing on ``modal_usd``.
A small fresh core, the fake encoder and the channel's fakes: CPU only, no network."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from anjo.sleep.budget import Ledger  # noqa: E402
from anjo.sleep.mind.core import api, store  # noqa: E402
from anjo.sleep.mind.core.latents import FakeEncoder  # noqa: E402
from anjo.sleep.mind.core.night import CoreNight  # noqa: E402
from anjo.sleep.mind.core.projector import Projector  # noqa: E402
from anjo.sleep.mind.core.types import CoreConfig  # noqa: E402
from anjo.sleep.night.core_dreams import core_dreams  # noqa: E402
from anjo.sleep.night.core_snapshot import CoreSnapshot, state_sha  # noqa: E402
from anjo.sleep.night.dreams import Terms  # noqa: E402
from anjo.sleep.night.run import GPU_STEPS, run_night_batch_with  # noqa: E402
from anjo.sleep.night.state import checkpoint_id  # noqa: E402
from anjo.sleep.night.sleep_pressure import SleepPolicy  # noqa: E402
from anjo.sleep.serving.client import WakeResult  # noqa: E402
from anjo.sleep.tests.night_fakes import NIGHT, World, step, steps, summary, thread  # noqa: E402
from anjo.sleep.tests.test_channel_night import ChannelTrainer, SoftGlimmer, _setup  # noqa: E402
from anjo.sleep.tests.test_mind_core_dreams import Glimmer  # noqa: E402
from anjo.sleep.tests.test_sleep_pressure import _holds_on  # noqa: E402
from anjo.sleep.tests.test_thought_channel import K, _anchors  # noqa: E402
from anjo.sleep.types import CheckpointRef  # noqa: E402

UID = "p1"
SMALL = CoreConfig(dim_in=16, width=16, slots=4)


def _core_night(root: Path) -> CoreNight:
    core = api.Core.fresh(SMALL, seed=0).with_thresholds(api.Thresholds(speak=0.5, ignite=0.0))
    return CoreNight(core, root / store.STORE_DIR, FakeEncoder(SMALL.dim_in), people=frozenset({UID}),
                     synthetic=lambda uid: True)


class SavingTrainer(ChannelTrainer):
    """The channel's trainer, saving the projector it trained where its receipt says."""

    def train_with_channel(self, user_id: str, night: Any, prev: CheckpointRef | None, data: Any, name: str, *,
                           arm: str, projector: Path) -> tuple[CheckpointRef, dict[str, Any]]:
        ckpt, receipt = super().train_with_channel(user_id, night, prev, data, name, arm=arm, projector=projector)
        kept = Path(receipt["projector"])
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_bytes(projector.read_bytes())
        return ckpt, receipt


def _rows_state(world: World, night: Any) -> set[str | None]:
    """The state each of tonight's transcripts was made under (None: none)."""
    return {None if t.core is None else CoreSnapshot.from_json(json.loads(t.core)).state_sha
            for t in world.transcripts(UID).for_night(night)}


def test_practice_reads_tonights_dream_and_the_unit_keeps_state_projector_and_lora_together(tmp_path: Path) -> None:
    # Arrange: the core serves her, the channel is on from the shared projector
    world, core = World(tmp_path), _core_night(tmp_path)
    world.targets(UID).upsert(thread(UID))
    trainer, glimmer = SavingTrainer(world), SoftGlimmer()
    shared = tmp_path / "shared.projector.npz"
    shared.write_bytes(Projector.fresh(anchor_rows=_anchors(), k=K, d=SMALL.width, width=16).save())
    states: list[str] = []
    projectors: list[str] = []
    loras: list[str] = []
    for n in range(2):
        tonight = NIGHT + timedelta(days=n)
        world.plan(UID, tonight)
        deps = world.deps(core=core, mind_state=core.reading, trainer=trainer, channel=_setup(world, glimmer, shared))

        # Act
        result = run_night_batch_with(deps, {UID: tonight})[UID]

        # Assert: the rows, the training job and the unit all name the dream's state
        dreamt, unit = step(result.log_path, "dream"), summary(result.log_path)["mind_unit"]
        assert result.status == "promoted" and glimmer.soft and all(glimmer.soft)
        states.append(dreamt["state"])
        assert _rows_state(world, tonight) == {dreamt["state"]}
        assert trainer.channel[-1]["states"] == set(states)  # tonight's rows, and the replayed rows of her earlier nights
        assert step(result.log_path, "practice")["channel"]["state"] == dreamt["state"] == unit["state"]
        serving = world.registry.last_promoted(UID)
        assert serving is not None and unit["lora"] == serving.id
        assert unit["projector"] == step(result.log_path, "train")["receipt"]["projector"]
        held = store.MindStore(tmp_path / store.STORE_DIR, UID).before(tonight + timedelta(days=1))
        assert held is not None and (held.state, held.projector, held.lora) == (dreamt["state"], unit["projector"], serving.id)
        projectors.append(unit["projector"])
        loras.append(serving.id)
    assert trainer.channel[0]["projector"] == shared  # the first night from the shared projector,
    assert trainer.channel[1]["projector"] == Path(projectors[0])  # the next from the one her unit kept
    assert step(result.log_path, "practice")["channel"]["own_projector"] is True

    # A night that trains nothing new keeps the projector and the LoRA that still serve her, together
    third = NIGHT + timedelta(days=2)
    world.plan(UID, third)
    failing = Failing(world)
    failed = run_night_batch_with(world.deps(core=core, mind_state=core.reading, trainer=failing,
                                             channel=_setup(world, glimmer, shared)), {UID: third})[UID]
    unit = summary(failed.log_path)["mind_unit"]
    assert failed.status == "failed" and failing.channel and unit["state"] == step(failed.log_path, "dream")["state"]
    assert (unit["projector"], unit["lora"]) == (projectors[1], loras[1])


class Failing(ChannelTrainer):
    """The channel's trainer, whose job fails."""

    def train_with_channel(self, user_id: str, night: Any, prev: CheckpointRef | None, data: Any, name: str, *,
                           arm: str, projector: Path) -> tuple[CheckpointRef, dict[str, Any]]:
        self.channel.append({"arm": arm, "projector": projector})
        raise RuntimeError("the training job failed")


def test_with_the_channel_off_the_core_still_dreams_and_its_unit_keeps_no_projector(tmp_path: Path) -> None:
    world, core = World(tmp_path), _core_night(tmp_path)
    world.plan(UID, NIGHT)
    result = run_night_batch_with(world.deps(core=core), {UID: NIGHT})[UID]
    assert result.status == "promoted" and "channel" not in step(result.log_path, "practice")
    assert _rows_state(world, NIGHT) == {None}  # no transcript carries a state
    unit = summary(result.log_path)["mind_unit"]
    assert unit["state"] == step(result.log_path, "dream")["state"] and unit["projector"] is None


def test_the_reading_is_tonights_dream_none_for_anyone_else_and_refused_before_it(tmp_path: Path) -> None:
    world, core = World(tmp_path), _core_night(tmp_path)
    assert core.reading("someone-else", NIGHT) is None and core.dreamt("someone-else", NIGHT) is None
    for asked in (core.reading, core.dreamt):
        with pytest.raises(ValueError, match="before the night's dream"):
            asked(UID, NIGHT)
    world.plan(UID, NIGHT)
    result = run_night_batch_with(world.deps(core=core), {UID: NIGHT})[UID]
    reading, dreamt = core.reading(UID, NIGHT), core.dreamt(UID, NIGHT)
    assert reading is not None and reading.projector is None  # her first night: the shared projector
    assert reading.state_id == store.MindStore(tmp_path / store.STORE_DIR, UID).units()[-1].state
    assert dreamt is not None and dreamt.reading() == reading
    logged = step(result.log_path, "dream")  # its receipt is the dream's record: the states the dreams start from
    assert {key: logged[key] for key in dreamt.receipt} == dict(dreamt.receipt) and "boundaries" in logged


def test_a_nap_with_the_core_dreams_on_cpu_and_holds_nothing_on_modal(tmp_path: Path) -> None:
    world, core = World(tmp_path), _core_night(tmp_path)
    world.plan(UID, NIGHT, count=1)  # a quiet day: she naps
    ledger, woke = Ledger(tmp_path / "budget.jsonl"), []

    def wake() -> WakeResult:
        woke.append(1)
        return WakeResult(seconds=1.0, ok=True)

    deps = world.deps(core=core, mind_state=core.reading, sleep_policy=SleepPolicy(), turns=lambda uid, window: 4,
                      ledger=ledger, wake=wake)
    nap = run_night_batch_with(deps, {UID: NIGHT})[UID]
    assert nap.status == "napped" and step(nap.log_path, "sleep")["kind"] == "nap"
    dreamt = step(nap.log_path, "dream")
    assert dreamt["status"] == "ok" and dreamt["state"] and dreamt["events"] >= 1  # observed, dreamt, consolidated
    assert summary(nap.log_path)["mind_unit"]["state"] == dreamt["state"]
    assert _holds_on(ledger, "modal_usd") == [] and woke == [] and world.gpu.opened == []
    assert world.glimmer.calls == 0 and world.trainer.calls == []
    skipped = {r["step"] for r in steps(nap.log_path) if r["status"] == "skipped"}
    assert "dream" not in skipped and skipped == {"probes", *GPU_STEPS}


def _shared(root: Path) -> Path:
    shared = root / "shared.projector.npz"
    shared.write_bytes(Projector.fresh(anchor_rows=_anchors(), k=K, d=SMALL.width, width=16).save())
    return shared


def _channel_night(world: World, core: CoreNight, trainer: Any, shared: Path, night: Any) -> Any:
    world.plan(UID, night)
    deps = world.deps(core=core, mind_state=core.reading, trainer=trainer, channel=_setup(world, SoftGlimmer(), shared))
    return run_night_batch_with(deps, {UID: night})[UID]


def test_a_promotion_keeps_her_unit_first_or_promotes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a first night promotes, its unit kept with the promotion, once
    world, core, shared = World(tmp_path), _core_night(tmp_path), _shared(tmp_path)
    trainer = SavingTrainer(world)
    world.targets(UID).upsert(thread(UID))
    held = store.MindStore(tmp_path / store.STORE_DIR, UID)
    first = _channel_night(world, core, trainer, shared, NIGHT)
    kept, serving = held.head(), world.registry.last_promoted(UID)
    assert first.status == "promoted" and kept is not None and serving is not None and kept.lora == serving.id
    assert step(first.log_path, "promote")["mind_unit"] == summary(first.log_path)["mind_unit"] and len(held.units()) == 1

    # Act: the next night's unit can't be saved
    def full_disk(self: store.MindStore, *args: Any, **kwargs: Any) -> store.Unit:
        raise OSError("the disk is full")

    with monkeypatch.context() as patched:
        patched.setattr(store.MindStore, "keep", full_disk)
        second = _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=1))

    # Assert: nothing was promoted, so the adapter and the projector that served keep serving together
    assert second.status == "failed" and step(second.log_path, "promote")["status"] == "failed"
    assert checkpoint_id(world.registry.last_promoted(UID)) == kept.lora and held.head() == kept
    third = _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=2))
    assert third.status == "promoted" and trainer.calls[-1]["prev"].id == kept.lora
    assert kept.projector is not None and trainer.channel[-1]["projector"] == Path(kept.projector)
    head = held.head()
    assert head is not None and head.lora == checkpoint_id(world.registry.last_promoted(UID))
    assert head.projector == step(third.log_path, "train")["receipt"]["projector"]


def test_a_unit_kept_for_a_promotion_that_then_fails_is_followed_by_one_naming_the_serving_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: a first night promotes; the next one keeps its unit, then its promotion raises
    world, core, shared = World(tmp_path), _core_night(tmp_path), _shared(tmp_path)
    trainer = SavingTrainer(world)
    world.targets(UID).upsert(thread(UID))
    held = store.MindStore(tmp_path / store.STORE_DIR, UID)
    assert _channel_night(world, core, trainer, shared, NIGHT).status == "promoted"
    (first,) = held.units()

    def refused(*args: Any, **kwargs: Any) -> None:
        raise OSError("the registry can't be written")

    # Act
    with monkeypatch.context() as patched:
        patched.setattr(world.registry, "promote", refused)
        second = _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=1))

    # Assert: the promotion's unit names an adapter that never served, so the night's end keeps one
    # naming the adapter that does, with the projector that served beside it
    assert second.status == "failed" and step(second.log_path, "promote")["status"] == "failed"
    _, unserved, last = held.units()
    assert unserved.lora != first.lora and unserved.lora != checkpoint_id(world.registry.last_promoted(UID))
    assert last.lora == first.lora == checkpoint_id(world.registry.last_promoted(UID))
    assert last.projector == first.projector and summary(second.log_path)["mind_unit"]["lora"] == first.lora
    third = _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=2))
    assert "projector_fallback" not in step(third.log_path, "dream")
    assert first.projector is not None and trainer.channel[-1]["projector"] == Path(first.projector)


def test_a_unit_naming_another_adapter_than_the_one_serving_falls_back_logged(tmp_path: Path) -> None:
    # Arrange: two promoted nights, then the first adapter serves again
    world, core, shared = World(tmp_path), _core_night(tmp_path), _shared(tmp_path)
    world.targets(UID).upsert(thread(UID))
    trainer = SavingTrainer(world)
    for n in range(2):
        assert _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=n)).status == "promoted"
    first, second = store.MindStore(tmp_path / store.STORE_DIR, UID).units()
    world.registry.rollback(UID, world.registry.history(UID)[0])

    # Act
    third = _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=2))

    # Assert: read through the newest unit that names the adapter that serves, and logged
    assert step(third.log_path, "dream")["projector_fallback"] == {
        "unit": second.version, "unit_lora": second.lora, "serving": first.lora, "from_unit": first.version}
    assert first.projector is not None and trainer.channel[-1]["projector"] == Path(first.projector)
    # No unit names the adapter that serves: the shared projector, logged
    world.registry.promote(UID, CheckpointRef("elsewhere", UID, NIGHT, "uri://elsewhere", None))
    fourth = _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=3))
    assert step(fourth.log_path, "dream")["projector_fallback"]["from_unit"] is None
    assert trainer.channel[-1]["projector"] == shared
    assert "projector_fallback" not in step(third.log_path, "promote") and "projector_fallback" not in step(
        _channel_night(world, core, trainer, shared, NIGHT + timedelta(days=4)).log_path, "dream")


class DreamRows(SavingTrainer):
    """The channel's trainer, keeping the dream rows each night trained on."""

    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.dream_rows: list[Any] = []

    def train_with_channel(self, user_id: str, night: Any, prev: CheckpointRef | None, data: Any, name: str, *,
                           arm: str, projector: Path) -> tuple[CheckpointRef, dict[str, Any]]:
        self.dream_rows.extend(data.dreams)
        return super().train_with_channel(user_id, night, prev, data, name, arm=arm, projector=projector)


def test_her_dreams_through_the_channel_reply_from_the_states_her_dream_kept(tmp_path: Path) -> None:
    # Arrange: the real core, the channel on, and the dreams through it
    world, core, shared = World(tmp_path), _core_night(tmp_path), _shared(tmp_path)
    world.targets(UID).upsert(thread(UID))
    world.plan(UID, NIGHT)
    trainer, soft, glimmer = DreamRows(world), SoftGlimmer(), Glimmer()
    channel = _setup(world, soft, shared)
    deps = world.deps(core=core, mind_state=core.reading, trainer=trainer, channel=channel,
                      terms=Terms(seed=True, dreams=True), dreams=core_dreams(core, world.evidence, glimmer, seed=7, channel=channel))

    # Act
    night = run_night_batch_with(deps, {UID: NIGHT})[UID]

    # Assert: every dream row carries one of the states her dream kept, never the consolidated one,
    # and each was read back from her store as it was kept
    dreamt = step(night.log_path, "dream")
    kept = {d["state"] for key in ("dreams", "boundaries") for d in dreamt[key]}
    assert night.status == "promoted" and trainer.dream_rows and len(glimmer.calls) == len(trainer.dream_rows)
    assert all(CoreSnapshot.from_json(json.loads(row.core)).dreamt for row in trainer.dream_rows)
    assert all(not CoreSnapshot.from_json(json.loads(t.core)).dreamt for t in world.transcripts(UID).for_night(NIGHT)
               if t.core is not None)  # practice's own rows carry her real state
    snaps = {CoreSnapshot.from_json(json.loads(row.core)).state_sha for row in trainer.dream_rows}
    assert snaps <= kept and dreamt["state"] not in snaps
    for ref in snaps:
        reading = core.imagined(UID, NIGHT, ref)
        assert reading is not None and state_sha(reading.state) == ref and reading.projector is None
    with pytest.raises(ValueError, match="kept"):
        core.imagined(UID, NIGHT, dreamt["state"])  # tonight's consolidated state is no dream
    assert core.imagined("someone-else", NIGHT, next(iter(snaps))) is None
