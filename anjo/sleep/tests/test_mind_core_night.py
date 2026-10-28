# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's core's own night (``mind.core.night.CoreNight``): it starts from the person's kept
state, observes the day through an encoder, dreams once, picks what practice rehearses,
consolidates and keeps the night's unit. A small fresh core and the fake encoder, CPU only."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anjo.sleep.mind.core import api, store  # noqa: E402
from anjo.sleep.mind.core import state as core_state  # noqa: E402
from anjo.sleep.mind.core.latents import FakeEncoder  # noqa: E402
from anjo.sleep.mind.core.night import CoreNight  # noqa: E402
from anjo.sleep.mind.core.types import CoreConfig  # noqa: E402
from anjo.sleep.night.deps import MindCall  # noqa: E402
from anjo.sleep.night.run import run_night_batch_with  # noqa: E402
from anjo.sleep.stops import Cancelled  # noqa: E402
from anjo.sleep.tests.mind_night_fakes import mind_call  # noqa: E402
from anjo.sleep.tests.night_fakes import NIGHT, World, own, step, summary, thread  # noqa: E402
from anjo.sleep.types import Evidence  # noqa: E402

UID = "p1"
SMALL = CoreConfig(dim_in=16, width=16, slots=4)
TALK = ("The rental house cancelled the camera again", "I still have no backup camera for Thursday",
        "The investor call moved to Friday", "I told the crew Thursday is fine", "The budget is short by a third")


def _day(night: date, n: int = 5) -> list[Evidence]:
    return [own(UID, night, i, TALK[i % len(TALK)]) for i in range(n)]


def _call(world: World, night: date, *, cancelled: bool = False) -> MindCall:
    call = mind_call(world, UID, night, datetime.combine(night + timedelta(days=1), time.min, tzinfo=timezone.utc))
    return replace(call, cancelled=lambda: True) if cancelled else call


def _core_night(root: Path, *, seed: int = 0, curriculum: bool = True) -> CoreNight:
    core = api.Core.fresh(SMALL, seed=seed).with_thresholds(api.Thresholds(speak=0.5, ignite=0.0))
    return CoreNight(core, root / store.STORE_DIR, FakeEncoder(SMALL.dim_in), curriculum=curriculum)


def test_a_first_night_starts_fresh_dreams_once_and_keeps_one_unit(tmp_path: Path) -> None:
    world, night = World(tmp_path), _core_night(tmp_path)
    day = _day(NIGHT)
    dreamt = night.dream(_call(world, NIGHT), day)
    held = store.MindStore(tmp_path / store.STORE_DIR, UID)
    slept = core_state.load(held.state(dreamt.state_id or ""), model_id=night.core.model_id)
    assert slept.observed == len(day) and slept.pending == 0.0  # consolidated: the day's pressure is spent
    assert dreamt.state is not None and core_state.equal(dreamt.state, slept)  # what practice reads
    assert dreamt.state_id == hashlib.sha256(core_state.save(slept)).hexdigest()[:16]  # the rows' state_sha
    assert set(dreamt.weights) == {item.id for item in day}
    assert np.mean(list(dreamt.weights.values())) == pytest.approx(1.0)
    assert dreamt.rehearse and set(dreamt.rehearse) <= {item.id for item in day}
    assert dreamt.view is not None and dreamt.view.predictions.slots
    assert dreamt.view.predictions == night.core.predict(slept)  # consolidation leaves the predictions as they were
    receipt: dict[str, Any] = dict(dreamt.receipt)
    assert receipt["fresh"] == "first night" and receipt["from_version"] is None
    assert receipt["state"] == dreamt.state_id and receipt["model"] == night.core.model_id
    assert receipt["encoder"] == FakeEncoder(SMALL.dim_in).spec.spec_id() and receipt["events"] == len(day)
    assert receipt["picks"]["evidence_ids"] == list(dreamt.rehearse) and receipt["ignitions"]
    assert receipt["pressure"] > 0 and json.loads(json.dumps(receipt)) == receipt  # the night log holds it as is
    assert isinstance(receipt["boundaries"], list)
    for dream in [*receipt["dreams"], *receipt["boundaries"]]:  # the imagined states, kept for the sleep objective
        imagined = core_state.load(held.state(dream["state"]), model_id=night.core.model_id)
        assert imagined.known <= slept.known and set(dream["rests_on"]) <= {item.id for item in day}
    unit = night.keep(_call(world, NIGHT), dreamt, lora="ad-1", projector="proj-1")
    assert (unit["version"], unit["state"], unit["lora"], unit["parent"]) == (1, dreamt.state_id, "ad-1", None)
    assert unit["model"] == night.core.model_id and unit["projector"] == "proj-1"


def test_the_next_night_continues_and_a_rerun_never_observes_a_day_twice(tmp_path: Path) -> None:
    world, night = World(tmp_path), _core_night(tmp_path)
    first = night.dream(_call(world, NIGHT), _day(NIGHT))
    night.keep(_call(world, NIGHT), first, lora="ad-1", projector=None)
    later = NIGHT + timedelta(days=1)
    second = night.dream(_call(world, later), _day(later, 3))
    assert second.receipt["from_version"] == 1 and second.receipt["fresh"] is None
    held = store.MindStore(tmp_path / store.STORE_DIR, UID)
    assert core_state.load(held.state(second.state_id or ""), model_id=night.core.model_id).observed == 5 + 3
    night.keep(_call(world, later), second, lora="ad-2", projector=None)
    again = night.dream(_call(world, later), _day(later, 3))
    assert again.state_id == second.state_id and again.receipt["from_version"] == 1  # the same start, the same night


def test_a_state_kept_under_another_core_starts_fresh_and_says_why(tmp_path: Path) -> None:
    world = World(tmp_path)
    old = _core_night(tmp_path, seed=0)
    old.keep(_call(world, NIGHT), old.dream(_call(world, NIGHT), _day(NIGHT)), lora=None, projector=None)
    later = NIGHT + timedelta(days=1)
    dreamt = _core_night(tmp_path, seed=1).dream(_call(world, later), _day(later, 2))
    assert dreamt.receipt["fresh"] == "another core model" and dreamt.receipt["from_version"] == 1


def test_without_the_curriculum_the_dream_picks_nothing(tmp_path: Path) -> None:
    dreamt = _core_night(tmp_path, curriculum=False).dream(_call(World(tmp_path), NIGHT), _day(NIGHT))
    assert dreamt.rehearse == () and dreamt.receipt["picks"] is None and dreamt.receipt["ignitions"]


def test_a_cancelled_night_writes_nothing(tmp_path: Path) -> None:
    world, night = World(tmp_path), _core_night(tmp_path)
    with pytest.raises(Cancelled):
        night.dream(_call(world, NIGHT, cancelled=True), _day(NIGHT))
    assert not (tmp_path / store.STORE_DIR / UID).exists()
    dreamt = night.dream(_call(world, NIGHT), _day(NIGHT))
    with pytest.raises(Cancelled):
        night.keep(_call(world, NIGHT, cancelled=True), dreamt, lora=None, projector=None)
    assert store.MindStore(tmp_path / store.STORE_DIR, UID).units() == ()


def test_an_empty_day_keeps_the_state_it_started_from(tmp_path: Path) -> None:
    world, night = World(tmp_path), _core_night(tmp_path)
    dreamt = night.dream(_call(world, NIGHT), [])
    assert dreamt.weights == {} and dreamt.rehearse == () and dreamt.receipt["events"] == 0
    assert dreamt.state_id is not None


def test_two_nights_run_with_the_core_through_the_nights_own_steps(tmp_path: Path) -> None:
    world, night = World(tmp_path), _core_night(tmp_path)
    world.targets(UID).upsert(thread(UID))
    records = []
    for n in range(2):
        world.plan(UID, NIGHT + timedelta(days=n))
        result = run_night_batch_with(world.deps(core=night), {UID: NIGHT + timedelta(days=n)})[UID]
        dreamt, record = step(result.log_path, "dream"), summary(result.log_path)
        assert record["mind_unit"]["state"] == dreamt["state"] and "mind_weighted" in step(result.log_path, "dataset")
        records.append((dreamt["from_version"], record["mind_unit"]["version"]))
    assert records == [(None, 1), (1, 2)]
