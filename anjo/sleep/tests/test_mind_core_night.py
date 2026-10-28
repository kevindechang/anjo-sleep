# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's core's own night (``mind.core.night.CoreNight``): it starts from the person's kept
state, observes what it has not (a killed night's day too) through an encoder, dreams once, opens
a question in her workspace for each ignition on her real evidence, picks what practice
rehearses, consolidates and keeps the night's unit. A small fresh core and the fake encoder, CPU
only."""
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
from anjo.sleep.mind.inquiry import kernel  # noqa: E402
from anjo.sleep.mind.inquiry import types as it  # noqa: E402
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog  # noqa: E402
from anjo.sleep.night.deps import MindCall  # noqa: E402
from anjo.sleep.night.run import run_night_batch_with  # noqa: E402
from anjo.sleep.stops import Cancelled  # noqa: E402
from anjo.sleep.tests.inquiry_fakes import Ws  # noqa: E402
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


def _core_night(root: Path, *, seed: int = 0, curriculum: bool = True, **options: Any) -> CoreNight:
    core = api.Core.fresh(SMALL, seed=seed).with_thresholds(api.Thresholds(speak=0.5, ignite=0.0))
    return CoreNight(core, root / store.STORE_DIR, FakeEncoder(SMALL.dim_in), curriculum=curriculum,
                     **{"people": frozenset({UID}), "synthetic": lambda uid: True, **options})


def _state(root: Path, night: CoreNight, state_id: str | None) -> core_state.CoreState:
    return core_state.load(store.MindStore(root / store.STORE_DIR, UID).state(state_id or ""), model_id=night.core.model_id)


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
    assert all(0.5 <= w <= 2.0 for w in dreamt.weights.values())  # bounded: no one event takes the night
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


def test_a_killed_nights_day_is_caught_up_the_next_night_and_never_observed_twice(tmp_path: Path) -> None:
    world, night = World(tmp_path), _core_night(tmp_path)
    second, third = NIGHT + timedelta(days=1), NIGHT + timedelta(days=2)
    days = {NIGHT: _day(NIGHT), second: _day(second, 3), third: _day(third, 2)}
    night.keep(_call(world, NIGHT), night.dream(_call(world, NIGHT), days[NIGHT]), lora=None, projector=None)
    killed = night.dream(_call(world, second), [*days[NIGHT], *days[second]])  # its unit never kept
    assert killed.receipt["events"] == 3 and killed.receipt["events_before_tonight"] == 0
    everything = [*days[NIGHT], *days[second], *days[third]]
    caught = night.dream(_call(world, third), everything)
    assert caught.receipt["from_version"] == 1 and caught.receipt["events"] == 3 + 2
    assert caught.receipt["events_before_tonight"] == 3 and set(caught.weights) == {i.id for i in [*days[second], *days[third]]}
    slept = _state(tmp_path, night, caught.state_id)
    assert slept.observed == 5 + 3 + 2 and slept.through is not None and slept.through[1] == days[third][-1].id
    assert night.dream(_call(world, third), everything).state_id == caught.state_id  # a rerun: the same night


def _seen(items: list[Evidence], *, tonight: it.RunCall | None, as_of: datetime) -> MemoryWorkspaceLog:
    """Her workspace holding ``items`` (the notice pass took them in), under a run of tonight's
    (``tonight``) or, with ``None``, of the night before."""
    w = Ws()
    when = NIGHT if tonight is not None else NIGHT - timedelta(days=1)
    w.ok(it.RunStart(when, as_of, tonight or "notice", it.FULL_RULES, "priority", None, it.Budget()), at=as_of)
    w.arrive(*items, at=as_of)
    return MemoryWorkspaceLog(tuple(w.events))


def _targeted(world: World) -> list[Evidence]:
    world.targets(UID).upsert(thread(UID))
    return [replace(item, target_id=f"{UID}-t1") for item in _day(NIGHT)]


def test_each_ignition_opens_a_question_through_the_kernel_on_her_real_evidence(tmp_path: Path) -> None:
    world = World(tmp_path)
    day, as_of = _targeted(world), _call(world, NIGHT).as_of
    log = _seen(day, tonight="notice", as_of=as_of)
    night = _core_night(tmp_path, workspace=lambda uid: log)
    before = len(log.events())
    dreamt = night.dream(_call(world, NIGHT), day)
    receipt: dict[str, Any] = dict(dreamt.receipt)
    ws = kernel.fold(log.events())
    assert receipt["ignite"] == "opened" and receipt["questions_opened"] == [q.id for q in ws.questions]
    assert receipt["questions_opened"] and all(set(q.origin_evidence) <= {i.id for i in day} for q in ws.questions)
    assert len(receipt["ignitions_dropped"]) == len(receipt["ignitions"]) - len(receipt["questions_opened"])
    assert {d["why"] for d in receipt["ignitions_dropped"]} <= {"already_open", "nothing_real", "no_targets"}
    assert all(not isinstance(e.step.entry, it.RunStart) for e in log.events()[before:])  # under tonight's notice pass
    again = night.dream(_call(world, NIGHT), day)  # a rerun: what is open stays open, once
    assert again.receipt["questions_opened"] == [] and len(kernel.fold(log.events()).questions) == len(ws.questions)


def test_without_a_run_tonight_a_notice_start_comes_first_and_after_the_inquiry_began_nothing_opens(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    day, as_of = _targeted(world), _call(world, NIGHT).as_of
    quiet = _seen(day, tonight=None, as_of=as_of)  # the notice pass found nothing new tonight
    before = len(quiet.events())
    _core_night(tmp_path, workspace=lambda uid: quiet).dream(_call(world, NIGHT), day)
    start = quiet.events()[before].step.entry
    assert isinstance(start, it.RunStart) and (start.night, start.call) == (NIGHT, "notice")
    assert kernel.fold(quiet.events()).questions

    begun = _seen(day, tonight="inquiry", as_of=as_of)  # a rerun of a night killed in its inquiry
    held = begun.events()
    dreamt = _core_night(tmp_path / "b", workspace=lambda uid: begun).dream(_call(world, NIGHT), day)
    assert dreamt.receipt["ignite"] == "skipped: tonight's inquiry began" and begun.events() == held


def test_fake_latents_are_refused_for_a_real_person_and_a_state_built_on_them_says_so(tmp_path: Path) -> None:
    world = World(tmp_path)
    with pytest.raises(ValueError, match="fake latents"):
        _core_night(tmp_path, synthetic=lambda uid: False).dream(_call(world, NIGHT), _day(NIGHT))
    assert not (tmp_path / store.STORE_DIR / UID).exists()
    allowed = _core_night(tmp_path, synthetic=lambda uid: False, fake_for_anyone=True)
    dreamt = allowed.dream(_call(world, NIGHT), _day(NIGHT))
    assert dreamt.receipt["fake_latents"] is True and _state(tmp_path, allowed, dreamt.state_id).fake_latents
