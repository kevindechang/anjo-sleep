# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A judge's two-phase night (``night/demo.py``) on the night's fakes: the profile and its fast gate,
phase 1 (the study in the background, MIND's early commitment, the note) before phase 2, the note's
rewrite after the gates, and the observer a judge's night view reads. Every other night is unchanged."""
from __future__ import annotations

import threading
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pytest

from anjo.sleep.mind import loop_focus
from anjo.sleep.night import demo
from anjo.sleep.night.deps import NightSettings, StudyOutcome
from anjo.sleep.night.meter import WonderMeter
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.note.hold import HELD_BACK_LINE
from anjo.sleep.practice.run import PracticeRun
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.sets.loader import load_collapse, load_general
from anjo.sleep.tests.night_fakes import NIGHT, Homework, World, step, steps, thread
from anjo.sleep.types import Target

UID = "judge-1"
PROFILE = demo.DemoProfile()
DEMO = NightSettings(token_cap=16384, min_new_examples=1, demo=PROFILE)
PHASE_ONE = ["close_day", "extract", "notice", "probes", "choose", "brief", "homework", "self_predict", "note"]
PHASE_TWO = ["practice", "dataset", "train", "gates", "promote", "refine", "finish"]


class Observed:
    """The observer a judge's night view passes: every (step, state) in order, and its receipts."""

    def __init__(self, fail: bool = False) -> None:
        self.events: list[tuple[str, str]] = []
        self.receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self.fail = fail

    def __call__(self, user_id: str, step: str, state: str, receipt: Any) -> None:
        assert user_id == UID
        self.events.append((step, state))
        self.receipts[(step, state)] = dict(receipt)
        if self.fail:
            raise RuntimeError("the view is down")


class ThreadedHomework(Homework):
    """The fake study, recording the thread it ran on."""

    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.threads: list[str] = []

    def study(self, user_id: str, target: Target, night: date, meter: WonderMeter, *, action: str = "",
              at: Any = None) -> StudyOutcome:
        self.threads.append(threading.current_thread().name)
        return super().study(user_id, target, night, meter, action=action, at=at)


class AdapterWriter:
    """A note writer that says which adapter wrote it (as Glimmer's writer uses the promoted one)."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.calls = 0

    def write(self, user_id: str, topic: str, facts: Any) -> str | None:
        self.calls += 1
        serving = self.world.registry.last_promoted(user_id)
        return f"I looked into {topic} (written by {serving.id if serving else 'the base'})."


class LoopPractice:
    """Practice that draws three scenes per focus target (each keeping her own reply) and three others."""

    max_spark_calls = 1

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    def run(self, user_id: str, night: date, evidence: Any, hw: Any, *, at: Any, focus: Any = ()) -> PracticeRun:
        self.seen.append(tuple(target.target_id for target in focus))
        plan: list[tuple[int, bool, str | None, Literal["glimmer", "teacher"]]] = [
            (i, True, target.target_id, "glimmer") for target in focus for i in range(3)]
        plan += [(100 + i, False, None, "teacher") for i in range(3)]
        scenes = tuple(PracticeTranscript(
            id=f"pr-{index}", person_id=user_id, night=night, scene=index, message=f"message {index}", reply="a reply",
            context=None, f_ids=(), ancestry=frozenset(), created_at=at, target=target, focus=on, target_id=tid,
            flagged=False) for index, on, tid, target in plan)
        return PracticeRun(scenes, tuple(item.id for item in evidence), 0)


def _world(root: Path) -> World:
    world = World(root)
    world.plan(UID, NIGHT)
    world.targets(UID).upsert(thread(UID))
    return world


def _night(world: World, **deps: Any) -> Any:
    return run_night_batch_with(world.deps(settings=deps.pop("settings", DEMO), **deps), {UID: NIGHT})[UID]


# ── the profile ──────────────────────────────────────────────────────────────


def test_the_profile_is_the_briefs_numbers_and_the_env_overrides_them() -> None:
    assert (PROFILE.practice_scenes, PROFILE.token_cap, PROFILE.general_items, PROFILE.collapse_prompts,
            PROFILE.train_steps, PROFILE.probes) == (8, 5000, 40, 10, 8, False)
    assert demo.from_env({}) == PROFILE
    chosen = demo.from_env({demo.SCENES_ENV: "10", demo.GENERAL_ENV: " 60 ", demo.STEPS_ENV: "12"})
    assert (chosen.practice_scenes, chosen.general_items, chosen.train_steps, chosen.token_cap) == (10, 60, 12, 5000)


@pytest.mark.parametrize("env, value", [(demo.SCENES_ENV, "3"), (demo.GENERAL_ENV, "201"),
                                        (demo.COLLAPSE_ENV, "ten"), (demo.TOKEN_CAP_ENV, "999")])
def test_a_profile_value_out_of_range_stops_the_night(env: str, value: str) -> None:
    with pytest.raises(demo.DemoProfileError):
        demo.from_env({env: value})


def test_the_fast_gate_is_a_fixed_recorded_subset_in_registered_order() -> None:
    general, collapse = PROFILE.general_set(), PROFILE.collapse_set()
    assert len(general) == 40 and len(collapse) == 10
    registered = [item.id for item in load_general()]
    assert [item.id for item in general] == [i for i in registered if i in {g.id for g in general}]
    again = demo.fast_subset(load_general(), 40, demo.SUBSET_SEED, lambda item: item.id)
    assert again == general  # whoever asks gets the same items
    other = demo.fast_subset(load_general(), 40, demo.SUBSET_SEED + 1, lambda item: item.id)
    assert other != general  # the seed decides them, and the receipt records it
    receipt = PROFILE.gate_receipt()
    assert receipt == {
        "general_items": 40, "general_ids_sha256": demo.ids_digest([i.id for i in general]),
        "collapse_prompts": 10, "collapse_ids_sha256": demo.ids_digest([p.id for p in collapse]),
        "seed": demo.SUBSET_SEED, "thresholds": "registered",
    }
    assert {p.id for p in collapse} <= {p.id for p in load_collapse()}


# ── the night's two phases ───────────────────────────────────────────────────


def test_phase_one_runs_before_any_gpu_step_and_the_study_runs_alongside(tmp_path: Path) -> None:
    world = _world(tmp_path)
    homework, seen = ThreadedHomework(world), Observed()
    result = _night(world, homework=homework, observer=seen)

    assert result.status == "promoted"
    assert [record["step"] for record in steps(result.log_path)] == PHASE_ONE + PHASE_TWO
    assert step(result.log_path, "probes") == {**step(result.log_path, "probes"), "status": "skipped",
                                               "why": demo.PROBES_SKIPPED}
    assert step(result.log_path, "choose")["study"] == {"background": True}
    assert homework.threads and homework.threads[0].startswith("night-study")  # off the night's thread
    started = seen.events.index(("homework", "running"))
    assert seen.events[started - 1] == ("choose", "running")  # started inside choose, before the brief
    assert seen.events.index(("brief", "running")) > started
    kept = step(result.log_path, "homework")
    assert kept["study"]["homework_items"] == 1 and kept["receipt"]["outcome"] == "ran"
    assert [e.kind for e in world.evidence(UID).stored(NIGHT)].count("homework") == 1


def test_the_note_is_written_in_phase_one_and_rewritten_by_the_promoted_adapter(tmp_path: Path) -> None:
    world = _world(tmp_path)
    writer = AdapterWriter(world)
    result = _night(world, homework=Homework(world), note_writer=writer)

    first, second = world.notifier.queued
    assert "written by the base" in first[1]  # phase 1: before tonight's weights existed
    assert f"written by {result.checkpoint.id}" in second[1] and second[2] == "sleep_note"
    assert step(result.log_path, "note")["outcome"] == "held"
    refined = step(result.log_path, "refine")
    assert refined["outcome"] == "refined" and refined["note_id"] == step(result.log_path, "note")["note_id"]
    assert writer.calls == 2


def test_a_rejected_nights_note_gets_the_held_back_line_without_a_second_write(tmp_path: Path) -> None:
    world = _world(tmp_path)
    world.failing_gates[UID] = {"general"}
    writer = AdapterWriter(world)
    result = _night(world, homework=Homework(world), note_writer=writer)

    assert result.status == "rejected"
    first, second = world.notifier.queued
    assert second[1] == f"{first[1]}\n\n{HELD_BACK_LINE}"
    assert step(result.log_path, "refine")["outcome"] == "held_back_line" and writer.calls == 1


def test_a_night_with_nothing_new_skips_phase_two_and_its_note_stands(tmp_path: Path) -> None:
    world = World(tmp_path)  # no planned evidence: nothing to train on
    world.targets(UID).upsert(thread(UID))
    seen = Observed()
    result = _night(world, homework=Homework(world), observer=seen)

    assert result.status == "nothing_new"
    for name in ("practice", "dataset", "train", "gates", "promote"):
        assert step(result.log_path, name)["status"] == "skipped"
        assert (name, "skipped") in seen.events
    assert step(result.log_path, "self_predict")["noop"] is True
    assert step(result.log_path, "refine")["noop"] is True
    assert len(world.notifier.queued) == 1


def test_the_fast_gate_is_recorded_with_the_gates(tmp_path: Path) -> None:
    result = _night(_world(tmp_path))
    assert step(result.log_path, "gates")["fast_gate"] == PROFILE.gate_receipt()


def test_mind_commits_its_prediction_in_phase_one_and_practice_resolves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anjo.sleep.tests.mind_night_fakes import LifeThinker, mind_night, workspace
    from anjo.sleep.tests.test_mind_loop_focus import _seeded

    monkeypatch.setenv(loop_focus.LOOP_ENV, "1")
    world = _world(tmp_path)
    w, _ = _seeded(second=False, ids=(f"{UID}-t1", f"{UID}-t2"))
    workspace(world, UID).extend(tuple(w.events))
    port = mind_night(world, LifeThinker(targets=(f"{UID}-t1",), notices=0))
    practice = LoopPractice()
    settings = replace(DEMO, mind_inquiry=True)
    result = _night(world, settings=settings, inquiry=port, notice=port, practice=lambda chat, night: practice)

    early = step(result.log_path, "self_predict")
    assert [c["target_id"] for c in early["committed"]] == [f"{UID}-t1"] and early["committed"][0]["claim"]
    assert early["loop"]["predictions"] == []  # committed, not yet resolved
    assert practice.seen == [(f"{UID}-t1",)]  # practice drew on the focus committed in phase 1
    prediction, = step(result.log_path, "practice")["loop"]["predictions"]
    assert prediction["prediction_id"] == early["committed"][0]["prediction_id"]  # the same bet, resolved once
    assert prediction["outcome"] == "true"


# ── the observer ─────────────────────────────────────────────────────────────


def test_the_observer_hears_each_step_start_and_end(tmp_path: Path) -> None:
    seen = Observed()
    result = _night(_world(tmp_path), observer=seen)
    for name in ("extract", "choose", "practice", "train", "gates", "promote"):
        assert seen.events.index((name, "running")) < seen.events.index((name, "ok"))
    assert seen.receipts[("extract", "ok")]["items"] == 3
    assert seen.receipts[("promote", "ok")]["promoted"] == result.checkpoint.id


def test_an_observer_that_fails_never_fails_the_night(tmp_path: Path) -> None:
    result = _night(_world(tmp_path), observer=Observed(fail=True))
    assert result.status == "promoted"


def test_every_other_night_has_no_judge_step(tmp_path: Path) -> None:
    world = _world(tmp_path)
    result = run_night_batch_with(world.deps(homework=Homework(world)), {UID: NIGHT})[UID]
    names = [record["step"] for record in steps(result.log_path)]
    assert not {"homework", "self_predict", "refine"} & set(names)
    assert names.index("note") > names.index("promote") and "fast_gate" not in step(result.log_path, "gates")
    assert step(result.log_path, "probes")["status"] == "ok"
