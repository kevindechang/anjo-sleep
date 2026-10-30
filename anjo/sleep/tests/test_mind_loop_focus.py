# mypy: disallow-untyped-decorators=False
"""The MIND loop (owner decision 2026-10-31): her questions choose what practice rehearses, and the
night's own practice record decides what her prediction about herself came to.

Fakes only, no network. Off (``ANJO_SLEEP_MIND_LOOP`` unset or 0) the night is today's: the same
scene prompt, the same scenes and the same summary.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.panel import night_record
from anjo.sleep.demo.timeline import TimelineSources, timeline_view
from anjo.sleep.mind import loop_focus, self_predict
from anjo.sleep.mind import week
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry import loop as inquiry_loop
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import MemoryWorkspaceLog
from anjo.sleep.mind.inquiry.loop_mark import LOOP_DECIDER
from anjo.sleep.night import mind as night_mind
from anjo.sleep.night.deps import NightSettings
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.practice.run import PracticeRun
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.tests.inquiry_fakes import (
    FakeResolver,
    Ws,
    day,
    ev,
    night,
    noon,
    notice_op,
    ports,
    ref,
    thread,
)
from anjo.sleep.tests.mind_night_fakes import LifeThinker, mind_night, workspace
from anjo.sleep.tests.night_fakes import NIGHT, World, step, summary, thread as night_thread
from anjo.sleep.types import Target

UID = "loop-life"
ON = NightSettings(token_cap=16384, min_new_examples=1, mind_inquiry=True)
LOOP_FILES = ("anjo/sleep/mind/loop_focus.py", "anjo/sleep/mind/self_predict.py",
              "anjo/sleep/mind/inquiry/loop_mark.py", "anjo/sleep/night/mind.py", "anjo/sleep/night/run.py")
NIGHT_START = date(2026, 10, 3)  # the day after her log begins


def _end(start: date) -> datetime:
    """The end of ``start``'s day, the night's clock."""
    return datetime.combine(start + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)


# ── a workspace she would have at a night's start ────────────────────────────


def _seeded(*, second: bool = True, ids: tuple[str, str] = ("t-1", "t-2")) -> tuple[Ws, tuple[Target, ...]]:
    """Her log with a question on the first target (no hypotheses yet: the top of the controller's
    ranking) and, with ``second``, a lower one on the second; and the open targets both name."""
    first, last = ids
    item = ev("e-1", "The kitchen can't take an 85 mm lens.", n=1, target=first)
    seen = ev("e-2", "The courtyard is locked on Thursday.", n=1, target=first)
    other = ev("e-3", "The grant covers the rental.", n=1, target=last)
    w = Ws()
    w.start(1)
    w.arrive(item, seen, other)
    w.ok(notice_op([item.id, seen.id], kind="gap", targets=(first,), subject="the kitchen shoot"), at=night(1))
    if second:
        w.ok(notice_op([other.id], kind="gap", targets=(last,), subject="the rental plan"), at=night(1))
        questioned = w.ws.questions[-1].id
        w.ok(it.HypothesizeOp(questioned, (it.Alternative("the grant covers it", 0.4),
                                           it.Alternative("it doesn't", 0.4)), 0.2), at=night(1))
        ids = [h.id for h in w.ws.hypotheses_of(questioned)]
        w.ok(it.PredictOp(questioned, "She says the rental is paid",
                          {ids[0]: 0.7, ids[1]: 0.2, ids[2]: 0.5}, 2), at=night(1))
    return w, (thread(first, "the kitchen shoot"), thread(last, "the rental plan"))


def _focus(events: Any, targets: tuple[Target, ...], *, start: date) -> tuple[str, ...]:
    return tuple(target.target_id for target in loop_focus.focus_targets(
        events, start, targets=targets, as_of=_end(start)).targets)

def _scene(index: int, *, focus: bool, target_id: str | None, kept: bool = True,
           target: str = "glimmer") -> PracticeTranscript:
    return PracticeTranscript(
        id=f"pr-{index}", person_id=UID, night=NIGHT_START, scene=index, message=f"message {index}",
        reply="a reply", context=None, f_ids=(), ancestry=frozenset(), created_at=noon(2),
        target=target, focus=focus, target_id=target_id, flagged=not kept)


def _record(kept: int, *, scenes: int = 3, other_kept: int = 0, other: int = 3) -> self_predict.Record:
    """A night's practice record: ``kept`` of its focus scenes held her reply, and the rest."""
    scenes_ = [_scene(index, focus=True, target_id="t-1", kept=index < kept) for index in range(scenes)]
    scenes_ += [_scene(100 + index, focus=False, target_id=None, kept=index < other_kept) for index in range(other)]
    return self_predict.record(NIGHT_START, scenes_)



def _commit(log: MemoryWorkspaceLog, w: Ws, targets: tuple[Target, ...], *,
            history: tuple[dict[str, Any], ...] = ()) -> self_predict.Commitment:
    """Tonight's commitment for every focus target, as the night makes it before practice."""
    focus = tuple((target.id, target.label, w.ws.questions[0].id) for target in targets if target.id != "t-2")
    return self_predict.commit(log, kernel.fold(log.events()), night=NIGHT_START, at=night(1),
                               focus=focus, history=history)


# ── her focus comes from her state ───────────────────────────────────────────


def test_focus_follows_the_workspace_it_is_read_from() -> None:
    first, targets = _seeded(second=False)
    second, _ = _seeded(second=True)
    assert _focus(first.events, targets, start=NIGHT_START) == ("t-1",)
    # The same targets, another workspace: both her questions' targets are in focus.
    assert _focus(second.events, targets, start=NIGHT_START) == ("t-1", "t-2")


def test_focus_is_read_at_the_nights_start_not_from_tonight() -> None:
    w, targets = _seeded(second=False)
    before = _focus(w.events, targets, start=NIGHT_START)
    # Tonight's own steps (her inquiry opening a question on t-2) don't move tonight's focus.
    w.start(2)
    w.ok(notice_op(["e-2"], kind="gap", targets=("t-2",), subject="something new"), at=night(2))
    assert _focus(w.events, targets, start=NIGHT_START) == before == ("t-1",)


# ── the prediction: committed before practice, resolved from the record ─────


@pytest.mark.parametrize(
    ("kept", "other_kept", "verdict"),
    [(3, 0, "true"), (1, 1, "false"), (3, 3, "false")],
    ids=["her own reply held up more", "the same share", "no more than the rest"],
)
def test_the_prediction_resolves_from_the_nights_counts(kept: int, other_kept: int, verdict: str) -> None:
    w, targets = _seeded(second=False)
    log = MemoryWorkspaceLog(tuple(w.events))
    committed = _commit(log, w, targets)
    prediction, = committed.predictions
    before = kernel.fold(log.events())
    assert next(p for p in before.predictions if p.id == prediction.prediction_id).outcome is None
    assert "the kitchen shoot" in prediction.claim

    outcome, = self_predict.resolve(log, before, night=NIGHT_START, at=night(2),
                                    record=_record(kept, other_kept=other_kept),
                                    predictions=committed.predictions, history=())
    assert outcome.verdict == verdict and outcome.scenes.scenes == 3
    after = kernel.fold(log.events())
    settled = next(p for p in after.predictions if p.id == prediction.prediction_id)
    assert settled.outcome == verdict
    assert settled.resolved_by == (f"practice:{NIGHT_START.isoformat()}",)


def test_her_credence_moves_through_a_resolved_prediction() -> None:
    w, targets = _seeded(second=False)
    log = MemoryWorkspaceLog(tuple(w.events))
    committed = _commit(log, w, targets)
    before = kernel.fold(log.events())
    question = before.question(committed.predictions[0].question_id)
    assert question is not None and question.status == "active"
    was = {h.id: h.credence for h in before.hypotheses_of(question.id)}

    self_predict.resolve(log, before, night=NIGHT_START, at=night(2), record=_record(3),
                         predictions=committed.predictions, history=())
    after = kernel.fold(log.events())
    update, = [u for u in after.updates if u.question_id == question.id]
    assert update.source == "resolved_prediction"
    assert update.evidence_ids == (f"practice:{NIGHT_START.isoformat()}",)
    improves = next(h for h in after.hypotheses_of(question.id)
                    if h.statement == self_predict.improves_statement("the kitchen shoot"))
    assert improves.credence > was[improves.id]  # the claim held, so it held more


def test_a_night_too_small_to_compare_decides_nothing() -> None:
    w, targets = _seeded(second=False)
    for record, words in ((_record(1, scenes=1), "too few scenes"),
                          (_record(3, other=1, other_kept=0), "compare")):
        log = MemoryWorkspaceLog(tuple(w.events))
        committed = _commit(log, w, targets)
        outcome, = self_predict.resolve(log, kernel.fold(log.events()), night=NIGHT_START, at=night(2),
                                        record=record, predictions=committed.predictions, history=())
        assert outcome.verdict == "undetermined" and words in (outcome.why or "")


def test_the_baseline_is_the_last_night_that_practised_the_target() -> None:
    """A second night compares with the first's own counts, as the claim it committed said it would."""
    w, targets = _seeded(second=False)
    history = ({"night": (NIGHT_START - timedelta(days=1)).isoformat(), "status": "promoted",
                "mind_loop": {"focus": [{"target_id": "t-1"}], "scenes": {"t-1": {"scenes": 4, "held_up": 1}}}},)
    log = MemoryWorkspaceLog(tuple(w.events))
    committed = _commit(log, w, targets, history=history)
    prediction, = committed.predictions
    assert (NIGHT_START - timedelta(days=1)).isoformat() in prediction.claim

    outcome, = self_predict.resolve(log, kernel.fold(log.events()), night=NIGHT_START, at=night(2),
                                    record=_record(2), predictions=committed.predictions, history=history)
    assert outcome.verdict == "true"  # 2 of 3 against 1 of 4
    assert outcome.baseline == self_predict.SceneCount(4, 1)
    assert outcome.against == NIGHT_START - timedelta(days=1)

    again = _commit(log, w, targets, history=history)  # the next night keeps her one question
    assert again.predictions[0].question_id == prediction.question_id
    assert len(kernel.fold(log.events()).questions) == 2


# ── two nights: the second's focus follows the first's resolution ───────────


def _second_night(kept: int, other_kept: int = 0) -> tuple[str, ...]:
    """Night 1 resolves its prediction on t-1 and that night's inquiry parks its own question; then
    night 2's focus is read from the log as it stands."""
    w, targets = _seeded()
    assert _focus(w.events, targets, start=NIGHT_START) == ("t-1", "t-2")
    log = MemoryWorkspaceLog(tuple(w.events))
    committed = _commit(log, w, targets)
    self_predict.resolve(log, kernel.fold(log.events()), night=NIGHT_START, at=night(2),
                         record=_record(kept, other_kept=other_kept),
                         predictions=committed.predictions, history=())
    ws = kernel.fold(log.events())
    applied = kernel.apply(ws, it.Step(ws.seq, night(2), it.ParkOp(w.ws.questions[0].id)))
    assert isinstance(applied, it.Applied)
    log.append(applied.event)
    return _focus(log.events(), targets, start=NIGHT_START + timedelta(days=1))


def test_the_second_nights_focus_follows_the_first_nights_resolution() -> None:
    settled = _second_night(3)
    assert settled == ("t-2",)  # she settled t-1: it leaves her focus
    kept = _second_night(1, 1)
    assert "t-1" in kept and "t-2" in kept  # she keeps failing on it: it stays


# ── a loop failure is not the night's ───────────────────────────────────────


class BrokenFocus:
    """An inquiry port whose focus read fails, as a log on a failing disk does."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def focus(self, call: Any) -> Any:
        raise OSError("the workspace log couldn't be read")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class BrokenResolve(BrokenFocus):
    """An inquiry port whose focus and commit work and whose resolution fails."""

    def focus(self, call: Any) -> Any:
        return self.inner.focus(call)

    def self_resolve(self, *args: Any, **kwargs: Any) -> Any:
        raise OSError("the workspace log couldn't be written")


def _seeded_loop_world(root: Path) -> tuple[World, Any]:
    world = _world(root)
    w, _ = _seeded(second=False, ids=(f"{UID}-t1", f"{UID}-t2"))
    workspace(world, UID).extend(tuple(w.events))
    return world, mind_night(world, LifeThinker(targets=(f"{UID}-t1",), notices=0))


def test_a_failing_focus_read_runs_practice_as_today_and_still_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(loop_focus.LOOP_ENV, "1")
    world, inner = _seeded_loop_world(tmp_path)
    result = _night(world, inquiry=BrokenFocus(inner), notice=inner)

    assert result.status == "promoted" and world.trainer.calls  # the night trained
    record = step(result.log_path, "practice")["loop"]
    assert record["focus"] == [] and record["predictions"] == []
    assert "OSError" in (record.get("why") or "")
    assert not any(t.focus for t in world.transcripts(UID).for_night(NIGHT))


def test_a_failing_resolution_censors_her_prediction_and_still_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(loop_focus.LOOP_ENV, "1")
    world, inner = _seeded_loop_world(tmp_path)
    result = _night(world, inquiry=BrokenResolve(inner), notice=inner, practice=lambda chat, night: LoopPractice())

    assert result.status == "promoted" and world.trainer.calls  # the night trained
    record = step(result.log_path, "practice")["loop"]
    prediction, = record["predictions"]
    assert prediction["outcome"] == "undetermined" and "OSError" in (prediction.get("why") or "")
    settled = next(p for p in kernel.fold(workspace(world, UID).events()).predictions
                   if p.id == prediction["prediction_id"])
    assert settled.outcome == "censored" and not settled.pending


def test_a_kept_rerun_censors_what_the_earlier_attempt_left_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun reuses practice, so it never reaches resolution: the prediction is closed as censored
    instead of waiting for MIND's resolver to read unrelated evidence."""
    monkeypatch.setenv(loop_focus.LOOP_ENV, "1")
    world = _world(tmp_path)
    w, _ = _seeded(second=False, ids=(f"{UID}-t1", f"{UID}-t2"))
    log = workspace(world, UID)
    log.extend(tuple(w.events))
    question, = kernel.fold(log.events()).questions
    started = self_predict.commit(
        log, kernel.fold(log.events()), night=NIGHT,
        at=datetime.combine(NIGHT, datetime.min.time(), tzinfo=timezone.utc),
        focus=((f"{UID}-t1", "the sorting line", question.id),), history=())
    assert started.predictions
    world.transcripts(UID).add([PracticeTranscript(
        id="pr-kept", person_id=UID, night=NIGHT, scene=0, message="A kept brief.", reply="A kept reply.",
        context=None, f_ids=(), ancestry=frozenset(), created_at=night(1), target="glimmer")])
    inner = mind_night(world, LifeThinker(targets=(f"{UID}-t1",), notices=0))

    result = _night(world, inquiry=inner, notice=inner)

    assert result.status == "promoted" and world.trainer.calls
    settled = next(p for p in kernel.fold(log.events()).predictions if p.id == started.predictions[0].prediction_id)
    assert settled.outcome == "censored" and not settled.pending


# ── MIND's own resolver never settles the loop's prediction ─────────────────


def test_minds_inquiry_never_resolves_or_censors_a_loop_prediction() -> None:
    w, targets = _seeded(second=False)
    log = MemoryWorkspaceLog(tuple(w.events))
    committed = _commit(log, w, targets)
    prediction, = committed.predictions
    # Unrelated evidence inside the prediction's window: MIND's resolver would call it a verdict.
    item = ev("e-unrelated", "The courtyard was open after all.", n=2, target="t-1")
    ws = kernel.fold(log.events())
    applied = kernel.apply(ws, it.Step(ws.seq, noon(2), it.Retrieved(None, "new_evidence", "", (ref(item),))))
    assert isinstance(applied, it.Applied)
    log.append(applied.event)
    resolver = FakeResolver({"courtyard": "true"})

    inquiry_loop.run_inquiry(kernel.fold(log.events()), ports((item,), [], log=log, resolver=resolver),
                             it.Budget(cycle_cap=2), night=day(3), targets=targets, as_of=night(3),
                             focus=prediction.question_id)

    ws = kernel.fold(log.events())
    settled = next(p for p in ws.predictions if p.id == prediction.prediction_id)
    assert settled.pending and resolver.calls == []  # neither the cycle nor the horizon touched it
    assert not any(u.source == "resolved_prediction" for u in ws.updates)

    # The night, not MIND, closes what it couldn't resolve.
    outcomes = self_predict.censor(log, ws, at=night(3), record=self_predict.record(NIGHT_START, ()),
                                   predictions=committed.predictions, why="the loop could not resolve it")
    assert outcomes[0].verdict == "undetermined" and "could not resolve" in (outcomes[0].why or "")
    closed = next(p for p in kernel.fold(log.events()).predictions if p.id == prediction.prediction_id)
    assert closed.outcome == "censored" and not closed.pending


# ── the loop's start is not a run to resume ─────────────────────────────────


def test_a_loop_start_never_displaces_the_nights_inquiry_start() -> None:
    """A loop night followed by a MIND resume on the same night: the inquiry start, its budget and
    its focus are still the ones a resume reads, and the loop's start records the current denylist."""
    w = Ws()
    w.ok(it.RunStart(day(1), night(1), "inquiry", it.FULL_RULES, "priority", "q1",
                     it.Budget(cycle_cap=5, token_cap=100), False, kernel.DENYLIST_VERSION), at=night(1))
    w.ok(it.Retrieved(None, "new_evidence", "", ()), cycle=1, focus="q1", tokens=10, at=night(1))
    loop_start = it.RunStart(day(1), night(1), "inquiry", it.FULL_RULES, "priority", None,
                             it.Budget(), False, kernel.DENYLIST_VERSION, LOOP_DECIDER)
    w.ok(loop_start, at=night(1))

    point = inquiry_loop.resume(tuple(w.events))
    assert point.start.call == "inquiry" and point.start.focus == "q1"
    assert point.budget.cycles_used == 1 and point.budget.tokens_used == 10
    assert point.focus == "q1"
    assert loop_start.denylist == kernel.DENYLIST_VERSION
    assert kernel.denylist_version(kernel.fold(w.events)) == kernel.DENYLIST_VERSION
    assert night_mind.cut_off(tuple(w.events), day(1)) is False
    assert len(week.slices(tuple(w.events))) == 1  # the loop's start adds no second night


# ── the morning probes are the evaluation, never her input ──────────────────


def test_the_loop_never_imports_the_probes_or_the_reports() -> None:
    root = Path(__file__).parents[3]
    for name in LOOP_FILES:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        modules += [node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module is not None]
        assert not [m for m in modules if m.startswith(("anjo.sleep.probes", "anjo.sleep.reports"))], name


# ── practice draws its share from her focus ─────────────────────────────────


class Scenes:
    """Spark Open's scene calls: the focus prompt answers with target-named scenes, the plain one as
    today (``night_fakes.SparkChat``'s shape)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None,
                 max_tokens: int | None = None, seed: int | None = None) -> Any:
        from anjo.sleep.tests.night_fakes import result

        content = messages[-1]["content"]
        self.prompts.append(content)
        if content.startswith("# scenes_v2"):
            scenes: Any = [{"target": "t-1", "message": f"Focus {index}: the kitchen needs a new angle."}
                           for index in range(12)]
        elif content.startswith("# scenes_v1"):
            scenes = [f"New brief {index}: a sorter for film plastics." for index in range(12)]
        else:
            raise AssertionError("an unexpected Spark Open call")
        return result(json.dumps({"scenes": scenes}))


def test_a_focus_practice_draws_its_share_from_the_target(tmp_path: Path) -> None:
    from anjo.sleep.tests.night_fakes import own

    world = World(tmp_path)
    world.evidence(UID).add([own(UID, NIGHT, 0), own(UID, NIGHT, 1)])
    chat = Scenes()
    practice = world.practice(chat, NIGHT)
    focus = (loop_focus.FocusTarget("t-1", "the kitchen shoot", "q1", 1.0,
                                    evidence=("- (lesson) the kitchen can't take an 85 mm lens.",)),)

    run = practice.run(UID, NIGHT, list(world.evidence(UID).stored(NIGHT)), None,
                       at=datetime.combine(NIGHT, datetime.min.time(), tzinfo=timezone.utc), focus=focus)

    assert len(run.transcripts) == 8 == practice.max_spark_calls * 8  # eight scenes, two calls
    drawn = [t for t in run.transcripts if t.focus]
    assert len(drawn) == int(8 * loop_focus.FOCUS_SHARE) == 4
    assert {t.target_id for t in drawn} == {"t-1"}
    assert all(t.target_id is None for t in run.transcripts if not t.focus)
    focus_prompt, plain_prompt = chat.prompts
    assert focus_prompt.startswith("# scenes_v2")
    assert "the kitchen shoot" in focus_prompt and "the kitchen can't take an 85 mm lens" in focus_prompt
    assert plain_prompt.startswith("# scenes_v1") and "COUNT" not in plain_prompt


def test_malformed_focus_scenes_leave_todays_practice_standing(tmp_path: Path) -> None:
    """A focus reply that isn't the JSON asked for is the loop's failure: practice draws today's
    scenes, so the night still has something to train on."""
    from anjo.sleep.tests.night_fakes import own, result

    world = World(tmp_path)
    world.evidence(UID).add([own(UID, NIGHT, 0), own(UID, NIGHT, 1)])

    class Malformed:
        def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *,
                     temperature: float | None = None, max_tokens: int | None = None, seed: int | None = None) -> Any:
            content = messages[-1]["content"]
            if content.startswith("# scenes_v2"):
                return result("this is not the JSON asked for")
            return result(json.dumps({"scenes": [f"New brief {index}: a sorter." for index in range(12)]}))

    practice = world.practice(Malformed(), NIGHT)
    focus = (loop_focus.FocusTarget("t-1", "the kitchen shoot", "q1", 1.0),)
    run = practice.run(UID, NIGHT, list(world.evidence(UID).stored(NIGHT)), None,
                       at=datetime.combine(NIGHT, datetime.min.time(), tzinfo=timezone.utc), focus=focus)
    assert len(run.transcripts) == 8 and not any(t.focus for t in run.transcripts)


def test_practice_records_each_focus_scene_and_its_target(tmp_path: Path) -> None:
    """The night's transcript log keeps what the record needs to decide her prediction."""
    store = World(tmp_path).transcripts(UID)
    store.add([_scene(0, focus=True, target_id="t-1")])
    kept, = store.for_night(NIGHT_START)
    assert kept.focus and kept.target_id == "t-1" and not kept.flagged
    assert not _scene(1, focus=False, target_id=None).focus


# ── the night: off is today's, on records the loop ──────────────────────────


@dataclass
class LoopPractice:
    """Practice that draws its scenes from the focus it is handed, as the real one does: three scenes
    per focus target, all keeping her own reply here, and three that keep the teacher's."""

    max_spark_calls: int = 1
    seen: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, user_id: str, night: date, evidence: Any, hw: Any, *, at: datetime,
            focus: Any = ()) -> PracticeRun:
        self.seen.append(tuple(target.target_id for target in focus))
        scenes = [_scene(index, focus=True, target_id=target.target_id)
                  for target in focus for index in range(3)]
        scenes += [_scene(100 + index, focus=False, target_id=None, target="teacher") for index in range(3)]
        return PracticeRun(
            tuple(PracticeTranscript(**{**scene.__dict__, "night": night, "created_at": at}) for scene in scenes),
            tuple(item.id for item in evidence), 0)


def _world(root: Path) -> World:
    world = World(root)
    world.targets(UID).upsert(night_thread(UID, label="the sorting line"))
    world.plan(UID, NIGHT)
    return world


def _night(world: World, **deps: Any) -> Any:
    return run_night_batch_with(world.deps(settings=ON, **deps), {UID: NIGHT})[UID]


def test_loop_off_is_the_same_scene_prompt_scenes_and_summary(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    runs = []
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv(loop_focus.LOOP_ENV, raising=False)
        else:
            monkeypatch.setenv(loop_focus.LOOP_ENV, value)
        world = _world(tmp_path / f"off-{value}")
        result = _night(world)
        record = summary(result.log_path)
        record.pop("log_path", None)
        runs.append({
            "prompts": [prompt for prompt in world.spark.prompts if prompt.startswith("# scenes")],
            "scenes": [(t.scene, t.message, t.focus, t.target_id)
                       for t in world.transcripts(UID).for_night(NIGHT)],
            "summary": record,
        })
    assert runs[0] == runs[1]  # unset and 0 are the same night
    assert runs[0]["prompts"] and all(prompt.startswith("# scenes_v1") for prompt in runs[0]["prompts"])
    assert runs[0]["scenes"] and all(not focus_scene for _, _, focus_scene, _ in runs[0]["scenes"])
    assert "mind_loop" not in runs[0]["summary"]


def test_a_loop_night_draws_from_her_focus_and_records_what_it_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(loop_focus.LOOP_ENV, "1")
    world = _world(tmp_path)
    w, _ = _seeded(second=False, ids=(f"{UID}-t1", f"{UID}-t2"))
    workspace(world, UID).extend(tuple(w.events))  # the workspace she starts the night with
    port = mind_night(world, LifeThinker(targets=(f"{UID}-t1",), notices=0))
    practice = LoopPractice()
    result = _night(world, inquiry=port, notice=port, practice=lambda chat, night: practice)

    assert practice.seen == [(f"{UID}-t1",)]  # practice was handed her own focus
    record = step(result.log_path, "practice")["loop"]
    assert record["record"] == f"practice:{NIGHT.isoformat()}"
    assert [target["target_id"] for target in record["focus"]] == [f"{UID}-t1"]
    assert record["scenes"][f"{UID}-t1"] == {"scenes": 3, "held_up": 3}
    prediction, = record["predictions"]
    assert prediction["outcome"] == "true"
    assert prediction["baseline"] == {"scenes": 3, "held_up": 0, "night": None}

    kept = summary(result.log_path)["mind_loop"]
    assert kept["focus"] == record["focus"] and kept["predictions"] == record["predictions"]
    assert [t.focus for t in world.transcripts(UID).for_night(NIGHT)].count(True) == 3

    ws = kernel.fold(workspace(world, UID).events())
    question, = [q for q in ws.questions if q.subject == self_predict.question_subject("the sorting line")]
    assert question.status == "active"
    settled = next(p for p in ws.predictions if p.question_id == question.id)
    assert settled.resolved_by == (f"practice:{NIGHT.isoformat()}",) and settled.outcome == "true"


def test_the_week_report_gets_a_short_loop_section() -> None:
    from anjo.sleep.mind.week import Moment, WeekHistory
    from anjo.sleep.reports.week import ProbeTrace, render

    resolution = Moment(
        night(2), "prediction resolved", ("q1", "p5", f"practice:{NIGHT.isoformat()}"), 12,
        "Practising the sorting line holds my own replies up in more of tonight's focus scenes "
        "than in tonight's other scenes outcome=true horizon=2026-10-29T00:00:00+00:00")
    history = WeekHistory(as_of=night(2), nights=(), moments=(resolution,), digest="digest")
    trace = ProbeTrace(prompt="p", adapter="a", model="m", seed=1, temperature=0.0, week_brief="b",
                       empty_brief="e", week_reply="r", empty_reply="s")
    text = render(history, trace)
    assert "## One loop: what she practised, and what came of it" in text
    assert "Practising the sorting line holds my own replies up" in text
    assert "morning probes are not read" in text
    assert "One loop" not in render(WeekHistory(as_of=night(2), nights=(), moments=(), digest="digest"), trace)


# ── the page shows it ───────────────────────────────────────────────────────


def test_the_timeline_and_the_panel_show_what_she_practised_and_what_came_of_it() -> None:
    focus = [{"target_id": "t-1", "label": "the kitchen shoot", "question_id": "q1"}]
    record = {
        "night": NIGHT.isoformat(), "status": "promoted", "choices": [], "gates": {},
        "mind_loop": {
            "record": f"practice:{NIGHT.isoformat()}", "focus": focus,
            "scenes": {"t-1": {"scenes": 3, "held_up": 3}},
            "predictions": [{"target_id": "t-1", "label": "the kitchen shoot", "question_id": "q3",
                             "prediction_id": "p5", "claim": "Practising the kitchen shoot holds my own replies up",
                             "outcome": "true", "scenes": {"scenes": 3, "held_up": 3}}],
        },
    }
    marked = night_record(record, [])
    assert [mark.target_id for mark in marked.practised] == ["t-1"]
    assert marked.practised[0].kept == 3 and marked.practised[0].scenes == 3

    as_of = _end(NIGHT)
    evidence = (ev("e-1", "The kitchen can't take an 85 mm lens.", n=0, target="t-1"),
                ev("e-2", "The courtyard is locked on Thursday.", n=1, target="t-1"))
    view = timeline_view(TimelineSources(events=(), evidence=evidence, labels={"t-1": "the kitchen shoot"},
                                         as_of=as_of, practised=marked.practised), HeldOutText([]))
    practice_node, = [node for node in view["items"] if node["kind"] == "practice"]
    assert practice_node["target"] == {"text": "the kitchen shoot", "withheld": False}
    assert practice_node["outcome"] == "true"

    # Her prediction and its resolution, with her exact words, come from her own workspace log.
    w, targets = _seeded(second=False)
    log = MemoryWorkspaceLog(tuple(w.events))
    committed = _commit(log, w, targets)
    self_predict.resolve(log, kernel.fold(log.events()), night=NIGHT_START, at=night(2), record=_record(3),
                         predictions=committed.predictions, history=())
    claimed = committed.predictions[0].claim
    held_view = timeline_view(TimelineSources(
        events=log.events(), evidence=evidence, as_of=_end(NIGHT_START),
        labels={"t-1": "the kitchen shoot"}, practised=marked.practised), HeldOutText([]))
    assert any(node["kind"] == "prediction" and node["detail"]["claim"]["text"] == claimed
               for node in held_view["items"])
    resolved, = [node for node in held_view["items"] if node["kind"] == "resolution"]
    assert resolved["detail"]["claim"]["text"] == claimed and resolved["outcome"] == "true"
