# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The night (Task 5): steps 1 to 11 with her chooser (Task 7a) on fakes, RF1 (a night that dies
halfway keeps yesterday's checkpoint serving, holds no note, and the next night trains from the last
promoted one), batches, the allowance and the note."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.context.brief import MAX_TOKENS, RETRY_SEED, RETRY_TEMPERATURE, SECTIONS
from anjo.sleep.context.brief_store import TrainingBriefs
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.night.deps import InquiryOutcome, MindCall, NightSettings, NoInquiry, StudyOutcome
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, spark_usd
from anjo.sleep.night.meter import WonderMeter
from anjo.sleep.night.run import BRIEF_MAX_CALLS, PROBES_WORST_USD, run_night_batch_with
from anjo.sleep.night.story import Kept, Learn, Wake, story_from
from anjo.sleep.note.hold import HELD_BACK_LINE
from anjo.sleep.stops import BudgetStop
from anjo.sleep.tests.night_fakes import (
    NIGHT,
    SOURCE_URL,
    Homework,
    World,
    own,
    step,
    steps,
    summary,
    thread,
    window,
)
from anjo.sleep.types import (
    Choice,
    ChoiceContext,
    NightResult,
    Opportunity,
    Result,
    Want,
    WantSnapshot,
)

STEP_ORDER = ["close_day", "extract", "notice", "probes", "choose", "brief", "practice", "dataset", "train",
              "gates", "promote", "note", "finish"]


def _run(world: World, nights: dict[str, date], **deps: Any) -> dict[str, NightResult]:
    return run_night_batch_with(world.deps(**deps), nights)


def _one(world: World, uid: str = "p1", night: date = NIGHT, **deps: Any) -> NightResult:
    return _run(world, {uid: night}, **deps)[uid]


def _with_thread(world: World, uid: str = "p1") -> None:
    world.targets(uid).upsert(thread(uid))


class Recording(StateChooser):
    """Her chooser, recording every snapshot and context it was given."""

    def __init__(self) -> None:
        self.snapshots: list[WantSnapshot] = []
        self.contexts: list[ChoiceContext] = []

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        self.snapshots.append(snapshot)
        return super().derive(snapshot)

    def choose(self, opportunity: Opportunity, wants: Sequence[Want], ctx: ChoiceContext) -> Choice:
        self.contexts.append(ctx)
        return super().choose(opportunity, wants, ctx)


class AlwaysNote(StateChooser):
    """Picks a note on the thread whenever a note is offered."""

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        return tuple(Want(f"share:{t.id}", "share", t.id, "hold_note", "active_thread", 0.9) for t in snapshot.targets)


# ── a whole night ─────────────────────────────────────────────────────────────


def test_night_completes_without_task_7(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world)
    assert result.status == "promoted" and result.failed_at is None
    assert world.registry.current("p1") == result.checkpoint
    choose = step(result.log_path, "choose")
    assert choose["action"] is None and choose["not_offered"] == ["understand:p1-t1"]
    assert "revisit_thread" not in choose["offered"] and "prepare_commitment" not in choose["offered"]
    assert choose["ports"]["homework"].startswith("no-op")
    assert choose["receipt"]["candidates"][0]["reason"] == "action_not_offered"
    assert [c["examples"] for c in world.trainer.calls] == [8]


def test_practice_and_training_run_without_homework(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = _one(world)
    practice = step(result.log_path, "practice")
    assert practice["scenes"] == 8 and practice["homework"] is None
    assert step(result.log_path, "train")["receipt"]["steps"] == 8


def test_minimal_practice_trains_a_night_without_task_7(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = _one(world)
    dataset = step(result.log_path, "dataset")
    assert dataset["examples"] == 8 and dataset["new"] == 8 and dataset["replay"] == 0
    assert len(world.transcripts("p1")._all()) == 8
    assert sum(1 for e in world.evidence("p1").stored(NIGHT) if e.kind == "practice") == 8


def test_night_log_holds_every_receipt(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = _one(world)
    assert [record["step"] for record in steps(result.log_path)] == STEP_ORDER
    done = summary(result.log_path)
    assert done["status"] == "promoted" and set(done["gates"]) == {"general", "collapse", "safety"}
    assert done["checkpoint"]["id"] == result.checkpoint.id  # type: ignore[union-attr]
    assert done["cost_parts"]["training"] == 0.5  # 12 GPU-minutes at $2.50 an hour
    assert done["cost_usd"] == result.cost_usd > 0
    assert done["steps"][-1] == "finish" and "wonder_calls" in done
    train = step(result.log_path, "train")
    assert train["receipt"]["gpu_minutes"] == 12.0 and train["adapter"] == result.checkpoint.id  # type: ignore[union-attr]


def test_every_night_log_step_says_new_or_existing(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = _one(world)
    for record in steps(result.log_path):
        assert record["uses"] in ("new", "existing")
        assert (record["uses"] == "existing") == (record["system"] is not None)
    assert step(result.log_path, "gates")["system"] == "crisis check"
    assert step(result.log_path, "extract")["system"] == "history store"
    assert step(result.log_path, "note")["system"] == "outreach"
    assert step(result.log_path, "notice")["noop"] is True


def test_probes_registered_before_dataset_built(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _one(world)
    order = [event[0] for event in world.events]
    assert order.index("probes") < order.index("decontam")
    assert next(e for e in world.events if e[0] == "decontam")[2] == 1  # the sealed probe was checked


# ── RF1 ───────────────────────────────────────────────────────────────────────


def test_failed_step_keeps_previous_checkpoint_serving(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    first = _one(world)
    world.plan("p1", NIGHT + timedelta(days=1))
    world.train_fails["p1"] = RuntimeError("the GPU went away")
    second = _one(world, night=NIGHT + timedelta(days=1))
    assert second.status == "failed" and second.failed_at == "train" and second.checkpoint is None
    assert world.registry.current("p1") == first.checkpoint
    assert step(second.log_path, "train")["status"] == "failed"
    assert [r["step"] for r in steps(second.log_path)][-1] == "finish"


def test_failed_night_queues_no_note(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    world.train_fails["p1"] = RuntimeError("boom")
    result = _one(world, chooser=AlwaysNote())
    assert result.status == "failed" and world.notifier.queued == [] and world.writer.calls == []
    assert "note" not in [r["step"] for r in steps(result.log_path)]


def test_budget_stop_is_a_failed_night(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    first = _one(world)
    world.plan("p1", NIGHT + timedelta(days=1))
    world.train_fails["p1"] = BudgetStop("colab_cu can't hold this training's worst case")
    second = _one(world, night=NIGHT + timedelta(days=1))
    assert second.status == "failed" and second.failed_at == "train"
    assert "BudgetStop" in step(second.log_path, "train")["error"]
    assert world.registry.current("p1") == first.checkpoint and world.notifier.queued == []


def test_next_night_trains_from_last_promoted(tmp_path: Path) -> None:
    world = World(tmp_path)
    nights = [NIGHT + timedelta(days=i) for i in range(4)]
    for night in nights:
        world.plan("p1", night)
    first = _one(world, night=nights[0])
    world.failing_gates["p1"] = {"collapse"}
    rejected = _one(world, night=nights[1])
    world.failing_gates.clear()
    world.train_fails["p1"] = RuntimeError("cut off mid-training")
    _one(world, night=nights[2])
    world.train_fails.clear()
    _one(world, night=nights[3])
    prevs = [call["prev"] for call in world.trainer.calls]
    assert prevs[0] is None
    assert prevs[1] == first.checkpoint and prevs[2] == first.checkpoint and prevs[3] == first.checkpoint
    assert rejected.status == "rejected" and world.registry.history("p1")[0] == first.checkpoint
    names = [call["name"] for call in world.trainer.calls]
    assert len(set(names)) == 4  # a name is never reused


def test_rejected_night_records_gate_and_reason(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    world.failing_gates["p1"] = {"safety"}
    result = _one(world)
    assert result.status == "rejected" and result.checkpoint is None
    assert not result.gates["safety"].passed and result.gates["safety"].reason == "new reply trips on case indices 3"
    assert step(result.log_path, "promote")["rejected_by"] == ["safety"]
    assert summary(result.log_path)["gates"]["safety"]["reason"] == "new reply trips on case indices 3"
    assert world.registry.last_promoted("p1") is None


# ── cadence ───────────────────────────────────────────────────────────────────


def test_nothing_new_skips_training_but_still_offers_the_note(tmp_path: Path) -> None:
    world = World(tmp_path)
    _with_thread(world)
    result = _one(world, chooser=AlwaysNote())
    assert result.status == "nothing_new" and world.trainer.calls == []
    assert [r["status"] for r in steps(result.log_path) if r["step"] in ("practice", "train", "gates")] == ["skipped"] * 3
    assert step(result.log_path, "note")["outcome"] == "held" and len(world.notifier.queued) == 1


def test_below_min_new_examples_skips_practice_and_training(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT, count=2)
    settings = NightSettings(token_cap=16384, min_new_examples=3)
    result = _one(world, settings=settings)
    assert result.status == "nothing_new" and world.trainer.calls == [] and world.glimmer.calls == 0
    assert step(result.log_path, "extract")["new_since_last_training"] == 2
    assert step(result.log_path, "brief")["folded"] == 2  # the brief still folds


def test_skipped_evidence_carries_to_next_training_night(tmp_path: Path) -> None:
    world = World(tmp_path)
    settings = NightSettings(token_cap=16384, min_new_examples=3)
    world.plan("p1", NIGHT, count=2)
    world.plan("p1", NIGHT + timedelta(days=1), count=1)
    quiet = _one(world, settings=settings)
    trained = _one(world, night=NIGHT + timedelta(days=1), settings=settings)
    assert quiet.status == "nothing_new" and trained.status == "promoted"
    assert step(trained.log_path, "extract")["new_since_last_training"] == 3
    assert step(trained.log_path, "practice")["f_items"] == 3  # yesterday's two are in F tonight


def test_hold_note_choice_still_runs_practice_and_training(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, chooser=AlwaysNote())
    assert step(result.log_path, "choose")["action"] == "hold_note"
    assert result.status == "promoted" and len(world.trainer.calls) == 1
    assert step(result.log_path, "note")["outcome"] == "held"


# ── the note ──────────────────────────────────────────────────────────────────


def test_note_homework_facts_cite_sources(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, homework=Homework(world))
    assert step(result.log_path, "choose")["action"] == "revisit_thread"
    note = step(result.log_path, "note")
    assert note["outcome"] == "held" and note["second_offer"] is True and note["sources"] == 1
    [(uid, text, source)] = world.notifier.queued
    assert SOURCE_URL in text and source == "sleep_note"
    assert world.writer.calls[0][1][0].source == SOURCE_URL
    assert all(r.consumed for r in world.targets("p1").results())  # the note settled its target


def test_rejected_night_keeps_homework_readable(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    world.failing_gates["p1"] = {"general"}
    result = _one(world, homework=Homework(world))
    assert result.status == "rejected"
    homework = [e for e in world.evidence("p1").stored(NIGHT) if e.kind == "homework"]
    assert [e.source for e in homework] == [SOURCE_URL]
    assert [r.homework_id for r in world.targets("p1").results()] == [homework[0].id]
    [(_, text, _)] = world.notifier.queued
    assert text.endswith(HELD_BACK_LINE) and SOURCE_URL in text


def test_no_reason_no_note(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, homework=Homework(world, verified=0))
    note = step(result.log_path, "note")
    assert note["outcome"] == "no_reason" and world.notifier.queued == [] and world.writer.calls == []
    assert len(result.choices) == 1  # no second offer without a verified result


def test_writer_decline_is_recorded(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    world.writer_declines = True
    result = _one(world, homework=Homework(world))
    note = step(result.log_path, "note")
    assert note["outcome"] == "note_declined" and world.notifier.queued == []
    assert note["receipt"]["outcome"] == "declined" and note["receipt"]["outcome_reason"] == "note_declined"
    assert all(r.consumed for r in world.targets("p1").results())  # a declined note settles too


def test_at_most_one_note_per_night(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, chooser=AlwaysNote(), homework=Homework(world))
    assert len(world.notifier.queued) == 1 and len(world.writer.calls) == 1
    assert len(result.choices) == 1  # a step-4 note leaves no second offer


# ── a same-night result is news ───────────────────────────────────────────────


class SameMoment(Homework):
    """Studies as the real homework port does: its item and result stamped at the night's ``at``,
    which in real time is the window's end."""

    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.stamps: list[datetime | None] = []

    def study(self, user_id: str, target: Any, night: date, meter: WonderMeter, *, action: str = "",
              at: datetime | None = None) -> StudyOutcome:
        self.stamps.append(at)
        done = super().study(user_id, target, night, meter, action=action, at=at)
        assert at is not None and done.result is not None
        return replace(done, result=replace(done.result, created_at=at),
                       evidence=tuple(replace(item, created_at=at) for item in done.evidence))


def _said_at_the_end(world: World, night: date, uid: str = "p1") -> None:
    """Her day as the history source stamps it in real time: every item at the window's end."""
    world.planned[(uid, night)] = [replace(own(uid, night, i), created_at=window(night)[1]) for i in range(3)]


def _paid_an_hour_ago(world: World, uid: str = "p1") -> None:
    """An earlier run whose three paid actions fall inside the rolling 24 hours before ``NIGHT``'s end."""
    folder = world.logs(uid) / (NIGHT - timedelta(days=1)).isoformat()
    folder.mkdir(parents=True)
    ended = window(NIGHT)[1] - timedelta(hours=1)
    (folder / "summary.json").write_text(json.dumps({"night_end": ended.isoformat(), "paid_actions": 3}))


def _share(note: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """The second offer's candidates, as (want, status, reason)."""
    record = note.get("receipt") or note["second_offer_receipt"]
    return [(c["want_id"], c["status"], c["reason"]) for c in record["candidates"]]


@pytest.mark.parametrize(("gate", "deps", "outcome", "share"), [
    ("none", {}, "held", ("won", None)),
    ("allowance", {"settings": NightSettings(token_cap=16384, min_new_examples=1, rolling_actions=1)}, "no_reason",
     ("held_back", "budget")),
    ("cadence", {}, "no_reason", ("held_back", "budget")),  # the rolling 24 hours' paid actions
    ("crisis", {"note_check": lambda text: False}, "note_declined", ("won", None)),
])
def test_a_same_night_study_stamped_with_her_last_words_is_news_under_every_gate(
    tmp_path: Path, gate: str, deps: dict[str, Any], outcome: str, share: tuple[str, str | None],
) -> None:
    """Behaviour change (``>=`` in the news check): in real time her day and the study's result are
    both stamped at the window's end, and the result is news, so a note follows when every other
    gate allows it. On the trunk (``>``) the share want never rose and no note followed."""
    world = World(tmp_path)
    _said_at_the_end(world, NIGHT)
    _with_thread(world)
    if gate == "cadence":
        _paid_an_hour_ago(world)
    homework = SameMoment(world)
    result = _one(world, homework=homework, **deps)
    assert homework.stamps == [window(NIGHT)[1]]
    [found] = world.targets("p1").results()
    assert found.created_at == window(NIGHT)[1]  # the same moment as her last words
    assert step(result.log_path, "choose")["action"] == "revisit_thread"
    note = step(result.log_path, "note")
    assert note["outcome"] == outcome and _share(note) == [("share:p1-t1", *share)]
    assert len(world.notifier.queued) == (outcome == "held")


@pytest.mark.parametrize(("wrote", "outcome", "queued"), [(False, "held", 1), (True, "no_reason", 0)],
                         ids=["quiet since", "wrote again"])
def test_an_earlier_nights_result_stops_being_news_once_she_writes_again(
    tmp_path: Path, wrote: bool, outcome: str, queued: int,
) -> None:
    world = World(tmp_path)
    _said_at_the_end(world, NIGHT)
    _with_thread(world)
    tight = NightSettings(token_cap=16384, min_new_examples=1, rolling_actions=1)  # the study, then no note
    _one(world, settings=tight, homework=SameMoment(world))
    [left] = world.targets("p1").results()
    assert not left.consumed and world.notifier.queued == []
    after = NIGHT + timedelta(days=1)
    if wrote:
        _said_at_the_end(world, after)
    result = _one(world, night=after)
    assert step(result.log_path, "note")["outcome"] == outcome and len(world.notifier.queued) == queued


# ── the allowance ─────────────────────────────────────────────────────────────


def test_second_offer_gets_allowance_after_actual_calls(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    chooser = Recording()
    _one(world, chooser=chooser, homework=Homework(world, calls=3))
    first, second = (ctx.allowance for ctx in chooser.contexts)
    assert (first.actions_left, first.calls_left) == (2, 12) and first.usd_left == 3.0
    assert (second.actions_left, second.calls_left) == (1, 9) and abs(second.usd_left - 2.7) < 1e-9
    assert chooser.contexts[1].allowed_actions == frozenset({"hold_note"})


def test_retries_and_failed_calls_are_debited(tmp_path: Path) -> None:
    class Flaky(Homework):
        def study(self, user_id: str, target: Any, night: date, meter: WonderMeter, **kw: Any) -> Any:
            for _ in range(2):  # two calls fail, and are retried
                meter.require()
                meter.debit(0.05)
            return super().study(user_id, target, night, meter)

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, homework=Flaky(world))
    study = step(result.log_path, "choose")["study"]
    assert study["wonder_calls"] == 3 and abs(study["wonder_usd"] - 0.2) < 1e-9
    assert summary(result.log_path)["wonder_calls"] == 3


def test_a_study_that_runs_out_of_wonder_records_budget(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    settings = NightSettings(token_cap=16384, min_new_examples=1, homework_call_cap=2)
    result = _one(world, settings=settings, homework=Homework(world, calls=3))
    choose = step(result.log_path, "choose")
    assert choose["study"]["stopped"] == "budget" and choose["receipt"]["outcome"] == "budget"
    assert result.status == "promoted"  # a study's budget stop isn't the night's


def test_spent_wonder_leaves_inquire_and_note_available(tmp_path: Path) -> None:
    class Inquiry(NoInquiry):
        available = True

        def inquire(self, *args: Any) -> InquiryOutcome:
            return InquiryOutcome(result=None, cycles=3)

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    chooser = Recording()
    settings = NightSettings(token_cap=16384, min_new_examples=1, wonder_usd_month=0.0, mind_inquiry=True)
    result = _one(world, chooser=chooser, settings=settings, homework=Homework(world), inquiry=Inquiry())
    ctx = chooser.contexts[0]
    assert ctx.allowance.available_model_calls == 0
    assert ctx.allows("inquire") and ctx.allows("hold_note") and ctx.allows("revisit_thread")
    reasons = {c["want_id"]: c["reason"] for c in step(result.log_path, "choose")["receipt"]["candidates"]}
    assert reasons == {"understand:p1-t1": "budget"}  # the Wonder-backed study waits for the budget


def test_inquiry_cycles_not_debited_from_actions(tmp_path: Path) -> None:
    class Inquiry(NoInquiry):
        available = True

        def inquire(self, call: MindCall, question_id: str) -> InquiryOutcome:
            anchored = Result("r-inq", "p1-t1", None, 0, window(call.night)[1] - timedelta(minutes=5), False,
                              basis="evidence", user_cites=2)
            return InquiryOutcome(result=anchored, cycles=7, research_calls=0)

    class Inquiring(Recording):
        def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
            self.snapshots.append(snapshot)
            if len(self.snapshots) == 1:
                return (Want("understand:p1-t1", "understand", "p1-t1", "inquire", "active_thread", 0.9,
                             question_id="q1"),)
            return ()

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    chooser = Inquiring()
    settings = NightSettings(token_cap=16384, min_new_examples=1, mind_inquiry=True)
    result = _one(world, chooser=chooser, settings=settings, inquiry=Inquiry())
    assert step(result.log_path, "choose")["inquiry"] == {"cycles": 7, "research_calls": 0, "result": "r-inq"}
    assert chooser.contexts[1].allowance.actions_left == 2  # thinker cycles cost no action
    assert summary(result.log_path)["paid_actions"] == 0


# ── batches ───────────────────────────────────────────────────────────────────


def test_batch_failure_stops_only_that_person(tmp_path: Path) -> None:
    world = World(tmp_path)
    for uid in ("a", "b", "c"):
        world.plan(uid, NIGHT)
    world.extract_fails["a"] = ValueError("the reply model is not Spark Open")
    world.train_fails["b"] = RuntimeError("boom")
    results = _run(world, {"a": NIGHT, "b": NIGHT, "c": NIGHT})
    assert (results["a"].status, results["a"].failed_at) == ("failed", "extract")
    assert step(results["a"].log_path, "extract")["error"] == "ValueError"  # never another library's text
    assert (results["b"].status, results["b"].failed_at) == ("failed", "train")
    assert results["c"].status == "promoted"


def test_batch_runs_gpu_steps_for_due_people_only(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("due1", NIGHT)
    world.plan("due2", NIGHT)
    results = _run(world, {"due1": NIGHT, "quiet": NIGHT, "due2": NIGHT})
    assert world.gpu.opened == [("due1", "due2")]  # one GPU job, the due people only
    assert [c["user"] for c in world.trainer.calls] == ["due1", "due2"]
    assert {uid for uid, _ in world.gates.checked} == {"due1", "due2"}
    assert results["quiet"].status == "nothing_new" and results["due1"].status == "promoted"


def test_batch_dates_anchor_to_each_persons_night(tmp_path: Path) -> None:
    world = World(tmp_path)
    early, late = NIGHT, NIGHT + timedelta(days=9)
    world.plan("a", early)
    world.plan("b", late)
    chooser = Recording()
    windows = {"a": window(early), "b": window(late)}
    results = run_night_batch_with(world.deps(chooser=chooser), {"a": early, "b": late}, windows=windows,
                                   accelerated=True)
    extracts = {e[1]: (e[2], e[3]) for e in world.events if e[0] == "extract"}
    assert extracts == {"a": (early, windows["a"]), "b": (late, windows["b"])}
    assert [s.as_of for s in chooser.snapshots] == [windows["a"][1], windows["b"][1]]
    assert summary(results["b"].log_path)["night_end"] == windows["b"][1].isoformat()
    assert {c["night"] for c in world.trainer.calls} == {early, late}


def test_a_cancelled_night_stops_at_its_next_step(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = _one(world, cancel=lambda uid: True)
    assert result.status == "failed" and result.failed_at == "close_day"
    assert world.trainer.calls == []


def test_a_rerun_of_the_same_night_logs_apart_and_trains_a_new_name(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    world.train_fails["p1"] = RuntimeError("cut off mid-training")
    failed = _one(world)
    world.train_fails.clear()
    rerun = _one(world)
    again = _one(world)
    assert len({failed.log_path.parent, rerun.log_path.parent, again.log_path.parent}) == 3
    assert rerun.status == "promoted" and step(rerun.log_path, "extract")["new_since_last_training"] == 3
    assert [c["name"] for c in world.trainer.calls][0] != rerun.checkpoint.id  # type: ignore[union-attr]
    assert again.status == "nothing_new"  # the promoted night's evidence is in the serving weights
    assert step(rerun.log_path, "practice")["reused"] is True and world.glimmer.calls == 8  # practised once


def test_a_rejected_nights_evidence_stays_new(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    world.plan("p1", NIGHT + timedelta(days=1), count=1)
    world.failing_gates["p1"] = {"collapse"}
    _one(world)
    world.failing_gates.clear()
    nxt = _one(world, night=NIGHT + timedelta(days=1))
    assert step(nxt.log_path, "extract")["new_since_last_training"] == 4
    assert step(nxt.log_path, "dataset")["new"] == step(nxt.log_path, "dataset")["examples"]


def test_the_rolling_allowance_counts_paid_actions_on_the_night_clock(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    _one(world, homework=Homework(world))  # a study and a note: two paid actions
    chooser = Recording()
    world.targets("p1").upsert(replace(thread("p1"), id="p1-t2", label="the pitch"))
    _one(world, chooser=chooser, homework=Homework(world))  # same night again: 4 - 2 left
    assert chooser.contexts[0].allowance.actions_left == 2
    settings = NightSettings(token_cap=16384, min_new_examples=1, rolling_actions=3)
    later = Recording()
    world.targets("p1").upsert(replace(thread("p1"), id="p1-t3", label="the hire"))
    _one(world, chooser=later, settings=settings, homework=Homework(world))
    assert later.contexts[0].allowance.actions_left == 0  # 3 - (2 + 2) per 24 hours, never below 0


# ── review fixes ──────────────────────────────────────────────────────────────


class Raising:
    def write(self, user_id: str, topic: str, facts: Any) -> str | None:
        raise RuntimeError("Glimmer is down")


def test_a_note_writer_failure_keeps_the_night_promoted(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, chooser=AlwaysNote(), note_writer=Raising())
    assert result.status == "promoted" and result.checkpoint is not None and result.failed_at is None
    note = step(result.log_path, "note")
    assert note["status"] == "ok" and note["outcome"] == "note_failed" and note["note_error"] == "write: RuntimeError"
    assert world.notifier.queued == [] and summary(result.log_path)["status"] == "promoted"


def test_a_push_failure_keeps_the_night_promoted_and_the_note_queued(tmp_path: Path) -> None:
    class PushDown(type(World(tmp_path).notifier)):  # type: ignore[misc]
        def push(self, user_id: str, message: str) -> bool:
            raise ConnectionError("push service down")

    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, chooser=AlwaysNote(), notifier=PushDown(world))
    note = step(result.log_path, "note")
    assert result.status == "promoted" and note["outcome"] == "held" and note["note_error"] == "push: ConnectionError"
    assert note["pushed"] is False


def test_the_crisis_check_reads_the_final_text_with_sources(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, homework=Homework(world), note_check=lambda text: SOURCE_URL not in text)
    assert step(result.log_path, "note")["outcome"] == "note_declined" and world.notifier.queued == []


def test_a_budget_stopped_study_leaves_its_revision_unacted(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    settings = NightSettings(token_cap=16384, min_new_examples=1, homework_call_cap=2)
    _one(world, settings=settings, homework=Homework(world, calls=3))
    [target] = world.targets("p1").open_targets(window(NIGHT)[1])
    assert target.acted_revision == 0  # nothing ran to completion, so the want stays


def _tight(tmp_path: Path, **caps: float) -> Ledger:
    return Ledger(tmp_path / "budget.jsonl", caps={**CAPS, **caps})  # type: ignore[dict-item]


def test_a_pool_without_room_stops_extraction_before_any_model_call(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = _one(world, ledger=_tight(tmp_path, baseten_sleep_usd=1e-6, baseten_mind_usd=1e-6))
    assert (result.status, result.failed_at) == ("failed", "extract")
    assert "BudgetStop" in step(result.log_path, "extract")["error"]
    assert not [e for e in world.events if e[0] == "extract"]  # no model call was made


def test_every_paid_step_settles_its_hold_on_the_metered_cost(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    ledger = Ledger(tmp_path / "budget.jsonl")
    result = _one(world, chooser=AlwaysNote(), ledger=ledger)
    events = read_records(ledger.path, PLAIN)
    jobs = {e["job"].split(":")[1] for e in events if e["op"] == "reserve"}
    assert {"extract", "brief", "practice", "note"} <= jobs
    assert len([e for e in events if e["op"] == "reserve"]) == len([e for e in events if e["op"] in ("settle", "release")])
    brief = next(e for e in events if e["op"] == "settle" and e["job"].startswith("night:brief:"))
    assert brief["amount"] == pytest.approx(summary(result.log_path)["cost_parts"]["brief"])


def test_a_pool_without_room_skips_the_note_but_keeps_the_night(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    _with_thread(world)
    result = _one(world, chooser=AlwaysNote(), ledger=_tight(tmp_path, modal_usd=1e-6))
    assert result.status == "promoted"  # the GPU job is NoGpu here; only the note's call is held
    assert step(result.log_path, "note")["note_error"] == "write: BudgetStop" and world.writer.calls == []


class Refusing(Ledger):
    """A ledger whose pools have no room for one step's hold."""

    def __init__(self, path: Path, refused: str) -> None:
        super().__init__(path)
        self._refused = refused

    def reserve(self, job: str, pools: Any, bound: Any) -> Any:
        return None if job.startswith(f"night:{self._refused}:") else super().reserve(job, pools, bound)


@pytest.mark.parametrize("refused", ["brief", "practice", "probes"])
def test_a_pool_without_room_stops_the_brief_practice_and_probes_before_any_call(tmp_path: Path, refused: str) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = _one(world, ledger=Refusing(tmp_path / "budget.jsonl", refused), is_synthetic=lambda uid: False)
    assert (result.status, result.failed_at) == ("failed", refused)
    assert "BudgetStop" in step(result.log_path, refused)["error"]
    made = {"brief": len([p for p in world.spark.prompts if p.startswith("# brief_v1")]),
            "practice": len([p for p in world.spark.prompts if p.startswith("# scenes_v1")]) + world.glimmer.calls,
            "probes": len([e for e in world.events if e[0] == "probes"])}[refused]
    assert made == 0  # no call was made


def _story(result: NightResult) -> Any:
    return story_from(json.loads((result.log_path.parent / "story.json").read_text(encoding="utf-8")))


def test_a_brief_that_fails_twice_keeps_yesterdays_and_the_night_goes_on(tmp_path: Path) -> None:
    world = World(tmp_path)
    yesterday, tonight = NIGHT - timedelta(days=1), NIGHT
    world.plan("p1", yesterday)
    world.plan("p1", tonight)
    assert _one(world, night=yesterday).status == "promoted"
    kept = world.briefs("p1").latest()
    assert kept is not None and kept.night == yesterday
    world.spark.briefs_cut, world.spark.brief_calls = 2, []  # tonight the served brief runs away twice
    ledger = Ledger(tmp_path / "budget.jsonl")
    result = _one(world, night=tonight, ledger=ledger)

    assert (result.status, result.failed_at) == ("promoted", None)  # the night went on
    brief = step(result.log_path, "brief")
    assert brief["status"] == "failed" and brief["kept"] == yesterday.isoformat()
    assert brief["why"] == f"brief kept from {yesterday.isoformat()}" and "BriefError" in brief["error"]
    waiting = 3 + len(world.transcripts("p1").for_night(yesterday))  # tonight's items, and yesterday's practice
    assert (brief["folded"], brief["pending"]) == (0, waiting)
    assert [r["status"] for r in steps(result.log_path) if r["step"] in ("practice", "dataset", "train", "gates")] == ["ok"] * 4
    assert world.briefs("p1").records() == [kept]  # nothing written: both briefs stay yesterday's
    assert len(world.briefs("p1").unfolded(world.evidence("p1").stored(tonight))) >= waiting  # folded next night
    practiced = world.transcripts("p1").for_night(tonight)
    assert practiced and all(kept.training in (t.context or "") for t in practiced)  # practice ran on the kept brief
    trained_on = TrainingBriefs(world.briefs).served("p1")
    assert trained_on is not None and trained_on.compiled_at == kept.compiled_at

    assert [(c["max_tokens"], c["temperature"], c["seed"]) for c in world.spark.brief_calls] == [
        (MAX_TOKENS, 0.0, None), (MAX_TOKENS, RETRY_TEMPERATURE, RETRY_SEED)]  # capped, then retried once, sampled
    events = [e for e in read_records(ledger.path, PLAIN) if e["job"].startswith("night:brief:")]
    assert [e["op"] for e in events] == ["reserve", "settle"]
    assert events[0]["amount"] == pytest.approx(spark_usd(world.deps().prices, BRIEF_MAX_CALLS * SPARK_MAX_IN_TOKENS,
                                                          BRIEF_MAX_CALLS * MAX_TOKENS))  # every attempt held at its cap
    both = 2 * spark_usd(world.deps().prices, 1853, MAX_TOKENS)  # the runaway and its retry, each cut at its cap
    assert events[1]["amount"] == pytest.approx(both) == pytest.approx(summary(result.log_path)["cost_parts"]["brief"])
    assert summary(result.log_path)["brief"] == {"kept": yesterday.isoformat(), "why": brief["why"],
                                                 "error": brief["error"]}
    told = _story(result)
    assert told.brief == Kept(yesterday.isoformat(), brief["error"])
    assert isinstance(told.wake, Wake) and isinstance(told.learn, Learn)  # never "the night stopped at brief"


def test_a_first_night_without_a_brief_uses_the_empty_brief(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    world.spark.briefs_cut = 2
    result = _one(world)

    assert (result.status, result.failed_at) == ("promoted", None)
    brief = step(result.log_path, "brief")
    assert (brief["status"], brief["kept"], brief["why"]) == ("failed", "empty", "brief kept empty: none written yet")
    assert world.briefs("p1").records() == []  # no brief yet, as before her first fold: readers serve none
    assert TrainingBriefs(world.briefs).served("p1") is None
    practiced = world.transcripts("p1").for_night(NIGHT)
    assert practiced and not any(heading in (t.context or "") for t in practiced for heading in SECTIONS)
    assert _story(result).brief == Kept("empty", brief["error"])
    assert summary(result.log_path)["brief"]["kept"] == "empty"


def test_a_real_persons_probes_hold_is_settled_in_full(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    ledger = Ledger(tmp_path / "budget.jsonl")
    _one(world, ledger=ledger, is_synthetic=lambda uid: False)
    events = [e for e in read_records(ledger.path, PLAIN) if e["job"].startswith("night:probes:")]
    assert [e["op"] for e in events] == ["reserve", "settle"]
    assert events[1]["amount"] == pytest.approx(PROBES_WORST_USD) == pytest.approx(events[0]["amount"])
