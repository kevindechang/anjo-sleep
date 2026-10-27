# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's inquiry in the night (Task 17): steps 2b, 4 and 10 behind ``mind_inquiry``.

Off, the night is the trunk's to the byte (``frozen/night_mind_off.json`` was captured on the trunk
before this code existed). On, with fakes: through the night's own steps and the accelerated rig.
No model, no network."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.app.screen import SparkScreen
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.evidence.targets import EventLogTargetStore
from anjo.sleep.homework.research import WonderResearch
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.deps import MindCall, NightSettings
from anjo.sleep.night.meter import WonderMeter
from anjo.sleep.night.mind import tonight
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.ownership.delete import delete_all, erase_stored
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.provenance import may_train
from anjo.sleep.sim.accelerated import FAKE_PERSONA, FAKE_USER, PERSONAS, run_accelerated
from anjo.sleep.tests.accelerated_rig import FakeAccelerated
from anjo.sleep.tests.homework_fakes import GUIDE_URL, STUDY_URL, FakeSearch, FakeWonder, research_script, study_web
from anjo.sleep.tests.inquiry_fakes import FakeResolver, FakeScreen, Kill
from anjo.sleep.tests.mind_night_fakes import (
    RESEARCH_URL,
    STATEMENT,
    LifeThinker,
    Mumbles,
    Predicts,
    Researches,
    Retracts,
    Suspects,
    TwoQuestions,
    mind_call,
    mind_night,
    workspace,
)
from anjo.sleep.tests.night_fakes import NIGHT, Homework, World, step, steps, summary, thread
from anjo.sleep.tests.test_ownership import Ports
from anjo.sleep.types import NightResult, Target

FROZEN = Path(__file__).parent / "frozen" / "night_mind_off.json"
UID = "p1"
ON = NightSettings(token_cap=16384, min_new_examples=1, mind_inquiry=True)


def _commitment(uid: str, due: datetime) -> Target:
    return Target(id=f"{uid}-c1", person_id=uid, kind="commitment", label="the pilot demo", status="open",
                  check_after=due, revision=1, acted_revision=0, confidence=None, last_mentioned=NIGHT)


DUE_SOON = datetime.combine(NIGHT + timedelta(days=4), time(9), tzinfo=timezone.utc)  # within 7 days of the night
NIGHT_END = datetime.combine(NIGHT + timedelta(days=1), time.min, tzinfo=timezone.utc)


def _world(root: Path, *, due: datetime | None = None, nights: int = 1) -> World:
    world = World(root)
    world.targets(UID).upsert(thread(UID))
    if due is not None:
        world.targets(UID).upsert(_commitment(UID, due))
    for n in range(nights):
        world.plan(UID, NIGHT + timedelta(days=n))
    return world


def _night(world: World, mind: Any, n: int = 0, **deps: Any) -> NightResult:
    settings = deps.pop("settings", ON)
    ported = world.deps(settings=settings, inquiry=mind, notice=mind, **deps)
    return run_night_batch_with(ported, {UID: NIGHT + timedelta(days=n)})[UID]


@dataclass
class Capturing:
    """The fake trainer, keeping each night's dataset as JSONL."""

    world: World
    data: list[str] = field(default_factory=list)

    def train(self, user_id: str, night: date, prev: Any, data: Any, name: str) -> Any:
        self.data.append(data.to_jsonl())
        return self.world.trainer.train(user_id, night, prev, data, name)


# ── Flag off: the trunk's night ──────────────────────────────────────────────


def _off_scenario(root: Path, ported: bool) -> dict[str, Any]:
    """Three nights: a thread revisited, then a commitment prepared, then nothing (the frozen run)."""
    world = World(root)
    world.targets(UID).upsert(thread(UID))
    thinker = LifeThinker(targets=(f"{UID}-t1",))
    mind = mind_night(world, thinker)
    out: list[dict[str, Any]] = []
    for n in range(3):
        night = NIGHT + timedelta(days=n)
        world.plan(UID, night, count=3 - n)
        if n == 1:
            world.targets(UID).upsert(Target(
                id=f"{UID}-c1", person_id=UID, kind="commitment", label="the pilot demo", status="open",
                check_after=datetime.combine(night + timedelta(days=2), time(9), tzinfo=timezone.utc),
                revision=1, acted_revision=0, confidence=None, last_mentioned=night))
        deps = world.deps(homework=Homework(world), **({"inquiry": mind, "notice": mind} if ported else {}))
        result = run_night_batch_with(deps, {UID: night})[UID]
        record = summary(result.log_path)
        record.pop("log_path", None)
        points = [json.loads(line) for line in (result.log_path.parent / "mind_points.jsonl").read_text().splitlines()]
        out.append({"steps": steps(result.log_path), "summary": record, "points": points})
    assert thinker.prompts == []  # MIND's ports alone never run it
    store = world.evidence(UID)
    data = {"nights": out, "evidence": sorted(item.id for item in store.stored(NIGHT + timedelta(days=5))),
            "targets": world.targets(UID).export(), "queued": [list(q) for q in world.notifier.queued]}
    normal: dict[str, Any] = json.loads(json.dumps(data, sort_keys=True, default=str))
    return normal


@pytest.mark.parametrize("ported", [False, True], ids=["no-op ports", "mind ports, flag off"])
def test_flag_off_night_is_unchanged(tmp_path: Path, ported: bool) -> None:
    assert _off_scenario(tmp_path, ported) == json.loads(FROZEN.read_text(encoding="utf-8"))


# ── A life through the accelerated rig ───────────────────────────────────────


def test_life_notices_inquires_under_caps_and_shares_only_with_an_anchor(tmp_path: Path) -> None:
    start = date(2026, 10, 1)  # before the rig's wall clock, as every accelerated life is
    fake = FakeAccelerated(tmp_path, mind_from=start)  # the commitment is due nine days after night 1
    person = load_person_file(PERSONAS / f"{FAKE_PERSONA}.json").visible
    results = run_accelerated(FAKE_USER, person, 3, start, rig=fake.rig(sha="0" * 40))
    assert [r.status for r in results] == ["promoted"] * 3
    first, second, third = (r.log_path for r in results)

    assert step(first, "notice")["questions_opened"] == 1  # step 2b opened it, and acted on nothing
    choose = step(first, "choose")
    assert choose["action"] == "inquire" and choose["receipt"]["outcome"] == "ran"
    inquiry = choose["inquiry"]
    assert inquiry["ops"] == {"HYPOTHESIZE": 1, "INVESTIGATE": 1, "ASSESS": 1, "CONCLUDE": 1}
    budget = inquiry["budget"]
    assert budget["cycle_cap"] == it.CYCLE_CAP and budget["token_cap"] == it.TOKEN_CAP
    assert inquiry["cycles"] == 4 == budget["cycles_used"] and budget["tokens_used"] <= it.TOKEN_CAP
    assert [d["kind"] for d in inquiry["decisions"]][-1] == "stop" and inquiry["decisions"][0]["priorities"]
    assert inquiry["consumed"] == ["fake-life-c1", "fake-life-t1"] and inquiry["artifacts"] == []

    for log in (first, second):  # the commitment is more than 7 days off: no anchor, nothing shared
        assert step(log, "note")["outcome"] == "no_reason" and "shared" not in step(log, "note")
    assert step(second, "choose")["action"] is None  # a closed question, and nothing new on its targets
    note = step(third, "note")
    assert note["outcome"] == "held" and note["target_id"] == "fake-life-c1" and len(note["shared"]) == 1
    shared = next(r for r in fake.world.targets(FAKE_USER).results() if r.id == note["shared"][0])
    assert (shared.basis, shared.homework_id, shared.consumed) == ("evidence", None, True) and shared.user_cites >= 2
    topic, facts = fake.world.writer.calls[-1]
    assert topic == "the pilot demo" and [f.text for f in facts] == [STATEMENT]
    assert len(fake.world.notifier.queued) == 1
    kept = workspace(fake.world, FAKE_USER)
    assert kept.replay_digest() == kept.digest()  # her workspace folds again from its log


def test_a_prediction_made_on_night_one_is_resolved_by_a_later_nights_evidence(tmp_path: Path) -> None:
    """A past-dated life: each night's items are created at its own clock, so night 1's prediction
    window sees the items that arrive on nights 2 and 3."""
    start = date(2026, 10, 1)
    fake = FakeAccelerated(tmp_path, mind_from=start, thinker=Predicts(targets=("fake-life-t1",)),
                           resolver=FakeResolver(verdicts={"pilot": "true"}))
    person = load_person_file(PERSONAS / f"{FAKE_PERSONA}.json").visible
    results = run_accelerated(FAKE_USER, person, 3, start, rig=fake.rig(sha="0" * 40))
    assert [r.status for r in results] == ["promoted"] * 3
    assert step(results[0].log_path, "choose")["inquiry"]["ops"].get("PREDICT") == 1
    events = workspace(fake.world, FAKE_USER).events()
    [prediction] = workspace(fake.world, FAKE_USER).workspace().predictions
    assert prediction.committed_at == datetime.combine(start + timedelta(days=1), time.min, tzinfo=timezone.utc)
    [resolved] = [e.step for e in events if isinstance(e.step.entry, it.Resolved)]
    assert isinstance(resolved.entry, it.Resolved) and resolved.entry.outcome == "true"
    assert resolved.at > prediction.committed_at and {ref.day for ref in resolved.entry.seen} == {start + timedelta(days=1)}
    assert all(prediction.committed_at < ref.created_at <= resolved.at for ref in resolved.entry.seen)


# ── Close marks: on a close only, read from her log ──────────────────────────


def test_close_marks_survive_a_kill_before_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = _world(tmp_path)
    thinker = LifeThinker(targets=(f"{UID}-t1",))
    marking = EventLogTargetStore.mark_acted
    killed: list[str] = []

    def kill_first(store: EventLogTargetStore, target_id: str, revision: int) -> None:
        if not killed:
            killed.append(target_id)
            raise Kill()
        marking(store, target_id, revision)

    monkeypatch.setattr(EventLogTargetStore, "mark_acted", kill_first)
    with pytest.raises(Kill):  # the question concluded; the kill came before its close marks
        _night(world, mind_night(world, thinker))
    assert world.targets(UID).get(f"{UID}-t1").acted_revision == 0  # type: ignore[union-attr]

    rerun = _night(world, mind_night(world, thinker))
    assert step(rerun.log_path, "choose")["action"] is None  # marked first, so no paid revisit of a closed topic
    thread_now = world.targets(UID).get(f"{UID}-t1")
    assert thread_now is not None and thread_now.acted_revision == thread_now.revision


def test_close_marks_survive_a_failure_after_the_close(tmp_path: Path) -> None:
    world = _world(tmp_path)
    result = _night(world, mind_night(world, TwoQuestions(targets=(f"{UID}-t1",))))
    assert step(result.log_path, "notice")["questions_opened"] == 2
    inquiry = step(result.log_path, "choose")["inquiry"]
    assert inquiry["stopped"] == "failed" and inquiry["consumed"] == [f"{UID}-t1"]  # the first closed, the second failed
    assert result.status == "promoted"


def test_a_cycle_without_a_close_marks_nothing(tmp_path: Path) -> None:
    world = _world(tmp_path, due=DUE_SOON)
    result = _night(world, mind_night(world, Mumbles(targets=(f"{UID}-t1", f"{UID}-c1"))))
    inquiry = step(result.log_path, "choose")["inquiry"]
    assert inquiry["cycles"] > 0 and inquiry["rejected"] and inquiry["consumed"] == []
    assert {t.id: t.acted_revision for t in world.targets(UID).open_targets(NIGHT_END)} == {f"{UID}-t1": 0, f"{UID}-c1": 0}


# ── Step 10: share only with an anchor ───────────────────────────────────────


def test_finding_anchor_rules_for_share(tmp_path: Path) -> None:
    # No anchor: a finding on the thread alone, with no research, never shares.
    alone = _world(tmp_path / "alone")
    result = _night(alone, mind_night(alone, LifeThinker(targets=(f"{UID}-t1",))))
    assert step(result.log_path, "choose")["inquiry"]["ops"]["CONCLUDE"] == 1
    assert step(result.log_path, "note")["outcome"] == "no_reason" and alone.targets(UID).results() == []

    # A commitment due within 7 days that it bears on, citing 2 of the person's own items: shared once.
    due = _world(tmp_path / "due", due=DUE_SOON, nights=2)
    thinker = LifeThinker(targets=(f"{UID}-t1", f"{UID}-c1"))
    result = _night(due, mind_night(due, thinker))
    note = step(result.log_path, "note")
    assert note["outcome"] == "held" and note["target_id"] == f"{UID}-c1"
    [shared] = due.targets(UID).results()
    assert (shared.basis, shared.consumed) == ("evidence", True) and shared.user_cites >= 2
    again = _night(due, mind_night(due, thinker), n=1)  # consumed: the same finding never shares twice
    assert "shared" not in step(again.log_path, "note") and len(due.targets(UID).results()) == 1

    # A verified research source: shared on the thread, the note citing the source.
    sourced = _world(tmp_path / "sourced")
    research = Researches()
    result = _night(sourced, mind_night(sourced, LifeThinker(targets=(f"{UID}-t1",), tool="research",
                                                             queries=["demo rehearsal schedules"]), research=research))
    [shared] = sourced.targets(UID).results()
    assert (shared.basis, shared.verified_sources, shared.target_id) == ("research", 1, f"{UID}-t1")
    assert shared.homework_id is not None and shared.homework_id.startswith("wr-")
    assert step(result.log_path, "note")["outcome"] == "held"
    _, facts = sourced.writer.calls[-1]
    assert {f.source for f in facts} == {RESEARCH_URL} and STATEMENT in [f.text for f in facts]


class Broken:
    """A note writer that fails: the note never goes out, so the result it would carry stays unconsumed."""

    def write(self, user_id: str, topic: str, facts: Any) -> str | None:
        raise RuntimeError("the writer is down")


def _suspected(root: Path) -> tuple[World, Any]:
    """Night 1: a suspicion shared on the commitment due soon, its note never sent."""
    world = _world(root, due=DUE_SOON)
    mind = mind_night(world, Suspects(targets=(f"{UID}-t1", f"{UID}-c1"), tokens=4000))  # its tokens end the run
    result = _night(world, mind, note_writer=Broken())
    assert step(result.log_path, "note")["outcome"] == "note_failed"
    [shared] = world.targets(UID).results()
    assert (shared.id.endswith(".suspicion"), shared.basis, shared.consumed) == (True, "evidence", False)
    return world, mind


def test_a_shared_result_lapses_with_its_retracted_finding(tmp_path: Path) -> None:
    world, mind = _suspected(tmp_path)
    second = NIGHT + timedelta(days=1)
    call = mind_call(world, UID, second, NIGHT_END + timedelta(days=1))
    assert [f.text for f in mind.facts(call, f"{UID}-c1")] == [STATEMENT]  # it stands, until...
    question = workspace(world, UID).workspace().questions[0].id
    mind_night(world, Retracts()).inquire(call, question)  # ...night 2's inquiry withdraws the finding
    assert mind.results(call) == () and mind.facts(call, f"{UID}-c1") == []

    after = _night(world, mind, n=1)  # nothing new from the person: the stale result would have been news
    assert step(after.log_path, "choose")["action"] != "hold_note" and world.notifier.queued == []


def test_a_shared_result_lapses_with_its_anchor(tmp_path: Path) -> None:
    world, mind = _suspected(tmp_path)
    due = NIGHT + timedelta(days=4)  # the commitment's date has passed by this night's end
    lapsed = mind_call(world, UID, due, datetime.combine(due + timedelta(days=1), time.min, tzinfo=timezone.utc))
    assert mind.results(lapsed) == () and mind.facts(lapsed, f"{UID}-c1") == []  # it rides along on no later note


# ── Findings without a source never train ────────────────────────────────────


def test_sourceless_finding_never_trainable(tmp_path: Path) -> None:
    world = _world(tmp_path / "sourceless", due=DUE_SOON)
    trainer = Capturing(world)
    result = _night(world, mind_night(world, LifeThinker(targets=(f"{UID}-t1", f"{UID}-c1"))), trainer=trainer)
    assert result.status == "promoted" and step(result.log_path, "choose")["inquiry"]["artifacts"] == []
    assert step(result.log_path, "note")["shared"]  # it shares on its anchor; a result, never evidence
    assert not [item for item in world.evidence(UID).stored(NIGHT) if item.id.startswith("mind-")]
    assert trainer.data and all(STATEMENT not in data for data in trainer.data)

    # A resolution citing a verified source is the night's homework artifact; provenance decides training.
    world = _world(tmp_path / "sourced")
    trainer = Capturing(world)
    thinker = LifeThinker(targets=(f"{UID}-t1",), tool="research", queries=["demo rehearsal schedules"])
    result = _night(world, mind_night(world, thinker, research=Researches()), trainer=trainer)
    [made] = step(result.log_path, "choose")["inquiry"]["artifacts"]
    item = next(i for i in world.evidence(UID).stored(NIGHT) if i.id == made)
    assert (item.kind, item.source, item.held_out) == ("homework", RESEARCH_URL, False) and STATEMENT in item.text
    assert not may_train(item.ancestry, hosted_in_training=False)  # Wonder wrote its source: hosted
    assert all(STATEMENT not in data for data in trainer.data)


# ── Research: the personal-terms screen, and the paid caps ───────────────────


def test_research_query_carries_no_personal_terms(tmp_path: Path) -> None:
    world = _world(tmp_path)

    def allowing(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        return ChatResult('{"allowed": true}', (), SPARK_OPEN, 40, 5, "stop")

    research = Researches()
    thinker = LifeThinker(targets=(f"{UID}-t1",), tool="research",
                          queries=["how Maya Lin rehearses", "demo rehearsal schedules"])
    mind = mind_night(world, thinker, research=research,
                      screen=lambda spark: SparkScreen(names=("Maya Lin",), chat=spark(allowing)))
    result = _night(world, mind)
    inquiry = step(result.log_path, "choose")["inquiry"]
    assert research.queries == ["demo rehearsal schedules"]  # the named query never left the night
    assert inquiry["rejected"] == {"domain_screen": 1} and inquiry["spark_calls"] > 0
    assert inquiry["research_offered"] and inquiry["research_calls"] == 1
    record = summary(result.log_path)
    assert (record["paid_actions"], record["wonder_calls"]) == (2, 1)  # the inquiry's research, and the note
    assert record["cost_parts"]["mind"] > 0


def test_spent_wonder_offers_no_research(tmp_path: Path) -> None:
    world = _world(tmp_path)
    research = Researches()
    thinker = LifeThinker(targets=(f"{UID}-t1",), tool="research", queries=["demo rehearsal schedules"])
    settings = NightSettings(token_cap=16384, min_new_examples=1, mind_inquiry=True, wonder_usd_month=0.0)
    result = _night(world, mind_night(world, thinker, research=research), settings=settings)
    inquiry = step(result.log_path, "choose")["inquiry"]
    assert not inquiry["research_offered"] and inquiry["rejected"].get("unavailable") and research.made == []
    assert summary(result.log_path)["paid_actions"] == 0


def test_research_that_reached_wonder_spends_an_action_when_the_inquiry_then_fails(tmp_path: Path) -> None:
    world = _world(tmp_path)

    class Refused:  # Wonder answered; storing what it found then failed
        def __init__(self, call: MindCall) -> None:
            self.call = call

        def research(self, question_id: str, query: str) -> Any:
            self.call.wonder.require()
            self.call.wonder.debit(0.05)
            raise ValueError("the store refused the item")

    thinker = LifeThinker(targets=(f"{UID}-t1",), tool="research", queries=["demo rehearsal schedules"])
    result = _night(world, mind_night(world, thinker, research=Refused))
    inquiry = step(result.log_path, "choose")["inquiry"]
    assert (inquiry["stopped"], inquiry["research_calls"]) == ("failed", 1)
    assert (summary(result.log_path)["paid_actions"], summary(result.log_path)["wonder_calls"]) == (1, 1)


def test_wonder_research_screens_its_query_and_stores_hosted_sources(tmp_path: Path) -> None:
    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", UID, codec=PLAIN, guard=NO_GUARD)
    wonder, meter = FakeWonder(research_script()), WonderMeter(12, 3.0)
    at = datetime.combine(NIGHT + timedelta(days=1), time.min, tzinfo=timezone.utc)
    port = WonderResearch(user_id=UID, night=NIGHT, at=at, store=store, meter=meter, wonder=wonder,
                          search=FakeSearch(), fetcher=study_web().fetcher(), names=("Maya Lin",))
    assert port.research("q1", "how maya sorts film") == () and wonder.calls == []
    items = port.research("q1", "optical sorting film plastics")
    assert {i.source for i in items} == {STUDY_URL, GUIDE_URL} and {i.id for i in store.stored(NIGHT)} == {i.id for i in items}
    assert all(i.kind == "homework" and not may_train(i.ancestry, hosted_in_training=False) for i in items)
    assert "(nothing yet)" in wonder.calls[0]["messages"][1]["content"]  # none of the person's evidence went out
    assert meter.calls == 3 and meter.usd > 0
    again = replace(port, wonder=FakeWonder(research_script()))
    assert again.research("q1", "optical sorting film plastics") == items  # the same sources are stored once
    assert len(store.stored(NIGHT)) == len(items)
    spent = WonderResearch(user_id=UID, night=NIGHT, at=at, store=store, meter=WonderMeter(0, 3.0), wonder=wonder,
                           search=FakeSearch(), fetcher=study_web().fetcher())
    assert spent.research("q1", "optical sorting") == ()


# ── RF2, budget stops and kill-resume ────────────────────────────────────────


def test_delete_mid_inquiry_leaves_nothing_behind(tmp_path: Path) -> None:
    world = _world(tmp_path, due=DUE_SOON)
    owned = Ports(world, tmp_path)
    deleted: list[Any] = []

    def delete(prompts: int) -> None:  # the person deletes while her second cycle thinks
        if prompts == 3 and not deleted:
            deleted.append(delete_all(UID, owned.ports()))

    thinker = LifeThinker(targets=(f"{UID}-t1", f"{UID}-c1"), hook=delete)
    result = _night(world, mind_night(world, thinker), cancel=owned.flag.watcher(),
                    purge=lambda uid: erase_stored(uid, owned.ports()))
    assert deleted and deleted[0].cancelled
    assert len(thinker.prompts) == 3  # the run stopped at its next write: no cycle after the delete
    assert (result.status, result.failed_at) == ("failed", "choose")
    for left in (world.root / UID, world.logs(UID)):
        assert not left.exists(), left  # her workspace, evidence, targets and night log are gone
    assert world.notifier.queued == [] and world.registry.last_promoted(UID) is None


def test_budget_stop_ends_mind_steps_cleanly(tmp_path: Path) -> None:
    world = _world(tmp_path, nights=2)
    thinker = LifeThinker(targets=(f"{UID}-t1",), tokens=it.TOKEN_CAP)  # the notice pass spends the night's tokens
    first = _night(world, mind_night(world, thinker))
    inquiry = step(first.log_path, "choose")["inquiry"]
    assert inquiry["cycles"] == 0 and [d["reason"] for d in inquiry["decisions"]] == ["token_cap"]

    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "baseten_mind_usd": 0.01})  # no room for a hold
    held = len(workspace(world, UID).events())
    asked = len(thinker.prompts)
    second = _night(world, mind_night(world, thinker, ledger=ledger), n=1, ledger=ledger)
    assert step(second.log_path, "notice")["stopped"] == "budget"
    choose = step(second.log_path, "choose")
    assert choose["action"] == "inquire" and choose["receipt"]["outcome"] == "budget"
    assert choose["inquiry"]["stopped"] == "budget" and choose["inquiry"]["cycles"] == 0
    assert second.status == "promoted"  # the step ended; the night went on
    assert len(workspace(world, UID).events()) == held and len(thinker.prompts) == asked


def test_a_notice_pass_killed_before_its_questions_are_logged_offers_its_items_again(tmp_path: Path) -> None:
    world = _world(tmp_path)
    thinker = LifeThinker(targets=(f"{UID}-t1",), notices=2)
    killed: list[str] = []

    def kill_at_the_screen(text: str, purpose: it.ScreenPurpose) -> bool:  # after the reply, before its ops
        if not killed:
            killed.append(purpose)
            raise Kill()
        return False

    with pytest.raises(Kill):
        _night(world, mind_night(world, thinker, screen=FakeScreen(refuse=kill_at_the_screen)))
    assert killed == ["question"] and workspace(world, UID).events() == ()  # the pass wrote nothing

    rerun = _night(world, mind_night(world, thinker))
    notice = step(rerun.log_path, "notice")
    assert notice["new"] == 3 and notice["questions_opened"] == 1


def test_a_notice_pass_that_fails_after_its_reply_still_spends_tonights_tokens(tmp_path: Path) -> None:
    world = _world(tmp_path)
    thinker = LifeThinker(targets=(f"{UID}-t1",), notices=2, tokens=1500)

    def down(text: str, purpose: it.ScreenPurpose) -> bool:  # the screen fails after the thinker replied
        raise ModelApiError("the screen's provider is down")

    failed = _night(world, mind_night(world, thinker, screen=FakeScreen(refuse=down)))
    assert step(failed.log_path, "notice")["stopped"] == "failed" and len(thinker.prompts) == 1
    start, spent = workspace(world, UID).events()  # its RunStart and its spend: no item, no question
    assert isinstance(start.step.entry, it.RunStart) and start.step.entry.call == "notice"
    assert (spent.step.entry, spent.step.code, spent.step.tokens) == (None, "malformed", 1500)
    assert tonight(workspace(world, UID).events(), NIGHT).tokens_used == 1500

    rerun = _night(world, mind_night(world, thinker))
    notice = step(rerun.log_path, "notice")
    assert notice["new"] == 3 and notice["questions_opened"] == 1  # its items are offered again
    assert notice["budget"]["tokens_used"] == 1500 + 1500 + FakeScreen().tokens  # the failed pass counts


def test_killed_inquiry_resumes_its_cut_off_tool(tmp_path: Path) -> None:
    world = _world(tmp_path)
    found = Researches()
    killed: list[bool] = []

    def research(call: MindCall) -> Any:
        port = found(call)

        class Once:
            def research(self, question_id: str, query: str) -> Any:
                if not killed:
                    killed.append(True)
                    raise Kill()
                return port.research(question_id, query)
        return Once()

    thinker = LifeThinker(targets=(f"{UID}-t1",), tool="research", queries=["demo rehearsal schedules"])
    with pytest.raises(Kill):
        _night(world, mind_night(world, thinker, research=research))
    last = workspace(world, UID).events()[-1].step
    assert isinstance(last.entry, it.InvestigateOp) and last.applied  # the kill cut its tool off

    rerun = _night(world, mind_night(world, thinker, research=research))
    assert step(rerun.log_path, "notice")["new"] == 0  # the notice pass passes nothing twice
    inquiry = step(rerun.log_path, "choose")["inquiry"]
    assert inquiry["resumed"] and inquiry["ops"]["CONCLUDE"] == 1 and found.queries == ["demo rehearsal schedules"]
    log = workspace(world, UID)
    assert log.replay_digest() == log.digest()
