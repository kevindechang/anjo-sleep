# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The MIND experiment's deciders (Task 16): only her chooser acts, every decider sees the same
state, D1 never sees her choice, and the lesions are pure replays. Offline, fakes only."""
from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.mind import choose as choose_module
from anjo.sleep.mind import deciders, expect, points, snapshot, wants
from anjo.sleep.mind.choose import StateChooser, choose
from anjo.sleep.mind.deciders import (
    DeciderInput,
    d1_prompt,
    earliest_due,
    night_picks,
    no_confidence,
    no_revision_gating,
    replay_point,
)
from anjo.sleep.mind.inquiry.types import Priority
from anjo.sleep.mind.policy import PUBLIC_DEFAULT
from anjo.sleep.mind.points import DecisionPoint, QuestionState, RawSnapshot
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night import run as night_run
from anjo.sleep.night.deps import NightDeps
from anjo.sleep.night.nightlog import NightLog
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.sim.accelerated import PERSONAS, run_accelerated
from anjo.sleep.tests.accelerated_rig import FakeAccelerated
from anjo.sleep.tests.night_fakes import NIGHT, Homework, World, own, step, summary, thread, window
from anjo.sleep.types import (
    ChoiceContext,
    NightAllowance,
    QuestionView,
    Target,
    Want,
)

PERSON = "synthetic-life"
DAY = date(2026, 11, 3)
AS_OF = datetime(2026, 11, 4, tzinfo=timezone.utc)
CTX = ChoiceContext(policy=PUBLIC_DEFAULT, allowance=NightAllowance(actions_left=2, calls_left=8, usd_left=3.0))
NOTE_ONLY = replace(CTX, allowed_actions=frozenset({"hold_note"}))
STUDIES = frozenset({"prepare_commitment", "revisit_thread"})
FOUNDER = load_person_file(PERSONAS / "founder.json").visible


def _thread(target_id: str, revision: int = 2, acted: int = 0, **changes: Any) -> Target:
    base = Target(
        id=target_id, person_id=PERSON, kind="thread", label=f"label of {target_id}", check_after=None,
        status="open", revision=revision, acted_revision=acted, confidence=None, last_mentioned=DAY,
    )
    return replace(base, **changes)


def _commitment(target_id: str, check_after: datetime, **changes: Any) -> Target:
    base = Target(
        id=target_id, person_id=PERSON, kind="commitment", label=f"plan {target_id}", check_after=check_after,
        status="open", revision=1, acted_revision=1, confidence=None, last_mentioned=DAY,
    )
    return replace(base, **changes)


def _raw(*targets: Target, questions: tuple[QuestionState, ...] = ()) -> RawSnapshot:
    return RawSnapshot(targets=targets, results=(), as_of=AS_OF, last_message_at=AS_OF - timedelta(hours=6),
                       questions=questions)


def _point(raw: RawSnapshot, *, phase: Any = "pre_study", ctx: ChoiceContext = CTX, d0: bool = True) -> DecisionPoint:
    choice = choose("away", wants.derive_wants(points.build(raw)), ctx) if d0 else None
    return DecisionPoint(night=DAY, phase=phase, raw=raw, ctx=ctx, choice=choice)


def _labels(raw: RawSnapshot) -> dict[str, str]:
    return {t.id: t.label for t in raw.targets}


def _question(question_id: str, target_id: str, status: Any, *, u: float = 0.5) -> QuestionState:
    parts = Priority(question_id, stake=1.0, u=u, a=1.0, ready=False)
    view = QuestionView(id=question_id, target_ids=(target_id,), status=status, priority=parts.value, continues=False)
    return QuestionState(view=view, parts=parts)


class FakeChat:
    """Spark Open for D1: one canned JSON answer, every prompt and call kept."""

    def __init__(self, answer: dict[str, Any] | str) -> None:
        self.answer = answer if isinstance(answer, str) else json.dumps(answer)
        self.prompts: list[str] = []
        self.calls: list[tuple[Any, float | None]] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, *, temperature: float | None = None) -> ChatResult:
        self.prompts.append(messages[-1]["content"])
        self.calls.append((model, temperature))
        return ChatResult(text=self.answer, tool_calls=(), model=str(model), input_tokens=120, output_tokens=12,
                          finish_reason="stop")


class StudyingHomework(Homework):
    """The night's homework fake, keeping the targets it studied."""

    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.studied: list[str] = []

    def study(self, user_id: str, target: Target, night: date, meter: Any, *, action: str = "", at: Any = None) -> Any:
        self.studied.append(target.id)
        return super().study(user_id, target, night, meter, action=action, at=at)


def _world(tmp_path: Path) -> tuple[World, StudyingHomework]:
    world = World(tmp_path)
    world.planned[("p1", NIGHT)] = [own("p1", NIGHT, i) for i in range(3)]
    world.targets("p1").upsert(thread("p1", "sorting black film"))
    world.targets("p1").upsert(replace(thread("p1", "pricing the pilot"), id="p1-t2"))
    return world, StudyingHomework(world)


def _points(result_log: Path) -> list[DecisionPoint]:
    return [points.point_from_record(record) for record in NightLog(result_log.parent).points()]


# ── The spec's tests ──────────────────────────────────────────────────────────


def test_only_d0_executes(tmp_path: Path) -> None:
    world, homework = _world(tmp_path)
    result = run_night_batch_with(world.deps(homework=homework), {"p1": NIGHT})["p1"]
    assert homework.studied == ["p1-t1"]  # her chooser's pick ran, and only it
    pre, post = _points(result.log_path)
    assert (pre.phase, post.phase) == ("pre_study", "post_result")
    store = world.targets("p1")
    digest, queued, evidence = store.digest(), list(world.notifier.queued), len(world.evidence("p1").stored(NIGHT))
    labels = {t.id: t.label for t in store.targets()}
    picks = night_picks(
        replay_point(pre, labels, chat=FakeChat({"action": "study", "target_id": "p1-t2"})),
        replay_point(post, labels, chat=FakeChat({"action": "note", "target_id": "p1-t2"}), lesions={}),
    )
    assert picks.study["D0"] == "p1-t1" and picks.study["D1"] == "p1-t2"  # D1 would have studied another
    assert picks.notes["D1"] is not None and picks.notes["D1"].target_id == "p1-t2"
    # replaying every decider changed nothing: no study, note, result or mark beyond what D0 ran
    assert homework.studied == ["p1-t1"] and world.notifier.queued == queued
    assert store.digest() == digest and len(world.evidence("p1").stored(NIGHT)) == evidence
    imported = {
        node.module or "" for node in ast.walk(ast.parse(Path(deciders.__file__).read_text()))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith(("anjo.sleep.night", "anjo.sleep.homework", "anjo.sleep.note")) for name in imported)


def test_d1_never_sees_d0_choice() -> None:
    raw = _raw(_thread("t1", revision=4), _thread("t2", revision=2))
    picked = _point(raw)
    assert picked.choice is not None and picked.choice.target_id == "t1"
    other_want = Want(id="understand:t2", kind="understand", target_id="t2", action="revisit_thread",
                      priority_class="active_thread", strength=0.9)
    other = replace(picked, choice=choose("away", (other_want,), CTX))
    silent = replace(picked, choice=None)
    chat = FakeChat({"action": "study", "target_id": "t2"})
    replays = [replay_point(point, _labels(raw), chat=chat) for point in (picked, other, silent)]
    assert chat.prompts[0] == chat.prompts[1] == chat.prompts[2]  # her choice changes nothing D1 reads
    assert "choice" not in {field.name for field in dataclasses.fields(DeciderInput)}
    assert "understand:t1" not in chat.prompts[0] and "D0" not in chat.prompts[0]
    assert chat.calls == [(SPARK_OPEN, 0.0)] * 3
    assert all(r.d1 is not None and r.d1.target_id == "t2" for r in replays)
    assert [r.choices["D0"] for r in replays] == [picked.choice, other.choice, None]


def test_deciders_get_identical_snapshot_and_allowance() -> None:
    raw = _raw(_thread("t1", revision=3), _commitment("c1", AS_OF + timedelta(days=2)))
    ctx = replace(CTX, allowance=NightAllowance(actions_left=1, calls_left=3, usd_left=0.5),
                  allowed_actions=frozenset({"revisit_thread", "hold_note"}))
    point = _point(raw, ctx=ctx)
    seen: list[tuple[str, DeciderInput]] = []

    def spy(name: str) -> Any:
        def decide(inp: DeciderInput) -> Any:
            seen.append((name, inp))
            return choose("away", (), inp.ctx)
        return decide

    chat = FakeChat({"action": "silence", "target_id": None})
    replay = replay_point(point, _labels(raw), chat=chat, lesions={name: spy(name) for name in ("D2", "D3", "D4")})
    assert [name for name, _ in seen] == ["D2", "D3", "D4"]
    assert all(inp is replay.input for _, inp in seen)  # one input object, handed to each
    assert replay.input.raw == point.raw and replay.input.ctx == ctx and replay.input.night == DAY
    assert chat.prompts == [d1_prompt(replay.input)] and replay.input.render in chat.prompts[0]
    # the same permissions bind all of them: a study answer where only a note is offered is not chosen
    post = _point(raw, phase="post_result", ctx=replace(ctx, allowed_actions=frozenset({"hold_note"})), d0=False)
    held = replay_point(post, _labels(raw), chat=FakeChat({"action": "study", "target_id": "t1"}))
    assert held.d1 is not None and held.d1.choice.want is None
    assert [c.reason for c in held.d1.choice.considered] == ["action_not_offered"]
    assert all(c is None or c.action not in STUDIES for c in held.choices.values())
    # and the same allowance: with nothing left, nobody acts
    spent = _point(raw, ctx=replace(ctx, allowance=NightAllowance(actions_left=0, calls_left=0, usd_left=0.0)))
    broke = replay_point(spent, _labels(raw), chat=FakeChat({"action": "study", "target_id": "t1"}))
    assert all(c is None or c.want is None for c in broke.choices.values())


def test_lesions_are_pure_replays_and_serving_has_no_lesion_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    # the serving chooser keeps its signatures: there is nothing to switch
    assert list(inspect.signature(choose).parameters) == ["opportunity", "wants", "ctx"]
    assert list(inspect.signature(StateChooser.choose).parameters) == ["self", "opportunity", "wants", "ctx"]
    assert list(inspect.signature(wants.derive_wants).parameters) == ["snapshot"]
    assert list(inspect.signature(snapshot.build).parameters) == [
        "targets", "results", "as_of", "last_message_at", "intentions", "questions",
    ]
    for module in (choose_module, wants, snapshot, night_run):
        source = Path(str(module.__file__)).read_text()
        assert "lesion" not in source.lower() and "deciders" not in source, module.__name__
    assert not any("lesion" in field.name or "decider" in field.name for field in dataclasses.fields(NightDeps))
    # each lesion replays through snapshot.build (and D2, D3 through derive_wants), the same every time
    builds: list[int] = []
    derived: list[int] = []
    real_build, real_derive = snapshot.build, wants.derive_wants

    def counted_build(*args: Any, **kwargs: Any) -> Any:
        builds.append(1)
        return real_build(*args, **kwargs)

    def counted_derive(state: Any) -> Any:
        derived.append(1)
        return real_derive(state)

    monkeypatch.setattr(snapshot, "build", counted_build)
    monkeypatch.setattr(wants, "derive_wants", counted_derive)
    raw = _raw(_thread("t1", revision=3, confidence=0.9), _thread("t2", revision=1),
               _thread("t3", revision=2, acted=2))
    inp = replay_point(_point(raw), _labels(raw), chat=None, lesions={}).input
    for lesion in (no_confidence, no_revision_gating, earliest_due):
        before = len(builds)
        assert lesion(inp) == lesion(inp)
        assert len(builds) > before
    assert len(derived) >= 4
    # and they bite: D0 passes over the confident thread, D2 doesn't
    d0 = _point(raw).choice
    assert d0 is not None and d0.target_id == "t2"
    assert no_confidence(inp).target_id == "t1"
    # D3: an unchanged thread wakes at the strength of one unit of new evidence
    quiet = _raw(_thread("t3", revision=2, acted=2))
    quiet_inp = replay_point(_point(quiet), _labels(quiet), chat=None, lesions={}).input
    assert _point(quiet).choice is not None and _point(quiet).choice.want is None  # type: ignore[union-attr]
    woken = no_revision_gating(quiet_inp)
    assert woken.target_id == "t3" and woken.want is not None and woken.want.strength == pytest.approx(0.5)
    # D3: a parked question wakes without new evidence; D2 fixes its uncertainty factor at 1.0
    parked = _raw(_thread("t4", revision=1, acted=1), questions=(_question("q1", "t4", "parked", u=0.5),))
    parked_inp = replay_point(_point(parked), _labels(parked), chat=None, lesions={}).input
    assert no_revision_gating(parked_inp).question_id == "q1"
    live = _raw(_thread("t5", revision=2), questions=(_question("q2", "t5", "open", u=0.5),))
    live_inp = replay_point(_point(live), _labels(live), chat=None, lesions={}).input
    served, lesioned = _point(live).choice, no_confidence(live_inp)
    assert served is not None and served.want is not None and served.want.strength == pytest.approx(0.25)
    assert lesioned.want is not None and lesioned.want.strength == pytest.approx(0.5)
    # D4: the open commitment with the nearest date, else the open thread mentioned longest ago
    due = _raw(_thread("t6", revision=3), _commitment("c2", AS_OF + timedelta(days=5)),
               _commitment("c1", AS_OF + timedelta(days=1)))
    due_inp = replay_point(_point(due), _labels(due), chat=None, lesions={}).input
    assert (earliest_due(due_inp).target_id, earliest_due(due_inp).action) == ("c1", "prepare_commitment")
    old = _raw(_thread("t7", revision=1, acted=1, last_mentioned=DAY - timedelta(days=9)), _thread("t8", revision=3))
    old_inp = replay_point(_point(old), _labels(old), chat=None, lesions={}).input
    assert earliest_due(old_inp).target_id == "t7"


# ── The night's glue and the points ───────────────────────────────────────────


def test_inquire_counts_its_first_target() -> None:
    state = _question("q1", "t2", "open")
    two = QuestionState(view=replace(state.view, target_ids=("t2", "t1")), parts=state.parts)
    raw = _raw(_thread("t1", revision=2), _thread("t2", revision=1), questions=(two,))
    inquiry = Want(id="understand:t1", kind="understand", target_id="t1", action="inquire",
                   priority_class="active_thread", strength=0.5, question_id="q1")
    choice = choose("away", (inquiry,), CTX)
    assert choice.target_id == "t1" and points.study_target(choice, raw) == "t2"
    assert points.study_target(None, raw) is None
    note = Want(id="share:t1", kind="share", target_id="t1", action="hold_note", priority_class="active_thread",
                strength=0.7)
    assert points.study_target(choose("away", (note,), CTX), raw) is None


def test_point_records_carry_no_words_and_round_trip() -> None:
    raw = _raw(_thread("t1", revision=3, confidence=0.4), _commitment("c1", AS_OF + timedelta(days=2)),
               questions=(_question("q1", "t1", "open"),))
    point = _point(raw)
    record = points.point_record(point)
    text = json.dumps(record)
    assert "label of" not in text and "plan c1" not in text and PERSON not in text
    back = points.point_from_record(json.loads(text))
    assert back.choice == point.choice and back.ctx == point.ctx and back.night == point.night
    assert back.raw == replace(raw, targets=tuple(replace(t, label="", person_id="") for t in raw.targets))


def test_night_records_both_points_and_her_expectation(tmp_path: Path) -> None:
    world, homework = _world(tmp_path)
    coefficients = expect.Coefficients(intercept=-0.5, understand=0.4, unacted=0.3, days=-0.05, base_rate=0.3, n=12)
    result = run_night_batch_with(world.deps(homework=homework, estimator=coefficients), {"p1": NIGHT})["p1"]
    pre, post = _points(result.log_path)
    assert pre.choice is not None and pre.choice.target_id == "p1-t1"
    assert post.ctx.allowed_actions == frozenset({"hold_note"})
    assert post.choice is not None and post.choice.action == "hold_note"  # the second offer
    assert {t.id for t in pre.raw.targets} == {"p1-t1", "p1-t2"}
    [made] = world.targets("p1").expectations()
    end = window(NIGHT)[1]
    assert (made.target_id, made.made_at, made.horizon, made.outcome) == ("p1-t1", end, end + timedelta(days=7), None)
    assert made.p_needed == pytest.approx(expect.estimate(coefficients, expect.Features(True, 1, 7.0)))
    assert step(result.log_path, "choose")["expectation"] == made.id


def test_night_without_estimator_or_synthetic_person_records_less(tmp_path: Path) -> None:
    world, homework = _world(tmp_path)
    result = run_night_batch_with(world.deps(homework=homework), {"p1": NIGHT})["p1"]
    assert world.targets("p1").expectations() == []  # not registered yet: nothing is guessed
    assert step(result.log_path, "choose")["ports"]["expectations"] == "estimator not registered (Task 16)"
    assert len(_points(result.log_path)) == 2
    real, real_homework = _world(tmp_path / "real")
    done = run_night_batch_with(real.deps(homework=real_homework, is_synthetic=lambda uid: False), {"p1": NIGHT})
    assert _points(done["p1"].log_path) == []  # only a synthetic life's nights are recorded


# ── Review fixes: the night's clock, the offers, D1's study and complete records ──


def test_accelerated_night_decides_on_the_virtual_clock(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    virtual = date(2026, 11, 20)  # weeks after the wall clock the fake day is simulated on
    fake.world.targets("u1").upsert(replace(thread("u1", "sorting black film"), last_mentioned=virtual))
    homework = StudyingHomework(fake.world)
    coefficients = expect.Coefficients(intercept=-0.5, understand=0.4, unacted=0.3, days=-0.05, base_rate=0.3, n=12)
    rig = replace(fake.rig(), deps_for=lambda night: fake.world.deps(
        extractor=lambda chat: fake, homework=homework, estimator=coefficients))
    [result] = run_accelerated("u1", FOUNDER, 1, virtual, rig=rig)
    end = night_run.day_window(virtual)[1]
    assert datetime.fromisoformat(summary(result.log_path)["night_end"]) < end  # the window stays wall-clock
    pre, post = _points(result.log_path)
    assert pre.raw.as_of == end and post.raw.as_of == end
    assert homework.studied == ["u1-t1"]  # the target first seen on the virtual date is open on it
    [made] = fake.world.targets("u1").expectations()
    assert (made.made_at, made.horizon) == (end, end + timedelta(days=7))


def test_a_point_without_an_offer_allows_nothing_and_every_replay_holds_to_it(tmp_path: Path) -> None:
    world, _ = _world(tmp_path)
    result = run_night_batch_with(world.deps(), {"p1": NIGHT})["p1"]  # no homework: no result, no second offer
    pre, post = _points(result.log_path)
    assert post.choice is None and post.ctx.allowed_actions == frozenset()
    labels = _labels(post.raw)
    chat = FakeChat({"action": "note", "target_id": "p1-t1"})
    late = replay_point(post, labels, chat=chat)
    assert "Allowed now: silence." in chat.prompts[-1]
    assert all(c is None or c.want is None for c in late.choices.values())
    picks = night_picks(replay_point(pre, labels, chat=FakeChat({"action": "silence", "target_id": None})), late)
    assert picks.notes["D1"] is None  # D1 gets no contact chance D0 lacked
    # a recorded choice its own point didn't allow is refused, not scored
    want = Want(id="share:p1-t1", kind="share", target_id="p1-t1", action="hold_note",
                priority_class="active_thread", strength=0.7)
    forged = replace(post, choice=choose("away", (want,), NOTE_ONLY))
    with pytest.raises(ValueError, match="allow"):
        replay_point(forged, labels, chat=None)


def _two_targets(state: QuestionState, first: str, second: str) -> QuestionState:
    return QuestionState(view=replace(state.view, target_ids=(first, second)), parts=state.parts)


def test_d1_study_takes_an_offered_question_on_its_first_target() -> None:
    raw = _raw(_thread("t1", revision=2), _thread("t2", revision=1),
               questions=(_question("q1", "t1", "open"), _two_targets(_question("q2", "t9", "open"), "t1", "t2")))
    inquire_only = replace(CTX, allowed_actions=frozenset({"inquire", "hold_note"}))
    chat = FakeChat({"action": "study", "target_id": "t1"})
    replay = replay_point(_point(raw, ctx=inquire_only, d0=False), _labels(raw), chat=chat, lesions={})
    choice = replay.choices["D1"]
    assert choice is not None and (choice.action, choice.question_id) == ("inquire", "q1")
    assert points.study_target(choice, raw) == "t1"
    assert "Allowed now: study, note, silence." in chat.prompts[0]
    # the question comes first when both are offered; homework when no offered question starts with it
    both = replace(CTX, allowed_actions=frozenset({"inquire", "revisit_thread"}))
    first = replay_point(_point(raw, ctx=both, d0=False), _labels(raw), chat=chat, lesions={}).choices["D1"]
    assert first is not None and first.action == "inquire"
    other = FakeChat({"action": "study", "target_id": "t2"})  # q2 holds t2 second: not its first target
    homework = replay_point(_point(raw, ctx=both, d0=False), _labels(raw), chat=other, lesions={}).choices["D1"]
    assert homework is not None and homework.action == "revisit_thread"
    held = replay_point(_point(raw, ctx=inquire_only, d0=False), _labels(raw), chat=other, lesions={}).choices["D1"]
    assert held is not None and held.want is None
    parked = _raw(_thread("t1", revision=2), questions=(_question("q1", "t1", "parked"),))
    asleep = replay_point(_point(parked, ctx=inquire_only, d0=False), _labels(parked), chat=chat, lesions={})
    assert asleep.choices["D1"] is not None and asleep.choices["D1"].want is None


def test_night_picks_refuses_a_decider_seen_at_one_offer_only(tmp_path: Path) -> None:
    world, homework = _world(tmp_path)
    result = run_night_batch_with(world.deps(homework=homework), {"p1": NIGHT})["p1"]
    pre, post = _points(result.log_path)
    labels = _labels(pre.raw)
    asked = replay_point(pre, labels, chat=FakeChat({"action": "silence", "target_id": None}))
    with pytest.raises(ValueError, match="both offers"):
        night_picks(asked, replay_point(post, labels, chat=None))
    unasked = night_picks(replay_point(pre, labels, chat=None), replay_point(post, labels, chat=None))
    assert "D1" not in unasked.study and "D1" not in unasked.notes  # not observed, never silence
