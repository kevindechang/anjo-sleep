# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Her chooser (Task 7a): wants from state, one choice, content-free receipts. Offline, fakes only."""
from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep import mind
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.targets import EventLogTargetStore, Mention
from anjo.sleep.mind import receipt, snapshot
from anjo.sleep.mind.choose import StateChooser, choose
from anjo.sleep.mind.policy import PUBLIC_DEFAULT
from anjo.sleep.mind.rule_chooser import RuleChooser
from anjo.sleep.mind.wants import derive_wants
from anjo.sleep.ports import Chooser
from anjo.sleep.types import (
    Choice,
    ChoiceContext,
    Commitment,
    Evidence,
    NightAllowance,
    QuestionView,
    Result,
    Target,
    Want,
    WantSnapshot,
)

PERSON = "synthetic-film-student"
DAY = date(2026, 10, 27)
NIGHT = datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)
SPOKE = NIGHT - timedelta(hours=20)  # the person's last message
CTX = ChoiceContext(policy=PUBLIC_DEFAULT, allowance=NightAllowance(actions_left=2, calls_left=8, usd_left=3.0))
HOLD_NOTE_ONLY = dataclasses.replace(CTX, allowed_actions=frozenset({"hold_note"}))


def _thread(target_id: str = "t1", revision: int = 2, acted: int = 0, **changes: Any) -> Target:
    base = Target(
        id=target_id, person_id=PERSON, kind="thread", label=f"private label of {target_id}",
        check_after=None, status="open", revision=revision, acted_revision=acted,
        confidence=None, last_mentioned=DAY,
    )
    return dataclasses.replace(base, **changes)


def _commitment(
    target_id: str = "c1", revision: int = 1, acted: int = 0, due: bool = True, **changes: Any,
) -> Target:
    base = Target(
        id=target_id, person_id=PERSON, kind="commitment", label=f"private plan {target_id}",
        check_after=NIGHT - timedelta(days=1) if due else NIGHT + timedelta(days=90), status="open",
        revision=revision, acted_revision=acted, confidence=None, last_mentioned=DAY,
    )
    return dataclasses.replace(base, **changes)


def _result(target_id: str = "t1", sources: int = 1, **changes: Any) -> Result:
    base = Result(
        id=f"r-{target_id}-{sources}", target_id=target_id, homework_id="hw-1",
        verified_sources=sources, created_at=NIGHT - timedelta(hours=2), consumed=False,
    )
    return dataclasses.replace(base, **changes)


def _snap(
    *targets: Target,
    results: tuple[Result, ...] = (),
    as_of: datetime = NIGHT,
    last_message_at: datetime | None = SPOKE,
    questions: tuple[QuestionView, ...] = (),
    intentions: tuple[Commitment, ...] = (),
) -> WantSnapshot:
    return snapshot.build(
        list(targets), list(results), as_of, last_message_at, intentions=intentions, questions=questions,
    )


def _wants(*targets: Target, **kwargs: Any) -> tuple[Want, ...]:
    return derive_wants(_snap(*targets, **kwargs))


def _ids(wants: tuple[Want, ...]) -> list[str]:
    return [f"{w.kind}:{w.action}:{w.target_id}" for w in wants]


def _question(
    question_id: str = "q1", targets: tuple[str, ...] = ("t1",), status: Any = "open",
    priority: float = 1.2, continues: bool = True,
) -> QuestionView:
    return QuestionView(
        id=question_id, target_ids=targets, status=status, priority=priority, continues=continues,
    )


def _store(tmp_path: Path) -> EventLogTargetStore:
    return EventLogTargetStore(tmp_path / "targets.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)


_evidence_ids = itertools.count(1)


def _mention(store: EventLogTargetStore, target_id: str, kind: Any = "thread") -> None:
    """The person says something new about a target: its revision goes up by one."""
    n = next(_evidence_ids)
    store.record_mention(
        Mention(target_id=target_id, kind=kind, label="x", check_after=None, confidence=None),
        Evidence(
            id=f"e-{n}", person_id=PERSON, day=DAY, kind="lesson", text="said", source=f"message:{n}",
            held_out=False, created_at=NIGHT, target_id=target_id,
        ),
    )


def _seed(store: EventLogTargetStore, target: Target) -> None:
    """Put a target in the store at the revision and acted revision it carries, the way nights do."""
    store.upsert(dataclasses.replace(target, revision=1, acted_revision=0))
    for _ in range(target.revision - 1):
        _mention(store, target.id, target.kind)
    if target.acted_revision:
        store.mark_acted(target.id, target.acted_revision)


def _night(store: EventLogTargetStore, ctx: ChoiceContext = CTX, **kwargs: Any) -> Choice:
    """One night's step 4 against the store: snapshot, derive, choose."""
    shot = snapshot.build(
        store.open_targets(NIGHT), store.results(), NIGHT, kwargs.pop("last_message_at", SPOKE), **kwargs,
    )
    return choose("away", derive_wants(shot), ctx)


# ── Nothing, time and arrival ─────────────────────────────────────────────────


def test_nothing_open_chooses_nothing() -> None:
    closed = _thread("closed", status="closed")
    paused = _commitment("paused", status="paused")
    for targets in ((), (closed, paused)):
        choice = choose("away", _wants(*targets), CTX)
        assert choice.want is None and choice.action is None
        assert choice.considered == ()


def test_time_alone_changes_no_want() -> None:
    targets = (
        _thread("moved", revision=3, acted=1, confidence=0.3),
        _thread("still", revision=4, acted=4),
        _commitment("far", revision=1, acted=0, due=False),
        _commitment("undated", revision=1, acted=0, due=False, check_after=None),
    )
    baseline = _wants(*targets)
    assert baseline  # the fixture does pull
    for later in (timedelta(hours=1), timedelta(days=7), timedelta(days=30)):
        assert _wants(*targets, as_of=NIGHT + later) == baseline
    # the control: crossing a date the person set does strengthen the commitment
    crossed = _wants(*targets, as_of=NIGHT + timedelta(days=100))
    prepare = {w.target_id: w.strength for w in baseline if w.action == "prepare_commitment"}
    after = {w.target_id: w.strength for w in crossed if w.action == "prepare_commitment"}
    assert after["far"] > prepare["far"]
    assert after["undated"] == prepare["undated"]


def test_unchanged_target_wakes_no_want_next_night(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _thread("t1", revision=2, acted=0))
    assert [c.action for c in [_night(store)]] == ["revisit_thread"]
    store.mark_acted("t1", 2)
    for _ in range(3):  # three more nights with nothing new from the person
        assert _night(store).want is None
    _mention(store, "t1")  # a new mention
    assert _night(store).action == "revisit_thread"


def test_own_homework_never_rewakes_its_target(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _thread("t1", revision=2, acted=0, confidence=0.4))
    assert _night(store).action == "revisit_thread"
    store.mark_acted("t1", 2)
    store.add_result(_result("t1", sources=2))
    second = choose("away", derive_wants(snapshot.build(
        store.open_targets(NIGHT), store.results(), NIGHT, SPOKE)), CTX)
    assert second.action == "hold_note"  # her result is something to say, never to study again
    assert not [c for c in second.considered if c.action == "revisit_thread"]
    store.settle("t1")
    assert _night(store).want is None


# ── The follow-up path ────────────────────────────────────────────────────────


@pytest.mark.parametrize("sources", [0, 3], ids=["follow_up wins", "share wins on its result"])
def test_due_commitment_notes_once(tmp_path: Path, sources: int) -> None:
    store = _store(tmp_path)
    _seed(store, _commitment("c1", revision=1, acted=0))
    store.mark_acted("c1", 1)  # preparation done
    if sources:
        store.add_result(_result("c1", sources=sources))
    choice = _night(store)
    assert choice.action == "hold_note"
    assert choice.want is not None
    assert choice.want.kind == ("share" if sources else "follow_up")
    store.settle("c1")  # a held note settles its target, whichever want won
    assert _night(store).want is None
    assert _night(store, ctx=HOLD_NOTE_ONLY).want is None


def test_consumed_result_keeps_preparation_closed() -> None:
    commitment = _commitment("c1", revision=2, acted=1, due=False)
    assert [w.action for w in _wants(commitment)] == ["prepare_commitment"]
    consumed = _result("c1", sources=1, consumed=True)
    assert _wants(commitment, results=(consumed,)) == ()


def test_sourceless_study_waits_for_new_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _commitment("c1", revision=1, acted=0, due=False))
    assert _night(store).action == "prepare_commitment"
    store.mark_acted("c1", 1)
    store.add_result(_result("c1", sources=0))  # the study found nothing it could verify
    assert _night(store).want is None
    _mention(store, "c1", kind="commitment")  # the person says something new
    assert _night(store).action == "prepare_commitment"


def test_due_commitment_prepares_before_it_notes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _commitment("c1", revision=1, acted=0))
    wants = _wants(*store.open_targets(NIGHT))
    assert [w.action for w in wants] == ["prepare_commitment"]
    assert _night(store).action == "prepare_commitment"
    store.mark_acted("c1", 1)
    store.add_result(_result("c1", sources=1))
    assert _night(store).action == "hold_note"


def test_due_commitment_beats_open_thread() -> None:
    wants = _wants(_commitment("c1"), _thread("t9", revision=60, acted=0))
    choice = choose("away", wants, CTX)
    assert (choice.action, choice.target_id) == ("prepare_commitment", "c1")
    loser = next(c for c in choice.considered if c.target_id == "t9")
    assert (loser.status, loser.reason) == ("held_back", "lost_to_target")


def test_result_without_verified_source_is_no_change() -> None:
    thread = _thread("t1", revision=2, acted=2)
    sourceless = _result("t1", sources=0)
    shot = _snap(thread, results=(sourceless,))
    assert shot.results == ()
    assert derive_wants(shot) == ()


def test_accelerated_clock_is_virtual_everywhere() -> None:
    forbidden = re.compile(r"\.now\(|utcnow|\.today\(|time\.time|monotonic|perf_counter|os\.environ|getenv|import time")
    for path in Path(mind.__file__).parent.glob("*.py"):
        assert not forbidden.search(path.read_text()), f"{path.name} reads a clock or the environment"
    # seven virtual days on one calendar date: the pull follows as_of alone
    sim = datetime(2031, 3, 1, 2, 0, tzinfo=timezone.utc)
    plan = _commitment("c1", due=False, check_after=sim + timedelta(days=3))
    early = _wants(plan, as_of=sim)
    later = _wants(plan, as_of=sim + timedelta(days=4))
    assert early[0].strength < later[0].strength


def test_declined_note_consumes_its_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _commitment("c1", revision=1, acted=1))
    store.add_result(_result("c1", sources=2))
    choice = _night(store)
    assert choice.action == "hold_note"
    entry = receipt.record(choice, "declined")  # the writer found nothing worth saying
    assert entry["outcome"] == "declined"
    assert entry["outcome_reason"] == "note_declined"
    store.settle("c1")  # a declined note settles its target as a held one does
    assert _night(store).want is None


# ── The port, the allowed actions, the order ──────────────────────────────────


def test_disallowed_action_recorded_not_chosen() -> None:
    spent = dataclasses.replace(CTX, allowed_actions=frozenset({"hold_note"}), allowance=NightAllowance(1, 0, 0.0))
    # the study would also be over budget; the offered-ness is recorded first
    study = Want(
        id="understand:t1", kind="understand", target_id="t1", action="revisit_thread",
        priority_class="active_thread", strength=0.05,
    )
    note = Want(
        id="share:c1", kind="share", target_id="c1", action="hold_note",
        priority_class="user_commitment", strength=0.7,
    )
    choice = choose("away", (study, note), spent)
    assert choice.action == "hold_note"
    held = next(c for c in choice.considered if c.want_id == "understand:t1")
    assert (held.status, held.reason) == ("held_back", "action_not_offered")
    only = choose("away", (study,), spent)
    assert only.want is None and [c.reason for c in only.considered] == ["action_not_offered"]


def test_rule_chooser_and_chooser_share_the_port() -> None:
    def one_night(chooser: Chooser) -> Choice:
        return chooser.choose("away", chooser.derive(_snap(_thread("t1"))), CTX)

    for chooser in (RuleChooser(), StateChooser()):
        assert isinstance(chooser, Chooser)
        assert one_night(chooser).action == "revisit_thread"
    assert inspect.signature(RuleChooser.choose) == inspect.signature(StateChooser.choose)
    assert inspect.signature(RuleChooser.derive) == inspect.signature(StateChooser.derive)


def test_second_offer_allows_only_hold_note(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _thread("t1", revision=2, acted=2))
    store.add_result(_result("t1", sources=2))
    _seed(store, _thread("t2", revision=5, acted=0))  # still pulls, but it is a study
    both = _night(store, ctx=HOLD_NOTE_ONLY)
    assert both.action == "hold_note" and both.target_id == "t1"
    study = next(c for c in both.considered if c.target_id == "t2")
    assert study.reason == "action_not_offered"
    store.settle("t1")
    assert _night(store, ctx=HOLD_NOTE_ONLY).want is None


def test_order_is_independent_of_arrival() -> None:
    wants = _wants(
        _thread("a", revision=3, acted=1), _thread("b", revision=3, acted=1),
        _thread("c", revision=9, acted=0), _commitment("d", due=False),
    )
    assert len(wants) == 4
    choices = {choose("away", order, CTX) for order in itertools.permutations(wants)}
    assert len(choices) == 1
    assert choices.pop().target_id == "d"  # the class first, then strength, then id


def test_candidates_past_the_cap_are_rejected() -> None:
    wants = tuple(
        Want(
            id=f"understand:t{n:02d}", kind="understand", target_id=f"t{n:02d}", action="revisit_thread",
            priority_class="active_thread", strength=0.9 - n / 100,
        )
        for n in range(18)
    )
    choice = choose("away", wants, CTX)
    assert len(choice.considered) == 18
    assert PUBLIC_DEFAULT.max_candidates == 16
    rejected = [c for c in choice.considered if c.status == "rejected"]
    assert [c.target_id for c in rejected] == ["t16", "t17"]
    assert {c.reason for c in rejected} == {"candidate_limit"}


def test_thresholds_and_budget() -> None:
    weak = _wants(_thread("t1", revision=2, acted=1, confidence=0.9))  # 0.5 * 0.1
    choice = choose("away", weak, CTX)
    assert [(c.status, c.reason) for c in choice.considered] == [("held_back", "below_threshold")]
    spent = dataclasses.replace(CTX, allowance=NightAllowance(actions_left=2, calls_left=0, usd_left=3.0))
    study = choose("away", _wants(_thread("t1", revision=2, acted=0)), spent)
    assert [c.reason for c in study.considered] == ["budget"] and study.want is None
    inquiry = choose("away", _wants(_thread("t1", revision=2, acted=2), questions=(_question(),)), spent)
    assert inquiry.action == "inquire"  # Wonder being spent doesn't stop her thinking
    none_left = dataclasses.replace(CTX, allowance=NightAllowance(actions_left=0, calls_left=8, usd_left=3.0))
    assert choose("away", _wants(_thread("t1", revision=2, acted=2), questions=(_question(),)), none_left).want is None


# ── Receipts ──────────────────────────────────────────────────────────────────


def test_receipt_fields_are_content_free() -> None:
    wants = _wants(_commitment("c1"), _thread("t1", revision=4, acted=1, confidence=0.2))
    choice = choose("away", wants, CTX)
    entry = receipt.record(choice, "ran")
    assert set(entry) == {"opportunity", "winner", "question_id", "outcome", "outcome_reason", "candidates"}
    assert entry["outcome"] == "ran" and entry["winner"] == "follow_up:c1"
    for candidate in entry["candidates"]:
        assert set(candidate) == {"want_id", "kind", "target_id", "action", "strength", "weight", "status", "reason"}
    dumped = json.dumps(entry)
    assert "private" not in dumped
    assert receipt.record(choose("away", (), CTX), None)["winner"] is None
    with pytest.raises(ValueError, match="outcome"):
        receipt.record(choose("away", (), CTX), "ran")
    text_reason = dataclasses.replace(choice.considered[1], reason="she said she was lonely")
    leaky = dataclasses.replace(choice, considered=(choice.considered[0], text_reason))
    with pytest.raises(ValueError, match="reason"):
        receipt.record(leaky, None)


# ── Boundary ──────────────────────────────────────────────────────────────────


def test_mind_core_imports_no_private_anjo() -> None:
    for path in sorted(Path(mind.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level
                else []
            )
            for name in names:
                if name == "anjo" or name.startswith("anjo."):
                    assert name.startswith("anjo.sleep") and not name.startswith("anjo.sleep.app"), (
                        f"{path.name} imports {name}"
                    )


# ── Understanding and questions ───────────────────────────────────────────────


def test_understand_strength_scales_with_one_minus_confidence() -> None:
    def strength(confidence: float | None) -> float:
        (want,) = _wants(_thread("t1", revision=3, acted=0, confidence=confidence))
        assert (want.kind, want.action) == ("understand", "revisit_thread")
        return want.strength

    unknown, half, high = strength(None), strength(0.5), strength(0.9)
    assert unknown == pytest.approx(0.75)  # 3 / (3 + 1), no confidence yet
    assert half == pytest.approx(unknown * 0.5)
    assert high == pytest.approx(unknown * 0.1)
    assert strength(1.0) == 0.0


def test_open_question_continues_without_new_evidence() -> None:
    thread = _thread("t1", revision=3, acted=3)
    (want,) = _wants(thread, questions=(_question(status="open", priority=1.2, continues=True),))
    assert (want.action, want.question_id, want.strength) == ("inquire", "q1", pytest.approx(0.6))
    assert want.kind == "understand"
    (active,) = _wants(thread, questions=(_question(status="active", priority=2.0),))
    assert active.strength == 1.0
    assert _wants(thread, questions=(_question(continues=False),)) == ()  # its next step isn't ready
    # with new evidence it inquires whether or not it continues
    fresh = _thread("t1", revision=4, acted=3)
    (woken,) = _wants(fresh, questions=(_question(continues=False),))
    assert woken.action == "inquire"
    # a question on another target doesn't hold this one open
    assert _wants(thread, questions=(_question(targets=("other",)),)) == ()


def test_evidence_result_admitted_only_with_anchor() -> None:
    def evidence(target_id: str = "c1", cites: int = 2, **changes: Any) -> Result:
        return _result(target_id, sources=0, basis="evidence", homework_id=None, user_cites=cites, **changes)

    def admitted(target: Target, result: Result) -> bool:
        return bool(_snap(target, results=(result,)).results)

    def due_in(days: float) -> Target:
        return _commitment("c1", revision=1, acted=1, due=False, check_after=NIGHT + timedelta(days=days))

    assert admitted(due_in(3), evidence())
    assert admitted(due_in(0), evidence()) and admitted(due_in(7), evidence())  # both ends count
    assert not admitted(due_in(7.01), evidence())
    assert not admitted(due_in(-0.5), evidence())  # already past
    assert not admitted(due_in(3), evidence(cites=1))  # a single cite isn't an anchor
    assert not admitted(due_in(3), evidence(consumed=True))
    assert not admitted(_commitment("c1", due=False, check_after=None), evidence())
    assert not admitted(_thread("c1", revision=2, acted=0), evidence())  # a thread has no date
    assert not admitted(_commitment("c1", status="paused", check_after=NIGHT + timedelta(days=2)), evidence())
    (want,) = _wants(due_in(3), results=(evidence(),))
    assert (want.kind, want.action) == ("share", "hold_note")


def test_closing_a_question_marks_its_targets_acted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _thread("t1", revision=3, acted=1))
    _seed(store, _thread("t2", revision=2, acted=0))
    _seed(store, _thread("t3", revision=2, acted=0))
    targets = store.open_targets(NIGHT)
    for status in ("parked", "concluded", "retracted", "abandoned"):
        marks = snapshot.close_marks(_question(targets=("t1", "t2"), status=status), targets)
        assert marks == (("t1", 3), ("t2", 2)), status
    for status in ("open", "active"):
        assert snapshot.close_marks(_question(targets=("t1",), status=status), targets) == ()
    for target_id, revision in snapshot.close_marks(_question(targets=("t1", "t2"), status="parked"), targets):
        store.mark_acted(target_id, revision)
    by_id = {t.id: t for t in store.open_targets(NIGHT)}
    assert by_id["t1"].unacted == 0 and by_id["t2"].unacted == 0 and by_id["t3"].unacted == 2


def test_parked_question_wakes_only_on_new_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store, _thread("t1", revision=3, acted=0))
    parked = _question(status="parked", continues=True)
    for target_id, revision in snapshot.close_marks(parked, store.open_targets(NIGHT)):
        store.mark_acted(target_id, revision)
    for _ in range(2):  # nights pass; continuation never wakes a parked question
        assert _night(store, questions=(parked,)).want is None
    _mention(store, "t1")  # the person says something new
    woken = _night(store, questions=(parked,))
    assert (woken.action, woken.target_id) == ("revisit_thread", "t1")


def test_continuation_never_makes_contact_eligible() -> None:
    thread = _thread("t1", revision=3, acted=3)
    shot = _snap(thread, questions=(_question(continues=True),))
    assert [t.id for t in shot.targets] == ["t1"]  # the thread passes with no new evidence
    wants = derive_wants(shot)
    assert [w.action for w in wants] == ["inquire"]
    assert not [w for w in wants if w.action == "hold_note"]
    assert choose("away", wants, HOLD_NOTE_ONLY).want is None
    assert choose("away", wants, CTX).action == "inquire"


def test_intentions_supply_a_date_only_to_a_matching_commitment() -> None:
    plan = _commitment("c1", revision=1, acted=1, due=False, check_after=None)
    ledger = (Commitment(id="c1", label="her plan", check_after=NIGHT - timedelta(hours=1)),
              Commitment(id="elsewhere", label="not a target", check_after=NIGHT - timedelta(days=9)))
    assert _wants(plan) == ()
    (note,) = _wants(plan, intentions=ledger)
    assert (note.kind, note.action, note.target_id) == ("follow_up", "hold_note", "c1")


def test_inquiry_choice_ignores_a_question_that_is_not_ready() -> None:
    thread = _thread("t1", revision=3, acted=3)
    asleep = _question("A", status="open", priority=2.0, continues=False)
    ready = _question("B", status="active", priority=1.0, continues=True)
    (want,) = _wants(thread, questions=(asleep, ready))
    assert (want.action, want.question_id, want.strength) == ("inquire", "B", pytest.approx(0.5))
    fresh = _thread("t1", revision=4, acted=3)  # with new evidence any live question may be chosen
    (woken,) = _wants(fresh, questions=(asleep, ready))
    assert woken.question_id == "A"


def test_receipt_outcome_must_be_one_of_the_four() -> None:
    choice = choose("away", _wants(_thread("t1")), CTX)
    for bad in ("she was lonely", "RAN", 3):
        with pytest.raises(ValueError, match="outcome") as caught:
            receipt.record(choice, bad)  # type: ignore[arg-type]
        assert "lonely" not in str(caught.value)


def test_a_result_stamped_with_the_persons_last_words_is_news() -> None:
    """The night stamps her day's evidence and its own results at one moment: that result is new to her."""
    thread = _thread("t1", revision=2, acted=2)
    assert _ids(_wants(thread, results=(_result("t1", sources=2, created_at=SPOKE),))) == ["share:hold_note:t1"]
    earlier = _result("t1", sources=2, created_at=SPOKE - timedelta(seconds=1))
    assert _wants(thread, results=(earlier,)) == ()


def test_waiting_thread_result_shares_without_reopening_thread() -> None:
    thread = _thread("t1", revision=2, acted=2)
    shot = _snap(thread, results=(_result("t1", sources=2),))
    assert shot.targets == () and len(shot.results) == 1
    (want,) = derive_wants(shot)
    assert (want.kind, want.action, want.target_id) == ("share", "hold_note", "t1")
    assert want.priority_class == "active_thread"
