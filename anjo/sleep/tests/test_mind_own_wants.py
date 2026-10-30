# mypy: disallow-untyped-decorators=False
"""Her own wants (Task 18): pulls from her own inquiry state, and the quiet week they keep alive.

Offline, fakes only. The unit cases read ``derive_wants`` directly; the week runs on the accelerated
fake rig, once with ``ANJO_SLEEP_MIND_OWN_WANTS=1`` and once with ``0``, and counts what her own
thinking did on the same seed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from anjo.sleep.mind import snapshot, wants
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.ops import op_to_json
from anjo.sleep.mind.policy import PUBLIC_DEFAULT
from anjo.sleep.mind.week import NightSlice, QuietNight, quiet_week, slices
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.sim.day import DayRun
from anjo.sleep.tests.accelerated_rig import FakeAccelerated, life_targets
from anjo.sleep.tests.mind_night_fakes import Predicts, _ITEM, _section, workspace
from anjo.sleep.tests.test_accelerated import HERO, START
from anjo.sleep.types import (
    ChoiceContext,
    NightAllowance,
    OwnView,
    Result,
    Target,
    Want,
    WantSnapshot,
)

PERSON = "synthetic-film-student"
DAY = date(2026, 10, 27)
NIGHT = datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)
SPOKE = NIGHT - timedelta(hours=20)
CTX = ChoiceContext(policy=PUBLIC_DEFAULT, allowance=NightAllowance(actions_left=2, calls_left=8, usd_left=3.0))
PLAN_DUE = it.STAKE_DUE  # a question on a plan due within the week
PLAN_FAR = it.STAKE_NONE
UID = "fake-life"
SEED = 17  # one seed for both arms: the same simulated days, so only the switch differs
QUIET_DAYS = 4  # day one onward, on the same seed, with no message at all


def _target(target_id: str = "c1", kind: str = "commitment", due: datetime | None = None, **changes: Any) -> Target:
    base = Target(
        id=target_id, person_id=PERSON, kind=kind, label=f"private label of {target_id}",  # type: ignore[arg-type]
        check_after=due, status="open", revision=1, acted_revision=0, confidence=None, last_mentioned=DAY,
    )
    return replace(base, **changes)


def _view(**changes: Any) -> OwnView:
    """One question of her own state, last worked a day ago: a quiet week's ordinary night."""
    base = OwnView(
        question_id="q1", target_ids=("c1",), status="open", credences=(0.6, 0.2), stake=PLAN_DUE,
        uncertainty=0.5, prediction_due=None, resolved_prediction=False, finding=False,
        last_worked_at=NIGHT - timedelta(days=1),
    )
    return replace(base, **changes)


def _snap(*targets: Target, views: tuple[OwnView, ...] = (), results: tuple[Result, ...] = (),
          inquiry_offered: bool = True, own_wants: bool = True, as_of: datetime = NIGHT) -> WantSnapshot:
    shot = snapshot.build(list(targets), list(results), as_of, SPOKE)
    return replace(shot, own=views, inquiry_offered=inquiry_offered, own_wants=own_wants)


def _own(*targets: Target, **kwargs: Any) -> tuple[Want, ...]:
    return tuple(w for w in wants.derive_wants(_snap(*targets, **kwargs)) if w.kind in ("know", "settle"))


def _pulls(want: Want, ctx: ChoiceContext = CTX) -> bool:
    """Whether ``choose`` would let this want through its bar and the night's allowance."""
    from anjo.sleep.mind.choose import choose

    return choose("away", (want,), ctx).want is want


# ── Settle: a promise to know, or two answers she can't tell apart ───────────


def test_settle_waits_for_her_own_date_then_pulls_harder_overdue() -> None:
    plan = _target(due=NIGHT + timedelta(days=3))
    asked = _view(prediction_due=NIGHT + timedelta(days=2), last_worked_at=NIGHT)
    assert _own(plan, views=(asked,)) == ()  # the date hasn't come: nothing of hers to settle
    (due,) = _own(plan, views=(asked,), as_of=NIGHT + timedelta(days=2))
    assert (due.kind, due.action, due.question_id) == ("settle", "inquire", "q1")
    assert _pulls(due)
    (later,) = _own(plan, views=(asked,), as_of=NIGHT + timedelta(days=30))
    assert later.strength > due.strength  # the more overdue, the harder it pulls


def test_settle_is_stronger_the_more_the_plan_matters() -> None:
    asked = _view(prediction_due=NIGHT - timedelta(days=1))
    (plan,) = _own(_target("c1", "commitment", due=NIGHT + timedelta(days=3)), views=(asked,))
    (casual,) = _own(_target("c1", "thread"), views=(replace(asked, stake=PLAN_FAR),))
    assert plan.strength > casual.strength


def test_settle_two_alternatives_neither_leads() -> None:
    plan = _target(due=NIGHT + timedelta(days=3))
    (even,) = _own(plan, views=(_view(credences=(0.45, 0.42, 0.13), uncertainty=0.9),))
    assert even.kind == "settle"
    (led,) = _own(plan, views=(_view(credences=(0.8, 0.15, 0.05)),))
    assert led.kind == "know"  # one answer leads: she keeps asking, she doesn't settle


# ── Know: uncertainty on her own clock ───────────────────────────────────────


def test_know_grows_with_the_time_she_has_not_worked_it() -> None:
    plan = _target(due=NIGHT + timedelta(days=3))
    view = _view(uncertainty=0.8, last_worked_at=NIGHT - timedelta(days=3))
    (early,) = _own(plan, views=(view,))
    (stale,) = _own(plan, views=(replace(view, last_worked_at=NIGHT - timedelta(days=30)),))
    assert early.kind == stale.kind == "know"
    assert stale.strength > early.strength
    assert _own(plan, views=(replace(view, last_worked_at=NIGHT, uncertainty=0.9),)) == ()  # worked tonight


def test_know_leaves_a_parked_question_asleep() -> None:
    parked = _view(status="parked", uncertainty=0.9, last_worked_at=NIGHT - timedelta(days=30))
    assert _own(_target(due=NIGHT + timedelta(days=3)), views=(parked,)) == ()


# ── Share: her own thinking on the person's live plan ────────────────────────


def _notes(*targets: Target, **kwargs: Any) -> list[Want]:
    return [w for w in wants.derive_wants(_snap(*targets, **kwargs)) if w.kind == "share"]


def test_own_finding_notes_the_live_plan_once() -> None:
    plan = _target(due=NIGHT + timedelta(days=30))
    (note,) = _notes(plan, views=(_view(finding=True),))
    assert (note.action, note.target_id) == ("hold_note", "c1")
    assert note.strength >= PUBLIC_DEFAULT.surface_min
    assert _notes(plan, views=(_view(finding=False, resolved_prediction=False),)) == []  # nothing of hers to say
    assert _notes(replace(plan, due_noted=True), views=(_view(finding=True),)) == []  # the plan's note was held


def test_own_finding_does_not_repeat_a_results_share() -> None:
    plan = _target(due=NIGHT + timedelta(days=3))
    result = Result(id="r1", target_id="c1", homework_id="hw-1", verified_sources=2,
                    created_at=NIGHT - timedelta(hours=2), consumed=False)
    (note,) = _notes(plan, views=(_view(finding=True),), results=(result,))
    assert note.id == "share:c1"  # one note on the target, whichever source raised it


# ── The switch and the offer ─────────────────────────────────────────────────


def test_own_wants_switch_turns_every_own_source_off() -> None:
    plan = _target(due=NIGHT - timedelta(days=1))
    loud = _view(finding=True, resolved_prediction=True, prediction_due=NIGHT - timedelta(days=3))
    assert {"settle", "share"} <= {w.kind for w in wants.derive_wants(_snap(plan, views=(loud,)))}
    quiet = {w.kind for w in wants.derive_wants(_snap(plan, views=(loud,), own_wants=False))}
    assert not quiet & {"settle", "know", "share"}
    assert wants.own_wants_on({}) is True
    assert wants.own_wants_on({wants.OWN_WANTS_ENV: "0"}) is False


def test_own_wants_wait_for_a_night_that_may_inquire() -> None:
    asked = _view(prediction_due=NIGHT - timedelta(days=1))
    assert _own(_target(due=NIGHT + timedelta(days=3)), views=(asked,), inquiry_offered=False) == ()


def test_own_finding_is_still_said_when_only_a_note_is_offered() -> None:
    # The second offer of a night (step 10) offers the note alone; a finding made tonight is said.
    plan = _target(due=NIGHT + timedelta(days=30))
    (note,) = _notes(plan, views=(_view(finding=True),), inquiry_offered=False)
    assert (note.action, note.target_id) == ("hold_note", "c1")


# ── A week alone, on the fake rig, with her own wants on and off ─────────────


@dataclass
class NoticesOnTheWay(Predicts):
    """Commits its prediction, then, once that prediction is settled, opens one question it noticed
    while working the first: her own thinking finding more to think about, with no message."""

    noticed: bool = False

    def _cycle(self, prompt: str) -> str:
        settled = [line for line in _section(prompt, "## Settled predictions") if line.startswith("- ")]
        if not self.predicted or self.noticed or not settled:
            return super()._cycle(prompt)
        self.noticed = True
        shown = [m.group(1) for line in _section(prompt, "## Evidence") if (m := _ITEM.match(line))]
        op = it.NoticeOp(kind="gap", subject="the plan's missing budget", target_ids=self.targets,
                         origin_evidence=(shown[0],))
        return json.dumps(op_to_json(op))


def _week(root: Path, *, own: bool, monkeypatch: Any) -> tuple[QuietNight, ...]:
    """Five nights on one seed with the person's messages only on the first, so the rest is a week
    with nothing new said. The inquiry runs only when her wants pull it, as on a live night
    (``quiet_inquiry`` is the simulated alone run's own forcing, and stays off here)."""
    monkeypatch.setenv(wants.OWN_WANTS_ENV, "1" if own else "0")
    thinker = NoticesOnTheWay(targets=(f"{UID}-c1",), queries=["pilot"])
    fake = FakeAccelerated(root, mind_from=START, thinker=thinker)
    deps = fake.rig().deps_for(START)
    deps = replace(deps, settings=replace(deps.settings, force_deep=True))  # a quiet night still sleeps deeply
    assert not deps.settings.quiet_inquiry
    end = START
    for index in range(1 + QUIET_DAYS):
        night = START + timedelta(days=index)
        if index == 0:
            day = fake.simulate(UID, HERO, index, SEED)
            # The person's plan and topic were already studied: nothing of theirs is unacted, so a
            # quiet night's wants can only be her own.
            store = fake.world.targets(UID)
            for target in life_targets(UID, START):
                store.upsert(target)
                store.mark_acted(target.id, target.revision)
        else:  # a positive, empty window: no turn, no evidence, nothing to extract
            began = max(fake.clock.now, end)
            day = DayRun((began, began + timedelta(microseconds=1)), 0, 0, ())
        end = day.window[1]
        done = run_night_batch_with(deps, {UID: night}, accelerated=True, days={UID: day})
        assert done[UID].status != "failed", done[UID].failed_at
    events = tuple(workspace(fake.world, UID).events())
    quiet = {START + timedelta(days=index) for index in range(1, 1 + QUIET_DAYS)}
    nights = [replace(section, alone=section.night in quiet) for section in slices(events)]
    # A night her own thinking did nothing on logs nothing: it is still one of the week's nights.
    nights += [NightSlice(night, (), True) for night in quiet - {section.night for section in nights}]
    return quiet_week(tuple(sorted(nights, key=lambda section: section.night)))


def test_quiet_week_her_own_wants_open_questions_and_settle_predictions(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    on = _week(tmp_path / "on", own=True, monkeypatch=monkeypatch)
    off = _week(tmp_path / "off", own=False, monkeypatch=monkeypatch)
    assert [n.night for n in on] == [n.night for n in off] and len(on) == QUIET_DAYS
    assert all(n.questions_opened == n.predictions_committed == n.predictions_resolved == 0 for n in off), off
    assert sum(n.questions_opened for n in on) > sum(n.questions_opened for n in off)
    assert sum(n.predictions_resolved for n in on) > sum(n.predictions_resolved for n in off)
    assert sum(n.credence_change for n in on) > 0.0  # her beliefs moved with nothing said to her
