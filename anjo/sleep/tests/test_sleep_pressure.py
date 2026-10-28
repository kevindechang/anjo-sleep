# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Sleep pressure: a nap or a deep sleep, decided once a night (``night/sleep_pressure.py``), on the
night's own fakes. A nap runs the CPU steps, wakes no Glimmer and defers MIND's passes and the note to
the next deep sleep; the month's cap always wins."""
from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.night import cost, past
from anjo.sleep.night.deps import STUDY_ACTIONS, NightSettings
from anjo.sleep.night.gpu import batch_digest
from anjo.sleep.night.holds import usd
from anjo.sleep.night.past import COLAB_USD_PER_CU, days_since_deep, gpu_batches, last_deep, month_spent, notice_from, turns_since
from anjo.sleep.night.run import GPU_STEPS, run_night_batch_with
from anjo.sleep.night.nightlog import summaries
from anjo.sleep.night.sleep_night import CAPPED_WHY, DEFER_WHY, owed_days
from anjo.sleep.night.sleep_pressure import (
    CORE_ENV,
    DEEP_FLOOR_USD,
    FRESH_NIGHT_USD,
    GPU_ENV,
    MAX_CATCH_UP_DAYS,
    MEASURED_MARGIN,
    MEASURED_REUSED_USD,
    NAP_USD,
    PRACTICE_REUSED_SAVES_USD,
    SleepPolicy,
    SleepPolicyError,
    SleepState,
    decide,
    deep_estimate,
    deep_night_usd,
    fits,
    high_risk,
    policy_from,
    sleep_pressure,
)
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.tests.mind_night_fakes import LifeThinker, mind_night
from anjo.sleep.tests.night_fakes import NIGHT, Homework, World, step, steps, summary, thread
from anjo.sleep.tests.test_night_mind import NIGHT_END, ON, _commitment, _world
from anjo.sleep.types import NightResult, Target

UID = "p1"
POLICY = SleepPolicy()
END = datetime.combine(NIGHT + timedelta(days=1), time.min, tzinfo=timezone.utc)


def _night(world: World, night: date = NIGHT, **deps: Any) -> NightResult:
    return run_night_batch_with(world.deps(**{"sleep_policy": POLICY, **deps}), {UID: night})[UID]


def _turns(count: int) -> Any:
    return lambda user_id, window: count


def _lost(uid: str, n: int) -> Target:
    return replace(thread(uid), id=f"{uid}-lost{n}", confidence=0.1)


def _state(**changes: Any) -> SleepState:
    base = SleepState(new_turns=0, new_evidence=0, open_high_risk=0, days_since_deep=None, month_spent_usd=0.0)
    return replace(base, **changes)


def _settle(ledger: Ledger, job: str, amount: float, pool: Any = "modal_usd") -> None:
    held = ledger.reserve(job, (pool,), usd(amount))
    assert held is not None
    held.settle(amount)


def _holds_on(ledger: Ledger, pool: str) -> list[str]:
    """Every hold ever taken on ``pool``, settled, released or open."""
    events = read_records(ledger.path, PLAIN) if ledger.path.exists() else []
    return [event["job"] for event in events if event["op"] == "reserve" and pool in event["pools"]]


def _note_due(world: World) -> Target:
    """A plan whose date has come, already prepared: the night's chooser holds a note on it."""
    due = _commitment(UID, NIGHT_END - timedelta(hours=1))
    world.targets(UID).upsert(due)
    world.targets(UID).mark_acted(due.id, due.revision)
    return due


# ── the night ─────────────────────────────────────────────────────────────────


def test_quiet_day_naps_and_costs_no_gpu(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=1)
    result = _night(world, turns=_turns(4))
    assert result.status == "napped" and result.checkpoint is None and result.failed_at is None
    sleep = step(result.log_path, "sleep")
    assert (sleep["kind"], sleep["reason"]) == ("nap", "below_threshold")
    assert sleep["pressure"] < sleep["threshold"] == POLICY.base_threshold
    assert (sleep["turns"], sleep["new_turns"], sleep["new_evidence"]) == (4, 4, 1)
    assert (sleep["gpu"], sleep["core"], sleep["channel"], sleep["practice_fresh"]) == (
        "L40S", False, False, True)  # what prices later deep nights
    assert world.gpu.opened == [] and world.trainer.calls == [] and world.glimmer.calls == 0
    skipped = {r["step"]: r["why"] for r in steps(result.log_path) if r["status"] == "skipped"}
    assert set(skipped) == {"probes", *GPU_STEPS} and all(why.startswith("napped") for why in skipped.values())
    done = summary(result.log_path)
    assert done["status"] == "napped" and done["sleep"]["kind"] == "nap"
    assert all(done["cost_parts"].get(part, 0.0) == 0.0 for part in ("practice", "training", "gates"))
    assert step(result.log_path, "brief")["folded"] == 1  # a nap still folds the day


def test_pressure_over_threshold_goes_deep(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=12)
    result = _night(world, turns=_turns(60))
    sleep = step(result.log_path, "sleep")
    assert (sleep["kind"], sleep["reason"]) == ("deep", "pressure") and sleep["pressure"] >= sleep["threshold"]
    assert result.status == "promoted" and world.gpu.opened == [(UID,)] and len(world.trainer.calls) == 1
    assert summary(result.log_path)["sleep"]["kind"] == "deep"


def test_max_days_forces_deep(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=1)
    first = _night(world)
    later = NIGHT + timedelta(days=POLICY.max_days)
    world.plan(UID, later, count=1)
    forced = _night(world, night=later)
    sleep = step(forced.log_path, "sleep")
    assert first.status == "napped" and step(first.log_path, "sleep")["days_since_deep"] is None
    assert (sleep["kind"], sleep["reason"], sleep["days_since_deep"]) == ("deep", "max_days", POLICY.max_days)
    assert sleep["pressure"] < sleep["threshold"] and forced.status == "promoted"
    world.plan(UID, later + timedelta(days=1), count=1)
    after = _night(world, night=later + timedelta(days=1))  # the deep sleep restarts the count
    assert step(after.log_path, "sleep")["days_since_deep"] == 1 and after.status == "napped"


def test_month_cap_turns_deep_into_logged_nap(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=12)
    for n in range(3):
        world.targets(UID).upsert(_lost(UID, n))
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": 1000.0})
    yesterday = (NIGHT - timedelta(days=1)).isoformat()
    _settle(ledger, f"night:practice:{UID}:{yesterday}:a1", 10.5)  # room for her naps and Wonder, not a deep night
    _settle(ledger, f"night:practice:someone:{yesterday}:a1", 400.0)  # another person's: not hers
    result = _night(world, turns=_turns(60), ledger=ledger)
    sleep = step(result.log_path, "sleep")
    assert (sleep["kind"], sleep["reason"]) == ("nap", "month_cap")
    assert sleep["pressure"] == 1.0 >= sleep["threshold"] > POLICY.base_threshold  # it would have gone deep
    assert sleep["month_spent_usd"] == pytest.approx(10.5) and sleep["month_usd"] == POLICY.month_usd
    assert sleep["capped"] is False and step(result.log_path, "extract")["status"] == "ok"  # a nap with room
    assert result.status == "napped" and world.gpu.opened == [] and world.trainer.calls == []
    assert summary(result.log_path)["sleep"]["reason"] == "month_cap"  # logged, never silent


def test_a_nap_wakes_no_glimmer_and_defers_mind_and_the_note(tmp_path: Path) -> None:
    world = _world(tmp_path, nights=2)
    due = _note_due(world)
    ledger = Ledger(tmp_path / "budget.jsonl")
    thinker, woke = LifeThinker(targets=(f"{UID}-t1", due.id)), []
    mind = mind_night(world, thinker, ledger=ledger)

    def wake() -> WakeResult:
        woke.append(1)
        return WakeResult(seconds=1.0, ok=True)

    nap = _night(world, settings=ON, inquiry=mind, notice=mind, ledger=ledger, wake=wake)
    assert nap.status == "napped" and step(nap.log_path, "sleep")["kind"] == "nap"
    assert _holds_on(ledger, "modal_usd") == [] and woke == [] and thinker.prompts == []  # Glimmer slept on
    assert world.writer.calls == [] and world.notifier.queued == [] and world.gpu.opened == []
    notice = step(nap.log_path, "notice")
    assert (notice["status"], notice["why"]) == ("skipped", DEFER_WHY)
    choose = step(nap.log_path, "choose")
    assert "inquire" not in choose["offered"] and choose["ports"]["inquiry"] == DEFER_WHY
    assert (choose["action"], choose["target_id"]) == ("hold_note", due.id)
    note = step(nap.log_path, "note")
    assert (note["outcome"], note["target_id"], note["why"]) == ("deferred", due.id, DEFER_WHY)
    assert not next(t for t in world.targets(UID).open_targets(NIGHT_END) if t.id == due.id).due_noted  # unsettled

    later = NIGHT + timedelta(days=1)  # her next deep sleep notices the nap's day and words the note
    deep = _night(world, night=later, settings=replace(ON, force_deep=True), inquiry=mind, notice=mind, ledger=ledger)
    assert deep.status == "promoted" and step(deep.log_path, "sleep")["kind"] == "deep"
    noticed = step(deep.log_path, "notice")
    assert noticed["new"] == 6 and noticed["notice_from"] == NIGHT.isoformat()  # both days' items, the nap's first
    worded = step(deep.log_path, "note")
    assert (worded["outcome"], worded["target_id"]) == ("held", due.id) and len(world.notifier.queued) == 1
    assert thinker.prompts and any(job.startswith(f"night:note:{UID}:{later.isoformat()}") for job in _holds_on(ledger, "modal_usd"))


def test_force_deep_overrides(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=1)
    settings = NightSettings(token_cap=16384, min_new_examples=1, force_deep=True)
    result = _night(world, settings=settings)
    sleep = step(result.log_path, "sleep")
    assert (sleep["kind"], sleep["reason"], sleep["forced"]) == ("deep", "forced", True)
    assert sleep["pressure"] < sleep["threshold"] and sleep["new_evidence"] < POLICY.min_new_evidence
    assert result.status == "promoted" and world.gpu.opened == [(UID,)]
    capped = decide(_state(month_spent_usd=14.0), POLICY, pressure=0.0, force_deep=True)
    assert (capped.kind, capped.reason, capped.forced) == ("nap", "month_cap", True)  # the cap still wins


LAST, FIRST = date(2026, 10, 31), date(2026, 11, 1)  # October's last night, then November's first


def _nap(folder: Path, spent: float) -> tuple[World, NightResult, Ledger]:
    """October's last night, a nap after ``spent`` of the month."""
    world = World(folder)
    world.plan(UID, LAST, count=1)
    world.targets(UID).upsert(thread(UID))
    ledger = Ledger(folder / "budget.jsonl", caps={**CAPS, "modal_usd": 1000.0})
    _settle(ledger, f"night:practice:{UID}:{(LAST - timedelta(days=1)).isoformat()}:a1", spent)
    return world, _night(world, night=LAST, turns=_turns(4), ledger=ledger, homework=Homework(world)), ledger


def test_a_capped_nap_makes_no_paid_hold(tmp_path: Path) -> None:
    """A night with no room under the month's cap even for a nap is a capped nap: it skips its paid steps
    (extract, the study and the brief), says so and holds nothing; the next night with room extracts the
    day it left first. The same nap with room extracts, folds the day and may study."""
    last, first, nap = LAST, FIRST, _nap
    _, roomy, roomy_ledger = nap(tmp_path / "room", 1.0)
    assert step(roomy.log_path, "sleep")["capped"] is False and step(roomy.log_path, "brief")["folded"] >= 1
    assert any(job.startswith(f"night:brief:{UID}") for job in _holds_on(roomy_ledger, "baseten_sleep_usd"))
    world, capped, ledger = nap(tmp_path / "capped", POLICY.month_usd - 0.5)  # Wonder's $3 held back: no room
    sleep = step(capped.log_path, "sleep")
    assert (sleep["kind"], sleep["capped"], capped.status) == ("nap", True, "napped")
    for skipped in ("extract", "brief"):
        assert (step(capped.log_path, skipped)["status"], step(capped.log_path, skipped)["why"]) == ("skipped", CAPPED_WHY)
    choose = step(capped.log_path, "choose")
    assert choose["action"] not in STUDY_ACTIONS and not set(choose["offered"]) & STUDY_ACTIONS
    assert choose["ports"]["homework"] == CAPPED_WHY and summary(capped.log_path)["wonder_usd"] == 0.0
    held = [e["job"] for e in read_records(ledger.path, PLAIN) if e["op"] == "reserve" and e["job"].startswith("night:")]
    assert held == [f"night:practice:{UID}:{(last - timedelta(days=1)).isoformat()}:a1"]  # only the seeded spend
    assert not [event for event in world.events if event[0] == "extract"]

    world.plan(UID, first, count=1)  # a new month: room again, and October 31's day is extracted first
    caught = _night(world, night=first, turns=_turns(4), ledger=ledger)
    assert [event[2] for event in world.events if event[0] == "extract"] == [last, first]
    assert step(caught.log_path, "sleep")["caught_up"] == [last.isoformat()]  # settled: owed no more
    assert {item.day for item in world.evidence(UID).stored(first)} >= {last, first}
    assert owed_days(summaries(world.logs(UID)), first + timedelta(days=1)) == []


def test_an_owed_day_stays_owed_until_an_extract_settles_it(tmp_path: Path) -> None:
    """A night that fails at extract settles nothing: the capped nap's day is still owed, and the next
    night with room extracts it first and marks it settled."""
    world, _, ledger = _nap(tmp_path, POLICY.month_usd - 0.5)
    world.plan(UID, FIRST, count=1)
    world.extract_fails[UID] = RuntimeError("extract down")
    failed = _night(world, night=FIRST, turns=_turns(4), ledger=ledger)
    assert failed.failed_at == "extract" and [e[2] for e in world.events if e[0] == "extract"] == [LAST]
    later = FIRST + timedelta(days=1)
    assert [day for day, _, _ in owed_days(summaries(world.logs(UID)), later)] == [LAST]
    del world.extract_fails[UID]
    world.plan(UID, later, count=1)
    caught = _night(world, night=later, turns=_turns(4), ledger=ledger)
    assert [e[2] for e in world.events if e[0] == "extract"] == [LAST, LAST, later]
    assert step(caught.log_path, "sleep")["caught_up"] == [LAST.isoformat()]
    assert owed_days(summaries(world.logs(UID)), later + timedelta(days=1)) == []


def _logged(world: World, day: date, status: str, sleep: Mapping[str, Any]) -> None:
    """An earlier night's summary in her log, as a finished night leaves it."""
    folder = world.logs(UID) / day.isoformat()
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"night": day.isoformat(), "status": status,
                                                     "trained": status == "promoted", "sleep": dict(sleep)}))


def _owing(world: World, count: int) -> list[date]:
    """``count`` capped naps before ``NIGHT``, each leaving its day unextracted (owed), oldest first."""
    days = [NIGHT - timedelta(days=n) for n in range(count, 0, -1)]
    for day in days:
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        deferred = {"window": [start.isoformat(), (start + timedelta(days=1)).isoformat()],
                    "as_of": (start + timedelta(days=1)).isoformat()}
        _logged(world, day, "napped", {"kind": "nap", "capped": True, "extract_deferred": deferred})
    return days


def test_a_night_catches_up_at_most_a_weeks_owed_days_and_holds_back_the_rest(tmp_path: Path) -> None:
    """Owed days are held back under the cap like naps; one night extracts the oldest
    ``MAX_CATCH_UP_DAYS`` of them, and the rest stay owed for the nights after."""
    world = World(tmp_path)
    days = _owing(world, MAX_CATCH_UP_DAYS + 2)
    assert [day for day, _, _ in owed_days(summaries(world.logs(UID)), NIGHT)] == days
    world.plan(UID, NIGHT, count=1)
    result = _night(world, turns=_turns(4))
    assert [e[2] for e in world.events if e[0] == "extract"] == [*days[:MAX_CATCH_UP_DAYS], NIGHT]
    assert step(result.log_path, "sleep")["caught_up"] == [day.isoformat() for day in days[:MAX_CATCH_UP_DAYS]]
    after = NIGHT + timedelta(days=1)
    assert [day for day, _, _ in owed_days(summaries(world.logs(UID)), after)] == days[MAX_CATCH_UP_DAYS:]
    # every owed day is held back like a nap: the room a nap needs shrinks by a nap's price per day owed
    room = POLICY.month_usd - 3.0 - 10 * POLICY.nap_usd - POLICY.nap_usd  # Wonder's $3, 10 later nights, tonight
    state = _state(month_spent_usd=room, nights_left=10, wonder_left_usd=3.0)
    assert fits(state, POLICY, POLICY.nap_usd) and not fits(replace(state, owed_days=1), POLICY, POLICY.nap_usd)


@pytest.mark.parametrize(("spent", "capped", "owed"), [(11.0, True, 9), (9.15, False, 2)])
def test_the_nights_hold_back_every_owed_day(tmp_path: Path, spent: float, capped: bool, owed: int) -> None:
    """Nine days owed: before extract (``at_cap``) all nine are held back like naps; tonight's decision,
    the two its catch-up leaves. At $11.00 spent a nap fits only without the nine, so the night is
    capped; at $9.15 a deep night fits only without the two, so it naps at the cap."""
    world = World(tmp_path)
    _owing(world, MAX_CATCH_UP_DAYS + 2)  # 9 days since her first night: deep sleep is due (max_days)
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": 1000.0})
    _settle(ledger, f"night:practice:{UID}:{date(2026, 10, 2).isoformat()}:a1", spent)
    room = POLICY.month_usd - spent - 3 * POLICY.nap_usd - 3.0  # October 28th: 3 later nights and Wonder's $3 held back
    tonight = POLICY.nap_usd if capped else deep_night_usd()  # nothing measured: the model's deep night
    assert tonight <= room < tonight + owed * POLICY.nap_usd  # it fits only without the owed days held back
    world.plan(UID, NIGHT, count=1)
    sleep = step(_night(world, turns=_turns(60), ledger=ledger).log_path, "sleep")
    assert (sleep["kind"], sleep["reason"], sleep["capped"], sleep["owed_days"]) == ("nap", "month_cap", capped, owed)


def test_a_capped_night_never_sleeps_deep_and_a_deep_night_is_never_priced_under_a_nap(tmp_path: Path) -> None:
    """A gated night that settled almost nothing (a ledger that missed its spend) can't price a deep
    night under a nap: it is floored, so a night capped before extract is a capped nap, wakes no
    Glimmer and runs no probe; and a capped night is a nap whatever the decision would want."""
    world = World(tmp_path)
    cheap = NIGHT - timedelta(days=POLICY.max_days)  # deep sleep is due again tonight (max_days)
    _logged(world, cheap, "promoted", {"kind": "deep", "gpu": POLICY.gpu, "core": False, "channel": False,
                                       "practice_fresh": True})
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": 1000.0})
    _settle(ledger, f"night:gates:{UID}:{cheap.isoformat()}:a1", 0.01)
    # 3 later nights and Wonder's $3 held back: room for a $0.0115 deep night, none for a nap
    _settle(ledger, f"night:practice:{UID}:{date(2026, 10, 2).isoformat()}:a1", POLICY.month_usd - 3.3 - 0.05 - 0.01)
    world.plan(UID, NIGHT, count=1)
    result = _night(world, turns=_turns(60), ledger=ledger)
    sleep = step(result.log_path, "sleep")
    assert (sleep["kind"], sleep["reason"], sleep["capped"]) == ("nap", "month_cap", True)
    assert sleep["measured_deep_usd"] == [pytest.approx(0.01)] and sleep["deep_usd"] >= max(POLICY.nap_usd, DEEP_FLOOR_USD)
    assert step(result.log_path, "probes")["status"] == "skipped" and world.glimmer.calls == 0
    assert step(result.log_path, "extract")["why"] == CAPPED_WHY and result.status == "napped"
    floored = _state(days_since_deep=POLICY.max_days, measured_deep_usd=(0.01,))
    assert deep_estimate(floored, POLICY) == DEEP_FLOOR_USD > POLICY.nap_usd
    capped = decide(floored, POLICY, pressure=1.0, force_deep=True, at_cap=True)  # room or not: capped wins
    assert (capped.kind, capped.reason, capped.capped) == ("nap", "month_cap", True)


def test_a_deep_night_is_priced_on_her_measured_ones() -> None:
    """Tonight's deep night: the highest of her last three measured at tonight's settings, plus the
    margin; the model's only before she has one."""
    state = _state(new_turns=60, new_evidence=12, open_high_risk=3, nights_left=10, wonder_left_usd=3.0)
    assert deep_estimate(state, POLICY) == deep_night_usd()  # nothing measured yet
    measured = replace(state, measured_deep_usd=(1.19, 1.05, 1.10, 9.0))  # newest first; the fourth is too old
    assert deep_estimate(measured, POLICY) == pytest.approx(1.19 * (1 + MEASURED_MARGIN))
    assert decide(measured, POLICY, pressure=1.0).deep_usd == pytest.approx(1.19 * (1 + MEASURED_MARGIN))
    on, before, older = NIGHT, NIGHT - timedelta(days=2), NIGHT - timedelta(days=5)
    failed, empty, other_gpu = (NIGHT - timedelta(days=n) for n in (1, 3, 4))
    key = {"gpu": "L40S", "core": False, "channel": False, "practice_fresh": True}
    history = [{**_summary(before), "sleep": {"kind": "deep", **key}},
               {**_summary(older), "sleep": {"kind": "deep", **key, "core": True}},  # another setting
               {**_summary(other_gpu), "sleep": {"kind": "deep", **key, "gpu": "H100"}},  # another serving GPU
               # deep nights that never reached the gates: a cheap failure is no deep night's price
               {**_summary(failed, status="failed"), "sleep": {"kind": "deep", **key}},
               {**_summary(empty, status="nothing_new"), "sleep": {"kind": "deep", **key}}]
    settled = {f"night:gates:{UID}:{before.isoformat()}:a1": 0.30, f"morning:{before.isoformat()}:{UID}:judge:a1": 0.50,
               f"night_gpu:{before.isoformat()}:{batch_digest([UID])}:a1": 0.40,
               f"night:homework:{UID}:{before.isoformat()}:a1": 0.23,  # Wonder: held back apart
               f"night:gates:{UID}:{older.isoformat()}:a1": 2.0,
               **{f"night:extract:{UID}:{day.isoformat()}:a1": 0.01 for day in (failed, empty, other_gpu)}}
    assert past.measured_deep(history, settled, UID, on, key) == pytest.approx((1.20,))
    rejected = [*history[:3], {**_summary(failed, status="rejected"), "sleep": {"kind": "deep", **key}}]
    assert past.measured_deep(rejected, settled, UID, on, key) == pytest.approx((0.01, 1.20))  # it reached the gates


def test_the_cap_holds_back_the_months_later_nights_and_wonder() -> None:
    deep = deep_night_usd()
    room = POLICY.month_usd - deep - 10 * POLICY.nap_usd - 1.0  # tonight's deep night, 10 naps and $1 of Wonder
    state = _state(new_turns=60, new_evidence=12, open_high_risk=3, month_spent_usd=room, nights_left=10,
                   wonder_left_usd=1.0)
    made = decide(state, POLICY, pressure=1.0)
    assert (made.kind, made.deep_usd) == ("deep", deep)
    tight = decide(replace(state, nights_left=11), POLICY, pressure=1.0)  # one more nap to hold back: no room
    assert (tight.kind, tight.reason, tight.capped) == ("nap", "month_cap", False)
    rerun = decide(replace(state, nights_left=11, practice_fresh=False), POLICY, pressure=1.0)
    assert rerun.kind == "deep" and rerun.deep_usd == pytest.approx(deep - PRACTICE_REUSED_SAVES_USD)
    full = decide(replace(state, month_spent_usd=POLICY.month_usd - 0.5), POLICY, pressure=0.0)
    assert (full.kind, full.capped) == ("nap", True) and full.record()["capped"] is True


def test_without_a_policy_the_night_is_unchanged(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan(UID, NIGHT, count=1)
    result = run_night_batch_with(world.deps(), {UID: NIGHT})[UID]
    assert result.status == "promoted" and "sleep" not in summary(result.log_path)
    assert "sleep" not in [r["step"] for r in steps(result.log_path)]


# ── the decision ─────────────────────────────────────────────────────────────


def test_the_threshold_rises_with_the_months_spend() -> None:
    assert [POLICY.threshold(spent) for spent in (0.0, 7.5, 15.0, 40.0)] == [0.5, 0.75, 1.0, 1.0]


def test_pressure_is_bounded_and_grows_with_each_signal() -> None:
    quiet, full = _state(), _state(new_turns=600, new_evidence=120, open_high_risk=30)
    assert sleep_pressure(quiet) == 0.0 and sleep_pressure(full) == 1.0
    for change in ({"new_turns": 10}, {"new_evidence": 2}, {"open_high_risk": 1}):
        assert 0.0 < sleep_pressure(_state(**change)) < 1.0


def test_too_little_new_material_naps_under_high_pressure() -> None:
    made = decide(_state(new_evidence=POLICY.min_new_evidence - 1), POLICY, pressure=0.9)
    assert (made.kind, made.reason) == ("nap", "too_little_new")
    assert decide(_state(new_evidence=POLICY.min_new_evidence), POLICY, pressure=0.9).kind == "deep"


def test_a_pressure_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(SleepPolicyError):
        decide(_state(), POLICY, pressure=1.5)


def test_high_risk_counts_due_commitments_and_lost_threads() -> None:
    due = Target(id="c1", person_id=UID, kind="commitment", label="x", check_after=END + timedelta(days=1),
                 status="open", revision=1, acted_revision=0, confidence=None, last_mentioned=NIGHT)
    far = replace(due, id="c2", check_after=END + timedelta(days=9))
    noted = replace(due, id="c3", due_noted=True)
    sure = replace(thread(UID), confidence=0.9)
    closed = replace(_lost(UID, 1), status="closed")
    assert high_risk([due, far, noted, _lost(UID, 0), sure, closed, thread(UID)], END) == 2


def test_policy_from_the_environment() -> None:
    assert policy_from({}) == SleepPolicy(month_usd=15.0, max_days=7) == policy_from({"ANJO_SLEEP_PRESSURE": "1"})
    assert policy_from({"ANJO_SLEEP_MONTH_USD": "10", "ANJO_SLEEP_MAX_DAYS": "3"}) == SleepPolicy(10.0, 3)
    assert policy_from({"ANJO_SLEEP_PRESSURE": "0", "ANJO_SLEEP_MAX_DAYS": "never"}) is None  # off: every night deep
    assert policy_from({CORE_ENV: "1", GPU_ENV: "H100"}) == SleepPolicy(gpu="H100", core=True)  # sized by its settings
    assert SleepPolicy(gpu="H100", core=True).deep_usd == deep_night_usd(gpu="H100", core=True) > POLICY.deep_usd
    from anjo.sleep.mind.core.bridge import CORE_ENV as BRIDGE_CORE_ENV
    from anjo.sleep.serving.config import SERVE_GPU_ENV, SERVE_GPUS

    assert (CORE_ENV, GPU_ENV, set(FRESH_NIGHT_USD)) == (BRIDGE_CORE_ENV, SERVE_GPU_ENV, set(SERVE_GPUS))
    for bad in ({"ANJO_SLEEP_MONTH_USD": "ten"}, {"ANJO_SLEEP_MONTH_USD": "0"}, {"ANJO_SLEEP_MAX_DAYS": "0"},
                {"ANJO_SLEEP_MAX_DAYS": "2.5"}, {"ANJO_SLEEP_MONTH_USD": "nan"}, {"ANJO_SLEEP_PRESSURE": "off"},
                {GPU_ENV: "T4"}):
        with pytest.raises(SleepPolicyError):
            policy_from(bad)


def test_the_deep_night_estimate_covers_the_cost_model_at_every_setting() -> None:
    """At every setting the cost model projects, on every serving GPU, the core on or off, the base cache
    hit or missed and practice fresh or reused, the decision's deep night is at or above the model's."""
    from anjo.sleep.serving.config import SERVE_GPUS

    settings = [cost.CURRENT, *cost.SCENARIOS, *(replace(cost.ONE_SYSTEM, name=gpu, gpu=gpu) for gpu in SERVE_GPUS)]
    for measured in (cost.Measured(extract=0.02, brief=0.06, training=None),  # the live night's own numbers, rounded up
                     cost.Measured(extract=0.001, brief=0.022, training=0.107)):  # 2026-10-29's
        for setting, core, hit in itertools.product(settings, (False, True), (False, True)):
            p = cost.project(replace(setting, mind_core=core, base_cached=hit), cost.UNITS, cost.DEFAULT_PRICES, measured)
            practice = math.fsum(usd for line, usd in p.trained.items() if line.startswith("practice"))
            assert deep_night_usd(gpu=setting.gpu, core=core, cache_hit=hit) >= p.deep, (setting.name, core, hit)
            assert deep_night_usd(gpu=setting.gpu, core=core, cache_hit=hit, practice_fresh=False) >= p.deep - practice
            assert NAP_USD >= math.fsum(p.quiet.values())
    assert deep_night_usd() >= 2.42 and deep_night_usd(core=True) >= 2.63  # a cache miss and the core's night, as reviewed
    assert deep_night_usd(practice_fresh=False) >= MEASURED_REUSED_USD  # 2026-10-29's night as measured, practice reused
    cheapest = min(cost.project(replace(setting, mind_core=core, base_cached=hit), cost.UNITS, cost.DEFAULT_PRICES,
                                measured).practice_reused().deep
                   for setting, core, hit in itertools.product(settings, (False, True), (False, True)))
    assert NAP_USD < DEEP_FLOOR_USD <= cheapest  # a floor no modelled deep night is under, practice reused and all savings on


# ── her past and her month ───────────────────────────────────────────────────


def _summary(night: date, *, kind: str | None = None, status: str = "promoted", turns: int | None = None,
             trained: bool = False) -> Mapping[str, Any]:
    sleep = {} if kind is None else {"sleep": {"kind": kind, "turns": turns}}
    return {"night": night.isoformat(), "status": status, "trained": trained, **sleep}


def test_the_last_deep_sleep_and_the_turns_since() -> None:
    day = [NIGHT + timedelta(days=n) for n in range(6)]
    history = [
        _summary(day[0], trained=True),  # from before naps were logged: it trained, so it slept deep
        _summary(day[1], kind="nap", status="napped", turns=5),
        _summary(day[2], kind="deep", status="failed", turns=7),  # a failed deep sleep doesn't count
        _summary(day[3], kind="nap", status="napped", turns=11),
        _summary(day[3], kind="deep", status="rejected", turns=11),  # its rerun: the last run counts
    ]
    assert last_deep(history, day[3]) == day[0] and last_deep(history, day[4]) == day[3]
    assert days_since_deep(history, day[5]) == 2 and days_since_deep([], day[0]) is None
    assert turns_since(history, day[0], day[4]) == 5 + 7 + 11 and turns_since(history, day[3], day[5]) == 0
    naps = [_summary(day[1], kind="nap", status="napped"), _summary(day[2], kind="nap", status="napped")]
    assert days_since_deep(naps, day[5]) == 4  # never deep: counted from her first logged night
    assert notice_from(history, day[4]) == day[4] and notice_from([], day[0]) == day[0]  # her last night woke
    assert notice_from(history[:2], day[3]) == day[1]  # the day after the last night that woke
    assert notice_from([*history, *naps[1:]], day[3]) == day[1]  # day 2's last run napped
    assert notice_from(naps, day[3]) == day[1]  # only naps: everything since her first night


def test_month_spent_counts_only_her_keys_this_month() -> None:
    on, other = NIGHT.isoformat(), "2026-09-30"
    settled = {
        f"night:extract:{UID}:{on}:a1": 0.02, f"night:inquire:{UID}:{on}:spark:a1": 0.03,
        f"morning:{on}:{UID}:judge:a1": 0.25, f"dailies:{on}:{UID}:a1": 0.05, f"modal_train:{on}:{UID}:a1": 0.12,
        f"night_gpu:{on}:{batch_digest([UID])}:a1": 0.10,  # her own batch
        f"night_gpu:{on}:{batch_digest([UID, 'p2'])}:a1": 9.0,  # a shared batch: not split per person
        f"night:extract:p2:{on}:a1": 9.0, f"night:extract:{UID}:{other}:a1": 9.0,  # another person, last month
        f"night:extract:{UID}:2026-10-29:a1": 9.0,  # after the night
        f"sim:{UID}:d0:s1:a1": 9.0, "colab_train:x": 9.0,  # the simulated day; not a dollar key of hers
        f"modal_latents:{on}:{UID}:a1": 0.07,  # a live night's latent encode, once one runs it
        "modal_latents:0123abcd:a1": 9.0,  # the research runner's job: no one's night
    }
    assert month_spent(settled, UID, NIGHT) == pytest.approx(0.02 + 0.03 + 0.25 + 0.05 + 0.12 + 0.10 + 0.07)


def test_a_rerun_counts_once_a_shared_session_splits_and_colab_units_are_priced() -> None:
    on, pair = NIGHT.isoformat(), batch_digest([UID, "p2"])
    settled = {
        f"night:extract:{UID}:{on}:a1": 0.02, f"night:extract:{UID}:{on}:a2": 0.03,  # a rerun: its last run counts
        f"night_gpu:{on}:{pair}:a1": 0.90,  # her batch with one other person: half is hers
        f"night_gpu:{on}:{batch_digest(['p2', 'p3'])}:a1": 5.0,  # a batch she wasn't in
    }
    units = {f"colab_train:{on}:{UID}:a1": 2.0, f"colab_life:{on}:{UID}:a1": 1.0, f"colab_train:{on}:p2:a1": 9.0}
    history = [{"night": on, "gpu_batch": {"digest": pair, "people": 2}},
               {"night": on, "gpu_batch": {"digest": "x", "people": 0}}]  # not a batch: left out
    assert gpu_batches(history) == {pair: 2}
    spent = month_spent(settled, UID, NIGHT, batches=gpu_batches(history), units=units)
    assert spent == pytest.approx(0.03 + 0.45 + 3.0 * COLAB_USD_PER_CU)


def test_a_shared_batch_records_its_gpu_session_for_each_person(tmp_path: Path) -> None:
    world = World(tmp_path)
    for uid in (UID, "p2"):
        world.plan(uid, NIGHT, count=1)
    results = run_night_batch_with(world.deps(), {UID: NIGHT, "p2": NIGHT})
    expected = {"digest": batch_digest([UID, "p2"]), "people": 2}
    assert all(summary(result.log_path)["gpu_batch"] == expected for result in results.values())
    world.plan("p3", NIGHT, count=1)
    alone = run_night_batch_with(world.deps(), {"p3": NIGHT})["p3"]  # her key's digest is hers alone: nothing to keep
    assert alone.status == "promoted" and "gpu_batch" not in summary(alone.log_path)


def test_the_month_command_prints_her_spend_and_the_room_left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    import json

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    ledger = Ledger(tmp_path / "sleep" / "budget.jsonl")
    on, pair = NIGHT.isoformat(), batch_digest([UID, "p2"])
    _settle(ledger, f"night:extract:{UID}:{on}:a1", 0.5)
    _settle(ledger, f"morning:{on}:{UID}:judge:a1", 1.0)
    _settle(ledger, f"night_gpu:{on}:{pair}:a1", 3.0)
    _settle(ledger, f"colab_train:{on}:{UID}:a1", 10.0, pool="colab_cu")
    folder = tmp_path / "nights" / on
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"night": on, "gpu_batch": {"digest": pair, "people": 2}}))
    assert past.main(["--month", UID, "--for-date", on, "--nights", str(tmp_path / "nights")]) == 0
    report = json.loads(capsys.readouterr().out)
    spent = 0.5 + 1.0 + 1.5 + 10.0 * COLAB_USD_PER_CU
    assert report["spent_usd"] == pytest.approx(spent) and report["cap_usd"] == POLICY.month_usd
    assert report["left_usd"] == pytest.approx(POLICY.month_usd - spent) and report["month"] == on[:7]
    assert report["by_kind"] == pytest.approx({"night": 0.5, "morning": 1.0, "night_gpu": 1.5, "colab_train": 1.0})
    assert report["deep_night_usd"] == {"modelled fresh night": POLICY.deep_usd, "measured, practice reused": 1.19}
    nights_left = 31 - NIGHT.day  # October's nights after this one, held back as naps
    assert report["deep_nights_left"] == int((POLICY.month_usd - spent - nights_left * NAP_USD) // POLICY.deep_usd)
    assert UID not in json.dumps(report)  # numbers only
