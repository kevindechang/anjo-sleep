# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Sleep pressure: a nap or a deep sleep, decided once a night (``night/sleep_pressure.py``), on the
night's own fakes. A nap runs MIND and the note and nothing on the GPU; the month's cap always wins."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.night import cost
from anjo.sleep.night.deps import NightSettings
from anjo.sleep.night.gpu import batch_digest
from anjo.sleep.night.holds import usd
from anjo.sleep.night.past import days_since_deep, last_deep, month_spent, turns_since
from anjo.sleep.night.run import GPU_STEPS, run_night_batch_with
from anjo.sleep.night.sleep_pressure import (
    DEEP_NIGHT_USD,
    SleepPolicy,
    SleepPolicyError,
    SleepState,
    decide,
    high_risk,
    policy_from,
    sleep_pressure,
)
from anjo.sleep.tests.mind_night_fakes import LifeThinker, mind_night
from anjo.sleep.tests.night_fakes import NIGHT, World, step, steps, summary, thread
from anjo.sleep.tests.test_night_mind import DUE_SOON, ON, _world
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


def _settle(ledger: Ledger, job: str, amount: float) -> None:
    held = ledger.reserve(job, ("modal_usd",), usd(amount))
    assert held is not None
    held.settle(amount)


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
    _settle(ledger, f"night:practice:{UID}:{yesterday}:a1", 13.5)
    _settle(ledger, f"night:practice:someone:{yesterday}:a1", 400.0)  # another person's: not hers
    result = _night(world, turns=_turns(60), ledger=ledger)
    sleep = step(result.log_path, "sleep")
    assert (sleep["kind"], sleep["reason"]) == ("nap", "month_cap")
    assert sleep["pressure"] == 1.0 >= sleep["threshold"] > POLICY.base_threshold  # it would have gone deep
    assert sleep["month_spent_usd"] == pytest.approx(13.5) and sleep["month_usd"] == POLICY.month_usd
    assert result.status == "napped" and world.gpu.opened == [] and world.trainer.calls == []
    assert summary(result.log_path)["sleep"]["reason"] == "month_cap"  # logged, never silent


def test_nap_still_runs_mind_and_notes(tmp_path: Path) -> None:
    world = _world(tmp_path, due=DUE_SOON)
    mind = mind_night(world, LifeThinker(targets=(f"{UID}-t1", f"{UID}-c1")))
    result = _night(world, settings=ON, inquiry=mind, notice=mind)
    assert result.status == "napped" and world.gpu.opened == [] and world.trainer.calls == []
    assert step(result.log_path, "notice")["noop"] is False
    choose = step(result.log_path, "choose")
    assert choose["action"] == "inquire" and choose["inquiry"]["ops"]["CONCLUDE"] == 1
    note = step(result.log_path, "note")
    assert note["outcome"] == "held" and note["target_id"] == f"{UID}-c1"
    assert len(world.notifier.queued) == 1


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
    assert policy_from({}) == SleepPolicy(month_usd=15.0, max_days=7)
    assert policy_from({"ANJO_SLEEP_MONTH_USD": "10", "ANJO_SLEEP_MAX_DAYS": "3"}) == SleepPolicy(10.0, 3)
    for bad in ({"ANJO_SLEEP_MONTH_USD": "ten"}, {"ANJO_SLEEP_MONTH_USD": "0"}, {"ANJO_SLEEP_MAX_DAYS": "0"},
                {"ANJO_SLEEP_MAX_DAYS": "2.5"}, {"ANJO_SLEEP_MONTH_USD": "nan"}):
        with pytest.raises(SleepPolicyError):
            policy_from(bad)


def test_deep_night_usd_covers_the_cost_models_deep_night() -> None:
    measured = cost.Measured(extract=0.02, brief=0.06, training=None)  # the live night's own numbers, rounded up
    projected = cost.project(cost.CURRENT, cost.UNITS, cost.DEFAULT_PRICES, measured)
    assert DEEP_NIGHT_USD >= sum(projected.trained.values()) + projected.morning


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
    }
    assert month_spent(settled, UID, NIGHT) == pytest.approx(0.02 + 0.03 + 0.25 + 0.05 + 0.12 + 0.10)
