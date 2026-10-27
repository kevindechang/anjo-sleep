"""The batch's GPU session (Task 5): ``modal_usd`` holds the worst case first, and is settled with
what the session really held; a pool that can't hold it stops the batch before any GPU work."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.night.gpu import ReservedGpu
from anjo.sleep.stops import BudgetStop

NIGHT = date(2026, 10, 28)


def test_the_session_reserves_the_worst_case_and_settles_what_it_held(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    clock = iter([100.0, 100.0 + 1800.0])
    gpu = ReservedGpu(ledger, NIGHT, usd_per_hour=2.0, hold_per_person_s=3600.0, timer=lambda: next(clock),
                      stamp=lambda: 1)
    with gpu.open(["a", "b"]):
        assert ledger.used("modal_usd") == pytest.approx(4.0)  # 2 people x 1 hour x $2
    assert ledger.used("modal_usd") == pytest.approx(1.0)  # held half an hour


def test_a_batch_the_pool_cannot_hold_stops_before_any_gpu_work(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "modal_usd": 1.0})
    gpu = ReservedGpu(ledger, NIGHT, usd_per_hour=2.0, hold_per_person_s=3600.0)
    ran = []
    with pytest.raises(BudgetStop), gpu.open(["a"]):
        ran.append(True)
    assert ran == [] and ledger.used("modal_usd") == 0.0


def test_an_empty_batch_holds_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    with ReservedGpu(ledger, NIGHT, usd_per_hour=2.0).open([]):
        pass
    assert ledger.used("modal_usd") == 0.0
