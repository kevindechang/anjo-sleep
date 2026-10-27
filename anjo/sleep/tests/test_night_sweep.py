# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The sweep at the start of every night command (Task 5): a run killed hard leaves its holds open
and maybe a VM up; the next start stops the VMs, settles those holds once at their full bound,
never touches a live run's, and the night log records the counts."""
from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.budget import Bound, Ledger, Pool
from anjo.sleep.night.gpu import ReservedGpu
from anjo.sleep.night.holds import attempt_key
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.night.sweep import NightBusy, Sweep, swept_night
from anjo.sleep.tests.night_fakes import NIGHT, World, step, steps
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, hold_base

GPU = Bound(1.95, 3 * 3600.0)
COLAB = Bound(A100_CU_PER_HOUR, 7200.0)


def _killed(ledger: Ledger) -> None:
    """What a night killed mid-training leaves: its GPU, Colab and step holds, all open."""
    killed: tuple[tuple[str, tuple[Pool, ...], Bound], ...] = (
        (ReservedGpu(ledger, NIGHT, usd_per_hour=1.95).job(["u1"], 1), ("modal_usd",), GPU),
        (attempt_key(hold_base(NIGHT.isoformat(), "u1"), 1), ("colab_cu",), COLAB),
        (attempt_key(f"night:brief:u1:{NIGHT.isoformat()}", 1), ("baseten_sleep_usd",), Bound(0.1, 3600.0)),
    )
    for job, pools, bound in killed:
        assert ledger.reserve(job, pools, bound) is not None


def test_a_killed_runs_gpu_and_colab_holds_are_settled_once_at_the_next_start(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    _killed(ledger)
    other = ledger.reserve("mind_run:q1", ("baseten_mind_usd",), Bound(0.2, 3600.0))  # not the night's
    with swept_night(ledger, lambda: (0, 0)) as swept:
        assert swept == Sweep(holds_settled=3, sessions_stopped=0, sessions_not_ours=0)
    assert ledger.used("modal_usd") == pytest.approx(GPU.amount)  # the worst case: work may have run
    assert ledger.used("colab_cu") == pytest.approx(COLAB.amount)
    with swept_night(ledger, lambda: (0, 0)) as again:
        assert again.holds_settled == 0  # settled once, never charged twice
    assert ledger.used("modal_usd") == pytest.approx(GPU.amount) and ledger.used("colab_cu") == pytest.approx(COLAB.amount)
    assert [hold.job for hold in ledger.open_holds()] == [other.job if other else ""]
    with ReservedGpu(ledger, NIGHT, usd_per_hour=1.95).open(["u1"]):  # the rerun takes a fresh hold
        assert ledger.used("modal_usd") == pytest.approx(2 * GPU.amount)


def test_a_live_runs_holds_are_never_swept(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    stops: list[int] = []

    def stop() -> tuple[int, int]:
        stops.append(1)
        return 0, 0

    with swept_night(ledger, lambda: (0, 0)):
        _killed(ledger)  # this run's own holds, open while it is live
        with pytest.raises(NightBusy), swept_night(ledger, stop):
            pass
        assert len(ledger.open_holds()) == 3 and stops == []  # nothing touched, no VM stopped


def test_vms_are_stopped_before_any_hold_is_settled(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    _killed(ledger)
    seen: list[int] = []

    def stop() -> tuple[int, int]:
        seen.append(len(ledger.open_holds()))
        return 1, 2

    with swept_night(ledger, stop) as swept:
        assert (swept.sessions_stopped, swept.sessions_not_ours) == (1, 2)
    assert seen == [3]

    def fails() -> tuple[int, int]:
        raise RuntimeError("stop it by hand")

    second = Ledger(tmp_path / "other" / "budget.jsonl")
    _killed(second)
    with pytest.raises(RuntimeError), swept_night(second, fails):
        pass
    assert len(second.open_holds()) == 3  # a VM that may run keeps its hold open


def test_the_night_log_records_the_sweep_as_counts(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    result = run_night_batch_with(world.deps(), {"p1": NIGHT}, sweep=Sweep(3, 1, 2))["p1"]
    assert steps(result.log_path)[0]["step"] == "sweep"
    assert step(result.log_path, "sweep") == {"step": "sweep", "status": "ok", "uses": "new", "system": None,
                                             "noop": False, "holds_settled": 3, "sessions_stopped": 1,
                                             "sessions_not_ours": 2}
