# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's lock and sweep (Task 17 review): each APD command takes MIND's lock, and only the holder
stops leftover MIND sessions and settles stale MIND holds. The night's sweep and MIND's never touch
each other's holds, and a second MIND command at once is refused and touches nothing."""
from __future__ import annotations

import contextvars
import threading
from pathlib import Path

import pytest

from anjo.sleep.budget import Bound, Ledger
from anjo.sleep.mind.apd.__main__ import BUSY
from anjo.sleep.mind.apd.__main__ import main as apd_main
from anjo.sleep.mind.apd.journal import SpendJournal
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.sweep import MIND_KINDS, MindBusy, lock_path, mind_lock, swept_mind
from anjo.sleep.night.holds import spark_worst_usd
from anjo.sleep.night.sweep import NIGHT_KINDS, swept_night
from anjo.sleep.train.runners.colab import MIND_SESSION_PREFIX, hold_base
from anjo.sleep.tests.test_apd_safety import PRICES

BOUND = Bound(0.5, 3600.0)
NIGHTS = ("night:brief:u1:2026-10-28:a1", f"{hold_base('2026-10-28', 'u1')}:a1", "colab_life:2026-10-28:u1:a1")
MINDS = ("apd:run:life-1:M:2026-10-28:a1", f"{hold_base('2026-10-28', 'apd-replay', MIND_SESSION_PREFIX)}:a1",
         "replay:branches:s1:a1", "colab_serve:anjo-mind-glimmer-ab12:a1", "apd:gen:development:3:0:spark:a1")


def _killed(ledger: Ledger) -> None:
    for job in (*NIGHTS, *MINDS):
        assert ledger.reserve(job, ("baseten_mind_usd",), BOUND) is not None


def _open(ledger: Ledger) -> set[str]:
    return {held.job for held in ledger.open_holds()}


def test_mind_and_night_hold_kinds_share_no_prefix() -> None:
    assert not [(a, b) for a in MIND_KINDS for b in NIGHT_KINDS if a.startswith(b) or b.startswith(a)]
    assert all(job.startswith(MIND_KINDS) and not job.startswith(NIGHT_KINDS) for job in MINDS)
    assert all(job.startswith(NIGHT_KINDS) and not job.startswith(MIND_KINDS) for job in NIGHTS)


def test_the_night_sweep_never_settles_minds_holds(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    _killed(ledger)
    with swept_night(ledger, lambda: (0, 0)) as swept:
        assert swept.holds_settled == len(NIGHTS)
    assert _open(ledger) == set(MINDS)  # a MIND command running beside the night keeps its holds


def test_minds_sweep_never_settles_the_nights_holds(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    _killed(ledger)
    with swept_mind(lock_path(ledger, tmp_path), ledger, lambda: (1, 2), lambda: 0.0) as swept:
        assert swept is not None and swept.holds_settled == len(MINDS) and swept.sessions_stopped == 1
        with swept_mind(lock_path(ledger, tmp_path), ledger, lambda: (9, 9), lambda: 9.0) as inner:
            assert inner is None  # a step inside the command: the lock is the command's, no second sweep
    assert _open(ledger) == set(NIGHTS)


def test_a_second_mind_command_is_refused_and_touches_nothing(tmp_path: Path) -> None:
    data = DataDir(tmp_path)
    ledger = Ledger(tmp_path / "fake-budget.jsonl")  # the fake chain's own ledger, in the data dir
    _killed(ledger)
    held, release = threading.Event(), threading.Event()

    def other_command() -> None:
        with mind_lock(lock_path(ledger, data.root)):
            held.set()
            release.wait(10)

    thread = threading.Thread(target=contextvars.Context().run, args=(other_command,))
    thread.start()
    assert held.wait(10)
    lines: list[str] = []
    try:
        assert apd_main(["--fake", "--data", str(data.root), "judge", "--held-out"], out=lines.append) == BUSY
        with pytest.raises(MindBusy):
            with swept_mind(lock_path(ledger, data.root), ledger, lambda: (0, 0), lambda: 0.0):
                pass
    finally:
        release.set()
        thread.join(10)
    assert "another MIND command is live" in lines[-1]
    assert _open(ledger) == {*NIGHTS, *MINDS}  # nothing was swept


def test_each_command_starts_by_settling_killed_runs_from_their_journals(tmp_path: Path) -> None:
    data = DataDir(tmp_path)
    ledger = Ledger(tmp_path / "fake-budget.jsonl")
    _killed(ledger)
    night = MINDS[0]
    journal = SpendJournal(data.journal("development-1.M"))
    journal.held(night)
    journal.call(night, 1000, 200, 0.01)  # what the killed run priced before it died
    lines: list[str] = []
    assert apd_main(["--fake", "--data", str(data.root), "judge", "--held-out"], out=lines.append) == 0
    assert _open(ledger) == set(NIGHTS)
    settled = {night.job: night.recovered for night in journal.jobs()}
    assert settled == {MINDS[0]: pytest.approx(0.01 + spark_worst_usd(PRICES, 1))}  # its calls, and one in flight
    assert any(line.startswith("MIND's sweep:") and f"{len(MINDS) - 1} stale MIND hold(s)" in line for line in lines)
