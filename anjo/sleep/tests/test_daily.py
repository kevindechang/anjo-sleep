# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The real-time daily schedule: the night phase closes yesterday's Taipei day under its own date,
the day phase plays today's index, and no account id is in the code. Fakes only."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.night import daily
from anjo.sleep.sim.day import DayRun
from anjo.sleep.types import NightResult

AT_2AM = datetime(2026, 10, 30, 2, 0, tzinfo=daily.TAIPEI)  # 18:00 UTC on the 29th
AT_2PM = datetime(2026, 10, 30, 14, 0, tzinfo=daily.TAIPEI)


def test_night_phase_uses_previous_day_window_and_virtual_date_equals_real_date(tmp_path: Path) -> None:
    plan = daily.night_plan(AT_2AM)
    assert plan.night == date(2026, 10, 29)  # the real date the day was lived on
    assert plan.window == (datetime(2026, 10, 28, 16, 0, tzinfo=timezone.utc),
                           datetime(2026, 10, 29, 16, 0, tzinfo=timezone.utc))  # 29 Oct 00:00 to 30 Oct 00:00, Taipei
    seen: dict[str, Any] = {}

    def nights(**kw: Any) -> dict[str, NightResult]:
        seen.update(kw)
        return {"acct": NightResult("promoted", None, None, {}, (), 0.0, {}, tmp_path / "log.jsonl")}

    scored: list[tuple[Any, ...]] = []
    result = daily.run_night_phase(
        "acct", "founder", AT_2AM, nights=nights, morning=lambda u, p, n: ("ports", u, p, n),
        score=lambda *a, **kw: scored.append((a, kw)))
    assert result.status == "promoted"
    assert seen["night"] == plan.night and seen["windows"] == {"acct": plan.window}
    assert seen["users"] == ["acct"] and seen["personas"] == {"acct": "founder"}
    (args, kw), = scored  # the morning run follows the night, on the same date
    assert args == ("acct", plan.night, None) and kw["promoted"] is True
    assert kw["morning"] == ("ports", "acct", "founder", plan.night)


def test_a_failed_night_is_not_scored(tmp_path: Path) -> None:
    scored: list[object] = []
    failed = NightResult("failed", "practice", None, {}, (), 0.0, {}, tmp_path / "log.jsonl")
    result = daily.run_night_phase("acct", "founder", AT_2AM, nights=lambda **kw: {"acct": failed},
                                   morning=lambda *a: None, score=lambda *a, **kw: scored.append(a))
    assert result.status == "failed" and scored == []


def test_day_phase_plays_todays_index() -> None:
    assert daily.day_plan(AT_2PM) == daily.DayPlan(date(2026, 10, 30), 2)
    asked: list[tuple[Any, ...]] = []

    def simulate(user: str, persona: str, index: int, seed: int) -> DayRun:
        asked.append((user, persona, index, seed))
        return DayRun((AT_2PM, AT_2PM), 0, 0, ())

    daily.run_day_phase("acct", "founder", AT_2PM, simulate=simulate)
    assert asked == [("acct", "founder", 2, daily.SEED)]
    with pytest.raises(ValueError, match="starts on"):
        daily.run_day_phase("acct", "founder", datetime(2026, 10, 27, 14, 0, tzinfo=daily.TAIPEI), simulate=simulate)


def test_users_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert daily.daily_users({daily.USERS_ENV: "founder=abc, other=def"}) == {"founder": "abc", "other": "def"}
    assert daily.daily_users({}) == {}
    monkeypatch.delenv(daily.USERS_ENV, raising=False)
    with pytest.raises(SystemExit):  # no account for the persona: the run stops
        daily.main(["--phase", "day", "--persona", "founder"])
