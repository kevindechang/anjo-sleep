# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The real-time daily schedule: the night phase closes yesterday's Taipei day under its own date,
the day phase plays today's index, and no account id is in the code. Fakes only."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.checkpoints import DATA_DIR_ENV
from anjo.sleep.night import __main__ as night_command
from anjo.sleep.night import daily
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.sim.day import DayRun, TurnFailed, run_day
from anjo.sleep.types import NightResult

AT_2AM = datetime(2026, 10, 30, 2, 0, tzinfo=daily.TAIPEI)  # 18:00 UTC on the 29th
AT_2PM = datetime(2026, 10, 30, 14, 0, tzinfo=daily.TAIPEI)
PERSONA = load_person_file(Path(__file__).parents[1] / "personas" / "founder.json").visible


class _Model:
    def complete(self, system: str, messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
        return "A line about the pilot."


class _Served:
    def of_reply(self, session_id: str) -> frozenset[str] | None:
        return frozenset({"glimmer"})


def _good_day(tmp_path: Path) -> Path:
    folder = tmp_path / "markers"
    daily.write_marker(folder, "founder", date(2026, 10, 29), 1, DayRun((AT_2PM, AT_2PM), 4, 0, ()))
    return folder


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
        "acct", "founder", AT_2AM, markers=_good_day(tmp_path), nights=nights,
        morning=lambda u, p, n: ("ports", u, p, n),
        score=lambda *a, **kw: scored.append((a, kw)))
    assert result.status == "promoted"
    assert seen["night"] == plan.night and seen["windows"] == {"acct": plan.window}
    assert seen["as_of"] == {"acct": plan.window[1]}  # real time: the night's clock is the window's end
    assert seen["users"] == ["acct"] and seen["personas"] == {"acct": "founder"}
    (args, kw), = scored  # the morning run follows the night, on the same date
    assert args == ("acct", plan.night, None) and kw["promoted"] is True
    assert kw["morning"] == ("ports", "acct", "founder", plan.night)


def test_a_failed_night_is_not_scored(tmp_path: Path) -> None:
    scored: list[object] = []
    failed = NightResult("failed", "practice", None, {}, (), 0.0, {}, tmp_path / "log.jsonl")
    result = daily.run_night_phase("acct", "founder", AT_2AM, markers=_good_day(tmp_path), nights=lambda **kw: {"acct": failed},
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


# ── the marker: the night sees the day it closes ─────────────────────────────

import json  # noqa: E402

from anjo.sleep.night.run import run_night_batch_with  # noqa: E402
from anjo.sleep.tests.night_fakes import World, step  # noqa: E402


def _nights_on(world: World, seen: dict[str, Any]) -> Any:
    def nights(**kw: Any) -> dict[str, NightResult]:
        seen.update(kw)
        world.plan("acct", kw["night"])
        return run_night_batch_with(world.deps(), {"acct": kw["night"]}, windows=kw["windows"],
                                    days=kw.get("days") or None)

    return nights


def test_night_records_missing_day(tmp_path: Path) -> None:
    scored: list[object] = []
    seen: dict[str, Any] = {}
    result = daily.run_night_phase("acct", "founder", AT_2AM, markers=tmp_path / "daily",
                                   nights=_nights_on(World(tmp_path / "w"), seen),
                                   morning=lambda *a: None, score=lambda *a, **kw: scored.append(a))
    assert result.status == "failed" and result.failed_at == "simulate" and scored == []
    assert step(result.log_path, "simulate")["error"] == "day_missing"
    assert seen["days"]["acct"].window == daily.night_plan(AT_2AM).window


@pytest.mark.parametrize(("status", "failure", "code"), [
    ("failed", "glimmer_not_ready", "glimmer_not_ready"), ("short", None, "short_day")])
def test_night_records_failed_day_and_skips_morning(tmp_path: Path, status: str, failure: str | None, code: str) -> None:
    markers = tmp_path / "daily"
    day = DayRun((AT_2PM, AT_2PM), 1, 0 if failure else 2, (), failure=failure)
    daily.write_marker(markers, "founder", date(2026, 10, 29), 1, day)
    assert json.loads((markers / "founder-2026-10-29.json").read_text())["status"] == status
    scored: list[object] = []
    result = daily.run_night_phase("acct", "founder", AT_2AM, markers=markers,
                                   nights=_nights_on(World(tmp_path / "w"), {}),
                                   morning=lambda *a: None, score=lambda *a, **kw: scored.append(a))
    assert result.status == "failed" and result.failed_at == "simulate" and scored == []
    assert step(result.log_path, "simulate")["error"] == code


def test_night_runs_on_a_good_days_marker(tmp_path: Path) -> None:
    markers = tmp_path / "daily"
    daily.write_marker(markers, "founder", date(2026, 10, 29), 1, DayRun((AT_2PM, AT_2PM), 4, 0, ()))
    seen: dict[str, Any] = {}
    result = daily.run_night_phase("acct", "founder", AT_2AM, markers=markers,
                                   nights=_nights_on(World(tmp_path / "w"), seen),
                                   morning=lambda *a: None, score=lambda *a, **kw: None)
    assert result.status != "failed" and not seen.get("days")


def _day_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, day: DayRun, *extra: str) -> int:
    monkeypatch.setenv(daily.USERS_ENV, "founder=acct")
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(night_command, "_named", lambda path: lambda *a: day)
    return daily.main(["--phase", "day", "--persona", "founder", *extra])


@pytest.mark.parametrize("day", [
    DayRun((AT_2PM, AT_2PM), 2, 1, ()),  # Glimmer died mid-day: a short day
    DayRun((AT_2PM, AT_2PM), 0, 0, (), failure="glimmer_not_ready")])
def test_day_phase_exits_1_unless_the_day_ran_whole(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, day: DayRun) -> None:
    assert _day_main(monkeypatch, tmp_path, day, "--for-date", "2026-10-30") == 1
    assert (tmp_path / "sleep" / "daily" / "founder-2026-10-30.json").is_file()


def test_day_phase_marker_carries_the_intended_date(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ok = DayRun((AT_2PM, AT_2PM), 4, 0, ())
    assert _day_main(monkeypatch, tmp_path, ok, "--for-date", "2026-10-30") == 0
    marker = json.loads((tmp_path / "sleep" / "daily" / "founder-2026-10-30.json").read_text())
    assert marker == {"persona": "founder", "date": "2026-10-30", "day_index": 2, "status": "ok",
                      "failure": None, "failed_turns": 0}


# ── dailies: the morning card's before and after, while Glimmer is still awake ─


def _promoted(tmp_path: Path) -> Any:
    folder = tmp_path / "night"
    folder.mkdir()
    return lambda **kw: {"acct": NightResult("promoted", None, None, {}, (), 0.0, {}, folder / "log.jsonl")}


def test_night_phase_runs_dailies_after_morning_and_survives_its_failure(tmp_path: Path) -> None:
    from anjo.sleep.serving.backend import phase

    order: list[str] = []

    def score(*args: Any, **kw: Any) -> None:
        with phase(lambda: order.append("sleep")):  # the morning's own phase: its sleep waits for the outer one
            order.append("morning")

    def dailies(user: str, persona: str, night: date, result: NightResult) -> None:
        order.append("dailies")
        assert (user, persona, night, result.status) == ("acct", "founder", daily.night_plan(AT_2AM).night, "promoted")
        raise RuntimeError("Glimmer lost the thread")

    ticks = iter([0.0, 90.0, 90.0, 102.5])
    result = daily.run_night_phase("acct", "founder", AT_2AM, markers=_good_day(tmp_path), nights=_promoted(tmp_path),
                                   morning=lambda *a: None, score=score, dailies=dailies, timer=lambda: next(ticks))
    assert result.status == "promoted"  # a failed line never fails the night
    assert order == ["morning", "dailies", "sleep"]  # after the morning, before Glimmer is put to sleep
    steps = [json.loads(line) for line in result.log_path.read_text().splitlines()]
    assert [(s["step"], s["status"]) for s in steps] == [("morning", "ok"), ("dailies", "failed")]
    assert steps[0]["seconds"] == 90.0 and steps[1]["seconds"] == 12.5  # the card's GPU-minutes, measured
    assert steps[1]["error"] == "RuntimeError" and "thread" not in json.dumps(steps)  # a type, never a message

    ran: list[str] = []
    daily.run_night_phase("acct", "founder", AT_2AM, markers=_good_day(tmp_path / "again"),
                          nights=_promoted(tmp_path / "again"), morning=lambda *a: None, score=lambda *a, **kw: None,
                          dailies=lambda *a: ran.append("dailies"))
    steps = [json.loads(line) for line in (tmp_path / "again" / "night" / "log.jsonl").read_text().splitlines()]
    assert ran == ["dailies"] and [(s["step"], s["status"]) for s in steps] == [("morning", "ok"), ("dailies", "ok")]


@pytest.mark.parametrize("status", ["rejected", "nothing_new"])
def test_night_phase_generates_no_line_without_a_new_checkpoint(tmp_path: Path, status: str) -> None:
    ran: list[str] = []
    night = NightResult(status, None, None, {}, (), 0.0, {}, tmp_path / "log.jsonl")  # type: ignore[arg-type]
    daily.run_night_phase("acct", "founder", AT_2AM, markers=_good_day(tmp_path), nights=lambda **kw: {"acct": night},
                          morning=lambda *a: None, score=lambda *a, **kw: None, dailies=lambda *a: ran.append("x"))
    assert ran == []  # nothing to say differently: the card says so, and nothing is generated


# ── --for-date on the night phase, and why a day failed ─


def test_night_plan_for_date_closes_that_day() -> None:
    plan = daily.night_plan(AT_2AM, date(2026, 10, 28))  # the wall says the 30th; he asked for the 28th
    assert plan.night == date(2026, 10, 28)
    assert plan.window == (datetime(2026, 10, 27, 16, 0, tzinfo=timezone.utc),
                           datetime(2026, 10, 28, 16, 0, tzinfo=timezone.utc))  # 28 Oct, Taipei, whole day


def test_night_phase_without_for_date_is_unchanged(tmp_path: Path) -> None:
    assert daily.night_plan(AT_2AM, None) == daily.night_plan(AT_2AM)
    assert daily.night_plan(AT_2AM).night == date(2026, 10, 29)
    seen: dict[str, Any] = {}

    def nights(**kw: Any) -> dict[str, NightResult]:
        seen.update(kw)
        return {"acct": NightResult("failed", "practice", None, {}, (), 0.0, {}, tmp_path / "log.jsonl")}

    daily.run_night_phase("acct", "founder", AT_2AM, markers=_good_day(tmp_path), nights=nights)
    assert seen["night"] == date(2026, 10, 29)


def test_night_for_date_in_future_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="future"):
        daily.night_plan(AT_2AM, date(2026, 10, 31))
    with pytest.raises(ValueError, match="starts on"):
        daily.night_plan(AT_2AM, date(2026, 10, 27))
    monkeypatch.setenv(daily.USERS_ENV, "founder=acct")
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    with pytest.raises(SystemExit) as refused:
        daily.main(["--phase", "night", "--persona", "founder", "--for-date", "2999-01-01"])
    assert refused.value.code == 2


def test_failed_day_marker_names_the_reason_code(tmp_path: Path) -> None:
    class Stale:
        def send(self, session_id: str, text: str, *, cleared: bool) -> str:
            raise TurnFailed("a sentence that must not travel", reason="stale session")

    day = run_day(PERSONA, 1, 7, model=_Model(), transport=Stale(), replies=_Served(), clock=lambda: AT_2PM)
    assert day.failure_detail == "stale session"
    markers = tmp_path / "daily"
    daily.write_marker(markers, "founder", date(2026, 10, 29), 1, day)
    text = (markers / "founder-2026-10-29.json").read_text()
    assert json.loads(text)["failure_detail"] == "stale session" and "sentence" not in text
    assert daily.missed_day(markers, "founder", daily.night_plan(AT_2AM)).failure_detail == "stale session"  # type: ignore[union-attr]


def test_old_marker_without_reason_still_reads(tmp_path: Path) -> None:
    markers = tmp_path / "daily"
    markers.mkdir()
    (markers / "founder-2026-10-29.json").write_text(json.dumps(
        {"persona": "founder", "date": "2026-10-29", "day_index": 1, "status": "short",
         "failure": None, "failed_turns": 1}))
    missed = daily.missed_day(markers, "founder", daily.night_plan(AT_2AM))
    assert missed is not None and missed.failure == daily.SHORT_DAY and missed.failure_detail is None
    ok = DayRun((AT_2PM, AT_2PM), 4, 0, ())
    daily.write_marker(markers, "founder", date(2026, 10, 29), 1, ok)  # an ok day: the field stays out
    assert "failure_detail" not in json.loads((markers / "founder-2026-10-29.json").read_text())


def test_night_phase_mind_flag_reaches_the_night(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[dict[str, Any]] = []

    def nights(**kw: Any) -> dict[str, NightResult]:
        seen.append(kw)
        return {"acct": NightResult("failed", "practice", None, {}, (), 0.0, {}, tmp_path / "log.jsonl")}

    for mind in (False, True):
        daily.run_night_phase("acct", "founder", AT_2AM, markers=_good_day(tmp_path), nights=nights, mind=mind)
    assert [kw["mind"] for kw in seen] == [False, True]
    monkeypatch.setenv(daily.USERS_ENV, "founder=acct")
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    real = daily.run_night_phase  # main's default `nights` is bound at import, so hand it the fake here
    monkeypatch.setattr(daily, "run_night_phase", lambda *a, **kw: real(*a, **kw, nights=nights))
    for env, flags, expected in ((None, [], False), ("1", [], True), (None, ["--mind"], True)):
        monkeypatch.delenv(daily.MIND_ENV, raising=False)
        if env:
            monkeypatch.setenv(daily.MIND_ENV, env)
        seen.clear()
        daily.main(["--phase", "night", "--persona", "founder", "--for-date", "2026-10-28", *flags])
        assert seen[0]["mind"] is expected
        assert json.loads(capsys.readouterr().out)["mind"] is expected
