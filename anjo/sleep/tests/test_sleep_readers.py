# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Every reader shows how a night slept: "napped" or "deep sleep", with its pressure and its reason
(the what-changed lines, the mind panel, the Dailies card, the copy), and the cost model projects a
month from a pressure trace by the same decision."""
from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.demo.dailies import DailySources, NightFiles, dailies_view
from anjo.sleep.demo.heldout import HeldOutText
from anjo.sleep.demo.panel import MindSources, PanelError, mind_view, night_record
from anjo.sleep.night import cost
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.night.sleep_pressure import REASONS, SleepMark, SleepPolicy, SleepRecordError, sleep_mark
from anjo.sleep.ownership.changes import ChangesError, night_views, what_changed
from anjo.sleep.tests.night_fakes import NIGHT, World, steps, summary

UID = "p1"
DEEP = NIGHT + timedelta(days=1)


def _life(tmp_path: Path) -> tuple[Path, Any, Any]:
    """A nap on a quiet day, then a deep sleep on a busy one, logged where the readers look."""
    world = World(tmp_path)
    person = tmp_path / "results" / UID
    world.plan(UID, NIGHT, count=1)
    world.plan(UID, DEEP, count=12)
    deps = world.deps(sleep_policy=SleepPolicy(), logs=lambda uid: person / "nights", turns=lambda uid, window: 40)
    nap = run_night_batch_with(deps, {UID: NIGHT})[UID]
    deep = run_night_batch_with(deps, {UID: DEEP})[UID]
    return person, nap, deep


def test_the_nights_and_what_changed_say_napped_or_deep_with_the_pressure(tmp_path: Path) -> None:
    person, nap, deep = _life(tmp_path)
    views = night_views(person)
    assert [(v.status, v.sleep.kind if v.sleep else None) for v in views] == [("napped", "nap"), ("promoted", "deep")]
    assert views[0].sleep == SleepMark("nap", views[0].sleep.pressure, 0.5, "below_threshold")  # type: ignore[union-attr]
    changes = what_changed(UID, NIGHT, results=lambda uid: tmp_path / "results" / uid)
    [napped] = [c for c in changes if c.kind == "napped"]
    pressure = summary(nap.log_path)["sleep"]["pressure"]
    assert napped.night == NIGHT and f"{pressure:.2f}" in napped.line and "0.50" in napped.line
    [slept] = [c for c in changes if c.night == DEEP and c.kind == "promoted"]
    assert "deep sleep" in slept.line and f"{summary(deep.log_path)['sleep']['pressure']:.2f}" in slept.line


def test_a_night_from_before_naps_reads_as_it_did(tmp_path: Path) -> None:
    world = World(tmp_path)
    person = tmp_path / "results" / UID
    world.plan(UID, NIGHT, count=1)
    run_night_batch_with(world.deps(logs=lambda uid: person / "nights"), {UID: NIGHT})
    [view] = night_views(person)
    assert view.sleep is None and view.status == "promoted"
    [line] = [c for c in what_changed(UID, NIGHT, results=lambda uid: person) if c.kind == "promoted"]
    assert "deep sleep" not in line.line


def test_a_malformed_sleep_record_fails_where_it_is_read(tmp_path: Path) -> None:
    person, nap, _ = _life(tmp_path)
    path = nap.log_path.parent / "summary.json"
    record = json.loads(path.read_text())
    record["sleep"]["kind"] = "hibernate"
    path.write_text(json.dumps(record))
    with pytest.raises(ChangesError):
        night_views(person)
    with pytest.raises(PanelError):
        night_record(record, steps(nap.log_path))
    assert sleep_mark(None) is None  # a night from before naps, or run without a policy
    with pytest.raises(SleepRecordError):
        sleep_mark({"kind": "nap", "reason": "below_threshold", "pressure": 1.5, "threshold": 0.5})


def test_the_panel_and_the_dailies_card_show_how_each_night_slept(tmp_path: Path) -> None:
    _, nap, deep = _life(tmp_path)
    files = [NightFiles(summary(r.log_path), steps(r.log_path)) for r in (nap, deep)]
    shown = mind_view(MindSources(nights=[f.record for f in files]), HeldOutText([]))
    sleeps = {item["date"]: item["sleep"] for item in shown["choices"]["items"]}
    assert sleeps[NIGHT.isoformat()]["kind"] == "nap" and sleeps[DEEP.isoformat()]["kind"] == "deep"
    assert set(sleeps[NIGHT.isoformat()]) == {"kind", "pressure", "threshold", "reason"}
    card = dailies_view(DailySources(person=UID, nights=files), HeldOutText([]), passes=lambda text: True, night=1)
    assert card["status"] == "napped" and card["sleep"]["kind"] == "nap" and card["sleep"]["reason"] == "below_threshold"
    latest = dailies_view(DailySources(person=UID, nights=files), HeldOutText([]), passes=lambda text: True)
    assert latest["sleep"]["kind"] == "deep" and latest["nights"][-1]["sleep"] == "nap"


def test_the_copy_names_every_kind_and_reason() -> None:
    assert words.COPY["night_status_napped"] and words.COPY["dailies_slate_sleep"]
    assert all(words.COPY[f"sleep_kind_{kind}"] for kind in ("nap", "deep"))
    assert all(words.COPY[f"sleep_reason_{reason}"] for reason in REASONS)


def test_the_cost_model_projects_a_month_from_a_pressure_trace() -> None:
    measured = cost.Measured(extract=0.02, brief=0.06, training=None)
    projected = cost.project(cost.CURRENT, cost.UNITS, cost.DEFAULT_PRICES, measured)
    quiet, busy = cost.DayLoad(turns=4, evidence=1), cost.DayLoad(turns=60, evidence=12)
    calm = cost.month_from_trace([quiet] * 30, SleepPolicy(), projected)
    assert calm.deep == 4 and calm.naps == 26  # the seventh day without a deep sleep forces one
    assert [n.reason for n in calm.nights if n.kind == "deep"] == ["max_days"] * 4
    hectic = cost.month_from_trace([busy] * 30, SleepPolicy(), projected)
    last = max(i for i, n in enumerate(hectic.nights) if n.kind == "deep")
    # the threshold rises, then the cap holds every deep sleep (naps after it still pay their own lines)
    assert math.fsum(n.usd for n in hectic.nights[:last + 1]) <= SleepPolicy().month_usd and hectic.deep < 30
    assert any(n.reason == "month_cap" or n.reason == "below_threshold" for n in hectic.nights)
    assert [n.pressure for n in hectic.nights][:1] == [pytest.approx(0.75)]
    deep_usd = sum(projected.trained.values()) + projected.morning
    assert hectic.usd == pytest.approx(hectic.deep * deep_usd + hectic.naps * sum(projected.quiet.values()))
    with pytest.raises(cost.CostError):
        cost.read_trace('[{"turns": -1, "evidence": 0}]')
    assert cost.read_trace('[{"turns": 3, "evidence": 2, "high_risk": 1}]') == [cost.DayLoad(3, 2, 1)]
