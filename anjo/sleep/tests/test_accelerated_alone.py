# mypy: disallow-untyped-decorators=False
"""Quiet synthetic days still have nights, inquiry, and dated alone labels (offline)."""
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

from anjo.sleep.personas.model import Persona
from anjo.sleep.sim.day import DayRun

import pytest

from anjo.sleep.mind.inquiry.types import RunStart
from anjo.sleep.night.nightlog import NightLog
from anjo.sleep.night.run import day_window
from anjo.sleep.night.sleep_pressure import SleepPolicy
from anjo.sleep.reports.week import ProbeTrace, render, to_json
from anjo.sleep.sim import accelerated
from anjo.sleep.tests.accelerated_rig import FakeAccelerated
from anjo.sleep.tests.mind_night_fakes import workspace
from anjo.sleep.tests.test_accelerated import HERO, START
from anjo.sleep.tools import week


@pytest.mark.parametrize('policy', [False, True])
def test_quiet_days_keep_inquiry_and_week_labels(tmp_path: Path, policy: bool) -> None:
    fake = FakeAccelerated(tmp_path, mind_from=START)
    rig = fake.rig()
    simulated: list[int] = []

    def simulate(uid: str, person: Persona, i: int, seed: int) -> DayRun:
        simulated.append(i)
        return fake.simulate(uid, person, i, seed)

    deps_for = rig.deps_for
    rig = replace(rig, simulate=simulate, deps_for=lambda night: replace(
        deps_for(night), sleep_policy=SleepPolicy(base_threshold=1.0) if policy else None))
    results = accelerated.run_accelerated('fake-life', HERO, 4, START, rig=rig, alone_from=3)
    assert len(results) == 4 and all(r.status != 'failed' for r in results)
    assert simulated == [0, 1]
    summaries = [json.loads((r.log_path.parent / 'summary.json').read_text()) for r in results]
    assert [s['alone'] for s in summaries] == [False, False, True, True]
    for result in results[2:]:
        steps = NightLog(result.log_path.parent).steps()
        sim = next(s for s in steps if s['step'] == 'simulate')
        assert sim['answered'] == sim['failed_turns'] == 0
        assert next(s for s in steps if s['step'] == 'extract')['items'] == 0
        assert next(s for s in steps if s['step'] == 'notice')['status'] == 'ok'
        assert next(s for s in steps if s['step'] == 'inquire')['status'] == 'ok'
        assert next(s for s in steps if s['step'] == 'morning')['status'] == 'ok'
    events = tuple(workspace(fake.world, 'fake-life').events())
    inquiry_nights = {entry.night for e in events if isinstance(entry := e.step.entry, RunStart)
                      and entry.call == 'inquiry'}
    alone = tuple(START + timedelta(days=i) for i in (2, 3))
    assert set(alone) <= inquiry_nights
    source = week.Records(Path('/fake/workspace.jsonl'), events, (), (), (), None, alone_nights=alone)
    end = START + timedelta(days=3)
    chosen = week._window(source, START, end, day_window(end)[1])
    assert [n.alone for n in chosen.nights] == [False, False, True, True]
    trace = ProbeTrace('fake prompt', 'base', 'fake', 0, 0.0, 'week', 'empty', 'a', 'b')
    markdown = render(chosen, trace)
    assert all(f'## {n.isoformat()} (alone)' in markdown for n in alone)
    assert [n['alone'] for n in json.loads(to_json(chosen, trace))['nights']] == [False, False, True, True]


def test_fake_command_alone(capsys: pytest.CaptureFixture[str]) -> None:
    assert accelerated.main(['--fake', '--days', '4', '--mind', '--alone-from', '3']) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report['nights']) == 4
    assert [n['alone'] for n in report['nights']] == [False, False, True, True]
    assert report['scored_nights'] == 4


@pytest.mark.parametrize('value', ['0', '-1', '5'])
def test_alone_out_of_range_is_refused(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        accelerated.main(['--fake', '--days', '4', '--mind', '--alone-from', value])
    assert error.value.code == 2
    assert '--alone-from must be between 1 and --days' in capsys.readouterr().err


def test_alone_from_first_day_never_simulates(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path, mind_from=START)

    def forbidden(*args: Any) -> DayRun:
        pytest.fail('quiet day simulated a conversation')

    results = accelerated.run_accelerated('fake-life', HERO, 2, START,
        rig=replace(fake.rig(), simulate=forbidden), alone_from=1)
    assert len(results) == 2 and all(r.status != 'failed' for r in results)
    assert not fake.heard


def test_quiet_inquiry_keeps_month_cap(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path, mind_from=START)
    rig = fake.rig()
    deps_for = rig.deps_for
    rig = replace(rig, deps_for=lambda night: replace(deps_for(night), sleep_policy=SleepPolicy(month_usd=0.000001)))
    results = accelerated.run_accelerated('fake-life', HERO, 2, START, rig=rig, alone_from=1)
    assert [r.status for r in results] == ['napped', 'napped']
    assert not fake.scores_dir('fake-life').exists()
    assert not tuple(workspace(fake.world, 'fake-life').events())
