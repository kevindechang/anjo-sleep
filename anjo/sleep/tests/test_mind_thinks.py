# mypy: disallow-untyped-decorators=False
"""Inquiry activation regressions, entirely on the existing accelerated fake rig."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from anjo.sleep.sim.accelerated import run_accelerated
from anjo.sleep.tests.accelerated_rig import FakeAccelerated, life_targets
from anjo.sleep.tests.test_accelerated import HERO, START


def off_fingerprints(root: Path) -> dict[str, str]:
    """Capture decision rows, every model prompt, and brief bytes on the pre-change fixture."""
    fake = FakeAccelerated(root)
    for target in life_targets('fake-life', START):
        fake.world.targets('fake-life').upsert(target)
    rig = fake.rig()
    deps = rig.deps_for(START)
    prompts: list[Any] = []
    chat = deps.spark_chat

    def capture(*args: Any, **kwargs: Any) -> Any:
        prompts.append(args[1])
        return chat(*args, **kwargs)

    deps = replace(deps, spark_chat=capture)
    results = run_accelerated('fake-life', HERO, 6, START,
                              rig=replace(rig, deps_for=lambda night: deps), alone_from=3)
    assert all(r.status != 'failed' for r in results)
    payloads = {
        'decisions': b''.join((r.log_path.parent / 'mind_points.jsonl').read_bytes() for r in results),
        'prompts': json.dumps(prompts, sort_keys=True).encode(),
        'briefs': fake.world.briefs('fake-life').path.read_bytes(),
    }
    return {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}


def test_inquiry_off_is_byte_identical_to_prechange_fixture(tmp_path: Path) -> None:
    # Captured before implementation, from the existing six-night film_founder fake life,
    # with its existing thread and nine-day commitment, quiet from day three.
    assert off_fingerprints(tmp_path) == {
        'decisions': '0f4feff155d51fce0cfb867ae9c22d686fcc55d98c437aa3f57ab804012235a1',
        'prompts': '09e0c0a3d6553afd05dba4e51ddd17949d4f1b88789bf7fd6c5f7c0ed21f5399',
        'briefs': '00d0e62b420a9bea5f23299bbb110665a3c29567532756a30af32a20e3a5ad35',
    }


def test_fake_six_night_command_works_questions_on_quiet_nights(
    tmp_path: Path, capsys: Any, monkeypatch: Any,
) -> None:
    from anjo.sleep.mind.inquiry import kernel
    from anjo.sleep.sim import accelerated
    from anjo.sleep.tests.mind_night_fakes import workspace

    made: list[FakeAccelerated] = []
    original = FakeAccelerated.rig

    def capture(self: FakeAccelerated, *args: Any, **kwargs: Any) -> Any:
        made.append(self)
        return original(self, *args, **kwargs)

    # Retain the CLI's temporary fake workspace for assertions; never use a data directory.
    from contextlib import nullcontext
    import tempfile
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', lambda **kw: nullcontext(str(tmp_path)))
    monkeypatch.setattr(FakeAccelerated, 'rig', capture)
    assert accelerated.main(['--fake', '--days', '6', '--mind', '--alone-from', '3', '--start', START.isoformat()]) == 0
    assert len(json.loads(capsys.readouterr().out)['nights']) == 6
    fake, = made
    ws = kernel.fold(tuple(workspace(fake.world, 'fake-life').events()))
    assert ws.hypotheses
    assert ws.findings or ws.predictions
    from datetime import timedelta
    assert any(r.night >= START + timedelta(days=2) for r in fake.world.briefs('fake-life').records())
    assert fake.thinker is not None
    notices = [p for p in fake.thinker.prompts if 'NOTICE operations.' in p]
    assert notices and all(p.startswith('# notice_v2\n') for p in notices)


def test_notice_port_never_runs_with_inquiry_off(tmp_path: Path) -> None:
    from anjo.sleep.tests.mind_night_fakes import LifeThinker, mind_night

    fake = FakeAccelerated(tmp_path)
    thinker = LifeThinker(targets=('fake-life-c1',))
    rig = fake.rig()
    deps = rig.deps_for(START)
    port = mind_night(fake.world, thinker)
    deps = replace(deps, notice=port, inquiry=port)
    run_accelerated('fake-life', HERO, 2, START, rig=replace(rig, deps_for=lambda night: deps))
    assert not thinker.prompts  # default false even when the real ports are available


def test_new_mind_homework_refreshes_quiet_brief_even_after_two_bad_folds(tmp_path: Path) -> None:
    from datetime import timedelta
    from anjo.sleep.mind.inquiry import kernel
    from anjo.sleep.tests.mind_night_fakes import LifeThinker, Researches, mind_night, workspace
    from anjo.sleep.tests.night_fakes import step

    thinker = LifeThinker(targets=('fake-life-c1',), tool='research', tokens=11_000)
    fake = FakeAccelerated(tmp_path, mind_from=START, thinker=thinker)
    rig = fake.rig()
    deps = rig.deps_for(START)
    port = mind_night(fake.world, thinker, research=Researches())
    deps = replace(deps, notice=port, inquiry=port)
    # A saturated/looping compiler must not strand new MIND homework on a quiet night.
    fake.world.spark.briefs_cut = 100
    results = run_accelerated('fake-life', HERO, 6, START,
                              rig=replace(rig, deps_for=lambda night: deps), alone_from=3)
    ws = kernel.fold(tuple(workspace(fake.world, 'fake-life').events()))
    assert ws.hypotheses and ws.findings
    homework = [e for e in fake.world.evidence('fake-life').stored(START + timedelta(days=5))
                if e.kind == 'homework' and e.id.startswith('mind-')]
    assert homework and any(e.day >= START + timedelta(days=2) for e in homework)
    for item in homework:
        record = next(r for r in fake.world.briefs('fake-life').records() if item.id in r.folded)
        assert record.night == item.day
        assert item.id in record.served and item.source in record.served
        assert item.id not in record.training  # hosted ancestry never enters training
        index = (item.day - START).days
        assert step(results[index].log_path, 'brief')['status'] == 'ok'


def test_fallback_folds_complete_homework_and_leaves_a_backlog_pending(tmp_path: Path) -> None:
    from datetime import timedelta
    from anjo.sleep.context.brief import SECTIONS
    from anjo.sleep.context.recall_common import estimate_tokens
    from anjo.sleep.tests.inquiry_fakes import ev
    from anjo.sleep.tests.night_fakes import World

    world = World(tmp_path)
    store = world.briefs('fake-life')
    item = replace(ev('mind-f1', 'The full finding that must survive the backlog', n=1,
                      kind='homework', source='https://example.org/source'), person_id='fake-life')
    too_large = replace(item, id='mind-too-large', text='x' * 9000)
    backlog = [replace(item, id=f'lesson-{i}', kind='lesson', text='Ordinary evidence') for i in range(1000)]
    world.spark.briefs_cut = 100
    at = item.created_at
    record = store.fold(item.day, [too_large, item, *backlog], at=at,
                        hosted_in_training=False, chat=world.spark, mind_refresh=True)
    assert item.text in record.served and item.source in record.served
    assert record.folded == (item.id,)
    assert len(store.unfolded([too_large, item, *backlog])) == 1001
    assert estimate_tokens(record.served) <= 2000
    assert all(heading in record.served for heading in SECTIONS)
    # Repeat against an existing brief, preserving its sections within the same cap.
    second = replace(item, id='mind-f2', day=item.day + timedelta(days=1))
    again = store.fold(second.day, [second, *backlog], at=at + timedelta(days=1),
                       hosted_in_training=False, chat=world.spark, mind_refresh=True)
    assert second.text in again.served and estimate_tokens(again.served) <= 2000


def test_failed_fold_is_still_kept_without_new_mind_homework(tmp_path: Path) -> None:
    import pytest
    from anjo.sleep.context.brief import BriefError
    from anjo.sleep.tests.inquiry_fakes import ev
    from anjo.sleep.tests.night_fakes import World

    world = World(tmp_path)
    world.spark.briefs_cut = 100
    store = world.briefs('fake-life')
    for enabled, item_id in [(False, 'mind-f1'), (True, 'regular-homework')]:
        item = replace(ev(item_id, 'A result', n=1, kind='homework'), person_id='fake-life')
        with pytest.raises(BriefError):
            store.fold(item.day, [item], at=item.created_at, hosted_in_training=False,
                       chat=world.spark, mind_refresh=enabled)
        assert store.latest() is None
