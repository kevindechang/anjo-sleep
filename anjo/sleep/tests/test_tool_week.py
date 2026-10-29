"""A plan is inert; fake paired replies exercise the real arms.reply seam offline."""
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.model_api import ChatResult
from anjo.sleep.reports.week import render, to_json
from anjo.sleep.tests.inquiry_fakes import PERSON, day, night
from anjo.sleep.tests.test_mind_week import EXPECTATIONS, ITEMS, TARGETS, fixture_events
from anjo.sleep.tools import week
from anjo.sleep.types import CheckpointRef


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def complete(self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *,
                 temperature: float, seed: int | None = None) -> ChatResult:
        self.calls.append((adapter, messages, temperature, seed))
        text = 'Try the courtyard.' if '[q1]' in messages[0]['content'] else 'What shot do you want?'
        return ChatResult(text, (), 'fake', 12, 8, 'stop')


def source() -> week.Records:
    return week.Records(Path('/synthetic/workspace.jsonl'), fixture_events(), TARGETS, EXPECTATIONS, ITEMS,
                        CheckpointRef('personal-test', PERSON, day(3), 'fixture://adapter', None),
                        '[memory:fixture] seq=external Fixed memory\n[brief:fixture] seq=external Fixed brief\n')


def args(root: Path) -> list[str]:
    return ['--user', PERSON, '--start', day(1).isoformat(), '--prompt', 'What shot should I try?', '--out', str(root)]


def test_plan_never_constructs_a_client_or_writes(tmp_path: Path) -> None:
    lines = []
    def forbidden(as_of: Any) -> Any:
        pytest.fail('plan tried to construct a model client')
    seen = []
    def records(user: str, persona: str, as_of: Any) -> week.Records:
        seen.append((user, persona, as_of))
        return source()
    root = tmp_path / 'report'
    assert week.main(args(root), out=lines.append, records=records, model_factory=forbidden) == 0
    output = '\n'.join(lines)
    assert seen[0][:2] == (PERSON, 'film_founder')
    for text in ('/synthetic/workspace.jsonl', '7 calendar nights', 'RunStart', 'NoticeOp',
                 '1 prompt x 2 replies', 'estimated cost unknown', 'personal-test', week.MODEL, 'plan only'):
        assert text in output
    assert not root.exists()


def test_go_fake_calls_match_everything_except_mind(tmp_path: Path) -> None:
    model = FakeModel()
    root = tmp_path / 'report'
    assert week.main([*args(root), '--go'], out=lambda _: None, records=lambda *_: source(),
                     model_factory=lambda _: model) == 0
    assert len(model.calls) == 2
    first, second = model.calls
    assert first[0] == second[0] == source().adapter
    assert first[2:] == second[2:] and first[2] == 0.0 and isinstance(first[3], int)
    a, b = first[1], second[1]
    assert a[1] == b[1] == {'role': 'user', 'content': 'What shot should I try?'}
    assert a[0]['role'] == b[0]['role'] == 'system'
    left, right = a[0]['content'], b[0]['content']
    assert left.split('<MIND>')[0] == right.split('<MIND>')[0]
    assert left.split('</MIND>')[1] == right.split('</MIND>')[1]
    assert left != right
    assert set(p.name for p in root.iterdir()) == {'week.md', 'week.json', 'week-reply.txt', 'empty-reply.txt', 'probe-inputs.local.json'}
    doc = json.loads((root / 'week.json').read_text())
    assert doc['probe']['seed'] == first[3]
    local = json.loads((root / 'probe-inputs.local.json').read_text())
    assert local['week'] == left and local['empty'] == right
    assert local['brief_diff'] and local['reply_diff']
    assert '<SERVED_BRIEF>' not in doc['probe']['week_brief']
    assert doc['brief_diff'] and doc['reply_diff']
    assert (root / 'week-reply.txt').read_text() == 'Try the courtyard.'
    assert (root / 'empty-reply.txt').read_text() == 'What shot do you want?'
    md = (root / 'week.md').read_text()
    assert 'finding retraction' in md and 'finding resolution' in md and 'seq=external' in md
    assert 'Reply diff' in md and 'Brief diff' in md
    assert 'No inquiry log recorded for this night.' in md


def test_window_keeps_prior_state_but_only_displays_selected_nights(tmp_path: Path) -> None:
    model = FakeModel()
    argv = args(tmp_path / 'window')
    argv[argv.index('--start') + 1] = day(3).isoformat()
    assert week.main([*argv, '--days', '1', '--go'], out=lambda _: None,
                     records=lambda *_: source(), model_factory=lambda _: model) == 0
    doc = json.loads((tmp_path / 'window' / 'week.json').read_text())
    assert [n['night'] for n in doc['nights']] == [day(3).isoformat()]
    assert not any(m['kind'] == 'question raised' for m in doc['moments'])
    assert '[q1]' in model.calls[0][1][0]['content']
    assert 'The courtyard now works' in model.calls[0][1][0]['content']


@pytest.mark.parametrize('field', ['targets', 'evidence', 'adapter'])
def test_wrong_person_refused_before_model(tmp_path: Path, field: str) -> None:
    data = source()
    if field == 'adapter':
        bad = replace(data, adapter=replace(data.adapter, person_id='other'))
    else:
        values = getattr(data, field)
        bad = replace(data, **{field: (replace(values[0], person_id='other'), *values[1:])})
    with pytest.raises(ValueError, match='another person'):
        week.main(args(tmp_path), records=lambda *_: bad, model_factory=lambda _: pytest.fail('model called'))


def test_adapter_mismatch_and_existing_output_refused_before_model(tmp_path: Path) -> None:
    for suffix in (['--adapter', 'wrong'], ['--go']):
        (tmp_path / 'week.md').write_text('keep me')
        with pytest.raises(ValueError):
            week.main([*args(tmp_path), *suffix], out=lambda _: None, records=lambda *_: source(),
                      model_factory=lambda _: pytest.fail('model called'))
        assert (tmp_path / 'week.md').read_text() == 'keep me'


def test_synthetic_gate_precedes_content_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    named = []
    monkeypatch.setattr(week, 'host_check', lambda: lambda _: False)
    def resolve(name: str) -> Any:
        named.append(name)
        if name != 'anjo.core.ids:validate_user_id':
            pytest.fail('content store opened before synthetic validation')
        return lambda _: None
    monkeypatch.setattr(week, '_named', resolve)
    with pytest.raises(ValueError, match='registered synthetic'):
        week.host_records('not-synthetic', 'film_founder', night(3))
    assert named == ['anjo.core.ids:validate_user_id']


def test_renderers_are_deterministic() -> None:
    data = source()
    model = FakeModel()
    trace = week.paired(data, PERSON, 'One probe', day(3), night(3), model)
    chosen = week._window(data, day(1), day(3), night(3))
    assert render(chosen, trace) == render(chosen, trace)
    assert json.loads(to_json(chosen, trace))['digest'] == chosen.digest


def test_host_omits_future_target_and_freezes_person_brief(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All stores are in-memory fakes: no host or data-directory access."""
    from types import SimpleNamespace

    future = replace(TARGETS[0], id='future-target', label='Future knowledge', last_mentioned=day(9))
    latest = SimpleNamespace(night=day(3), compiled_at=night(3), folded=(),
                             served='A fixed synthetic brief' + ' detail' * 100 + '\n  Verbatim tail.\n')
    later = SimpleNamespace(night=day(9), compiled_at=night(9), served='Future brief', folded=())
    folder = tmp_path / 'nights' / 'quiet'
    folder.mkdir(parents=True)
    (folder / 'summary.json').write_text(json.dumps({'night': day(3).isoformat(), 'alone': True}))
    ports = {
        'anjo.sleep.app.ownership:night_logs': lambda _: folder.parent,
        'anjo.core.ids:validate_user_id': lambda _: None,
        'anjo.sleep.app.storage:workspace_log': lambda _: SimpleNamespace(
            path=Path('/fixture/log'), events=fixture_events),
        'anjo.sleep.app.storage:target_store': lambda _: SimpleNamespace(
            targets=lambda: [*TARGETS, future], expectations=lambda: EXPECTATIONS),
        'anjo.sleep.app.storage:evidence_store': lambda _: SimpleNamespace(stored=lambda _: ITEMS),
        'anjo.sleep.app.storage:brief_store': lambda _: SimpleNamespace(records=lambda: [latest, later]),
        'anjo.sleep.app.storage:checkpoint_registry': lambda: week.FixedAdapter(source().adapter),
    }
    monkeypatch.setattr(week, '_named', ports.__getitem__)
    monkeypatch.setattr(week, 'host_check', lambda: lambda _: True)
    monkeypatch.setattr(week, 'host_resolution', lambda *_: None)
    loaded = week.host_records(PERSON, 'film_founder', night(3))
    assert loaded.alone_nights == (day(3),)
    assert week._window(loaded, day(1), day(3), night(3)).nights[-1].alone
    assert loaded.targets == TARGETS
    assert loaded.common == latest.served
    assert latest.served in week._briefs(loaded, day(3), night(3))[0]
    brief, _ = week._briefs(loaded, day(3), night(3))
    assert 'Future knowledge' not in brief and 'future-target' not in brief


def test_window_includes_the_last_nights_midnight_clock(tmp_path: Path) -> None:
    from datetime import datetime, time, timezone

    from anjo.sleep.mind.inquiry import kernel
    from anjo.sleep.mind.inquiry import types as it

    cutoff = datetime.combine(day(2), time.min, tzinfo=timezone.utc)
    start = it.RunStart(day(1), cutoff, 'inquiry', it.FULL_RULES, 'priority', None, it.Budget())
    applied = kernel.apply(it.Workspace(), it.Step(0, cutoff, start))
    assert isinstance(applied, it.Applied)
    loaded = replace(source(), events=(applied.event,), expectations=(), targets=(), evidence=())
    root = tmp_path / 'midnight'
    assert week.main([*args(root), '--days', '1', '--go'], out=lambda _: None,
                     records=lambda *_: loaded, model_factory=lambda _: FakeModel()) == 0
    document = json.loads((root / 'week.json').read_text())
    assert document['nights'] == [{'night': day(1).isoformat(), 'event_seqs': [0], 'alone': False}]
    assert document['digest'] == applied.event.after


def test_full_brief_reaches_both_calls_and_only_local_inputs_keep_it(tmp_path: Path) -> None:
    full = ('## The project and its constraints\n  A fact with  two spaces.\n'
            + 'A long synthetic fact. ' * 80 + '\nFULL_BRIEF_TAIL\n')
    data = replace(source(), common=full)
    model = FakeModel()
    root = tmp_path / 'full'
    assert week.main([*args(root), '--go'], out=lambda _: None, records=lambda *_: data,
                     model_factory=lambda _: model) == 0
    for call in model.calls:
        assert full in call[1][0]['content']
    local = json.loads((root / 'probe-inputs.local.json').read_text())
    assert full in local['week'] and full in local['empty']
    for name in ('week.md', 'week.json'):
        public = (root / name).read_text()
        assert 'FULL_BRIEF_TAIL' not in public and full not in public
        assert 'served-brief' in public


def test_mind_findings_in_served_prose_are_ablated_inside_the_same_block() -> None:
    ordinary = '## The project and its constraints\nKeep the 85 mm look. [e0]\n'
    finding = 'Use a wider lens. [mind-f4]\n'
    data = replace(source(), common=ordinary + finding + 'Other preference. [e1]\n',
                   mind_evidence_ids=('mind-f4',))
    model = FakeModel()
    week.paired(data, PERSON, 'One probe', day(3), night(3), model)
    a, b = (call[1][0]['content'] for call in model.calls)
    assert data.common in a and finding not in b
    assert ordinary in a and ordinary in b and 'Other preference. [e1]\n' in b
    assert a.partition('<MIND>')[0] == b.partition('<MIND>')[0]
    assert a.partition('</MIND>')[2] == b.partition('</MIND>')[2]
    assert model.calls[0][0] == model.calls[1][0] and model.calls[0][2:] == model.calls[1][2:]


@pytest.mark.parametrize('text', ['An uncited MIND paraphrase.', 'Mixed facts. [mind-f4] [e0]'])
def test_inseparable_mind_prose_is_refused_before_any_call(text: str) -> None:
    model = FakeModel()
    data = replace(source(), common=text, mind_evidence_ids=('mind-f4',))
    with pytest.raises(ValueError, match='exact removal is unavailable'):
        week.paired(data, PERSON, 'One probe', day(3), night(3), model)
    assert model.calls == []


def test_host_loads_the_same_frozen_life_inputs_as_the_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.mind import collect
    from anjo.sleep.probes.relevance import freeze
    from anjo.sleep.reports import results

    root = tmp_path / PERSON
    root.mkdir()
    labels = freeze('fixture', {('t1', 'arrival'): True}, frozenset())
    (root / 'relevance.json').write_text(json.dumps({
        'prompt_sha256': labels.prompt_sha256, 'digest': labels.digest,
        'labels': [['t1', 'arrival', True]], 'unreadable': [],
    }))
    folders = (root / 'nights' / day(0).isoformat(), root / 'nights' / day(8).isoformat())
    monkeypatch.setattr(results, 'results_root', lambda: tmp_path)
    monkeypatch.setattr(collect, '_collectible_folders', lambda *_: folders)
    monkeypatch.setattr(collect, '_life_start', lambda fs: day(0) if len(fs) == 2 else day(8))
    from anjo.sleep.reports.mind import Arrival
    arrivals = (Arrival('arrival', day(4)),)
    monkeypatch.setattr(collect, '_arrivals', lambda _person, start: arrivals if start == day(0) else ())
    loaded = week.host_resolution(PERSON, Path(collect.PERSONAS_DIR) / 'film_founder.json')
    assert loaded is not None and loaded.labels.digest == labels.digest and loaded.arrivals == arrivals
    assert loaded.life_end.date() == day(9) and loaded.life_end.hour == 0
    assert week.host_resolution('missing-life', Path('unused')) is None
    (root / 'relevance.json').write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        week.host_resolution(PERSON, Path('unused'))


def test_full_reply_difference_is_visible_after_a_long_shared_prefix(tmp_path: Path) -> None:
    class TailModel(FakeModel):
        def complete(self, adapter: Any, messages: Any, *, temperature: float, seed: Any = None) -> ChatResult:
            self.calls.append((adapter, messages, temperature, seed))
            tail = 'WEEK_TAIL' if len(self.calls) == 1 else 'EMPTY_TAIL'
            return ChatResult('Same prefix ' * 40 + tail, (), 'fake', 12, 8, 'stop')

    model = TailModel()
    root = tmp_path / 'tail'
    assert week.main([*args(root), '--go'], out=lambda _: None, records=lambda *_: source(),
                     model_factory=lambda _: model) == 0
    document = json.loads((root / 'week.json').read_text())
    assert document['comparison']['reply_changed'] is True
    assert document['comparison']['week_reply_sha256'] != document['comparison']['empty_reply_sha256']
    assert 'Full replies differ: True' in (root / 'week.md').read_text()
    local = json.loads((root / 'probe-inputs.local.json').read_text())
    assert any('WEEK_TAIL' in line for line in local['reply_diff'])
    assert any('EMPTY_TAIL' in line for line in local['reply_diff'])


def test_markdown_keeps_expectations_on_a_night_without_inquiry() -> None:
    data = replace(source(), events=())
    chosen = week._window(data, day(1), day(8), night(8))
    trace = week.paired(data, PERSON, 'Probe', day(8), night(8), FakeModel())
    markdown = render(chosen, trace)
    assert 'No inquiry log recorded' in markdown
    assert 'expectation made' in markdown and 'expectation resolved' in markdown
    assert 'returns only' in markdown


def test_plan_marks_quiet_nights_alone(tmp_path: Path) -> None:
    lines: list[str] = []
    data = replace(source(), alone_nights=(day(3), day(4)))
    assert week.main(args(tmp_path / 'out'), out=lines.append, records=lambda *_: data) == 0
    timeline = next(line for line in lines if line.startswith('recorded nights:'))
    assert f'{day(3).isoformat()} (alone)' in timeline
    assert f'{day(4).isoformat()} (alone)' in timeline  # even without workspace events
    assert f'{day(1).isoformat()} (alone)' not in timeline
