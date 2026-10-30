"""Fake-only coverage of the end-to-end intervention run: state → brief → replies → effect.

The talkers are callables, the embedder is injected, the ledger is a temp file: no network and no
model. A state-changing edit must move the replies beyond the null re-render; a talker that ignores
its brief must not; and the 2x2 must attribute a state swap to the claims while a talker swap
leaves them alone.
"""
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.night.deps import Prices
from anjo.sleep.tests.inquiry_fakes import PERSON, Ws, commitment, day, ev, hypothesize, night, noon
from anjo.sleep.tools import mind_causal
from anjo.sleep.tools.week import Records
from anjo.sleep.types import CheckpointRef

NIGHT = day(2)  # the second life's state lands on this night; the first life's is unchanged
PRICES = Prices(spark_in_per_m=1.0, spark_out_per_m=2.0, train_gpu_per_hour=3.0, serve_gpu_per_hour=4.0)
TARGETS = (commitment('t1', 'Shoot in the kitchen', due=noon(7)),)
ITEMS = (ev('e0', 'I want an 85 mm shot.', n=0, target='t1', model=None),
         ev('e1', 'The kitchen is three metres.', n=1, target='t1', model=None))
CITE = re.compile(r'\[([^\[\]\n]{1,120})\]')


def own_events() -> tuple[it.WorkspaceEvent, ...]:
    """A night with a live question, a clear lead and a pending prediction: every edit applies."""
    w = Ws()
    w.start(1)
    w.arrive(*ITEMS, at=night(1))
    w.ok(it.NoticeOp('contradiction', 'Does the shot fit?', ('t1',), ('e0', 'e1')), at=night(1))
    w.ok(hypothesize('q1', 0.6, 0.2), at=night(1))
    ids = w.ids('q1')
    w.ok(it.PredictOp('q1', 'The courtyard will open', dict(zip(ids, (0.05, 0.95, 0.95), strict=True)), 5),
         at=night(1))
    return tuple(w.events)


def other_events() -> tuple[it.WorkspaceEvent, ...]:
    """Another life's log: its question is q3 and p5, so its citations cannot pass for this life's."""
    w = Ws()
    w.start(1)
    w.arrive(*ITEMS, at=night(1))
    w.ok(it.NoticeOp('contradiction', 'Does the light fit?', ('t1',), ('e0', 'e1')), at=night(1))
    w.ok(it.ParkOp('q1'), at=night(1))
    w.start(2, wipe=True)  # the op counter survives the wipe, so the ids here differ from own
    w.arrive(*ITEMS, at=night(2))
    w.ok(it.NoticeOp('contradiction', 'Does the lens fit?', ('t1',), ('e1', 'e0')), at=night(2))
    w.ok(hypothesize('q3', 0.5, 0.3), at=night(2))
    ids = w.ids('q3')
    w.ok(it.PredictOp('q3', 'The lens holds', dict(zip(ids, (0.9, 0.1, 0.1), strict=True)), 4), at=night(2))
    return tuple(w.events)


def records(events: tuple[it.WorkspaceEvent, ...], person: str = PERSON) -> Records:
    return Records(Path('/synthetic/workspace.jsonl'), events,
                   tuple(replace(t, person_id=person) for t in TARGETS), (),
                   tuple(replace(e, person_id=person) for e in ITEMS),
                   CheckpointRef('personal-test', person, day(3), 'fixture://adapter', None),
                   '[memory:fixture] seq=external Fixed memory\n')


def echo(tag: str):
    """A talker whose reply is its tag plus the whole MIND block: claims and credences both show."""
    def talker(brief: str, prompt: str) -> str:
        del prompt
        return f'{tag} {brief.partition("<MIND>")[2]}'
    return talker


class Counting:
    """A talker that records every call, so a plan can prove it made none."""

    def __init__(self, talk) -> None:
        self.talk = talk
        self.asked: list[tuple[str, str]] = []
        self.calls = 0
        self.gpu_s = 0.0

    def __call__(self, brief: str, prompt: str) -> str:
        self.asked.append((brief, prompt))
        self.calls += 1
        return self.talk(brief, prompt)


def plan(*, other: bool = True):
    return mind_causal.planned(PERSON, records(own_events()), NIGHT,
                               other=records(other_events(), 'life-other') if other else None,
                               probes=mind_causal.PROBES[:3], prices=PRICES)


def test_plan_prints_calls_cost_and_holds_and_makes_no_call(tmp_path: Path) -> None:
    lines: list[str] = []

    def forbidden(*_: object) -> object:
        pytest.fail('a plan constructed a talker or held budget')

    out = tmp_path / 'results'
    code = mind_causal.main(
        ['--user', PERSON, '--night', NIGHT.isoformat(), '--probes', '2', '--out', str(out)],
        out=lines.append, records=lambda *_: records(own_events()),
        talker_a=forbidden, talker_b=forbidden,
        pool=lambda: 'baseten_sleep_usd', prices=PRICES, prices_for=lambda: PRICES,
        ledger_for=forbidden,
    )
    assert code == 0
    text = '\n'.join(lines)
    for kept in ('conditions: own, null, empty, swap_lead, delete_prediction, inject_hypothesis',
                 'calls: talker A 6 conditions x 2 probes = 12; talker B 1 x 2 = 2',
                 'tokens: ~', 'cost: A ~$', 'holds: A $', 'baseten_sleep_usd',
                 'plan only: no model client, calls or writes'):
        assert kept in text
    assert not out.exists()


def test_go_requires_prices_and_the_ledger_before_any_talker(tmp_path: Path) -> None:
    def forbidden(*_: object) -> object:
        pytest.fail('a refused run constructed a talker')

    with pytest.raises(ValueError, match='--go needs prices'):
        mind_causal.main(['--user', PERSON, '--night', NIGHT.isoformat(), '--out', str(tmp_path / 'r'), '--go'],
                         out=lambda _: None, records=lambda *_: records(own_events()),
                         talker_a=forbidden, talker_b=forbidden,
                         prices=None, prices_for=lambda: None, ledger_for=lambda: None)


def test_go_writes_json_and_markdown_from_fake_talkers(tmp_path: Path) -> None:
    out = tmp_path / 'results'
    talker_a, talker_b = Counting(echo('A')), Counting(echo('B'))

    def source(user: str, *_: object) -> Records:
        return records(other_events(), 'life-other') if user == 'life-other' else records(own_events())

    code = mind_causal.main(
        ['--user', PERSON, '--night', NIGHT.isoformat(), '--other', 'life-other',
         '--probes', '2', '--out', str(out), '--go'],
        out=lambda _: None, records=source,
        talker_a=lambda _records, _as_of: talker_a, talker_b=lambda _model: talker_b,
        pool=lambda: 'baseten_sleep_usd', prices=PRICES, prices_for=lambda: PRICES,
        ledger_for=lambda: Ledger(tmp_path / 'budget.jsonl'),
    )
    assert code == 0
    assert {p.name for p in out.iterdir()} == {'mind-causal.json', 'mind-causal.md'}
    document = json.loads((out / 'mind-causal.json').read_text())
    assert document['load_bearing'] is True
    assert [c['name'] for c in document['conditions']] == [
        'own', 'null', 'empty', 'swap_lead', 'delete_prediction', 'inject_hypothesis', 'swapped']
    assert len(document['two_by_two']['rows']) == 2 and len(document['two_by_two']['columns']) == 2
    assert document['spend']['calls_a'] == 7 * 2 and document['spend']['calls_b'] == 2 * 2
    assert all('brief_sha256' in c and c['brief_tokens'] > 0 for c in document['conditions'])
    markdown = (out / 'mind-causal.md').read_text()
    for kept in ('Does her state change what she says?', 'Null floor', 'Load-bearing on this night: yes',
                 '## The 2x2', '## Replies'):
        assert kept in markdown
    # The ledger settled both holds; a settled hold is money actually spent, not a plan.
    assert (tmp_path / 'budget.jsonl').is_file()


def test_state_changing_edit_moves_output_beyond_the_null_floor() -> None:
    found = plan()
    result = mind_causal.execute(found, talker_a=echo('A'), talker_b=echo('B'),
                                 distances=mind_causal.Distances(None), prices=PRICES)
    by_name = {effect.name: effect for effect in result.effects}
    assert result.floor == mind_causal.Interval(0.0, 0.0, 0.0)  # a deterministic re-render
    for name in ('swap_lead', 'delete_prediction', 'inject_hypothesis', 'empty'):
        assert by_name[name].semantic.low > 0.0, name
        assert by_name[name].load_bearing, name
    assert by_name['inject_hypothesis'].cited.low > 0.0  # the injected claim is cited
    assert by_name['swap_lead'].cited.mean == 0.0  # credences move; the ids cited do not
    assert result.load_bearing and result.share == 1.0
    assert mind_causal.load_bearing_share([result, result]) == 1.0


def test_a_talker_that_ignores_its_brief_stays_inside_the_null_floor() -> None:
    """A no-op condition cannot be load-bearing: the measure credits only difference the state makes."""
    result = mind_causal.execute(plan(), talker_a=lambda *_: 'Nothing changes.', talker_b=lambda *_: 'Nothing.',
                                 distances=mind_causal.Distances(None))
    assert not result.load_bearing
    assert result.share == 0.0
    assert all(effect.semantic == mind_causal.Interval(0.0, 0.0, 0.0) for effect in result.effects)
    assert mind_causal.load_bearing_share([result]) == 0.0


def test_two_by_two_swaps_state_for_claims_and_the_talker_for_nothing() -> None:
    found = plan()
    talker_a, talker_b = Counting(echo('A')), Counting(echo('B'))
    result = mind_causal.execute(found, talker_a=talker_a, talker_b=talker_b,
                                 distances=mind_causal.Distances(None), prices=PRICES)
    assert result.grid is not None
    assert result.grid.cells == ('own/A', 'own/B', 'swapped/A', 'swapped/B')
    for row in result.grid.rows:  # state swap: other ids, so fewer shared claims
        assert row.claims < 1.0 and row.semantic > 0.0
    for column in result.grid.columns:  # talker swap: same ids, so the claims are preserved
        assert column.claims == 1.0
    assert len(talker_a.asked) == len(found.conditions) * len(found.probes)
    assert len(talker_b.asked) == 2 * len(found.probes)


def test_swap_needs_another_life_not_the_same_person() -> None:
    same = records(other_events(), PERSON)
    with pytest.raises(ValueError, match='another life'):
        mind_causal.planned(PERSON, records(own_events()), NIGHT, other=same, probes=('Why?',))


def test_missing_edits_are_recorded_as_skips_not_dropped() -> None:
    empty = Records(Path('/synthetic/workspace.jsonl'), (), TARGETS, (), ITEMS,
                    CheckpointRef('personal-test', PERSON, day(3), 'fixture://adapter', None), 'Served.\n')
    found = mind_causal.planned(PERSON, empty, NIGHT, probes=('Why?',))
    assert [c.name for c in found.conditions] == ['own', 'null', 'empty']
    assert {skip.name for skip in found.skips} == {'swap_lead', 'delete_prediction', 'inject_hypothesis', 'swapped'}


def test_injected_embedder_is_cached_and_scored_by_cosine() -> None:
    vectors = {'red': (1.0, 0.0), 'crimson': (1.0, 0.0), 'blue': (0.0, 1.0)}
    calls: list[str] = []

    def embed(text: str) -> tuple[float, ...]:
        calls.append(text)
        return vectors[text]

    distances = mind_causal.Distances(embed)
    assert distances.name == 'minilm cosine'
    assert distances.semantic('red', 'crimson') == pytest.approx(0.0)
    assert distances.semantic('red', 'blue') == pytest.approx(1.0)
    assert distances.semantic('red', 'blue') == pytest.approx(1.0)
    assert calls == ['red', 'crimson', 'blue']
    assert mind_causal.cites('I think [q1.h2a] and [p3] (not [a phrase]).') == frozenset({'q1.h2a', 'p3'})
