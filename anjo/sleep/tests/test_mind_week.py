"""Fixture-only regression coverage of intact-prefix replay and the two state briefs."""
from dataclasses import replace

import pytest

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import EMPTY_DIGEST
from anjo.sleep.mind.week import diff, fold_to, history, slices, state_brief
from anjo.sleep.tests.inquiry_fakes import Ws, commitment, day, ev, hypothesize, night, noon, ref
from anjo.sleep.types import Expectation

TARGETS = (commitment('t1', 'Shoot in the kitchen', due=noon(7)),)
ITEMS = (ev('e0', 'I want an 85 mm shot.', n=0, target='t1', model=None),
         ev('e1', 'The kitchen is three metres.', n=1, target='t1', model=None),
         ev('e2', 'The courtyard is locked.', n=2, target='t1', model=None),
         ev('e3', 'The courtyard opened.', n=3, target='t1', model=None))
EXPECTATIONS = (Expectation('x-1', 't1', night(1), noon(7), 0.6),)


def fixture_events() -> tuple[it.WorkspaceEvent, ...]:
    """A finding survives in history after later evidence causes its retraction."""
    w = Ws()
    w.start(1)
    w.arrive(*ITEMS[:2], at=night(1))
    w.ok(it.NoticeOp('contradiction', 'Does the shot fit?', ('t1',), ('e0', 'e1')), at=night(1))
    w.ok(hypothesize('q1', 0.4, 0.4), at=night(1))
    ids = w.ids('q1')
    w.ok(it.PredictOp('q1', 'The courtyard will open', dict(zip(ids, (0.05, 0.95, 0.95), strict=True)), 5),
         at=night(1))
    w.start(1)  # another run that same night is not another night
    w.start(2)
    w.ok(it.Retrieved('q1', 'evidence_search', 'courtyard', (ref(ITEMS[2]),)), at=night(2))
    w.ok(it.AssessOp('q1', 'e2', dict(zip(ids, (4.0, 0.25, 0.25), strict=True))), at=night(2))
    w.ok(it.ConcludeOp('q1', ids[0], 'resolution', 'Use a wider lens', ('e2',)), at=night(2))
    w.start(3)
    prediction = w.ws.predictions[0]
    w.ok(it.Resolved(prediction.id, 'true', (ref(ITEMS[3]),)), at=night(3))
    w.ok(it.RetractOp('q1', 'The courtyard now works', ('e3',)), at=night(3))
    w.start(4)
    return tuple(w.events)


def test_fold_to_is_valid_prefix_not_filtered_night() -> None:
    events = fixture_events()
    for n in (0, 1, 2, 3, 4, 9):
        ws = fold_to(events, day(n))
        cut = next((i for i, e in enumerate(events) if isinstance(e.step.entry, it.RunStart)
                    and e.step.entry.night > day(n)), len(events))
        assert ws == kernel.fold(events[:cut])
        assert kernel.digest(ws) == (events[cut - 1].after if cut else EMPTY_DIGEST)
    with pytest.raises(ValueError):
        kernel.fold(slices(events)[1].events)


def test_slices_merge_runs_and_preserve_every_event() -> None:
    events = fixture_events()
    grouped = slices(events)
    assert tuple(s.night for s in grouped) == tuple(day(n) for n in (1, 2, 3, 4))
    assert tuple(e for s in grouped for e in s.events) == events
    assert sum(isinstance(e.step.entry, it.RunStart) for e in grouped[0].events) == 2
    assert slices(()) == ()
    with pytest.raises(ValueError):
        slices(events[1:])


def test_history_retains_transitions_and_own_origin_quotes() -> None:
    events = fixture_events()
    result = history(events, TARGETS, EXPECTATIONS, ITEMS, as_of=night(3))
    kinds = [m.kind for m in result.moments]
    for kind in ('question raised', 'credence moved', 'prediction committed', 'prediction resolved',
                 'finding resolution', 'finding retraction', 'expectation made', 'expectation resolved'):
        assert kind in kinds
    assert kinds.index('finding resolution') < kinds.index('finding retraction')
    origins = [m for m in result.moments if m.kind == 'origin evidence']
    assert [m.text for m in origins] == [repr(e.text) for e in ITEMS[:2]]
    assert all(m.seq is not None and m.sources for m in origins)
    assert result.digest == kernel.digest(fold_to(events, day(3)))
    made, resolved = [m for m in result.moments if m.kind.startswith('expectation')]
    assert (made.at, resolved.at, resolved.text) == (night(1), noon(2), 'needed (returns only)')
    assert resolved.sources == ('x-1', 't1', 'e2', 'e3')
    assert made.seq is None


@pytest.mark.parametrize('bad', [replace(ITEMS[0], held_out=True),
                                 replace(ITEMS[0], kind='homework')])
def test_origin_text_never_comes_from_non_person_evidence(bad: object) -> None:
    result = history(fixture_events(), TARGETS, (), (bad, *ITEMS[1:]), as_of=night(3))  # type: ignore[arg-type]
    origins = [m for m in result.moments if m.kind == 'origin evidence']
    assert [m.sources for m in origins] == [('e1',)]


def test_history_as_of_and_unresolved_expectation() -> None:
    result = history(fixture_events(), TARGETS, EXPECTATIONS, (), as_of=night(1))
    assert not any(m.kind in ('prediction resolved', 'expectation resolved') for m in result.moments)
    assert result.digest == kernel.digest(fold_to(fixture_events(), day(1)))
    later = history((), TARGETS, EXPECTATIONS, (), as_of=night(8))
    assert later.moments[-1].text == 'not_needed (returns only)'


def test_brief_snapshots_empty_and_week_and_diff() -> None:
    events = fixture_events()
    ws = fold_to(events, day(1))
    week = state_brief(ws, TARGETS, EXPECTATIONS, night(1), variant='week')
    empty = state_brief(ws, TARGETS, EXPECTATIONS, night(1), variant='empty')
    common = (f'[clock] seq=external As of {night(1).isoformat()}\n'
              "[t1] seq=external Target reference\n"
              f'[x-1] [t1] seq=external Expectation: p_needed=0.600000 horizon={noon(7).isoformat()}\n')
    assert empty == common + f'<MIND>\n[MIND] seq=none digest={EMPTY_DIGEST}\n</MIND>\n'
    assert week == common + (f'<MIND>\n[MIND] seq=5 digest={kernel.digest(ws)}\n'
                            "[q1] seq=2,3,4 Question: 'Does the shot fit?' status=active origins=e0,e1\n"
                            "[q1.h2a] [q1] seq=2,3,4 Heuristic credence=0.400000: 'alternative 1 about the plan'\n"
                            "[q1.h2b] [q1] seq=2,3,4 Heuristic credence=0.400000: 'alternative 2 about the plan'\n"
                            "[q1.else] [q1] seq=2,3,4 Heuristic credence=0.200000: 'something else'\n"
                            "[p3] [q1] seq=2,3,4 Prediction: 'The courtyard will open' outcome=None "
                            f'horizon={night(6).isoformat()}\n</MIND>\n')
    assert week.split('<MIND>')[0] == empty.split('<MIND>')[0]
    assert diff(week, empty)[:2] == ('--- week', '+++ empty')
    assert diff(week, week) == ()


def test_resolution_sequence_traces_updated_credences() -> None:
    events = fixture_events()
    ws = fold_to(events, day(3))
    resolved = next(e.step.seq for e in events if isinstance(e.step.entry, it.Resolved))
    brief = state_brief(ws, TARGETS, (), night(3), variant='week')
    for line in brief.splitlines():
        if 'Heuristic credence=' in line:
            assert str(resolved) in line.split('seq=')[1].split()[0].split(',')


def test_extracted_person_evidence_keeps_its_origin_quote() -> None:
    extracted = replace(ITEMS[0], model='extractor')
    result = history(fixture_events(), TARGETS, (), (extracted, *ITEMS[1:]), as_of=night(3))
    assert any(m.kind == 'origin evidence' and m.sources == ('e0',) for m in result.moments)


@pytest.mark.parametrize('outcome', ['needed', 'censored'])
def test_recorded_outcome_is_not_overwritten_by_incomplete_returns(outcome: str) -> None:
    made = replace(EXPECTATIONS[0], outcome=outcome)
    result = history((), TARGETS, (made,), (), as_of=night(8))
    assert not any(m.kind == 'expectation resolved' and m.text == 'not_needed' for m in result.moments)
    assert result.recorded_outcomes == (f'[x-1] [t1] seq=external {outcome}',)


def test_target_mutable_fields_cannot_leak_into_brief() -> None:
    ws = fold_to(fixture_events(), day(1))
    target = replace(TARGETS[0], label='A future secret', status='settled', revision=99)
    assert state_brief(ws, (target,), (), night(1), variant='week') == state_brief(
        ws, TARGETS, (), night(1), variant='week')


def test_relevant_arrivals_use_the_fits_virtual_clock_and_censoring() -> None:
    from anjo.sleep.mind import expect
    from anjo.sleep.mind.week import ResolutionInputs
    from anjo.sleep.probes.relevance import freeze
    from anjo.sleep.reports.mind import Arrival

    labels = freeze('fixture', {('t1', 'held-work'): True}, frozenset())
    inputs = ResolutionInputs(labels, (Arrival('held-work', day(4)),), night(8))
    result = history((), TARGETS, EXPECTATIONS, (), as_of=night(8), resolution=inputs)
    resolved, = (m for m in result.moments if m.kind == 'expectation resolved')
    arrival_at = inputs.life_end.replace(year=day(4).year, month=day(4).month, day=day(4).day,
                                         hour=0, minute=0)
    fitted = expect.resolve(EXPECTATIONS[0], returns=(), arrivals=(arrival_at,),
                            life_end=inputs.life_end, as_of=night(8))
    assert resolved.text == f'{fitted} (labels={labels.digest})'
    assert resolved.sources == ('x-1', 't1', 'held-work') and resolved.at == arrival_at
    assert history((), TARGETS, EXPECTATIONS, (), as_of=night(3), resolution=inputs).moments[-1].kind == 'expectation made'
    # collect._returns uses midnight of day, even when extraction happened later.
    returned = history((), TARGETS, EXPECTATIONS, ITEMS, as_of=night(8), resolution=inputs)
    assert next(m for m in returned.moments if m.kind == 'expectation resolved').at.hour == 0
    censored = replace(inputs, life_end=night(5))
    result = history((), TARGETS, EXPECTATIONS, (), as_of=night(8), resolution=censored)
    assert result.moments[-1].text == f'censored (labels={labels.digest})'


def test_irrelevant_arrival_does_not_resolve_needed() -> None:
    from anjo.sleep.mind.week import ResolutionInputs
    from anjo.sleep.probes.relevance import freeze
    from anjo.sleep.reports.mind import Arrival

    labels = freeze('fixture', {('t1', 'held-work'): False}, frozenset())
    inputs = ResolutionInputs(labels, (Arrival('held-work', day(4)),), night(8))
    result = history((), TARGETS, EXPECTATIONS, (), as_of=night(8), resolution=inputs)
    assert result.moments[-1].text == f'not_needed (labels={labels.digest})'


def test_available_labels_require_coverage_instead_of_silent_returns_fallback() -> None:
    from anjo.sleep.mind.week import ResolutionInputs
    from anjo.sleep.probes.relevance import LabelError, freeze
    from anjo.sleep.reports.mind import Arrival

    inputs = ResolutionInputs(freeze('fixture', {}, frozenset()), (Arrival('missing', day(4)),), night(8))
    with pytest.raises(LabelError):
        history((), TARGETS, EXPECTATIONS, (), as_of=night(8), resolution=inputs)


def test_labeled_resolution_agrees_with_expect_register_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from anjo.sleep.mind import expect_register
    from anjo.sleep.mind.week import ResolutionInputs
    from anjo.sleep.probes.relevance import freeze
    from anjo.sleep.reports.mind import Arrival, LifeRecord

    labels = freeze('fixture', {('t1', 'arrival'): True}, frozenset())
    life = LifeRecord('film_founder', 'development', (), (Arrival('arrival', day(4)),),
                      labels, day(8), expectations=EXPECTATIONS, returns={})
    monkeypatch.setattr(expect_register, '_studies', lambda *_: ((night(1), 'follow_up', TARGETS[0]),))
    row, = expect_register._fit_rows(life, Path('/fixture'))
    inputs = ResolutionInputs(labels, life.arrivals, expect_register._virtual(day(9)))
    timeline = history((), TARGETS, EXPECTATIONS, (), as_of=inputs.life_end, resolution=inputs)
    resolved, = (m for m in timeline.moments if m.kind == 'expectation resolved')
    assert row.needed and resolved.text.startswith('needed (labels=')
