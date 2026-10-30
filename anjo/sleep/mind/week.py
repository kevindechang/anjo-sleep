"""A dated, replayable MIND history and a controlled empty-state comparison.

Pure: callers supply the log, stored evidence and clock. State is always folded from
an intact prefix; a display window must never become an input to the kernel.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
from difflib import unified_diff
from typing import Literal

from anjo.sleep.mind import expect
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import RunStart, Workspace, WorkspaceEvent
from anjo.sleep.probes.relevance import FrozenLabels
from anjo.sleep.reports.mind import Arrival
from anjo.sleep.types import Evidence, Expectation, Target


@dataclass(frozen=True)
class NightSlice:
    night: date
    events: tuple[WorkspaceEvent, ...]
    alone: bool = False


@dataclass(frozen=True)
class Moment:
    at: datetime
    kind: str
    sources: tuple[str, ...]
    seq: int | None
    text: str
    night: date | None = None

    @property
    def line(self) -> str:
        tags = ' '.join(f'[{source}]' for source in self.sources)
        source = f'seq={self.seq}' if self.seq is not None else 'seq=external'
        return f'{tags} {source} {self.kind}: {self.text}'


@dataclass(frozen=True)
class WeekHistory:
    as_of: datetime
    nights: tuple[NightSlice, ...]
    moments: tuple[Moment, ...]
    digest: str
    recorded_outcomes: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuietNight:
    """What her own thinking did on one night the person wrote nothing, counted from her log.

    ``credence_change`` sums the absolute move of every alternative, a new one from zero, so a
    night that only added an alternative still counts. ``predictions_resolved`` counts a prediction
    that left ``pending``: the resolver answered it (true, false or undetermined) or its horizon
    censored it, so a censored prediction is a resolution, not a stillness.
    """

    night: date
    questions_opened: int
    credence_change: float
    predictions_committed: int
    predictions_resolved: int


def quiet_week(nights: Sequence[NightSlice]) -> tuple[QuietNight, ...]:
    """One count per night stamped ``alone``. The nights must be ``slices``' own contiguous run, so
    each night's state folds from the whole prefix before it and never from a display window."""
    seen: tuple[WorkspaceEvent, ...] = ()
    out: list[QuietNight] = []
    for section in nights:
        before = kernel.fold(seen)
        seen = (*seen, *section.events)
        if section.alone:
            out.append(_quiet_counts(section.night, before, kernel.fold(seen)))
    return tuple(out)


def _quiet_counts(night: date, before: Workspace, after: Workspace) -> QuietNight:
    held = {h.id: h.credence for h in before.hypotheses}
    change = math.fsum(abs(h.credence - held.get(h.id, 0.0)) for h in after.hypotheses)
    was = {p.id: p.pending for p in before.predictions}
    opened = {q.id for q in after.questions} - {q.id for q in before.questions}
    committed = {p.id for p in after.predictions} - {p.id for p in before.predictions}
    resolved = {p.id for p in after.predictions if was.get(p.id, True) and not p.pending}
    return QuietNight(night, len(opened), round(change, 6), len(committed), len(resolved))


def _prefix(events: Sequence[WorkspaceEvent], night: date) -> tuple[WorkspaceEvent, ...]:
    cut = next((i for i, event in enumerate(events)
                if isinstance(event.step.entry, RunStart) and event.step.entry.night > night), len(events))
    return tuple(events[:cut])


def fold_to(events: Sequence[WorkspaceEvent], night: date) -> Workspace:
    """State through a night, cut before the first later night's RunStart."""
    return kernel.fold(_prefix(events, night))


def slices(events: Sequence[WorkspaceEvent]) -> tuple[NightSlice, ...]:
    """Group contiguous runs of the same night; these slices are display-only."""
    starts = tuple((i, e.step.entry.night) for i, e in enumerate(events)
                   if isinstance(e.step.entry, RunStart))
    if events and (not starts or starts[0][0] != 0):
        raise ValueError('the log must start with RunStart')
    boundaries = tuple((i, night) for n, (i, night) in enumerate(starts)
                       if n == 0 or starts[n - 1][1] != night)
    if any(a[1] >= b[1] for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('nights must advance in date order')
    return tuple(NightSlice(night, tuple(events[i:boundaries[n + 1][0] if n + 1 < len(boundaries)
                                               else len(events)]))
                 for n, (i, night) in enumerate(boundaries))


def _quote(text: str) -> str:
    return repr(' '.join(text.split())[:180])


def _changes(before: Workspace, after: Workspace, event: WorkspaceEvent,
             night: date) -> tuple[Moment, ...]:
    """Record transitions rather than just the final finding (which may be retracted)."""
    step = event.step
    questions = tuple(Moment(step.at, 'question raised' if before.question(q.id) is None else 'question status',
                             (q.id, *q.origin_evidence), step.seq, f'{q.status} {_quote(q.subject)}', night)
                      for q in after.questions if q not in before.questions)
    hypotheses = tuple(Moment(step.at, 'credence moved', (h.question_id, h.id), step.seq,
                              f'{next((old.credence for old in before.hypotheses if old.id == h.id), None)}'
                              f' -> {h.credence:.6f} {_quote(h.statement)}', night)
                       for h in after.hypotheses if h not in before.hypotheses)
    predictions = tuple(Moment(step.at, 'prediction committed' if not any(p.id == x.id for x in before.predictions)
                               else 'prediction resolved', (p.question_id, p.id, *p.resolved_by), step.seq,
                               f'{_quote(p.claim)} outcome={p.outcome} horizon={p.horizon.isoformat()}', night)
                        for p in after.predictions if p not in before.predictions)
    findings = tuple(Moment(step.at, f'finding {f.kind}', (f.question_id, f.id, *f.cites), step.seq,
                           f'p={f.p:.6f} {_quote(f.statement)}', night)
                     for f in after.findings if f not in before.findings)
    return (*questions, *hypotheses, *predictions, *findings)


@dataclass(frozen=True)
class ResolutionInputs:
    """The fit's frozen relevance, virtual arrivals and actual observation endpoint."""

    labels: FrozenLabels
    arrivals: tuple[Arrival, ...]
    life_end: datetime


def _virtual(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _expectations(expectations: Sequence[Expectation], returns: Sequence[Evidence],
                  as_of: datetime, resolution: ResolutionInputs | None) -> tuple[Moment, ...]:
    moments: tuple[Moment, ...] = ()
    scope = 'returns only' if resolution is None else f'labels={resolution.labels.digest}'
    for made in expectations:
        if made.made_at > as_of:
            continue
        own = expect.returning(returns, made.target_id)
        # collect._returns and expect_register use midnight of the evidence's virtual day.
        returned = tuple((e.id, _virtual(e.day) if resolution else e.created_at) for e in own)
        arrivals = (() if resolution is None else
                    tuple((a.item_id, _virtual(a.day)) for a in resolution.arrivals
                          if resolution.labels.relevant(made.target_id, a.item_id)))
        endpoint = max(as_of, made.horizon) if resolution is None else resolution.life_end
        outcome = expect.resolve(made, returns=tuple(at for _, at in returned),
                                 arrivals=tuple(at for _, at in arrivals), life_end=endpoint, as_of=as_of)
        moments += (Moment(made.made_at, 'expectation made', (made.id, made.target_id), None,
                           f'p_needed={made.p_needed:.6f} horizon={made.horizon.isoformat()} ({scope})'),)
        if outcome is not None and made.outcome in (None, outcome):
            hits = tuple((source, at) for source, at in (*returned, *arrivals)
                         if made.made_at <= at < made.horizon and at <= as_of)
            at = (endpoint if outcome == 'censored' else
                  min(at for _, at in hits) if hits else made.horizon)
            if at <= as_of:
                moments += (Moment(at, 'expectation resolved', (made.id, made.target_id, *(s for s, _ in hits)),
                                   None, f'{outcome} ({scope})'),)
    return moments


def history(events: Sequence[WorkspaceEvent], targets: Sequence[Target], expectations: Sequence[Expectation],
            returns: Sequence[Evidence], *, as_of: datetime,
            resolution: ResolutionInputs | None = None) -> WeekHistory:
    """Replay transitions with short origin quotes from stored, own evidence only.

    Expectations have no workspace event sequence; label that separate provenance explicitly.
    ``targets`` is the fixed target snapshot shared with both probe briefs.
    """
    del targets  # targets do not supply origin-evidence text
    cut = next((i for i, event in enumerate(events) if event.step.at > as_of), len(events))
    prefix = tuple(events[:cut])
    nights = slices(prefix)
    ws = kernel.fold(())
    moments: tuple[Moment, ...] = ()
    own = {e.id: e for e in returns if e.kind in expect.PERSON_KINDS and not e.held_out
           and e.created_at <= as_of}
    for section in nights:
        for event in section.events:
            after = kernel.fold(prefix[:event.step.seq + 1])
            changes = _changes(ws, after, event, section.night)
            origins = tuple(Moment(event.step.at, 'origin evidence', (source,), event.step.seq,
                                   _quote(own[source].text), section.night)
                            for q in after.questions if ws.question(q.id) is None
                            for source in q.origin_evidence if source in own)
            moments += (*changes, *origins)
            ws = after
    clocks = {e.step.at: n.night for n in nights for e in n.events if isinstance(e.step.entry, RunStart)}
    forecasts = tuple(replace(m, night=clocks.get(m.at)) for m in _expectations(expectations, returns, as_of, resolution))
    return WeekHistory(as_of, nights, tuple(sorted((*moments, *forecasts), key=lambda m: m.at)), kernel.digest(ws),
                       tuple(f'[{e.id}] [{e.target_id}] seq=external {e.outcome}'
                             for e in expectations if e.outcome is not None and e.made_at <= as_of))


def _seq(ws: Workspace, record_id: str, question_id: str) -> str:
    relevant = tuple(s.seq for s in ws.history if s.code is None and (
        getattr(s.entry, 'question_id', None) == question_id
        or getattr(s.entry, 'prediction_id', None) in
        {p.id for p in ws.predictions if p.question_id == question_id}
        or (getattr(s.entry, 'subject', None) == getattr(ws.question(question_id), 'subject', None)
            and hasattr(s.entry, 'subject'))))
    return ','.join(map(str, relevant)) or 'none'


def state_brief(ws: Workspace, targets: Sequence[Target], expectations: Sequence[Expectation], as_of: datetime,
                *, variant: Literal['week', 'empty']) -> str:
    """Change only the delimited MIND block; empty is the actual empty kernel state."""
    if variant not in ('week', 'empty'):
        raise ValueError('variant must be week or empty')
    mind = ws if variant == 'week' else kernel.fold(())
    fixed = (f'[clock] seq=external As of {as_of.isoformat()}',
             *(f'[{t.id}] seq=external Target reference' for t in targets),
             *(f'[{e.id}] [{e.target_id}] seq=external Expectation: p_needed={e.p_needed:.6f} '
               f'horizon={e.horizon.isoformat()}' for e in expectations if e.made_at <= as_of))
    lines = (f'[MIND] seq={mind.seq - 1 if mind.history else "none"} digest={kernel.digest(mind)}',
             *(f'[{q.id}] seq={_seq(mind, q.id, q.id)} Question: {_quote(q.subject)} '
               f'status={q.status} origins={",".join(q.origin_evidence)}' for q in mind.questions),
             *(f'[{h.id}] [{h.question_id}] seq={_seq(mind, h.id, h.question_id)} '
               f'Heuristic credence={h.credence:.6f}: {_quote(h.statement)}' for h in mind.hypotheses),
             *(f'[{p.id}] [{p.question_id}] seq={_seq(mind, p.id, p.question_id)} '
               f'Prediction: {_quote(p.claim)} outcome={p.outcome} horizon={p.horizon.isoformat()}'
               for p in mind.predictions),
             *(f'[{f.id}] [{f.question_id}] seq={_seq(mind, f.id, f.question_id)} Finding {f.kind}: '
               f'{_quote(f.statement)} p={f.p:.6f} cites={",".join(f.cites)}' for f in mind.findings))
    return '\n'.join((*fixed, '<MIND>', *lines, '</MIND>')) + '\n'


def probe_brief(ws: Workspace, targets: Sequence[Target], expectations: Sequence[Expectation], as_of: datetime,
                served: str) -> str:
    """The probe's whole input: the served brief inside its delimiters, ``ws``'s state around it.

    The week probe's two arms and the causal run compose their briefs here, so two conditions
    differ only in the state block or the served prose they are handed, never in the framing.
    """
    fixed, _, mind = state_brief(ws, targets, expectations, as_of, variant='week').partition('<MIND>\n')
    return fixed + '<MIND>\n<SERVED_BRIEF>\n' + served + '\n</SERVED_BRIEF>\n' + mind


def diff(week: str, empty: str) -> tuple[str, ...]:
    """A stable unified diff, usable for either briefs or replies."""
    return tuple(unified_diff(week.splitlines(), empty.splitlines(), fromfile='week', tofile='empty', lineterm=''))
