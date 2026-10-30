"""Pure Markdown and JSON renderers for a week's trace and its paired probe."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from anjo.sleep.mind.week import QuietNight, WeekHistory, diff, quiet_week


@dataclass(frozen=True)
class ProbeTrace:
    prompt: str
    adapter: str
    model: str
    seed: int
    temperature: float
    week_brief: str
    empty_brief: str
    week_reply: str
    empty_reply: str


def _quote(text: str) -> str:
    return repr(' '.join(text.split())[:180])


def _public_brief(text: str) -> str:
    """Bound the served excerpt before diffing, so diff context cannot leak the full brief."""
    before, marker, rest = text.partition('<SERVED_BRIEF>\n')
    if not marker:
        return _quote(text)
    served, closing, after = rest.partition('\n</SERVED_BRIEF>')
    if not closing:
        raise ValueError('probe input has no closing served-brief delimiter')
    return before + '[served-brief] seq=external ' + _quote(served) + after


def _public_trace(trace: ProbeTrace) -> ProbeTrace:
    """Reports contain excerpts only; full inputs and replies are separate local artifacts."""
    return replace(trace, prompt=_quote(trace.prompt), week_brief=_public_brief(trace.week_brief),
                   empty_brief=_public_brief(trace.empty_brief), week_reply=_quote(trace.week_reply),
                   empty_reply=_quote(trace.empty_reply))


def _comparison(trace: ProbeTrace) -> dict[str, object]:
    """Compare full strings before excerpting; a tail-only effect must remain visible."""
    def sha(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    return {'brief_changed': trace.week_brief != trace.empty_brief,
            'reply_changed': trace.week_reply != trace.empty_reply,
            'week_brief_sha256': sha(trace.week_brief), 'empty_brief_sha256': sha(trace.empty_brief),
            'week_reply_sha256': sha(trace.week_reply), 'empty_reply_sha256': sha(trace.empty_reply)}


def _quiet(history: WeekHistory) -> tuple[QuietNight, ...]:
    """The quiet nights' counts, when the week has any: a week alone still works."""
    return quiet_week(history.nights)


def render(history: WeekHistory, trace: ProbeTrace) -> str:
    """Show dated transitions and both diffs without implying that different means better."""
    comparison = _comparison(trace)
    trace = _public_trace(trace)
    lines = ['# A week alone', '', 'Does the state she built change what she says?', '',
             f'As of {history.as_of.isoformat()}; digest {history.digest}.', '',
             'Synthetic evidence only. Credences are heuristic. A reply difference is not a quality score.', '']
    for night in history.nights:
        lines += [f'## {night.night.isoformat()}' + (' (alone)' if night.alone else ''), '']
        if not night.events:
            lines += ['- No inquiry log recorded for this night.', '']
        moments = tuple(m for m in history.moments if (m.night or m.at.date()) == night.night)
        lines += [f'- {m.at.isoformat()} {m.line}' for m in moments] or ['- No recorded state transition.']
        lines += ['']
    other = tuple(m for m in history.moments if (m.night or m.at.date()) not in {n.night for n in history.nights})
    if other:
        lines += ['## Outside recorded inquiry nights', '', *(f'- {m.at.isoformat()} {m.line}' for m in other), '']
    if history.recorded_outcomes:
        lines += ['## Recorded ledger outcomes', '',
                  'Current snapshot; resolution timestamps are unavailable. These are not dated transitions.',
                  '', *(f'- {line}' for line in history.recorded_outcomes), '']
    quiet = _quiet(history)
    if quiet:
        lines += ['## A quiet week', '',
                  'Nights the person wrote nothing, counted from her own log. No message opened these: '
                  'her state did.', '',
                  *(f'- {q.night.isoformat()}: {q.questions_opened} question(s) opened, '
                    f'{q.credence_change:.6f} of credence moved, {q.predictions_committed} prediction(s) '
                    f'committed, {q.predictions_resolved} resolved' for q in quiet), '',
                  f'Totals: {sum(q.questions_opened for q in quiet)} opened, '
                  f'{sum(q.predictions_committed for q in quiet)} committed, '
                  f'{sum(q.predictions_resolved for q in quiet)} resolved.', '']
    lines += ['## One probe, two replies', '', trace.prompt, '',
              f'Full replies differ: {comparison["reply_changed"]}; full briefs differ: {comparison["brief_changed"]}.',
              'Diffs below compare short excerpts. Full inputs and exact diffs are in probe-inputs.local.json.',
              f'Week reply SHA-256: {comparison["week_reply_sha256"]}',
              f'Empty reply SHA-256: {comparison["empty_reply_sha256"]}', '',
              f'Model: {trace.model}; adapter: {trace.adapter}; seed: {trace.seed}; temperature: {trace.temperature}.',
              '', '### Week reply', '', trace.week_reply, '', '### Empty reply', '', trace.empty_reply, '',
              '### Brief diff (week → empty)', '', '```diff', *diff(trace.week_brief, trace.empty_brief), '```', '',
              '### Reply diff (week → empty)', '', '```diff', *diff(trace.week_reply, trace.empty_reply), '```', '']
    return '\n'.join(lines)


def to_json(history: WeekHistory, trace: ProbeTrace) -> str:
    """The same trace as JSON, including source ids, sequence numbers and generation controls."""
    comparison = _comparison(trace)
    trace = _public_trace(trace)
    document = {'as_of': history.as_of.isoformat(), 'digest': history.digest,
                'nights': [{'night': n.night.isoformat(), 'alone': n.alone, 'event_seqs': [e.step.seq for e in n.events]}
                           for n in history.nights],
                'moments': [asdict(m) for m in history.moments], 'probe': asdict(trace),
                'quiet_week': [asdict(q) for q in _quiet(history)],
                'recorded_outcomes_undated': history.recorded_outcomes, 'comparison': comparison,
                'brief_diff': diff(trace.week_brief, trace.empty_brief),
                'reply_diff': diff(trace.week_reply, trace.empty_reply)}
    return json.dumps(document, indent=2, sort_keys=True, default=str) + '\n'
