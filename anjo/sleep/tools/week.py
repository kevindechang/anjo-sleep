"""Plan a synthetic week alone; only --go asks the paired probe and writes reports.

Host ports are loaded lazily, like mind_dev_prepare. Planning reads a verified synthetic
account's logs only: no context builder, model client, GPU wake or writer is called.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Literal
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.mind.inquiry.types import WorkspaceEvent
from anjo.sleep.mind.week import NightSlice, ResolutionInputs, WeekHistory, diff, fold_to, history, state_brief
from anjo.sleep.ports import EvidenceStore
from anjo.sleep.probes import arms
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.reports.week import ProbeTrace, render, to_json
from anjo.sleep.serving.config import MODEL
from anjo.sleep.tools.fit_estimator import _named, host_check
from anjo.sleep.types import CheckpointRef, Evidence, Expectation, Probe, Target


@dataclass(frozen=True)
class Records:
    log_path: Path
    events: tuple[WorkspaceEvent, ...]
    targets: tuple[Target, ...]
    expectations: tuple[Expectation, ...]
    evidence: tuple[Evidence, ...]
    adapter: CheckpointRef | None
    common: str = ''  # the full served brief, verbatim; never a report excerpt
    mind_evidence_ids: tuple[str, ...] = ()
    resolution: ResolutionInputs | None = None
    alone_nights: tuple[date, ...] = ()


@dataclass(frozen=True)
class FixedBrief:
    """The existing ContextBuilder seam, frozen separately for each arm-4 call."""

    text: str

    def build(self, user_id: str, probe: Probe) -> str:
        return self.text


@dataclass(frozen=True)
class FixedAdapter:
    checkpoint: CheckpointRef | None

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        return self.checkpoint


@dataclass(frozen=True)
class NoLogprobs:
    def continuation_logprobs(self, adapter: CheckpointRef | None, messages: list[dict[str, str]],
                             continuation: str) -> Sequence[float]:
        raise RuntimeError('the week probe must be free text')


def _no_evidence(user_id: str) -> EvidenceStore:
    raise RuntimeError('arm 4 must use only the fixed brief')


def host_resolution(user: str, persona_path: Path) -> ResolutionInputs | None:
    """Read the same frozen life inputs as expect_register, without replay or labeling."""
    from anjo.sleep.mind import collect
    from anjo.sleep.personas.model import load_person_file
    from anjo.sleep.reports.results import results_root

    root = results_root() / user
    if not (root / collect.LABELS_FILE).is_file():
        return None
    labels = collect.load_frozen_labels(root / collect.LABELS_FILE)
    folders = collect._collectible_folders(root / 'nights', 'development')
    first, last = collect._life_start(folders), collect._life_start((folders[-1],))
    arrivals = collect._arrivals(load_person_file(persona_path), first)
    endpoint = datetime.combine(last + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return ResolutionInputs(labels, arrivals, endpoint)


def host_records(user: str, persona: str, as_of: datetime) -> Records:
    """Verify synthetic status before reading content; never open a real person's stores."""
    _named('anjo.core.ids:validate_user_id')(user)
    if not host_check()(user):
        raise ValueError('only registered synthetic accounts may be reported')
    persona_path = Path(__file__).parents[1] / 'personas' / f'{persona}.json'
    if not persona_path.is_file():
        raise ValueError('unknown synthetic persona')
    storage = 'anjo.sleep.app.storage:'
    log = _named(storage + 'workspace_log')(user)
    targets = _named(storage + 'target_store')(user)
    evidence = _named(storage + 'evidence_store')(user)
    brief = _named(storage + 'brief_store')(user)
    records = tuple(b for b in brief.records() if b.compiled_at <= as_of)
    if not records:
        raise ValueError('no served brief compiled by the cutoff; cannot run a valid comparison')
    latest = max(enumerate(records), key=lambda pair: (pair[1].compiled_at, pair[0]))[1]
    mind_ids = tuple(sorted({item for b in records for item in b.folded if item.startswith('mind-')}))
    events = tuple(log.events())
    stored = tuple(e for e in evidence.stored(as_of.date()) if e.created_at <= as_of)
    expectations = tuple(e for e in targets.expectations() if e.made_at <= as_of)
    # The target ledger has no mutation timestamps. Use only ids witnessed by the
    # cutoff; state_brief deliberately emits no current labels/statuses or revisions.
    known = {e.target_id for e in stored} | {e.target_id for e in expectations}
    known |= {t for q in fold_to(events, as_of.date()).questions for t in q.target_ids}
    fixed = tuple(t for t in targets.targets() if t.id in known)
    from anjo.sleep.night.nightlog import summaries

    alone = tuple(date.fromisoformat(s['night']) for s in
                  summaries(_named('anjo.sleep.app.ownership:night_logs')(user)) if s.get('alone') is True)
    return Records(log.path, events, fixed, expectations, stored,
                   _named(storage + 'checkpoint_registry')().last_promoted(user), latest.served, mind_ids,
                   host_resolution(user, persona_path), alone)


def host_model(as_of: datetime) -> ModelClient:
    """Construct the serving client only after --go, pinned to the same virtual day."""
    from anjo.sleep.serving.backend import from_env

    return from_env().client().on(as_of.date())


def _empty_served(text: str, mind_ids: tuple[str, ...], workspace_ids: frozenset[str]) -> str:
    """Remove exactly MIND-supported prose lines under brief_v1's citation contract.

    Mixed or missing artifact citations cannot establish an exact ablation: refuse instead
    of stripping unrelated facts or leaving an untraceable finding in the empty control.
    """
    lines = text.splitlines(keepends=True)
    cited = frozenset(re.findall(r'\[([^\]\n]+)\]', text))
    if set(mind_ids) - cited:
        raise ValueError('served brief has folded MIND artifacts without citations; exact removal is unavailable')
    mind = frozenset(mind_ids) | workspace_ids | frozenset(c for c in cited if c.startswith('mind-'))
    kept: tuple[str, ...] = ()
    for line in lines:
        refs = frozenset(re.findall(r'\[([^\]\n]+)\]', line))
        if refs & mind:
            if refs - mind:
                raise ValueError('served brief mixes MIND and other citations on one line; exact removal is unavailable')
        else:
            kept += (line,)
    return ''.join(kept)


def _briefs(source: Records, end: date, as_of: datetime) -> tuple[str, str]:
    ws = fold_to(source.events, end)
    if not source.common.strip():
        raise ValueError('the probe requires the full served brief')
    if any(marker in source.common for marker in ('<MIND>', '</MIND>', '<SERVED_BRIEF>', '</SERVED_BRIEF>')):
        raise ValueError('served brief contains reserved probe delimiters; exact framing is unavailable')
    ids = frozenset(r.id for records in (ws.questions, ws.hypotheses, ws.predictions, ws.findings) for r in records)
    empty_served = _empty_served(source.common, source.mind_evidence_ids, ids)

    def build(variant: Literal['week', 'empty'], served: str) -> str:
        state = state_brief(ws, source.targets, source.expectations, as_of, variant=variant)
        fixed, _, mind = state.partition('<MIND>\n')
        # The envelope retains all unrelated served text verbatim in both variants;
        # only MIND-cited lines and the kernel state change inside its delimiters.
        return fixed + '<MIND>\n<SERVED_BRIEF>\n' + served + '\n</SERVED_BRIEF>\n' + mind

    return build('week', source.common), build('empty', empty_served)


def paired(source: Records, user: str, prompt: str, end: date, as_of: datetime,
           model: ModelClient) -> ProbeTrace:
    """Use the same adapter snapshot, probe id and run seed for both arm-4 calls."""
    week, empty = _briefs(source, end, as_of)
    probe = Probe('week-alone', user, 'new_work', prompt, None, None, ())
    deps = arms.ArmDeps(model, FixedAdapter(source.adapter), FixedBrief(week), _no_evidence,
                        NoLogprobs(), {}, end, temperature=0.0, run_seed=0)
    first = arms.reply('4', user, probe, deps=deps)
    second = arms.reply('4', user, probe,
                        deps=replace(deps, contexts=FixedBrief(empty), context_cache={}))
    return ProbeTrace(prompt, source.adapter.id if source.adapter else 'base', MODEL,
                      arms.reply_seed(0, probe.id, '4'), 0.0, week, empty, first.text, second.text)


def _window(source: Records, start: date, end: date, as_of: datetime) -> WeekHistory:
    whole = history(source.events, source.targets, source.expectations, source.evidence, as_of=as_of,
                    resolution=source.resolution)
    by_night = {n.night: n for n in whole.nights}
    calendar = tuple(start + timedelta(days=i) for i in range((end - start).days + 1))
    return replace(whole, nights=tuple(replace(by_night.get(n, NightSlice(n, ())), alone=n in source.alone_nights) for n in calendar),
                   moments=tuple(m for m in whole.moments if start <= (m.night or m.at.date()) <= end))


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print,
         records: Callable[[str, str, datetime], Records] = host_records,
         model_factory: Callable[[datetime], ModelClient] = host_model,
         prices: tuple[float, float] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', required=True)
    parser.add_argument('--persona', default='film_founder')
    parser.add_argument('--start', type=date.fromisoformat, required=True)
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--adapter', help='checkpoint id (must match this account snapshot)')
    parser.add_argument('--go', action='store_true')
    args = parser.parse_args(argv)
    if args.days < 1 or not args.prompt.strip():
        parser.error('--days must be positive and --prompt nonempty')
    if Path(args.persona).name != args.persona:
        parser.error('--persona must be a name, not a path')
    end = args.start + timedelta(days=args.days - 1)
    as_of = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc)
    source = records(args.user, args.persona, as_of)
    if any(t.person_id != args.user for t in source.targets) or any(e.person_id != args.user for e in source.evidence):
        raise ValueError('stored content belongs to another person')
    if source.adapter is not None and source.adapter.person_id != args.user:
        raise ValueError('adapter belongs to another person')
    if args.adapter and (source.adapter is None or source.adapter.id != args.adapter):
        raise ValueError('--adapter must match the captured personal checkpoint')
    chosen = _window(source, args.start, end, as_of)
    counts = Counter(type(e.step.entry).__name__ for n in chosen.nights for e in n.events)
    briefs = _briefs(source, end, as_of)
    tokens = sum(estimate_tokens(b) + estimate_tokens(args.prompt) for b in briefs)
    cost = ('unknown (GPU billing; no token prices supplied)' if prices is None else
            f'${(tokens * prices[0] + 2 * 1024 * prices[1]) / 1_000_000:.6f} at supplied USD/million prices')
    out(f'log: {source.log_path}')
    out(f'window: {args.start} through {end} ({args.days} calendar nights)')
    out('recorded nights: ' + (', '.join(n.night.isoformat() + (' (alone)' if n.alone else '') for n in chosen.nights if n.events or n.alone) or 'none'))
    out('events per kind: ' + str(dict(sorted(counts.items()))))
    out(f'probes: 1 prompt x 2 replies; estimated input tokens {tokens}; estimated cost {cost}')
    out(f'adapter: {source.adapter.id if source.adapter else "base"}; model: {MODEL}; temperature: 0; run seed: 0')
    if not args.go:
        out('plan only: no model client, calls or writes')
        return 0
    outputs = ('week.md', 'week.json', 'week-reply.txt', 'empty-reply.txt', 'probe-inputs.local.json')
    if any((args.out / name).exists() for name in outputs):
        raise ValueError('output already exists; choose a new --out directory')
    trace = paired(source, args.user, args.prompt, end, as_of, model_factory(as_of))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in zip(outputs, (render(chosen, trace), to_json(chosen, trace),
                                       trace.week_reply, trace.empty_reply,
                                       json.dumps({'week': trace.week_brief, 'empty': trace.empty_brief,
                                                   'brief_diff': diff(trace.week_brief, trace.empty_brief),
                                                   'reply_diff': diff(trace.week_reply, trace.empty_reply)},
                                                  ensure_ascii=False, indent=2)), strict=True):
        (args.out / name).write_text(content, encoding='utf-8')
    out(f'wrote reports and both replies under {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
