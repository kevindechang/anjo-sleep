"""Does MIND's state cause what she says? An end-to-end intervention run, with a null floor.

The week probe shows one thing only: two reply strings differ (``anjo.sleep.reports.week`` and
``mind.week.diff``). That is compatible with a decorative state, because any re-render changes a
byte. This runs the intervention test all the way to the talker instead: for one person's MIND
state on one night it sends the same served brief, per condition, to the same talkers on N fixed
probes.

- ``own``: the state the night left in her log;
- each edit the snapshot test already uses (``swap_lead``, ``delete_prediction``,
  ``inject_hypothesis``; ``anjo.sleep.mind.apd.interventions.edit_for``), on the question the
  night last worked;
- ``empty``: the kernel's empty workspace, with its served prose ablated the way the week probe
  does it;
- ``null``: ``own`` rendered and asked again, unchanged: the noise floor;
- ``swapped``: another synthetic life's state, for the 2x2.

**The measure.** A semantic distance between the two replies to a probe (MiniLM, the embedder
Anjo already ships, or token-set Jaccard where it isn't importable) and the distance between the
ids each reply cites. An edit's effect is its distance from ``own`` minus the null edit's, paired
over probes, with a seeded bootstrap 95% interval. ``own`` against ``empty`` is measured the same
way, so the week probe's comparison gains the noise floor it lacked. An edit is load-bearing when
its semantic interval's lower bound is above zero; a night is load-bearing when any condition is,
and ``load_bearing_share`` counts that across a run of nights.

**The 2x2.** State {own, another life's} x talker {served Glimmer, Spark Open}: a state swap should
move the claims (the ids cited), a talker swap should leave them. The talkers are injected
callables, so tests run the whole thing on fakes.

Plan only unless ``--go``: the plan prints the calls, the token estimate, the cost and the holds
the run would take. ``--go`` reserves them, runs, settles what it spent, and writes
``mind-causal.json`` and ``mind-causal.md`` into the night's results folder. Nothing here scores a
reply's quality or says anything about experience: a distance is a distance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as clock, timedelta, timezone
from pathlib import Path
from typing import Final

from anjo.sleep.budget import Bound, Ledger, default_ledger
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.mind.apd.interventions import EDIT_KINDS, edit_for
from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.types import (
    FULL_RULES,
    Applied,
    Edit,
    EditKind,
    Hypothesis,
    KernelRules,
    Question,
    RunStart,
    Step,
    Workspace,
)
from anjo.sleep.mind.week import fold_to, probe_brief
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import held, spark_usd, spark_worst_usd, usd
from anjo.sleep.probes.arms import reply_seed
from anjo.sleep.probes.clients import ModelClient
from anjo.sleep.serving.client import REQUEST_TIMEOUT_S, WAKE_TOTAL_TIMEOUT_S
from anjo.sleep.serving.config import MODEL
from anjo.sleep.tools.fit_estimator import _named
from anjo.sleep.tools.week import Records, host_model, host_records, served_without_mind
from anjo.sleep.types import CheckpointRef

Talker = Callable[[str, str], str]  # what the talker receives (brief, prompt), what it says
Embed = Callable[[str], Sequence[float]]
Replies = Mapping[str, tuple[str, ...]]  # a condition's reply per probe, in probe order

PROBES: Final[tuple[str, ...]] = (
    'What have you noticed about today?',
    'What are you unsure about?',
    'What would you do next, and why?',
    'What do you expect to happen?',
    'What changed your mind recently?',
)
RUN_SEED: Final = 0  # the probe's seed, so one probe is one draw in every condition
REPLY_MAX_TOKENS: Final = 600
SPARK_TIMEOUT_S: Final = REQUEST_TIMEOUT_S
INJECTED: Final = 'the plan was dropped and the work moved to something already in hand'
BOOTSTRAP_RESAMPLES: Final = 2_000
BOOTSTRAP_SEED: Final = 17  # the intervention test's registered seed
CONFIDENCE: Final = 0.95
JSON_FILE: Final = 'mind-causal.json'
MARKDOWN_FILE: Final = 'mind-causal.md'
# Loaded by name: the engine loads the app's embedder seam rather than importing anjo.memory.
EMBEDDER: Final = 'anjo.sleep.app.recall:default_embed'
_STATES: Final[tuple[str, ...]] = ('own', 'swapped')
_TALKERS: Final[tuple[str, ...]] = ('A', 'B')
_TOKEN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9']+")
_CITED: Final[re.compile] = re.compile(r'\[([^\[\]\n]{1,120})\]')
_ID: Final[re.compile] = re.compile(r'^[A-Za-z][\w.:-]*$')


# ── The state, and what each condition sends ──────────────────────────────────


@dataclass(frozen=True)
class Condition:
    """One rendered state: its name, what the talker receives, the MIND digest it carries, and the
    edit that made it (``None`` for own, the empty twin, the null re-render and the swap)."""

    name: str
    brief: str
    digest: str
    edit: EditKind | None = None


@dataclass(frozen=True)
class Skip:
    """A condition the night's state can't supply, and why. Recorded, never silent."""

    name: str
    reason: str


@dataclass(frozen=True)
class FixedHypothesis:
    """The injected alternative, a fixed string rather than a model's: it is the same in every
    condition and for every talker, so the comparison stays paired. APD's own test writes one that
    is plausible and wrong; this one only has to move the state."""

    statement: str

    def write(self, question: Question, hypotheses: Sequence[Hypothesis]) -> str:
        del question, hypotheses
        return self.statement


def rules_of(ws: Workspace) -> KernelRules:
    """The rules the latest ``RunStart`` in the log ran under, so an edit is recorded under them."""
    return next((step.entry.rules for step in reversed(ws.history)
                 if isinstance(step.entry, RunStart)), FULL_RULES)


def focus_question(ws: Workspace) -> str | None:
    """The question the night last focused that holds hypotheses, else the one with most: edits
    apply to a question she worked, never to a fresh one."""
    held = {q.id: len(ws.hypotheses_of(q.id)) for q in ws.questions}
    seen = tuple(step.focus for step in reversed(ws.history) if step.focus is not None)
    return next((q for q in seen if held.get(q, 0)), None) or max(
        (q for q in held if held[q]), key=lambda q: (held[q], q), default=None)


def edited(ws: Workspace, kind: EditKind, question_id: str, *, at: datetime,
           statement: str) -> tuple[Workspace, Edit] | str:
    """``ws`` after ``kind``, or the reason the edit doesn't apply (an intervention's own reason)."""
    edit = edit_for(kind, ws, question_id, FixedHypothesis(statement))
    if not isinstance(edit, Edit):
        return edit
    applied = kernel.apply(ws, Step(seq=ws.seq, at=at, entry=edit), rules=rules_of(ws))
    if not isinstance(applied, Applied):  # an edit is a mark, recorded and never rejected
        raise ValueError('the kernel rejected an edit, which is a mark')
    return applied.ws, edit


def conditions(source: Records, night: date, *, other: Records | None = None,
               statement: str = INJECTED) -> tuple[tuple[Condition, ...], tuple[Skip, ...]]:
    """Every condition the night can render: own, the null re-render, the empty twin, each edit
    that applies, and another life's state when one is given."""
    as_of = scope(night)
    ws = fold_to(source.events, night)
    own = probe_brief(ws, source.targets, source.expectations, as_of, source.common)
    empty_served = served_without_mind(source, ws)
    found = (
        Condition('own', own, kernel.digest(ws)),
        Condition('null', own, kernel.digest(ws)),  # the same bytes, asked again: the floor
        Condition('empty', probe_brief(kernel.fold(()), source.targets, source.expectations, as_of,
                                       empty_served), kernel.digest(kernel.fold(()))),
    )
    skips: tuple[Skip, ...] = ()
    question = focus_question(ws)
    for kind in EDIT_KINDS:
        if question is None:
            skips = (*skips, Skip(kind, 'no_question_with_hypotheses'))
            continue
        result = edited(ws, kind, question, at=as_of, statement=statement)
        if isinstance(result, str):
            skips = (*skips, Skip(kind, result))
            continue
        state, edit = result
        found += (Condition(kind, probe_brief(state, source.targets, source.expectations, as_of, source.common),
                            kernel.digest(state), edit.kind),)
    if other is None:
        return found, (*skips, Skip('swapped', 'no_other_life'))
    swapped = fold_to(other.events, night)
    found += (Condition('swapped', probe_brief(swapped, source.targets, source.expectations, as_of, source.common),
                        kernel.digest(swapped)),)
    return found, skips


def scope(night: date) -> datetime:
    """The night's cutoff: midnight after it, the clock both the week probe and the state briefs use."""
    return datetime.combine(night + timedelta(days=1), clock.min, tzinfo=timezone.utc)


# ── The plan ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Plan:
    """What a run would send, computed without a model call: the conditions, the probes, and the
    calls and tokens each talker costs."""

    user: str
    night: date
    as_of: datetime
    conditions: tuple[Condition, ...]
    probes: tuple[str, ...]
    skips: tuple[Skip, ...]
    prices: Prices | None = None

    def _with(self, name: str) -> Condition | None:
        return next((c for c in self.conditions if c.name == name), None)

    @property
    def own(self) -> Condition:
        found = self._with('own')
        assert found is not None
        return found

    @property
    def swapped(self) -> Condition | None:
        return self._with('swapped')

    @property
    def calls_a(self) -> int:
        return len(self.conditions) * len(self.probes)

    @property
    def calls_b(self) -> int:
        return (2 if self.swapped is not None else 1) * len(self.probes)

    def tokens(self) -> tuple[int, int]:
        """(input tokens for talker A, input tokens for talker B), estimated the way the week probe does."""
        prompts = sum(estimate_tokens(probe) for probe in self.probes)
        a = sum(estimate_tokens(c.brief) for c in self.conditions) * len(self.probes) + prompts
        b_briefs = [self.own] + ([self.swapped] if self.swapped is not None else [])
        b = sum(estimate_tokens(c.brief) for c in b_briefs if c is not None) * len(self.probes) + prompts
        return a, b


def planned(user: str, source: Records, night: date, *, other: Records | None = None,
            probes: Sequence[str] = PROBES, prices: Prices | None = None) -> Plan:
    """The plan for ``user``'s night, or ``ValueError`` when a store belongs to someone else."""
    if any(t.person_id != user for t in source.targets) or any(e.person_id != user for e in source.evidence):
        raise ValueError('stored content belongs to another person')
    if other is not None:
        other_person(source, other)  # the swap must be another life's, not this person's other night
    if not probes:
        raise ValueError('the run needs at least one probe')
    found, skips = conditions(source, night, other=other)
    return Plan(user, night, scope(night), found, tuple(probes), skips, prices)


def person_of(records: Records) -> str:
    """The one person a life's own records name."""
    ids = {t.person_id for t in records.targets} | {e.person_id for e in records.evidence}
    if len(ids) != 1:
        raise ValueError("the life's records name no single person")
    return ids.pop()


def other_person(source: Records, other: Records) -> str:
    """The other life's id, read from its own records; refused when it is the same person."""
    found = person_of(other)
    if found == person_of(source):
        raise ValueError("the swap needs another life, not the same person's other night")
    return found


# ── The measures ─────────────────────────────────────────────────────────────


def tokens(text: str) -> frozenset[str]:
    return frozenset(match.group(0).casefold() for match in _TOKEN.finditer(text))


def cites(text: str) -> frozenset[str]:
    """The ids a reply cites: bracketed tokens with an id's shape, so ``[e0]`` and ``[q1.h2a]``
    count and a bracketed phrase does not."""
    return frozenset(match.group(1).strip() for match in _CITED.finditer(text) if _ID.match(match.group(1).strip()))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """The overlap of two sets; two empty sets are wholly alike."""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError('embeddings of different sizes')
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def default_embed() -> Embed | None:
    """MiniLM, the embedder Anjo already ships, through the app's seam; ``None`` where it isn't
    importable (the public tree), where the measure is token-set Jaccard. A model that fails to
    load mid-run is raised, never swapped for another measure: a distance may not change scale
    inside one comparison."""
    try:
        embed: Embed = _named(EMBEDDER)()
    except (ImportError, AttributeError):
        return None
    return embed


class Distances:
    """Semantic and citation distance between two replies, one embedding kept per distinct text."""

    def __init__(self, embed: Embed | None = None) -> None:
        self._embed = embed
        self._vectors: dict[str, Sequence[float]] = {}

    @property
    def name(self) -> str:
        return 'token-set jaccard' if self._embed is None else 'minilm cosine'

    def _vector(self, text: str) -> Sequence[float]:
        if text not in self._vectors:
            assert self._embed is not None
            self._vectors[text] = self._embed(text)
        return self._vectors[text]

    def semantic(self, left: str, right: str) -> float:
        """0 when the replies say the same thing, 1 when they share nothing."""
        if self._embed is None:
            return 1.0 - jaccard(tokens(left), tokens(right))
        return 1.0 - cosine(self._vector(left), self._vector(right))

    def cited(self, left: str, right: str) -> float:
        """1 - the overlap of the ids the replies cite; two replies that cite nothing overlap."""
        return 1.0 - jaccard(cites(left), cites(right))


@dataclass(frozen=True)
class Interval:
    """A paired mean and its seeded bootstrap 95% interval over probes."""

    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True)
class Effect:
    """A condition's distance from ``own`` less the null edit's, paired over probes, in both
    measures."""

    name: str
    semantic: Interval
    cited: Interval

    @property
    def load_bearing(self) -> bool:
        """The condition moved the replies further than a re-render does, not merely differently."""
        return self.semantic.low > 0.0


def bootstrap(values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
              seed: int = BOOTSTRAP_SEED) -> Interval:
    """The mean of ``values`` and a percentile interval over the probes resampled with replacement.
    Seeded, so the same replies give the same interval on a rerun."""
    if not values:
        raise ValueError('an interval needs at least one probe')
    count = len(values)
    mean = math.fsum(values) / count
    rng = random.Random(seed)
    draws = sorted(math.fsum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(resamples))
    low, high = int((1.0 - CONFIDENCE) / 2.0 * resamples), int((1.0 + CONFIDENCE) / 2.0 * resamples)
    return Interval(mean, draws[low], draws[min(high, resamples - 1)])


def effect(name: str, replies: Replies, distances: Distances) -> Effect:
    """``name``'s distance from own, less the null re-render's, probe by probe."""
    own, found, null = replies['own'], replies[name], replies['null']
    if not len(own) == len(found) == len(null):
        raise ValueError('every condition is asked the same probes')
    semantic = tuple(distances.semantic(a, b) - distances.semantic(a, n)
                     for a, b, n in zip(own, found, null, strict=True))
    cited = tuple(distances.cited(a, b) - distances.cited(a, n)
                  for a, b, n in zip(own, found, null, strict=True))
    return Effect(name, bootstrap(semantic), bootstrap(cited))


def null_floor(replies: Replies, distances: Distances) -> Interval:
    """How far ``own``'s own re-render lands from it: what no edit may be credited for."""
    return bootstrap(tuple(distances.semantic(a, b)
                           for a, b in zip(replies['own'], replies['null'], strict=True)))


@dataclass(frozen=True)
class Contrast:
    """Two cells of the 2x2, paired over probes: how far apart they read and how many claims they
    share (``claims`` 1.0: the same ids cited)."""

    left: str
    right: str
    semantic: float
    claims: float


@dataclass(frozen=True)
class Grid:
    """The 2x2 and its two ways of differing: a state swap (rows) and a talker swap (columns)."""

    cells: tuple[str, ...]
    rows: tuple[Contrast, ...]
    columns: tuple[Contrast, ...]


def grid(cells: Mapping[tuple[str, str], tuple[str, ...]], distances: Distances) -> Grid:
    """Every row and column of the 2x2, paired over probes."""
    def contrast(left: tuple[str, str], right: tuple[str, str]) -> Contrast:
        semantic = tuple(distances.semantic(a, b) for a, b in zip(cells[left], cells[right], strict=True))
        claims = tuple(1.0 - distances.cited(a, b) for a, b in zip(cells[left], cells[right], strict=True))
        return Contrast(f'{left[0]}/{left[1]}', f'{right[0]}/{right[1]}',
                        math.fsum(semantic) / len(semantic), math.fsum(claims) / len(claims))

    return Grid(
        cells=tuple(f'{state}/{talker}' for state in _STATES for talker in _TALKERS),
        rows=(contrast(('own', 'A'), ('swapped', 'A')), contrast(('own', 'B'), ('swapped', 'B'))),
        columns=(contrast(('own', 'A'), ('own', 'B')), contrast(('swapped', 'A'), ('swapped', 'B'))),
    )


# ── The run ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spend:
    """What the run asked and spent: calls per talker, tokens, GPU seconds, dollars when priced."""

    calls_a: int
    calls_b: int
    gpu_s: float
    tokens_in: int
    tokens_out: int
    usd_a: float
    usd_b: float


@dataclass(frozen=True)
class Result:
    """One night's run: its plan, every talker's replies, the measure and what it cost."""

    user: str
    night: date
    as_of: datetime
    digest: str
    probes: tuple[str, ...]
    conditions: tuple[Condition, ...]
    replies: Replies
    effects: tuple[Effect, ...]
    floor: Interval
    grid: Grid | None
    skips: tuple[Skip, ...]
    spend: Spend
    embedder: str

    @property
    def load_bearing(self) -> bool:
        """Any condition moved the replies beyond the floor on this night."""
        return any(found.load_bearing for found in self.effects)

    @property
    def share(self) -> float:
        """The share of the night's conditions that were load-bearing."""
        return sum(found.load_bearing for found in self.effects) / len(self.effects) if self.effects else 0.0


def load_bearing_share(results: Sequence[Result]) -> float:
    """The share of nights whose state was load-bearing: the headline across a run of nights."""
    return sum(result.load_bearing for result in results) / len(results) if results else 0.0


def _messages(brief: str, prompt: str) -> list[dict[str, str]]:
    return [{'role': 'system', 'content': brief}, {'role': 'user', 'content': prompt}]


@dataclass
class ServedTalker:
    """Talker A: the served Glimmer client the week probe uses, the person's adapter, temperature
    0. The seed comes from the probe alone, so one probe is one draw in every condition."""

    model: ModelClient
    adapter: CheckpointRef | None
    calls: int = 0
    gpu_s: float = 0.0
    timer: Callable[[], float] = time.monotonic

    def __call__(self, brief: str, prompt: str) -> str:
        self.calls += 1
        started = self.timer()
        try:
            reply = self.model.complete(self.adapter, _messages(brief, prompt), temperature=0.0,
                                        seed=reply_seed(RUN_SEED, prompt, '4'))
        finally:
            self.gpu_s += max(0.0, self.timer() - started)
        return reply.text


@dataclass
class SparkTalker:
    """Talker B: Spark Open on the provider ``ANJO_SLEEP_SPARK`` selects, at temperature 0."""

    chat: Callable[..., ChatResult]
    model: str
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def __call__(self, brief: str, prompt: str) -> str:
        self.calls += 1
        result = self.chat(self.model, _messages(brief, prompt), None, SPARK_TIMEOUT_S, temperature=0.0,
                           max_tokens=REPLY_MAX_TOKENS, seed=reply_seed(RUN_SEED, prompt, '4'))
        self.tokens_in += result.input_tokens
        self.tokens_out += result.output_tokens
        return result.text


def spend_of(talker_a: Talker, talker_b: Talker, prices: Prices | None) -> Spend:
    """What the two talkers report; a product price is needed for the dollars, not for the counts."""
    rate = prices.serve_gpu_per_hour if prices is not None else 0.0
    gpu_s = getattr(talker_a, 'gpu_s', 0.0)
    tokens_in, tokens_out = getattr(talker_b, 'tokens_in', 0), getattr(talker_b, 'tokens_out', 0)
    return Spend(
        calls_a=getattr(talker_a, 'calls', 0), calls_b=getattr(talker_b, 'calls', 0), gpu_s=gpu_s,
        tokens_in=tokens_in, tokens_out=tokens_out,
        usd_a=rate * gpu_s / 3600.0,
        usd_b=spark_usd(prices, tokens_in, tokens_out) if prices is not None else 0.0,
    )


def execute(plan: Plan, *, talker_a: Talker, talker_b: Talker, distances: Distances,
            prices: Prices | None = None) -> Result:
    """Ask both talkers every planned probe and measure each condition against the null floor.
    Nothing is written and no budget is held here: the CLI wraps this in the holds."""
    asked = {condition.name: tuple(talker_a(condition.brief, probe) for probe in plan.probes)
             for condition in plan.conditions}
    swapped = plan.swapped
    cells: dict[tuple[str, str], tuple[str, ...]] = {('own', 'A'): asked['own'],
                                                     ('own', 'B'): tuple(talker_b(plan.own.brief, probe)
                                                                         for probe in plan.probes)}
    found_grid: Grid | None = None
    if swapped is not None:
        cells[('swapped', 'A')] = asked['swapped']
        cells[('swapped', 'B')] = tuple(talker_b(swapped.brief, probe) for probe in plan.probes)
        found_grid = grid(cells, distances)
    names = tuple(c.name for c in plan.conditions if c.edit is not None) + ('empty',)
    return Result(
        user=plan.user, night=plan.night, as_of=plan.as_of, digest=plan.own.digest, probes=plan.probes,
        conditions=plan.conditions, replies=asked,
        effects=tuple(effect(name, asked, distances) for name in names),
        floor=null_floor(asked, distances), grid=found_grid, skips=plan.skips,
        spend=spend_of(talker_a, talker_b, prices), embedder=distances.name,
    )


# ── The reports ──────────────────────────────────────────────────────────────


def _quote(text: str) -> str:
    return repr(' '.join(text.split())[:180])


def _interval(found: Interval) -> str:
    return f'{found.mean:.3f} [{found.low:.3f}, {found.high:.3f}]'


def render(result: Result) -> str:
    """The night's causal run as Markdown: conditions, effects against the floor, the 2x2, skips
    and spend. Reply excerpts only; a distance is not a quality score."""
    lines = ['# Does her state change what she says?', '',
             f'{result.user}, night {result.night.isoformat()}; as of {result.as_of.isoformat()}.',
             f'MIND digest {result.digest}. {len(result.probes)} fixed probes; distance: {result.embedder}.', '',
             'Effect = the distance from `own` to a condition, less the distance from `own` to the null '
             're-render, paired over probes; the interval is a seeded bootstrap 95%. A distance is not a '
             'quality score, and nothing here says a reply is better. Credences are heuristic; no claim '
             'is made about experience.', '',
             f'Null floor (own against its re-render): {_interval(result.floor)}.', '',
             f'Load-bearing on this night: {"yes" if result.load_bearing else "no"} '
             f'({sum(e.load_bearing for e in result.effects)} of {len(result.effects)} conditions).', '',
             '## Conditions', '', '| condition | edit | state digest |', '|---|---|---|']
    lines += [f'| {c.name} | {c.edit or "-"} | {c.digest[:12]} |' for c in result.conditions]
    lines += ['', '## Effect against the null floor', '',
              '| condition | semantic mean [95%] | cited mean [95%] | load-bearing |', '|---|---|---|---|']
    lines += [f'| {e.name} | {_interval(e.semantic)} | {_interval(e.cited)} | '
              f'{"yes" if e.load_bearing else "no"} |' for e in result.effects]
    if result.grid is not None:
        grid_found = result.grid
        lines += ['', '## The 2x2', '',
                   'State {own, another life} x talker {A, B}: `claims` is the mean overlap of the ids each '
                   'pair cites (1.0: the same claims), `semantic` their mean distance.', '',
                   '| contrast | semantic | claims |', '|---|---|---|']
        lines += [f'| {c.left} vs {c.right} | {c.semantic:.3f} | {c.claims:.3f} |'
                  for c in (*grid_found.rows, *grid_found.columns)]
    if result.skips:
        lines += ['', '## Conditions the night could not supply', '']
        lines += [f'- {skip.name}: {skip.reason}' for skip in result.skips]
    spent = result.spend
    lines += ['', '## What it asked and spent', '',
              f'- talker A (served Glimmer, {MODEL}): {spent.calls_a} calls, {spent.gpu_s:.1f} s of GPU, '
              f'${spent.usd_a:.4f} when priced',
              f'- talker B (Spark Open): {spent.calls_b} calls, {spent.tokens_in} tokens in and '
              f'{spent.tokens_out} out, ${spent.usd_b:.4f} when priced', '',
              '## Replies', '']
    for condition in result.conditions:
        lines += [f'### {condition.name}', '', *(_quote(reply) for reply in result.replies[condition.name]), '']
    return '\n'.join(lines).rstrip() + '\n'


def to_json(result: Result) -> str:
    """The same run as JSON: every reply in full, with digests and hashes in place of the briefs."""
    document = {
        'user': result.user, 'night': result.night.isoformat(), 'as_of': result.as_of.isoformat(),
        'digest': result.digest, 'embedder': result.embedder, 'probes': list(result.probes),
        'conditions': [{'name': c.name, 'edit': c.edit, 'digest': c.digest,
                        'brief_sha256': hashlib.sha256(c.brief.encode('utf-8')).hexdigest(),
                        'brief_tokens': estimate_tokens(c.brief)} for c in result.conditions],
        'replies': {name: list(texts) for name, texts in result.replies.items()},
        'effects': [{'name': e.name, 'semantic': asdict(e.semantic), 'cited': asdict(e.cited),
                     'load_bearing': e.load_bearing} for e in result.effects],
        'null_floor': asdict(result.floor), 'load_bearing': result.load_bearing, 'share': result.share,
        'two_by_two': None if result.grid is None else {
            'cells': list(result.grid.cells),
            'rows': [asdict(c) for c in result.grid.rows],
            'columns': [asdict(c) for c in result.grid.columns],
        },
        'skips': [asdict(skip) for skip in result.skips], 'spend': asdict(result.spend),
    }
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + '\n'


# ── The host's seams ─────────────────────────────────────────────────────────


def host_talker_a(source: Records, as_of: datetime) -> ServedTalker:
    """Talker A after --go: the served Glimmer path the week probe uses, on the night's virtual day."""
    return ServedTalker(host_model(as_of), source.adapter)


def host_talker_b(model: str | None) -> SparkTalker:
    """Talker B after --go: Spark Open on the configured provider, or ``model`` on it."""
    from anjo.sleep import spark_provider

    provider = spark_provider.current()
    return SparkTalker(provider.chat, model or provider.model)


def host_pool() -> str:
    """The pool Spark Open's provider bills: the ledger's name for it."""
    from anjo.sleep import spark_provider

    return spark_provider.current().pool


def host_prices() -> Prices | None:
    """The registered prices, when this shell has them; a plan without them says so."""
    try:
        return _named('anjo.sleep.app.night:prices')()
    except (ImportError, ValueError, KeyError):
        return None


def host_ledger() -> Ledger | None:
    """The ledger the run holds on, or ``None`` when this shell names no data dir."""
    try:
        return default_ledger()
    except ValueError:
        return None


def night_dir(user: str, night: date) -> Path:
    """Where a night's results live: ``<root>/<user>/nights/<night>/`` (``reports.results``)."""
    from anjo.sleep.reports.results import results_root

    return results_root() / user / 'nights' / night.isoformat()


# ── The command ──────────────────────────────────────────────────────────────


def _money(prices: Prices | None, calls_a: int, tokens_b: int, calls_b: int) -> tuple[str, str]:
    if prices is None:
        unknown = 'unknown (no prices supplied)'
        return unknown, unknown
    gpu = prices.serve_gpu_per_hour * (calls_a * REQUEST_TIMEOUT_S) / 3600.0
    spark = (tokens_b * prices.spark_in_per_m + calls_b * REPLY_MAX_TOKENS * prices.spark_out_per_m) / 1e6
    return f'~${gpu:.4f} (GPU time, worst case)', f'~${spark:.4f}'


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print,
         records: Callable[[str, str, datetime], Records] = host_records,
         talker_a: Callable[[Records, datetime], Talker] = host_talker_a,
         talker_b: Callable[[str | None], Talker] = host_talker_b,
         pool: Callable[[], str] = host_pool, prices: Prices | None = None,
         prices_for: Callable[[], Prices | None] = host_prices,
         distances: Distances | None = None, ledger_for: Callable[[], Ledger | None] = host_ledger) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', required=True)
    parser.add_argument('--night', type=date.fromisoformat, required=True)
    parser.add_argument('--other', help='a second synthetic life, for the state swap of the 2x2')
    parser.add_argument('--persona', default='film_founder')
    parser.add_argument('--talker-b', help='the model Spark Open serves for talker B (its default otherwise)')
    parser.add_argument('--probes', type=int, default=3)
    parser.add_argument('--out', type=Path, help="the night's results folder (default: its results tree)")
    parser.add_argument('--go', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.probes <= len(PROBES):
        parser.error(f'--probes is 1 to {len(PROBES)}')
    if Path(args.persona).name != args.persona:
        parser.error('--persona must be a name, not a path')
    if args.other == args.user:
        parser.error('--other must be another life than --user')
    night, as_of = args.night, scope(args.night)
    source = records(args.user, args.persona, as_of)
    other = None if args.other is None else records(args.other, args.persona, as_of)
    if any(t.person_id != args.user for t in source.targets) or any(e.person_id != args.user for e in source.evidence):
        raise ValueError('stored content belongs to another person')
    if other is not None and other_person(source, other) != args.other:
        raise ValueError('the swapped life is not the one --other names')
    listed = prices if prices is not None else prices_for()
    plan = planned(args.user, source, night, other=other, probes=PROBES[:args.probes], prices=listed)
    tokens_a, tokens_b = plan.tokens()
    cost_a, cost_b = _money(listed, plan.calls_a, tokens_b, plan.calls_b)
    target = args.out or night_dir(args.user, night)
    gpu_hold = (None if listed is None else Bound(
        listed.serve_gpu_per_hour, plan.calls_a * REQUEST_TIMEOUT_S + WAKE_TOTAL_TIMEOUT_S))
    spark_hold = None if listed is None else usd(spark_worst_usd(listed, plan.calls_b))
    out(f'log: {source.log_path}')
    out(f'night: {night.isoformat()} (as of {as_of.isoformat()}); MIND digest {plan.own.digest[:12]}')
    out(f'conditions: {", ".join(c.name for c in plan.conditions)}; probes: {len(plan.probes)} fixed')
    if plan.skips:
        out('skipped: ' + '; '.join(f'{skip.name} ({skip.reason})' for skip in plan.skips))
    out(f'calls: talker A {len(plan.conditions)} conditions x {len(plan.probes)} probes = {plan.calls_a}; '
        f'talker B {2 if plan.swapped is not None else 1} x {len(plan.probes)} = {plan.calls_b}')
    out(f'tokens: ~{tokens_a} in for A, ~{tokens_b} in for B; at most {REPLY_MAX_TOKENS} out per reply')
    out(f'cost: A {cost_a}; B {cost_b}')
    out('holds: ' + ('unknown (no prices supplied)' if listed is None or gpu_hold is None or spark_hold is None else
                     f'A ${gpu_hold.amount:.2f} on modal_usd; B ${spark_hold.amount:.2f} on {pool()}'))
    out(f'writes: {target / JSON_FILE} and {target / MARKDOWN_FILE}')
    if not args.go:
        out('plan only: no model client, calls or writes')
        return 0
    if any((target / name).exists() for name in (JSON_FILE, MARKDOWN_FILE)):
        raise ValueError('output already exists; choose a new --out')
    ledger = ledger_for()
    if listed is None or gpu_hold is None or spark_hold is None or ledger is None:
        raise ValueError('--go needs prices and the budget ledger; nothing was called')
    a, b = talker_a(source, as_of), talker_b(args.talker_b)
    key = f'mind-causal:{args.user}:{night.isoformat()}'
    with ExitStack() as stack:
        stack.enter_context(held(ledger, f'{key}:a', ('modal_usd',), gpu_hold, lambda: listed.serve_gpu_per_hour
                                 * getattr(a, 'gpu_s', 0.0) / 3600.0))
        stack.enter_context(held(ledger, f'{key}:b', (pool(),), spark_hold, lambda: spark_usd(
            listed, getattr(b, 'tokens_in', 0), getattr(b, 'tokens_out', 0))))
        result = execute(plan, talker_a=a, talker_b=b,
                         distances=distances if distances is not None else Distances(default_embed()),
                         prices=listed)
    target.mkdir(parents=True, exist_ok=True)
    (target / JSON_FILE).write_text(to_json(result), encoding='utf-8')
    (target / MARKDOWN_FILE).write_text(render(result), encoding='utf-8')
    out(f'load-bearing: {"yes" if result.load_bearing else "no"} '
        f'({sum(found.load_bearing for found in result.effects)} of {len(result.effects)} conditions); '
        f'{result.spend.calls_a + result.spend.calls_b} calls')
    out(f'wrote the run to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
