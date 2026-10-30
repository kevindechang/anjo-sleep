"""Replay's start states and branches (Task 17b).

- **Start states** (``start_states``): in a development life's M run, night n runs from the first
  ``RunStart`` naming it, so the workspace at its end is the fold of the log before night n+1's
  start (the whole log after the last night). Each open or active question with at least two
  hypotheses there is one start state, with its hypothesis set frozen, the catch-all last.
- **Branches** (``run_branches``): ``REPLAY_K`` per start state, each on a fresh copy of the log
  with its own thinker (``ReplayModels.thinker_for``: Glimmer at ``BRANCH_TEMPERATURE``, one seed
  per branch). Each runs one night's inquiry budget on the real loop, focused on the start state's
  question, with every tool masked to the start state's night (``as_of``, m = 0): a branch can
  never see what came later. Its prompts are dated on that night too, the night whose evidence it
  sees, as an arm's are (``ReplayModels.thinker_for``). Only the branch's ``RunStart`` names the next night
  (``branch_night``), so K5's stall count starts afresh, as a new night's would. Nothing arrives, so
  there is no notice pass. Each branch ends with its credence vector over the frozen set, a
  hypothesis it added counted in the catch-all (``frozen_credences``), and how each of its
  thinker's calls ended (its thinker records them: ``RecordingThinker``), so only applied turns
  whose reply ended at a stop token train (``Branch.targets``).
- **Provenance:** a start state's ancestry is every model whose text its workspace holds: the
  life's evidence, the M run's thinker turns, and the model each ``world_query`` answer
  (``Answered``) and resolver verdict (``Resolved``) came from, as its API reported it and the M
  run recorded it (``ArmRun.written``). A workspace holding one the run didn't record (a failed
  call's empty answer and a censored prediction aside) has unknown ancestry and never trains; the
  dataset then trains only on cleared models. A branch adds its own thinker turns, the model each
  world answer names, and the model each resolver reply named, as Spark Open's API reported it
  (``ReplayModels.resolver_chat``).

Only a start state with a usable label branches (``sample_all``): an undetermined one, or one whose
label isn't Spark Open's by the API's report, is never sampled.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Protocol

from anjo.sleep.mind.apd.arms import BriefedThinker, LifeSearch, MemoryLog, MindCore, load_brief, mind_core, seen_by
from anjo.sleep.mind.apd.types import ArmRun, Life, model_written
from anjo.sleep.mind.inquiry.ports import InquiryPorts, Thinker, WorldQuery
from anjo.sleep.mind.inquiry.types import (
    LIVE_STATUSES,
    Budget,
    Hypothesis,
    ModelReply,
    RunStart,
    WorkspaceEvent,
    catch_all_id,
)
from anjo.sleep.mind.replay.types import (
    REPLAY_K,
    Branch,
    CallEnd,
    LabelBook,
    ReplayRefused,
    StartState,
    check_synthetic,
)
from anjo.sleep.mind.replay.spend import Holds, SparkMeter, held_gpu, held_spark
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.provenance import union
from anjo.sleep.spark_inquiry import ChatFn, SparkResolver


class RecordingThinker(Thinker, Protocol):
    """A thinker that records how each of its calls ended, one per call, in order."""

    @property
    def calls(self) -> Sequence[CallEnd]: ...


ThinkerFactory = Callable[[int, date], RecordingThinker]  # a branch's seed and its night


@dataclass(frozen=True)
class ReplayModels:
    """The branches' models. ``thinker_for(seed, night)`` is the thinker for one branch (Glimmer
    at ``BRANCH_TEMPERATURE`` with that seed, dated on that night: ``served.branch_thinkers``); the
    world query is the M arm's. The resolver is the M arm's too, ``SparkResolver`` over
    ``resolver_chat`` (Spark Open), built per branch on a chat that keeps the model each reply
    names."""

    thinker_for: ThinkerFactory
    world: WorldQuery
    resolver_chat: ChatFn


# ── Start states ──────────────────────────────────────────────────────────────


def _check_run(life: Life, run: ArmRun) -> None:
    check_synthetic(life)
    if run.arm != "M" or run.life_id != life.id:
        raise ReplayRefused(f"start states come from the life's own M run, not {run.arm} on {run.life_id}")


def night_bounds(events: Sequence[WorkspaceEvent]) -> dict[date, int]:
    """Each night's start: the index of the first ``RunStart`` naming it."""
    bounds: dict[date, int] = {}
    for index, event in enumerate(events):
        entry = event.step.entry
        if isinstance(entry, RunStart) and entry.night not in bounds:
            bounds[entry.night] = index
    return bounds


def _as_of(events: Sequence[WorkspaceEvent]) -> datetime:
    starts = [event.step.entry for event in events if isinstance(event.step.entry, RunStart)]
    return starts[-1].as_of


def _unrecorded(events: Sequence[WorkspaceEvent], written: Mapping[int, str]) -> bool:
    """Whether the log holds model output whose model the M run didn't record: a world answer
    with text, or a resolver's verdict (a censored prediction is the code's)."""
    return any(model_written(event.step.entry) is not None and event.step.seq not in written for event in events)


def _evidence_ancestry(life: Life, as_of: datetime) -> frozenset[str] | None:
    return union(
        None if item.ancestry is None else item.ancestry | ({item.model} if item.model else frozenset())
        for item in seen_by(life.evidence, as_of)
    )


def _of_night(life: Life, run: ArmRun, night: date, events: tuple[WorkspaceEvent, ...], core: MindCore) -> list[StartState]:
    ws, as_of = core.fold(events), _as_of(events)
    end = events[-1].step.seq
    turns = frozenset(t.reply.model for t in run.turns if t.seq <= end)
    written = {w.seq: w.model for w in run.written if w.seq <= end}
    ancestry = None if _unrecorded(events, written) else union(
        [_evidence_ancestry(life, as_of), turns, frozenset(written.values())])
    found: list[StartState] = []
    for question in ws.questions:
        held = ws.hypotheses_of(question.id)
        catch = [h for h in held if h.id == catch_all_id(question.id)]
        alternatives = [h for h in held if h.id != catch_all_id(question.id)]
        if question.status not in LIVE_STATUSES or len(alternatives) < 2 or len(catch) != 1:
            continue
        found.append(StartState(
            id=f"{life.id}:{night.isoformat()}:{question.id}", life_id=life.id, night=night, as_of=as_of,
            question_id=question.id, hypotheses=tuple((h.id, h.statement) for h in (*alternatives, *catch)),
            events=events, targets=tuple(life.targets[night]), ancestry=ancestry,
        ))
    return found


def start_states(life: Life, run: ArmRun, *, core: MindCore | None = None) -> tuple[StartState, ...]:
    """Every start state of ``life``'s M run, night by night."""
    _check_run(life, run)
    mind = core or mind_core()
    bounds = night_bounds(run.events)
    nights = sorted(bounds)
    found: list[StartState] = []
    for n, night in enumerate(nights):
        end = bounds[nights[n + 1]] if n + 1 < len(nights) else len(run.events)
        found.extend(_of_night(life, run, night, run.events[:end], mind))
    return tuple(found)


# ── Branches ──────────────────────────────────────────────────────────────────


def frozen_credences(hypotheses: Sequence[Hypothesis], start: StartState) -> dict[str, float] | None:
    """The credence vector over the start state's frozen set: a hypothesis outside it (one a
    branch added) counts in the catch-all. ``None`` when the question holds no hypotheses."""
    frozen, catch = frozenset(start.frozen_ids), start.frozen_ids[-1]
    vector = dict.fromkeys(start.frozen_ids, 0.0)
    for hypothesis in hypotheses:
        key = hypothesis.id if hypothesis.id in frozen else catch
        vector[key] += hypothesis.credence
    total = sum(vector.values())
    return {key: value / total for key, value in vector.items()} if total > 0 else None


def branch_night(start: StartState) -> date:
    """The night a branch's ``RunStart`` names: the next one, so K5 counts afresh. Its prompts
    stay dated on the start state's night."""
    return start.night + timedelta(days=1)


def branch_seed(start_id: str, index: int, seed: int) -> int:
    digest = hashlib.sha256(f"{start_id}\x1f{index}\x1f{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class _SeenChat:
    """A ``ChatFn`` keeping the model each reply named, as named: the resolver's provenance."""

    inner: ChatFn
    models: list[str] = field(default_factory=list)

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        result = self.inner(messages, max_tokens)
        self.models.append(result.model)
        return result


@dataclass
class _SeenWorld:
    """The world query, keeping the model of every answer the branch's prompts may show."""

    inner: WorldQuery
    models: list[str] = field(default_factory=list)

    def ask(self, text: str, max_tokens: int) -> ModelReply:
        reply = self.inner.ask(text, max_tokens)
        self.models.append(reply.model)
        return reply


def _branch(start: StartState, life: Life, models: ReplayModels, index: int, seed: int, budget: Budget,
            core: MindCore, brief: str) -> Branch:
    world, resolver = _SeenWorld(models.world), _SeenChat(models.resolver_chat)
    thinker = models.thinker_for(seed, start.night)
    ports = InquiryPorts(
        thinker=BriefedThinker(thinker, brief), search=LifeSearch(life.evidence), world=world,
        resolver=SparkResolver(chat=resolver), log=MemoryLog(start.events),
    )
    begun = core.fold(start.events)
    run = core.run_inquiry(
        begun, ports, budget, night=branch_night(start), targets=start.targets, as_of=start.as_of,
        focus=start.question_id,
    )
    if len(thinker.calls) != len(run.turns):
        raise ValueError("a branch's thinker records how each of its calls ended, one per turn")
    ends = {turn.seq: end for turn, end in zip(run.turns, thinker.calls, strict=True)}
    on_question = {event.step.seq for event in run.events if event.step.focus == start.question_id}
    turns = tuple(turn for turn in run.turns if turn.seq in on_question)
    ended = frozen_credences(run.ws.hypotheses_of(start.question_id), start)
    ancestry = union([
        start.ancestry, frozenset(turn.reply.model for turn in run.turns), frozenset(world.models),
        frozenset(resolver.models),
    ])
    branch = Branch(
        start.id, index, seed, ended or frozen_credences(begun.hypotheses_of(start.question_id), start) or {},
        run.budget.tokens_used, turns, False, ancestry, ends={seq: ends[seq] for seq in on_question if seq in ends},
        applied=frozenset(e.step.seq for e in run.events if e.step.applied and e.step.seq in on_question),
        failed=sum(1 for end in thinker.calls if end.failed), day=start.night,
    )
    return dataclasses.replace(branch, complete=ended is not None and not branch.failed and bool(branch.targets))


def run_branches(
    start: StartState, life: Life, models: ReplayModels, *, k: int = REPLAY_K, seed: int = 0,
    budget: Budget = Budget(), core: MindCore | None = None, brief: str | None = None,
) -> tuple[Branch, ...]:
    """``k`` branches from ``start``, each from ``budget`` (one night's), on ``life``'s own tools.
    ``core`` defaults to the real kernel and loop; ``brief`` to APD's ``brief_v1.md``."""
    check_synthetic(life)
    if start.life_id != life.id:
        raise ReplayRefused(f"start state {start.id} isn't from life {life.id}")
    mind, text = core or mind_core(), load_brief() if brief is None else brief
    return tuple(
        _branch(start, life, models, index, branch_seed(start.id, index, seed), budget, mind, text)
        for index in range(k)
    )


def sample_all(
    pairs: Sequence[tuple[Life, Sequence[StartState]]], models: ReplayModels, labels: LabelBook, *,
    k: int = REPLAY_K, seed: int = 0, budget: Budget = Budget(), core: MindCore | None = None,
    brief: str | None = None, holds: Holds | None = None,
) -> Mapping[str, tuple[Branch, ...]]:
    """The branches of every start state with a usable label; no other is ever sampled. With
    ``holds``, each start state's branches hold their Glimmer time and their resolver's Spark Open
    calls first (``spend``)."""
    out: dict[str, tuple[Branch, ...]] = {}
    for life, starts in pairs:
        for start in starts:
            labels.holds(start.id)  # every start state is labelled before any branches
            if not labels.labels[start.id].usable:
                continue
            meter = None if holds is None else SparkMeter(holds.prices)
            metered = models if meter is None else dataclasses.replace(
                models, resolver_chat=meter.wrap(models.resolver_chat))
            with held_gpu(holds, f"branches:{start.id}", k), \
                    held_spark(holds, f"branches:{start.id}:spark", k * MIND_SPARK_CALLS, meter):
                out[start.id] = run_branches(start, life, metered, k=k, seed=seed, budget=budget, core=core, brief=brief)
    return out
