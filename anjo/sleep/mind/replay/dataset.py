"""Replay's two datasets (Task 17b): thinker-1 and thinker-shuffled, as Task 5's ``Dataset``.

- **The start states both draw on:** each labelled one (an undetermined start state is dropped for
  all its branches) whose branches all carry cleared provenance and at least one of which is
  complete. Both datasets take exactly one trajectory from each.
- **thinker-1** takes each start state's best-scoring complete trajectory; a tie goes to fewer
  tokens, then the lower branch index (``best``).
- **thinker-shuffled** permutes the scores among the start state's k branches *before* the winner
  is picked (``permute``, then the same ``best``), so it trains on a random branch's trajectory.
  The permutation is seeded per start state.
- **Rows:** each of the trajectory's targets (``Branch.targets``: its turns on the start state's
  question that the kernel applied, each reply ended at a stop token) is one example: the prompt
  is the brief and the rendered workspace as one user message, exactly what the thinker was sent
  (``BriefedThinker``), the completion is Glimmer's raw reply, and the row is rendered on the date
  it was served on (the branch's night). A row past ``DEFAULT_MAX_SEQ_TOKENS`` is dropped
  (``truncated``).
- **Same size and settings:** the dataset with more rows is cut to the other's count by a seeded
  draw (``over_cap``), so both train with the same step plan.
- **Provenance:** every text in a row comes from Spark Open or deterministic code, and Glimmer's
  own replies are the targets. A start state whose label isn't Spark Open's by the API's report,
  whose workspace or any branch holds text of another model (or of unknown ancestry), or with a
  turn any other model wrote, is refused for both datasets (``provenance``). No hosted output trains anything (``may_train`` with hosted output off).
- **Synthetic lives only:** every start state must come from one of the lives given, each a
  synthetic development life (``check_synthetic``); anything else is ``ReplayRefused``.
"""
from __future__ import annotations

import dataclasses
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.mind.inquiry.types import Turn
from anjo.sleep.mind.replay.score import score_start
from anjo.sleep.mind.replay.types import (
    TOKEN_PENALTY,
    Branch,
    LabelBook,
    ReplayRefused,
    Scored,
    StartState,
    check_synthetic,
)
from anjo.sleep.provenance import GLIMMER, may_train, union
from anjo.sleep.train.dataset import DEFAULT_MAX_SEQ_TOKENS, ROLE_OVERHEAD_TOKENS, Dataset, Example

REPLAY_DROPS: Final = ("undetermined", "provenance", "incomplete")


@dataclass(frozen=True)
class ReplayData:
    """Both datasets, the number of start states they each draw one trajectory from, and the
    start states dropped, by reason (``REPLAY_DROPS``)."""

    thinker1: Dataset
    shuffled: Dataset
    starts: int
    dropped: Mapping[str, int] = field(hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dropped", MappingProxyType(dict(self.dropped)))


# ── Selection ─────────────────────────────────────────────────────────────────


def _rank(scored: Scored) -> tuple[float, int, int]:
    return -scored.score, scored.branch.tokens, scored.branch.index


def best(scored: Sequence[Scored]) -> Scored | None:
    """The best-scoring complete branch (ties: fewer tokens, then the lower index), if any."""
    complete = [s for s in scored if s.branch.complete]
    return min(complete, key=_rank) if complete else None


def permute(scored: Sequence[Scored], rng: random.Random) -> tuple[Scored, ...]:
    """The same branches with their scores permuted among them."""
    values = [(s.brier, s.score) for s in scored]
    rng.shuffle(values)
    return tuple(dataclasses.replace(s, brier=b, score=v) for s, (b, v) in zip(scored, values, strict=True))


def pick_shuffled(scored: Sequence[Scored], rng: random.Random) -> Scored | None:
    """thinker-shuffled's pick: the scores permuted first, then ``best``."""
    return best(permute(scored, rng))


# ── Rows ──────────────────────────────────────────────────────────────────────


def served_prompt(brief: str, turn: Turn) -> str:
    """The text the thinker was sent for ``turn``: the brief, then the rendered workspace."""
    return f"{brief}\n\n{turn.prompt}"


def _example(branch: Branch, turn: Turn, brief: str) -> Example:
    if branch.day is None:
        raise ValueError(f"branch {branch.index} of {branch.start_id} records no date its prompts were served on")
    prompt = (("user", served_prompt(brief, turn)),)
    tokens = sum(estimate_tokens(text) + ROLE_OVERHEAD_TOKENS for _, text in prompt)
    tokens += estimate_tokens(turn.reply.text) + ROLE_OVERHEAD_TOKENS
    return Example(
        id=f"{branch.start_id}:b{branch.index}:t{turn.seq}", prompt=prompt, completion=turn.reply.text,
        context_dropped=False, tokens=tokens, new=True, current_date=branch.day,
    )


def _rows(winners: Iterable[Branch], brief: str, max_tokens: int) -> tuple[list[Example], int]:
    rows = [_example(branch, turn, brief) for branch in winners for turn in branch.targets]
    kept = [row for row in rows if row.tokens <= max_tokens]
    return kept, len(rows) - len(kept)


def _dataset(rows: list[Example], truncated: int, n: int, rng: random.Random) -> Dataset:
    keep = frozenset(rng.sample(range(len(rows)), n)) if len(rows) > n else frozenset(range(len(rows)))
    examples = tuple(row for index, row in enumerate(rows) if index in keep)
    return Dataset(
        examples=examples, tokens=sum(e.tokens for e in examples),
        dropped={"truncated": truncated, "over_cap": len(rows) - len(examples)},
    )


# ── Building both ─────────────────────────────────────────────────────────────


def _lives(lives: Iterable[Life]) -> dict[str, Life]:
    found: dict[str, Life] = {}
    for life in lives:
        check_synthetic(life)
        found[life.id] = life
    return found


def cleared(start: StartState, branches: Sequence[Branch]) -> bool:
    """Every text the start state's rows could hold is Spark Open's, Glimmer's or code's, and
    every turn a dataset could train on is Glimmer's own."""
    ancestry = union([start.ancestry, *(branch.ancestry for branch in branches)])
    if not may_train(ancestry, hosted_in_training=False):
        return False
    return all(turn.reply.model == GLIMMER for branch in branches for turn in branch.turns)


def _winners(
    start: StartState, branches: Sequence[Branch], labels: LabelBook, penalty: float, seed: int,
) -> tuple[str | None, Branch | None, Branch | None]:
    """Why the start state is dropped, or its two winners."""
    if labels.holds(start.id) is None:
        return "undetermined", None, None
    if not labels.labels[start.id].usable or not cleared(start, branches):
        return "provenance", None, None
    scored = score_start(branches, start, labels.labels[start.id], penalty=penalty)
    first, shuffled = best(scored), pick_shuffled(scored, random.Random(f"{seed}:{start.id}"))
    if first is None or shuffled is None:
        return "incomplete", None, None
    return None, first.branch, shuffled.branch


def build(
    lives: Iterable[Life], starts: Sequence[StartState], branches: Mapping[str, Sequence[Branch]],
    labels: LabelBook, *, brief: str, penalty: float = TOKEN_PENALTY, seed: int = 0,
    max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS,
) -> ReplayData:
    """thinker-1's and thinker-shuffled's datasets from the labelled start states' branches."""
    held = _lives(lives)
    dropped = dict.fromkeys(REPLAY_DROPS, 0)
    first: list[Branch] = []
    second: list[Branch] = []
    for start in starts:
        if start.life_id not in held:
            raise ReplayRefused(f"start state {start.id} isn't from a synthetic development life given")
        reason, one, other = _winners(start, branches.get(start.id, ()), labels, penalty, seed)
        if reason is not None or one is None or other is None:
            dropped[reason or "incomplete"] += 1
            continue
        first.append(one)
        second.append(other)
    rows1, cut1 = _rows(first, brief, max_seq_tokens)
    rows2, cut2 = _rows(second, brief, max_seq_tokens)
    n, rng = min(len(rows1), len(rows2)), random.Random(f"{seed}:trim")
    return ReplayData(
        thinker1=_dataset(rows1, cut1, n, rng), shuffled=_dataset(rows2, cut2, n, rng), starts=len(first),
        dropped=dropped,
    )
