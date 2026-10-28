"""``python -m anjo.sleep.mind.replay --fake``: replay end to end on offline fakes (Task 17b).

Every step runs on the real kernel, loop and code, with ``replay.fakes`` in each model's place:
the development lives' M runs, the start states, Spark Open's labels, the branches and their
scores, both datasets, the training plan (a dry run: no runner, no hold), and the 17b arms on
held-out lives under a fake judge. The fakes' numbers mean nothing; the run shows the whole path
works with no key and no spend. Nothing is written to disk.

A live run (Glimmer on vLLM, Spark Open, Colab) needs Kevin's go and isn't wired here: without
``--fake`` the command stops.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from anjo.sleep.mind.apd.arms import load_brief, run_arm
from anjo.sleep.mind.replay import compare, dataset, fakes, resolve, sample, score
from anjo.sleep.mind.replay.dataset import ReplayData
from anjo.sleep.mind.replay.types import Branch, LabelBook, StartState
from anjo.sleep.reports import replay as report
from anjo.sleep.train import thinker

DEV_LIVES: Final = 3
HELD_OUT_LIVES: Final = 4
LIVE_STOP: Final = (
    "A live replay run calls Glimmer, Spark Open and Colab, which needs Kevin's go; only --fake runs here."
)


def _sampled(lives: int, seed: int, brief: str) -> tuple[list[StartState], LabelBook, Mapping[str, tuple[Branch, ...]], ReplayData]:
    """Every development life's start states, labels and branches, and both datasets."""
    developed = [fakes.fake_life(n) for n in range(lives)]
    starts: list[StartState] = []
    book = LabelBook()
    branches: dict[str, tuple[Branch, ...]] = {}
    for life in developed:
        found = sample.start_states(life, run_arm("M", life, fakes.fake_models(), brief=brief))
        labels = resolve.label_all(found, {life.id: life}, resolve.SparkLabeler(chat=fakes.FakeLabelChat()))
        for label in labels.labels.values():
            book = book.with_label(label)
        branches.update(sample.sample_all([(life, found)], fakes.fake_replay_models(), labels, seed=seed, brief=brief))
        starts.extend(found)
    return starts, book, branches, dataset.build(developed, starts, branches, book, brief=brief, seed=seed)


def _summary(starts: Sequence[StartState], book: LabelBook, branches: Mapping[str, tuple[Branch, ...]], data: ReplayData) -> str:
    determined = [start for start in starts if book.holds(start.id) is not None]
    scores = [s.score for start in determined for s in score.score_start(branches[start.id], start, book.labels[start.id])]
    dropped = ", ".join(f"{reason} {count}" for reason, count in data.dropped.items())
    return "\n".join((
        f"Start states: {len(starts)} ({len(determined)} labelled, {len(starts) - len(determined)} undetermined); "
        f"{sum(len(b) for b in branches.values())} branches.",
        f"Branch scores: best {max(scores):.3f}, worst {min(scores):.3f}." if scores else "Branch scores: none.",
        f"thinker-1: {len(data.thinker1.examples)} rows, {data.thinker1.tokens:,} tokens; thinker-shuffled: "
        f"{len(data.shuffled.examples)} rows, {data.shuffled.tokens:,} tokens; from {data.starts} start states "
        f"(dropped: {dropped}).",
    ))


def _arms(held_out: int, brief: str) -> str:
    """The 17b arms on held-out fake lives; the two adapters are stand-in thinkers."""
    lives = [fakes.fake_life(100 + n, split="held_out") for n in range(held_out)]
    thinkers = compare.ArmThinkers(
        m=fakes.FakeGlimmer(), replay=fakes.FakeGlimmer(seed=1, temperature=0.4),
        shuffled=fakes.FakeGlimmer(seed=2, temperature=0.4),
    )
    runs = compare.run_replay_arms(lives, fakes.fake_models(), thinkers, brief=brief)
    problems = {life.id: fakes.fake_problems(life) for life in lives}
    scored = compare.score_replay_arms(lives, problems, runs, audited=fakes.FakeJudge(), gemini=fakes.FakeJudge())
    return report.section([sheet.score for sheet in scored.audited], scored.gemini)


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.mind.replay", description="Replay (Task 17b) on fakes.")
    parser.add_argument("--fake", action="store_true", help="run every step on offline fakes")
    parser.add_argument("--lives", type=int, default=DEV_LIVES, help="development lives to replay")
    parser.add_argument("--held-out", type=int, default=HELD_OUT_LIVES, help="held-out lives for the arms")
    parser.add_argument("--seed", type=int, default=0, help="the branches' and the shuffle's seed")
    args = parser.parse_args(argv)
    if not args.fake:
        out(LIVE_STOP)
        return 2
    brief = load_brief()
    out(f"Replay on fakes: {args.lives} development lives, {args.held_out} held-out lives. The numbers mean nothing.")
    starts, book, branches, data = _sampled(args.lives, args.seed, brief)
    out(_summary(starts, book, branches, data))
    if data.thinker1.examples:
        out(thinker.describe(thinker.plan(data)))
    out(_arms(args.held_out, brief))
    return 0
