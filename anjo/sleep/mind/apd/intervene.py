"""The intervention test on the live models (Task 17): does her state cause what she does next?

``INTERVENTION_SNAPSHOTS`` mid-inquiry snapshots are drawn from the development lives' M runs on
disk, in the order ``SNAPSHOT_SEED`` fixes (``interventions.draw_snapshots``), only once all
``DEVELOPMENT_LIVES`` of them are there (``DrawRefused`` before), so the draw is the same on every
rerun. Each is resumed after each of the three edits against its unedited twin
(``interventions.intervention_test``), on the APD session's base Glimmer at temperature 0, dated
on the snapshot's night, with the evidence held fixed: both twins read the life's evidence as of
the snapshot.

- **The wrong hypothesis** is written by Spark Open (``wrong_hypothesis_v1.md``, ``SparkWrong``),
  the model that generated the life, from the question, its hypotheses and the development life's
  planted problems, so the injected one is plausible and wrong. A held-out life's problem file is
  never read here (``DataDir.development_problems``).
- **Spend:** each snapshot-edit is one work item, its Spark Open calls (both twins' screen and
  resolver, and the writer's) held first on ``baseten_mind_usd`` at ``EDIT_SPARK_CALLS`` calls'
  worst case and journaled (``SparkCalls``), as the runs are held.
- **Never twice, never mixed:** each finished item is stored under its snapshot's identity
  (``item_name``: the life, its M run, the step the snapshot ends at and its question, then the
  edit) and skipped on a rerun; only the draw's own items are read back (``on_disk``). An item
  starts only when the session can carry ``run_s`` more. One with a failed call (after ``retry``'s
  tries) fails alone: recorded (``DataDir.record_failed``), never stored, run again on a rerun; the
  others go on.
- **One at a time:** vLLM at temperature 0 isn't batch-invariant (a reply can change with what
  else is in its batch), and each item's twins are compared token for token, so items run one
  after another (``parallel`` is 1; any other value is refused). MIND's lock keeps every other
  MIND command off the session.

The section of the report is ``reports.apd.intervention_section``.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

from anjo.sleep.mind.apd import interventions
from anjo.sleep.mind.apd.arms import date_night, mind_core
from anjo.sleep.mind.apd.backend import Backend
from anjo.sleep.mind.apd.batch import Out, check_failed, failed_alone, fresh_journal, in_parallel, serial
from anjo.sleep.mind.apd.colab_vllm import MIN_SERVE_S, ColabSession
from anjo.sleep.mind.apd.interventions import EDIT_KINDS, InterventionResult, Skipped, Snapshot
from anjo.sleep.mind.apd.live import Meter, RunFailed, SparkCalls, arm_models, held_run
from anjo.sleep.mind.apd.resume import resume_point
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import DEVELOPMENT_LIVES, INTERVENTION_SNAPSHOTS, ProblemFile
from anjo.sleep.mind.inquiry.types import EditKind, Hypothesis, Question
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.spark_inquiry import ChatFn

WRONG_PATH: Final = Path(__file__).parent / "prompts" / "wrong_hypothesis_v1.md"
WRONG_MAX_TOKENS: Final = 256
SNAPSHOT_SEED: Final = 17  # registered with the protocol
EDIT_SPARK_CALLS: Final = 2 * MIND_SPARK_CALLS + 1  # both twins' screen and resolver calls, and the writer's
Done = tuple[tuple[InterventionResult, ...], tuple[Skipped, ...]]
_UNSAFE: Final = re.compile(r"[^\w.+-]")


class DrawRefused(ValueError):
    """The snapshots asked for before every development life's M run is on disk."""


@dataclass(frozen=True)
class SparkWrong:
    """``WrongHypothesis`` on Spark Open: a plausible alternative the life's planted problems show
    is wrong. A reply without one fails the item, which a rerun tries again."""

    chat: ChatFn
    problems: ProblemFile

    def write(self, question: Question, hypotheses: Sequence[Hypothesis]) -> str:
        body = {
            "question": question.subject,
            "hypotheses": [h.statement for h in hypotheses if not h.is_catch_all],
            "planted": [{"topic": item.topic, "resolution": item.resolution} for item in self.problems.items],
        }
        result = self.chat([{"role": "system", "content": WRONG_PATH.read_text(encoding="utf-8")},
                            {"role": "user", "content": json.dumps(body)}], WRONG_MAX_TOKENS)
        try:
            statement = json.loads(result.text)["statement"]
        except (ValueError, KeyError, TypeError):
            statement = None
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("Spark Open wrote no wrong hypothesis")
        return statement.strip()


def draw(data: DataDir, *, seed: int = SNAPSHOT_SEED, count: int = INTERVENTION_SNAPSHOTS,
         whole: bool = True) -> tuple[Snapshot, ...]:
    """The snapshots, from the development lives' M runs on disk: all ``DEVELOPMENT_LIVES`` of them,
    or ``DrawRefused``, so a draw never changes under a rerun. ``whole=False`` is timing's probe,
    which keeps nothing."""
    pairs = [(life, run) for life in data.lives("development") if (run := data.run(life.id, "M")) is not None]
    if whole and len(pairs) < DEVELOPMENT_LIVES:
        raise DrawRefused(f"the intervention test draws from all {DEVELOPMENT_LIVES} development lives' M runs, "
                          f"and {len(pairs)} are on disk: run them first")
    return interventions.draw_snapshots(pairs, mind_core(), seed=seed, count=count)


def item_name(snap: Snapshot, kind: EditKind) -> str:
    """A snapshot-edit's name, by its snapshot's identity: the life, its M run, the step the
    snapshot ends at and the question it is on; then the edit."""
    return f"{snap.life_id}.M.s{len(snap.events)}.{_UNSAFE.sub('_', snap.question_id)}.{kind}"


def _item(
    data: DataDir, snap: Snapshot, kind: EditKind, session: ColabSession, backend: Backend, out: Out,
    meter: Meter | None, keep: bool, run_s: float,
) -> Done | None:
    session.check_room(run_s)
    life = data.life(snap.life_id)
    name = item_name(snap, kind)
    spend = fresh_journal(data, f"{'' if keep else 'timing-'}intervene-{name}", backend, out, life_id=life.id,
                          arm=f"intervention {name}")
    spark = SparkCalls(backend.spark, backend.prices, backend.ledger, f"intervene:{name}",
                       per_night=EDIT_SPARK_CALLS, journal=spend, wait=backend.sleep)
    models = arm_models(session, spark)
    date_night(models.clock, resume_point(snap.events).start.night)
    try:
        with held_run(spark):
            done = interventions.intervention_test([snap], {life.id: life}, models,
                                                   SparkWrong(spark, data.development_problems(life.id)), kinds=(kind,))
    except RunFailed as error:
        failed_alone(data, out, f"intervention {name}", life.id, error)
        return None
    if meter is not None:
        meter.add(models, spark, 0)
    if keep:
        data.save_intervention(name, done)
    out(f"intervention {name} on {life.id}: {len(done[0])} resumed, {len(done[1])} skipped")
    return done


def intervene(
    data: DataDir, session: ColabSession, backend: Backend, out: Out, *, parallel: int = 1,
    snapshots: Sequence[Snapshot] | None = None, keep: bool = True, meter: Meter | None = None,
    run_s: float = MIN_SERVE_S,
) -> Done:
    """Every snapshot-edit not on disk, one at a time; then every one's results and skips.
    ``keep=False`` (timing) runs the given snapshots and stores nothing."""
    if parallel != 1:  # twins compared token for token: never batched with another item's requests
        raise ValueError("the intervention test runs one at a time: vLLM at temperature 0 isn't batch-invariant")
    snaps = draw(data) if snapshots is None else tuple(snapshots)
    say = serial(out)
    todo = [(snap, kind) for snap in snaps for kind in EDIT_KINDS
            if not keep or data.intervention(item_name(snap, kind)) is None]
    ran = in_parallel([partial(_item, data, snap, kind, session, backend, say, meter, keep, run_s)
                       for snap, kind in todo], parallel)
    check_failed([f"intervention {item_name(snap, kind)}" for (snap, kind), done in zip(todo, ran, strict=True)
                  if done is None])
    found = [done for done in ran if done is not None] if not keep else [stored(data, snap, kind) for snap in snaps
                                                                         for kind in EDIT_KINDS]
    return (tuple(r for results, _ in found for r in results), tuple(s for _, skipped in found for s in skipped))


def stored(data: DataDir, snap: Snapshot, kind: EditKind) -> Done:
    found = data.intervention(item_name(snap, kind))
    if not (isinstance(found, tuple) and len(found) == 2):
        raise ValueError(f"intervention {item_name(snap, kind)} isn't on disk")
    return found[0], found[1]


def on_disk(data: DataDir) -> Done | None:
    """The draw's stored snapshot-edits' results and skips (no file of another snapshot is read),
    or ``None`` when the test hasn't run."""
    if not data.interventions.exists():
        return None
    names = (item_name(snap, kind) for snap in draw(data) for kind in EDIT_KINDS)
    found = [done for done in (data.intervention(name) for name in names) if done is not None]
    if not found:
        return None
    return (tuple(r for results, _ in found for r in results), tuple(s for _, skipped in found for s in skipped))
