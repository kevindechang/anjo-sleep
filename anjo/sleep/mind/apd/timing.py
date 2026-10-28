"""The Oct 30 timing (Task 17, with 17b): measured throughput, then the held-out projection.

``time_pilot`` runs at the chosen parallelism, on the one server:

1. B and M on ``ceil(parallel / 2)`` development lives at once (``batch.run_lives``, measuring), with
   a ``Meter`` counting Glimmer thinker calls and Spark Open calls and their real price;
2. 17b's labels and branches from one of those M runs, and its training planned as a dry run;
3. the intervention test on one snapshot from those M runs, its three edits at once (nothing kept).

It prints the measured throughput (thinker calls a minute, compute units per life-arm night) and
the expected Spark Open spend per call and per MIND night, then projects the held-out step for 16
and 12 lives, with and without M-control and 17b, kill-resume and the intervention test included,
against the ``colab_cu`` cap and ``COLAB_LINE``, and its expected Spark Open spend against the
``baseten_mind_usd`` cap. The measurements are kept (``timing.json``), so each plan shows the
expected spend beside its worst-case holds.

**The model:** the A100 is billed by the hour whatever runs on it, so a life-arm night costs the
batch's compute units over its life-arm nights at this parallelism. A step that keeps fewer
life-arms busy (kill-resume's 3 lives at once, each 3 M runs one after another, counted whole) costs
its nights' wall time. Training is planned for all 8 development lives at the measured life's rows.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from anjo.sleep.budget import CAPS
from anjo.sleep.mind.apd import intervene
from anjo.sleep.mind.apd.arms import date_night, load_brief
from anjo.sleep.mind.apd.backend import Backend
from anjo.sleep.mind.apd.batch import Out, run_lives
from anjo.sleep.mind.apd.colab_vllm import MIN_SERVE_S, ColabSession
from anjo.sleep.mind.apd.interventions import EDIT_KINDS
from anjo.sleep.mind.apd.live import Meter, SparkCalls, arm_models, held_run
from anjo.sleep.mind.apd.resume import KILL_RUNS  # each counted whole
from anjo.sleep.mind.apd.store import DataDir, write_text
from anjo.sleep.mind.apd.types import (
    DEVELOPMENT_LIVES,
    HELD_OUT_LIVES,
    HELD_OUT_LIVES_CUT,
    INTERVENTION_SNAPSHOTS,
    KILL_RESUME_LIVES,
    LIFE_DAYS,
    ArmName,
    ArmRun,
    Life,
)
from anjo.sleep.mind.replay import dataset, resolve, sample, served, settings
from anjo.sleep.mind.replay.spend import Holds
from anjo.sleep.mind.replay.types import REPLAY_K, Branch, StartState
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.train import thinker
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR

COLAB_CAP: Final = CAPS["colab_cu"]
COLAB_LINE: Final = 300.0  # compute units the held-out runs must stay under, or the cut order applies
SPARK_CAP: Final = CAPS["baseten_mind_usd"]
RUN_MARGIN: Final = 1.5  # a life-arm's measured wall time, with room for a slower one
DEV_ARMS: Final[tuple[ArmName, ...]] = ("B", "M")


def cu(seconds: float) -> float:
    return seconds / 3600.0 * A100_CU_PER_HOUR


@dataclass(frozen=True)
class Timing:
    """What timing measured, at ``parallel`` life-arms at once."""

    parallel: int
    setup_s: float
    batch_runs: int  # life-arm runs in the timed batch
    batch_s: float
    nights: int  # their life-arm nights
    mind_nights: int  # of which MIND's (M's)
    thinker_calls: int
    spark_calls: int
    spark_usd: float
    branches_s: float  # 17b, on one development life
    branches_usd: float
    training_cu: float  # planned for every development life
    intervene_items: int
    intervene_s: float
    intervene_usd: float

    @property
    def cu_per_night(self) -> float:
        return cu(self.batch_s) / max(1, self.nights)

    @property
    def night_wall_s(self) -> float:
        """The wall time of one night with the batch's load on the server."""
        waves = math.ceil(self.batch_runs / max(1, self.parallel))
        return self.batch_s / max(1, LIFE_DAYS * waves)

    @property
    def thinker_per_minute(self) -> float:
        return self.thinker_calls / max(self.batch_s / 60.0, 1e-9)

    @property
    def spark_per_call(self) -> float:
        return self.spark_usd / max(1, self.spark_calls)

    @property
    def spark_per_mind_night(self) -> float:
        return self.spark_usd / max(1, self.mind_nights)

    def intervene_cu(self, items: int) -> float:
        """``items`` snapshot-edits at this parallelism: the measured ones ran ``k`` at once."""
        if self.intervene_items == 0:
            return 0.0
        k = min(self.parallel, self.intervene_items)
        return items * cu(self.intervene_s) / self.intervene_items * k / self.parallel

    def held_out(self, lives: int, *, control: bool, replay: bool) -> tuple[float, float]:
        """The held-out step's compute units and expected Spark Open dollars: A and B at the
        measured night, every MIND arm too; kill-resume and the intervention test included."""
        mind_arms = 3 + (1 if control else 0) + (2 if replay else 0)  # M, M-persist, M-kernel (+ M-control, 17b's)
        units = cu(self.setup_s) + lives * (2 + mind_arms) * LIFE_DAYS * self.cu_per_night
        units += cu(KILL_RUNS * LIFE_DAYS * self.night_wall_s)  # its lives at once, each run after run
        items = INTERVENTION_SNAPSHOTS * len(EDIT_KINDS)
        units += self.intervene_cu(items)
        spark = (lives * mind_arms + KILL_RESUME_LIVES * KILL_RUNS) * LIFE_DAYS * self.spark_per_mind_night
        spark += items * self.intervene_usd / max(1, self.intervene_items)
        if replay:
            units += DEVELOPMENT_LIVES * cu(self.branches_s) + self.training_cu
            spark += DEVELOPMENT_LIVES * self.branches_usd
        return units, spark


def run_s(timing: Timing | None) -> float:
    """The session time a whole life-arm needs before it starts: its measured nights with a
    margin, or ``MIN_SERVE_S`` when timing hasn't run."""
    return MIN_SERVE_S if timing is None else LIFE_DAYS * timing.night_wall_s * RUN_MARGIN


def describe(timing: Timing, *, spark_left: float | None = None) -> str:
    t = timing
    lines = [
        f"Timed at --parallel {t.parallel}: {t.batch_runs} life-arm run(s) of B and M in {t.batch_s / 60:.1f} min "
        f"({cu(t.batch_s):.2f} CU): {t.thinker_calls} thinker calls, {t.thinker_per_minute:.1f} thinker calls per "
        f"minute; {t.cu_per_night:.4f} CU per life-arm night.",
        f"Spark Open, expected (measured per call): {t.spark_calls} call(s) at ${t.spark_per_call:.5f} each, "
        f"${t.spark_per_mind_night:.4f} per MIND night; 17b's labels and branches ${t.branches_usd:.4f} and "
        f"{cu(t.branches_s):.2f} CU per development life, training {t.training_cu:.2f} CU; the intervention test "
        f"${t.intervene_usd / max(1, t.intervene_items):.4f} per snapshot-edit; session setup {cu(t.setup_s):.2f} CU.",
        "Held-out projection (A and B at the measured night, every MIND arm at M's; kill-resume and the "
        f"{INTERVENTION_SNAPSHOTS * len(EDIT_KINDS)} snapshot-edits of the intervention test included):",
    ]
    for lives in (HELD_OUT_LIVES, HELD_OUT_LIVES_CUT):
        for control in (True, False):
            for replay in (False, True):
                units, spark = t.held_out(lives, control=control, replay=replay)
                lines.append(
                    f"N={lives}, {'with' if control else 'without'} M-control, {'with' if replay else 'without'} 17b: "
                    f"{units:.1f} CU ({_against(units, COLAB_CAP)} the {COLAB_CAP:.0f} CU cap, "
                    f"{_against(units, COLAB_LINE)} the {COLAB_LINE:.0f}-unit line); Spark Open expected ${spark:.2f} "
                    f"({_against(spark, SPARK_CAP)} the ${SPARK_CAP:.0f} baseten_mind_usd cap"
                    + ("" if spark_left is None else f", {_against(spark, spark_left)} the ${spark_left:.2f} left")
                    + ")"
                )
    return "\n".join(lines)


def _against(amount: float, limit: float) -> str:
    return "under" if amount <= limit else "OVER"


def save(data: DataDir, timing: Timing) -> None:
    write_text(data.timing, json.dumps(dataclasses.asdict(timing), sort_keys=True))


def load(data: DataDir) -> Timing | None:
    if not data.timing.exists():
        return None
    return Timing(**json.loads(data.timing.read_text(encoding="utf-8")))


def _timed(clock: Callable[[], float], work: Callable[[], object]) -> tuple[object, float]:
    started = clock()
    found = work()
    return found, max(0.0, clock() - started)


def _branches(life: Life, start: StartState, labels: resolve.LabelBook, session: ColabSession, backend: Backend,
              metered: SparkCalls, holds: Holds | None) -> tuple[Mapping[str, tuple[Branch, ...]], float]:
    """One start state's branches: their world query dated on its night, as their thinkers are,
    and their screen's Spark Open held for this start state alone, settled before the next."""
    screen = SparkCalls(backend.spark, backend.prices, backend.ledger, f"timing:branch-screens:{start.id}",
                        per_night=REPLAY_K * MIND_SPARK_CALLS)
    models = arm_models(session, screen)
    date_night(models.clock, start.night)
    replay_models = sample.ReplayModels(thinker_for=served.branch_thinkers(session), world=models.world,
                                        screen=models.screen, resolver_chat=metered)
    with held_run(screen):
        found = sample.sample_all([(life, [start])], replay_models, labels, holds=holds)
    return found, screen.total_usd


def _replay(life: Life, m_run: ArmRun, session: ColabSession, backend: Backend, out: Out,
            clock: Callable[[], float]) -> tuple[float, float, float]:
    """17b's labels and branches from ``m_run``: their seconds and Spark Open dollars, and the
    training planned as a dry run for all the development lives at this life's rows."""
    starts = sample.start_states(life, m_run)
    holds = None if backend.ledger is None or backend.prices is None else Holds(
        backend.ledger, backend.prices, clock, job="apd:timing", gpu=False)
    metered = SparkCalls(backend.spark, backend.prices, per_night=10**9)  # 17b holds these itself
    labels = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=metered), holds=holds)
    branches: dict[str, tuple[Branch, ...]] = {}
    screens, started = 0.0, clock()
    for start in starts:
        found, usd = _branches(life, start, labels, session, backend, metered, holds)
        branches.update(found)
        screens += usd
    seconds = max(0.0, clock() - started)
    replay = dataset.build([life], starts, branches, labels, brief=load_brief(), penalty=settings.token_penalty())
    out(f"17b on {life.id}: {len(starts)} start state(s), {sum(len(b) for b in branches.values())} branch(es), "
        f"{len(replay.thinker1.examples)} thinker-1 row(s)")
    training = 0.0
    if replay.thinker1.examples:
        planned = thinker.plan(replay, scale=DEVELOPMENT_LIVES)
        out(thinker.describe(planned))
        training = planned.compute_units
    return seconds, metered.total_usd + screens, training


def time_pilot(data: DataDir, session: ColabSession, backend: Backend, out: Out, *, setup_s: float, parallel: int,
               clock: Callable[[], float] = time.monotonic) -> Timing:
    """Measure, at ``parallel``, on the development lives (those without runs first; the new runs
    are kept as theirs), then keep the measurements."""
    lives = sorted(data.lives("development"), key=lambda life: any(data.run(life.id, a) for a in DEV_ARMS))
    chosen = lives[:max(1, math.ceil(parallel / 2))]
    if not chosen:
        raise ValueError("timing needs a development life: generate them first")
    meter = Meter()
    runs, batch_s = _timed(clock, lambda: run_lives(data, chosen, DEV_ARMS, session, backend, out, parallel=parallel,
                                                    meter=meter, measure=True))
    assert isinstance(runs, tuple)
    m_run = next(run for run in runs if run.arm == "M")
    branches_s, branches_usd, training_cu = _replay(data.life(m_run.life_id), m_run, session, backend, out, clock)
    snaps = intervene.draw(data, count=1, whole=False)  # a probe from the timed M runs, nothing kept
    probe = Meter()
    _, intervene_s = _timed(clock, lambda: intervene.intervene(data, session, backend, out, parallel=parallel,
                                                               snapshots=snaps, keep=False, meter=probe))
    timing = Timing(
        parallel=parallel, setup_s=setup_s, batch_runs=len(runs), batch_s=batch_s, nights=meter.nights,
        mind_nights=sum(len(data.life(r.life_id).days) for r in runs if r.arm == "M"),
        thinker_calls=meter.thinker_calls, spark_calls=meter.spark_calls, spark_usd=meter.spark_usd,
        branches_s=branches_s, branches_usd=branches_usd, training_cu=training_cu,
        intervene_items=len(snaps) * len(EDIT_KINDS), intervene_s=intervene_s, intervene_usd=probe.spark_usd,
    )
    save(data, timing)
    return timing
