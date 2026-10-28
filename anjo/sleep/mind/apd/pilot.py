"""The APD pilot's steps (Task 17, with 17b on the same server), on one data directory.

- **generate:** the 8 development lives now (``generate.development_specs``), through the gates;
  the held-out lives only at the freeze (``held_out_specs``). Each attempt's worst case is held
  first (``backend.held_attempt``). Each life's problem file is sealed (``store``).
- **run:** every arm on every life of a split, in one Colab session (``serving``: the command's own
  sweep of ``anjo-mind-`` leftovers first, then ``colab_vllm.serve``). A run already on disk is
  skipped, so a rerun resumes; 17b's adapters go up to the session first (``ColabPublisher``).
- **timing:** B and M on one development life, 17b's replay sampled from that M run (labels and
  branches) and its training planned as a dry run; then compute units per life-arm and the
  held-out projection for 16 and 12 lives, with and without M-control and 17b, against
  ``COLAB_LINE``.
- **judge** and **report:** ``judging`` and ``reports/apd.py``, with 17b's section when its arms ran.
- **kill-resume:** ``resume.kill_resume`` on held-out lives, its children attached to this
  session (``kill_setup``).

Times are measured with the clock passed in; nothing here reads the environment but
``kill_setup``, which runs in kill-resume's children.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx

from anjo.sleep.mind.apd import colab_vllm, fakes, gates, generate, judging, resume
from anjo.sleep.mind.apd.arms import load_brief, mind_core
from anjo.sleep.mind.apd.backend import Backend, held_attempt
from anjo.sleep.mind.apd.colab_vllm import ColabSession
from anjo.sleep.mind.apd.generate import GenerationError, GenerationFailed, Generated, LifeSpec
from anjo.sleep.mind.apd.live import SparkCalls, arm_models, held_run, run_live
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import (
    DEVELOPMENT_LIVES,
    HELD_OUT_LIVES,
    HELD_OUT_LIVES_CUT,
    ArmName,
    ArmRun,
    Life,
    LifeScore,
)
from anjo.sleep.mind.inquiry.log import JsonlWorkspaceLog
from anjo.sleep.mind.replay import dataset, resolve, sample, served, settings
from anjo.sleep.mind.replay.spend import SPARK_POOL, Holds
from anjo.sleep.mind.replay.types import REPLAY_K
from anjo.sleep.night.holds import held, spark_worst_usd, usd
from anjo.sleep.night.mind import MIND_SPARK_CALLS
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.prereg.ledger import LedgerEntry
from anjo.sleep.reports import apd as apd_report
from anjo.sleep.reports import replay as replay_report
from anjo.sleep.serving import config
from anjo.sleep.serving.config import ServeConfig
from anjo.sleep.train import thinker
from anjo.sleep.train.runners.colab import A100_CU_PER_HOUR, MIND_SESSION_PREFIX, stop_leftover_sessions
from anjo.sleep.types import CheckpointRef

DEV_ARMS: Final[tuple[ArmName, ...]] = ("B", "M")
HELD_OUT_ARMS: Final[tuple[ArmName, ...]] = ("A", "B", "M", "M-persist", "M-kernel", "M-control")
REPLAY_ARMS: Final[tuple[ArmName, ...]] = ("M+replay", "M+shuffled")
COLAB_LINE: Final = 300.0  # compute units the held-out runs must stay under
FREEZE_KIND: Final = "apd_freeze"
DRAW_SEED_KIND: Final = "apd_draw_seed"
SETUP_ENV: Final = "ANJO_APD_ATTACH"  # kill-resume's children: "<data>|<session>|<port>|<fake>"
KEY_ENV: Final = "ANJO_APD_KEY"
SETUP: Final = "anjo.sleep.mind.apd.pilot:kill_setup"
KILL_RUNS: Final = 3  # uninterrupted, killed, resumed
Out = Callable[[str], None]


class FreezeError(RuntimeError):
    """Held-out lives asked for before the freeze, or from a seed not committed before it."""


def cu(seconds: float) -> float:
    return seconds / 3600.0 * A100_CU_PER_HOUR


# ── Generation ────────────────────────────────────────────────────────────────


def generate_life(spec: LifeSpec, backend: Backend) -> Generated:
    """``generate.generate``'s attempts, each attempt's worst case held before it starts."""
    for attempt in range(generate.MAX_ATTEMPTS):
        with held_attempt(backend, f"{spec.split}:{spec.seed}:{attempt}") as (writer, reader, auditor):
            try:
                built, problems = generate.life(spec.field, spec.families, spec.prevalence,
                                                generate.attempt_seed(spec.seed, attempt), split=spec.split, chat=writer)
            except GenerationFailed:
                continue
            verdicts = gates.check(built, problems, reader=reader, auditor=auditor)
        if gates.passed(verdicts):
            return Generated(built, problems, attempt + 1, verdicts)
    raise GenerationError(f"no {spec.split} life passed the gates in {generate.MAX_ATTEMPTS} attempts")


def spec_ids(spec: LifeSpec) -> frozenset[str]:
    """The ids ``spec``'s life may take, one per attempt (``generate.life`` names a life by its
    split, field and attempt seed)."""
    field_ = re.sub(r"[^a-z0-9]+", "-", spec.field.lower()).strip("-")
    return frozenset(f"{spec.split}-{field_}-{generate.attempt_seed(spec.seed, n)}" for n in range(generate.MAX_ATTEMPTS))


def generate_lives(data: DataDir, specs: Sequence[LifeSpec], backend: Backend, out: Out) -> tuple[str, ...]:
    """Every life of ``specs`` not yet on disk, generated, gated and saved, its problems sealed."""
    made: list[str] = []
    for spec in specs:
        if any(data.has_life(life_id) for life_id in spec_ids(spec)):
            continue
        made_life = generate_life(spec, backend)
        data.save_life(made_life.life, made_life.problems)
        made.append(made_life.life.id)
        out(f"generated {made_life.life.id} ({made_life.attempts} attempt(s), {len(made_life.problems.items)} planted)")
    return tuple(made)


def check_freeze(entries: Sequence[LedgerEntry], draw_seed: str, count: int) -> None:
    """The held-out lives are drawn only at the freeze: the prereg ledger holds an ``apd_freeze``
    entry naming ``held_out_lives=<count>``, after an ``apd_draw_seed`` entry committing this
    seed's hash."""
    if count not in (HELD_OUT_LIVES, HELD_OUT_LIVES_CUT):
        raise FreezeError(f"the held-out lives number {HELD_OUT_LIVES} or {HELD_OUT_LIVES_CUT}")
    frozen = [entry for entry in entries if entry.kind == FREEZE_KIND]
    if not frozen:
        raise FreezeError("APD isn't frozen: the prereg ledger holds no apd_freeze entry")
    digest = hashlib.sha256(draw_seed.encode("utf-8")).hexdigest()
    seeded = [entry for entry in entries if entry.kind == DRAW_SEED_KIND and entry.sha256 == digest]
    if not seeded or seeded[0].seq > frozen[-1].seq:
        raise FreezeError("the draw seed wasn't committed to the prereg ledger before the freeze")
    if f"held_out_lives={count}" not in frozen[-1].note.split():
        raise FreezeError(f"the freeze registered another number of held-out lives than {count}")


def held_out_specs(entries: Sequence[LedgerEntry], draw_seed: str, count: int) -> tuple[LifeSpec, ...]:
    check_freeze(entries, draw_seed, count)
    return generate.draw_held_out(draw_seed, count=count)


# ── Serving and runs ──────────────────────────────────────────────────────────


@contextmanager
def serving(backend: Backend, settings_: ServeConfig, timeout_s: float, out: Out) -> Iterator[ColabSession]:
    """MIND's leftovers stopped, then one Glimmer session, stopped on the way out."""
    stopped, failed = stop_leftover_sessions(backend.cli, prefix=MIND_SESSION_PREFIX)
    if stopped or failed:
        out(f"stopped {stopped} leftover MIND session(s); {failed} failed")
    with colab_vllm.serve(backend.cli, backend.ledger, settings_, timeout_s=timeout_s, tunnels=backend.tunnels,
                          wait=backend.wait) as session:
        out(f"serving Glimmer on {session.session}")
        yield session


@dataclass
class ColabPublisher:
    """17b's ``AdapterPublisher`` for APD: a trained adapter stays in the local cache where
    training put it, and goes up to the APD session's LoRA directory when it serves (``upload``),
    its rendering checked there. Never to Modal."""

    published: dict[str, Path] = field(default_factory=dict)

    def __call__(self, cached: Path, name: str) -> None:
        config.check_adapter(cached, name)
        self.published[name] = cached

    def upload(self, session: ColabSession) -> None:
        for name, folder in self.published.items():
            session.upload_adapter(folder, name)


def run_lives(
    data: DataDir, lives: Sequence[Life], arms: Sequence[ArmName], session: ColabSession, backend: Backend, out: Out,
    *, adapters: Mapping[ArmName, CheckpointRef] | None = None,
) -> tuple[ArmRun, ...]:
    """Every arm on every life, each run saved as it ends; a run already on disk is kept."""
    runs: list[ArmRun] = []
    for life in lives:
        for arm in arms:
            found = data.run(life.id, arm)
            if found is None:
                adapter = None if adapters is None else adapters.get(arm)
                found = run_live(arm, life, session, backend.spark, ledger=backend.ledger, prices=backend.prices,
                                 adapter=adapter)
                data.save_run(found)
                out(f"ran {arm} on {life.id}: {len(found.reports)} report(s)")
            runs.append(found)
    return tuple(runs)


# ── Timing ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Timing:
    """Measured compute units: one B and one M run of a development life, 17b's branches from that
    M run, and the training plan's."""

    b_cu: float
    m_cu: float
    branches_cu: float
    training_cu: float
    setup_cu: float

    def held_out(self, lives: int, *, control: bool, replay: bool) -> float:
        """The held-out runs' projection: A as B, every MIND arm as M, one session's setup."""
        mind_arms = 4 + (1 if control else 0) + (2 if replay else 0)  # M, M-persist, M-kernel (+ M-control, 17b's)
        total = self.setup_cu + lives * (2 * self.b_cu + mind_arms * self.m_cu)
        return total + (DEVELOPMENT_LIVES * self.branches_cu + self.training_cu if replay else 0.0)


def describe_timing(timing: Timing) -> str:
    lines = [
        f"Per life-arm: B {timing.b_cu:.2f} CU, M {timing.m_cu:.2f} CU (A at B's, every MIND arm at M's); "
        f"17b branches {timing.branches_cu:.2f} CU per development life, training {timing.training_cu:.2f} CU; "
        f"session setup {timing.setup_cu:.2f} CU.",
    ]
    for lives in (HELD_OUT_LIVES, HELD_OUT_LIVES_CUT):
        for control in (True, False):
            for replay in (False, True):
                total = timing.held_out(lives, control=control, replay=replay)
                lines.append(f"N={lives}, {'with' if control else 'without'} M-control, "
                             f"{'with' if replay else 'without'} 17b: {total:.1f} CU "
                             f"({'under' if total <= COLAB_LINE else 'OVER'} the {COLAB_LINE:.0f}-unit line)")
    return "\n".join(lines)


def _timed(clock: Callable[[], float], work: Callable[[], object]) -> tuple[object, float]:
    started = clock()
    found = work()
    return found, max(0.0, clock() - started)


def _timed_run(data: DataDir, arm: ArmName, life: Life, session: ColabSession, backend: Backend,
               clock: Callable[[], float]) -> tuple[ArmRun, float]:
    """One development run, timed, and kept as that life's run when it has none yet."""
    found, seconds = _timed(clock, lambda: run_live(arm, life, session, backend.spark, ledger=backend.ledger,
                                                    prices=backend.prices))
    assert isinstance(found, ArmRun)
    if data.run(life.id, arm) is None:
        data.save_run(found)
    return found, seconds


def time_pilot(data: DataDir, life: Life, session: ColabSession, backend: Backend, out: Out, *, setup_s: float,
               clock: Callable[[], float] = time.monotonic) -> Timing:
    """B and M on ``life`` (kept as its development runs), 17b's labels and branches from that M
    run, and a training dry run. ``setup_s`` is how long the session took to serve."""
    _, b_s = _timed_run(data, "B", life, session, backend, clock)
    m_run, m_s = _timed_run(data, "M", life, session, backend, clock)
    starts = sample.start_states(life, m_run)
    holds = None if backend.ledger is None or backend.prices is None else Holds(
        backend.ledger, backend.prices, clock, job="apd-timing", gpu=False)
    labels = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=backend.spark), holds=holds)
    screen = SparkCalls(backend.spark, backend.prices, backend.ledger, f"timing:{life.id}:branch-screens",
                        per_night=max(1, len(starts)) * REPLAY_K * MIND_SPARK_CALLS)
    models = arm_models(session, screen)
    replay_models = sample.ReplayModels(thinker_for=served.branch_thinkers(session), world=models.world,
                                        screen=models.screen, resolver_chat=backend.spark)
    with held_run(screen):
        branches, branch_s = _timed(clock, lambda: sample.sample_all([(life, starts)], replay_models, labels,
                                                                     holds=holds))
    assert isinstance(branches, Mapping)
    replay = dataset.build([life], starts, branches, labels, brief=load_brief(), penalty=settings.token_penalty())
    planned = thinker.plan(replay)
    out(thinker.describe(planned))
    out(f"17b on {life.id}: {len(starts)} start state(s), {sum(len(b) for b in branches.values())} branch(es), "
        f"{len(replay.thinker1.examples)} thinker-1 row(s)")
    return Timing(b_cu=cu(b_s), m_cu=cu(m_s), branches_cu=cu(branch_s), training_cu=planned.compute_units,
                  setup_cu=cu(setup_s))


# ── Judge and report ──────────────────────────────────────────────────────────


def judges(data: DataDir, backend: Backend) -> tuple[judging.CachedJudge, judging.CachedJudge]:
    rates = backend.rates or {}
    return (
        judging.CachedJudge(judging.MUSE, backend.muse, data.judging / "muse-spark.jsonl", backend.ledger,
                            rates.get(judging.MUSE)),
        judging.CachedJudge(judging.GEMINI, backend.gemini, data.judging / "gemini.jsonl", backend.ledger,
                            rates.get(judging.GEMINI)),
    )


def runs_of(data: DataDir, lives: Sequence[Life], arms: Sequence[ArmName]) -> tuple[ArmRun, ...]:
    found = (data.run(life.id, arm) for life in lives for arm in arms)
    return tuple(run for run in found if run is not None)


def judge_lives(data: DataDir, lives: Sequence[Life], backend: Backend, out: Out) -> judging.Judged:
    """Every run on disk for ``lives``, judged; with the fakes, the audit answered by the fake."""
    muse, gemini = judges(data, backend)
    runs = runs_of(data, lives, (*HELD_OUT_ARMS, *REPLAY_ARMS))
    judged = judging.judge_runs(data, lives, runs, muse, gemini)
    if judged.pending and backend.claude is not None:
        judging.answer_pending(data, backend.claude)
        judged = judging.judge_runs(data, lives, runs, muse, gemini)
    out(f"judged {len(runs)} run(s): Muse Spark {muse.calls} call(s), Gemini {gemini.calls}; "
        f"{judged.pending} disagreement(s) await Claude's blind audit in {data.audit / 'pending.jsonl'}")
    return judged


def report(judged: judging.Judged) -> str:
    """The APD claim and per-life table, and 17b's section when its arms ran."""
    if judged.pending:
        raise ValueError(f"{judged.pending} disagreement(s) still await Claude's blind audit")
    sheets = {arm: [s for s in judged.audited if s.score.arm == arm] for arm in ("M", "B")}
    rescored = {arm: [s for s in judged.gemini if s.arm == arm] for arm in ("M", "B")}
    pilot = apd_report.pilot_of(sheets["M"], sheets["B"], rescored["M"], rescored["B"])
    scores: list[LifeScore] = [*(s.score for s in judged.audited), *judged.gemini]
    parts = [apd_report.claim(pilot), apd_report.per_life_table(scores)]
    if any(s.arm in REPLAY_ARMS for s in judged.gemini):
        replay = ("M", *REPLAY_ARMS)
        parts.append(replay_report.section([s.score for s in judged.audited if s.score.arm in replay],
                                           [s for s in judged.gemini if s.arm in replay]))
    return "\n\n".join(parts)


# ── Kill-resume ───────────────────────────────────────────────────────────────


def kill_lives(data: DataDir, lives: Sequence[Life], session: ColabSession, kill_at: int, backend: Backend,
               out: Out) -> tuple[resume.KillResume, ...]:
    """``resume.kill_resume`` on each life, its children attached to ``session``. The children
    hold nothing and report no spend, so each life's three runs are held here first, a night of
    Spark Open per run-night, and settled at that worst case."""
    found: list[resume.KillResume] = []
    with _attached(data, session, backend.fake):
        for life in lives:
            with _kill_held(backend, life):
                result = resume.kill_resume(SETUP, life.id, kill_at, workdir=data.root / "kill")
            out(f"kill-resume {life.id} at {kill_at}: digest {'equal' if result.digest_equal else 'DIFFERS'}, "
                f"conclusions {'agree' if result.conclusions_agree else 'differ'}")
            found.append(result)
    return tuple(found)


@contextmanager
def _kill_held(backend: Backend, life: Life) -> Iterator[None]:
    if backend.ledger is None or backend.prices is None:
        yield
        return
    worst = KILL_RUNS * len(life.days) * spark_worst_usd(backend.prices, MIND_SPARK_CALLS)
    with held(backend.ledger, f"apd:kill:{life.id}", (SPARK_POOL,), usd(worst), lambda: worst):
        yield


@contextmanager
def _attached(data: DataDir, session: ColabSession, fake: bool) -> Iterator[None]:
    before = {name: os.environ.get(name) for name in (SETUP_ENV, KEY_ENV)}
    os.environ[SETUP_ENV] = f"{data.root}|{session.session}|{session.local_port}|{int(fake)}"
    os.environ[KEY_ENV] = session.key
    try:
        (data.root / "kill").mkdir(parents=True, exist_ok=True)
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def kill_setup(life_id: str) -> resume.KillSetup:
    """Kill-resume's setup in each process: the life from the data dir, and the run's models on
    the session the parent serves, attached through its tunnel (the fakes' own server with
    ``--fake``). The parent holds the Spark Open these processes spend (``kill_lives``)."""
    root, session, port, fake = os.environ[SETUP_ENV].split("|")
    key = os.environ[KEY_ENV]
    data = DataDir(Path(root))
    transport = httpx.MockTransport(fakes.FakeVllm(key)) if fake == "1" else None
    attached = colab_vllm.attach(session, int(port), key, transport=transport)
    life = data.life(life_id)
    if fake == "1":
        spark = SparkCalls(fakes.FakeSparkChat(), None)
    else:
        from anjo.sleep.spark_inquiry import spark_open
        spark = SparkCalls(spark_open, None)
    return resume.KillSetup(
        life=life, models=arm_models(attached, spark), core=mind_core(),
        open_log=lambda path: JsonlWorkspaceLog(path, life.id, codec=PLAIN, guard=NO_GUARD),
    )

