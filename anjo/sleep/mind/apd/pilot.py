"""The APD pilot's steps (Task 17, with 17b on the same server), on one data directory.

- **generate:** the 8 development lives now (``generate.development_specs``), through the gates;
  the held-out lives only at the freeze (``held_out_specs``). Each attempt's worst case is held
  first (``backend.held_attempt``). Each life's problem file is sealed (``store``).
- **run:** every arm on every life of a split, in one Colab session (``serving``, after the
  command's own sweep of ``anjo-mind-`` leftovers under MIND's lock, ``mind.sweep``), ``parallel``
  life-arms at a time (``batch``). A run already on disk is skipped, so a rerun resumes; 17b's
  adapters go up to the session first (``ColabPublisher``). A held-out run, and kill-resume, start
  only on lives drawn at the freeze (``check_drawn``).
- **timing** (``timing``) and the **intervention test** (``intervene``), on the same session.
- **judge** and **report:** ``judging`` and ``reports/apd.py``, with 17b's section when its arms ran
  and the intervention test's when it has.
- **kill-resume:** ``resume.kill_resume`` on held-out lives, one at a time (vLLM at temperature 0
  isn't batch-invariant, and the runs are compared token for token), its children attached to this
  session (``kill_setup``), on the parent's Spark Open provider. Each process holds its own Spark
  Open a night at a time on that provider's pool and journals every call; the killed child's open
  night is settled from its journal (``batch.settle_orphans``).

Times are measured with the clock passed in; nothing here reads the environment but
``kill_setup``, which runs in kill-resume's children.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Final

import httpx

from anjo.sleep.budget import Ledger
from anjo.sleep.mind.apd import colab_vllm, fakes, gates, generate, judging, resume
from anjo.sleep.mind.apd.arms import mind_core
from anjo.sleep.mind.apd.backend import Backend, held_attempt
from anjo.sleep.mind.apd.batch import Out, fresh_journal, in_parallel, run_lives, serial, settle_orphans
from anjo.sleep.mind.apd.colab_vllm import ColabSession
from anjo.sleep.mind.apd.generate import GenerationError, GenerationFailed, Generated, LifeSpec
from anjo.sleep.mind.apd.interventions import InterventionResult, Skipped
from anjo.sleep.mind.apd.journal import SpendJournal
from anjo.sleep.mind.apd.live import SparkCalls, arm_models
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import HELD_OUT_LIVES, HELD_OUT_LIVES_CUT, ArmName, ArmRun, Life, LifeScore
from anjo.sleep.mind.inquiry.log import JsonlWorkspaceLog
from anjo.sleep.mind.spark_provider import named
from anjo.sleep.night.deps import Prices
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.prereg.ledger import LedgerEntry
from anjo.sleep.spark_inquiry import ChatFn
from anjo.sleep.reports import apd as apd_report
from anjo.sleep.reports import replay as replay_report
from anjo.sleep.serving import config
from anjo.sleep.serving.config import ServeConfig

__all__ = ["Out", "run_lives"]  # the runs live in ``batch``; the command reaches them here

DEV_ARMS: Final[tuple[ArmName, ...]] = ("B", "M")
HELD_OUT_ARMS: Final[tuple[ArmName, ...]] = ("A", "B", "M", "M-persist", "M-kernel", "M-control")
REPLAY_ARMS: Final[tuple[ArmName, ...]] = ("M+replay", "M+shuffled")
FREEZE_KIND: Final = "apd_freeze"
DRAW_SEED_KIND: Final = "apd_draw_seed"
SETUP_ENV: Final = "ANJO_APD_ATTACH"  # kill-resume's children: "<data>|<session>|<port>|<fake>|<provider>"
KEY_ENV: Final = "ANJO_APD_KEY"
SPEND_ENV: Final = "ANJO_APD_SPEND"  # and "<ledger>|<four prices>" when their Spark Open is held
SETUP: Final = "anjo.sleep.mind.apd.pilot:kill_setup"


class FreezeError(RuntimeError):
    """Held-out lives asked for before the freeze, or from a seed not committed before it."""


# ── Generation ────────────────────────────────────────────────────────────────


def generate_life(spec: LifeSpec, backend: Backend) -> Generated:
    """``generate.generate``'s attempts, each attempt's worst case held before it starts, Spark Open
    on the backend's provider."""
    model = backend.provider.model
    for attempt in range(generate.MAX_ATTEMPTS):
        if backend.ceiling is not None:
            backend.ceiling.check()  # before the attempt's hold
        with held_attempt(backend, f"{spec.split}:{spec.seed}:{attempt}") as (writer, reader, auditor):
            try:
                built, problems = generate.life(spec.field, spec.families, spec.prevalence,
                                                generate.attempt_seed(spec.seed, attempt), split=spec.split, chat=writer,
                                                model=model)
            except GenerationFailed:
                continue
            verdicts = gates.check(built, problems, reader=reader, auditor=auditor, model=model)
        if gates.passed(verdicts):
            return Generated(built, problems, attempt + 1, verdicts)
    raise GenerationError(f"no {spec.split} life passed the gates in {generate.MAX_ATTEMPTS} attempts")


def spec_ids(spec: LifeSpec) -> frozenset[str]:
    """The ids ``spec``'s life may take, one per attempt (``generate.life`` names a life by its
    split, field and attempt seed)."""
    field_ = re.sub(r"[^a-z0-9]+", "-", spec.field.lower()).strip("-")
    return frozenset(f"{spec.split}-{field_}-{generate.attempt_seed(spec.seed, n)}" for n in range(generate.MAX_ATTEMPTS))


def generate_lives(data: DataDir, specs: Sequence[LifeSpec], backend: Backend, out: Out) -> tuple[str, ...]:
    """Every life of ``specs`` not yet on disk, generated, gated and saved, its problems sealed;
    with a ceiling, what the run has settled per life printed after each."""
    made: list[str] = []
    todo = [spec for spec in specs if not any(data.has_life(life_id) for life_id in spec_ids(spec))]
    if backend.ceiling is not None:
        backend.ceiling.begin("life(s) generated", len(todo))
    for spec in todo:
        made_life = generate_life(spec, backend)
        data.save_life(made_life.life, made_life.problems)
        made.append(made_life.life.id)
        out(f"generated {made_life.life.id} ({made_life.attempts} attempt(s), {len(made_life.problems.items)} planted)")
        if backend.ceiling is not None:
            out(backend.ceiling.life_generated())
    return tuple(made)


def seed_digest(draw_seed: str) -> str:
    return hashlib.sha256(draw_seed.encode("utf-8")).hexdigest()


def check_freeze(entries: Sequence[LedgerEntry], draw_seed: str, count: int) -> None:
    """The held-out lives are drawn only at the freeze: the prereg ledger holds an ``apd_freeze``
    entry naming ``held_out_lives=<count>``, after an ``apd_draw_seed`` entry committing this
    seed's hash."""
    check_frozen(entries, seed_digest(draw_seed), count)


def check_drawn(entries: Sequence[LedgerEntry], drawn: tuple[str, int] | None) -> None:
    """A held-out run's freeze check: the lives on disk were drawn (``DataDir.drawn``) from a seed
    the ledger committed before its freeze, at the count it froze."""
    if drawn is None:
        raise FreezeError("no held-out lives were drawn at the freeze: generate them with --held-out first")
    check_frozen(entries, *drawn)


def check_frozen(entries: Sequence[LedgerEntry], digest: str, count: int) -> None:
    if count not in (HELD_OUT_LIVES, HELD_OUT_LIVES_CUT):
        raise FreezeError(f"the held-out lives number {HELD_OUT_LIVES} or {HELD_OUT_LIVES_CUT}")
    frozen = [entry for entry in entries if entry.kind == FREEZE_KIND]
    if not frozen:
        raise FreezeError("APD isn't frozen: the prereg ledger holds no apd_freeze entry")
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
    """One Glimmer session, stopped on the way out. MIND's leftovers were stopped by the command's
    sweep (``mind.sweep.swept_mind``), under MIND's lock."""
    with colab_vllm.serve(backend.cli, backend.ledger, settings_, timeout_s=timeout_s, tunnels=backend.tunnels,
                          wait=backend.wait, clock=backend.clock, sleep=backend.sleep, keygen=backend.keygen,
                          key_dir=backend.key_dir) as session:
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


# ── Judge and report ──────────────────────────────────────────────────────────


def judges(data: DataDir, backend: Backend) -> tuple[judging.CachedJudge, judging.CachedJudge]:
    rates = backend.rates or {}
    return (
        judging.CachedJudge(judging.MUSE, backend.muse, data.judging / "muse-spark.jsonl", backend.ledger,
                            rates.get(judging.MUSE), ceiling=backend.ceiling),
        judging.CachedJudge(judging.GEMINI, backend.gemini, data.judging / "gemini.jsonl", backend.ledger,
                            rates.get(judging.GEMINI), ceiling=backend.ceiling),
    )


def runs_of(data: DataDir, lives: Sequence[Life], arms: Sequence[ArmName]) -> tuple[ArmRun, ...]:
    found = (data.run(life.id, arm) for life in lives for arm in arms)
    return tuple(run for run in found if run is not None)


def judge_lives(data: DataDir, lives: Sequence[Life], backend: Backend, out: Out) -> judging.Judged:
    """Every run on disk for ``lives``, judged; with the fakes, the audit answered by the fake."""
    if backend.ceiling is not None:
        backend.ceiling.begin("new judge call(s)", None)
    muse, gemini = judges(data, backend)
    runs = runs_of(data, lives, (*HELD_OUT_ARMS, *REPLAY_ARMS))
    judged = judging.judge_runs(data, lives, runs, muse, gemini)
    if judged.pending and backend.claude is not None:
        judging.answer_pending(data, backend.claude)
        judged = judging.judge_runs(data, lives, runs, muse, gemini)
    out(f"judged {len(runs)} run(s): Muse Spark {muse.calls} call(s), Gemini {gemini.calls}; "
        f"{judged.pending} disagreement(s) await Claude's blind audit in {data.audit / 'pending.jsonl'}")
    return judged


def report(judged: judging.Judged,
           interventions: tuple[tuple[InterventionResult, ...], tuple[Skipped, ...]] | None = None) -> str:
    """The APD claim and per-life table, 17b's section when its arms ran, and the intervention
    test's when it has."""
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
    if interventions is not None:
        parts.append(apd_report.intervention_section(*interventions))
    return "\n\n".join(parts)


# ── Kill-resume ───────────────────────────────────────────────────────────────


def kill_lives(data: DataDir, lives: Sequence[Life], session: ColabSession, kill_at: int, backend: Backend,
               out: Out, *, parallel: int = 1, run_s: float = colab_vllm.MIN_SERVE_S) -> tuple[resume.KillResume, ...]:
    """``resume.kill_resume`` on each life, one at a time, its children attached to ``session``; a
    life done before is kept. A life starts only when the session can carry its three runs,
    ``run_s`` each."""
    if parallel != 1:  # the runs are compared token for token: never batched with another life's requests
        raise ValueError("kill-resume runs one at a time: vLLM at temperature 0 isn't batch-invariant")
    say = serial(out)
    with _attached(data, session, backend):
        return tuple(in_parallel([partial(_kill_one, data, life, kill_at, backend, say, session, run_s)
                                  for life in lives], parallel))


def _kill_one(data: DataDir, life: Life, kill_at: int, backend: Backend, out: Out, session: ColabSession,
              run_s: float) -> resume.KillResume:
    found: resume.KillResume | None = data.kill_result(life.id)
    if found is None:
        session.check_room(resume.KILL_RUNS * run_s)
        name = f"kill-{life.id}"
        spend = fresh_journal(data, name, backend, out, life_id=life.id, arm="kill-resume")
        log = resume.kill_log(data.kill, life.id, kill_at)
        if log.exists():  # an earlier attempt's: a kill starts on a fresh log
            os.replace(log, log.with_name(f"{log.name}.discarded-{os.getpid()}"))
        found = resume.kill_resume(SETUP, life.id, kill_at, workdir=data.kill,
                                   on_killed=lambda: _settled(backend, spend))
        data.save_kill_result(found)
    else:
        out(f"kill-resume {life.id} already done")
    out(f"kill-resume {life.id} at {kill_at}: digest {'equal' if found.digest_equal else 'DIFFERS'}, "
        f"conclusions {'agree' if found.conclusions_agree else 'differ'}")
    return found


@contextmanager
def _attached(data: DataDir, session: ColabSession, backend: Backend) -> Iterator[None]:
    before = {name: os.environ.get(name) for name in (SETUP_ENV, KEY_ENV, SPEND_ENV)}
    os.environ[SETUP_ENV] = (f"{data.root}|{session.session}|{session.local_port}|{int(backend.fake)}|"
                             f"{backend.provider.name}")
    os.environ[KEY_ENV] = session.key
    ledger, prices = backend.ledger, backend.prices
    os.environ[SPEND_ENV] = "" if ledger is None or prices is None else (
        f"{ledger.path}|{','.join(str(price) for price in dataclasses.astuple(prices))}")
    try:
        data.kill.mkdir(parents=True, exist_ok=True)
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _settled(backend: Backend, spend: SpendJournal) -> None:
    settle_orphans(backend, spend)


def kill_setup(life_id: str) -> resume.KillSetup:
    """Kill-resume's setup in each process: the life from the data dir, and the run's models on
    the session the parent serves, attached through its tunnel (the fakes' own server with
    ``--fake``). Each process holds its Spark Open, on the parent's provider, a night at a time
    under a key of its own on that provider's pool, and journals every call to the life's journal;
    ``close`` settles its last night."""
    root, session, port, fake, name = os.environ[SETUP_ENV].split("|")
    provider = named(name)
    key = os.environ[KEY_ENV]
    data = DataDir(Path(root))
    transport = httpx.MockTransport(fakes.FakeVllm(key)) if fake == "1" else None
    attached = colab_vllm.attach(session, int(port), key, transport=transport)
    life = data.life(life_id)
    chat: ChatFn = fakes.FakeSparkChat(model=provider.model) if fake == "1" else provider.chat
    ledger, prices = _spend_env(os.environ.get(SPEND_ENV, ""))
    spark = SparkCalls(chat, prices, ledger, f"kill:{life.id}:{os.getpid()}", provider.pool,
                       journal=SpendJournal(data.journal(f"kill-{life.id}")))
    return resume.KillSetup(
        life=life, models=arm_models(attached, spark), core=mind_core(),
        open_log=lambda path: JsonlWorkspaceLog(path, life.id, codec=PLAIN, guard=NO_GUARD), close=spark.close,
    )


def _spend_env(text: str) -> tuple[Ledger | None, Prices | None]:
    if not text:
        return None, None
    path, prices = text.split("|")
    return Ledger(Path(path)), Prices(*(float(price) for price in prices.split(",")))

