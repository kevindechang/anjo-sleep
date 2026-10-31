"""The accelerated runner (Task 12a): a synthetic life lived in minutes.

For each day *i* a life's simulated day runs first (``sim.day``), with the wall-clock window around
it recorded, and then its night runs on that window and the virtual date ``start + i``
(``night.run``). Several lives run in lockstep: each day is simulated per life, and the day's nights
go through one ``run_night_batch_with`` call, so Task 5's night lock, sweep, holds and newest-reply
path are reused, never re-implemented. Every night's ``summary.json`` is stamped with
``accelerated=True``, the git sha the run used, its label (``development`` or ``final``) and its day
index, so no accelerated result can be read as a real person's. A final-label run on real code must
also name the frozen sha (``--require-sha``): the run refuses any other commit before a day is
simulated or a night runs. An offline fake run is not on git, so it ignores ``--require-sha``.

A window that is empty, or starts before the life's previous window ended, stops the run: a
second night reading the first day's lines again would train on them twice.

**A life is dated in the past.** Its last virtual night must end by the wall time the run starts
(``check_dates``), and ``--start`` defaults to the latest such date, today minus ``--days`` on the
run's clock. Each record is stamped at its window's wall end or the night's clock, whichever is
earlier, which on a past date is always the virtual time: chat on the wall clock sees everything
that exists, and the night's masks, its prediction windows and the person's last message all read
one clock. A start after that is refused before any day is simulated or any night runs.

    python -m anjo.sleep.sim.accelerated --fake --days 3          # the daily integration gate, offline
    python -m anjo.sleep.sim.accelerated --fake --days 3 --mind   # the same, with MIND's inquiry on (Task 17)
    ANJO_SLEEP_HOST=1 python -m anjo.sleep.sim.accelerated --wiring anjo.sleep.app.night:build \\
        --simulate anjo.sleep.app.sim_transport:simulate --morning anjo.sleep.app.morning:build --user <account id> --persona film_founder \\
        --days 2                                                   # starts two days ago; --start 2026-10-01 dates it earlier

``--alone-from N`` stops person messages from 1-based day N, retaining a positive empty day
window. Quiet nights are stamped ``alone``. With MIND on they request deep sleep so the
pressure/empty-evidence nap guards do not defer inquiry; the monthly cap still wins. Training
still requires new evidence. An unfocused inquiry pass runs if the chooser did not inquire. Mornings retain their normal failed/nap guards and sealed probes.

Each night naps or sleeps deep by the same decision as every night (``night/sleep_pressure.py``,
under the wiring's policy), and the report says which. After each night that slept deep the morning
probe run scores every arm (`probes.morning`), or the subset ``ANJO_SLEEP_MORNING_SUBSET`` names
(``daily-v1``; any other value stops the run before it starts); a nap runs no GPU morning. Each morning is logged in
its night's log as the daily run logs it (``night/daily.py``), ``ok`` or ``failed`` with its reason,
and the night's story is told again with its wake. A judge that fails one morning never stops the
run: the next night goes on, the report says which morning failed, and the command exits 3
(``UNSCORED_EXIT``) when every night ran but a morning failed. The morning's own refusal
(``MorningError``: a rerun, or a checkpoint that moved since the night) still stops it. Real runs train synthetic
people only, on the runner ``ANJO_SLEEP_TRAIN_RUNNER`` names (``train/runners/choice.py``): Modal by
default, one call a night, or Colab, one VM per life (``ColabLives``).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Literal, cast

from anjo.sleep.night.daily import UNSCORED_EXIT, log_failed_morning, morning_outcome, morning_receipt
from anjo.sleep.night.deps import NightDeps, NightOptions
from anjo.sleep.night.dreams import Terms
from anjo.sleep.night.nightlog import SUMMARY, NightLog
from anjo.sleep.night.run import day_window, run_night_batch_with
from anjo.sleep.night.sweep import Sweep
from anjo.sleep.personas.model import Persona, load_person_file
from anjo.sleep.practice.full import MAX_SCENES, MIN_SCENES
from anjo.sleep.probes.morning import (
    MORNING_PROBES,
    Morning,
    MorningError,
    ProbeSubset,
    morning_subset,
    probe_set,
    run_morning,
)
from anjo.sleep.sim.day import DayRun
from anjo.sleep.types import NightResult

Label = Literal["development", "final"]
LABELS: Final = ("development", "final")
DEFAULT_SEED: Final = 20261028
PERSONAS: Final = Path(__file__).parents[1] / "personas"
REPO: Final = Path(__file__).parents[3]
GIT_TIMEOUT_S: Final = 10.0
LIFE_NIGHT_TIMEOUT_S: Final = 2.0 * 3600.0  # one night's Colab job, as the night command's trainer allows
LIFE_NIGHTS_HELD: Final = 3  # a life's VM hold covers this many nights at the job timeout, plus setup
FAKE_PERSONA: Final = "film_founder"
FAKE_USER: Final = "fake-life"


class AcceleratedError(RuntimeError):
    """The accelerated run broke a rule and stopped."""


@dataclass(frozen=True)
class Rig:
    """What a run needs from outside: the night's ports for a virtual date, the simulated day, and
    (optionally) the git sha to stamp. ``simulate`` is ``(user_id, person, day_index, seed)``.

    ``sha`` given marks a fake or offline run: its code is not read from git, so ``require_sha`` is
    ignored. Left ``None``, the run reads this commit's sha, and a final one must be on
    ``require_sha`` before any day is simulated or any night runs."""

    deps_for: Callable[[date], NightDeps]
    simulate: Callable[[str, Persona, int, int], DayRun]
    sha: str | None = None
    require_sha: str | None = None
    # How long a life's simulated day took (seconds), told before its night: the Colab runner stops a
    # VM that would idle through a long day. ``vm_record(user_id, night)`` is that night's content-free
    # VM record (kept or stopped, idle seconds), stamped into the summary.
    on_day: Callable[[str, float], None] | None = None
    vm_record: Callable[[str, str], Mapping[str, Any] | None] | None = None
    # The morning probe run after a night: ``score(user_id, night, label)`` scores every arm on the
    # checkpoint serving after it (``probes.morning``) and writes that night's score file.
    # ``score(user_id, night, label, result)`` gets the night's own result, whose checkpoint it scores.
    # ``on_scoring`` is told how long each scoring run took (seconds), as ``on_day`` is for the day.
    score: Callable[[str, date, str, NightResult], object] | None = None
    on_scoring: Callable[[str, float], None] | None = None
    subset: ProbeSubset | None = None  # the set ``score`` asks, recorded in each night's summary and a failed morning
    timer: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)  # the wall clock, read as the run starts


def latest_start(now: datetime, days: int) -> date:
    """The latest start whose ``days`` nights all end by ``now``: today (UTC) minus ``days``."""
    return now.astimezone(timezone.utc).date() - timedelta(days=days)


def check_dates(start: date, days: int, now: datetime) -> None:
    """Refuse a life whose last virtual night ends after ``now``: dated in the future, its records
    would sit on the wall clock while its nights read the virtual one."""
    last_end = day_window(start + timedelta(days=days - 1))[1]
    if last_end > now:
        raise AcceleratedError(
            f"the life's last night ends {last_end.isoformat()}, after the run starts ({now.isoformat()}): "
            f"accelerated lives are dated in the past, so start on or before {latest_start(now, days).isoformat()}")


def git_sha(repo: Path = REPO) -> str:
    """The commit this run is on (a local read: no fetch, no network)."""
    done = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=GIT_TIMEOUT_S, check=False)
    sha = done.stdout.strip()
    if done.returncode != 0 or len(sha) != 40:
        raise AcceleratedError("the git sha of this run can't be read")
    return sha


def is_sha(text: str) -> bool:
    """Whether ``text`` is a full 40-character hex git sha (either case)."""
    try:
        return len(text) == 40 and int(text, 16) >= 0
    except ValueError:
        return False


def _resolve_sha(supplied: str | None, require_sha: str | None, label: Label) -> str:
    """The sha the run stamps. A fake or offline run (``supplied`` given) is not on git, so any
    ``require_sha`` is ignored. Otherwise a final run must name the frozen sha (``require_sha``)
    before this commit's sha is even read, and be on it: refused before any day is simulated or any
    night runs."""
    if supplied is not None:
        return supplied
    if require_sha is None:
        if label == "final":
            raise AcceleratedError(
                "a final run must name the frozen code's sha with --require-sha (require_sha)")
        return git_sha()
    sha = git_sha()
    wanted = require_sha.lower()
    if sha != wanted:
        raise AcceleratedError(
            f"this run is on {sha}, not the frozen {wanted}: a final run must be on the frozen code, "
            f"a separate checkout of the sleep-freeze tag")
    return sha


def _check_label(label: str) -> Label:
    if label not in LABELS:
        raise ValueError(f"label is one of {', '.join(LABELS)}, got {label!r}")
    return cast(Label, label)


def _check_window(user_id: str, day: DayRun, last_end: datetime | None) -> None:
    start, end = day.window
    if end <= start:
        raise AcceleratedError(f"{user_id}: the simulated day's window is empty")
    if last_end is not None and start < last_end:
        raise AcceleratedError(f"{user_id}: the day's window starts before the last day's window ended")


def _stamp(result: NightResult, *, sha: str, label: Label, day_index: int, vm: Mapping[str, Any] | None,
           probes: str | None = None, alone: bool = False) -> None:
    """Add the run's labels to the night's summary, atomically, and the probe set its morning asks."""
    path = result.log_path.parent / SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update({"accelerated": True, "git_sha": sha, "label": label, "day_index": day_index, "alone": alone})
    if probes is not None:
        summary[MORNING_PROBES] = probes
    if vm is not None:
        summary["vm"] = dict(vm)
    pending = path.with_name(f".{SUMMARY}.stamp")
    pending.write_text(json.dumps(summary, sort_keys=True, indent=1, default=str), encoding="utf-8")
    os.replace(pending, path)


def _morning(rig: Rig, uid: str, night: date, label: Label, result: NightResult) -> None:
    """The morning probe run after a night, logged in its night's log, and the night's story told
    again with its wake. A failure is that morning's alone; the morning's refusal stops the run."""
    assert rig.score is not None
    log, began, failure = NightLog(result.log_path.parent), rig.timer(), None
    try:
        done = rig.score(uid, night, label, result)
    except MorningError:
        raise  # a rerun, or a checkpoint that moved since the night: the run's own rule broke
    except Exception as error:  # a judge that can't be read, timed out or is down: this morning only
        failure = error
    finally:
        seconds = max(0.0, rig.timer() - began)
        if rig.on_scoring is not None:
            rig.on_scoring(uid, seconds)
    if failure is not None:
        log_failed_morning(log, night, failure, seconds, subset=rig.subset)
    else:
        log.step("morning", "ok", uses="new", system=None, **morning_receipt(done, seconds))
    log.tell(night, result.status)


def run_accelerated_batch(
    lives: Sequence[tuple[str, Persona]], days: int, start: date, *, rig: Rig, seed: int = DEFAULT_SEED,
    label: str = "development", sweep: Sweep | None = None, require_sha: str | None = None,
    alone_from: int | None = None,
) -> dict[str, list[NightResult]]:
    """Run ``days`` days for every life in lockstep; each night is the life's own virtual date, in
    the past (``check_dates``, before anything runs). A final run on real code must be on the frozen
    ``require_sha`` (or the rig's), refused first of all, before ``check_dates``, any day or any night."""
    if days < 1:
        raise ValueError("days is at least 1")
    if alone_from is not None and not 1 <= alone_from <= days:
        raise ValueError("--alone-from must be between 1 and --days (inclusive)")
    kind = _check_label(label)
    if len({uid for uid, _ in lives}) != len(lives):
        raise ValueError("each life has its own user id")
    sha = _resolve_sha(rig.sha, require_sha if require_sha is not None else rig.require_sha, kind)
    check_dates(start, days, rig.now())
    out: dict[str, list[NightResult]] = {uid: [] for uid, _ in lives}
    last_end: dict[str, datetime] = {}
    for i in range(days):
        night = start + timedelta(days=i)
        alone = alone_from is not None and i + 1 >= alone_from
        runs: dict[str, DayRun] = {}
        for uid, person in lives:
            if alone:
                # A positive, disjoint wall window without waking a transport or sending a turn.
                began = max(rig.now(), last_end.get(uid, datetime.min.replace(tzinfo=timezone.utc)))
                day = DayRun((began, max(rig.now(), began + timedelta(microseconds=1))), 0, 0, ())
            else:
                day = rig.simulate(uid, person, i, seed)
            _check_window(uid, day, last_end.get(uid))
            last_end[uid] = day.window[1]
            if rig.on_day is not None:
                rig.on_day(uid, (day.window[1] - day.window[0]).total_seconds())
            runs[uid] = day
        deps = rig.deps_for(night)
        if alone and deps.settings.mind_inquiry:
            # Quiet inquiry needs a waking night; retain the month's cap and the training threshold.
            deps = replace(deps, settings=replace(deps.settings, force_deep=True, quiet_inquiry=True))
        results = run_night_batch_with(deps, {uid: night for uid in runs}, accelerated=True,
                                       days=runs, sweep=sweep if i == 0 else None,
                                       as_of={uid: day_window(night)[1] for uid in runs})
        for uid, result in results.items():
            vm = rig.vm_record(uid, night.isoformat()) if rig.vm_record is not None else None
            _stamp(result, sha=sha, label=kind, day_index=i, vm=vm,
                   probes=None if rig.score is None else probe_set(rig.subset), alone=alone)
            out[uid].append(result)
        if rig.score is not None:
            for uid, result in results.items():
                if result.status not in ("failed", "napped"):  # nothing new to score, and a nap runs no GPU morning
                    _morning(rig, uid, night, kind, result)
    return out


def run_accelerated(
    user_id: str, person: Persona, days: int, start: date, *, rig: Rig, seed: int = DEFAULT_SEED,
    label: str = "development", sweep: Sweep | None = None, require_sha: str | None = None,
    alone_from: int | None = None,
) -> list[NightResult]:
    """One life: a batch of one."""
    return run_accelerated_batch([(user_id, person)], days, start, rig=rig, seed=seed, label=label,
                                 sweep=sweep, require_sha=require_sha, alone_from=alone_from)[user_id]


# ── the command line ─────────────────────────────────────────────────────────


def _vm_record(lives: Any, user_id: str, night: str) -> dict[str, object] | None:
    return next((r for r in lives.records(user_id) if r["night"] == night), None)


@contextmanager
def _per_life(runner: Any, life_s: float) -> Iterator[tuple[Any, dict[str, Any]]]:
    """The runner the lives train on, and the rig's VM hooks. Colab keeps one VM per life
    (``ColabLives``), told how long each day and scoring run took; Modal runs one call a night, so
    there is no VM to keep between nights and nothing to tell it."""
    from anjo.sleep.train.runners.colab import ColabRunner

    if not isinstance(runner, ColabRunner):
        yield runner, {}
        return
    with runner.lives(life_timeout_s=life_s) as lives:
        yield lives, {"on_day": lives.note_day, "on_scoring": lives.note_scoring,
                      "vm_record": lambda uid, night: _vm_record(lives, uid, night)}


def _normal(summary: Mapping[str, Any]) -> dict[str, Any]:
    """A summary without the paths that differ between runs, for the fake gate's digest."""
    return {k: v for k, v in summary.items() if k != "log_path"}


def _trained(result: NightResult) -> dict[str, Any]:
    """The objective the night trained under and its pair count, from its log's train step."""
    steps = NightLog(result.log_path.parent).steps()
    receipt = next((s.get("receipt") or {} for s in steps if s["step"] == "train"), {})
    return {"objective": receipt.get("objective"), "pairs": receipt.get("pairs")}


def _report(
    results: Sequence[NightResult], *, fake: bool, days: int, sha: str, scores: Sequence[Mapping[str, Any]] = (),
    practice_scenes: int = 16,
) -> dict[str, Any]:
    summaries = [_normal(json.loads((r.log_path.parent / SUMMARY).read_text(encoding="utf-8"))) for r in results]
    timeless = [{k: v for k, v in sc.items() if k != "seconds"} for sc in scores]  # a morning's wall time varies
    digest = hashlib.sha256(json.dumps([summaries, timeless], sort_keys=True, default=str).encode()).hexdigest()
    nights = [{"night": s["night"], "alone": s.get("alone", False), "status": s["status"], "failed_at": s["failed_at"],
               "checkpoint": (s["checkpoint"] or {}).get("id") if isinstance(s["checkpoint"], dict) else None,
               "cost_usd": s["cost_usd"], "sleep": (s.get("sleep") or {}).get("kind"), "morning": morning_outcome(r),
               **_trained(r)}
              for s, r in zip(summaries, results, strict=True)]
    return {"accelerated": True, "fake": fake, "days": days, "git_sha": sha, "nights": nights, "digest": digest,
            "practice_scenes": practice_scenes,
            "scored_nights": len(scores), "scores": [{"night": sc["night"], "arms": sc["arms"], "pairs": sc["pairs"],
                                                      "unavailable": sc["unavailable"]} for sc in scores]}


def _exit(results: Sequence[NightResult]) -> int:
    """1 when a night failed; ``UNSCORED_EXIT`` (3) when every night ran but a morning failed; else 0."""
    if any(r.status == "failed" for r in results):
        return 1
    return UNSCORED_EXIT if any(morning_outcome(r).startswith("failed") for r in results) else 0


def _dated(parser: argparse.ArgumentParser, start: date | None, days: int, now: datetime) -> date:
    """``start``, or the latest past one on the run's clock; a future life is refused here."""
    dated = start or latest_start(now, days)
    try:
        check_dates(dated, days, now)
    except AcceleratedError as late:
        parser.error(str(late))
    return dated


def _fake(days: int, start: date | None, seed: int, mind: bool, label: str,
          parser: argparse.ArgumentParser, alone_from: int | None = None, practice_scenes: int = 16) -> int:
    """The whole loop on fakes: no model, no GPU, no network, no data dir, on the rig's fake clock.
    ``mind`` turns on MIND's inquiry on a fake thinker. A fake run is not on git, so a final one
    (``label``) needs no ``--require-sha`` and any given one is ignored."""
    rig_module = importlib.import_module("anjo.sleep.tests.accelerated_rig")  # test support, loaded only here
    person = load_person_file(PERSONAS / f"{FAKE_PERSONA}.json").visible
    clock = rig_module.FakeClock()
    start = _dated(parser, start, days, clock.now)
    with tempfile.TemporaryDirectory(prefix="anjo-sleep-fake-") as root:
        fake = rig_module.FakeAccelerated(Path(root), clock=clock, mind_from=start if mind else None)
        rig = fake.rig(sha="0" * 40)
        deps_for = rig.deps_for

        def configured(night: date) -> NightDeps:
            deps = deps_for(night)
            return replace(deps, settings=replace(deps.settings, practice_scenes=practice_scenes))

        results = run_accelerated(FAKE_USER, person, days, start, rig=replace(rig, deps_for=configured),
                                  seed=seed, label=label, alone_from=alone_from)
        scores = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fake.scores_dir(FAKE_USER).glob("night-*.json"))]
        print(json.dumps(_report(results, fake=True, days=days, sha="0" * 40, scores=scores,
                                 practice_scenes=practice_scenes), indent=1, sort_keys=True))
        return _exit(results)  # while the run's folder still holds its logs


def _named(path: str) -> Any:
    module, _, name = path.partition(":")
    if not module or not name:
        raise SystemExit(f"{path} is not <module>:<function>")
    return getattr(importlib.import_module(module), name)


def match_by_night(path: Path, start: date, days: int) -> tuple[tuple[date, float], ...]:
    """The live comparison's matched control (``--match-file``): a JSON object of each night's date
    and its training budget in minutes, one for every night of the run, checked before anything runs."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        budgets = {date.fromisoformat(night): minutes for night, minutes in raw.items()}
    except (OSError, ValueError, AttributeError) as error:
        raise AcceleratedError(f"the match file isn't each night's date and minutes ({error})") from None
    nights = [start + timedelta(days=i) for i in range(days)]
    missing = [night.isoformat() for night in nights if night not in budgets]
    if missing or not all(isinstance(m, int | float) and not isinstance(m, bool) for m in budgets.values()):
        raise AcceleratedError(f"the match file has no budget, in minutes, for {', '.join(missing) or 'every night'}")
    return tuple((night, float(budgets[night])) for night in nights)


def morning_scorer(build: Callable[[str, str, date], Morning], personas: Mapping[str, str],
                   subset: ProbeSubset | None) -> Callable[[str, date, str, NightResult], object]:
    def score(uid: str, night: date, label: str, result: NightResult) -> object:
        """The morning probe run on the night's own checkpoint (the serving one if it promoted nothing), on
        ``subset`` (``ANJO_SLEEP_MORNING_SUBSET``; None: the full set)."""
        return run_morning(uid, night, result.checkpoint, morning=build(uid, personas[uid], night), label=label,
                           promoted=result.status == "promoted", subset=subset)
    return score


def _with_terms(deps: NightDeps, terms: Terms) -> NightDeps:
    """The wiring's ports with sft_dpo_seed's terms; with the switches' explicit off values, untouched
    (today's night)."""
    return replace(deps, terms=terms) if terms != Terms() else deps


def _real(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from anjo.sleep.budget import default_ledger
    from anjo.sleep.checkpoints import DATA_DIR_ENV
    from anjo.sleep.night import __main__ as night_command
    from anjo.sleep.night.sweep import NightBusy, swept_night
    from anjo.sleep.serving.backend import phase
    from anjo.sleep.serving.config import REVISION_ENV
    from anjo.sleep.train import objective
    from anjo.sleep.train.lora import LoraCfgError
    from anjo.sleep.train.trainer import LoraTrainer

    if not (args.wiring and args.simulate and args.morning and args.users and args.personas):
        parser.error("a real run needs --wiring, --simulate, --morning and a --user with its --persona (or --fake)")
    if len(args.personas) != len(args.users):
        parser.error("each --user has one --persona: accelerated runs are synthetic people only")
    try:
        subset = morning_subset(os.environ)  # an unknown ANJO_SLEEP_MORNING_SUBSET stops here, before any call
    except ValueError as error:
        parser.error(str(error))
    if subset is not None and args.label == "final":
        parser.error("a final run scores the full morning: unset ANJO_SLEEP_MORNING_SUBSET")
    if args.label == "final" and args.require_sha is None:  # a fake run never reaches here
        parser.error("a final run must name the frozen code's sha: --require-sha <40 hex chars>")
    try:  # the sha first of all: wrong code is refused before the sweep, any VM or any call
        _resolve_sha(None, args.require_sha, _check_label(args.label))
    except AcceleratedError as refused:
        raise SystemExit(str(refused)) from None
    start = _dated(parser, args.start, args.days, datetime.now(timezone.utc))  # before any sweep, VM or call
    cfg = night_command.night_cfg()  # likewise: an unknown ANJO_SLEEP_OBJECTIVE stops here
    if args.match_file is not None:
        try:
            cfg = replace(cfg, match_by_night=match_by_night(args.match_file, start, args.days))
        except (AcceleratedError, LoraCfgError) as error:
            parser.error(str(error))
    terms = objective.night_terms(os.environ)  # sft_dpo_seed's extra rows, as the night command sets them
    data_dir = Path(os.environ.get(DATA_DIR_ENV, "").strip() or parser.error(f"{DATA_DIR_ENV} is not set"))
    personas = dict(zip(args.users, args.personas, strict=True))
    people = [(uid, load_person_file(PERSONAS / f"{name}.json").visible) for uid, name in personas.items()]
    ledger = default_ledger()
    chosen = night_command.chosen_runner(data_dir, ledger, personas)  # synthetic people only: no real one named
    life_s = LIFE_NIGHT_TIMEOUT_S * (LIFE_NIGHTS_HELD + args.days)
    wiring, simulate, morning = _named(args.wiring), _named(args.simulate), _named(args.morning)

    score = morning_scorer(morning, personas, subset)
    try:
        with swept_night(ledger, chosen.sweep) as swept, _per_life(chosen.runner, life_s) as (runner, hooks):
            trainer = LoraTrainer(runner, root=data_dir / "sleep" / "training",
                                  publish=night_command._publisher(chosen.name),
                                  revision=os.environ.get(REVISION_ENV) or None, cfg=cfg)
            rig = Rig(deps_for=lambda night: _with_terms(wiring(trainer, NightOptions(
                night=night, personas=personas, mind_inquiry=args.mind, practice_scenes=args.practice_scenes)), terms),
                      simulate=lambda uid, person, i, seed: simulate(uid, personas[uid], i, seed),
                      score=score, subset=subset, require_sha=args.require_sha, **hooks)
            with phase():  # each day, night and morning follows the last: Glimmer sleeps once, at the end
                results = run_accelerated_batch(people, args.days, start, rig=rig, seed=args.seed,
                                                label=args.label, sweep=swept, alone_from=args.alone_from)
    except NightBusy as busy:
        raise SystemExit(str(busy)) from None
    sha = git_sha()
    print(json.dumps({uid: _report(done, fake=False, days=args.days, sha=sha, practice_scenes=args.practice_scenes)
                      for uid, done in results.items()},
                     indent=1, sort_keys=True))
    return _exit([r for done in results.values() for r in done])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic lives day by day, a night after each.")
    parser.add_argument("--fake", action="store_true", help="the whole loop on fakes, offline (the daily gate)")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="the first virtual date, in the past (default: today minus --days)")
    parser.add_argument("--alone-from", type=int, help="first day without person messages (1-based, through --days)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--label", choices=LABELS, default="development")
    parser.add_argument("--wiring", help="the app's NightDeps builder, <module>:<function>")
    parser.add_argument("--simulate", help="the app's day runner, <module>:<function>")
    parser.add_argument("--morning", help="the app's morning-run builder (user, persona, night), <module>:<function>")
    parser.add_argument("--user", action="append", default=[], dest="users")
    parser.add_argument("--persona", action="append", default=[], dest="personas")
    parser.add_argument("--mind", action="store_true", help="MIND's inquiry in each night (Task 17)")
    parser.add_argument("--practice-scenes", type=int, default=16, help="full practice's scenes, 16 to 20")
    parser.add_argument("--match-file", type=Path, default=None,
                        help="the live comparison's matched control only: JSON of each night's date and budget minutes")
    parser.add_argument("--require-sha", default=None,
                        help="the frozen code's 40-character git sha a real final run must be on")
    args = parser.parse_args(argv)
    if not MIN_SCENES <= args.practice_scenes <= MAX_SCENES:
        parser.error(f"--practice-scenes must be between {MIN_SCENES} and {MAX_SCENES} (inclusive)")
    if args.days < 1:
        parser.error("--days is at least 1")
    if args.alone_from is not None and not 1 <= args.alone_from <= args.days:
        parser.error("--alone-from must be between 1 and --days (inclusive)")
    if args.fake:  # a fake run is not on git: it ignores --require-sha, well-formed or not
        return _fake(args.days, args.start, args.seed, args.mind, args.label, parser, args.alone_from,
                     practice_scenes=args.practice_scenes)
    if args.require_sha is not None and not is_sha(args.require_sha):
        parser.error("--require-sha is a full 40-character hex git sha")
    return _real(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
