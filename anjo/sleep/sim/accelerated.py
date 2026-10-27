"""The accelerated runner (Task 12a): a synthetic life lived in minutes.

For each day *i* a life's simulated day runs first (``sim.day``), with the wall-clock window around
it recorded, and then its night runs on that window and the virtual date ``start + i``
(``night.run``). Several lives run in lockstep: each day is simulated per life, and the day's nights
go through one ``run_night_batch_with`` call, so Task 5's night lock, sweep, holds and newest-reply
path are reused, never re-implemented. Every night's ``summary.json`` is stamped with
``accelerated=True``, the git sha the run used, its label (``development`` or ``final``) and its day
index, so no accelerated result can be read as a real person's.

A window that is empty, or starts before the life's previous window ended, stops the run: a
second night reading the first day's lines again would train on them twice.

    python -m anjo.sleep.sim.accelerated --fake --days 3          # the daily integration gate, offline
    python -m anjo.sleep.sim.accelerated --fake --days 3 --mind   # the same, with MIND's inquiry on (Task 17)
    ANJO_SLEEP_HOST=1 python -m anjo.sleep.sim.accelerated --wiring anjo.sleep.app.night:build \\
        --simulate anjo.sleep.app.sim_transport:simulate --morning anjo.sleep.app.morning:build --user <account id> --persona founder \\
        --days 2 --start 2026-11-03

After each night the morning probe run scores every arm (`probes.morning`). Real runs train on Colab, one VM per life (``ColabLives``), synthetic people only.
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, cast

from anjo.sleep.night.deps import NightDeps, NightOptions
from anjo.sleep.night.nightlog import SUMMARY
from anjo.sleep.night.run import day_window, run_night_batch_with
from anjo.sleep.night.sweep import Sweep
from anjo.sleep.personas.model import Persona, load_person_file
from anjo.sleep.probes.morning import run_morning
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
FAKE_PERSONA: Final = "founder"
FAKE_USER: Final = "fake-life"


class AcceleratedError(RuntimeError):
    """The accelerated run broke a rule and stopped."""


@dataclass(frozen=True)
class Rig:
    """What a run needs from outside: the night's ports for a virtual date, the simulated day, and
    (optionally) the git sha to stamp. ``simulate`` is ``(user_id, person, day_index, seed)``."""

    deps_for: Callable[[date], NightDeps]
    simulate: Callable[[str, Persona, int, int], DayRun]
    sha: str | None = None
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
    timer: Callable[[], float] = time.monotonic


def git_sha(repo: Path = REPO) -> str:
    """The commit this run is on (a local read: no fetch, no network)."""
    done = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=GIT_TIMEOUT_S, check=False)
    sha = done.stdout.strip()
    if done.returncode != 0 or len(sha) != 40:
        raise AcceleratedError("the git sha of this run can't be read")
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


def _stamp(result: NightResult, *, sha: str, label: Label, day_index: int, vm: Mapping[str, Any] | None) -> None:
    """Add the run's labels to the night's summary, atomically."""
    path = result.log_path.parent / SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.update({"accelerated": True, "git_sha": sha, "label": label, "day_index": day_index})
    if vm is not None:
        summary["vm"] = dict(vm)
    pending = path.with_name(f".{SUMMARY}.stamp")
    pending.write_text(json.dumps(summary, sort_keys=True, indent=1, default=str), encoding="utf-8")
    os.replace(pending, path)


def run_accelerated_batch(
    lives: Sequence[tuple[str, Persona]], days: int, start: date, *, rig: Rig, seed: int = DEFAULT_SEED,
    label: str = "development", sweep: Sweep | None = None,
) -> dict[str, list[NightResult]]:
    """Run ``days`` days for every life in lockstep; each night is the life's own virtual date."""
    if days < 1:
        raise ValueError("days is at least 1")
    kind = _check_label(label)
    if len({uid for uid, _ in lives}) != len(lives):
        raise ValueError("each life has its own user id")
    sha = rig.sha if rig.sha is not None else git_sha()
    out: dict[str, list[NightResult]] = {uid: [] for uid, _ in lives}
    last_end: dict[str, datetime] = {}
    for i in range(days):
        night = start + timedelta(days=i)
        runs: dict[str, DayRun] = {}
        for uid, person in lives:
            day = rig.simulate(uid, person, i, seed)
            _check_window(uid, day, last_end.get(uid))
            last_end[uid] = day.window[1]
            if rig.on_day is not None:
                rig.on_day(uid, (day.window[1] - day.window[0]).total_seconds())
            runs[uid] = day
        results = run_night_batch_with(rig.deps_for(night), {uid: night for uid in runs}, accelerated=True,
                                       days=runs, sweep=sweep if i == 0 else None,
                                       as_of={uid: day_window(night)[1] for uid in runs})
        for uid, result in results.items():
            vm = rig.vm_record(uid, night.isoformat()) if rig.vm_record is not None else None
            _stamp(result, sha=sha, label=kind, day_index=i, vm=vm)
            out[uid].append(result)
        if rig.score is not None:
            for uid, result in results.items():
                if result.status != "failed":  # a failed night keeps yesterday's checkpoint: nothing new to score
                    began = rig.timer()
                    rig.score(uid, night, kind, result)
                    if rig.on_scoring is not None:
                        rig.on_scoring(uid, max(0.0, rig.timer() - began))
    return out


def run_accelerated(
    user_id: str, person: Persona, days: int, start: date, *, rig: Rig, seed: int = DEFAULT_SEED,
    label: str = "development", sweep: Sweep | None = None,
) -> list[NightResult]:
    """One life: a batch of one."""
    return run_accelerated_batch([(user_id, person)], days, start, rig=rig, seed=seed, label=label,
                                 sweep=sweep)[user_id]


# ── the command line ─────────────────────────────────────────────────────────


def _vm_record(lives: Any, user_id: str, night: str) -> dict[str, object] | None:
    return next((r for r in lives.records(user_id) if r["night"] == night), None)


def _normal(summary: Mapping[str, Any]) -> dict[str, Any]:
    """A summary without the paths that differ between runs, for the fake gate's digest."""
    return {k: v for k, v in summary.items() if k != "log_path"}


def _report(
    results: Sequence[NightResult], *, fake: bool, days: int, sha: str, scores: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    summaries = [_normal(json.loads((r.log_path.parent / SUMMARY).read_text(encoding="utf-8"))) for r in results]
    digest = hashlib.sha256(json.dumps([summaries, list(scores)], sort_keys=True, default=str).encode()).hexdigest()
    nights = [{"night": s["night"], "status": s["status"], "failed_at": s["failed_at"],
               "checkpoint": (s["checkpoint"] or {}).get("id") if isinstance(s["checkpoint"], dict) else None,
               "cost_usd": s["cost_usd"]} for s in summaries]
    return {"accelerated": True, "fake": fake, "days": days, "git_sha": sha, "nights": nights, "digest": digest,
            "scored_nights": len(scores), "scores": [{"night": sc["night"], "arms": sc["arms"], "pairs": sc["pairs"],
                                                      "unavailable": sc["unavailable"]} for sc in scores]}


def _fake(days: int, start: date, seed: int, mind: bool = False) -> int:
    """The whole loop on fakes: no model, no GPU, no network, no data dir. ``mind`` turns on MIND's
    inquiry on a fake thinker."""
    rig_module = importlib.import_module("anjo.sleep.tests.accelerated_rig")  # test support, loaded only here
    person = load_person_file(PERSONAS / f"{FAKE_PERSONA}.json").visible
    with tempfile.TemporaryDirectory(prefix="anjo-sleep-fake-") as root:
        fake = rig_module.FakeAccelerated(Path(root), mind_from=start if mind else None)
        results = run_accelerated(FAKE_USER, person, days, start, rig=fake.rig(sha="0" * 40), seed=seed)
        scores = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fake.scores_dir(FAKE_USER).glob("night-*.json"))]
        print(json.dumps(_report(results, fake=True, days=days, sha="0" * 40, scores=scores), indent=1, sort_keys=True))
    return 0 if all(r.status != "failed" for r in results) else 1


def _named(path: str) -> Any:
    module, _, name = path.partition(":")
    if not module or not name:
        raise SystemExit(f"{path} is not <module>:<function>")
    return getattr(importlib.import_module(module), name)


def _real(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from anjo.sleep.budget import default_ledger
    from anjo.sleep.checkpoints import DATA_DIR_ENV
    from anjo.sleep.night import __main__ as night_command
    from anjo.sleep.night.sweep import NightBusy, swept_night
    from anjo.sleep.serving.backend import phase
    from anjo.sleep.serving.config import REVISION_ENV
    from anjo.sleep.train.runners.colab import ColabCli, ColabRunner, stop_leftover_sessions
    from anjo.sleep.train.trainer import LoraTrainer

    if not (args.wiring and args.simulate and args.morning and args.users and args.personas and args.start):
        parser.error("a real run needs --wiring, --simulate, --morning, --start and a --user with its --persona (or --fake)")
    if len(args.personas) != len(args.users):
        parser.error("each --user has one --persona: accelerated runs are synthetic people only")
    data_dir = Path(os.environ.get(DATA_DIR_ENV, "").strip() or parser.error(f"{DATA_DIR_ENV} is not set"))
    personas = dict(zip(args.users, args.personas, strict=True))
    people = [(uid, load_person_file(PERSONAS / f"{name}.json").visible) for uid, name in personas.items()]
    cli, ledger = ColabCli(), default_ledger()
    runner = ColabRunner(cli=cli, ledger=ledger, is_synthetic=lambda uid: uid in personas, allow_real=False)
    life_s = LIFE_NIGHT_TIMEOUT_S * (LIFE_NIGHTS_HELD + args.days)
    wiring, simulate, morning = _named(args.wiring), _named(args.simulate), _named(args.morning)

    def score(uid: str, night: date, label: str, result: NightResult) -> object:
        """The morning probe run on the night's own checkpoint (the serving one if it promoted nothing)."""
        ports = morning(uid, personas[uid], night)
        return run_morning(uid, night, result.checkpoint, morning=ports, label=label,
                           promoted=result.status == "promoted")

    try:
        with swept_night(ledger, lambda: stop_leftover_sessions(cli)) as swept, \
                runner.lives(life_timeout_s=life_s) as lives:
            trainer = LoraTrainer(lives, root=data_dir / "sleep" / "training", publish=night_command._publish,
                                  revision=os.environ.get(REVISION_ENV) or None)
            rig = Rig(deps_for=lambda night: wiring(trainer, NightOptions(night=night, personas=personas,
                                                                          mind_inquiry=args.mind)),
                      simulate=lambda uid, person, i, seed: simulate(uid, personas[uid], i, seed),
                      on_day=lives.note_day, on_scoring=lives.note_scoring, vm_record=lambda uid, night: _vm_record(lives, uid, night),
                      score=score)
            with phase():  # each day, night and morning follows the last: Glimmer sleeps once, at the end
                results = run_accelerated_batch(people, args.days, args.start, rig=rig, seed=args.seed,
                                                label=args.label, sweep=swept)
    except NightBusy as busy:
        raise SystemExit(str(busy)) from None
    sha = git_sha()
    print(json.dumps({uid: _report(done, fake=False, days=args.days, sha=sha) for uid, done in results.items()},
                     indent=1, sort_keys=True))
    return 0 if all(r.status != "failed" for done in results.values() for r in done) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic lives day by day, a night after each.")
    parser.add_argument("--fake", action="store_true", help="the whole loop on fakes, offline (the daily gate)")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 11, 3))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--label", choices=LABELS, default="development")
    parser.add_argument("--wiring", help="the app's NightDeps builder, <module>:<function>")
    parser.add_argument("--simulate", help="the app's day runner, <module>:<function>")
    parser.add_argument("--morning", help="the app's morning-run builder (user, persona, night), <module>:<function>")
    parser.add_argument("--user", action="append", default=[], dest="users")
    parser.add_argument("--persona", action="append", default=[], dest="personas")
    parser.add_argument("--mind", action="store_true", help="MIND's inquiry in each night (Task 17)")
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days is at least 1")
    if args.fake:
        return _fake(args.days, args.start, args.seed, mind=args.mind)
    return _real(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
