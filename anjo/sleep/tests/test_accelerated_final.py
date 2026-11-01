# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A final run's schedule (Task 12b, ``sim.final_batch``): the lives run as one batch, each is scored
once, on its last night, after every life's last night (so every 3x donor has trained its last); a
failed night stops the run before any morning, and ``resume`` continues it without scoring a life
twice. Fakes only: no model, judge, GPU or network."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.night.daily import UNSCORED_EXIT, morning_outcome
from anjo.sleep.night.nightlog import STORY
from anjo.sleep.night.story import NotRun, story_from
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.probes.morning import MorningError, run_morning
from anjo.sleep.probes.registered_judge import require_registered
from anjo.sleep.sim import accelerated, final_batch
from anjo.sleep.sim.accelerated import FinalStopped, Rig, run_accelerated_batch
from anjo.sleep.sim.day import DayRun
from anjo.sleep.tests.accelerated_rig import FakeAccelerated
from anjo.sleep.tests.night_fakes import steps
from anjo.sleep.types import NightResult

START = date(2026, 10, 20)  # before the rig's wall clock (2026-10-28): an accelerated life is dated in the past
LAST = START + timedelta(days=1)
PERSONAS = Path(__file__).parents[1] / "personas"
FOUNDER = load_person_file(PERSONAS / "founder.json").visible
DONOR = load_person_file(PERSONAS / "founder_donor.json").visible
LIVES = [("u1", FOUNDER), ("u2", DONOR)]
DONORS = {"u1": "u2", "u2": "u1"}  # the pair: each the other's 3x donor


class Scored:
    """The fake morning with the pair's donors, recording what had run when each morning began."""

    def __init__(self, fake: FakeAccelerated, fail: dict[str, Exception] | None = None) -> None:
        self.fake, self.fail = fake, dict(fail or {})
        self.calls: list[dict[str, Any]] = []

    def __call__(self, uid: str, night: date, label: str, result: NightResult) -> object:
        donor = self.fake.world.registry.last_promoted(DONORS[uid])
        self.calls.append({"uid": uid, "night": night, "label": label, "donor_night": donor and donor.night,
                           "extracted": {(e[1], e[2]) for e in self.fake.world.events if e[0] == "extract"}})
        if uid in self.fail:
            raise self.fail.pop(uid)
        base = self.fake.morning(uid)
        morning = replace(base, arms=replace(base.arms, donors=DONORS))
        return run_morning(uid, night, result.checkpoint, morning=morning, label=label,
                           promoted=result.status == "promoted")


def _rig(fake: FakeAccelerated, scored: Scored, simulated: list[tuple[str, int]] | None = None) -> Rig:
    rig = fake.rig()
    seen = simulated if simulated is not None else []

    def simulate(uid: str, person: Any, i: int, seed: int) -> DayRun:
        seen.append((uid, i))
        return fake.simulate(uid, person, i, seed)

    return replace(rig, score=scored, simulate=simulate, logs=fake.world.logs, scores=fake.scores_dir)


def _scores(fake: FakeAccelerated, uid: str) -> list[dict[str, Any]]:
    folder = fake.scores_dir(uid)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("night-*.json"))] if folder.is_dir() else []


def _morning_steps(result: NightResult) -> list[dict[str, Any]]:
    return [s for s in steps(result.log_path) if s["step"] == "morning"]


# ── one batch, each life scored once, after every life's last night ────────────


def test_a_final_batch_scores_each_life_once_after_every_lifes_last_night(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    scored = Scored(fake)
    out = run_accelerated_batch(LIVES, 2, START, rig=_rig(fake, scored), label="final")
    assert [r.status for v in out.values() for r in v] == ["promoted"] * 4
    # no morning until both lives' second nights ran; then exactly one per life, on its last night
    assert [(c["uid"], c["night"], c["label"]) for c in scored.calls] == [("u1", LAST, "final"), ("u2", LAST, "final")]
    for call in scored.calls:
        assert {("u1", LAST), ("u2", LAST)} <= call["extracted"]
        assert call["donor_night"] == LAST  # the donor's last night's adapter serves: 3x reads it
    for uid in DONORS:
        (record,) = _scores(fake, uid)
        assert (record["label"], record["night"], record["checkpoint_basis"]) == ("final", LAST.isoformat(), "promoted")
        assert record["unavailable"] == {} and "3x" in record["arms"] and "3_vs_3x" in record["pairs"]
        first, last = out[uid]
        assert [s["status"] for s in _morning_steps(first)] == ["skipped"]
        assert _morning_steps(first)[0]["why"] == final_batch.DEFERRED
        told = story_from(json.loads((first.log_path.parent / STORY).read_text(encoding="utf-8")))
        assert told.wake.morning == NotRun(final_batch.DEFERRED)  # type: ignore[union-attr]
        (ok,) = _morning_steps(last)
        assert ok["status"] == "ok" and ok[final_batch.RECIPE] == final_batch.recipe(os.environ)
        assert [morning_outcome(r) for r in out[uid]] == ["skipped", "ok"]


def test_every_final_night_records_the_recipe_it_ran_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_PRESSURE", "0")
    fake = FakeAccelerated(tmp_path)
    out = run_accelerated_batch(LIVES, 2, START, rig=_rig(fake, Scored(fake)), label="final")
    for result in (r for v in out.values() for r in v):
        recorded = json.loads((result.log_path.parent / "summary.json").read_text(encoding="utf-8"))[final_batch.RECIPE]
        assert set(recorded) == set(final_batch.RECIPE_ENV) and recorded["ANJO_SLEEP_PRESSURE"] == "0"
        assert recorded["ANJO_SLEEP_OBJECTIVE"] is None  # unset is recorded as unset


def test_the_recipe_names_the_switches_the_code_reads() -> None:
    from anjo.sleep import harness, parallel
    from anjo.sleep.agent_baseline import night as agent_night
    from anjo.sleep.night import core_snapshot, sleep_pressure
    from anjo.sleep.practice import roles
    from anjo.sleep.probes import morning
    from anjo.sleep.serving import backend, config
    from anjo.sleep.train import objective
    from anjo.sleep.train.runners import choice

    named = (objective.OBJECTIVE_ENV, objective.CAPACITY_ENV, objective.DAY_DREAMS_ENV, objective.DREAMS_ENV,
             objective.FOLD_ENV, objective.PROTECT_ENV, harness.HARNESS_ENV, roles.ROLES_ENV, core_snapshot.CHANNEL_ENV,
             "ANJO_SLEEP_DREAM_CURRICULUM", agent_night.FLAG, sleep_pressure.PRESSURE_ENV, choice.RUNNER_ENV,
             backend.BACKEND_ENV, config.REVISION_ENV, config.DETERMINISTIC_ENV, parallel.CONCURRENCY_ENV,
             morning.ARMS_EXTRA_ENV, morning.MORNING_SUBSET_ENV, "ANJO_SLEEP_JUDGE")
    # by name: mind.core.night.CURRICULUM_ENV needs torch; app.morning.JUDGE_ENV is checked with the app
    assert final_batch.RECIPE_ENV == named
    assert not any("TOKEN" in name or "KEY" in name for name in final_batch.RECIPE_ENV)  # flags only


def test_a_last_night_that_napped_is_still_scored_on_the_checkpoint_serving(tmp_path: Path) -> None:
    from anjo.sleep.night.sleep_pressure import SleepPolicy

    fake = FakeAccelerated(tmp_path)
    scored = Scored(fake)
    rig = _rig(fake, scored)
    naps = replace(rig, deps_for=lambda night: replace(rig.deps_for(night), sleep_policy=SleepPolicy(
        base_threshold=0.0 if night == START else 1.0)))
    out = run_accelerated_batch(LIVES, 2, START, rig=naps, label="final")
    assert [r.status for r in out["u1"]] == ["promoted", "napped"]
    assert [c["uid"] for c in scored.calls] == ["u1", "u2"]
    (record,) = _scores(fake, "u1")
    assert record["checkpoint_basis"] == "not_promoted"
    assert record["checkpoint"] == out["u1"][0].checkpoint.id  # type: ignore[union-attr]  # its last promoted


def test_a_napped_last_night_whose_final_morning_failed_exits_unscored(tmp_path: Path) -> None:
    from anjo.sleep.night.sleep_pressure import SleepPolicy

    fake = FakeAccelerated(tmp_path)
    rig = _rig(fake, Scored(fake, fail={"u1": TimeoutError("the judge timed out")}))
    naps = replace(rig, deps_for=lambda night: replace(rig.deps_for(night), sleep_policy=SleepPolicy(
        base_threshold=0.0 if night == START else 1.0)))
    out = run_accelerated_batch(LIVES, 2, START, rig=naps, label="final")
    assert out["u1"][1].status == "napped" and morning_outcome(out["u1"][1]).startswith("failed (")
    assert accelerated._exit([r for v in out.values() for r in v]) == UNSCORED_EXIT  # never a clean 0


def test_the_development_path_still_scores_every_night_as_it_runs(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    scored = Scored(fake)
    out = run_accelerated_batch(LIVES, 2, START, rig=_rig(fake, scored))
    assert [(c["uid"], c["night"], c["label"]) for c in scored.calls] == [
        ("u1", START, "development"), ("u2", START, "development"), ("u1", LAST, "development"),
        ("u2", LAST, "development")]
    assert ("u2", LAST) not in scored.calls[0]["extracted"]  # each morning right after its own night
    for result in (r for v in out.values() for r in v):
        summary = json.loads((result.log_path.parent / "summary.json").read_text(encoding="utf-8"))
        assert final_batch.RECIPE not in summary
        (ok,) = _morning_steps(result)
        assert ok["status"] == "ok" and final_batch.RECIPE not in ok
    assert all(len(_scores(fake, uid)) == 2 for uid in DONORS)


# ── a failed night stops the run; a resume never scores a life twice ──────────


def test_a_failed_night_stops_the_final_run_before_any_morning_and_a_resume_continues(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    scored, simulated = Scored(fake), []
    rig = _rig(fake, scored, simulated)
    inner = rig.simulate

    def simulate(uid: str, person: Any, i: int, seed: int) -> DayRun:
        if (uid, i) == ("u2", 1):
            fake.world.train_fails["u2"] = RuntimeError("Modal is down")
        return inner(uid, person, i, seed)

    with pytest.raises(FinalStopped, match="u2 at") as stopped:
        run_accelerated_batch(LIVES, 2, START, rig=replace(rig, simulate=simulate), label="final")
    assert "--resume" in str(stopped.value) and "no life was scored" in str(stopped.value)
    assert scored.calls == [] and not any(_scores(fake, uid) for uid in DONORS)
    assert [r.status for r in stopped.value.done["u2"]] == ["promoted", "failed"]
    assert [r.status for r in stopped.value.done["u1"]] == ["promoted", "promoted"]

    del fake.world.train_fails["u2"]
    before = len(fake.world.events)
    out = run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", resume=True)
    assert simulated == [("u1", 0), ("u2", 0), ("u1", 1), ("u2", 1)]  # no day simulated again
    reran = [(e[1], e[2]) for e in fake.world.events[before:] if e[0] == "extract"]
    assert reran == [("u2", LAST)]  # only the failed night, on the day it already had
    windows = [e[3] for e in fake.world.events if e[0] == "extract" and e[1:3] == ("u2", LAST)]
    assert len(windows) == 2 and windows[0] == windows[1]
    assert out["u2"][1].log_path.parent.name == f"{LAST.isoformat()}.2" and out["u2"][1].status == "promoted"
    assert [(c["uid"], c["night"], c["donor_night"]) for c in scored.calls] == [("u1", LAST, LAST), ("u2", LAST, LAST)]
    assert all(len(_scores(fake, uid)) == 1 for uid in DONORS)

    again = run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", resume=True)
    assert len(scored.calls) == 2 and len(simulated) == 4  # nothing simulated, run or scored again
    assert [r.status for r in again["u2"]] == ["promoted", "promoted"]
    assert all(len(_scores(fake, uid)) == 1 for uid in DONORS)


def test_a_night_failing_on_the_last_day_puts_off_the_others_so_no_backfill_scores_one_early(
    tmp_path: Path,
) -> None:
    """The review's case: on the last day u1 promotes and u2 fails. u1's last night is put off, so a
    backfill can't score it on u2's first-night adapter; the resume scores both, after u2's last night."""
    from anjo.sleep.budget import Ledger
    from anjo.sleep.tools import morning_backfill as backfill_tool

    fake = FakeAccelerated(tmp_path)
    scored = Scored(fake)
    rig = _rig(fake, scored)
    inner = rig.simulate

    def simulate(uid: str, person: Any, i: int, seed: int) -> DayRun:
        if (uid, i) == ("u2", 1):
            fake.world.train_fails["u2"] = RuntimeError("Modal is down")
        return inner(uid, person, i, seed)

    with pytest.raises(FinalStopped) as stopped:
        run_accelerated_batch(LIVES, 2, START, rig=replace(rig, simulate=simulate), label="final")
    assert [r.status for r in stopped.value.done["u1"]] == ["promoted", "promoted"]
    del fake.world.train_fails["u2"]
    last = fake.world.logs("u1") / LAST.isoformat()
    assert [(s["status"], s.get("why")) for s in steps(last / "log.jsonl") if s["step"] == "morning"] == [
        ("skipped", final_batch.DEFERRED)]
    ports = backfill_tool.Ports(morning=lambda uid, persona, night: replace(
        fake.morning(uid), judge_model=fireworks_api.DEEPSEEK_V4P1_FLASH), is_synthetic=lambda uid: True,
        logs=fake.world.logs, registry=fake.world.registry, ledger=Ledger(tmp_path / "budget.jsonl"))
    with pytest.raises(backfill_tool.Refused, match="put off"):
        backfill_tool.plan("u1", "founder", LAST, ports)
    assert scored.calls == [] and not any(_scores(fake, uid) for uid in DONORS)

    run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", resume=True)
    assert [(c["uid"], c["night"], c["donor_night"]) for c in scored.calls] == [("u1", LAST, LAST), ("u2", LAST, LAST)]


def test_a_resume_as_another_command_is_refused_before_anything_runs(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    scored, simulated = Scored(fake), []
    rig = _rig(fake, scored, simulated)
    fake.world.train_fails["u2"] = RuntimeError("Modal is down")
    with pytest.raises(FinalStopped):  # night 1 stops the run: nothing scored yet
        run_accelerated_batch(LIVES, 2, START, rig=rig, label="final")
    del fake.world.train_fails["u2"]
    ran = len(simulated)
    for changed, kw in (("days", {"days": 3}), ("start", {"start": START - timedelta(days=1)}),
                        ("practice_scenes", {"rig": replace(rig, shape={"practice_scenes": 18})}),
                        ("lives", {"lives": LIVES[::-1]})):
        args: dict[str, Any] = {"lives": LIVES, "days": 2, "start": START, "rig": rig, **kw}
        with pytest.raises(final_batch.ResumeRefused, match=changed if changed != "start" else "isn't this final"):
            run_accelerated_batch(args["lives"], args["days"], args["start"], rig=args["rig"], label="final",
                                  resume=True)
    assert len(simulated) == ran and scored.calls == []  # refused before any day, night or morning


def test_a_stopped_final_run_rerun_without_resume_is_refused_before_any_life_trains_again(tmp_path: Path) -> None:
    """The review's case: night 2 fails and the run stops; run again without ``--resume``, every life
    would train both nights again (``<date>.2``) and then be scored. Refused before any day."""
    fake = FakeAccelerated(tmp_path)
    scored, simulated = Scored(fake), []
    rig = _rig(fake, scored, simulated)
    inner = rig.simulate

    def simulate(uid: str, person: Any, i: int, seed: int) -> DayRun:
        if (uid, i) == ("u2", 1):
            fake.world.train_fails["u2"] = RuntimeError("Modal is down")
        return inner(uid, person, i, seed)

    with pytest.raises(FinalStopped):
        run_accelerated_batch(LIVES, 2, START, rig=replace(rig, simulate=simulate), label="final")
    del fake.world.train_fails["u2"]
    ran, events = len(simulated), len(fake.world.events)
    with pytest.raises(final_batch.ResumeRefused, match="u1 already has a final night .*--resume"):
        run_accelerated_batch(LIVES, 2, START, rig=rig, label="final")
    assert len(simulated) == ran and len(fake.world.events) == events and scored.calls == []
    assert not any(p.name.endswith(".2") for uid in DONORS for p in fake.world.logs(uid).iterdir())
    run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", resume=True)  # the way on
    assert [c["uid"] for c in scored.calls] == ["u1", "u2"] and all(len(_scores(fake, uid)) == 1 for uid in DONORS)


def test_a_life_is_never_scored_twice_by_a_changed_resume_or_a_fresh_run(tmp_path: Path) -> None:
    """The review's case: a finished 2-day run, then ``--resume --days 3``, would score every life again
    on a third night, and ``reports/final.py`` would read only the newer score."""
    fake = FakeAccelerated(tmp_path)
    scored, simulated = Scored(fake), []
    rig = _rig(fake, scored, simulated)
    run_accelerated_batch(LIVES, 2, START, rig=rig, label="final")
    ran = len(simulated)
    with pytest.raises(final_batch.ResumeRefused, match="already has a final score"):
        run_accelerated_batch(LIVES, 3, START, rig=rig, label="final", resume=True)
    with pytest.raises(final_batch.ResumeRefused, match="already has a final score"):  # a fresh run on the lives
        run_accelerated_batch(LIVES, 2, START + timedelta(days=2), rig=rig, label="final")
    assert len(simulated) == ran and len(scored.calls) == 2
    assert all(len(_scores(fake, uid)) == 1 for uid in DONORS)
    run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", resume=True)  # the same command: a no-op
    assert len(simulated) == ran and len(scored.calls) == 2


def test_every_final_night_records_the_command_it_ran_as(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    rig = replace(_rig(fake, Scored(fake)), shape={"practice_scenes": 16, "mind": False})
    out = run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", seed=5)
    for result in (r for v in out.values() for r in v):
        summary = json.loads((result.log_path.parent / "summary.json").read_text(encoding="utf-8"))
        assert summary[final_batch.RUN] == {"days": 2, "start": START.isoformat(), "seed": 5, "alone_from": None,
                                            "lives": [["u1", "founder"], ["u2", "founder_donor"]],
                                            "practice_scenes": 16, "mind": False}


def test_a_resume_scores_only_the_life_whose_final_morning_failed(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    scored = Scored(fake, fail={"u1": TimeoutError("the judge timed out")})
    rig = _rig(fake, scored)
    out = run_accelerated_batch(LIVES, 2, START, rig=rig, label="final")
    assert [c["uid"] for c in scored.calls] == ["u1", "u2"]  # the next life's morning still ran
    assert accelerated._exit([r for v in out.values() for r in v]) == UNSCORED_EXIT
    assert not _scores(fake, "u1") and len(_scores(fake, "u2")) == 1
    run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", resume=True)
    assert [c["uid"] for c in scored.calls] == ["u1", "u2", "u1"]  # u2 is never scored again
    assert len(_scores(fake, "u1")) == 1 and len(_scores(fake, "u2")) == 1


def test_a_resume_under_another_recipe_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAccelerated(tmp_path)
    scored = Scored(fake, fail={"u1": TimeoutError("down")})
    rig = _rig(fake, scored)
    run_accelerated_batch(LIVES, 2, START, rig=rig, label="final")
    monkeypatch.setenv("ANJO_SLEEP_CAPACITY", "1")
    with pytest.raises(final_batch.ResumeRefused, match="ANJO_SLEEP_CAPACITY"):
        run_accelerated_batch(LIVES, 2, START, rig=rig, label="final", resume=True)
    assert len(scored.calls) == 2


def test_only_a_final_run_resumes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeAccelerated(tmp_path)
    with pytest.raises(ValueError, match="only a final run resumes"):
        run_accelerated_batch(LIVES, 1, START, rig=_rig(fake, Scored(fake)), resume=True)
    with pytest.raises(ValueError, match="night logs"):
        run_accelerated_batch(LIVES, 1, START, rig=fake.rig(), label="final", resume=True)
    for argv in (["--fake", "--label", "final", "--resume"],
                 ["--wiring", "x:y", "--simulate", "x:z", "--morning", "x:w", "--user", "u1", "--persona", "founder",
                  "--resume"]):
        with pytest.raises(SystemExit):
            accelerated.main(argv)
        assert "--resume continues a real final run" in capsys.readouterr().err


# ── the registered judge ──────────────────────────────────────────────────────


def test_a_final_run_with_an_unregistered_judge_is_refused_before_any_day(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    simulated: list[tuple[str, int]] = []
    rig = replace(_rig(fake, Scored(fake), simulated),
                  preflight=lambda uid: require_registered("muse-spark-1.3", "final"))
    with pytest.raises(MorningError, match="registered judge"):
        run_accelerated_batch(LIVES, 2, START, rig=rig, label="final")
    assert simulated == [] and fake.world.events == []
    run_accelerated_batch(LIVES, 1, START, rig=rig)  # a development run is never checked here
    assert simulated == [("u1", 0), ("u2", 0)]


def test_a_bad_token_for_any_life_stops_the_final_run_before_any_lifes_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real run's preflight checks each life's judge, then its sim token against its account
    (``app.sim_transport.check_day``), for every life before any day: a bad token for the last life
    leaves no earlier life's day posted."""
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    from anjo.sleep.night import __main__ as night_command

    class TokenRefused(RuntimeError):
        """Stands in for ``app.sim_transport.TokenRefused``: the core never names the adapters' error, it only
        lets it through (the published tree leaves the adapters out, test_example_synthetic)."""

    @contextmanager
    def sweep(ledger: object, stop: object) -> Any:
        yield SimpleNamespace()

    monkeypatch.setattr("anjo.sleep.night.sweep.swept_night", sweep)
    monkeypatch.setattr("anjo.sleep.serving.backend.phase", nullcontext)
    monkeypatch.setattr("anjo.sleep.train.trainer.LoraTrainer", lambda *a, **k: object())
    monkeypatch.setattr(night_command, "night_cfg", lambda: object())
    monkeypatch.setattr(night_command, "_publisher", lambda name, landed: (lambda *a, **k: None))
    monkeypatch.setattr(night_command, "chosen_runner",
                        lambda *a, **k: SimpleNamespace(sweep=lambda: (0, 0), runner=object(), name="modal", landed=None))
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    checked: list[str] = []

    def check_day(uid: str) -> None:
        checked.append(uid)
        if uid == "u2":
            raise TokenRefused("the sim token belongs to another account than the day's: no line is posted")

    built = SimpleNamespace(judge_model=fireworks_api.DEEPSEEK_V4P1_FLASH, root=tmp_path / "results")
    named = {"t:wiring": print, "t:simulate": print, "t:morning": lambda uid, persona, night: built,
             accelerated.SIM_CHECK: check_day, accelerated.NIGHT_LOGS: lambda uid: tmp_path / "logs" / uid}
    monkeypatch.setattr(accelerated, "_named", named.__getitem__)
    seen: dict[str, Any] = {}

    def batch(people: Any, days: int, start: date, *, rig: Rig, **kw: Any) -> Any:
        seen["scores"], seen["shape"] = rig.scores("u1") if rig.scores else None, rig.shape
        return run_accelerated_batch(people, days, start, rig=replace(rig, simulate=simulate), **kw)

    days: list[str] = []

    def simulate(uid: str, person: Any, i: int, seed: int) -> DayRun:
        days.append(uid)
        raise AssertionError("a day ran")

    monkeypatch.setattr(accelerated, "run_accelerated_batch", batch)
    args = argparse.Namespace(wiring="t:wiring", simulate="t:simulate", morning="t:morning", users=["u1", "u2"],
                              personas=["founder", "founder_donor"], label="final", require_sha=accelerated.git_sha(),
                              days=2, start=START, seed=1, mind=False, match_file=None, alone_from=None,
                              resume=False, practice_scenes=17)
    with pytest.raises(TokenRefused, match="another account"):
        accelerated._real(args, argparse.ArgumentParser())
    assert checked == ["u1", "u2"] and days == []
    assert seen == {"scores": tmp_path / "results" / "u1" / "scores", "shape": {"practice_scenes": 17, "mind": False}}


def test_the_morning_scorer_refuses_a_final_morning_under_an_unregistered_judge(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    (result,) = run_accelerated_batch([("u1", FOUNDER)], 1, START, rig=replace(fake.rig(), score=None))["u1"]
    judged: dict[str, str] = {}

    def build(uid: str, persona: str, night: date) -> Any:
        return replace(fake.morning(uid), judge_model=judged["model"])

    score = accelerated.morning_scorer(build, {"u1": "founder"}, None)
    judged["model"] = "muse-spark-1.3"
    with pytest.raises(MorningError, match="registered judge"):
        score("u1", START, "final", result)
    assert not _scores(fake, "u1")  # refused before any call: nothing written
    judged["model"] = fireworks_api.DEEPSEEK_V4P1_FLASH
    score("u1", START, "final", result)
    (record,) = _scores(fake, "u1")
    assert (record["label"], record["judge"]) == ("final", "deepseek")
