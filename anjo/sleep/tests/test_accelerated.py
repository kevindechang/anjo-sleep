# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The accelerated runner (Task 12a): each day is simulated, its wall-clock window recorded, and the
night run on that window and the virtual date, in the past; records are labeled; the fake gate is
offline and deterministic."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from anjo.sleep.night.daily import UNSCORED_EXIT, morning_outcome
from anjo.sleep.night.deps import NightDeps
from anjo.sleep.night.nightlog import STORY
from anjo.sleep.night.run import day_window
from anjo.sleep.night.story import MORNING_FAILED, SELF_MODEL_NOT_YET, Failed, NotRun, Wake, story_from
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.personas.model import Persona, load_person_file
from anjo.sleep.sim import accelerated
from anjo.sleep.sim.accelerated import AcceleratedError, run_accelerated, run_accelerated_batch
from anjo.sleep.probes.morning import COMMITTED_ARMS, MorningError, PAIRS, TooManyNoVerdicts
from anjo.sleep.sim.day import DayRun, run_day
from anjo.sleep.types import CheckpointRef, NightResult
from anjo.sleep.tests import accelerated_rig
from anjo.sleep.tests.accelerated_rig import CLOCK_START, FAKE_SHA, FakeAccelerated
from anjo.sleep.tests.night_fakes import step

START = date(2026, 10, 20)  # before the rig's wall clock (2026-10-28): an accelerated life is dated in the past
PERSONAS = Path(__file__).parents[1] / "personas"
FOUNDER = load_person_file(PERSONAS / "founder.json").visible
DONOR = load_person_file(PERSONAS / "founder_donor.json").visible
HERO = load_person_file(PERSONAS / "film_founder.json").visible


def _summary(path: Path) -> dict[str, object]:
    found: dict[str, object] = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    return found


@pytest.mark.parametrize("person", [FOUNDER, HERO], ids=["founder", "hero"])
def test_accelerated_results_are_labeled(tmp_path: Path, person: Persona) -> None:
    fake = FakeAccelerated(tmp_path)
    results = run_accelerated("u1", person, 2, START, rig=fake.rig())
    assert [r.status for r in results] == ["promoted", "promoted"]
    for i, result in enumerate(results):
        summary = _summary(result.log_path)
        assert summary["accelerated"] is True and summary["git_sha"] == FAKE_SHA
        assert summary["label"] == "development" and summary["day_index"] == i
        assert summary["night"] == (START + timedelta(days=i)).isoformat()
    final = run_accelerated("u2", FOUNDER, 1, START, rig=fake.rig(), label="final")
    assert _summary(final[0].log_path)["label"] == "final"
    with pytest.raises(ValueError, match="label"):
        run_accelerated("u3", FOUNDER, 1, START, rig=fake.rig(), label="draft")


# ── a final run must be on the frozen code (Task 12b) ─────────────────────────


def test_final_label_without_require_sha_is_refused_on_real_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeAccelerated(tmp_path)
    touched: list[object] = []

    def simulate(uid: str, person: object, i: int, seed: int) -> DayRun:
        touched.append(uid)
        raise AssertionError("a refused run simulates no day")

    def deps_for(night: date) -> NightDeps:
        touched.append(night)
        raise AssertionError("a refused run runs no night")

    rig = replace(fake.rig(sha=None), simulate=simulate, deps_for=deps_for)  # sha=None: this commit's code
    with pytest.raises(AcceleratedError, match="frozen"):
        run_accelerated("u1", FOUNDER, 1, START, rig=rig, label="final")
    assert touched == [] and fake.world.events == []
    real = ["--wiring", "x:y", "--simulate", "x:z", "--morning", "x:w", "--days", "2", "--start", "2026-10-20",
            "--user", "u1", "--persona", "film_founder", "--label", "final"]
    with pytest.raises(SystemExit):  # the command refuses it before any sweep, VM or call
        accelerated.main(real)
    assert "require-sha" in capsys.readouterr().err


def test_accelerated_refuses_when_head_differs_from_require_sha(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    wrong = "0" * 40 if accelerated.git_sha() != "0" * 40 else "f" * 40
    with pytest.raises(AcceleratedError, match="frozen"):
        run_accelerated("u1", FOUNDER, 1, START, rig=fake.rig(sha=None), label="final", require_sha=wrong)
    assert fake.world.events == []  # refused before any day simulated or night run


def test_accelerated_accepts_matching_require_sha(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    head = accelerated.git_sha()
    results = run_accelerated("u1", FOUNDER, 1, START, rig=fake.rig(sha=None), label="final", require_sha=head)
    assert [r.status for r in results] == ["promoted"]
    summary = _summary(results[0].log_path)
    assert summary["git_sha"] == head and summary["label"] == "final"


def test_fake_final_runs_ignore_require_sha(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    results = run_accelerated("u1", FOUNDER, 1, START, rig=fake.rig(), label="final", require_sha="0" * 40)
    assert [r.status for r in results] == ["promoted"]
    assert _summary(results[0].log_path)["git_sha"] == FAKE_SHA  # a fake run's own sha, not the given one
    assert accelerated.main(["--fake", "--days", "1", "--label", "final"]) == 0
    assert accelerated.main(["--fake", "--days", "1", "--label", "final", "--require-sha", "0" * 40]) == 0
    # not on git, a fake run ignores --require-sha whatever it looks like, not only a well-formed sha
    assert accelerated.main(["--fake", "--days", "1", "--label", "final", "--require-sha", "abcd"]) == 0


def test_a_real_final_run_refuses_a_wrong_sha_before_the_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from anjo.sleep.night import __main__ as night_command

    swept: list[bool] = []

    @contextmanager
    def sweep(ledger: object, stop: object) -> Iterator[object]:
        swept.append(True)  # the sweep stops leftover Colab/Modal sessions and settles open holds
        yield SimpleNamespace()

    monkeypatch.setattr("anjo.sleep.night.sweep.swept_night", sweep)
    monkeypatch.setattr(night_command, "night_cfg", lambda: object())
    monkeypatch.setattr(night_command, "chosen_runner",
                        lambda *a, **k: SimpleNamespace(sweep=lambda: (0, 0), runner=object(), name="modal"))
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    wrong = "0" * 40 if accelerated.git_sha() != "0" * 40 else "f" * 40
    args = argparse.Namespace(wiring="builtins:print", simulate="builtins:print", morning="builtins:print",
                              users=["u1"], personas=["film_founder"], label="final", require_sha=wrong,
                              days=1, start=START, seed=1, mind=False, match_file=None)
    with pytest.raises(SystemExit, match="frozen"):
        accelerated._real(args, argparse.ArgumentParser())
    assert swept == []  # wrong code touches nothing: no VM stopped, no hold settled, no day, no night


def test_a_final_run_names_the_frozen_sha_before_it_reads_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fake = FakeAccelerated(tmp_path)

    def down(*args: object, **kw: object) -> str:
        raise AcceleratedError("the git sha of this run can't be read")

    monkeypatch.setattr(accelerated, "git_sha", down)
    with pytest.raises(AcceleratedError, match="frozen"):  # the missing flag, not a git that can't be read
        run_accelerated("u1", FOUNDER, 1, START, rig=fake.rig(sha=None), label="final")


def test_the_frozen_sha_is_refused_before_a_future_start(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    wrong = "0" * 40 if accelerated.git_sha() != "0" * 40 else "f" * 40
    with pytest.raises(AcceleratedError, match="frozen"):  # the wrong code, not the future date
        run_accelerated("u1", FOUNDER, 1, date(2099, 1, 1), rig=fake.rig(sha=None), label="final", require_sha=wrong)


def test_a_real_run_does_not_turn_a_late_accelerated_error_into_a_system_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    from anjo.sleep.night import __main__ as night_command

    @contextmanager
    def sweep(ledger: object, stop: object) -> Iterator[object]:
        yield SimpleNamespace()

    monkeypatch.setattr("anjo.sleep.night.sweep.swept_night", sweep)
    monkeypatch.setattr("anjo.sleep.serving.backend.phase", nullcontext)
    monkeypatch.setattr("anjo.sleep.train.trainer.LoraTrainer", lambda *a, **k: object())
    monkeypatch.setattr(night_command, "night_cfg", lambda: object())
    monkeypatch.setattr(night_command, "_publisher", lambda name, landed: (lambda *a, **k: None))
    monkeypatch.setattr(night_command, "chosen_runner",
                        lambda *a, **k: SimpleNamespace(sweep=lambda: (0, 0), runner=object(), name="modal", landed=None))
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    head = accelerated.git_sha()

    def late(*a: object, **k: object) -> object:
        raise AcceleratedError("u1: the simulated day's window is empty")  # after night 1, not the sha

    monkeypatch.setattr(accelerated, "run_accelerated_batch", late)
    args = argparse.Namespace(wiring="builtins:print", simulate="builtins:print", morning="builtins:print",
                              users=["u1"], personas=["film_founder"], label="final", require_sha=head,
                              days=2, start=START, seed=1, mind=False, match_file=None, alone_from=None,
                              resume=False, practice_scenes=16)
    with pytest.raises(AcceleratedError, match="window"):  # a mid-run failure still raises, not a clean exit
        accelerated._real(args, argparse.ArgumentParser())


def test_two_accelerated_days_extract_separate_evidence(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    run_accelerated("u1", FOUNDER, 2, START, rig=fake.rig())
    extracts = [e for e in fake.world.events if e[0] == "extract"]
    assert [e[2] for e in extracts] == [START, START + timedelta(days=1)]
    (first_start, first_end), (second_start, second_end) = extracts[0][3], extracts[1][3]
    assert first_start < first_end <= second_start < second_end  # the second night never re-reads day one
    store = fake.world.evidence("u1")
    one, two = ([i for i in store.for_day(d) if i.kind == "lesson"] for d in (START, START + timedelta(days=1)))
    assert one and two and not {i.id for i in one} & {i.id for i in two}
    # stamped as the history source stamps them, at their own night's clock: earlier than their window
    assert all(i.created_at == day_window(START)[1] < first_end for i in one)
    assert all(i.created_at == day_window(START + timedelta(days=1))[1] < second_end for i in two)


def test_a_window_that_overlaps_the_last_day_is_refused(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    stale = fake.rig()

    def same_window(uid: str, person: object, i: int, seed: int) -> DayRun:
        day = fake.simulate(uid, FOUNDER, i, seed)
        return replace(day, window=(day.window[0] - timedelta(days=i), day.window[1] - timedelta(days=i)))

    with pytest.raises(AcceleratedError, match="window"):
        run_accelerated("u1", FOUNDER, 2, START, rig=replace(stale, simulate=same_window))


def test_lives_run_in_lockstep_through_one_batch_per_day(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    out = run_accelerated_batch([("u1", FOUNDER), ("u2", DONOR)], 2, START, rig=fake.rig())
    assert set(out) == {"u1", "u2"} and all(len(v) == 2 for v in out.values())
    order = [(e[1], e[2]) for e in fake.world.events if e[0] == "extract"]
    assert order == [("u1", START), ("u2", START), ("u1", START + timedelta(days=1)), ("u2", START + timedelta(days=1))]
    assert all(r.status == "promoted" for v in out.values() for r in v)


def test_a_failed_night_does_not_stop_the_life(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    fake.world.extract_fails["u1"] = RuntimeError("boom")
    results = run_accelerated("u1", FOUNDER, 2, START, rig=fake.rig())
    assert [r.status for r in results] == ["failed", "failed"]


class Down:
    """Spark Open refusing after every retry, its message carrying text that must not travel."""

    def complete(self, system: str, messages: list[dict[str, str]], *, temperature: float, seed: int) -> str:
        raise ModelApiError("Fireworks chat with muse-spark-open returned HTTP 503: an upstream body")


def test_a_day_whose_spark_open_call_fails_is_recorded_and_the_life_goes_on(tmp_path: Path) -> None:
    """Day 2 of 3 has no Spark Open reply after its waits: its day is short, days 1 and 3 still run."""
    fake = FakeAccelerated(tmp_path)
    rig = fake.rig()
    failed: list[DayRun] = []

    class Never:
        def send(self, session_id: str, text: str, *, cleared: bool) -> str:
            raise AssertionError("a day whose Spark Open call failed sends no turn")

    def simulate(uid: str, person: Persona, i: int, seed: int) -> DayRun:
        if i != 1:
            return rig.simulate(uid, person, i, seed)
        day = run_day(person, i, seed, model=Down(), transport=Never(), replies=accelerated_rig.Replies(),
                      clock=fake.clock)
        failed.append(day)
        return day

    results = run_accelerated("u1", FOUNDER, 3, START, rig=replace(rig, simulate=simulate))
    assert len(results) == 3 and [r.status for r in results[:1] + results[2:]] == ["promoted", "promoted"]
    assert all(r.failed_at is None for r in results)  # the night ran, never a dead life
    (day,) = failed  # day 2's own summary: no turns, one failure, the error class only
    assert (day.answered, day.failed, day.turns) == (0, 1, ()) and day.short
    assert day.failure_detail == "ModelApiError" and "503" not in day.failure_detail and "body" not in day.failure_detail
    record = step(results[1].log_path, "simulate")  # and the night that closed it records the short day
    assert (record["answered"], record["failed_turns"], record["short_day"]) == (0, 1, True)
    extracted = [e[2] for e in fake.world.events if e[0] == "extract"]
    assert extracted == [START, START + timedelta(days=1), START + timedelta(days=2)]  # every day's night ran


def test_days_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="days"):
        run_accelerated("u1", FOUNDER, 0, START, rig=FakeAccelerated(tmp_path).rig())


def test_fake_gate_runs_offline_and_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    started = time.monotonic()
    assert accelerated.main(["--fake", "--days", "3"]) == 0
    first = capsys.readouterr().out
    assert accelerated.main(["--fake", "--days", "3"]) == 0
    second = capsys.readouterr().out
    assert first == second and time.monotonic() - started < 60
    report = json.loads(first)
    assert report["accelerated"] is True and report["fake"] is True and report["days"] == 3
    assert [n["status"] for n in report["nights"]] == ["promoted"] * 3
    assert len(report["digest"]) == 64


def test_fake_gate_digest_changes_when_the_run_does(capsys: pytest.CaptureFixture[str]) -> None:
    accelerated.main(["--fake", "--days", "2"])
    two = json.loads(capsys.readouterr().out)["digest"]
    accelerated.main(["--fake", "--days", "3"])
    assert json.loads(capsys.readouterr().out)["digest"] != two


def test_a_life_dated_after_the_run_starts_is_refused_before_any_paid_step(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    touched: list[object] = []

    def simulate(uid: str, person: object, i: int, seed: int) -> DayRun:
        touched.append(uid)
        raise AssertionError("a refused life simulates no day")

    def deps_for(night: date) -> NightDeps:
        touched.append(night)
        raise AssertionError("a refused life runs no night")

    rig = replace(fake.rig(), simulate=simulate, deps_for=deps_for)
    last = CLOCK_START.date() - timedelta(days=1)  # two nights from here: the second ends after the run starts
    with pytest.raises(AcceleratedError, match="dated in the past, so start on or before 2026-10-26"):
        run_accelerated("u1", FOUNDER, 2, last, rig=rig)
    with pytest.raises(AcceleratedError, match="dated in the past"):
        run_accelerated("u1", FOUNDER, 1, date(2026, 11, 3), rig=rig)
    assert touched == [] and fake.world.events == [] and fake.clock.now == CLOCK_START  # no day, no night
    assert [r.status for r in run_accelerated("u1", FOUNDER, 2, last - timedelta(days=1), rig=fake.rig())] == ["promoted"] * 2


def test_the_command_starts_a_life_in_the_past_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert accelerated.main(["--fake", "--days", "3"]) == 0
    nights = [n["night"] for n in json.loads(capsys.readouterr().out)["nights"]]
    assert nights == ["2026-10-25", "2026-10-26", "2026-10-27"]  # the rig's clock reads 2026-10-28: the last ended at midnight
    with pytest.raises(SystemExit):
        accelerated.main(["--fake", "--days", "3", "--start", "2026-11-03"])
    assert "dated in the past" in capsys.readouterr().err
    now = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    assert accelerated.latest_start(now, 7) == date(2026, 11, 2)
    accelerated.check_dates(date(2026, 11, 2), 7, now)  # its last night ends exactly as the run starts


def test_a_real_run_names_its_wiring_and_a_persona() -> None:
    with pytest.raises(SystemExit):
        accelerated.main(["--days", "2", "--start", "2026-10-20", "--user", "u1"])
    with pytest.raises(SystemExit):  # a user without a persona is a real person: never accelerated
        accelerated.main(["--wiring", "x:y", "--simulate", "x:z", "--days", "2", "--start", "2026-10-20", "--user", "u1"])


def test_the_git_sha_is_read_from_the_repo() -> None:
    sha = accelerated.git_sha()
    assert len(sha) == 40 and int(sha, 16) >= 0



def test_the_vm_record_and_day_lengths_pass_through_the_rig(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    seen: list[tuple[str, float]] = []
    scored: list[tuple[str, float]] = []
    rig = replace(fake.rig(), on_day=lambda uid, seconds: seen.append((uid, seconds)),
                  on_scoring=lambda uid, seconds: scored.append((uid, seconds)),
                  vm_record=lambda uid, night: {"night": night, "vm_kept": False, "idle_s_before": None})
    results = run_accelerated("u1", FOUNDER, 2, START, rig=rig)
    assert [uid for uid, _ in seen] == ["u1", "u1"] and all(seconds > 0 for _, seconds in seen)
    assert [uid for uid, _ in scored] == ["u1", "u1"] and all(seconds >= 0 for _, seconds in scored)
    assert _summary(results[0].log_path)["vm"] == {"night": START.isoformat(), "vm_kept": False, "idle_s_before": None}


def test_fake_gate_includes_scoring(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_SLEEP_HARNESS", "0")  # the registered six, before the reported-only 2H
    assert accelerated.main(["--fake", "--days", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["scored_nights"] == 2
    for scored in report["scores"]:
        assert set(scored["arms"]) == {"1", "2C", "2F", "3", "3x", "4"} and scored["unavailable"] == {}
        # A full run without ANJO_SLEEP_ARMS_EXTRA asks the committed arms, so their pairs alone answer.
        assert set(scored["pairs"]) == {n for n, (a, b) in PAIRS.items() if a in COMMITTED_ARMS and b in COMMITTED_ARMS}
    assert report["scores"][0]["night"] != report["scores"][1]["night"]


def test_each_accelerated_night_is_scored_once_after_it_runs(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    run_accelerated("u1", FOUNDER, 2, START, rig=fake.rig())
    assert sorted(p.name for p in fake.scores_dir("u1").iterdir()) == ["night-01.json", "night-02.json"]
    bad = FakeAccelerated(tmp_path / "bad")
    bad.world.extract_fails["u1"] = RuntimeError("boom")
    run_accelerated("u1", FOUNDER, 1, START, rig=bad.rig())
    assert not bad.scores_dir("u1").exists()  # a failed night scores nothing


def test_the_morning_scores_the_nights_own_checkpoint(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    run_accelerated("u1", FOUNDER, 1, START, rig=fake.rig())
    record = json.loads((fake.scores_dir("u1") / "night-01.json").read_text(encoding="utf-8"))
    assert record["checkpoint"] == fake.world.registry.last_promoted("u1").id  # type: ignore[union-attr]
    assert record["checkpoint_basis"] == "promoted"


def test_a_registry_change_between_the_night_and_the_morning_refuses_the_run(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    original = fake.score

    class Moved:
        def last_promoted(self, person_id: str) -> object:
            return CheckpointRef("someone-elses-n9", person_id, START, "fake://elsewhere", None)

    def moved(uid: str, night: date, label: str, result: object) -> object:
        fake.world.registry = Moved()  # type: ignore[assignment]
        return original(uid, night, label, result)  # type: ignore[arg-type]

    with pytest.raises(MorningError, match="serving"):
        run_accelerated("u1", FOUNDER, 1, START, rig=replace(fake.rig(), score=moved))
    assert not fake.scores_dir("u1").exists()


def test_a_napped_night_is_not_scored_and_the_report_says_how_each_night_slept(tmp_path: Path) -> None:
    from anjo.sleep.night.sleep_pressure import SleepPolicy

    fake = FakeAccelerated(tmp_path)
    rig = fake.rig()
    naps = replace(rig, deps_for=lambda night: replace(rig.deps_for(night), sleep_policy=SleepPolicy(base_threshold=1.0)))
    results = run_accelerated("u1", FOUNDER, 2, START, rig=naps)
    assert [r.status for r in results] == ["napped"] * 2 and not fake.scores_dir("u1").exists()  # no GPU morning
    deep = replace(rig, deps_for=lambda night: replace(rig.deps_for(night), sleep_policy=SleepPolicy(base_threshold=0.0)))
    slept = run_accelerated("u2", FOUNDER, 1, START, rig=deep)
    report = accelerated._report([*results, *slept], fake=True, days=3, sha=FAKE_SHA)
    assert [n["sleep"] for n in report["nights"]] == ["nap", "nap", "deep"]
    assert [n["status"] for n in report["nights"]] == ["napped", "napped", "promoted"]


# ── a morning that fails (the judge) never stops the run ──────────────────────


def _flaky(fake: FakeAccelerated, error: Exception) -> tuple[Callable[..., object], list[date]]:
    """The fake's scoring, failing with ``error`` on the run's first morning only."""
    original, mornings = fake.score, []

    def score(uid: str, night: date, label: str, result: NightResult) -> object:
        mornings.append(night)
        if len(mornings) == 1:
            raise error
        return original(uid, night, label, result)

    return score, mornings


def test_a_judge_that_fails_one_morning_is_logged_and_the_run_goes_on(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    unread = TooManyNoVerdicts("the judge gave no readable verdict on 3 of 3 probes", no_verdict=3)
    score, mornings = _flaky(fake, unread)
    scored: list[float] = []
    results = run_accelerated("u1", FOUNDER, 2, START,
                              rig=replace(fake.rig(), score=score, on_scoring=lambda uid, s: scored.append(s)))
    assert [r.status for r in results] == ["promoted", "promoted"] and mornings == [START, START + timedelta(days=1)]
    assert len(scored) == 2  # the VM is told about the failed morning's time too
    first = step(results[0].log_path, "morning")
    assert first["status"] == "failed" and first["no_verdict"] == 3
    assert first["why"] == f"{MORNING_FAILED}TooManyNoVerdicts: the judge gave no readable verdict on 3 of 3 probes"
    assert step(results[1].log_path, "morning")["status"] == "ok"
    assert [morning_outcome(r) for r in results] == [
        "failed (TooManyNoVerdicts: the judge gave no readable verdict on 3 of 3 probes)", "ok"]
    told = story_from(json.loads((results[0].log_path.parent / STORY).read_text(encoding="utf-8")))
    assert told.wake == Wake(Failed("morning", "TooManyNoVerdicts: the judge gave no readable verdict on 3 of 3 probes"),
                             NotRun(SELF_MODEL_NOT_YET))  # the story told again with its wake, as the daily run does


def test_a_failed_morning_never_quotes_another_librarys_error(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    score, _ = _flaky(fake, TimeoutError("POST https://judge.example/v1 timed out for u1"))
    results = run_accelerated("u1", FOUNDER, 1, START, rig=replace(fake.rig(), score=score))
    assert step(results[0].log_path, "morning")["why"] == f"{MORNING_FAILED}TimeoutError"


def test_the_command_exits_3_when_a_morning_failed_and_says_which(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, calls = accelerated_rig.FakeAccelerated.score, []

    def score(self: FakeAccelerated, uid: str, night: date, label: str, result: NightResult) -> object:
        calls.append(night)
        if len(calls) == 1:
            raise TooManyNoVerdicts("the judge gave no readable verdict", no_verdict=2)
        return original(self, uid, night, label, result)

    monkeypatch.setattr(accelerated_rig.FakeAccelerated, "score", score)
    assert accelerated.main(["--fake", "--days", "2"]) == UNSCORED_EXIT == 3
    report = json.loads(capsys.readouterr().out)
    assert [n["status"] for n in report["nights"]] == ["promoted", "promoted"]
    assert [n["morning"] for n in report["nights"]] == ["failed (TooManyNoVerdicts: the judge gave no readable verdict)", "ok"]
