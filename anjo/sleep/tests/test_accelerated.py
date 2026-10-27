# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The accelerated runner (Task 12a): each day is simulated, its wall-clock window recorded, and the
night run on that window and the virtual date; records are labeled; the fake gate is offline and
deterministic."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from anjo.sleep.personas.model import load_person_file
from anjo.sleep.sim import accelerated
from anjo.sleep.sim.accelerated import AcceleratedError, run_accelerated, run_accelerated_batch
from anjo.sleep.probes.morning import MorningError
from anjo.sleep.sim.day import DayRun
from anjo.sleep.types import CheckpointRef
from anjo.sleep.tests.accelerated_rig import FAKE_SHA, FakeAccelerated

START = date(2026, 11, 3)
PERSONAS = Path(__file__).parents[1] / "personas"
FOUNDER = load_person_file(PERSONAS / "founder.json").visible
DONOR = load_person_file(PERSONAS / "founder_donor.json").visible


def _summary(path: Path) -> dict[str, object]:
    found: dict[str, object] = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    return found


def test_accelerated_results_are_labeled(tmp_path: Path) -> None:
    fake = FakeAccelerated(tmp_path)
    results = run_accelerated("u1", FOUNDER, 2, START, rig=fake.rig())
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
    # stamped as the history source stamps them: at their own window's end (the night's clock is later)
    assert all(i.created_at == first_end for i in one)
    assert all(i.created_at == second_end for i in two)


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


def test_a_real_run_names_its_wiring_and_a_persona() -> None:
    with pytest.raises(SystemExit):
        accelerated.main(["--days", "2", "--start", "2026-11-03", "--user", "u1"])
    with pytest.raises(SystemExit):  # a user without a persona is a real person: never accelerated
        accelerated.main(["--wiring", "x:y", "--simulate", "x:z", "--days", "2", "--start", "2026-11-03", "--user", "u1"])


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


def test_fake_gate_includes_scoring(capsys: pytest.CaptureFixture[str]) -> None:
    assert accelerated.main(["--fake", "--days", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["scored_nights"] == 2
    for scored in report["scores"]:
        assert set(scored["arms"]) == {"1", "2C", "2F", "3", "3x", "4"} and scored["unavailable"] == {}
        assert set(scored["pairs"]) == {"3_vs_3x", "4_vs_2C"}
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
