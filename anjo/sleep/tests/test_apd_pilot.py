# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The APD pilot command (Task 17): refused without ``--live``, a plan that spends nothing, the
held-out lives drawn only at the freeze, blind judging with Claude's audit, and the whole chain on
fakes. No model, no VM, no network."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import fakes, generate, intervene, judging, live, pilot, timing
from anjo.sleep.mind.apd.interventions import EDIT_KINDS
from anjo.sleep.mind.apd.store import DataDir, StoreError
from anjo.sleep.mind.apd.types import HELD_OUT_LIVES, HELD_OUT_LIVES_CUT, INTERVENTION_SNAPSHOTS
from anjo.sleep.mind.sweep import MIND_KINDS
from anjo.sleep.reports import apd as apd_report
from anjo.sleep.prereg.ledger import read_entries, register
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.config import ServeConfigError

LIVE = ["--live", "--prices", "0.6,2.4,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]


def _main(argv: list[str]) -> tuple[int, str]:
    lines: list[str] = []
    code = cli.main(argv, out=lines.append)
    return code, "\n".join(lines)


def test_live_refused_without_flag_and_plan_spends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_backend(*args: object, **kwargs: object) -> None:
        raise AssertionError("a plan builds no live backend")

    monkeypatch.setattr(backends, "live_backend", no_backend)
    monkeypatch.setattr("anjo.sleep.budget.default_ledger", no_backend)
    code, text = _main(["--data", str(tmp_path), "run", "--held-out"])
    assert code == 2 and "--live" in text and "Plan" not in text
    for command in (["generate"], ["run", "--held-out"], ["timing"], ["intervene"], ["judge"],
                    ["kill-resume", "--kill-at", "9"]):
        code, text = _main([*LIVE, "--data", str(tmp_path), "--hours", "4", *command])
        assert code == 0 and text.startswith("Plan:") and text.endswith("Nothing has run. Add --yes to start it.")
        assert "Hold on " in text
        if command[0] in ("run", "timing", "intervene", "kill-resume"):
            assert "hard timeout 4.0 h" in text and "colab_cu: 30.0 CU" in text and "baseten_mind_usd" in text
    assert list(tmp_path.iterdir()) == []  # nothing written, nothing held
    assert "meta_usd" in _main([*LIVE, "--data", str(tmp_path), "judge"])[1]
    with pytest.raises(SystemExit):
        cli.main(["--live", "--data", str(tmp_path), "run"], out=lambda _line: None)  # unpriced: refused


def _ledger(tmp_path: Path, *entries: tuple[str, str, str]) -> list[object]:
    path = tmp_path / "prereg.jsonl"
    for kind, digest, note in entries:
        register(kind, digest, note, path=path)
    return list(read_entries(path))


def test_heldout_generation_refused_before_freeze(tmp_path: Path) -> None:
    seed = "a seed committed before the freeze"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    seeded = ("apd_draw_seed", digest, "draw seed")
    frozen = ("apd_freeze", "f" * 64, f"APD frozen held_out_lives={HELD_OUT_LIVES}")
    cases = {
        "no freeze": [],
        "seed only": [seeded],
        "seed after the freeze": [frozen, seeded],
        "another seed": [("apd_draw_seed", "e" * 64, "x"), frozen],
    }
    for name, entries in cases.items():
        with pytest.raises(pilot.FreezeError):
            pilot.held_out_specs(_ledger(tmp_path / name, *entries), seed, HELD_OUT_LIVES)  # type: ignore[arg-type]
    good = _ledger(tmp_path / "good", seeded, frozen)
    with pytest.raises(pilot.FreezeError, match="another number"):
        pilot.held_out_specs(good, seed, HELD_OUT_LIVES_CUT)  # type: ignore[arg-type]
    specs = pilot.held_out_specs(good, seed, HELD_OUT_LIVES)  # type: ignore[arg-type]
    assert len(specs) == HELD_OUT_LIVES and {s.split for s in specs} == {"held_out"}
    data = tmp_path / "data"
    code, text = _main(["--fake", "--data", str(data), "generate", "--held-out", "--draw-seed", seed,
                        "--prereg", str(tmp_path / "seed only" / "prereg.jsonl")])
    assert code == cli.REFUSED_CODE and "Refused: APD isn't frozen" in text  # through the command, refused cleanly
    assert not (data / "lives").exists() and not (data / "sealed").exists()


def _judged_life(tmp_path: Path) -> tuple[DataDir, generate.Generated]:
    made = fakes.fake_life(next(s for s in generate.draw_held_out("judge-test") if s.prevalence >= 2))
    data = DataDir(tmp_path / "data")
    data.save_life(made.life, made.problems)
    with fakes.serving() as session:
        run = live.run_live("M", made.life, session, fakes.FakeSparkChat(), ledger=None, prices=None)
    data.save_run(run)
    data.save_run(dataclasses.replace(run, arm="B", events=(), turns=(), written=()))  # the same reports as B's
    return data, made


def test_judge_blind_to_arm_and_sees_report_time_evidence_only(tmp_path: Path) -> None:
    data, made = _judged_life(tmp_path)
    seen: dict[str, list[str]] = {"muse": [], "gemini": []}

    def model(name: str, contrary: bool) -> judging.TextModel:
        def complete(prompt: str) -> tuple[str, int, int]:
            seen[name].append(prompt)
            ruling = json.loads(fakes.fake_judge(False)(prompt))
            return json.dumps({**ruling, "holds": ruling["holds"] != contrary}), 100, 10
        return complete

    muse = judging.CachedJudge(judging.MUSE, model("muse", False), data.judging / "muse.jsonl")
    gemini = judging.CachedJudge(judging.GEMINI, model("gemini", True), data.judging / "gemini.jsonl")  # differs
    runs = (data.run(made.life.id, "M"), data.run(made.life.id, "B"))
    judged = judging.judge_runs(data, [made.life], [r for r in runs if r is not None], muse, gemini)
    prompts = seen["muse"] + seen["gemini"]
    assert prompts and len(seen["muse"]) == len(set(seen["muse"]))  # M's and B's same reports: asked once
    assert not any(made.life.id in p or '"arm"' in p or "Arm:" in p for p in prompts)
    later = {e.id for e in made.life.evidence if e.day == made.life.days[-1]}
    first = min(r.day for r in runs[0].reports)  # type: ignore[union-attr]
    early = [p for p in prompts if f"{first}" in p and not any(f"[{i}]" in p for i in later)]
    assert early  # a report's prompt shows the evidence up to its night, never the last day's
    pending = [json.loads(line) for line in (data.audit / "pending.jsonl").read_text().splitlines()]
    assert judged.pending == len(pending) > 0 and set(pending[0]) == {"id", "prompt"}
    assert [row["id"] for row in pending] == sorted(row["id"] for row in pending)  # hash order, not arm order
    assert all(made.life.id not in row["prompt"] for row in pending)
    with pytest.raises(ValueError, match="blind audit"):
        pilot.report(judged)
    judging.answer_pending(data, fakes.fake_judge(False))
    again = judging.judge_runs(data, [made.life], [r for r in runs if r is not None], muse, gemini)
    assert again.pending == 0 and muse.calls == len(seen["muse"])  # every prompt judged once, then cached


def test_adapters_stay_in_the_cache_and_go_up_only_to_the_apd_session(tmp_path: Path) -> None:
    publisher = pilot.ColabPublisher()
    good = fakes.adapter_folder(tmp_path / "p-0123456789ab-n0", framing.rendering(config.reasoning_strength()))
    publisher(good, good.name)
    with pytest.raises(ServeConfigError):
        publisher(tmp_path / "missing", "p-ba9876543210-n0")
    vm = fakes.FakeColab()
    with fakes.serving(vm=vm) as session:
        publisher.upload(session)
    assert vm.adapters == {good.name} and good.exists()  # uploaded from the cache, which keeps it


def test_fake_pilot_runs_end_to_end(tmp_path: Path) -> None:
    code, text = _main(["--fake", "--data", str(tmp_path), "--only", "2"])
    assert code == 0
    for step in ("generate", "timing", "run", "kill-resume", "judge --held-out", "report"):
        assert f" {step}" in text
    assert "N=16, with M-control, with 17b" in text and "300-unit line" in text
    assert text.count("digest equal, conclusions agree") == 2
    assert "| Life | M (audited) |" in text and "## Replay (Task 17b)" in text
    data = DataDir(tmp_path)
    for life in data.lives("held_out"):
        assert all(data.run(life.id, arm) is not None for arm in (*pilot.HELD_OUT_ARMS, *pilot.REPLAY_ARMS))
    for life in data.lives("development"):
        assert all(data.run(life.id, arm) is not None for arm in pilot.DEV_ARMS)
    assert _main(["--fake", "--data", str(tmp_path), "--only", "2", "run"])[1].count("ran ") == 0  # resumes


def test_the_store_rebuilds_only_anjo_sleeps_own_types(tmp_path: Path) -> None:
    from anjo.sleep.mind.apd import store

    made = fakes.fake_life(generate.development_specs()[1])
    assert store.loads(store.dumps(made.problems), type(made.problems)) == made.problems
    for forged in ({"$type": "subprocess:Popen", "fields": {}}, {"$type": "anjo.sleep.mind.apd.types:os", "fields": {}},
                   {"$other": 1}):
        with pytest.raises(store.StoreError):
            store.decode(forged)
    data = DataDir(tmp_path)
    data.save_life(made.life, made.problems)
    with pytest.raises(store.StoreError):
        data.life("../etc/passwd")


def test_intervention_test_runs_each_snapshot_edit_once_and_reports(tmp_path: Path) -> None:
    for command in (["generate"], ["run"]):
        assert _main(["--fake", "--data", str(tmp_path), "--only", "2", *command])[0] == 0
    code, text = _main(["--fake", "--data", str(tmp_path), "--parallel", "4", "intervene"])
    assert code == 0 and "the intervention test:" in text
    stored = sorted(path.stem for path in (tmp_path / "interventions").glob("*.json"))
    assert len(stored) == INTERVENTION_SNAPSHOTS * len(EDIT_KINDS)
    again = _main(["--fake", "--data", str(tmp_path), "intervene"])[1]
    assert "intervention 0" not in again  # every snapshot-edit was kept: none runs twice
    found = intervene.on_disk(DataDir(tmp_path))
    assert found is not None
    section = apd_report.intervention_section(*found)
    assert section.startswith("## Intervention test") and all(f"| {kind} |" in section for kind in EDIT_KINDS)
    with pytest.raises(StoreError):  # the writer reads a development life's problem file, never a held-out one
        made = fakes.fake_life(generate.draw_held_out("guard")[0])
        DataDir(tmp_path).save_life(made.life, made.problems)
        DataDir(tmp_path).development_problems(made.life.id)


def test_timing_measures_throughput_and_projects_against_the_cap_and_the_line(tmp_path: Path) -> None:
    assert _main(["--fake", "--data", str(tmp_path), "--only", "2", "generate"])[0] == 0
    code, text = _main(["--fake", "--data", str(tmp_path), "--parallel", "4", "timing"])
    assert code == 0 and "Timed at --parallel 4:" in text
    assert "thinker calls per minute" in text and "CU per life-arm night" in text and "per MIND night" in text
    for lives in (HELD_OUT_LIVES, HELD_OUT_LIVES_CUT):
        for control in ("with", "without"):
            for replay in ("with", "without"):
                line = next(line for line in text.splitlines()
                            if line.startswith(f"N={lives}, {control} M-control, {replay} 17b:"))
                assert "the 100 CU cap" in line and "the 300-unit line" in line and "baseten_mind_usd cap" in line
    measured = timing.load(DataDir(tmp_path))
    assert measured is not None and measured.parallel == 4 and measured.thinker_calls > 0 and measured.spark_calls > 0
    ledger = Ledger(tmp_path / "fake-budget.jsonl")
    jobs = {str(event["job"]) for event in read_records(ledger.path, PLAIN) if event["op"] == "reserve"}
    starts = {job.rsplit(":spark:", 1)[0] for job in jobs if job.startswith("apd:timing:branches:")}
    assert len(starts) > 1  # 17b's Spark Open held one start state at a time, never all its branches at once
    assert all(job.startswith(MIND_KINDS) for job in jobs) and ledger.open_holds() == ()  # MIND's kinds, all settled
    assert measured.branches_s > 0  # the branches ran: their world query was dated on their start state's night
    plan_text = _main([*LIVE, "--data", str(tmp_path), "run"])[1]
    assert "expected about $" in plan_text and "expected about" in plan_text.split("Colab:")[1].splitlines()[0]
