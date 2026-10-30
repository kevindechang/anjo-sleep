"""Type-2 sensitivity and calibration over synthetic sequences, plus the report's offline CLI.

Fakes only: confidence records are built by hand where the question is the arithmetic, and the CLI
reads a temporary results tree with a plaintext workspace log and one night folder. No network, no
model, no clock.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.log import JsonlWorkspaceLog
from anjo.sleep.mind.metacog import ConfidenceRecord
from anjo.sleep.mind.points import (
    DecisionPoint,
    RawSnapshot,
    point_record,
)
from anjo.sleep.reports import metacog
from anjo.sleep.reports.results import ResultsError
from anjo.sleep.tests.inquiry_fakes import (
    PERSON,
    Ws,
    commitment,
    day,
    ev,
    hypothesize,
    night,
    notice_op,
)
from anjo.sleep.types import Choice, ChoiceContext, Considered, NightAllowance, Policy, Want

SCENE = "t-scene4"
TARGET = commitment(SCENE, "Scene 4 shoot", due=None)


# ── The measures on sequences whose answer is known ──────────────────────────


def test_an_informative_confidence_ranks_at_one_and_a_coin_at_a_half() -> None:
    informed = [(0.9, True)] * 20 + [(0.1, False)] * 20
    assert metacog.auroc2(informed) == 1.0
    assert metacog.auroc2([(0.5, index % 2 == 0) for index in range(40)]) == 0.5  # all ties
    backwards = [(0.2, True), (0.8, False), (0.4, True), (0.6, False)]
    assert metacog.auroc2(backwards) == 0.0  # an anti-informative confidence ranks every miss above every hit
    assert metacog.auroc2([(0.9, True)]) is None  # one class is not a perfect score


def test_brier_is_zero_for_certainty_and_names_the_gap() -> None:
    assert metacog.brier([(1.0, True), (0.0, False)]) == 0.0
    assert metacog.brier([(0.9, True), (0.9, False)]) == pytest.approx(0.41)
    assert metacog.brier([]) is None


def test_bins_cover_the_unit_interval_and_include_one() -> None:
    found = metacog.bins([(0.0, False), (0.55, True), (1.0, True)], count=10)
    assert len(found) == 10 and found[0].n == 1 and found[0].rate == 0.0
    assert found[5].n == 1 and found[5].confidence == pytest.approx(0.55)
    assert found[-1].n == 1 and found[-1].rate == 1.0  # 1.0 belongs to the last bin, not a missing one
    assert sum(b.n for b in found) == 3
    assert found[1].rate is None


def test_sensitivity_says_insufficient_and_why() -> None:
    few = [(0.9, True)] * 3 + [(0.1, False)] * 2
    assert metacog.sensitivity(few).why == f"5 resolved < {metacog.TRIALS_MIN} trials"
    one_sided = [(0.9, index % 2 == 0) for index in range(40)]
    why = metacog.sensitivity(one_sided).why
    assert why is not None and "on a side of" in why


def test_a_confidence_that_tracks_the_outcome_has_a_positive_ratio() -> None:
    pairs = [(0.9, True)] * 20 + [(0.1, False)] * 20
    found = metacog.sensitivity(pairs)
    assert found.why is None
    assert found.meta_d is not None and found.d_prime is not None
    assert found.meta_d > 0 and found.d_prime > 0
    assert found.ratio is not None and found.ratio > 1.0


# ── Groups, calibration and the words ────────────────────────────────────────


def _record(night_number: int, *, raw: float, calibrated: float, outcome: bool | None,
            kind: str = "decision") -> ConfidenceRecord:
    return ConfidenceRecord(
        night=day(night_number), kind=kind, ref=f"w-{night_number}", question_id="q1", target_id=SCENE,  # type: ignore[arg-type]
        evidence=2, entropy=0.5, raw=raw, calibrated=calibrated, calibrator="raw" if raw == calibrated else "platt",
        trials=0, outcome=outcome, resolved_on=None if outcome is None else day(night_number),
    )


def test_calibration_lowers_brier_for_an_overconfident_heuristic() -> None:
    recs = [_record(index, raw=0.9, calibrated=0.7, outcome=index < 28) for index in range(40)]
    found = metacog.group("decision", recs)
    assert found.resolved == 40
    assert found.raw.brier is not None and found.calibrated.brier is not None
    assert found.calibrated.brier < found.raw.brier
    assert found.gain is not None and found.gain > 0


def test_unresolved_records_are_counted_but_never_scored() -> None:
    recs = [_record(0, raw=0.8, calibrated=0.8, outcome=True),
            _record(1, raw=0.8, calibrated=0.8, outcome=None)]
    report = metacog.report(recs, user=PERSON)
    assert (report.overall.n, report.overall.resolved, report.unresolved) == (2, 1, 1)
    assert (report.first_night, report.last_night) == (day(0), day(1))
    text = metacog.render(report)
    assert "2 second-order confidences, 1 resolved, 1 unresolved" in text
    assert "insufficient:" in text
    assert PERSON in text


def test_the_report_scores_each_kind_apart_from_the_pool() -> None:
    recs = (
        _record(0, raw=0.9, calibrated=0.9, outcome=True, kind="decision"),
        _record(0, raw=0.6, calibrated=0.6, outcome=False, kind="hypothesis"),
    )
    report = metacog.report(recs, user=PERSON)
    kinds = {g.label: g for g in report.kinds}
    assert set(kinds) == set(metacog.KINDS)
    assert (kinds["decision"].resolved, kinds["hypothesis"].resolved, kinds["prediction"].resolved) == (1, 1, 0)
    assert report.overall.resolved == 2


# ── The CLI over a results tree ─────────────────────────────────────────────


def _events() -> tuple[it.WorkspaceEvent, ...]:
    item = ev("e-1", "Scene 4 wants an 85 mm shot.", n=0, target=SCENE)
    w = Ws()
    w.start(0)
    w.arrive(item)
    w.ok(notice_op([item.id], kind="gap"))
    w.ok(hypothesize("q1", 0.5, 0.3))
    return tuple(w.events)


def _point(event_night: date) -> DecisionPoint:
    want = Want(id="w-1", kind="understand", target_id=SCENE, action="inquire", priority_class="P3",
                strength=0.8, question_id="q1")
    return DecisionPoint(
        night=event_night, phase="pre_study",
        raw=RawSnapshot(targets=(TARGET,), results=(), as_of=night(0), last_message_at=None),
        ctx=ChoiceContext(
            policy=Policy(("P3",), 8, 0.2, 0.3, frozenset()),
            allowance=NightAllowance(actions_left=1, calls_left=3, usd_left=1.0),
        ),
        choice=Choice(opportunity="away", want=want,
                      considered=(Considered("w-1", "understand", SCENE, "inquire", 0.8, 1.0, "won", None),)),
    )


def _tree(root: Path, *, pick: str = "ran", winner: str = "w-1") -> Path:
    base = root / PERSON
    base.mkdir(parents=True)
    log = JsonlWorkspaceLog(base / metacog.WORKSPACE, PERSON, codec=PLAIN, guard=NO_GUARD)
    for event in _events():
        log.append(event)
    folder = base / metacog.NIGHTS / day(0).isoformat()
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"night": day(0).isoformat(), "label": "development"}))
    (folder / "mind_points.jsonl").write_text(json.dumps(point_record(_point(day(0)))) + "\n")
    (folder / "log.jsonl").write_text(json.dumps({
        "step": "choose", "status": "ok", "uses": "new", "system": None, "noop": False,
        "receipt": {"opportunity": "away", "winner": winner, "question_id": "q1",
                    "outcome": pick, "outcome_reason": None, "candidates": []},
    }) + "\n")
    return root


def test_the_cli_reads_the_tree_and_prints_the_report(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results")
    lines: list[str] = []
    assert metacog.main(["--user", PERSON, "--root", str(root)], out=lines.append) == 0
    text = "\n".join(lines)
    assert "4 second-order confidences, 1 resolved, 3 unresolved" in text
    assert "| decision | 1 | 1 |" in text
    assert f"insufficient: 1 resolved < {metacog.TRIALS_MIN} trials" in text


def test_the_cli_writes_a_file_when_asked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _tree(tmp_path / "results")
    out = tmp_path / "metacog.md"
    assert metacog.main(["--user", PERSON, "--root", str(root), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Metacognition over her nights")
    assert "wrote" in capsys.readouterr().out


def test_night_facts_carry_the_log_views_and_the_pre_study_pick(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results")
    facts = metacog.night_facts(root, PERSON)
    assert len(facts) == 1
    one = facts[0]
    assert (one.night, one.logged, one.choice is not None) == (day(0), True, True)
    assert one.after.question("q1") is not None and one.at == night(0)
    assert one.pick == "ran"  # the choose receipt says the inquiry ran
    assert metacog.load(root, PERSON)


def test_a_pick_that_never_ran_has_no_outcome_to_score(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results", pick="budget")
    assert metacog.night_facts(root, PERSON)[0].pick == "budget"
    decisions = [record for record in metacog.load(root, PERSON) if record.kind == "decision"]
    assert decisions and all(not record.resolved for record in decisions)


def test_a_receipt_for_another_winner_is_refused(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results", winner="w-other")
    with pytest.raises(ResultsError, match="the choose receipt names"):
        metacog.night_facts(root, PERSON)


def test_a_missing_or_unreadable_tree_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ResultsError):
        metacog.night_facts(tmp_path, PERSON)
    base = tmp_path / PERSON
    base.mkdir()
    (base / metacog.WORKSPACE).write_text("enc1:a-ciphertext\n", encoding="utf-8")
    (base / metacog.NIGHTS).mkdir()
    with pytest.raises(ResultsError, match="plaintext workspace log"):
        metacog.night_facts(tmp_path, PERSON)


def test_the_report_makes_no_claim_about_her_state_reporting_itself() -> None:
    text = metacog.CLAIM.casefold()
    assert "never against a report" in text
    for word in ("conscious", "sentient", "first-person"):
        assert word not in text


def test_the_root_defaults_to_the_results_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    base = tmp_path / "sleep" / "results" / PERSON
    base.mkdir(parents=True)
    (base / metacog.WORKSPACE).write_text("", encoding="utf-8")
    (base / metacog.NIGHTS).mkdir()
    assert metacog.night_facts(None, PERSON) == ()
    with pytest.raises(ResultsError):
        metacog.night_facts(None, "life-missing")


def test_a_summary_without_points_is_not_a_scored_night(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results")
    folder = root / PERSON / metacog.NIGHTS / "2026-10-02"
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps({"night": "2026-10-02"}), encoding="utf-8")
    assert [facts.night for facts in metacog.night_facts(root, PERSON)] == [day(0)]
