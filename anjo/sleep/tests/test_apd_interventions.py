# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The intervention test and kill-resume (Task 17-eval): an edited snapshot and its unedited twin
resume on the same evidence, each on its own log; a killed run rebuilds from its log alone, with
the digest the live run logged, and resumes."""
from __future__ import annotations

import dataclasses
import json
import signal
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from anjo.sleep.mind.apd import arms, interventions, resume
from anjo.sleep.mind.apd import types as apd
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.ports import InquiryPorts
from anjo.sleep.tests import apd_eval_fakes as fx


def _m_run(life: apd.Life, reply: fx.Reply = fx.mind_reply) -> tuple[apd.ArmRun, fx.FakeCore, apd.ArmModels]:
    fake = fx.FakeCore()
    models = fx.models(fx.ScriptThinker(reply))
    return arms.run_arm("M", life, models, core=fake.core()), fake, models


def _first(events: Sequence[it.WorkspaceEvent], kind: type) -> int:
    return next(n for n, e in enumerate(events) if isinstance(e.step.entry, kind))


@dataclass
class Wrong:
    calls: list[tuple[str, int]] = field(default_factory=list)

    def write(self, question: it.Question, hypotheses: Sequence[it.Hypothesis]) -> str:
        self.calls.append((question.id, len(hypotheses)))
        return "The plan lacks a lens rental"


# ── The spec's test ───────────────────────────────────────────────────────────


def test_interventions_hold_evidence_fixed() -> None:
    life = fx.life(2)
    whole, fake, models = _m_run(life)
    snapshot = whole.events[: _first(whole.events, it.InvestigateOp)]  # after PREDICT, before the search
    point = resume.resume_point(snapshot)
    start = point.start
    question = point.focus
    assert question is not None and start.call == "inquiry"
    ports = arms.mind_ports(models, life, arms.MemoryLog(), arms.load_brief())
    edit = it.Edit("swap_lead", question, hypothesis_id=f"{question}.h2")
    fake.calls.clear()
    fake.searches.clear()

    resumed = interventions.resume_after_edit(
        snapshot, edit, ports, point.budget, night=start.night, targets=life.targets[start.night], as_of=start.as_of,
        core=fake.core(),
    )
    unedited, edited = resumed.unedited, resumed.edited

    first, second = (c for c in fake.calls if c["call"] == "inquiry")
    for name in ("night", "as_of", "targets", "budget", "focus", "rules", "control"):
        assert first[name] == second[name], name
    assert first["as_of"] == start.as_of and first["focus"] == question
    a, b = first["ports"], second["ports"]
    assert isinstance(a, InquiryPorts) and isinstance(b, InquiryPorts)
    assert a.search is b.search is ports.search and a.resolver is b.resolver  # the same evidence, held fixed
    assert a.log is not b.log and ports.log.events() == ()  # each branch on its own fresh log
    assert a.log.events() == (*snapshot, *unedited.events)
    edit_event = b.log.events()[len(snapshot)]
    assert edit_event.step.entry == edit and edit_event.step.at == start.as_of
    assert b.log.events() == (*snapshot, edit_event, *edited.events)
    assert fx.fold(b.log.events()) == edited.ws  # the edited branch is a fold of its own log
    assert fx.fold(b.log.events()[: len(snapshot) + 1]) == resumed.after_edit
    (q1, at1, found1), (q2, at2, found2) = fake.searches
    assert (q1, at1, found1) == (q2, at2, found2) and at1 == start.as_of
    assert found1 and all(e.day <= start.night for e in life.evidence if e.id in found1)
    concluded = {run.ws.finding_of(question).hypothesis_id for run in (unedited, edited)}  # type: ignore[union-attr]
    assert concluded == {f"{question}.h1", f"{question}.h2"}  # her state, not the evidence, moved the conclusion


# ── The harness ───────────────────────────────────────────────────────────────


def test_snapshots_are_mid_inquiry_with_a_live_hypothesized_focus() -> None:
    life = fx.life(3)
    whole, fake, _ = _m_run(life)
    core = fake.core()
    snapshots = interventions.draw_snapshots([(life, whole)], core, seed=3, count=4)
    assert len(snapshots) == 4 and snapshots == interventions.draw_snapshots([(life, whole)], core, seed=3, count=4)
    for snap in snapshots:
        ws = fx.fold(snap.events)
        question = ws.question(snap.question_id)
        assert question is not None and question.status in it.LIVE_STATUSES and ws.hypotheses_of(question.id)
        last, following = snap.events[-1].step, whole.events[len(snap.events)].step
        assert last.cycle is not None and not isinstance(following.entry, it.RunStart)  # the run goes on
    with pytest.raises(ValueError, match="development"):
        interventions.draw_snapshots([(fx.life(3, split="held_out"), whole)], core, seed=3)


def test_three_edits_per_snapshot_and_skips_are_recorded() -> None:
    life = fx.life(1)
    whole, fake, models = _m_run(life)
    hypothesized = interventions.Snapshot(life.id, whole.events[: _first(whole.events, it.PredictOp)], "q2")
    predicted = interventions.Snapshot(life.id, whole.events[: _first(whole.events, it.InvestigateOp)], "q2")
    wrong = Wrong()
    results, skipped = interventions.intervention_test(
        (hypothesized, predicted), {life.id: life}, models, wrong, core=fake.core(),
    )
    assert [(r.at, r.edit.kind) for r in results] == [
        (len(hypothesized.events), "swap_lead"), (len(hypothesized.events), "inject_hypothesis"),
        (len(predicted.events), "swap_lead"), (len(predicted.events), "delete_prediction"),
        (len(predicted.events), "inject_hypothesis"),
    ]
    assert skipped == (interventions.Skipped(life.id, len(hypothesized.events), "delete_prediction",
                                             "no_pending_prediction"),)
    assert wrong.calls == [("q2", 3), ("q2", 3)]
    measured = {(r.at, r.edit.kind): r.measures for r in results}
    swap = results[0]
    assert swap.edit.hypothesis_id == "q2.h2" and swap.conclusion_differs
    assert swap.unedited.conclusion == ("q2.h1", "resolution") and swap.edited.conclusion == ("q2.h2", "resolution")
    # Her next test is of the new lead after the swap, and of the old one without it.
    assert swap.measures == (interventions.Measure("acts_on_new_lead", True, False),)
    late_swap = measured[len(predicted.events), "swap_lead"]
    assert late_swap == (interventions.Measure("acts_on_new_lead", False, False),)  # nothing left to test: no credit
    deleted = results[3]
    assert deleted.measures == (
        interventions.Measure("deleted_not_resolved", True, True),
        interventions.Measure("replaced_or_moved_on", True, False),  # she commits a new test; unedited, she needn't
    )
    assert json.loads(deleted.edited.ops[0])["op"] == "PREDICT" and json.loads(deleted.unedited.ops[0])["op"] == "INVESTIGATE"
    assert measured[len(hypothesized.events), "inject_hypothesis"] == (
        interventions.Measure("injected_tested", True, None),  # the injected hypothesis leads at 0.5, so it is tested
        interventions.Measure("injected_dropped", False, None),  # no evidence moved it here
    )


def test_injection_skips_a_question_that_already_holds_four() -> None:
    def four(prompt: str) -> str:
        reply = json.loads(fx.mind_reply(prompt)) if "\nCYCLE " in prompt else None
        if reply is None or reply["op"] != "HYPOTHESIZE":
            return fx.mind_reply(prompt)
        reply["hypotheses"] = [{"statement": f"alternative {n}", "credence": 0.4 - 0.1 * n} for n in range(4)]
        return json.dumps(reply)

    life = fx.life(1)
    whole, fake, models = _m_run(life, four)
    snapshot = interventions.Snapshot(life.id, whole.events[: _first(whole.events, it.HypothesizeOp) + 1], "q2")
    assert len(fx.fold(snapshot.events).hypotheses_of("q2")) == 5
    results, skipped = interventions.intervention_test((snapshot,), {life.id: life}, models, Wrong(), core=fake.core())
    assert [r.edit.kind for r in results] == ["swap_lead"]
    assert {(s.kind, s.reason) for s in skipped} == {
        ("inject_hypothesis", "four_alternatives"), ("delete_prediction", "no_pending_prediction"),
    }


def test_an_injected_hypothesis_is_tested_then_dropped_on_evidence() -> None:
    def doubts_the_injected(prompt: str) -> str:
        if "\nCYCLE " not in prompt:
            return fx.mind_reply(prompt)
        question = prompt.split("FOCUS ", 1)[1].split("\n", 1)[0]
        ids = prompt.split("ids: ", 1)[1].split("\n", 1)[0].split(",")
        injected = [h for h in ids if ".x" in h]
        if injected and "searched: yes" in prompt and "updates: 0" in prompt:
            return json.dumps({"op": "ASSESS", "question_id": question, "evidence_id": "e-01-2",
                               "likelihoods": {h: 0.25 if h in injected else 1.0 for h in ids}})
        return fx.mind_reply(prompt)

    life = fx.life(1)
    whole, fake, models = _m_run(life, doubts_the_injected)
    snapshot = interventions.Snapshot(life.id, whole.events[: _first(whole.events, it.HypothesizeOp) + 1], "q2")
    results, _ = interventions.intervention_test((snapshot,), {life.id: life}, models, Wrong(), core=fake.core())
    injected = next(r for r in results if r.edit.kind == "inject_hypothesis")
    assert injected.measures == (
        interventions.Measure("injected_tested", True, None), interventions.Measure("injected_dropped", True, None),
    )
    assert injected.edited.conclusion == ("q2.h1", "resolution") and not injected.conclusion_differs
    assert any(json.loads(op)["op"] == "ASSESS" for op in injected.edited.ops)


def test_aims_at_reads_which_hypothesis_an_operation_singles_out() -> None:
    def predict(values: dict[str, float]) -> it.PredictOp:
        return it.PredictOp("q1", "claim", values, 2)

    assert interventions.aims_at(predict({"h1": 0.9, "h2": 0.2, "else": 0.5}), "h1")
    assert not interventions.aims_at(predict({"h1": 0.9, "h2": 0.2, "else": 0.5}), "h2")  # a prediction tests its top
    assert not interventions.aims_at(predict({"h1": 0.9, "h2": 0.9, "else": 0.5}), "h1")  # shared: not singled out
    against = it.AssessOp("q1", "e1", likelihoods={"h1": 1.0, "h2": 0.25, "else": 1.0})
    assert interventions.aims_at(against, "h2") and not interventions.aims_at(against, "h1")  # for or against
    assert interventions.aims_at(it.AssessOp("q1", "e1", credences={"h1": 0.7, "h2": 0.2, "else": 0.1}), "h1")
    assert not interventions.aims_at(it.ParkOp("q1"), "h1")


# ── Kill-resume ───────────────────────────────────────────────────────────────


def test_resume_point_rebuilds_budget_and_focus_from_the_log() -> None:
    life = fx.life(1)
    whole, _, _ = _m_run(life)
    upto = _first(whole.events, it.InvestigateOp) + 2  # the INVESTIGATE and its search
    point = resume.resume_point(whole.events[:upto])
    assert point.start.call == "inquiry" and point.focus == "q2"
    notice_tokens = point.start.budget.tokens_used
    assert point.budget == it.Budget(cycles_used=3, tokens_used=notice_tokens + 300)
    with pytest.raises(ValueError, match="RunStart"):
        resume.resume_point(())


SETUP = "anjo.sleep.tests.apd_eval_fakes:kill_setup"


def test_kill_resume_digest_identical(tmp_path: Path) -> None:
    life = fx.life(3, split="held_out")
    whole, _, _ = _m_run(life)
    night2 = next(n for n, e in enumerate(whole.events)
                  if isinstance(e.step.entry, it.RunStart) and e.step.entry.call == "inquiry" and e.step.entry.night == life.days[1])
    for kill_at in (night2 + 2, 0):  # mid-inquiry, and in the first notice pass
        result = resume.kill_resume(SETUP, life.id, kill_at, workdir=tmp_path)
        assert result.digest_equal and result.conclusions_agree, kill_at
        assert result.resumed == result.uninterrupted and len(result.resumed) == 3
        logged = fx.DiskLog(tmp_path / f"{life.id}.kill-{kill_at}.log").events()
        assert len(logged) > kill_at + 1  # the fresh process resumed on the killed run's own file
        assert sum(isinstance(e.step.entry, it.RunStart) for e in logged) == sum(
            isinstance(e.step.entry, it.RunStart) for e in whole.events) + 1  # one RunStart more: the resume's
    with pytest.raises(ValueError, match="fresh log"):
        resume.kill_resume(SETUP, life.id, 0, workdir=tmp_path)
    with pytest.raises(ValueError, match="before operation"):
        resume.kill_resume(SETUP, life.id, len(whole.events) + 5, workdir=tmp_path)


def test_the_kill_is_a_real_process_kill_and_the_rebuild_reads_only_the_file(tmp_path: Path) -> None:
    path = tmp_path / "life-1.log"
    killed = resume._child("kill", SETUP, "life-1", path, 60, "--kill-at", "5")
    assert killed.returncode == -signal.SIGKILL
    assert [e.step.seq for e in fx.DiskLog(path).events()] == list(range(6))  # durable through the kill, no further
    tampered = tmp_path / "tampered.log"
    events = fx.DiskLog(path).events()
    log = fx.DiskLog(tampered)
    for event in (*events[:-1], dataclasses.replace(events[-1], after="0" * 64)):
        log.append(event)
    rebuilt = resume.rebuild(fx.kill_setup("life-1"), tampered)
    assert rebuilt == {"digest_equal": False, "conclusions": None}  # a digest that doesn't hold resumes nothing


def test_killing_log_kills_once_its_event_is_appended() -> None:
    whole, _, _ = _m_run(fx.life(1))
    kills: list[int] = []
    inner = arms.MemoryLog()
    log = resume.KillingLog(inner, 2, kill=lambda: kills.append(len(inner.events())))
    for event in whole.events[:4]:
        log.append(event)
    assert kills == [3] and log.events() == whole.events[:4]  # killed right after event 2 was logged
