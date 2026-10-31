"""MIND's records as arm 4M's lines (``mind.records``): her findings, then the questions she
noticed that have none, ids kept and bounded. Pure: a folded workspace in, lines out. Fakes only."""
from __future__ import annotations

from datetime import datetime, timezone

from anjo.sleep.mind import records
from anjo.sleep.mind.inquiry.types import Finding, Question, Workspace

AT = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)


def question(qid: str, *, subject: str = "the lens plan against the kitchen") -> Question:
    return Question(id=qid, target_ids=("t-scene4",), kind="contradiction", origin_evidence=("e-1",),
                    status="open", opened_at=AT, subject=subject)


def finding(fid: str, qid: str, *, statement: str = "the lens plan will not fit") -> Finding:
    return Finding(id=fid, question_id=qid, statement=statement, p=0.3, cites=("e-1",),
                   kind="retraction", hypothesis_id=None)


def test_a_person_without_records_serves_no_lines() -> None:
    assert records.notes(Workspace()) == ()


def test_findings_come_first_and_a_question_with_one_is_not_repeated() -> None:
    workspace = Workspace(questions=(question("q1"), question("q2", subject="the ferry cut")),
                          findings=(finding("f1", "q1"),))
    assert records.notes(workspace) == (
        "- [f1] (retraction) the lens plan will not fit",
        "- [q2] (noticed) the ferry cut",
    )


def test_records_are_bounded_to_the_notes_the_arm_may_carry() -> None:
    workspace = Workspace(questions=tuple(question(f"q{n}") for n in range(records.NOTES_MAX + 5)))
    lines = records.notes(workspace)
    assert len(lines) == records.NOTES_MAX and lines[0] == "- [q0] (noticed) the lens plan against the kitchen"


def test_more_findings_than_the_bound_keep_the_newest_and_leave_no_room_for_questions() -> None:
    # kernel._with_finding moves an updated finding to the end, so the last ones are the newest.
    count = records.NOTES_MAX + 5
    workspace = Workspace(
        questions=tuple(question(f"q{n}") for n in range(count)),
        findings=tuple(finding(f"f{n}", f"q{n}", statement=f"statement {n}") for n in range(count)),
    )
    lines = records.notes(workspace)
    assert len(lines) == records.NOTES_MAX
    assert lines[0] == "- [f5] (retraction) statement 5"
    assert lines[-1] == f"- [f{count - 1}] (retraction) statement {count - 1}"
    assert all("(noticed)" not in line for line in lines)


def test_questions_fill_only_the_room_the_findings_leave() -> None:
    workspace = Workspace(
        questions=tuple(question(f"q{n}", subject=f"subject {n}") for n in range(records.NOTES_MAX + 2)),
        findings=(finding("f1", "q0"), finding("f2", "q1")),
    )
    lines = records.notes(workspace)
    assert len(lines) == records.NOTES_MAX
    assert lines[:2] == ("- [f1] (retraction) the lens plan will not fit",
                         "- [f2] (retraction) the lens plan will not fit")
    assert lines[2] == "- [q2] (noticed) subject 2"
    assert lines[-1] == f"- [q{records.NOTES_MAX - 1}] (noticed) subject {records.NOTES_MAX - 1}"
