"""What the thinker sees: the rendered workspace (Task 17). Pure, but for reading its two fixed
instruction files once, at import.

``render`` is an inquiry cycle's whole input: ``think_v1.md``, then the focal question, its
hypotheses and credences (labeled heuristic), its open and settled predictions, its finding, the
answers its world queries got, the evidence items the loop fetched by id, and the last
``RECENT_OPS`` operations. ``render_notice`` is the notice pass's: ``notice_v1.md``, the targets,
the questions she holds, earlier items and the day's new ones. Neither takes the conversation, so
neither can show it: evidence items are what extraction kept, by id. The output depends only on
its arguments, so a resumed run renders the same.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from anjo.sleep.mind.inquiry.kernel import hits, used_items
from anjo.sleep.mind.inquiry.types import (
    LIVE_STATUSES,
    OP_CLASSES,
    RECENT_OPS,
    Answered,
    AssessOp,
    ConcludeOp,
    InvestigateOp,
    NoticeOp,
    Prediction,
    Step,
    Workspace,
)
from anjo.sleep.types import Evidence, Target

_PROMPTS: Final = Path(__file__).parent / "prompts"
THINK: Final = (_PROMPTS / "think_v1.md").read_text()
NOTICE: Final = (_PROMPTS / "notice_v1.md").read_text()
ANSWERS_SHOWN: Final = 3


def _line(text: str) -> str:
    return " ".join(text.split())


def _item(item: Evidence, label: str | None = None) -> str:
    tag = f", {label}" if label else ""
    return f"- [{item.id}] {item.day.isoformat()}{tag}: {_line(item.text)}"


def _p_values(prediction: Prediction) -> str:
    return ", ".join(f"{h} {p:.2f}" for h, p in prediction.p_true_given.items())


def _prediction(prediction: Prediction) -> str:
    when = f"committed {prediction.committed_at.date().isoformat()}, horizon {prediction.horizon.date().isoformat()}"
    state = "pending" if prediction.pending else f"resolved {prediction.outcome}"
    return f'- {prediction.id} ({state}; {when}): "{_line(prediction.claim)}"; P(true | h): {_p_values(prediction)}'


def _op_summary(step: Step) -> str:
    entry = step.entry
    if entry is None:
        return f"- #{step.seq} (no operation: {step.code})"
    if not isinstance(entry, OP_CLASSES):
        return ""
    detail = ""
    if isinstance(entry, InvestigateOp):
        detail = f' {entry.tool} "{_line(entry.query)}"'
    elif isinstance(entry, AssessOp):
        detail = f" {entry.evidence_id}"
    elif isinstance(entry, ConcludeOp):
        detail = f" {entry.kind} {entry.hypothesis_id}"
    where = "" if isinstance(entry, NoticeOp) else f" {entry.question_id}"
    outcome = "applied" if step.code is None else f"rejected: {step.code}"
    return f"- #{step.seq} {entry.name}{where}{detail} ({outcome})"


def _recent(ws: Workspace) -> list[str]:
    ops = [step for step in ws.history if step.entry is None or isinstance(step.entry, OP_CLASSES)]
    return [_op_summary(step) for step in ops[-RECENT_OPS:]] or ["(none yet)"]


def _evidence(ws: Workspace, focus: str, evidence: Sequence[Evidence]) -> list[str]:
    question = ws.question(focus)
    origin = frozenset(question.origin_evidence) if question is not None else frozenset()
    returned = frozenset(item.id for item in hits(ws, focus))
    used = used_items(ws, focus)

    def label(item_id: str) -> str | None:
        if item_id in origin:
            return "raised the question"
        if item_id in used:
            return "assessed"
        return "search hit, not yet assessed" if item_id in returned else None

    return [_item(item, label(item.id)) for item in evidence] or ["(none fetched)"]


def _answers(ws: Workspace, focus: str) -> list[str]:
    answers = [
        step.entry for step in ws.history if isinstance(step.entry, Answered) and step.entry.question_id == focus
    ]
    return [
        f'- "{_line(a.query)}": {_line(a.answer) or "(no answer)"}' for a in answers[-ANSWERS_SHOWN:]
    ] or ["(none)"]


def render(ws: Workspace, focus: str, evidence: Sequence[Evidence]) -> str:
    """One inquiry cycle's whole input, for the question ``focus``."""
    question = ws.question(focus)
    if question is None:
        raise ValueError(f"question {focus} isn't held")
    hypotheses = [
        f"- {h.id}: {h.credence:.2f}  {_line(h.statement)}" for h in ws.hypotheses_of(focus)
    ] or ["(none yet: HYPOTHESIZE first)"]
    predictions = [p for p in ws.predictions if p.question_id == focus]
    finding = ws.finding_of(focus)
    reported = (
        [f"- {finding.id} ({finding.kind}, p {finding.p:.2f}): {_line(finding.statement)}"] if finding else ["(none)"]
    )
    sections = [
        THINK.rstrip(),
        "## Your question",
        f"{question.id} ({question.kind}, {question.status}, opened {question.opened_at.date().isoformat()}): "
        f"{_line(question.subject)}",
        f"Targets: {', '.join(question.target_ids)}. Raised by: {', '.join(question.origin_evidence)}.",
        "## Hypotheses (credences are heuristic)", *hypotheses,
        "## Open predictions", *([_prediction(p) for p in predictions if p.pending] or ["(none)"]),
        "## Settled predictions", *([_prediction(p) for p in predictions if not p.pending] or ["(none)"]),
        "## Your finding", *reported,
        "## Answers to your world queries (never evidence)", *_answers(ws, focus),
        "## Evidence", *_evidence(ws, focus, evidence),
        "## Your last operations", *_recent(ws),
        "Reply with one JSON operation.",
    ]
    return "\n".join(sections) + "\n"


def render_notice(
    ws: Workspace, new: Sequence[Evidence], earlier: Sequence[Evidence], targets: Sequence[Target],
) -> str:
    """The notice pass's whole input: the day's ``new`` items and ``earlier`` ones she holds."""
    shown_targets = [
        f"- {t.id} ({t.kind}{', due ' + t.check_after.date().isoformat() if t.check_after else ''}): {_line(t.label)}"
        for t in targets if t.status == "open"
    ] or ["(none)"]
    questions = [
        f"- {q.id} ({q.kind}, {q.status}; {', '.join(q.target_ids)}): {_line(q.subject)}"
        for q in ws.questions if q.status in LIVE_STATUSES or q.status == "parked"
    ] or ["(none)"]
    sections = [
        NOTICE.rstrip(),
        "## Open targets", *shown_targets,
        "## Questions you hold", *questions,
        "## Earlier evidence you hold", *([_item(item) for item in earlier] or ["(none)"]),
        "## Today's new evidence", *([_item(item) for item in new] or ["(none)"]),
        "Reply with a JSON array of NOTICE operations.",
    ]
    return "\n".join(sections) + "\n"
