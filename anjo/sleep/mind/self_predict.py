"""Sleep → MIND: her own prediction about a night's practice, committed before it and resolved after.

The loop's second arrow (owner decision 2026-10-31). For each focus target (``loop_focus``), before
practice she opens a question about her own learning on it — *does practising <label> make my own
replies hold up?* — with two alternatives, improves and no change, and commits a prediction through
the kernel's normal path: *practising <label> holds my own replies up in more of tonight's focus
scenes than on the last night that practised it* (or, with no earlier night, than in tonight's other
scenes). Every operation goes through ``kernel.apply`` and is appended to her workspace log, so her
credences and the timeline are the kernel's own, never a second store.

**The outcome is code, not a model.** ``record`` counts, from the night's practice transcripts, how
many of the target's focus scenes kept her own candidate — Glimmer's reply, not the teacher's, and
not one the critic listed as missing a constraint — and how many scenes there were. ``resolve``
compares that share with the baseline the claim named: the last night that practised the target with
``MIN_SCENES`` scenes or more, else tonight's other scenes. A night with fewer than ``MIN_SCENES``
scenes on the target, or nothing to compare with, is ``undetermined``. The verdict is appended as a
``Resolved`` step whose ``seen`` cites the night's practice record id (``practice:<night>``), so the
kernel's existing ``resolved_prediction`` update moves her credence.

**No leakage.** The only evidence this module names is that practice record: no probe, probe text,
score, sealed set or results score reaches it, and neither does the store. That is the morning
evaluation's business, never hers (``tests/test_mind_loop_focus.py`` holds the import line).

Pure but for the log it appends to, which the caller owns; every time is passed in on the night's
clock.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final, Literal

from anjo.sleep.mind.inquiry import kernel
from anjo.sleep.mind.inquiry.loop_mark import LOOP_DECIDER, SUBJECT_PREFIX, SUBJECT_SUFFIX, is_self_practice
from anjo.sleep.mind.inquiry.ports import WorkspaceLog
from anjo.sleep.mind.inquiry.types import (
    FULL_RULES,
    Alternative,
    Applied,
    Budget,
    EvidenceRef,
    HypothesizeOp,
    KernelRules,
    NoticeOp,
    PredictOp,
    Prediction,
    Question,
    Resolved,
    RunStart,
    Step,
    Workspace,
)
from anjo.sleep.mind.loop_focus import FocusTarget
from anjo.sleep.practice.transcripts import PracticeTranscript

Verdict = Literal["true", "false", "undetermined"]

MIN_SCENES: Final = 2  # a night decides nothing on fewer scenes than this, on the target or its baseline
HORIZON_DAYS: Final = 1  # the night it is committed for
PRIORS: Final[tuple[float, float]] = (0.5, 0.3)  # improves, no change; the catch-all takes the rest
CATCH_ALL: Final = 0.2
P_IMPROVES: Final = 0.9  # P(the claim | practising improves my replies)
P_NO_CHANGE: Final = 0.1
P_OTHER: Final = 0.5  # the catch-all, and any alternative she adds later
QUESTION_KIND: Final = "gap"
RECORD_PREFIX: Final = "practice:"  # the night's practice record id, which a resolution cites


def question_subject(label: str) -> str:
    """The question she opens about her own learning on a target."""
    return f"{SUBJECT_PREFIX}{label}{SUBJECT_SUFFIX}"


def is_self_question(question: Question) -> bool:
    """Whether ``question`` is one this module opened about her own practice."""
    return is_self_practice(question.subject)


def label_of(question: Question) -> str:
    """The target label her practice question names."""
    return question.subject[len(SUBJECT_PREFIX):-len(SUBJECT_SUFFIX)]


def is_self_prediction(ws: Workspace, prediction: Prediction) -> bool:
    """Whether ``prediction`` is a loop commitment: one on a question this module opened."""
    question = ws.question(prediction.question_id)
    return question is not None and is_self_question(question)


def improves_statement(label: str) -> str:
    return f"Practising {label} makes my own replies hold up"


def no_change_statement(label: str) -> str:
    return f"Practising {label} leaves my own replies as they are"


def claim_for(label: str, previous: date | None) -> str:
    """The prediction's words: what tonight's practice must show, against what it is compared to."""
    than = f"on {previous.isoformat()}" if previous is not None else "in tonight's other scenes"
    return f"Practising {label} holds my own replies up in more of tonight's focus scenes than {than}"


@dataclass(frozen=True)
class SceneCount:
    """The scenes one target's focus held, and how many kept her own candidate."""

    scenes: int
    held_up: int

    @property
    def share(self) -> float:
        return self.held_up / self.scenes if self.scenes else 0.0

    def record(self) -> dict[str, int]:
        return {"scenes": self.scenes, "held_up": self.held_up}


@dataclass(frozen=True)
class Record:
    """The night's practice record: its id (which a resolution cites), and what each focus target's
    scenes kept, beside the scenes that weren't focus."""

    night: date
    scenes: Mapping[str, SceneCount] = field(default_factory=dict)
    other: SceneCount = SceneCount(0, 0)

    @property
    def id(self) -> str:
        return f"{RECORD_PREFIX}{self.night.isoformat()}"

    def of(self, target_id: str) -> SceneCount:
        return self.scenes.get(target_id, SceneCount(0, 0))


@dataclass(frozen=True)
class SelfPrediction:
    """One focus target's commitment: her question, the prediction's id and words, and the earlier
    night the claim compares to (None: tonight's other scenes)."""

    target_id: str
    label: str
    question_id: str
    prediction_id: str
    claim: str
    previous: date | None


@dataclass(frozen=True)
class Commitment:
    """A night's commitments, and the targets it couldn't commit for (why, for the night log)."""

    predictions: tuple[SelfPrediction, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Outcome:
    """One prediction resolved from the night's record: the verdict, the counts it rests on, and why
    when it decided nothing."""

    target_id: str
    label: str
    question_id: str
    prediction_id: str
    claim: str
    verdict: Verdict
    scenes: SceneCount
    baseline: SceneCount | None
    against: date | None = None  # the night the baseline is, None for tonight's other scenes
    why: str | None = None

    def record(self) -> dict[str, object]:
        return {
            "target_id": self.target_id, "label": self.label, "question_id": self.question_id,
            "prediction_id": self.prediction_id, "claim": self.claim, "outcome": self.verdict,
            "scenes": self.scenes.record(),
            **({} if self.baseline is None else {"baseline": {**self.baseline.record(), "night":
                                                             None if self.against is None else self.against.isoformat()}}),
            **({} if self.why is None else {"why": self.why}),
        }


# ── The night's record, and what it decides ───────────────────────────────────


def keeps(transcript: PracticeTranscript) -> bool:
    """Whether her own candidate held up in a scene: Glimmer's reply is the target and the critic
    didn't list it as missing a constraint."""
    return transcript.target == "glimmer" and not transcript.flagged


def record(night: date, transcripts: Sequence[PracticeTranscript]) -> Record:
    """The night's practice record, from the scenes practice kept (its focus scenes and its rest)."""
    focus: dict[str, list[PracticeTranscript]] = {}
    other: list[PracticeTranscript] = []
    for transcript in transcripts:
        if transcript.focus and transcript.target_id is not None:
            focus.setdefault(transcript.target_id, []).append(transcript)
        else:
            other.append(transcript)
    return Record(
        night=night,
        scenes={target_id: SceneCount(len(scenes), sum(keeps(scene) for scene in scenes))
                for target_id, scenes in focus.items()},
        other=SceneCount(len(other), sum(keeps(scene) for scene in other)),
    )


def previous(history: Sequence[Mapping[str, object]], target_id: str) -> tuple[date, SceneCount] | None:
    """The last night's record that practised ``target_id`` with ``MIN_SCENES`` scenes or more, from
    the finished nights' summaries (newest first)."""
    for summary in reversed(history):
        loop = summary.get("mind_loop")
        if not isinstance(loop, Mapping):
            continue
        scenes = loop.get("scenes")
        night = summary.get("night")
        if not isinstance(scenes, Mapping) or not isinstance(night, str):
            continue
        counts = scenes.get(target_id)
        if not isinstance(counts, Mapping):
            continue
        found = SceneCount(int(counts.get("scenes", 0)), int(counts.get("held_up", 0)))
        if found.scenes >= MIN_SCENES:
            return date.fromisoformat(night), found
    return None


def _decide(now: SceneCount, baseline: SceneCount | None) -> tuple[Verdict, str | None]:
    """The verdict the record supports, and why nothing was decided when it decided nothing."""
    if now.scenes < MIN_SCENES:
        return "undetermined", f"too few scenes on the target (a night needs {MIN_SCENES})"
    if baseline is None:
        return "undetermined", f"nothing to compare with (fewer than {MIN_SCENES} scenes)"
    return ("true" if now.share > baseline.share else "false"), None


def receipt(
    focus: Sequence[FocusTarget], record: Record, commitment: Commitment, outcomes: Sequence[Outcome],
    *, why: str | None = None,
) -> dict[str, object]:
    """The practice step's loop record, ids and counts only: what practice drew from, what each focus
    target's scenes kept, the predictions she committed and what they resolved to. It is written to
    the night log with the step and carried in the summary as ``mind_loop``."""
    return {
        "record": record.id,
        "focus": [{"target_id": target.target_id, "label": target.label, "question_id": target.question_id}
                  for target in focus],
        "scenes": {target_id: count.record() for target_id, count in record.scenes.items()},
        "other": record.other.record(),
        "predictions": [outcome.record() for outcome in outcomes],
        **({} if why is None else {"why": why}),
        **({"skipped": [{"target_id": target_id, "why": reason} for target_id, reason in commitment.skipped]}
           if commitment.skipped else {}),
    }


# ── The kernel's path: commit before practice, resolve after ─────────────────


def _rules(ws: Workspace) -> KernelRules:
    """The rules her log's latest run recorded, so a step applies as the fold will read it."""
    return next((step.entry.rules for step in reversed(ws.history) if isinstance(step.entry, RunStart)),
                FULL_RULES)


def _append(log: WorkspaceLog, ws: Workspace, *, at: datetime, entry: Any, rules: KernelRules
            ) -> Workspace | None:
    """Apply one step and append its event; None when the kernel refuses it (the caller records why)."""
    result = kernel.apply(ws, Step(seq=ws.seq, at=at, entry=entry), rules=rules)
    if not isinstance(result, Applied):
        return None
    log.append(result.event)
    return result.ws


def _held_evidence(ws: Workspace, origin: Sequence[str]) -> tuple[str, ...]:
    """The origin ids her workspace still holds as known items, so a NOTICE may cite them."""
    known = kernel.known_items(ws)
    return tuple(item for item in origin if item in known)


def _existing(ws: Workspace, target_id: str, label: str) -> Question | None:
    """The live question she already holds about her own learning on this target, if any."""
    subject = question_subject(label)
    return next((q for q in ws.questions
                 if q.subject == subject and target_id in q.target_ids and q.status in {"open", "active"}),
                None)


def _given(ws: Workspace, question_id: str, label: str) -> dict[str, float]:
    """P(claim | each hypothesis): improves high, no change low, the catch-all and anything she adds
    about the middle. K2's spread holds whatever set she holds."""
    improves, no_change = improves_statement(label), no_change_statement(label)
    out = {}
    for hypothesis in ws.hypotheses_of(question_id):
        if hypothesis.statement == improves:
            out[hypothesis.id] = P_IMPROVES
        elif hypothesis.statement == no_change:
            out[hypothesis.id] = P_NO_CHANGE
        else:
            out[hypothesis.id] = P_OTHER
    return out


def commit(
    log: WorkspaceLog, ws: Workspace, *, night: date, at: datetime,
    focus: Sequence[tuple[str, str, str]], history: Sequence[Mapping[str, object]],
) -> Commitment:
    """Open (or find) her question on each ``(target_id, label, focus_question_id)`` and commit
    tonight's prediction on it, before practice. A target whose question can't be opened — no
    origin item her log still knows — is skipped with its reason."""
    rules = _rules(ws)
    started = _append(log, ws, at=at, rules=rules, entry=RunStart(
        night=night, as_of=at, call="inquiry", rules=rules, control="priority", focus=None, budget=Budget(),
        denylist=kernel.DENYLIST_VERSION, decider=LOOP_DECIDER))
    if started is None:  # a RunStart always applies; keep the log honest if one ever doesn't
        return Commitment(skipped=tuple((target_id, "no run start") for target_id, _, _ in focus))
    ws = started
    made: list[SelfPrediction] = []
    skipped: list[tuple[str, str]] = []
    for target_id, label, source_question in focus:
        question = _existing(ws, target_id, label)
        if question is None:
            source = ws.question(source_question)
            origin = _held_evidence(ws, () if source is None else source.origin_evidence)
            if not origin:
                skipped.append((target_id, "no origin item her log still knows"))
                continue
            held_ids = {q.id for q in ws.questions}
            opened = _append(log, ws, at=at, rules=rules, entry=NoticeOp(
                kind=QUESTION_KIND, subject=question_subject(label), target_ids=(target_id,),
                origin_evidence=origin))
            if opened is None:
                skipped.append((target_id, "the kernel refused the question"))
                continue
            ws = opened
            question = next(q for q in ws.questions if q.id not in held_ids)  # the one it just opened
        if not ws.hypotheses_of(question.id):
            set_ = _append(log, ws, at=at, rules=rules, entry=HypothesizeOp(
                question_id=question.id,
                hypotheses=(Alternative(improves_statement(label), PRIORS[0]),
                            Alternative(no_change_statement(label), PRIORS[1])),
                catch_all=CATCH_ALL))
            if set_ is None:
                skipped.append((target_id, "the kernel refused the hypotheses"))
                continue
            ws = set_
        before = previous(history, target_id)
        claim = claim_for(label, None if before is None else before[0])
        predicted = _append(log, ws, at=at, rules=rules, entry=PredictOp(
            question_id=question.id, claim=claim, p_true_given=_given(ws, question.id, label),
            horizon_days=HORIZON_DAYS))
        if predicted is None:
            skipped.append((target_id, "the kernel refused the prediction"))
            continue
        ws = predicted
        committed = ws.predictions[-1]
        made.append(SelfPrediction(target_id, label, question.id, committed.id, claim,
                                   None if before is None else before[0]))
    return Commitment(tuple(made), tuple(skipped))


def resolve(
    log: WorkspaceLog, ws: Workspace, *, night: date, at: datetime, record: Record,
    predictions: Sequence[SelfPrediction], history: Sequence[Mapping[str, object]],
) -> tuple[Outcome, ...]:
    """Resolve each commitment from the night's record: the counts, the baseline its claim named,
    the verdict, and a ``Resolved`` step citing ``record.id`` so the kernel updates her credence."""
    rules = _rules(ws)
    out: list[Outcome] = []
    for prediction in predictions:
        now = record.of(prediction.target_id)
        if prediction.previous is None:
            against, counts = None, record.other
        else:
            against, counts = prediction.previous, _find(history, prediction.target_id, prediction.previous)
        if counts is None or counts.scenes < MIN_SCENES:
            counts = None
        verdict, why = _decide(now, counts)
        entry = Resolved(prediction.prediction_id, verdict,
                         (EvidenceRef(id=record.id, day=night, created_at=at, target_id=prediction.target_id),))
        applied = _append(log, ws, at=at, rules=rules, entry=entry)
        if applied is None:
            continue  # already settled, or a question no longer watched: her log says so
        ws = applied
        out.append(Outcome(prediction.target_id, prediction.label, prediction.question_id,
                           prediction.prediction_id, prediction.claim, verdict, now, counts, against, why))
    return tuple(out)


def _committed(ws: Workspace) -> tuple[SelfPrediction, ...]:
    """Every loop prediction still pending in her workspace, including ones a commit made before it
    failed and so never returned to the night."""
    out: list[SelfPrediction] = []
    for question in ws.questions:
        if not is_self_question(question) or not question.target_ids:
            continue
        for prediction in ws.predictions:
            if prediction.question_id == question.id and prediction.pending:
                out.append(SelfPrediction(question.target_ids[0], label_of(question), question.id,
                                          prediction.id, prediction.claim, None))
    return tuple(out)


def outcomes(
    ws: Workspace, predictions: Sequence[SelfPrediction], record: Record, *, why: str | None = None,
) -> tuple[Outcome, ...]:
    """What her workspace now says each commitment came to, for a night that couldn't finish the
    normal resolution: a true or false verdict stands, a censored prediction reads undetermined with
    the reason, and one still pending is named as such."""
    out: list[Outcome] = []
    for prediction in predictions:
        current = next((p for p in ws.predictions if p.id == prediction.prediction_id), None)
        if current is None:
            continue
        if current.outcome in ("true", "false"):
            verdict: Verdict = current.outcome
            note = None
        elif current.outcome == "censored":
            verdict, note = "undetermined", why or "censored"
        else:
            verdict, note = "undetermined", why or "still pending"
        out.append(Outcome(prediction.target_id, prediction.label, prediction.question_id,
                           prediction.prediction_id, prediction.claim, verdict,
                           record.of(prediction.target_id), None, None, note))
    return tuple(out)


def censor(
    log: WorkspaceLog, ws: Workspace, *, at: datetime, record: Record,
    predictions: Sequence[SelfPrediction] = (), why: str = "the loop could not resolve it",
) -> tuple[Outcome, ...]:
    """Settle pending loop predictions as censored and report what her workspace holds for them.
    The night calls this when the loop can't resolve them (a failed commit or resolution, or a kept
    rerun). MIND's inquiry resolver never touches these predictions — it skips the questions this
    module opened — so a stranded one can only be closed here."""
    rules = _rules(ws)
    known = list(predictions)
    have = {prediction.prediction_id for prediction in known}
    known += [prediction for prediction in _committed(ws) if prediction.prediction_id not in have]
    for prediction in known:
        current = next((p for p in ws.predictions if p.id == prediction.prediction_id), None)
        if current is None or not current.pending:
            continue
        applied = _append(log, ws, at=at, rules=rules, entry=Resolved(prediction.prediction_id, "censored", ()))
        if applied is not None:
            ws = applied
    return outcomes(ws, known, record, why=why)


def _find(history: Sequence[Mapping[str, object]], target_id: str, night: date) -> SceneCount | None:
    """The earlier night's counts for a target, as its summary recorded them."""
    for summary in history:
        if summary.get("night") != night.isoformat():
            continue
        loop = summary.get("mind_loop")
        scenes = loop.get("scenes") if isinstance(loop, Mapping) else None
        counts = scenes.get(target_id) if isinstance(scenes, Mapping) else None
        if isinstance(counts, Mapping):
            return SceneCount(int(counts.get("scenes", 0)), int(counts.get("held_up", 0)))
    return None
