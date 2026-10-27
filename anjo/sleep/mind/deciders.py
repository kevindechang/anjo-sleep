"""The MIND experiment's five deciders (Task 16), replayed on her recorded decision points.

- **D0, her chooser:** the choice the night recorded. It is the only one that ever ran.
- **D1, a plain model:** Spark Open with ``decider_v1.md`` at temperature 0, through a ``ChatFn``
  port. It answers a target id from the snapshot with ``study`` or ``note``, or silence, which is a
  normal, unpenalized answer. It reads only the render, which never holds her choice. A study goes
  through an offered, live question whose first target it names, else through homework.
- **D2, no confidence:** her chooser without the (1 - ``confidence``) multiplier on ``understand``,
  and with the inquiry priority's uncertainty factor fixed at 1.0, so the lesion still bites when
  ``understand`` acts through ``inquire``.
- **D3, no revision gating:** her chooser without the ``revision > acted_revision`` rule: a target
  with no unacted revision gets one unit of new evidence, so her own homework and time can wake it,
  and a parked question wakes without new evidence.
- **D4, earliest due:** the open commitment with the earliest ``check_after``, else the open thread
  mentioned longest ago, studied the way D1's study is.

D2 and D3 change the raw input and replay it through ``snapshot.build``, ``derive_wants`` and
``choose``; D4 reads ``snapshot.build``'s dates and picks by rule. All three are pure, and the
serving chooser has no lesion switch: a lesion lives here and nowhere else. Every decider gets the
same ``DeciderInput`` at a point: the same raw snapshot, its one render, the same context (policy,
allowance and the actions the point really offered, none at a night's missing second offer) and the
same contact rules, which ``choose`` applies to each answer; a replay refuses any choice its point
didn't allow. A decider not asked at a point (D1 without a chat port) is not observed there, which is
never silence. Nothing here executes a choice; the replay only returns what each would have done.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.template import render as fill
from anjo.sleep.mind import points, wants
from anjo.sleep.mind.choose import choose
from anjo.sleep.mind.points import STUDY_ACTIONS, DecisionPoint, Phase, QuestionState, RawSnapshot
from anjo.sleep.mind.snapshot import LIVE_QUESTION
from anjo.sleep.mind.wants import COMMITMENT_CLASS, THREAD_CLASS
from anjo.sleep.types import Action, Choice, ChoiceContext, QuestionView, Result, Target, Want, WantKind

DECIDERS: Final = ("D0", "D1", "D2", "D3", "D4")
CONTACTED: Final = ("D0", "D1")  # the deciders whose notes are compared (contact discipline)
D1_MODEL: Final = SPARK_OPEN
D1_TEMPERATURE: Final = 0.0
D1_TIMEOUT_S: Final = 120.0
DECIDER_PROMPT: Final = Path(__file__).parents[1] / "probes" / "prompts" / "decider_v1.md"
DECIDER_TEMPLATE: Final = DECIDER_PROMPT.read_text(encoding="utf-8")
ONE_UNIT: Final = 1  # D3: what a target with no unacted revision is given

D1Action = Literal["study", "note", "silence"]
D1_ACTIONS: Final = frozenset({"study", "note", "silence"})


@dataclass(frozen=True)
class DeciderInput:
    """What every decider gets at one point. It holds no choice: D0's is the point's own."""

    night: date
    phase: Phase
    raw: RawSnapshot
    render: str
    ctx: ChoiceContext


Decider = Callable[[DeciderInput], Choice]


def input_of(point: DecisionPoint, labels: Mapping[str, str]) -> DeciderInput:
    """The one input every decider gets at ``point``. ``labels`` gives each target's label by id
    (a point's record holds none)."""
    text = render(point.night, point.raw, point.ctx, labels)
    return DeciderInput(night=point.night, phase=point.phase, raw=point.raw, render=text, ctx=point.ctx)


# ── The render: the same text for every decider ──────────────────────────────


def render(night: date, raw: RawSnapshot, ctx: ChoiceContext, labels: Mapping[str, str]) -> str:
    """The state as text: targets (with labels looked up by id), ledger dates, results, questions,
    the allowance and the actions the offer allows. Never a choice."""
    last = "never" if raw.last_message_at is None else raw.last_message_at.isoformat()
    lines = [f"Night of {night.isoformat()}, as of {raw.as_of.isoformat()}.", f"The person last wrote: {last}.",
             "", "Targets (the topics and dated plans tracked for the person):"]
    lines += [_target_line(t, labels.get(t.id, "")) for t in sorted(raw.targets, key=lambda t: t.id)] or ["- none"]
    lines += ["", "Dated plans from the person's own ledger:"]
    lines += [f"- {c.id}: date {_date(c.check_after)}" for c in raw.intentions] or ["- none"]
    lines += ["", "Results already found:"]
    lines += [_result_line(r) for r in sorted(raw.results, key=lambda r: r.id)] or ["- none"]
    lines += ["", "Questions in the workspace:"]
    lines += [_question_line(q) for q in raw.questions] or ["- none"]
    allowance = ctx.allowance
    lines += ["", f"Tonight: {allowance.actions_left} actions left, {allowance.calls_left} research calls left, "
              f"${allowance.usd_left:.2f} of research left.", f"Allowed now: {', '.join(_allowed(ctx))}."]
    return "\n".join(lines)


def _date(value: datetime | None) -> str:
    return "none" if value is None else value.isoformat()


def _target_line(target: Target, label: str) -> str:
    confidence = "none yet" if target.confidence is None else f"{target.confidence:.2f}"
    label = " ".join(label.split())
    return (f"- {target.id} ({target.kind}) \"{label}\": {target.unacted} new from the person since it was last "
            f"studied; confidence {confidence} (heuristic); last mentioned {target.last_mentioned.isoformat()}; "
            f"date {_date(target.check_after)}; status {target.status}; due note sent: "
            f"{'yes' if target.due_noted else 'no'}")


def _result_line(r: Result) -> str:
    said = "already said" if r.consumed else "not yet said"
    return (f"- {r.id} on {r.target_id}: {r.basis}, {r.verified_sources} verified sources, {r.user_cites} of the "
            f"person's items cited, found {r.created_at.isoformat()}, {said}")


def _question_line(state: QuestionState) -> str:
    view = state.view
    return f"- {view.id} on {', '.join(view.target_ids)}: {view.status}, priority {view.priority:.2f}"


def _allowed(ctx: ChoiceContext) -> list[str]:
    allowed = []
    if any(ctx.allows(action) for action in STUDY_ACTIONS) or ctx.allows("inquire"):
        allowed.append("study")
    if ctx.allows("hold_note"):
        allowed.append("note")
    return [*allowed, "silence"]


# ── D1: a plain model, through a port ────────────────────────────────────────


@dataclass(frozen=True)
class D1Answer:
    """D1's answer at one point, its choice under the point's context, and its tokens (its
    overhead, reported apart). ``valid`` is False when the reply named no listed target or wasn't
    one of the three answers; it then counts as silence."""

    action: D1Action
    target_id: str | None
    valid: bool
    input_tokens: int
    output_tokens: int
    choice: Choice


def d1_prompt(inp: DeciderInput) -> str:
    return fill(DECIDER_TEMPLATE, {"STATE": inp.render})


def ask(inp: DeciderInput, chat: ChatFn) -> D1Answer:
    """One Spark Open call at temperature 0 on the render alone."""
    reply = chat(D1_MODEL, [{"role": "user", "content": d1_prompt(inp)}], None, D1_TIMEOUT_S,
                 temperature=D1_TEMPERATURE)
    action, target_id, valid = parse_d1(reply.text, inp)
    return D1Answer(action, target_id, valid, int(reply.input_tokens), int(reply.output_tokens),
                    _plain_choice(inp, action, target_id))


def parse_d1(text: str, inp: DeciderInput) -> tuple[D1Action, str | None, bool]:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        data = json.loads(body)
    except ValueError:
        return "silence", None, False
    action = data.get("action") if isinstance(data, dict) else None
    target_id = data.get("target_id") if isinstance(data, dict) else None
    if action == "silence":
        return "silence", None, target_id is None
    known = {t.id for t in inp.raw.targets}
    if action in ("study", "note") and isinstance(target_id, str) and target_id in known:
        return action, target_id, True
    return "silence", None, False


def _plain_choice(inp: DeciderInput, action: D1Action, target_id: str | None) -> Choice:
    target = next((t for t in inp.raw.targets if t.id == target_id), None)
    if action == "silence" or target is None:
        return choose("away", (), inp.ctx)
    want = _rule_want(target, note=True) if action == "note" else _study_want(target, inp.raw, inp.ctx)
    return choose("away", (want,), inp.ctx)


def _study_want(target: Target, raw: RawSnapshot, ctx: ChoiceContext) -> Want:
    """A study of ``target``: an offered inquiry into a live question whose first target it is, else
    homework (which ``choose`` holds back when the point doesn't offer it)."""
    question = _first_question(target.id, raw) if ctx.allows("inquire") else None
    homework = _rule_want(target, note=False)
    return homework if question is None else replace(homework, action="inquire", question_id=question.id)


def _first_question(target_id: str, raw: RawSnapshot) -> QuestionView | None:
    live = [q.view for q in raw.questions if q.view.status in LIVE_QUESTION and q.view.target_ids[0] == target_id]
    return min(live, key=lambda q: (-q.priority, q.id), default=None)


def _rule_want(target: Target, *, note: bool) -> Want:
    """A study or a note on ``target`` that a rule or a model picked: full strength, so only the
    point's context (offered actions, allowance) can hold it back."""
    commitment = target.kind == "commitment"
    kind: WantKind
    action: Action
    if note:
        kind, action = ("follow_up" if commitment else "share"), "hold_note"
    else:
        kind = "follow_up" if commitment else "understand"
        action = "prepare_commitment" if commitment else "revisit_thread"
    return Want(id=f"{kind}:{target.id}", kind=kind, target_id=target.id, action=action,
                priority_class=COMMITMENT_CLASS if commitment else THREAD_CLASS, strength=1.0)


# ── D2, D3 and D4: pure replays ───────────────────────────────────────────────


def _replay(raw: RawSnapshot, ctx: ChoiceContext) -> Choice:
    return choose("away", wants.derive_wants(points.build(raw)), ctx)


def no_confidence(inp: DeciderInput) -> Choice:
    """D2: confidence never lowers ``understand``, and every question's u is 1.0."""
    raw = inp.raw
    targets = tuple(replace(t, confidence=None) for t in raw.targets)
    questions = tuple(_u_fixed(q) for q in raw.questions)
    return _replay(replace(raw, targets=targets, questions=questions), inp.ctx)


def _u_fixed(state: QuestionState) -> QuestionState:
    parts = replace(state.parts, u=1.0)
    return QuestionState(view=replace(state.view, priority=parts.value), parts=parts)


def no_revision_gating(inp: DeciderInput) -> Choice:
    """D3: a target with no unacted revision gets one unit of new evidence, and a parked question
    is awake."""
    raw = inp.raw
    targets = tuple(t if t.unacted > 0 else replace(t, revision=t.revision + ONE_UNIT) for t in raw.targets)
    questions = tuple(
        QuestionState(view=replace(q.view, status="open"), parts=q.parts) if q.view.status == "parked" else q
        for q in raw.questions
    )
    return _replay(replace(raw, targets=targets, questions=questions), inp.ctx)


def earliest_due(inp: DeciderInput) -> Choice:
    """D4: the open commitment with the earliest date (its own, or its ledger entry's), else the
    open thread mentioned longest ago; ties go to the smaller id."""
    dated = [(t.check_after, t.id, t) for t in points.build(inp.raw).targets
             if t.kind == "commitment" and t.check_after is not None]
    threads = [(t.last_mentioned, t.id, t) for t in inp.raw.targets if t.kind == "thread" and t.status == "open"]
    if dated:
        pick = min(dated, key=lambda row: (row[0], row[1]))[2]
    elif threads:
        pick = min(threads, key=lambda row: (row[0], row[1]))[2]
    else:
        return choose("away", (), inp.ctx)
    return choose("away", (_study_want(pick, inp.raw, inp.ctx),), inp.ctx)


LESIONS: Final[Mapping[str, Decider]] = MappingProxyType({
    "D2": no_confidence, "D3": no_revision_gating, "D4": earliest_due,
})


# ── One night's replay ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Replay:
    """Every decider's choice at one point: D0's as recorded, D1's (when asked) and the lesions'."""

    input: DeciderInput
    choices: Mapping[str, Choice | None]
    d1: D1Answer | None


def replay_point(
    point: DecisionPoint, labels: Mapping[str, str], *, chat: ChatFn | None,
    lesions: Mapping[str, Decider] = LESIONS,
) -> Replay:
    """Replay ``point``: D0 is its recorded choice; D1 is asked through ``chat`` (``None`` leaves
    it unobserved); each lesion gets the same input. Every choice is held to what the point allowed."""
    inp = input_of(point, labels)
    choices: dict[str, Choice | None] = {"D0": point.choice}
    answer = None if chat is None else ask(inp, chat)
    if answer is not None:
        choices["D1"] = answer.choice
    for name, decide in lesions.items():
        choices[name] = decide(inp)
    for name, choice in choices.items():
        if choice is not None and choice.want is not None and not point.ctx.allows(choice.want.action):
            raise ValueError(f"{name} chose {choice.want.action}, which its point didn't allow")
    return Replay(input=inp, choices=MappingProxyType(choices), d1=answer)


@dataclass(frozen=True)
class Note:
    """The night's one note of a decider: its target, the offer it came at, and whether it cites a
    due commitment or a changed target."""

    target_id: str
    phase: Phase
    anchored: bool


@dataclass(frozen=True)
class NightPicks:
    """One night, per decider: what it would have studied (``None`` is silence) and, for D0 and
    D1, its note. ``studied`` is whether her step-4 choice was a study (what ran)."""

    night: date
    studied: bool
    study: Mapping[str, str | None]
    notes: Mapping[str, Note | None]
    d1: tuple[D1Answer, ...]


def night_picks(pre: Replay, post: Replay | None) -> NightPicks:
    """The night from its pre-study replay and its post-result one (``None`` when the night
    stopped before step 10)."""
    if pre.input.phase != "pre_study":
        raise ValueError("a night's first replay is its pre-study point")
    if post is not None and (post.input.phase != "post_result" or post.input.night != pre.input.night):
        raise ValueError("a night's second replay is the same night's post-result point")
    study = {name: points.study_target(choice, pre.input.raw) for name, choice in pre.choices.items()}
    notes = {name: _night_note(name, pre, post) for name in CONTACTED if name in pre.choices}
    answers = tuple(r.d1 for r in (pre, post) if r is not None and r.d1 is not None)
    return NightPicks(night=pre.input.night, studied=study.get("D0") is not None,
                      study=MappingProxyType(study), notes=MappingProxyType(notes), d1=answers)


def _night_note(name: str, pre: Replay, post: Replay | None) -> Note | None:
    """The first offer at which ``name`` chose a note: one note a night, for every decider."""
    for done in (pre, post):
        choice = None if done is None else done.choices.get(name)
        if done is not None and choice is not None and choice.action == "hold_note" and choice.target_id:
            return Note(choice.target_id, done.input.phase, anchored(choice.target_id, done.input.raw))
    return None


def anchored(target_id: str, raw: RawSnapshot) -> bool:
    """A note is anchored when its target is a due commitment not yet noted, or a changed target:
    one with a result admitted by ``snapshot.build`` that she hasn't said since the person last wrote."""
    built = points.build(raw)
    due = any(
        t.id == target_id and t.kind == "commitment" and t.check_after is not None
        and t.check_after <= raw.as_of and not t.due_noted
        for t in built.targets
    )
    changed = any(
        r.target_id == target_id and not r.consumed
        and (raw.last_message_at is None or r.created_at > raw.last_message_at)
        for r in built.results
    )
    return due or changed
