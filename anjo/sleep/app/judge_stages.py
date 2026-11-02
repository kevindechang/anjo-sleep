"""A judge's night, as the night view shows it (2026-11-02): the night's steps folded into the
contract's twelve stages, each with a short plain summary.

``fold`` takes the stages as they stand and one step event from the night's observer (``night/deps.py``
``StepObserver``: the step, ``running``/``ok``/``failed``/``skipped``, its receipt) and returns the new
stages; it never changes the ones it is handed. Steps the view has no stage for (``close_day``,
``brief``, ``probes``, ``dataset``, ``harness``, ``sleep``) change nothing. ``mind_inquire`` has no
step of its own: MIND's inquiry runs inside ``choose``, so its stage is settled when ``choose`` ends.
``refine`` (the note rewritten after the gates) updates the ``note`` stage's summary. ``serve`` is set
by the dispatcher after the night (``app/judge_night.py``).

Summaries are counts and labels (a target's own label, as the judge's MIND panel shows it), never a
line of the judge's chat.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

STAGES: Final = ("extract", "mind_notice", "mind_choose", "mind_inquire", "homework", "self_predict", "note",
                 "practice", "train", "gate", "promote", "serve")
PHASE_TWO: Final = frozenset({"practice", "train", "gate", "promote", "serve"})
STEP_STAGE: Final[Mapping[str, str]] = {
    "extract": "extract", "notice": "mind_notice", "choose": "mind_choose", "homework": "homework",
    "self_predict": "self_predict", "note": "note", "practice": "practice", "train": "train", "gates": "gate",
    "promote": "promote",
}
CLAIM_CHARS: Final = 120  # a longer self-bet is named by its target instead
STATES: Final[Mapping[str, str]] = {"running": "running", "ok": "done", "failed": "failed", "skipped": "skipped"}
ACTIONS: Final[Mapping[str, str]] = {
    "prepare_commitment": "Chose to prepare", "revisit_thread": "Chose to study", "inquire": "Chose to ask",
    "hold_note": "Chose to write",
}

MIND_RESTED: Final = "A live thought still held her mind, so MIND rested tonight"
RESTED: Final = ("mind_notice", "mind_inquire", "self_predict")  # what a night without MIND leaves out

Stages = dict[str, dict[str, Any]]
Label = Callable[[str], "str | None"]


def empty() -> Stages:
    return {name: {"state": "pending", "started_at": None, "ended_at": None, "summary": ""} for name in STAGES}


def as_list(stages: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The stages in the contract's order, as the night route returns them."""
    return [{"name": name, **stages.get(name, empty()[name])} for name in STAGES]


def _set(stages: Stages, name: str, state: str, at: str, summary: str | None) -> Stages:
    old = stages[name]
    new = {**old, "state": state}
    if state == "running":
        new["started_at"] = old["started_at"] or at
    else:
        new["started_at"] = old["started_at"] or at
        new["ended_at"] = at
    if summary is not None:
        new["summary"] = summary
    return {**stages, name: new}


def _part(receipt: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """A receipt's nested record, or an empty one."""
    found = receipt.get(key)
    return found if isinstance(found, Mapping) else {}


def _n(receipt: Mapping[str, Any], key: str) -> int:
    value = receipt.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _why(receipt: Mapping[str, Any]) -> str | None:
    why = receipt.get("why") or receipt.get("skipped")
    return why[:1].upper() + why[1:] if isinstance(why, str) and why else None


def _extract(r: Mapping[str, Any], label: Label) -> str:
    if r.get("training_night") is False:
        return f"Read {_plural(_n(r, 'items'), 'thing')}; nothing new to learn from yet"
    return f"Read {_plural(_n(r, 'items'), 'thing')}, {_n(r, 'new_since_last_training')} new"


def _notice(r: Mapping[str, Any], label: Label) -> str:
    opened = _n(r, "questions_opened")
    return f"Opened {_plural(opened, 'question')}" if opened else "Noticed nothing that needs a question"


def _target(r: Mapping[str, Any], label: Label) -> str | None:
    target = r.get("target_id")
    return label(target) if isinstance(target, str) and target else None


def _choose(r: Mapping[str, Any], label: Label) -> str:
    action = str(r.get("action") or "")
    said = ACTIONS.get(action, "Chose to rest" if action in ("", "none", "rest") else f"Chose: {action.replace('_', ' ')}")
    target = _target(r, label)
    return f"{said}: {target}" if target else said


def _homework(r: Mapping[str, Any], label: Label) -> str:
    study = _part(r, "study")
    if study.get("background"):
        return "Studying in the background"
    if study.get("stopped"):
        return f"Study stopped ({study['stopped']})"
    target = _target(r, label)
    found = _plural(_n(study, "homework_items"), "thing")
    return f"Studied: {target} ({found} found)" if target else f"Studied ({found} found)"


def _self_predict(r: Mapping[str, Any], label: Label) -> str:
    committed = r.get("committed")
    if isinstance(committed, list) and committed and isinstance(committed[0], Mapping):
        first = committed[0]
        claim = first.get("claim")
        if isinstance(claim, str) and 0 < len(claim) <= CLAIM_CHARS:
            return f"Bet: {claim}"
        return f"Bet that practising {first.get('label') or 'tonight'} helps"
    return "No bet tonight"


def _note(r: Mapping[str, Any], label: Label) -> str:
    outcome = str(r.get("outcome") or "")
    if outcome in ("held", "ran", "sent", "queued") or r.get("note_id"):
        return "Wrote a note for the morning"
    return "No note tonight" if not outcome else f"Note {outcome.replace('_', ' ')}"


def _practice(r: Mapping[str, Any], label: Label) -> str:
    pairs = _n(r, "pairs")
    said = f"Practised {_plural(_n(r, 'scenes'), 'scene')}"
    return f"{said}, {_plural(pairs, 'pair')} to learn from" if pairs else said


def _train(r: Mapping[str, Any], label: Label) -> str:
    receipt = _part(r, "receipt")
    steps = _n(receipt, "steps")
    warm = receipt.get("runner") == "modal_warm"
    return f"Updated her weights: {_plural(steps, 'step')} on a {'warm' if warm else 'fresh'} GPU"


def _gate(r: Mapping[str, Any], label: Label) -> str:
    gates = _part(r, "gates")
    ran = {name: g for name, g in gates.items() if isinstance(g, Mapping) and not g.get("skipped")}
    failed = sorted(name for name, g in ran.items() if not g.get("passed"))
    if failed:
        return f"Gate held it back: {', '.join(failed)}"
    return f"Gate kept it: {len(ran)} of {len(ran)} checks passed"


def _promote(r: Mapping[str, Any], label: Label) -> str:
    if r.get("promoted"):
        return "Tonight's weights now serve her"
    return "Kept last night's weights"


SUMMARIES: Final[Mapping[str, Callable[[Mapping[str, Any], Label], str]]] = {
    "extract": _extract, "mind_notice": _notice, "mind_choose": _choose, "homework": _homework,
    "self_predict": _self_predict, "note": _note, "practice": _practice, "train": _train, "gate": _gate,
    "promote": _promote,
}


def _summary(stage: str, state: str, receipt: Mapping[str, Any], label: Label) -> str | None:
    if state == "running":
        return None
    if state == "failed":
        return "Didn't finish"
    if state == "skipped" or receipt.get("noop"):
        return _why(receipt) or "Not needed tonight"
    return SUMMARIES[stage](receipt, label)


def _inquire(stages: Stages, state: str, receipt: Mapping[str, Any], at: str) -> Stages:
    inquiry = receipt.get("inquiry")
    if state == "ok" and isinstance(inquiry, Mapping):
        calls = _n(inquiry, "research_calls")
        return _set(stages, "mind_inquire", "done", at, f"Asked {_plural(calls, 'research question')}")
    if state in ("ok", "skipped"):
        return _set(stages, "mind_inquire", "skipped", at, "Nothing to ask tonight")
    return _set(stages, "mind_inquire", "failed", at, "Didn't finish")


def rest_mind(stages: Stages, at: str) -> Stages:
    """MIND's stages ``skipped`` with ``MIND_RESTED``, when the live-pass lock kept MIND from running."""
    for name in RESTED:
        if stages[name]["state"] != "skipped" or stages[name]["summary"] != MIND_RESTED:
            stages = _set(stages, name, "skipped", at, MIND_RESTED)
    return stages


def fold(stages: Stages, step: str, state: str, receipt: Mapping[str, Any], at: str,
         label: Label = lambda target_id: None) -> Stages:
    """``stages`` after one step event at time ``at`` (an ISO string)."""
    if state not in STATES:
        return stages
    if step == "refine" and state == "ok":
        outcome = receipt.get("outcome")
        line = {"refined": "Rewritten by tonight's weights", "held_back_line": "Wrote a note; tonight's weights wait"}
        return _set(stages, "note", stages["note"]["state"], at, line.get(str(outcome))) if outcome in line else stages
    name = STEP_STAGE.get(step)
    if name is None:
        return stages
    folded = _set(stages, name, STATES[state], at, _summary(name, state, receipt, label))
    if step == "choose" and state != "running":
        folded = _inquire(folded, state, receipt, at)
    return folded


def phase(stages: Mapping[str, Mapping[str, Any]]) -> int:
    """2 once any phase-2 stage has started, else 1."""
    return 2 if any(stages[name]["state"] != "pending" for name in PHASE_TWO if name in stages) else 1


def close(stages: Stages, at: str, *, failed: bool) -> Stages:
    """At the night's end: a stage still running ends failed (or done), and one never reached is skipped."""
    done: Stages = {}
    for name, stage in stages.items():
        if stage["state"] == "running":
            done[name] = {**stage, "state": "failed" if failed else "done", "ended_at": at}
        elif stage["state"] == "pending" and name != "serve":
            done[name] = {**stage, "state": "skipped", "summary": stage["summary"] or "Not reached tonight"}
        else:
            done[name] = stage
    return done
