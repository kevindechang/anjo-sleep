"""The workspace's JSON form: its records, its steps and its log's events (Task 17). Pure.

One form serves the log and the digest: ``log.py`` stores each ``WorkspaceEvent`` as
``event_to_json``, and ``kernel.digest`` hashes ``workspace_document``, so what is stored is what is
hashed. Floats are written by ``json`` (the shortest form that reads back exactly) and times as
ISO 8601, so a decoded event folds to the same digest. Decoding is strict: a record that isn't one
raises ``CodecError``, which names the field and never repeats a stored value.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Final

from anjo.sleep.mind.inquiry.types import (
    Answered,
    Budget,
    Edit,
    EvidenceRef,
    KernelRules,
    Mark,
    Observation,
    Op,
    OpFormatError,
    Prediction,
    Resolved,
    Retrieved,
    RunStart,
    Step,
    Wake,
    Workspace,
    WorkspaceEvent,
    op_from_json,
    op_to_json,
)

Entry = Op | Observation | Mark


class CodecError(ValueError):
    """A stored record that isn't a workspace event."""


def canonical(document: Mapping[str, Any]) -> str:
    """The one serialization the digest hashes: sorted keys, no spaces, ASCII."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ── Encoding ──────────────────────────────────────────────────────────────────


def _ref(ref: EvidenceRef) -> dict[str, Any]:
    return {
        "id": ref.id, "day": ref.day.isoformat(), "created_at": ref.created_at.isoformat(), "target_id": ref.target_id,
    }


def _budget(budget: Budget) -> dict[str, Any]:
    return {
        "cycles_used": budget.cycles_used, "tokens_used": budget.tokens_used,
        "cycle_cap": budget.cycle_cap, "token_cap": budget.token_cap,
    }


def entry_to_json(entry: Entry | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    if isinstance(entry, Retrieved):
        return {
            "type": "retrieved", "question_id": entry.question_id, "tool": entry.tool, "query": entry.query,
            "items": [_ref(item) for item in entry.items],
        }
    if isinstance(entry, Answered):
        return {"type": "answered", "question_id": entry.question_id, "query": entry.query, "answer": entry.answer}
    if isinstance(entry, Resolved):
        return {
            "type": "resolved", "prediction_id": entry.prediction_id, "outcome": entry.outcome,
            "seen": [_ref(item) for item in entry.seen],
        }
    if isinstance(entry, RunStart):
        decided = {} if entry.decider is None else {"decider": entry.decider}  # the controller's logs stay byte for byte
        return {
            "type": "run_start", "night": entry.night.isoformat(), "as_of": entry.as_of.isoformat(),
            "call": entry.call, "control": entry.control, "focus": entry.focus, "wipe": entry.wipe,
            "rules": {"k1": entry.rules.k1, "k2": entry.rules.k2, "k4": entry.rules.k4},
            "budget": _budget(entry.budget), "denylist": entry.denylist, **decided,
        }
    if isinstance(entry, Wake):
        return {"type": "wake", "question_id": entry.question_id, "items": [_ref(item) for item in entry.items]}
    if isinstance(entry, Edit):
        return {
            "type": "edit", "kind": entry.kind, "question_id": entry.question_id,
            "hypothesis_id": entry.hypothesis_id, "prediction_id": entry.prediction_id,
            "statement": entry.statement,
        }
    return {"type": "op", "op": op_to_json(entry)}


def step_to_json(step: Step) -> dict[str, Any]:
    return {
        "seq": step.seq, "at": step.at.isoformat(), "entry": entry_to_json(step.entry), "code": step.code,
        "cycle": step.cycle, "focus": step.focus, "tokens": step.tokens,
    }


def event_to_json(event: WorkspaceEvent) -> dict[str, Any]:
    return {"step": step_to_json(event.step), "before": event.before, "after": event.after}


def records_document(ws: Workspace) -> dict[str, Any]:
    """The workspace's records without its history: what an export shows a person."""
    return {
        "questions": [
            {
                "id": q.id, "target_ids": list(q.target_ids), "kind": q.kind,
                "origin_evidence": list(q.origin_evidence), "status": q.status,
                "opened_at": q.opened_at.isoformat(), "subject": q.subject,
            }
            for q in ws.questions
        ],
        "hypotheses": [
            {"id": h.id, "question_id": h.question_id, "statement": h.statement, "credence": h.credence}
            for h in ws.hypotheses
        ],
        "predictions": [_prediction(p) for p in ws.predictions],
        "updates": [
            {
                "question_id": u.question_id, "ref": u.ref, "source": u.source,
                "evidence_ids": list(u.evidence_ids), "likelihoods": dict(u.likelihoods),
            }
            for u in ws.updates
        ],
        "findings": [
            {
                "id": f.id, "question_id": f.question_id, "statement": f.statement, "p": f.p,
                "cites": list(f.cites), "kind": f.kind, "hypothesis_id": f.hypothesis_id,
            }
            for f in ws.findings
        ],
    }


def _prediction(p: Prediction) -> dict[str, Any]:
    return {
        "id": p.id, "question_id": p.question_id, "claim": p.claim, "p_true_given": dict(p.p_true_given),
        "committed_at": p.committed_at.isoformat(), "horizon": p.horizon.isoformat(), "outcome": p.outcome,
        "resolved_by": list(p.resolved_by),
    }


def workspace_document(ws: Workspace) -> dict[str, Any]:
    """The canonical workspace, history and op counter included: what ``kernel.digest`` hashes."""
    return {**records_document(ws), "history": [step_to_json(step) for step in ws.history], "ops": ws.ops}


# ── Decoding ──────────────────────────────────────────────────────────────────


def _str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _opt_str(value: object, name: str) -> str | None:
    return None if value is None else _str(value, name)


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a whole number")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _map(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _ref_from(value: object) -> EvidenceRef:
    data = _map(value, "an item")
    return EvidenceRef(
        id=_str(data["id"], "id"), day=date.fromisoformat(_str(data["day"], "day")),
        created_at=datetime.fromisoformat(_str(data["created_at"], "created_at")),
        target_id=_opt_str(data.get("target_id"), "target_id"),
    )


def _refs(value: object, name: str) -> tuple[EvidenceRef, ...]:
    return tuple(_ref_from(item) for item in _list(value, name))


def _run_start(data: Mapping[str, Any]) -> RunStart:
    rules = _map(data["rules"], "rules")
    budget = _map(data["budget"], "budget")
    return RunStart(
        night=date.fromisoformat(_str(data["night"], "night")),
        as_of=datetime.fromisoformat(_str(data["as_of"], "as_of")),
        call=data["call"], control=data["control"], focus=_opt_str(data["focus"], "focus"),
        wipe=_bool(data["wipe"], "wipe"),
        rules=KernelRules(k1=_bool(rules["k1"], "k1"), k2=_bool(rules["k2"], "k2"), k4=_bool(rules["k4"], "k4")),
        budget=Budget(**{name: _int(budget[name], name) for name in _BUDGET_FIELDS}),
        denylist=_opt_str(data.get("denylist"), "denylist"), decider=_opt_str(data.get("decider"), "decider"),
    )


_BUDGET_FIELDS: Final = ("cycles_used", "tokens_used", "cycle_cap", "token_cap")


def _entry_from(value: object) -> Entry | None:
    if value is None:
        return None
    data = _map(value, "entry")
    kind = data.get("type")
    if kind == "op":
        return op_from_json(data["op"])
    if kind == "retrieved":
        return Retrieved(
            question_id=_opt_str(data["question_id"], "question_id"), tool=data["tool"],
            query=_str(data["query"], "query"), items=_refs(data["items"], "items"),
        )
    if kind == "answered":
        return Answered(
            question_id=_str(data["question_id"], "question_id"), query=_str(data["query"], "query"),
            answer=_str(data["answer"], "answer"),
        )
    if kind == "resolved":
        return Resolved(
            prediction_id=_str(data["prediction_id"], "prediction_id"), outcome=data["outcome"],
            seen=_refs(data["seen"], "seen"),
        )
    if kind == "run_start":
        return _run_start(data)
    if kind == "wake":
        return Wake(question_id=_str(data["question_id"], "question_id"), items=_refs(data["items"], "items"))
    if kind == "edit":
        return Edit(
            kind=data["kind"], question_id=_str(data["question_id"], "question_id"),
            hypothesis_id=_opt_str(data["hypothesis_id"], "hypothesis_id"),
            prediction_id=_opt_str(data["prediction_id"], "prediction_id"),
            statement=_opt_str(data["statement"], "statement"),
        )
    raise ValueError("unknown entry type")


def step_from_json(value: object) -> Step:
    data = _map(value, "step")
    cycle = data["cycle"]
    return Step(
        seq=_int(data["seq"], "seq"), at=datetime.fromisoformat(_str(data["at"], "at")),
        entry=_entry_from(data["entry"]), code=data["code"],
        cycle=None if cycle is None else _int(cycle, "cycle"), focus=_opt_str(data["focus"], "focus"),
        tokens=_int(data["tokens"], "tokens"),
    )


def event_from_json(value: object) -> WorkspaceEvent:
    """The event a stored record holds; ``CodecError`` when it isn't one."""
    try:
        data = _map(value, "event")
        return WorkspaceEvent(
            step=step_from_json(data["step"]), before=_str(data["before"], "before"),
            after=_str(data["after"], "after"),
        )
    except OpFormatError as exc:
        raise CodecError(f"not a workspace event ({exc})") from None
    except (KeyError, TypeError, ValueError) as exc:
        raise CodecError(f"not a workspace event ({type(exc).__name__})") from None

