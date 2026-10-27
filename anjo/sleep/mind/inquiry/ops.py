"""The nine operations the thinker may propose, and their reference JSON codec (Task 17).

Pure. ``anjo.sleep.mind.inquiry.types`` re-exports every name here, so the contract has one import
point; the operations live in their own module only to keep each file small.

**What the thinker emits.** In an inquiry cycle, one JSON object: one operation, in the shape its
class docstring shows, with ``"op"`` naming it. In the notice pass (night step 2b), a JSON array of
zero or more ``NOTICE`` objects. Hypotheses are named by their ids as the render shows them; the
catch-all is ``catch_all_id(question_id)``. ``op_to_json`` and ``op_from_json`` are the strict
reference codec: exactly the operation's fields, of the right JSON types. The lenient parser of
real model text is ``parse.parse_op`` (17-core), which maps any failure to
``Rejected("malformed")``.

An operation's constructor checks its shape only: non-empty ids and texts, numbers that are
finite, probabilities in [0, 1], likelihoods positive. Whether it may apply (counts, buckets,
spreads, references, the domain rule) is the kernel's to decide, as a ``RejectCode``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal, get_args

from anjo.sleep.mind.inquiry._check import (
    check_number,
    check_one_of,
    check_positive,
    check_text,
    check_unit,
    freeze_ids,
    freeze_map,
    freeze_tuple,
)

QuestionKind = Literal["contradiction", "gap", "opportunity", "anomaly"]
ConclusionKind = Literal["suspicion", "resolution"]  # what CONCLUDE may report
Tool = Literal["evidence_search", "world_query", "research"]
OpName = Literal[
    "NOTICE", "HYPOTHESIZE", "PREDICT", "INVESTIGATE", "ASSESS", "CONCLUDE", "RETRACT", "PARK",
    "ABANDON",
]
QUESTION_KINDS: Final = frozenset(get_args(QuestionKind))
TOOLS: Final = frozenset(get_args(Tool))


@dataclass(frozen=True)
class Alternative:
    """One hypothesis HYPOTHESIZE proposes: its statement and its prior (clamped by the kernel)."""

    statement: str
    credence: float

    def __post_init__(self) -> None:
        check_text("statement", self.statement)
        check_unit("credence", self.credence)


@dataclass(frozen=True)
class NoticeOp:
    """Open a question nobody assigned. Only the notice pass and an inquiry cycle may emit it.

    {"op": "NOTICE", "kind": "contradiction", "subject": "Scene 4's long-lens look against the small kitchen it is shot in", "target_ids": ["t-scene4"], "origin_evidence": ["e-d2-1", "e-d5-3"]}
    """

    name: ClassVar[Literal["NOTICE"]] = "NOTICE"
    kind: QuestionKind
    subject: str
    target_ids: tuple[str, ...]
    origin_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        check_one_of("kind", self.kind, QUESTION_KINDS)
        check_text("subject", self.subject)
        freeze_ids(self, "target_ids")
        freeze_ids(self, "origin_evidence")


@dataclass(frozen=True)
class HypothesizeOp:
    """Set a question's hypotheses once: 2-4 mutually exclusive alternatives, each with a prior,
    and the catch-all's prior; the kernel clamps and normalizes. A second one only splits the
    catch-all, while it holds more than ``SPLIT_MIN`` and the set stays within 4 alternatives:
    ``hypotheses`` and ``catch_all`` then share the catch-all's current credence in proportion,
    and each new hypothesis inherits the catch-all's ``p_true_given`` in every prediction already
    committed on the question.

    {"op": "HYPOTHESIZE", "question_id": "q1", "hypotheses": [{"statement": "The plan assumes room for an 85 mm lens that the kitchen lacks", "credence": 0.5}, {"statement": "She will shoot through the window from the courtyard", "credence": 0.3}], "catch_all": 0.2}
    """

    name: ClassVar[Literal["HYPOTHESIZE"]] = "HYPOTHESIZE"
    question_id: str
    hypotheses: tuple[Alternative, ...]
    catch_all: float

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        freeze_tuple(self, "hypotheses", Alternative)
        check_unit("catch_all", self.catch_all)


@dataclass(frozen=True)
class PredictOp:
    """Commit a test before its outcome exists: P(claim | hypothesis) for every hypothesis id
    (K2: spread of at least ``MIN_SPREAD``), resolved blind within ``horizon_days``.

    {"op": "PREDICT", "question_id": "q1", "claim": "She names a place other than the kitchen for scene 4", "p_true_given": {"q1.h1": 0.2, "q1.h2": 0.9, "q1.else": 0.5}, "horizon_days": 3}
    """

    name: ClassVar[Literal["PREDICT"]] = "PREDICT"
    question_id: str
    claim: str
    p_true_given: Mapping[str, float] = field(hash=False)
    horizon_days: int

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_text("claim", self.claim)
        freeze_map(self, "p_true_given", check_unit)
        if isinstance(self.horizon_days, bool) or not isinstance(self.horizon_days, int) or self.horizon_days < 1:
            raise ValueError("horizon_days must be a whole number of days, at least 1")


@dataclass(frozen=True)
class InvestigateOp:
    """Run one tool for a question. ``evidence_search`` is time-masked to the night;
    ``world_query`` never moves credence; ``research`` (Wonder) runs in the live night only and
    its query carries topic terms only.

    {"op": "INVESTIGATE", "question_id": "q1", "tool": "evidence_search", "query": "courtyard window scene 4"}
    """

    name: ClassVar[Literal["INVESTIGATE"]] = "INVESTIGATE"
    question_id: str
    tool: Tool
    query: str

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_one_of("tool", self.tool, TOOLS)
        check_text("query", self.query)


@dataclass(frozen=True)
class AssessOp:
    """Weigh one item a search or verified research returned for the question: a likelihood
    per hypothesis id from ``ASSESS_BUCKETS``. Under M-kernel only (K1 off), ``credences``
    replaces ``likelihoods`` and writes the vector directly. Exactly one of the two is set.

    {"op": "ASSESS", "question_id": "q1", "evidence_id": "e-d3-2", "likelihoods": {"q1.h1": 0.5, "q1.h2": 4, "q1.else": 1}}
    """

    name: ClassVar[Literal["ASSESS"]] = "ASSESS"
    question_id: str
    evidence_id: str
    likelihoods: Mapping[str, float] | None = field(default=None, hash=False)
    credences: Mapping[str, float] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_text("evidence_id", self.evidence_id)
        if (self.likelihoods is None) == (self.credences is None):
            raise ValueError("ASSESS carries likelihoods or credences, exactly one")
        freeze_map(self, "likelihoods", check_positive)
        freeze_map(self, "credences", check_unit)


@dataclass(frozen=True)
class ConcludeOp:
    """Report a finding on the question's leading hypothesis: a ``resolution`` (K4) or a
    ``suspicion`` with its credence. A question has one finding, so a later CONCLUDE updates it
    in place.

    {"op": "CONCLUDE", "question_id": "q1", "hypothesis_id": "q1.h1", "kind": "resolution", "statement": "The 85 mm look for scene 4 can't be shot in the kitchen, and the courtyard is locked on Thursday", "cites": ["e-d2-1", "e-d5-3", "e-d6-1"]}
    """

    name: ClassVar[Literal["CONCLUDE"]] = "CONCLUDE"
    question_id: str
    hypothesis_id: str
    kind: ConclusionKind
    statement: str
    cites: tuple[str, ...]

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_text("hypothesis_id", self.hypothesis_id)
        check_one_of("kind", self.kind, frozenset(get_args(ConclusionKind)))
        check_text("statement", self.statement)
        freeze_ids(self, "cites")


@dataclass(frozen=True)
class RetractOp:
    """Withdraw the question's finding, or the question itself when it dissolved.

    {"op": "RETRACT", "question_id": "q2", "statement": "The early budget worry is gone: the grant covers the rental", "cites": ["e-d7-2"]}
    """

    name: ClassVar[Literal["RETRACT"]] = "RETRACT"
    question_id: str
    statement: str
    cites: tuple[str, ...]

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)
        check_text("statement", self.statement)
        freeze_ids(self, "cites")


@dataclass(frozen=True)
class ParkOp:
    """Set the question aside until new evidence on one of its targets wakes it.

    {"op": "PARK", "question_id": "q1"}
    """

    name: ClassVar[Literal["PARK"]] = "PARK"
    question_id: str

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)


@dataclass(frozen=True)
class AbandonOp:
    """Close the question without a finding.

    {"op": "ABANDON", "question_id": "q1"}
    """

    name: ClassVar[Literal["ABANDON"]] = "ABANDON"
    question_id: str

    def __post_init__(self) -> None:
        check_text("question_id", self.question_id)


Op = NoticeOp | HypothesizeOp | PredictOp | InvestigateOp | AssessOp | ConcludeOp | RetractOp | ParkOp | AbandonOp
OP_TYPES: Final[Mapping[OpName, type[Op]]] = MappingProxyType({
    "NOTICE": NoticeOp, "HYPOTHESIZE": HypothesizeOp, "PREDICT": PredictOp,
    "INVESTIGATE": InvestigateOp, "ASSESS": AssessOp, "CONCLUDE": ConcludeOp, "RETRACT": RetractOp,
    "PARK": ParkOp, "ABANDON": AbandonOp,
})
OP_CLASSES: Final = tuple(OP_TYPES.values())


# ── The reference codec ───────────────────────────────────────────────────────


class OpFormatError(ValueError):
    """A JSON object that isn't one well-formed operation. The message names the field, never
    the text the model wrote."""


def op_to_json(op: Op) -> dict[str, Any]:
    """The JSON object the thinker emits for ``op``."""
    out: dict[str, Any] = {"op": op.name}
    for f in fields(op):
        value = getattr(op, f.name)
        if value is not None:
            out[f.name] = _plain(value)
    return out


def _plain(value: object) -> object:
    if isinstance(value, Alternative):
        return {"statement": value.statement, "credence": value.credence}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return dict(value)
    return value


def op_from_json(data: object) -> Op:
    """The operation ``data`` holds, read strictly: exactly its fields, of the right types.
    ``OpFormatError`` otherwise."""
    if not isinstance(data, Mapping):
        raise OpFormatError("an operation is a JSON object")
    name = data.get("op")
    spec = _DECODERS.get(name) if isinstance(name, str) else None
    if spec is None:
        raise OpFormatError("unknown op")
    required, optional, decode = spec
    keys = set(data) - {"op"}
    if not required <= keys <= required | optional:
        raise OpFormatError(f"{name} takes {sorted(required | optional)}")
    try:
        return decode(data)
    except (TypeError, ValueError) as exc:
        raise OpFormatError(f"{name}: {exc}") from None


def _s(data: Mapping[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _ids(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(value)


def _numbers(data: Mapping[str, Any], key: str) -> dict[str, float]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object of numbers")
    return {_key(k): check_number(f"a value in {key}", v) for k, v in value.items()}


def _key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("a key must be a string")
    return value


def _whole(data: Mapping[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be a whole number")
    return int(value)


def _alternatives(data: Mapping[str, Any]) -> tuple[Alternative, ...]:
    value = data["hypotheses"]
    if not isinstance(value, list):
        raise TypeError("hypotheses must be a list")
    out: list[Alternative] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"statement", "credence"}:
            raise TypeError("each hypothesis is {statement, credence}")
        out.append(Alternative(statement=_s(item, "statement"), credence=check_number("credence", item["credence"])))
    return tuple(out)


def _optional_numbers(data: Mapping[str, Any], key: str) -> dict[str, float] | None:
    return _numbers(data, key) if key in data else None


Decoder = Callable[[Mapping[str, Any]], Op]
_NONE: Final[frozenset[str]] = frozenset()
_DECODERS: Final[Mapping[str, tuple[frozenset[str], frozenset[str], Decoder]]] = MappingProxyType({
    "NOTICE": (frozenset({"kind", "subject", "target_ids", "origin_evidence"}), _NONE, lambda d: NoticeOp(
        kind=d["kind"], subject=_s(d, "subject"), target_ids=_ids(d, "target_ids"),
        origin_evidence=_ids(d, "origin_evidence"),
    )),
    "HYPOTHESIZE": (frozenset({"question_id", "hypotheses", "catch_all"}), _NONE, lambda d: HypothesizeOp(
        question_id=_s(d, "question_id"), hypotheses=_alternatives(d),
        catch_all=check_number("catch_all", d["catch_all"]),
    )),
    "PREDICT": (frozenset({"question_id", "claim", "p_true_given", "horizon_days"}), _NONE, lambda d: PredictOp(
        question_id=_s(d, "question_id"), claim=_s(d, "claim"),
        p_true_given=_numbers(d, "p_true_given"), horizon_days=_whole(d, "horizon_days"),
    )),
    "INVESTIGATE": (frozenset({"question_id", "tool", "query"}), _NONE, lambda d: InvestigateOp(
        question_id=_s(d, "question_id"), tool=d["tool"], query=_s(d, "query"),
    )),
    "ASSESS": (frozenset({"question_id", "evidence_id"}), frozenset({"likelihoods", "credences"}), lambda d: AssessOp(
        question_id=_s(d, "question_id"), evidence_id=_s(d, "evidence_id"),
        likelihoods=_optional_numbers(d, "likelihoods"), credences=_optional_numbers(d, "credences"),
    )),
    "CONCLUDE": (frozenset({"question_id", "hypothesis_id", "kind", "statement", "cites"}), _NONE, lambda d: ConcludeOp(
        question_id=_s(d, "question_id"), hypothesis_id=_s(d, "hypothesis_id"), kind=d["kind"],
        statement=_s(d, "statement"), cites=_ids(d, "cites"),
    )),
    "RETRACT": (frozenset({"question_id", "statement", "cites"}), _NONE, lambda d: RetractOp(
        question_id=_s(d, "question_id"), statement=_s(d, "statement"), cites=_ids(d, "cites"),
    )),
    "PARK": (frozenset({"question_id"}), _NONE, lambda d: ParkOp(question_id=_s(d, "question_id"))),
    "ABANDON": (frozenset({"question_id"}), _NONE, lambda d: AbandonOp(question_id=_s(d, "question_id"))),
})
