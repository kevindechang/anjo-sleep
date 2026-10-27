"""Reads the thinker's raw reply into operations, leniently (Task 17). Pure.

A model wraps JSON in prose or code fences, changes a name's case, adds a key or writes a number
as text. The parser takes the first JSON value in the reply that holds what it wants, keeps only
the operation's own fields and coerces what is safe to coerce:

- the operation's name, in any case;
- a lone id where a list of ids belongs;
- a number written as text, and a whole number of days written as a float;
- extra keys, on the operation and on each hypothesis, dropped.

Then the strict reference codec (``ops.op_from_json``) decides. Anything it refuses is
``Rejected("malformed")``, which the loop logs and which costs its cycle.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import fields
from typing import Any, Final

from anjo.sleep.mind.inquiry.types import OP_TYPES, NoticeOp, Op, OpFormatError, Rejected, op_from_json

_DECODER: Final = json.JSONDecoder()
_ID_LISTS: Final = frozenset({"target_ids", "origin_evidence", "cites"})
_MAPS: Final = frozenset({"p_true_given", "likelihoods", "credences"})
_NUMBERS: Final = frozenset({"catch_all"})
MALFORMED: Final = Rejected("malformed")


def _values(text: str) -> Iterator[object]:
    """Every JSON object or array in ``text``, in order, each from its first character."""
    at = 0
    while True:
        starts = [n for n in (text.find("{", at), text.find("[", at)) if n >= 0]
        if not starts:
            return
        start = min(starts)
        try:
            value, end = _DECODER.raw_decode(text, start)
        except ValueError:
            at = start + 1
            continue
        yield value
        at = end


def _number(value: object) -> object:
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def _whole(value: object) -> object:
    number = _number(value)
    if isinstance(number, float) and number.is_integer():
        return int(number)
    return number


def _field(key: str, value: object) -> object:
    if key in _ID_LISTS and isinstance(value, str):
        return [value]
    if key in _MAPS and isinstance(value, Mapping):
        return {str(k): _number(v) for k, v in value.items()}
    if key in _NUMBERS:
        return _number(value)
    if key == "horizon_days":
        return _whole(value)
    if key == "hypotheses" and isinstance(value, list):
        return [
            {"statement": item.get("statement"), "credence": _number(item.get("credence"))}
            if isinstance(item, Mapping) else item
            for item in value
        ]
    return value


def _clean(data: object) -> dict[str, Any] | None:
    """The operation ``data`` names, with only its own fields; ``None`` if it names none."""
    if not isinstance(data, Mapping):
        return None
    name = data.get("op")
    if not isinstance(name, str):
        return None
    name = name.strip().upper()
    kind = next((cls for op_name, cls in OP_TYPES.items() if op_name == name), None)
    if kind is None:
        return None
    own = {f.name for f in fields(kind)}
    return {"op": name, **{key: _field(key, value) for key, value in data.items() if key in own}}


def _decode(data: object) -> Op | None:
    cleaned = _clean(data)
    if cleaned is None:
        return None
    try:
        return op_from_json(cleaned)
    except OpFormatError:
        return None


def parse_op(text: str) -> Op | Rejected:
    """The one operation in an inquiry cycle's reply: the first JSON object (or the first object
    of an array) that reads as an operation, or ``Rejected("malformed")``."""
    for value in _values(text):
        for candidate in value if isinstance(value, list) else [value]:
            op = _decode(candidate)
            if op is not None:
                return op
    return MALFORMED


def parse_notices(text: str) -> tuple[NoticeOp, ...] | Rejected:
    """The notice pass's reply: the first JSON array, every element a NOTICE (an empty array is
    nothing noticed); a lone NOTICE object counts as an array of one. Anything else is
    ``Rejected("malformed")``."""
    for value in _values(text):
        items = value if isinstance(value, list) else [value]
        if isinstance(value, dict) and _clean(value) is None:
            continue
        ops = [_decode(item) for item in items]
        if all(isinstance(op, NoticeOp) for op in ops):
            return tuple(op for op in ops if isinstance(op, NoticeOp))
        return MALFORMED
    return MALFORMED
