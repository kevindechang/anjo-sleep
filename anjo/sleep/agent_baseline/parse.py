"""Reads the baseline agent's raw reply as one step, leniently. Pure.

A model wraps JSON in prose or a fence, or cases the operation's name. The parser takes the first
JSON object in the reply that names one of the agent's four operations, keeps only what that
operation uses, and takes a lone id where a list of ids belongs. Anything else is ``None``: a reply
that isn't one of its operations ends the pass, which is what a conventional agent does rather than
guess what was meant.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final

from anjo.sleep.agent_baseline.types import AgentStep

OPS: Final = frozenset({"recall", "research", "note", "done"})
_DECODER: Final = json.JSONDecoder()


def _objects(text: str) -> Iterator[Mapping[str, Any]]:
    """Every JSON object in ``text``, in order, each from its first character."""
    at = 0
    while True:
        start = text.find("{", at)
        if start < 0:
            return
        try:
            value, end = _DECODER.raw_decode(text, start)
        except ValueError:
            at = start + 1
            continue
        if isinstance(value, Mapping):
            yield value
        at = end


def _text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _cites(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def parse_step(text: str) -> AgentStep | None:
    """The one operation in a step's reply, or ``None`` when the reply holds none it can run."""
    for data in _objects(text):
        name = data.get("op")
        if not isinstance(name, str) or name.strip().casefold() not in OPS:
            continue
        op = name.strip().casefold()
        if op in ("recall", "research"):
            query = _text(data.get("query"))
            if query is None:
                return None
            return AgentStep("recall", query=query) if op == "recall" else AgentStep("research", query=query)
        if op == "note":
            said = _text(data.get("text"))
            return None if said is None else AgentStep("note", text=said, cites=_cites(data.get("cites")))
        return AgentStep("done")
    return None
