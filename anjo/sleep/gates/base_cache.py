"""The base model's gate answers, kept across nights (§5.5, the gate-set lever).

Every gate compares a checkpoint with the base model on the same items. The base model is fixed:
on the same server (the model's pinned revision, its precision, the vLLM release, the rendering,
the date every gate prompt carries and the reply cap, all in ``serving``) the same items, seeds,
temperature and max_tokens give the same answers. So a gate asks the base once per key and reads
its answers back on every later night; only the checkpoint's side is asked every night.

- **The key** is a SHA-256 over everything the base side depends on: ``serving``, the gate's name,
  this cache's ``VERSION`` and the gate's own parts (its items, seeds, temperature, max_tokens and
  any prompt it renders). Any change is a new key, so the base is asked again; the old key stays
  kept for a night that goes back to it. ``serving`` is what the server itself reports, read once
  a run at the first gate (the app's ``gate_cache``); when it can't be read, or names no revision,
  it is None and every base answer is asked and none is kept: a miss, never a hit.
- **Only the base side is kept.** The gates hand this cache nothing but the base model's answers;
  the checkpoint's side never reaches it. The safety gate keeps the base model's replies, never
  their flags, so tonight's reply check reads them afresh. The collapse gate keeps its flags and,
  beside them (``read`` and ``keep``, written only when the base was just asked), the base model's
  replies, so a later night can show one next to the checkpoint's without asking the base again.
- **A kept answer is checked before use** (``valid``): one of the wrong shape is asked again.
- Up to vLLM's batch nondeterminism, which a base side computed once per run already carries, a
  kept answer is the answer the base model gives.

``JsonlBaseStore`` keeps the answers in one append-only JSON Lines file through a codec: the app
passes its at-rest encryption and the sleep data folder; tests pass ``PLAIN``.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TypeVar

from anjo.sleep.evidence.jsonl import Codec, append_records, canonical, read_records

VERSION: Final = 1
FILE_NAME: Final = "gate_base.jsonl"
T = TypeVar("T")
logger = logging.getLogger(__name__)


class BaseStore(Protocol):
    def get(self, key: str) -> list[Any] | None:
        """The answers last kept under ``key``, or None."""

    def put(self, key: str, gate: str, answers: Sequence[Any]) -> None:
        """Keep ``answers`` under ``key``."""


class JsonlBaseStore:
    """Every kept base side, one record a key, in one file."""

    def __init__(self, path: Path, *, codec: Codec) -> None:
        self.path = path
        self._codec = codec

    def get(self, key: str) -> list[Any] | None:
        found = [r for r in read_records(self.path, self._codec) if r.get("key") == key]
        answers = found[-1].get("answers") if found else None
        return answers if isinstance(answers, list) else None

    def put(self, key: str, gate: str, answers: Sequence[Any]) -> None:
        append_records(self.path, self._codec, [{"key": key, "gate": gate, "answers": list(answers)}])


def base_key(gate: str, serving: str, parts: Mapping[str, Any]) -> str:
    """The SHA-256 of everything ``gate``'s base side depends on."""
    material = canonical({"version": VERSION, "gate": gate, "serving": serving, "parts": dict(parts)})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def digest(text: str) -> str:
    """An item's text in a key without its words (the safety cases')."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BaseCache:
    """A gate's view of the kept base sides: the store and the server's fingerprint."""

    store: BaseStore
    serving: str | Callable[[], str | None]  # the server's fingerprint, or how to ask for it

    def _key(self, gate: str, parts: Mapping[str, Any]) -> str | None:
        serving = self.serving() if callable(self.serving) else self.serving
        return None if serving is None else base_key(gate, serving, parts)

    def read(self, gate: str, parts: Mapping[str, Any], valid: Callable[[list[Any]], bool]) -> list[Any] | None:
        """What is kept under this key, never computed: None when nothing valid is."""
        key = self._key(gate, parts)
        kept = None if key is None else self.store.get(key)
        return kept if kept is not None and valid(kept) else None

    def keep(self, gate: str, parts: Mapping[str, Any], answers: Sequence[Any]) -> None:
        """Keep base answers already in hand under this key (nothing without a fingerprint)."""
        key = self._key(gate, parts)
        if key is not None:
            self.store.put(key, gate, answers)

    def answers(
        self, gate: str, parts: Mapping[str, Any], compute: Callable[[], list[T]], valid: Callable[[list[Any]], bool],
    ) -> list[T]:
        """The base side under this key: kept, or computed now and kept. With no fingerprint it is
        computed and not kept."""
        serving = self.serving() if callable(self.serving) else self.serving
        if serving is None:
            logger.info("%s gate: the server's fingerprint is unknown, so the base model is asked", gate)
            return compute()
        key = base_key(gate, serving, parts)
        kept = self.store.get(key)
        if kept is not None and valid(kept):
            logger.info("%s gate: the base model's answers read back (key %s)", gate, key[:12])
            return kept
        logger.info("%s gate: no base answers under key %s, so the base model is asked", gate, key[:12])
        fresh = compute()
        self.store.put(key, gate, fresh)
        return fresh


def base_side(cache: BaseCache | None, gate: str, parts: Callable[[], Mapping[str, Any]],
              compute: Callable[[], list[T]], valid: Callable[[list[Any]], bool]) -> list[T]:
    """``compute()`` without a cache; otherwise the kept base side (``parts`` built only then)."""
    if cache is None:
        return compute()
    return cache.answers(gate, parts(), compute, valid)
