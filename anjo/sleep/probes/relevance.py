"""The relevance labeling pass (Task 16): which studied targets were relevant to held-out work.

Muse Spark labels each (target, held-out ``new_work`` item) pair once with ``relevance_v1.md`` at
temperature 0. The pairs are every target any decider picked, crossed with the life's held-out items:
who picked a target is never an input, so the labels are blind to the decider. The labels are then
frozen: a read-only mapping whose SHA-256 (with the prompt's) is fixed when they are made, and a
label changed afterwards no longer matches its hash. The report scores only labels whose hash was
registered before any decider was scored (``verify``).

The held-out items reach only this pass and the report; no decider ever sees them.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.template import render
from anjo.sleep.personas.model import NewWork

RELEVANCE_PROMPT: Final = Path(__file__).parent / "prompts" / "relevance_v1.md"
RELEVANCE_MODEL: Final = "muse-spark"
RELEVANCE_TEMPERATURE: Final = 0.0
RELEVANCE_TIMEOUT_S: Final = 180.0

PairKey = tuple[str, str]  # (target id, held-out item id)


class LabelError(ValueError):
    """A pair the frozen labels don't hold, or labels that don't match their hash."""


@dataclass(frozen=True)
class Pair:
    """One (target, held-out item) pair to label: the target's id and label, and the item."""

    target_id: str
    target_label: str
    item: NewWork

    @property
    def key(self) -> PairKey:
        return self.target_id, self.item.id


def pairs_to_label(picked: Iterable[str], labels: Mapping[str, str], items: Sequence[NewWork]) -> tuple[Pair, ...]:
    """Each (target, item) pair once, for every target picked, sorted by key. ``labels`` gives each
    target's label by id."""
    targets = sorted(set(picked))
    missing = [t for t in targets if t not in labels]
    if missing:
        raise LabelError(f"no label for picked targets {missing}")
    return tuple(
        Pair(target_id=t, target_label=labels[t], item=item)
        for t in targets for item in sorted(items, key=lambda w: w.id)
    )


def digest_of(prompt_sha256: str, labels: Mapping[PairKey, bool], unreadable: frozenset[PairKey]) -> str:
    document = {
        "prompt_sha256": prompt_sha256,
        "labels": sorted([target, item, bool(value)] for (target, item), value in labels.items()),
        "unreadable": sorted([target, item] for target, item in unreadable),
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenLabels:
    """The labels, read-only, with their hash. An unreadable answer is labeled not relevant and
    listed in ``unreadable``."""

    prompt_sha256: str
    labels: Mapping[PairKey, bool] = field(hash=False)
    unreadable: frozenset[PairKey]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        if digest_of(self.prompt_sha256, self.labels, self.unreadable) != self.digest:
            raise LabelError("the labels don't match their hash")

    def relevant(self, target_id: str, item_id: str) -> bool:
        key = (target_id, item_id)
        if key not in self.labels:
            raise LabelError(f"pair {key} is not labeled")
        return self.labels[key]


def freeze(prompt_sha256: str, labels: Mapping[PairKey, bool], unreadable: frozenset[PairKey]) -> FrozenLabels:
    return FrozenLabels(prompt_sha256, labels, unreadable, digest_of(prompt_sha256, labels, unreadable))


def prompt_sha256() -> str:
    return hashlib.sha256(RELEVANCE_PROMPT.read_bytes()).hexdigest()


def label_pairs(pairs: Sequence[Pair], chat: ChatFn) -> FrozenLabels:
    """Ask Muse Spark once per pair, then freeze the labels."""
    keys = [pair.key for pair in pairs]
    if len(set(keys)) != len(keys):
        raise LabelError("a pair is labeled once")
    template = RELEVANCE_PROMPT.read_text(encoding="utf-8")
    labels: dict[PairKey, bool] = {}
    unreadable: set[PairKey] = set()
    for pair in pairs:
        prompt = render(template, {"TARGET": pair.target_label, "PROJECT": pair.item.parent_project,
                                   "ITEM": pair.item.text})
        reply = chat(RELEVANCE_MODEL, [{"role": "user", "content": prompt}], None, RELEVANCE_TIMEOUT_S,
                     temperature=RELEVANCE_TEMPERATURE)
        verdict = parse_relevance(reply.text)
        labels[pair.key] = bool(verdict)
        if verdict is None:
            unreadable.add(pair.key)
    return freeze(prompt_sha256(), labels, frozenset(unreadable))


def parse_relevance(text: str) -> bool | None:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        data = json.loads(body)
    except ValueError:
        return None
    value = data.get("relevant") if isinstance(data, dict) else None
    return value if isinstance(value, bool) else None


def verify(labels: FrozenLabels, registered: str) -> None:
    """Refuse labels whose hash isn't the one registered before scoring."""
    if digest_of(labels.prompt_sha256, labels.labels, labels.unreadable) != labels.digest:
        raise LabelError("the labels don't match their hash")
    if labels.digest != registered:
        raise LabelError("the labels are not the registered ones")
