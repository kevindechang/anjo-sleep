# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Fakes for the arms, scoring, brief and recall tests: no network, no model."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.model_api import ChatResult
from anjo.sleep.types import CheckpointRef, Evidence, EvidenceKind, Probe, ProbeKind

DAY = date(2026, 10, 28)
T0 = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)
WONDER = "muse-wonder-1.0"


def ev(
    id: str, text: str = "", *, kind: EvidenceKind = "lesson", held_out: bool = False,
    ancestry: frozenset[str] | None = frozenset(), minutes: int = 0, person: str = "p1",
    day: date = DAY,
) -> Evidence:
    return Evidence(
        id=id, person_id=person, day=day, kind=kind, text=text or f"text of {id}",
        source=f"src:{id}", held_out=held_out, created_at=T0 + timedelta(minutes=minutes),
        model=None, ancestry=ancestry,
    )


def probe(
    id: str = "q1", kind: ProbeKind = "new_work", prompt: str = "How should I cut the ferry scene?",
    *, person: str = "p1", options: tuple[str, str] | None = None, expected: str | None = None,
    source_ids: tuple[str, ...] = (),
) -> Probe:
    return Probe(id, person, kind, prompt, options, expected, source_ids)


def taste(id: str = "t1", person: str = "p1", source: str = "item-1") -> Probe:
    options = ("hold on her face", "cut on the last word")
    return Probe(id, person, "taste_ab", "How should the cut come?", options, options[0], (source,))


def ckpt(person: str, night: int = 1) -> CheckpointRef:
    return CheckpointRef(f"{person}-n{night}", person, DAY, f"uri://{person}/{night}", None)


@dataclass
class FakeModel:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, adapter, messages, *, temperature):
        self.calls.append({"adapter": adapter, "messages": messages, "temperature": temperature})
        name = adapter.id if adapter else "base"
        words = sum(len(m["content"].split()) for m in messages)
        return ChatResult(text=f"reply from {name}", tool_calls=(), model=name,
                          input_tokens=words, output_tokens=3, finish_reason="stop")


@dataclass
class FakeRegistry:
    adapters: dict[str, CheckpointRef] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)

    def last_promoted(self, person_id):
        self.asked.append(person_id)
        return self.adapters.get(person_id)


@dataclass
class FakeContexts:
    """A builder that returns different text on every call, so only one call gives equal text."""

    builds: int = 0
    text: str = "CTX"

    def build(self, user_id, probe):
        self.builds += 1
        return f"{self.text} for {user_id}/{probe.id} build#{self.builds}"

    def recall(self, user_id, query, k, budget_tokens):
        return ()


@dataclass
class FakeLogprobs:
    """Token log-probabilities for a label. ``prefers`` picks the label by the messages shown."""

    prefers: Callable[[Any, list[dict[str, str]]], str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def continuation_logprobs(self, adapter, messages, continuation) -> Sequence[float]:
        self.calls.append({"adapter": adapter, "messages": messages, "continuation": continuation})
        return (-0.1,) if self.prefers(adapter, messages) == continuation else (-2.0,)


def options(p: Probe) -> tuple[str, str]:
    assert p.options is not None
    return p.options


class StoreSpy:
    """An EvidenceStore over a list that fails the test on any write."""

    def __init__(self, items: list[Evidence]) -> None:
        self.items = items
        self.writes = 0

    def add(self, items):
        self.writes += 1

    def stored(self, until):
        return [e for e in self.items if e.day <= until and not e.held_out]

    def for_day(self, day):
        return [e for e in self.items if e.day == day]

    def trainable(self, until):
        return []

    def held_out(self, day):
        return [e for e in self.items if e.held_out and e.day == day]

    def search(self, query, as_of, k):
        return []

    def export(self):
        return {}


def brief_text(body: str = "") -> str:
    heads = ["The project and its constraints", "Stated preferences", "Corrections",
             "What homework found, with sources", "What practice showed"]
    return "\n\n".join(f"## {h}\n{body or 'none'}" for h in heads)


SPARK = SPARK_OPEN


class LeakyStore(StoreSpy):
    """A store whose ``stored`` wrongly returns held-out items too."""

    def stored(self, until):
        return list(self.items)
