"""Context C for a reply (Task 2): one builder, so arms 2C and 4 get byte-identical text.

``ComposedContextBuilder`` is the ``ContextBuilder`` port over four sources it is given: Anjo's
memory for the turn, the served brief, the person's explicit preferences and corrections
recorded since that brief was compiled (so a same-day request works the same day), and
``evidence_recall``. The app wires the sources to private Anjo (``anjo.sleep.app.context_builder``);
the standalone example wires public ones. Training-time context is Task 5's and is not built here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from anjo.sleep.context.recall_common import RecallEngine
from anjo.sleep.types import Evidence, Probe

NONE: Final = "none"


class MemorySource(Protocol):
    def read(self, user_id: str, message: str) -> str:
        """Anjo's memory for a turn, as text ("" when it holds nothing)."""


@dataclass(frozen=True)
class StoredBrief:
    """The served brief and the moment it was compiled."""

    text: str
    compiled_at: datetime


class BriefSource(Protocol):
    def served(self, user_id: str) -> StoredBrief | None:
        """The latest served brief, or ``None`` before the first night."""


class PreferenceSource(Protocol):
    def since(self, user_id: str, since: datetime | None) -> tuple[str, ...]:
        """The person's explicit preferences and corrections recorded after ``since`` (all of
        them when ``None``), oldest first."""


@dataclass(frozen=True)
class RecallSettings:
    """``evidence_recall``'s k and token budget: set on development probes, frozen at the freeze."""

    k: int
    budget_tokens: int


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip() or NONE}"


def _evidence_lines(items: tuple[Evidence, ...]) -> str:
    return "\n".join(f"- [{item.id}] {item.text}" for item in items)


class ComposedContextBuilder:
    """``ContextBuilder`` over the four sources. ``build`` only reads."""

    def __init__(
        self,
        *,
        memory: MemorySource,
        briefs: BriefSource,
        preferences: PreferenceSource,
        recall: RecallEngine,
        settings: RecallSettings,
    ) -> None:
        self._memory = memory
        self._briefs = briefs
        self._preferences = preferences
        self._recall = recall
        self._settings = settings

    def recall(
        self, user_id: str, query: str, k: int, budget_tokens: int,
    ) -> tuple[Evidence, ...]:
        return self._recall.recall(user_id, query, k, budget_tokens)

    def build(self, user_id: str, probe: Probe) -> str:
        brief = self._briefs.served(user_id)
        since = self._preferences.since(user_id, brief.compiled_at if brief else None)
        found = self.recall(user_id, probe.prompt, self._settings.k, self._settings.budget_tokens)
        return "\n\n".join((
            _section("What you remember about this person", self._memory.read(user_id, probe.prompt)),
            _section("Their brief", brief.text if brief else ""),
            _section("Preferences and corrections since the brief",
                     "\n".join(f"- {text}" for text in since)),
            _section("Evidence that may help with this message", _evidence_lines(found)),
        ))
