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
    """A brief, the moment it was compiled, and the models it derives from (``None`` when the
    store does not know, so text built from it never trains)."""

    text: str
    compiled_at: datetime
    ancestry: frozenset[str] | None = None


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


@dataclass(frozen=True)
class ComposedContext:
    """The four sources as one build read them; ``text`` is context C."""

    memory: str
    brief: StoredBrief | None
    since: tuple[str, ...]
    found: tuple[Evidence, ...]

    @property
    def text(self) -> str:
        return "\n\n".join((
            _section("What you remember about this person", self.memory),
            _section("Their brief", self.brief.text if self.brief else ""),
            _section("Preferences and corrections since the brief",
                     "\n".join(f"- {text}" for text in self.since)),
            _section("Evidence that may help with this message", _evidence_lines(self.found)),
        ))


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

    def compose(self, user_id: str, probe: Probe) -> ComposedContext:
        """What ``build`` reads for ``probe``, each source once."""
        brief = self._briefs.served(user_id)
        return ComposedContext(
            memory=self._memory.read(user_id, probe.prompt),
            brief=brief,
            since=self._preferences.since(user_id, brief.compiled_at if brief else None),
            found=self.recall(user_id, probe.prompt, self._settings.k, self._settings.budget_tokens),
        )

    def build(self, user_id: str, probe: Probe) -> str:
        return self.compose(user_id, probe).text
