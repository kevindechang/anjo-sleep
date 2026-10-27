"""Context C on private Anjo (Task 2): the memory read and the person's own preferences.

The engine core builds context from four sources (``anjo.sleep.context.builder``). Here are the
two that need private Anjo, and the factory that wires all four:

- memory: ``anjo.memory.long_term.query_memories``, Anjo's existing memory read;
- preferences and corrections recorded since the brief: ``list_explicit_preferences`` for the
  person's explicit preferences, and their stored corrections read only through
  ``training_eligible`` (the same gate extraction uses), so a request made today reaches
  today's reply.

Both read live, so a request is in the context as soon as Anjo stores it. Anjo stores some of them
only after a session ends; those reach the context from the next session (spec 2.2).

``cleared_context_builder`` is the same builder for a cleared session: no memory, the training
brief, recall over items that may train (``sft`` mode) with homework left out, and only the
preferences and corrections in the person's own words.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Final

from anjo.sleep.app.flags import require_sleep_host
from anjo.sleep.app.recall import Embed, MiniLmRecall, StoredEvidence
from anjo.sleep.app.storage import evidence_store
from anjo.sleep.context.builder import BriefSource, ComposedContextBuilder, RecallSettings, StoredBrief
from anjo.sleep.context.recall_common import RecallMode
from anjo.sleep.types import Evidence

MEMORY_RESULTS: Final = 4  # Anjo's own default for a turn's memory
# Served turns and arms 2C and 4 share these, so their C matches. Set on development probes.
RECALL: Final = RecallSettings(k=6, budget_tokens=1200)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


class AnjoMemory:
    """Anjo's memory for a turn, one line per remembered session or episode."""

    def read(self, user_id: str, message: str) -> str:
        from anjo.memory import long_term

        found = long_term.query_memories(message, user_id, n_results=MEMORY_RESULTS)
        return "\n".join(f"- {summary}" for _score, summary in found)


class NoMemory:
    """A cleared session's memory: none of Anjo's."""

    def read(self, user_id: str, message: str) -> str:
        return ""


class NoBrief:
    """A brief store before one is set: no brief."""

    def served(self, user_id: str) -> StoredBrief | None:
        return None


class AnjoPreferences:
    """The person's explicit preferences and eligible corrections, newer than a moment. With
    ``own_words_only``, a preference with no receipt (stored as a field and value) is left out."""

    def __init__(self, *, own_words_only: bool = False) -> None:
        self._own_words_only = own_words_only

    def since(self, user_id: str, since: datetime | None) -> tuple[str, ...]:
        found = [*self._preferences(user_id), *self._corrections(user_id)]
        fresh = sorted(
            (at, text) for at, text in found if since is None or at > _aware(since)
        )
        return tuple(dict.fromkeys(text for _, text in fresh))

    def _preferences(self, user_id: str) -> list[tuple[datetime, str]]:
        from anjo.core import personal_preferences

        out: list[tuple[datetime, str]] = []
        for item in personal_preferences.list_explicit_preferences(user_id):
            receipt = str(item.get("receipt") or "").strip()
            if not receipt and self._own_words_only:
                continue
            text = receipt or f"{item['field']}: {item['value']}"
            out.append((_aware(datetime.fromisoformat(str(item["created_at"]))), text))
        return out

    @staticmethod
    def _corrections(user_id: str) -> list[tuple[datetime, str]]:
        from anjo.core import learning_evidence
        from anjo.core.db import get_db

        rows = get_db().execute(
            "SELECT * FROM learning_evidence WHERE user_id = ? AND kind = 'explicit_preference' "
            "ORDER BY created_at, rowid",
            (user_id,),
        ).fetchall()
        out: list[tuple[datetime, str]] = []
        for row in rows:
            if not learning_evidence.training_eligible(row):
                continue
            patch = learning_evidence.correction_of(row)
            quote = patch["quote"].strip() if patch is not None else ""
            if quote:
                out.append((_aware(datetime.fromisoformat(row["created_at"])), quote))
        return out


def anjo_context_builder(
    briefs: BriefSource,
    *,
    until: date,
    settings: RecallSettings,
    evidence: Callable[[str], StoredEvidence] = evidence_store,
    embed: Embed | None = None,
) -> ComposedContextBuilder:
    """The ``ContextBuilder`` for served turns and arms 2C and 4 on the dev host. ``briefs`` is
    the served-brief store (Task 5)."""
    require_sleep_host()
    return ComposedContextBuilder(
        memory=AnjoMemory(),
        briefs=briefs,
        preferences=AnjoPreferences(),
        recall=MiniLmRecall(evidence, until=until, embed=embed),
        settings=settings,
    )


@dataclass(frozen=True)
class _WithoutHomework:
    store: StoredEvidence

    def stored(self, until: date) -> list[Evidence]:
        return [item for item in self.store.stored(until) if item.kind != "homework"]


class ClearedRecall:
    """Recall for a cleared session: the training-input recall (``sft``: no practice, cleared
    ancestry only) over the person's evidence without homework, whatever mode is asked."""

    def __init__(
        self, evidence: Callable[[str], StoredEvidence], *, until: date, embed: Embed | None = None,
    ) -> None:
        self._ranker = MiniLmRecall(
            lambda user_id: _WithoutHomework(evidence(user_id)), until=until, embed=embed,
        )

    def recall(
        self, user_id: str, query: str, k: int, budget_tokens: int, *, mode: RecallMode = "sft",
    ) -> tuple[Evidence, ...]:
        return self._ranker.recall(user_id, query, k, budget_tokens, mode="sft")


def cleared_context_builder(
    briefs: BriefSource,
    *,
    until: date,
    settings: RecallSettings,
    evidence: Callable[[str], StoredEvidence] = evidence_store,
    embed: Embed | None = None,
) -> ComposedContextBuilder:
    """The ``ContextBuilder`` for a cleared session. ``briefs`` is the training-brief store
    (Task 5). Built only while hosted outputs can't train, as cleared sessions are."""
    require_sleep_host()
    return ComposedContextBuilder(
        memory=NoMemory(),
        briefs=briefs,
        preferences=AnjoPreferences(own_words_only=True),
        recall=ClearedRecall(evidence, until=until, embed=embed),
        settings=settings,
    )
