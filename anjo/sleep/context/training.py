"""Context C for a training input (Task 5): the snapshot each practice scene's example keeps.

It is built the way a cleared turn's C is (``anjo.sleep.app.context_builder.cleared_context_builder``)
so the weights learn from inputs shaped like the ones they will be served: none of Anjo's memory,
the training brief, and ``evidence_recall`` in ``sft`` mode (no practice item and no item whose
ancestry isn't cleared). Preferences since the brief are left out: the night folds them into the
training brief just before practice, so none are newer.

The night takes the snapshot before tonight's practice exists (the pre-practice brief, and recall
that can't return practice), and the transcript keeps it, so a target is never in its own input.
"""
from __future__ import annotations

from datetime import datetime

from anjo.sleep.context.brief_store import union
from anjo.sleep.context.builder import BriefSource, ComposedContextBuilder, RecallSettings
from anjo.sleep.context.recall_common import RecallEngine, RecallMode
from anjo.sleep.types import Evidence, Probe


class _NoMemory:
    def read(self, user_id: str, message: str) -> str:
        return ""


class _NoPreferences:
    def since(self, user_id: str, since: datetime | None) -> tuple[str, ...]:
        return ()


class SftRecall:
    """A recall engine held to ``sft`` mode, whatever mode is asked."""

    def __init__(self, engine: RecallEngine) -> None:
        self._engine = engine

    def recall(
        self, user_id: str, query: str, k: int, budget_tokens: int, *, mode: RecallMode = "sft",
    ) -> tuple[Evidence, ...]:
        return self._engine.recall(user_id, query, k, budget_tokens, mode="sft")


class TrainingContext:
    """``snapshot(user_id, message)``: C's text (None when it says nothing) and its ancestry."""

    def __init__(self, briefs: BriefSource, recall: RecallEngine, settings: RecallSettings) -> None:
        self._builder = ComposedContextBuilder(
            memory=_NoMemory(), briefs=briefs, preferences=_NoPreferences(),
            recall=SftRecall(recall), settings=settings,
        )

    def snapshot(self, user_id: str, message: str) -> tuple[str | None, frozenset[str] | None]:
        probe = Probe(id="practice", person_id=user_id, kind="new_work", prompt=message,
                      options=None, expected=None, source_ids=())
        composed = self._builder.compose(user_id, probe)
        if composed.brief is None and not composed.found:
            return None, frozenset()
        ancestry = union((
            composed.brief.ancestry if composed.brief is not None else frozenset(),
            *(item.ancestry for item in composed.found),
        ))
        return composed.text, ancestry
