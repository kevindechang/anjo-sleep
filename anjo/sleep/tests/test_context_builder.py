# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""Context C (Task 2): memory, the served brief, preferences since the brief, and recall."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from anjo.sleep.context.builder import ComposedContextBuilder, RecallSettings, StoredBrief
from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.tests.sleep_fakes import DAY, T0, StoreSpy, brief_text, ev, probe


@dataclass
class Memory:
    text: str = "she is cutting a short about a night ferry"

    def read(self, user_id: str, message: str) -> str:
        return self.text


@dataclass
class Briefs:
    compiled_at: datetime = T0

    def served(self, user_id: str) -> StoredBrief | None:
        return StoredBrief(brief_text("likes long takes"), self.compiled_at)


@dataclass
class Prefs:
    rows: tuple[tuple[datetime, str], ...] = ()
    asked_since: datetime | None = None

    def since(self, user_id: str, since: datetime | None) -> tuple[str, ...]:
        self.asked_since = since
        return tuple(text for at, text in self.rows if since is None or at > since)


def make_builder(prefs: Prefs | None = None) -> ComposedContextBuilder:
    store = StoreSpy([ev("a", "long take on the ferry deck", minutes=1), ev("b", "sound mix", minutes=2)])
    return ComposedContextBuilder(
        memory=Memory(), briefs=Briefs(), preferences=prefs or Prefs(),
        recall=Bm25Recall(lambda u: store, until=DAY), settings=RecallSettings(k=5, budget_tokens=2000),
    )


def test_context_holds_memory_brief_and_recalled_evidence():
    text = make_builder().build("p1", probe(prompt="long take on the ferry deck?"))
    assert "night ferry" in text and "likes long takes" in text
    assert "[a] long take on the ferry deck" in text


def test_same_day_tone_request_reaches_context():
    prefs = Prefs(rows=((T0 - timedelta(hours=1), "old request"), (T0 + timedelta(hours=2), "keep it blunt")))
    text = make_builder(prefs).build("p1", probe())
    assert "keep it blunt" in text
    assert "old request" not in text  # already folded into the brief
    assert prefs.asked_since == T0


def test_build_is_deterministic_for_the_same_stores():
    builder = make_builder()
    assert builder.build("p1", probe()) == builder.build("p1", probe())


def test_empty_parts_are_marked_not_dropped():
    store = StoreSpy([])

    @dataclass
    class NoBrief:
        def served(self, user_id: str) -> StoredBrief | None:
            return None

    builder = ComposedContextBuilder(
        memory=Memory(""), briefs=NoBrief(), preferences=Prefs(),
        recall=Bm25Recall(lambda u: store, until=DAY), settings=RecallSettings(k=3, budget_tokens=100),
    )
    text = builder.build("p1", probe())
    assert text.count("none") >= 3
