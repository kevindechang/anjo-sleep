# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""evidence_recall (Task 2): top-k inside a token budget, never held-out, SFT mode strict."""
from __future__ import annotations

from datetime import date


from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.context.recall_common import estimate_tokens
from anjo.sleep.tests.sleep_fakes import DAY, SPARK, WONDER, LeakyStore, StoreSpy, ev

QUERY = "long take on the ferry deck"


def _items():
    return [
        ev("a", "long take on the ferry deck, held past the line", minutes=1),
        ev("b", "ferry deck crowd tracking shot", minutes=2),
        ev("c", "notes about sound mixing", minutes=3),
        ev("held", "long take on the ferry deck ferry deck ferry deck", held_out=True, minutes=4),
        ev("prac", "practice: long take on the ferry deck worked", kind="practice",
           ancestry=frozenset({SPARK}), minutes=5),
        ev("hw", "homework on the long take, ferry deck", kind="homework",
           ancestry=frozenset({WONDER}), minutes=6),
        ev("unk", "long take on the ferry deck, unknown source", ancestry=None, minutes=7),
    ]


def _recall(items=None) -> Bm25Recall:
    store = StoreSpy(items or _items())
    return Bm25Recall(lambda user_id: store, until=DAY)


def test_recall_never_returns_held_out():
    got = _recall().recall("p1", QUERY, k=20, budget_tokens=10_000)
    assert "held" not in {e.id for e in got}


def test_recall_never_returns_held_out_even_if_the_store_leaks_it():
    store = LeakyStore(_items())
    got = Bm25Recall(lambda u: store, until=DAY).recall("p1", QUERY, k=20, budget_tokens=10_000)
    assert "held" not in {e.id for e in got}


def test_recall_ranks_by_relevance_and_serves_homework_and_unknown_ancestry():
    got = [e.id for e in _recall().recall("p1", QUERY, k=3, budget_tokens=10_000)]
    assert got[0] in {"a", "prac", "unk", "hw"} and "c" not in got
    assert len(got) == 3
    assert "hw" in [e.id for e in _recall().recall("p1", QUERY, k=10, budget_tokens=10_000)]


def test_recall_respects_token_budget():
    items = [ev(f"i{n}", "ferry deck " + "word " * 40, minutes=n) for n in range(10)]
    one = estimate_tokens(items[0].text)
    got = _recall(items).recall("p1", QUERY, k=10, budget_tokens=one * 3)
    assert 1 <= len(got) <= 3
    assert sum(estimate_tokens(e.text) for e in got) <= one * 3


def test_recall_returns_nothing_for_a_budget_that_fits_no_item():
    assert _recall().recall("p1", QUERY, k=5, budget_tokens=1) == ()


def test_recall_is_deterministic():
    first = _recall().recall("p1", QUERY, k=5, budget_tokens=10_000)
    assert first == _recall().recall("p1", QUERY, k=5, budget_tokens=10_000)


def test_sft_recall_excludes_practice_and_uncleared():
    got = {e.id for e in _recall().recall("p1", QUERY, k=20, budget_tokens=10_000, mode="sft")}
    assert got == {"a", "b"}


def test_sft_recall_admits_hosted_only_when_hosted_may_train():
    store = StoreSpy(_items())
    recall = Bm25Recall(lambda u: store, until=DAY, hosted_in_training=True)
    got = {e.id for e in recall.recall("p1", QUERY, k=20, budget_tokens=10_000, mode="sft")}
    assert got == {"a", "b", "hw"}


def test_recall_ignores_items_after_the_night():
    store = StoreSpy([ev("late", "ferry deck", day=date(2026, 11, 3))] + _items())
    got = Bm25Recall(lambda u: store, until=DAY).recall("p1", QUERY, k=20, budget_tokens=10_000)
    assert "late" not in {e.id for e in got}
