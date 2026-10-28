# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A judge's reply is cached only when it holds a ruling (Task 17 review): a reply cut at its cap
is paid for, never cached, and asked again on the rerun."""
from __future__ import annotations

from pathlib import Path

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.mind.apd import judging
from anjo.sleep.mind.apd.score import JudgeError

RULING = '{"same_problem": true, "holds": false}'


def test_an_unparsable_reply_is_settled_never_cached_and_asked_again(tmp_path: Path) -> None:
    replies = iter(['{"same_problem": tr', RULING])  # the first cut off mid-thought
    asked: list[str] = []

    def model(prompt: str) -> tuple[str, int, int]:
        asked.append(prompt)
        return next(replies), 1000, 400

    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    rate = judging.Rate(0.75, 3.75)
    cache = tmp_path / "gemini.jsonl"
    judge = judging.CachedJudge(judging.GEMINI, model, cache, ledger, rate)
    with pytest.raises(JudgeError):
        judge.complete("a prompt")
    assert not cache.exists() and ledger.open_holds() == ()
    assert ledger.used(judging.POOLS[judging.GEMINI]) == pytest.approx(rate.usd(1000, 400))  # paid all the same
    again = judging.CachedJudge(judging.GEMINI, model, cache, ledger, rate)  # the rerun
    assert again.complete("a prompt") == RULING and again.complete("a prompt") == RULING
    assert asked == ["a prompt", "a prompt"]  # asked again once, then read from the cache
