"""The general-skills gate (Task 5): the registered 200 multiple-choice items (MMLU, ARC-Challenge
and HellaSwag), scored by log-likelihood. Each choice is the assistant's reply to the question,
and the choice the model gives the highest summed log-probability is its answer. The night is
rejected when the checkpoint's accuracy is more than 2.0 points below the base model's.

The base model's answers are kept across nights when the app passes a ``cache`` (``base_cache``),
keyed on every item's question and choices and the scoring rule; the checkpoint's are asked every
night.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from anjo.sleep.gates import GateThresholds, in_order, listed, load_thresholds
from anjo.sleep.gates.base_cache import BaseCache, base_side
from anjo.sleep.probes.clients import TokenLogprobs
from anjo.sleep.sets.loader import GeneralItem, load_general
from anjo.sleep.types import CheckpointRef, GateResult

# How an answer is read (a key part): the summed log-probability of each choice as the reply, at
# temperature 0 with one generated token, the first choice on a tie.
SCORING: Final = "continuation_logprobs:sum:t0:max_tokens1:first_on_tie"


def answer(client: TokenLogprobs, adapter: CheckpointRef | None, item: GeneralItem) -> int:
    """The index of the choice the model finds most likely (the first on a tie)."""
    messages = [{"role": "user", "content": item.question}]
    scores = [sum(client.continuation_logprobs(adapter, messages, choice)) for choice in item.choices]
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


class GeneralGate:
    """``check(ckpt)`` for the general-skills gate; the base model's answers are kept per run."""

    def __init__(
        self, client: TokenLogprobs, *, items: Sequence[GeneralItem] | None = None,
        thresholds: GateThresholds | None = None, cache: BaseCache | None = None,
    ) -> None:
        self._client = client
        self._items = tuple(items) if items is not None else load_general()
        self._threshold = (thresholds or load_thresholds()).general_points
        self._cache = cache
        self._base: list[bool] | None = None

    def _answers(self, adapter: CheckpointRef | None) -> list[int]:
        return in_order(lambda item: answer(self._client, adapter, item), self._items)

    def _correct(self, picks: Sequence[int]) -> list[bool]:
        return [pick == item.answer for pick, item in zip(picks, self._items, strict=True)]

    def _key(self) -> dict[str, Any]:
        return {"scoring": SCORING, "items": [[i.id, i.question, list(i.choices)] for i in self._items]}

    def _valid(self, kept: list[Any]) -> bool:
        return len(kept) == len(self._items) and all(
            isinstance(pick, int) and not isinstance(pick, bool) and 0 <= pick < len(item.choices)
            for pick, item in zip(kept, self._items, strict=True))

    def check(self, ckpt: CheckpointRef) -> GateResult:
        if self._base is None:
            self._base = self._correct(base_side(self._cache, "general", self._key, lambda: self._answers(None), self._valid))
        mine = self._correct(self._answers(ckpt))
        total = len(self._items)
        lost = 100.0 * (sum(self._base) - sum(mine)) / total
        passed = lost <= self._threshold
        newly_wrong = [item.id for item, b, m in zip(self._items, self._base, mine, strict=True) if b and not m]
        reason = None if passed else (
            f"accuracy fell {lost:.1f} points (over {self._threshold:g}); newly wrong: {listed(newly_wrong)}"
        )
        return GateResult("general", passed, lost, self._threshold, reason)
