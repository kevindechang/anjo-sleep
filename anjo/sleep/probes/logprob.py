"""``taste_ab`` by log-probability forced choice (Task 2, spec 5.1).

The two options are shown as "A" and "B"; the arm's score is the log-probability it gives each
label as its answer. That is repeated with the options swapped. If the same option wins both
times it is the arm's choice; if the order decides, the arm has no choice, and any comparison
with that arm is a tie (spec 5.1: "an order flip is a tie"; ``score.taste_outcome``).
There is no judge and no sampling, so the result is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from anjo.sleep.probes.clients import TokenLogprobs
from anjo.sleep.types import CheckpointRef, Probe

LABELS: Final = ("A", "B")
ORDERS: Final = ((0, 1), (1, 0))
INSTRUCTION: Final = "Answer with the letter of your choice only."


@dataclass(frozen=True)
class TasteChoice:
    """One arm's forced choice. ``order_1`` and ``order_2`` are the log-probabilities of the
    labels ("A", "B") with the options shown in order and swapped. ``picked`` is the option text
    chosen in both orders, or ``None`` when the order flipped the answer."""

    order_1: tuple[float, float]
    order_2: tuple[float, float]
    picked: str | None

    @property
    def flipped(self) -> bool:
        return self.picked is None

    def correct(self, expected: str | None) -> bool | None:
        """Whether the expected option won both orders; ``None`` for a flip (no choice)."""
        return None if self.picked is None else self.picked == expected


def present(
    probe: Probe, order: tuple[int, int], context: str | None,
) -> list[dict[str, str]]:
    """The messages for one order: the context (if any) as the system turn, then the question
    with the options under the labels."""
    if probe.options is None:
        raise ValueError(f"{probe.id} has no options")
    first, second = (probe.options[index] for index in order)
    question = f"{probe.prompt}\n\nA. {first}\nB. {second}\n\n{INSTRUCTION}"
    messages = [{"role": "user", "content": question}]
    return [{"role": "system", "content": context}, *messages] if context else messages


def _preferred(logprobs: tuple[float, float], order: tuple[int, int]) -> int | None:
    """The option index the labels favour, or ``None`` when they are equal."""
    label_a, label_b = logprobs
    if label_a == label_b:
        return None
    return order[0] if label_a > label_b else order[1]


def forced_choice(
    client: TokenLogprobs, adapter: CheckpointRef | None, context: str | None, probe: Probe,
) -> TasteChoice:
    """Score ``probe``'s two options in both orders with ``client``."""
    if probe.options is None:
        raise ValueError(f"{probe.id} has no options")
    results: list[tuple[float, float]] = []
    for order in ORDERS:
        messages = present(probe, order, context)
        results.append((
            sum(client.continuation_logprobs(adapter, messages, LABELS[0])),
            sum(client.continuation_logprobs(adapter, messages, LABELS[1])),
        ))
    first = _preferred(results[0], ORDERS[0])
    second = _preferred(results[1], ORDERS[1])
    picked = probe.options[first] if first is not None and first == second else None
    return TasteChoice(results[0], results[1], picked)
