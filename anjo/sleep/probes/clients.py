"""The ports the arms and the scorers reach outside through (Task 2).

Task 4 (the serving route) fills ``ModelClient``, ``AdapterRegistry`` and ``TokenLogprobs``; the
app wires the judges. None of them is called outside a port, so every test runs on fakes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from anjo.sleep.model_api import ChatResult
from anjo.sleep.types import CheckpointRef


@runtime_checkable
class ModelClient(Protocol):
    """Samples a reply from the base model (``adapter=None``) or from a person's adapter."""

    def complete(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], *, temperature: float,
    ) -> ChatResult:
        """One sampled reply. ``input_tokens`` is the prompt's length as the server counted it."""


@runtime_checkable
class AdapterRegistry(Protocol):
    """Which adapter serves a person: the adapter choice (Tasks 4 and 5)."""

    def last_promoted(self, person_id: str) -> CheckpointRef | None:
        """The person's last promoted checkpoint, or ``None`` when no night was promoted."""


@runtime_checkable
class TokenLogprobs(Protocol):
    """Log-probabilities of a continuation, for forced-choice scoring (``taste_ab``)."""

    def continuation_logprobs(
        self, adapter: CheckpointRef | None, messages: list[dict[str, str]], continuation: str,
    ) -> Sequence[float]:
        """The log-probability of each token of ``continuation`` given ``messages``."""


@runtime_checkable
class VertexClient(Protocol):
    """Gemini on Vertex, for the second judge. The real client lives in the app."""

    def generate(self, model: str, prompt: str, *, temperature: float) -> str:
        """The model's text answer to ``prompt``."""
